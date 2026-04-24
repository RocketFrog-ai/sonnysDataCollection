"""Realistic greenfield benchmark using unseen-site annual totals.

This script evaluates model ideas under assumptions that match the pre-opening
use case better than the existing time holdouts:

- Split by ``site_client_id`` (GroupKFold), not by later dates from the same site.
- Use only train-site geography to fit the 12 km DBSCAN peer map.
- Assign held-out sites to the nearest train-cluster centroid.
- Score at the business grain: annual totals per site.

Two views are reported for the <2y cohort:

1. ``greenfield_preopen``:
   no wash-history lags are used, which matches a brand new site before open.
2. ``with_history_upper_bound``:
   lag columns are allowed, which is closer to a rolling reforecast after the
   site has started operating.

For the >2y cohort, only a mature-site analogue is measurable from current data.
The repo does not contain realized operational Years 3-4 for the same sites, so
those future years still need to be extrapolated after the level model step.

Run:
  python daily_data/daily-data-modelling/clustering_v2/eval_realistic_site_holdout.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


V2 = Path(__file__).resolve().parent
REPO_ROOT = V2.parents[2]
RESULTS_DIR = V2 / "results"
OUT_JSON = RESULTS_DIR / "realistic_site_holdout_eval.json"

RADIUS_KM = 12.0
DBSCAN_EPS_RAD = RADIUS_KM / 6371.0088
DBSCAN_MIN = 2
N_SPLITS = 5


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = float(np.abs(y_true).sum())
    return float(np.abs(y_true - y_pred).sum() / max(denom, 1e-9))


def _pct_within(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    ape = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.mean(ape <= threshold) * 100.0)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _fit_train_dbscan(train_sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = (
        train_sites[["site_client_id", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")
        .reset_index(drop=True)
    )
    if sites.empty:
        empty = pd.DataFrame(columns=["site_client_id", "cluster"])
        empty_c = pd.DataFrame(columns=["cluster", "lat", "lon", "size"])
        return empty, empty_c

    coords = np.radians(sites[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(
        eps=DBSCAN_EPS_RAD,
        min_samples=DBSCAN_MIN,
        metric="haversine",
    ).fit(coords).labels_

    labeled = sites.copy()
    labeled["cluster"] = labels.astype(int)
    centroids = (
        labeled[labeled["cluster"] != -1]
        .groupby("cluster", as_index=False)
        .agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            size=("site_client_id", "nunique"),
        )
    )
    return labeled[["site_client_id", "cluster"]], centroids


def _assign_nearest_train_cluster(test_sites: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    sites = (
        test_sites[["site_client_id", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")
        .reset_index(drop=True)
    )
    if sites.empty:
        return pd.DataFrame(columns=["site_client_id", "cluster"])
    if centroids.empty:
        out = sites.copy()
        out["cluster"] = -1
        return out[["site_client_id", "cluster"]]

    centroid_rows = list(centroids[["cluster", "lat", "lon"]].itertuples(index=False, name=None))
    clusters: list[int] = []
    for site in sites.itertuples(index=False):
        best = min(centroid_rows, key=lambda c: _haversine_km(site.latitude, site.longitude, c[1], c[2]))
        clusters.append(int(best[0]))
    out = sites.copy()
    out["cluster"] = clusters
    return out[["site_client_id", "cluster"]]


def _yearly_metrics(df: pd.DataFrame) -> dict[str, Any]:
    by_year = (
        df.groupby(["site_client_id", "year_block"], as_index=False)
        .agg(actual=("actual", "sum"), pred=("pred", "sum"))
    )
    y = by_year["actual"].to_numpy(float)
    p = by_year["pred"].to_numpy(float)
    return {
        "n_site_years": int(len(by_year)),
        "actual_sum": round(float(np.sum(y)), 2),
        "pred_sum": round(float(np.sum(p)), 2),
        "mae": round(float(mean_absolute_error(y, p)), 2),
        "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 2),
        "r2": round(float(r2_score(y, p)), 4),
        "wape": round(_wape(y, p), 4),
        "pct_within_10pct": round(_pct_within(y, p, 0.10), 2),
        "pct_within_15pct": round(_pct_within(y, p, 0.15), 2),
        "pct_within_20pct": round(_pct_within(y, p, 0.20), 2),
        "pct_within_30pct": round(_pct_within(y, p, 0.30), 2),
    }


def _safe_fill(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in cols:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(median)
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(median)
    return train_df, test_df


def _geo_knn_monthly_predict(
    train_sites: pd.DataFrame,
    train_monthly: pd.DataFrame,
    test_rows: pd.DataFrame,
    month_col: str,
    *,
    k: int = 25,
) -> np.ndarray:
    lookup = train_sites[["site_client_id", "latitude", "longitude", "state_enc"]].drop_duplicates("site_client_id")
    coords = np.radians(lookup[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    lookup = lookup.reset_index(drop=True)
    month_maps = {
        key: grp.set_index("site_client_id")["wash_count_total"].to_dict()
        for key, grp in train_monthly.groupby(month_col)
    }
    global_month = train_monthly.groupby(month_col)["wash_count_total"].median().to_dict()

    preds: list[float] = []
    max_k = min(k, len(lookup))
    for row in test_rows[["latitude", "longitude", "state_enc", month_col]].itertuples(index=False):
        dist, ind = tree.query(np.radians([[row.latitude, row.longitude]]), k=max_k)
        month_map = month_maps.get(getattr(row, month_col), {})
        vals: list[float] = []
        weights: list[float] = []
        for d, idx in zip(dist[0], ind[0]):
            peer = lookup.iloc[idx]
            sid = int(peer["site_client_id"])
            if sid not in month_map:
                continue
            wt = 1.0 / max(float(d), 1e-6)
            if pd.notna(peer["state_enc"]) and peer["state_enc"] == row.state_enc:
                wt *= 1.5
            vals.append(float(month_map[sid]))
            weights.append(wt)
        if vals:
            preds.append(float(np.average(vals, weights=weights)))
        else:
            preds.append(float(global_month.get(getattr(row, month_col), train_monthly["wash_count_total"].median())))
    return np.asarray(preds, float)


def _less_than_monthly() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    df = df.dropna(subset=["wash_count_total", "latitude", "longitude"]).copy()
    df["op_month"] = df["month_number"].astype(int)
    df["year_block"] = np.where(df["year_number"].astype(int) == 1, 1, 2)
    return df


def _more_than_monthly() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv", low_memory=False)
    df = df.dropna(subset=["wash_count_total", "latitude", "longitude"]).copy()
    monthly = (
        df.groupby(["site_client_id", "year_number", "month_number"], as_index=False)
        .agg(
            wash_count_total=("wash_count_total", "sum"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            region_enc=("region_enc", "first"),
            state_enc=("state_enc", "first"),
            costco_enc=("costco_enc", "first"),
            tunnel_count=("tunnel_count", "first"),
            carwash_type_encoded=("carwash_type_encoded", "first"),
            current_count=("current_count", "first"),
            previous_count=("previous_count", "first"),
            weather_total_precipitation_mm=("weather_total_precipitation_mm", "first"),
            weather_rainy_days=("weather_rainy_days", "first"),
            weather_days_below_freezing=("weather_days_below_freezing", "first"),
            weather_days_pleasant_temp=("weather_days_pleasant_temp", "first"),
            weather_avg_daily_max_windspeed_ms=("weather_avg_daily_max_windspeed_ms", "first"),
            nearest_gas_station_distance_miles=("nearest_gas_station_distance_miles", "first"),
            nearest_gas_station_rating=("nearest_gas_station_rating", "first"),
            nearest_gas_station_rating_count=("nearest_gas_station_rating_count", "first"),
            competitors_count_4miles=("competitors_count_4miles", "first"),
            competitor_1_google_rating=("competitor_1_google_rating", "first"),
            competitor_1_distance_miles=("competitor_1_distance_miles", "first"),
            competitor_1_rating_count=("competitor_1_rating_count", "first"),
            distance_nearest_costco_5mile=("distance_nearest_costco(5 mile)", "first"),
            distance_nearest_walmart_5mile=("distance_nearest_walmart(5 mile)", "first"),
            distance_nearest_target_5mile=("distance_nearest_target (5 mile)", "first"),
            other_grocery_count_1mile=("other_grocery_count_1mile", "first"),
            count_food_joints_0_5miles=("count_food_joints_0_5miles (0.5 mile)", "first"),
        )
    )
    monthly["month_of_year"] = ((monthly["month_number"].astype(int) - 1) % 12) + 1
    monthly["year_block"] = monthly["year_number"].astype(int)
    return monthly


def _prepare_cluster_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    month_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_sites = train_df[["site_client_id", "latitude", "longitude"]].drop_duplicates("site_client_id")
    test_sites = test_df[["site_client_id", "latitude", "longitude"]].drop_duplicates("site_client_id")
    train_map, centroids = _fit_train_dbscan(train_sites)
    test_map = _assign_nearest_train_cluster(test_sites, centroids)

    train_df = train_df.merge(train_map, on="site_client_id", how="left")
    test_df = test_df.merge(test_map, on="site_client_id", how="left")

    cluster_overall = (
        train_df.groupby("cluster", as_index=False)
        .agg(
            cluster_mean=("wash_count_total", "mean"),
            cluster_median=("wash_count_total", "median"),
            cluster_std=("wash_count_total", "std"),
            cluster_size=("site_client_id", "nunique"),
        )
    )
    cluster_month = (
        train_df.groupby(["cluster", month_col], as_index=False)
        .agg(
            cluster_month_median=("wash_count_total", "median"),
            cluster_month_mean=("wash_count_total", "mean"),
        )
    )
    global_month = (
        train_df.groupby(month_col, as_index=False)["wash_count_total"]
        .median()
        .rename(columns={"wash_count_total": "global_month_median"})
    )

    train_df = train_df.merge(cluster_overall, on="cluster", how="left")
    train_df = train_df.merge(cluster_month, on=["cluster", month_col], how="left")
    train_df = train_df.merge(global_month, on=month_col, how="left")

    test_df = test_df.merge(cluster_overall, on="cluster", how="left")
    test_df = test_df.merge(cluster_month, on=["cluster", month_col], how="left")
    test_df = test_df.merge(global_month, on=month_col, how="left")

    fill_cols = [
        "cluster_mean",
        "cluster_median",
        "cluster_std",
        "cluster_size",
        "cluster_month_median",
        "cluster_month_mean",
        "global_month_median",
    ]
    return _safe_fill(train_df, test_df, fill_cols)


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    pipeline: Pipeline


def _less_than_greenfield_recipes() -> list[ModelRecipe]:
    return [
        ModelRecipe(
            "ridge",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("m", Ridge(alpha=3.0)),
                ]
            ),
        ),
        ModelRecipe(
            "rf",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "m",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=14,
                            min_samples_leaf=3,
                            random_state=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        ModelRecipe(
            "extra_trees",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "m",
                        ExtraTreesRegressor(
                            n_estimators=600,
                            min_samples_leaf=2,
                            random_state=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _less_than_history_recipes() -> list[ModelRecipe]:
    return [
        ModelRecipe(
            "rf_with_lags",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "m",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=14,
                            min_samples_leaf=3,
                            random_state=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        ModelRecipe(
            "extra_trees_with_lags",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "m",
                        ExtraTreesRegressor(
                            n_estimators=600,
                            min_samples_leaf=2,
                            random_state=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def _more_than_recipes() -> list[ModelRecipe]:
    recipes = [
        ModelRecipe(
            "ridge",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("m", Ridge(alpha=3.0)),
                ]
            ),
        ),
        ModelRecipe(
            "rf",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "m",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=16,
                            min_samples_leaf=2,
                            random_state=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        ModelRecipe(
            "extra_trees",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    (
                        "m",
                        ExtraTreesRegressor(
                            n_estimators=500,
                            min_samples_leaf=2,
                            random_state=0,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
    ]
    if XGBRegressor is not None:
        recipes.append(
            ModelRecipe(
                "xgboost",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        (
                            "m",
                            XGBRegressor(
                                n_estimators=500,
                                max_depth=5,
                                learning_rate=0.05,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                reg_alpha=0.2,
                                reg_lambda=1.0,
                                objective="reg:squarederror",
                                random_state=0,
                                n_jobs=4,
                            ),
                        ),
                    ]
                ),
            )
        )
    return recipes


def _rank_blocks(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {"model": name, **vals}
        for name, vals in metrics.items()
    ]
    rows.sort(key=lambda r: (r["wape"], -r["pct_within_20pct"]))
    return rows


def eval_less_than() -> dict[str, Any]:
    df = _less_than_monthly()
    gkf = GroupKFold(n_splits=N_SPLITS)

    greenfield_parts: dict[str, list[pd.DataFrame]] = {"geo_knn": [], "blend_70_geo_30_extra_trees": []}
    for recipe in _less_than_greenfield_recipes():
        greenfield_parts[recipe.name] = []

    history_parts: dict[str, list[pd.DataFrame]] = {}
    for recipe in _less_than_history_recipes():
        history_parts[recipe.name] = []

    base_cols = [
        "latitude",
        "longitude",
        "region_enc",
        "state_enc",
        "site_count",
        "op_month",
        "year_block",
        "cluster",
        "cluster_mean",
        "cluster_median",
        "cluster_std",
        "cluster_size",
        "cluster_month_median",
        "cluster_month_mean",
        "global_month_median",
    ]
    history_cols = base_cols + [
        "prev_wash_count",
        "last_week_same_day",
        "running_avg_7_days",
    ]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["site_client_id"]), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, test_df = _prepare_cluster_features(train_df, test_df, month_col="op_month")

        geo_pred = _geo_knn_monthly_predict(
            train_df[["site_client_id", "latitude", "longitude", "state_enc"]],
            train_df[["site_client_id", "op_month", "wash_count_total"]],
            test_df[["latitude", "longitude", "state_enc", "op_month"]],
            "op_month",
        )
        greenfield_parts["geo_knn"].append(
            test_df[["site_client_id", "year_block"]].assign(
                actual=test_df["wash_count_total"].to_numpy(float),
                pred=geo_pred,
            )
        )

        for recipe in _less_than_greenfield_recipes():
            recipe.pipeline.fit(train_df[base_cols], train_df["wash_count_total"].to_numpy(float))
            pred = recipe.pipeline.predict(test_df[base_cols])
            greenfield_parts[recipe.name].append(
                test_df[["site_client_id", "year_block"]].assign(
                    actual=test_df["wash_count_total"].to_numpy(float),
                    pred=np.asarray(pred, float),
                )
            )

        # Blend the strongest no-history local baseline with the strongest tree model.
        et_pred = greenfield_parts["extra_trees"][-1]["pred"].to_numpy(float)
        greenfield_parts["blend_70_geo_30_extra_trees"].append(
            test_df[["site_client_id", "year_block"]].assign(
                actual=test_df["wash_count_total"].to_numpy(float),
                pred=(0.70 * geo_pred + 0.30 * et_pred),
            )
        )

        for recipe in _less_than_history_recipes():
            recipe.pipeline.fit(train_df[history_cols], train_df["wash_count_total"].to_numpy(float))
            pred = recipe.pipeline.predict(test_df[history_cols])
            history_parts[recipe.name].append(
                test_df[["site_client_id", "year_block"]].assign(
                    actual=test_df["wash_count_total"].to_numpy(float),
                    pred=np.asarray(pred, float),
                )
            )

        print(f"[<2y] fold {fold}/{N_SPLITS} complete")

    greenfield_metrics = {
        name: _yearly_metrics(pd.concat(parts, ignore_index=True))
        for name, parts in greenfield_parts.items()
    }
    history_metrics = {
        name: _yearly_metrics(pd.concat(parts, ignore_index=True))
        for name, parts in history_parts.items()
    }

    return {
        "cohort": "less_than_2yrs",
        "business_grain": "annual totals per site (Year 1 and Year 2)",
        "greenfield_preopen": {
            "definition": "Unseen-site holdout with no wash-history lag features allowed.",
            "models": greenfield_metrics,
            "ranked": _rank_blocks(greenfield_metrics),
        },
        "with_history_upper_bound": {
            "definition": "Unseen-site holdout where lag features are allowed. This is closer to a rolling reforecast after opening, not a day-0 greenfield estimate.",
            "models": history_metrics,
            "ranked": _rank_blocks(history_metrics),
        },
    }


def eval_more_than() -> dict[str, Any]:
    df = _more_than_monthly()
    gkf = GroupKFold(n_splits=N_SPLITS)

    parts: dict[str, list[pd.DataFrame]] = {
        "cluster_month_median": [],
        "geo_knn": [],
        "blend_70_geo_30_extra_trees": [],
    }
    for recipe in _more_than_recipes():
        parts[recipe.name] = []

    feature_cols = [
        "latitude",
        "longitude",
        "region_enc",
        "state_enc",
        "costco_enc",
        "tunnel_count",
        "carwash_type_encoded",
        "current_count",
        "previous_count",
        "weather_total_precipitation_mm",
        "weather_rainy_days",
        "weather_days_below_freezing",
        "weather_days_pleasant_temp",
        "weather_avg_daily_max_windspeed_ms",
        "nearest_gas_station_distance_miles",
        "nearest_gas_station_rating",
        "nearest_gas_station_rating_count",
        "competitors_count_4miles",
        "competitor_1_google_rating",
        "competitor_1_distance_miles",
        "competitor_1_rating_count",
        "distance_nearest_costco_5mile",
        "distance_nearest_walmart_5mile",
        "distance_nearest_target_5mile",
        "other_grocery_count_1mile",
        "count_food_joints_0_5miles",
        "month_of_year",
        "year_block",
        "cluster",
        "cluster_mean",
        "cluster_median",
        "cluster_std",
        "cluster_size",
        "cluster_month_median",
        "cluster_month_mean",
        "global_month_median",
    ]

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["site_client_id"]), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, test_df = _prepare_cluster_features(train_df, test_df, month_col="month_of_year")

        parts["cluster_month_median"].append(
            test_df[["site_client_id", "year_block"]].assign(
                actual=test_df["wash_count_total"].to_numpy(float),
                pred=test_df["cluster_month_median"].to_numpy(float),
            )
        )

        geo_pred = _geo_knn_monthly_predict(
            train_df[["site_client_id", "latitude", "longitude", "state_enc"]],
            train_df[["site_client_id", "month_of_year", "wash_count_total"]],
            test_df[["latitude", "longitude", "state_enc", "month_of_year"]],
            "month_of_year",
        )
        parts["geo_knn"].append(
            test_df[["site_client_id", "year_block"]].assign(
                actual=test_df["wash_count_total"].to_numpy(float),
                pred=geo_pred,
            )
        )

        extra_trees_pred: np.ndarray | None = None
        for recipe in _more_than_recipes():
            recipe.pipeline.fit(train_df[feature_cols], train_df["wash_count_total"].to_numpy(float))
            pred = np.asarray(recipe.pipeline.predict(test_df[feature_cols]), float)
            parts[recipe.name].append(
                test_df[["site_client_id", "year_block"]].assign(
                    actual=test_df["wash_count_total"].to_numpy(float),
                    pred=pred,
                )
            )
            if recipe.name == "extra_trees":
                extra_trees_pred = pred

        if extra_trees_pred is None:
            raise RuntimeError("extra_trees recipe missing from >2y benchmark")

        parts["blend_70_geo_30_extra_trees"].append(
            test_df[["site_client_id", "year_block"]].assign(
                actual=test_df["wash_count_total"].to_numpy(float),
                pred=(0.70 * geo_pred + 0.30 * extra_trees_pred),
            )
        )

        print(f"[>2y] fold {fold}/{N_SPLITS} complete")

    metrics = {
        name: _yearly_metrics(pd.concat(rows, ignore_index=True))
        for name, rows in parts.items()
    }
    return {
        "cohort": "more_than_2yrs",
        "business_grain": "annual totals per site (2024 and 2025 mature analogue years)",
        "note": (
            "Current data supports an honest mature-site analogue benchmark, but does not contain realized operational "
            "Years 3-4 for the same sites. Use this to choose the mature-level model, then extrapolate Years 3-4 separately."
        ),
        "models": metrics,
        "ranked": _rank_blocks(metrics),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_name": "Realistic greenfield site-holdout annual benchmark",
        "split_design": "GroupKFold by site_client_id, train-only DBSCAN 12 km, test sites assigned to nearest train centroid",
        "important_caveats": [
            "This is stricter than the current time-based holdout because held-out sites are never seen during training.",
            "The <2y greenfield_preopen view removes wash-history lag features on purpose.",
            "The <2y with_history_upper_bound view is a useful post-opening update benchmark, not a pre-opening benchmark.",
            "The repo does not contain realized operational Years 3-4 for the same sites, so Years 3-4 remain an extrapolation problem.",
        ],
        "less_than_2yrs": eval_less_than(),
        "more_than_2yrs": eval_more_than(),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
