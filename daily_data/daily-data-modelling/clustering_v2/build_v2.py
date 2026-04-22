"""V2 level models: Ridge (portable JSON) + RandomForest (joblib), >2y daily, <2y monthly. Writes models/<cohort>/. See APPROACH.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = Path(__file__).resolve().parent
MODELS_DIR = V2_DIR / "models"

TARGET = "wash_count_total"
COHORT_MORE_THAN = "more_than_2yrs"
COHORT_LESS_THAN = "less_than_2yrs"

RADIUS_KM = 12.0
DBSCAN_EPS_RAD = RADIUS_KM / 6371.0088
DBSCAN_MIN = 2


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_dbscan(site_df: pd.DataFrame, cluster_label: str) -> pd.DataFrame:
    sites = (
        site_df[["site_client_id", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")
        .reset_index(drop=True)
    )
    if sites.empty:
        return pd.DataFrame(columns=["site_client_id", cluster_label])
    coords = np.radians(sites[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(
        eps=DBSCAN_EPS_RAD, min_samples=DBSCAN_MIN, metric="haversine"
    ).fit(coords).labels_
    sites[cluster_label] = labels.astype(int)
    return sites[["site_client_id", cluster_label]]


def align_train_test_clusters_to_train_refit(
    train: pd.DataFrame,
    test: pd.DataFrame,
    site_train_labeled: pd.DataFrame,
    label_col: str = "cluster_id_train",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use train-only refit DBSCAN labels on all rows; fall back to CSV cluster if site not in refit map."""
    m = site_train_labeled[["site_client_id", label_col]].rename(columns={label_col: "_refit_cl"})
    train = train.copy()
    test = test.copy()
    train["_dc_csv"] = pd.to_numeric(train["dbscan_cluster_12km"], errors="coerce")
    train = train.merge(m, on="site_client_id", how="left")
    train["dbscan_cluster_12km"] = (
        pd.to_numeric(train["_refit_cl"], errors="coerce")
        .combine_first(train["_dc_csv"])
        .fillna(-1)
        .astype(int)
    )
    train = train.drop(columns=["_refit_cl", "_dc_csv"])
    test["_dc_csv"] = pd.to_numeric(test["dbscan_cluster_12km"], errors="coerce")
    test = test.merge(m, on="site_client_id", how="left")
    test["dbscan_cluster_12km"] = (
        pd.to_numeric(test["_refit_cl"], errors="coerce")
        .combine_first(test["_dc_csv"])
        .fillna(-1)
        .astype(int)
    )
    test = test.drop(columns=["_refit_cl", "_dc_csv"])
    return train, test


def _cluster_centroids(site_clusters: pd.DataFrame, cluster_label: str) -> dict[str, Any]:
    centroids: list[dict[str, Any]] = []
    for cid, grp in site_clusters.groupby(cluster_label):
        if int(cid) == -1:
            continue
        centroids.append({
            "cluster_id": int(cid),
            "size": int(len(grp)),
            "lat": float(grp["latitude"].mean()),
            "lon": float(grp["longitude"].mean()),
        })
    centroids.sort(key=lambda d: d["cluster_id"])
    return {"radius_km": RADIUS_KM, "cluster_label": cluster_label, "centroids": centroids}


def _build_cluster_context(
    train: pd.DataFrame,
    cluster_col: str,
    context_cols: list[str],
    include_target_aggs: bool = True,
) -> pd.DataFrame:
    usable = [c for c in context_cols if c in train.columns]
    agg_spec: dict[str, Any] = {c: "median" for c in usable}
    if include_target_aggs:
        agg_spec[TARGET] = ["median", "mean", "std", "min", "max"]
    if not agg_spec:
        return train[[cluster_col]].drop_duplicates().rename(columns={cluster_col: "cluster_id"}).reset_index(drop=True)
    grp = train.groupby(cluster_col).agg(agg_spec)
    grp.columns = [
        "ctx_" + ("_".join(str(x) for x in col) if isinstance(col, tuple) else str(col))
        for col in grp.columns
    ]
    grp = grp.reset_index().rename(columns={cluster_col: "cluster_id"})
    return grp


def _cluster_monthly_series(train: pd.DataFrame, cluster_col: str, date_col: str, freq: str) -> dict[int, list[dict[str, Any]]]:
    df = train[["site_client_id", cluster_col, date_col, TARGET]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[df[cluster_col].astype(int) != -1]
    out: dict[int, list[dict[str, Any]]] = {}
    for cid, grp in df.groupby(cluster_col):
        per_site_month = (
            grp.groupby(["site_client_id", pd.Grouper(key=date_col, freq=freq)])[TARGET]
            .sum()
            .reset_index()
        )
        monthly = (
            per_site_month.groupby(date_col)[TARGET]
            .median()
            .sort_index()
        )
        out[int(cid)] = [
            {"month": d.strftime("%Y-%m-%d"), "value": float(v)}
            for d, v in monthly.items()
            if np.isfinite(v)
        ]
    return out


def _cluster_local_feature_medians(
    train: pd.DataFrame,
    cluster_col: str,
    feature_cols: list[str],
) -> dict[int, dict[str, float]]:
    skip = {cluster_col, TARGET, "latitude", "longitude"}
    cols = [
        c
        for c in feature_cols
        if c not in skip and not str(c).startswith("ctx_") and c in train.columns
    ]
    if not cols:
        return {}
    out: dict[int, dict[str, float]] = {}
    for cid, grp in train.groupby(cluster_col, sort=False):
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        med = grp[cols].median(numeric_only=True)
        d = {str(k): float(v) for k, v in med.items() if pd.notna(v) and np.isfinite(float(v))}
        if d:
            out[cid_int] = d
    return out


def _context_records_with_local_medians(
    context_df: pd.DataFrame,
    local_by_cluster: dict[int, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in context_df.to_dict(orient="records"):
        r = dict(rec)
        cid = int(r["cluster_id"])
        r["local_feature_medians"] = dict(local_by_cluster.get(cid, {}))
        rows.append(r)
    return rows


def _train_ridge_portable(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    X_tr, y_tr = train[feature_cols].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float)
    X_te, y_te = test[feature_cols].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=1.0, random_state=0)),
    ])
    pipe.fit(X_tr, y_tr)

    imp: SimpleImputer = pipe.named_steps["imp"]
    sc: StandardScaler = pipe.named_steps["sc"]
    reg: Ridge = pipe.named_steps["m"]

    portable = {
        "feature_order": list(feature_cols),
        "imputer": {"strategy": "median", "statistics": imp.statistics_.tolist()},
        "scaler": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
        "ridge": {
            "coef": reg.coef_.tolist(),
            "intercept": float(reg.intercept_),
            "alpha": 1.0,
        },
    }

    tr_pred = pipe.predict(X_tr)
    te_pred = pipe.predict(X_te)
    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_mae": float(mean_absolute_error(y_tr, tr_pred)),
        "test_mae": float(mean_absolute_error(y_te, te_pred)),
        "train_rmse": _rmse(y_tr, tr_pred),
        "test_rmse": _rmse(y_te, te_pred),
        "train_r2": float(r2_score(y_tr, tr_pred)),
        "test_r2": float(r2_score(y_te, te_pred)),
        "train_wape": _wape(y_tr, tr_pred),
        "test_wape": _wape(y_te, te_pred),
    }
    return portable, metrics


def _rf_estimator() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=4,
        random_state=0,
        n_jobs=-1,
    )


def _train_rf_joblib(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Same preprocessing as Ridge; tree model saved for sklearn inference in project_site."""
    X_tr, y_tr = train[feature_cols].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float)
    X_te, y_te = test[feature_cols].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", _rf_estimator()),
        ]
    )
    pipe.fit(X_tr, y_tr)
    tr_pred = pipe.predict(X_tr)
    te_pred = pipe.predict(X_te)
    bundle = {"pipeline": pipe, "feature_order": list(feature_cols)}
    metrics = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_mae": float(mean_absolute_error(y_tr, tr_pred)),
        "test_mae": float(mean_absolute_error(y_te, te_pred)),
        "train_rmse": _rmse(y_tr, tr_pred),
        "test_rmse": _rmse(y_te, te_pred),
        "train_r2": float(r2_score(y_tr, tr_pred)),
        "test_r2": float(r2_score(y_te, te_pred)),
        "train_wape": _wape(y_tr, tr_pred),
        "test_wape": _wape(y_te, te_pred),
    }
    return bundle, metrics


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


SITE_FEATURES_STATIC = [
    "latitude", "longitude", "region_enc", "state_enc",
    "tunnel_count", "age", "carwash_type_encoded",
    "costco_enc",
    "current_site_count", "previous_site_count",
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_rating_count",
    "distance_nearest_costco(5 mile)",
    "distance_nearest_walmart(5 mile)",
    "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile",
    "count_food_joints_0_5miles (0.5 mile)",
]

ANNUAL_WEATHER_FEATURES = [
    "weather_total_precipitation_mm",
    "weather_rainy_days",
    "weather_snowy_days",
    "weather_avg_temperature_c",
    "weather_max_temperature_c",
    "weather_min_temperature_c",
    "weather_days_below_freezing",
    "weather_days_pleasant_temp",
]

DAILY_WEATHER_FEATURES = [
    "daily_weather_temperature_2m_mean",
    "daily_weather_temperature_2m_max",
    "daily_weather_temperature_2m_min",
    "daily_weather_precipitation_sum",
    "daily_weather_rain_sum",
    "daily_weather_snowfall_sum",
    "daily_weather_precipitation_hours",
    "daily_weather_wind_speed_10m_max",
    "daily_weather_shortwave_radiation_sum",
    "daily_weather_sunshine_duration",
]


def assert_less_than_no_weather_columns(feature_cols: list[str]) -> None:
    bad = set(feature_cols) & (set(DAILY_WEATHER_FEATURES) | set(ANNUAL_WEATHER_FEATURES))
    if bad:
        raise RuntimeError(
            "<2y build: weather columns in feature list (reserved for daily >2y cohort): "
            + str(sorted(bad))
        )


TIME_LAG_FEATURES_DAILY = [
    "day_number", "month_number", "year_number", "day_of_week_feature",
    "prev_wash_count", "last_week_same_day", "running_avg_7_days",
]

CONTEXT_BASE_FEATURES = [
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_rating_count",
    "distance_nearest_costco(5 mile)",
    "distance_nearest_walmart(5 mile)",
    "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile",
    "count_food_joints_0_5miles (0.5 mile)",
    "daily_weather_precipitation_sum",
    "daily_weather_temperature_2m_mean",
    "daily_weather_wind_speed_10m_max",
]


def build_more_than() -> dict[str, Any]:
    print("[>2y] loading raw frames...")
    master = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
        low_memory=False,
    )
    more = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv",
        low_memory=False,
    )

    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    more["calendar_day"] = pd.to_datetime(more["calendar_day"], errors="coerce")

    extra_from_more = [
        c for c in (
            ["dbscan_cluster_12km"]
            + TIME_LAG_FEATURES_DAILY
            + ANNUAL_WEATHER_FEATURES
            + ["region_enc", "state_enc", "costco_enc", "carwash_type_encoded"]
        )
        if c in more.columns and c not in master.columns
    ]
    key_cols = ["site_client_id", "calendar_day"]
    df = master.merge(
        more[key_cols + extra_from_more],
        on=key_cols,
        how="left",
    )

    if "dbscan_cluster_12km" not in df.columns:
        raise RuntimeError(">2y: dbscan_cluster_12km missing after merge")

    print(f"[>2y] merged rows={len(df):,} sites={df['site_client_id'].nunique():,}")

    split_date = pd.Timestamp("2025-07-01")
    train = df[df["calendar_day"] < split_date].copy()
    test = df[df["calendar_day"] >= split_date].copy()
    print(f"[>2y] train={len(train):,}  test={len(test):,}  split={split_date.date()}")

    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = _fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    train, test = align_train_test_clusters_to_train_refit(train, test, site_train)
    centroids = _cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    print(f"[>2y] train centroids: {len(centroids['centroids'])} clusters")

    context_df = _build_cluster_context(
        train, "dbscan_cluster_12km", CONTEXT_BASE_FEATURES
    )
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    monthly_series = _cluster_monthly_series(
        train, "dbscan_cluster_12km", "calendar_day", freq="MS"
    )

    feature_candidates = (
        SITE_FEATURES_STATIC
        + ANNUAL_WEATHER_FEATURES
        + DAILY_WEATHER_FEATURES
        + TIME_LAG_FEATURES_DAILY
        + ["dbscan_cluster_12km"]
        + [c for c in train.columns if c.startswith("ctx_")]
    )
    feature_cols = [
        c for c in feature_candidates
        if c in train.columns and train[c].notna().any()
    ]
    print(f"[>2y] using {len(feature_cols)} features (dropped {len(feature_candidates)-len(feature_cols)} all-null columns)")

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    portable, metrics = _train_ridge_portable(train, test, feature_cols)
    print(f"[>2y] metrics: {metrics}")

    rf_bundle, rf_metrics = _train_rf_joblib(train, test, feature_cols)
    print(f"[>2y] RF metrics: {rf_metrics}")

    local_med = _cluster_local_feature_medians(train, "dbscan_cluster_12km", feature_cols)
    context_records = _context_records_with_local_medians(context_df, local_med)

    out_dir = MODELS_DIR / "more_than"
    _save_json(out_dir / "wash_count_model_12km.portable.json", portable)
    joblib.dump(rf_bundle, out_dir / "wash_count_model_12km.rf.joblib")
    _save_json(out_dir / "training_metrics_rf_12km.json", rf_metrics)
    _save_json(out_dir / "cluster_centroids_12km.json", centroids)
    _save_json(out_dir / "cluster_context_12km.json", {
        "cluster_label": "dbscan_cluster_12km",
        "feature_cols": [c for c in context_df.columns if c != "cluster_id"],
        "records": context_records,
    })
    _save_json(out_dir / "cluster_monthly_series_12km.json", {
        "frequency": "MS",
        "series": {str(k): v for k, v in monthly_series.items()},
    })
    _save_json(out_dir / "feature_spec_12km.json", {
        "site_features_static": SITE_FEATURES_STATIC,
        "annual_weather_features": ANNUAL_WEATHER_FEATURES,
        "daily_weather_features": DAILY_WEATHER_FEATURES,
        "time_lag_features": TIME_LAG_FEATURES_DAILY,
        "cluster_context_features": [c for c in context_df.columns if c != "cluster_id"],
        "final_feature_order": feature_cols,
        "cluster_col": "dbscan_cluster_12km",
    })
    _save_json(out_dir / "training_metrics_12km.json", metrics)
    return {
        "cohort": COHORT_MORE_THAN,
        "metrics": metrics,
        "metrics_rf": rf_metrics,
        "features": len(feature_cols),
        "clusters": len(centroids["centroids"]),
    }


SITE_FEATURES_MONTHLY = [
    "latitude", "longitude", "region_enc", "state_enc",
]
TIME_LAG_FEATURES_MONTHLY = [
    "period_index", "month_number", "year_number", "day_of_week_feature",
    "prev_wash_count", "last_week_same_day", "running_avg_7_days",
]
CONTEXT_BASE_FEATURES_MONTHLY: list[str] = []


def build_less_than() -> dict[str, Any]:
    print("[<2y] loading clustering-ready frame...")
    df = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    if "dbscan_cluster_12km" not in df.columns:
        raise RuntimeError("<2y: dbscan_cluster_12km missing")

    # period_index 1-12 = year_number 1 & month_number 1-12;
    # period_index 25-36 = year_number 2 & month_number 13-24. No rows with period_index 13-24.
    train = df[df["period_index"] <= 12].copy()
    test = df[df["period_index"] >= 25].copy()
    print(f"[<2y] train={len(train):,}  test={len(test):,}  (train=year1 idx<=12, test=year2 idx>=25)")

    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = _fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    train, test = align_train_test_clusters_to_train_refit(train, test, site_train)
    centroids = _cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    print(f"[<2y] train centroids: {len(centroids['centroids'])} clusters")

    context_df = _build_cluster_context(
        train,
        "dbscan_cluster_12km",
        CONTEXT_BASE_FEATURES_MONTHLY,
        include_target_aggs=True,
    )
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    monthly_series = _cluster_monthly_series(
        train, "dbscan_cluster_12km", "calendar_day", freq="MS"
    )

    feature_candidates = (
        SITE_FEATURES_MONTHLY
        + TIME_LAG_FEATURES_MONTHLY
        + ["dbscan_cluster_12km"]
        + [c for c in train.columns if c.startswith("ctx_")]
    )
    feature_cols = [
        c for c in feature_candidates
        if c in train.columns and train[c].notna().any()
    ]
    assert_less_than_no_weather_columns(feature_cols)
    print(f"[<2y] using {len(feature_cols)} features")

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    portable, metrics = _train_ridge_portable(train, test, feature_cols)
    print(f"[<2y] metrics: {metrics}")

    rf_bundle, rf_metrics = _train_rf_joblib(train, test, feature_cols)
    print(f"[<2y] RF metrics: {rf_metrics}")

    local_med = _cluster_local_feature_medians(train, "dbscan_cluster_12km", feature_cols)
    context_records = _context_records_with_local_medians(context_df, local_med)

    out_dir = MODELS_DIR / "less_than"
    _save_json(out_dir / "wash_count_model_12km.portable.json", portable)
    joblib.dump(rf_bundle, out_dir / "wash_count_model_12km.rf.joblib")
    _save_json(out_dir / "training_metrics_rf_12km.json", rf_metrics)
    _save_json(out_dir / "cluster_centroids_12km.json", centroids)
    _save_json(out_dir / "cluster_context_12km.json", {
        "cluster_label": "dbscan_cluster_12km",
        "feature_cols": [c for c in context_df.columns if c != "cluster_id"],
        "records": context_records,
    })
    _save_json(out_dir / "cluster_monthly_series_12km.json", {
        "frequency": "MS",
        "series": {str(k): v for k, v in monthly_series.items()},
    })
    _save_json(out_dir / "feature_spec_12km.json", {
        "site_features_static": SITE_FEATURES_MONTHLY,
        "time_lag_features": TIME_LAG_FEATURES_MONTHLY,
        "cluster_context_features": [c for c in context_df.columns if c != "cluster_id"],
        "final_feature_order": feature_cols,
        "cluster_col": "dbscan_cluster_12km",
    })
    _save_json(out_dir / "training_metrics_12km.json", metrics)
    return {
        "cohort": COHORT_LESS_THAN,
        "metrics": metrics,
        "metrics_rf": rf_metrics,
        "features": len(feature_cols),
        "clusters": len(centroids["centroids"]),
    }


def main() -> None:
    summary = {
        COHORT_MORE_THAN: build_more_than(),
        COHORT_LESS_THAN: build_less_than(),
    }
    _save_json(V2_DIR / "results" / "build_summary.json", summary)
    print("\n=== V2 build complete ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
