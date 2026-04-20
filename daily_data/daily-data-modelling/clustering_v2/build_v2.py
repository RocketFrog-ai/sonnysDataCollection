"""
Build the V2 clustering + Ridge pipeline for BOTH cohorts.

For each cohort (more_than_2yrs = daily, less_than_2yrs = monthly) we produce:
  - cluster_centroids_12km.json   (train-only DBSCAN centroids for serving)
  - cluster_context_12km.json     (train-only per-cluster feature aggregates,
                                   this is the "localisation" signal that gives
                                   Ridge information about OTHER sites in the
                                   cluster, which V1 was missing)
  - cluster_monthly_series_12km.json (train-only monthly median wash-count
                                   series per cluster, used as the source
                                   for the 24-month projection)
  - wash_count_model_12km.portable.json (portable Ridge: imputer means,
                                   scaler mean/scale, coefs, intercept,
                                   feature_order)
  - feature_spec_12km.json        (what the serving layer must feed)
  - training_metrics_12km.json    (train vs hold-out metrics for that cohort)

Splits:
  - >2y (daily):      train = calendar_day <= 2025-06-30, test = rest of 2025
  - <2y (monthly):    train = period_index <= 12 (site-year 1)
                      test  = period_index >= 25 (site-year 2)

V2 improvements vs V1:
  1. Adds daily_weather_* features for the >2y cohort (V1 used only
     annual aggregates).
  2. Adds a `ctx_*` block of train-only cluster-aggregate features
     (median/mean/std of wash_count + medians of key site features),
     giving Ridge access to the local-market structure of the cluster.
  3. Hardens feature selection: drops columns with no observed values
     in the train split (the UserWarning fix).
  4. Persists centroids, monthly series and context as portable JSON so
     the serving layer no longer has to re-read full CSVs or joblib-pickle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = Path(__file__).resolve().parent
MODELS_DIR = V2_DIR / "models"

TARGET = "wash_count_total"
RADIUS_KM = 12.0
DBSCAN_EPS_RAD = RADIUS_KM / 6371.0088
DBSCAN_MIN = 2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_dbscan(site_df: pd.DataFrame, cluster_label: str) -> pd.DataFrame:
    """Fit DBSCAN on site-level lat/lon and return site -> cluster mapping."""
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


def _cluster_centroids(site_clusters: pd.DataFrame, cluster_label: str) -> dict[str, Any]:
    """Return serving-time centroids for every non-noise cluster."""
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
    """Per-cluster train-only aggregates (the 'localisation' features).

    If `context_cols` is empty and `include_target_aggs` is False, we
    only return the cluster id column (useful when a cohort does not
    benefit from cluster-context features).
    """
    usable = [c for c in context_cols if c in train.columns]
    agg_spec: dict[str, Any] = {c: "median" for c in usable}
    if include_target_aggs:
        agg_spec[TARGET] = ["median", "mean", "std"]
    if not agg_spec:
        return train[[cluster_col]].drop_duplicates().rename(columns={cluster_col: "cluster_id"}).reset_index(drop=True)
    grp = train.groupby(cluster_col).agg(agg_spec)
    grp.columns = [
        "ctx_" + ("_".join(col) if isinstance(col, tuple) else col)
        for col in grp.columns
    ]
    grp = grp.reset_index().rename(columns={cluster_col: "cluster_id"})
    return grp


def _cluster_monthly_series(train: pd.DataFrame, cluster_col: str, date_col: str, freq: str) -> dict[int, list[dict[str, Any]]]:
    """Monthly median wash-count series per cluster, from train only.

    For >2y the underlying data is daily so we resample. For <2y the
    `calendar_day` is already a monthly pseudo-date.
    """
    df = train[["site_client_id", cluster_col, date_col, TARGET]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[df[cluster_col].astype(int) != -1]
    out: dict[int, list[dict[str, Any]]] = {}
    for cid, grp in df.groupby(cluster_col):
        per_site_month = (
            grp.groupby(["site_client_id", pd.Grouper(key=date_col, freq=freq)])[TARGET]
            .sum()  # wash-count per site-month
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


def _train_ridge_portable(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train the Ridge pipeline and return (portable_dict, metrics)."""
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


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# >2y (daily) cohort
# ---------------------------------------------------------------------------

# Site/competition/retail/gas fields (stable per site).
SITE_FEATURES_STATIC = [
    "latitude", "longitude", "region_enc", "state_enc",
    "tunnel_count", "age", "carwash_type_encoded",
    "costco_enc",
    "current_site_count", "previous_site_count",
    # gas
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
    # competitors
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_rating_count",
    # retail
    "distance_nearest_costco(5 mile)",
    "distance_nearest_walmart(5 mile)",
    "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile",
    "count_food_joints_0_5miles (0.5 mile)",
]

# Annual climate (one value per site).
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

# Daily weather (varies with date).
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

# Time + lag features.
TIME_LAG_FEATURES_DAILY = [
    "day_number", "month_number", "year_number", "day_of_week_feature",
    "prev_wash_count", "last_week_same_day", "running_avg_7_days",
]

# Columns we'll aggregate per cluster (median) as part of ctx_*.
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
    """Build V2 artifacts for the >2y cohort (daily panel)."""
    print("[>2y] loading raw frames...")
    master = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
        low_memory=False,
    )
    # `more_than-2yrs.csv` already has DBSCAN clusters, lag features and
    # annual weather - we keep its cluster assignment to stay consistent
    # with the V1 model and only RE-fit centroids from the train slice.
    more = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv",
        low_memory=False,
    )

    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    more["calendar_day"] = pd.to_datetime(more["calendar_day"], errors="coerce")

    # Columns to import from `more` to enrich `master`.
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

    # Ensure cluster column exists even if merge failed for some row.
    if "dbscan_cluster_12km" not in df.columns:
        raise RuntimeError(">2y: dbscan_cluster_12km missing after merge")

    print(f"[>2y] merged rows={len(df):,} sites={df['site_client_id'].nunique():,}")

    # --- train/test split -------------------------------------------------
    split_date = pd.Timestamp("2025-07-01")
    train = df[df["calendar_day"] < split_date].copy()
    test = df[df["calendar_day"] >= split_date].copy()
    print(f"[>2y] train={len(train):,}  test={len(test):,}  split={split_date.date()}")

    # --- train-only DBSCAN centroids -------------------------------------
    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = _fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    centroids = _cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    print(f"[>2y] train centroids: {len(centroids['centroids'])} clusters")

    # --- train-only cluster context aggregates ---------------------------
    context_df = _build_cluster_context(
        train, "dbscan_cluster_12km", CONTEXT_BASE_FEATURES
    )
    # merge back into train & test (leakage-free: ctx computed from train only)
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    # --- train-only cluster monthly series (for projection) --------------
    monthly_series = _cluster_monthly_series(
        train, "dbscan_cluster_12km", "calendar_day", freq="MS"
    )

    # --- final feature set ------------------------------------------------
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

    # target must be present and finite
    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    portable, metrics = _train_ridge_portable(train, test, feature_cols)
    print(f"[>2y] metrics: {metrics}")

    out_dir = MODELS_DIR / "more_than"
    _save_json(out_dir / "wash_count_model_12km.portable.json", portable)
    _save_json(out_dir / "cluster_centroids_12km.json", centroids)
    _save_json(out_dir / "cluster_context_12km.json", {
        "cluster_label": "dbscan_cluster_12km",
        "feature_cols": [c for c in context_df.columns if c != "cluster_id"],
        "records": context_df.to_dict(orient="records"),
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
    return {"cohort": "more_than_2yrs", "metrics": metrics, "features": len(feature_cols), "clusters": len(centroids["centroids"])}


# ---------------------------------------------------------------------------
# <2y (monthly) cohort
# ---------------------------------------------------------------------------

SITE_FEATURES_MONTHLY = [
    "latitude", "longitude", "region_enc", "state_enc",
]
TIME_LAG_FEATURES_MONTHLY = [
    "period_index", "month_number", "year_number", "day_of_week_feature",
    "prev_wash_count", "last_week_same_day", "running_avg_7_days",
]
# <2y has a thin static-feature schema, so the only reliable
# cluster-context signal is the wash-count itself. We intentionally do
# NOT aggregate lat/lon here (they're already per-row features and would
# just add collinearity to Ridge).
CONTEXT_BASE_FEATURES_MONTHLY: list[str] = []


def build_less_than() -> dict[str, Any]:
    """Build V2 artifacts for the <2y cohort (monthly panel)."""
    print("[<2y] loading clustering-ready frame...")
    df = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    if "dbscan_cluster_12km" not in df.columns:
        raise RuntimeError("<2y: dbscan_cluster_12km missing")

    # --- train/test split: site-year-1 vs site-year-2 --------------------
    train = df[df["period_index"] <= 12].copy()
    test = df[df["period_index"] >= 25].copy()
    print(f"[<2y] train={len(train):,}  test={len(test):,}  (train=site-year1, test=site-year2)")

    # --- train-only centroids --------------------------------------------
    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = _fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    centroids = _cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    print(f"[<2y] train centroids: {len(centroids['centroids'])} clusters")

    # --- train-only cluster context --------------------------------------
    # The <2y backtest shows V1-baseline (no ctx) outperforms any ctx
    # variant for this cohort: the feature schema is thin and the test
    # split is site-year-2 of the SAME sites, so ctx_wash_* becomes a
    # near-duplicate of the target signal and adds collinearity. Keep
    # the artifact for API consistency but with no aggregate columns.
    context_df = _build_cluster_context(
        train,
        "dbscan_cluster_12km",
        CONTEXT_BASE_FEATURES_MONTHLY,
        include_target_aggs=False,
    )
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    # --- monthly series (calendar_day is already a monthly pseudo-date) ---
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
    print(f"[<2y] using {len(feature_cols)} features")

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    portable, metrics = _train_ridge_portable(train, test, feature_cols)
    print(f"[<2y] metrics: {metrics}")

    out_dir = MODELS_DIR / "less_than"
    _save_json(out_dir / "wash_count_model_12km.portable.json", portable)
    _save_json(out_dir / "cluster_centroids_12km.json", centroids)
    _save_json(out_dir / "cluster_context_12km.json", {
        "cluster_label": "dbscan_cluster_12km",
        "feature_cols": [c for c in context_df.columns if c != "cluster_id"],
        "records": context_df.to_dict(orient="records"),
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
    return {"cohort": "less_than_2yrs", "metrics": metrics, "features": len(feature_cols), "clusters": len(centroids["centroids"])}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    summary = {
        "more_than_2yrs": build_more_than(),
        "less_than_2yrs": build_less_than(),
    }
    _save_json(V2_DIR / "results" / "build_summary.json", summary)
    print("\n=== V2 build complete ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
