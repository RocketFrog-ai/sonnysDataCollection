"""90/10 site-holdout model zoo runner.

Creates a versioned run folder under ../outputs_site_holdout_intersection/<run_id>/ that contains:
  - models/<cohort>/... artifacts (ridge portable JSON + rf joblib + clusters + monthly series)
  - results/eval_site_holdout_90_10.json
  - results/REPORT.md

This is intentionally "self-contained" so experiments don't disturb the default v2 artifacts in ../models/.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EX_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = V2_DIR.parents[2]
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(REPO_ROOT))

import build_v2 as b  # noqa: E402
import project_site as ps  # noqa: E402


TARGET = b.TARGET
RANDOM_SEED = 7
HOLDOUT_FRAC = 0.10
MAX_HOLDOUT_SITES_EVAL: int | None = None  # set to an int to cap runtime
RF_N_ESTIMATORS = 120
RF_MAX_DEPTH = 16
RF_MIN_SAMPLES_LEAF = 4

# (cohort_key, cluster_id, ts_method) -> forecast series (horizon=24) on the raw cluster track (no prefix)
_FORECAST_CACHE: dict[tuple[str, int, str], pd.Series] = {}


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))

def _git_rev() -> str | None:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        return out[:12]
    except Exception:
        return None


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _train_rf_joblib_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    X_tr, y_tr = train[feature_cols].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float)
    X_te, y_te = test[feature_cols].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)
    est = RandomForestRegressor(
        n_estimators=int(RF_N_ESTIMATORS),
        max_depth=int(RF_MAX_DEPTH),
        min_samples_leaf=int(RF_MIN_SAMPLES_LEAF),
        random_state=0,
        n_jobs=-1,
    )
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", est),
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
        "rf_params": {
            "n_estimators": int(RF_N_ESTIMATORS),
            "max_depth": int(RF_MAX_DEPTH),
            "min_samples_leaf": int(RF_MIN_SAMPLES_LEAF),
        },
    }
    return bundle, metrics


def _site_split(site_ids: np.ndarray, holdout_frac: float, seed: int) -> tuple[set[int], set[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array([int(x) for x in site_ids if pd.notna(x)], dtype=int)
    ids = np.unique(ids)
    rng.shuffle(ids)
    n_hold = int(np.ceil(len(ids) * holdout_frac))
    hold = set(int(x) for x in ids[:n_hold])
    train = set(int(x) for x in ids[n_hold:])
    return train, hold


def _site_intersection() -> np.ndarray:
    lt = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        usecols=["site_client_id"],
        low_memory=False,
    ).dropna()
    gt = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
        usecols=["site_client_id"],
        low_memory=False,
    ).dropna()
    ls = set(lt["site_client_id"].astype(int).unique())
    gs = set(gt["site_client_id"].astype(int).unique())
    inter = sorted(ls & gs)
    return np.asarray(inter, dtype=int)


def _metric_block(y_true: list[float], y_pred: list[float]) -> dict[str, Any]:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    abs_err = np.abs(err)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "actual_sum": round(float(np.sum(y)), 2),
        "pred_sum": round(float(np.sum(p)), 2),
        "bias_pct_of_actual_sum": round(float(np.sum(err) / max(np.sum(y), 1e-9)) * 100.0, 2),
        "mae": round(float(np.mean(abs_err)), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "wape": round(float(np.sum(abs_err) / max(np.sum(np.abs(y)), 1e-9)), 4),
    }
    pos = y > 0
    if np.any(pos):
        ape = abs_err[pos] / y[pos]
        out["median_abs_pct_error"] = round(float(np.median(ape)) * 100.0, 2)
        out["mean_abs_pct_error"] = round(float(np.mean(ape)) * 100.0, 2)
        for thr in (0.10, 0.15, 0.20, 0.30):
            out[f"pct_within_{int(thr * 100)}pct"] = round(float(np.mean(ape <= thr)) * 100.0, 2)
    return out


def _cluster_series_from_train(train: pd.DataFrame, cluster_col: str, date_col: str, freq: str) -> dict[int, list[dict[str, Any]]]:
    # Reuse build_v2 logic to keep series definition aligned.
    return b._cluster_monthly_series(train, cluster_col, date_col, freq=freq)


@dataclass(frozen=True)
class Variant:
    name: str
    level_model: str  # ridge|rf
    ts_method: str  # arima|holt_winters|blend|meta|prophet
    use_prefix: bool
    bridge: bool


VARIANTS: list[Variant] = [
    Variant("ridge_arima_prefix_bridge", "ridge", "arima", True, True),
    Variant("ridge_holt_prefix_bridge", "ridge", "holt_winters", True, True),
    Variant("ridge_blend_prefix_bridge", "ridge", "blend", True, True),
    Variant("ridge_meta_prefix_bridge", "ridge", "meta", True, True),
    Variant("rf_arima_prefix_bridge", "rf", "arima", True, True),
    Variant("rf_holt_prefix_bridge", "rf", "holt_winters", True, True),
    Variant("rf_blend_prefix_bridge", "rf", "blend", True, True),
    Variant("rf_meta_prefix_bridge", "rf", "meta", True, True),
    Variant("rf_arima_no_prefix", "rf", "arima", False, False),
]


def _portable_level_predict(model: dict[str, Any], feature_vec: dict[str, float], ctx_row: dict[str, Any] | None) -> float:
    return ps._score_portable(model, feature_vec, ctx_row)


def _rf_level_predict(bundle: dict[str, Any], feature_vec: dict[str, float], ctx_row: dict[str, Any] | None) -> float:
    return ps._score_rf_bundle(bundle, feature_vec, ctx_row)


def _load_assets(model_dir: Path) -> dict[str, Any]:
    return {
        "model": json.loads((model_dir / "wash_count_model_12km.portable.json").read_text()),
        "centroids": json.loads((model_dir / "cluster_centroids_12km.json").read_text()),
        "context": json.loads((model_dir / "cluster_context_12km.json").read_text()),
        "series": json.loads((model_dir / "cluster_monthly_series_12km.json").read_text()),
        "spec": json.loads((model_dir / "feature_spec_12km.json").read_text()),
        "rf_bundle": joblib.load(model_dir / "wash_count_model_12km.rf.joblib"),
    }


def _ctx_row_for_cluster(assets: dict[str, Any], cluster_id: int) -> dict[str, Any] | None:
    for r in assets["context"]["records"]:
        if int(r.get("cluster_id", -999)) == int(cluster_id):
            return r
    return None


def _cluster_series(assets: dict[str, Any], cluster_id: int) -> pd.Series:
    return ps._series_to_df(assets["series"]["series"].get(str(int(cluster_id)), []))


def _anchor_and_scale(
    *,
    assets: dict[str, Any],
    lat: float,
    lon: float,
    level_model: str,
    is_daily_model: bool,
    days_per_month: int,
) -> tuple[int, float, float, pd.Series]:
    centroids = assets["centroids"]["centroids"]
    nearest = ps._nearest_cluster(centroids, lat, lon)
    cluster_id = int(nearest["cluster_id"])
    ctx_row = _ctx_row_for_cluster(assets, cluster_id)
    vec = ps._feature_vector(assets["spec"], ctx_row, lat, lon, cluster_id)
    if level_model == "ridge":
        anchor_raw = _portable_level_predict(assets["model"], vec, ctx_row)
    elif level_model == "rf":
        anchor_raw = _rf_level_predict(assets["rf_bundle"], vec, ctx_row)
    else:
        raise ValueError(level_model)
    anchor_monthly = anchor_raw * days_per_month if is_daily_model else anchor_raw
    series = _cluster_series(assets, cluster_id)
    cluster_monthly_level = float(series.tail(6).mean()) if len(series) else float("nan")
    if len(series) >= 6 and np.isfinite(anchor_monthly) and cluster_monthly_level > 0:
        scale = float(anchor_monthly / cluster_monthly_level)
    else:
        scale = 1.0
    return cluster_id, float(scale), float(anchor_monthly), series


def _project_years_1_4(
    *,
    lat: float,
    lon: float,
    less_assets: dict[str, Any],
    more_assets: dict[str, Any],
    variant: Variant,
) -> dict[str, Any]:
    lt_cluster_id, lt_scale, _, lt_series = _anchor_and_scale(
        assets=less_assets,
        lat=lat,
        lon=lon,
        level_model=variant.level_model,
        is_daily_model=False,
        days_per_month=30,
    )
    lt_key = ("lt", int(lt_cluster_id), str(variant.ts_method))
    if lt_key not in _FORECAST_CACHE:
        _FORECAST_CACHE[lt_key] = ps._forecast(lt_series, 24, variant.ts_method)
    lt_fc = _FORECAST_CACHE[lt_key] * lt_scale
    lt_vals = np.maximum(lt_fc.to_numpy(dtype=float), 0.0)
    y1 = float(lt_vals[:12].sum()) if len(lt_vals) >= 12 else float(lt_vals.sum())
    y2 = float(lt_vals[12:24].sum()) if len(lt_vals) >= 24 else 0.0

    prefix: list[float] | None = None
    if variant.use_prefix and len(lt_vals) >= 24:
        prefix = [float(x) for x in lt_vals[:24].tolist()]

    gt_cluster_id, gt_scale, _, gt_series = _anchor_and_scale(
        assets=more_assets,
        lat=lat,
        lon=lon,
        level_model=variant.level_model,
        is_daily_model=True,
        days_per_month=30,
    )
    forecast_input = (
        ps._series_for_mature_forecast_with_opening_context(gt_series, prefix)
        if prefix is not None and len(prefix) == 24
        else gt_series
    )
    if prefix is None:
        gt_key = ("gt", int(gt_cluster_id), str(variant.ts_method))
        if gt_key not in _FORECAST_CACHE:
            _FORECAST_CACHE[gt_key] = ps._forecast(gt_series, 24, variant.ts_method)
        gt_fc = _FORECAST_CACHE[gt_key] * gt_scale
    else:
        gt_fc = ps._forecast(forecast_input, 24, variant.ts_method) * gt_scale
    gt_vals = np.maximum(gt_fc.to_numpy(dtype=float), 0.0)

    if variant.bridge and prefix is not None and len(prefix) == 24 and len(gt_vals) >= 1 and gt_vals[0] > 1e-12:
        factor = float(lt_vals[23] / gt_vals[0]) if len(lt_vals) >= 24 else 1.0
        if np.isfinite(factor) and factor > 0:
            gt_vals = gt_vals * factor

    y3 = float(gt_vals[:12].sum()) if len(gt_vals) >= 12 else float(gt_vals.sum())
    y4 = float(gt_vals[12:24].sum()) if len(gt_vals) >= 24 else 0.0
    return {
        "year_1": y1,
        "year_2": y2,
        "year_3": y3,
        "year_4": y4,
        "lt_cluster_id": int(lt_cluster_id),
        "gt_cluster_id": int(gt_cluster_id),
    }


def _actual_year_totals_less_than(df_lt: pd.DataFrame, site_id: int) -> dict[str, float]:
    s = df_lt[df_lt["site_client_id"].astype(int) == int(site_id)].copy()
    s[TARGET] = pd.to_numeric(s[TARGET], errors="coerce")
    s = s.dropna(subset=[TARGET])
    y1 = float(s.loc[s["year_number"] == 1, TARGET].sum())
    y2 = float(s.loc[s["year_number"] == 2, TARGET].sum())
    return {"actual_year_1": y1, "actual_year_2": y2}


def _actual_year_totals_more_than(df_gt: pd.DataFrame, site_id: int) -> dict[str, float]:
    s = df_gt[df_gt["site_client_id"].astype(int) == int(site_id)].copy()
    s["calendar_day"] = pd.to_datetime(s["calendar_day"], errors="coerce")
    s[TARGET] = pd.to_numeric(s[TARGET], errors="coerce")
    s = s.dropna(subset=["calendar_day", TARGET])
    y2024 = float(s.loc[s["calendar_day"].dt.year == 2024, TARGET].sum())
    y2025 = float(s.loc[s["calendar_day"].dt.year == 2025, TARGET].sum())
    return {"actual_2024": y2024, "actual_2025": y2025}


def _build_less_than_models(out_dir: Path, *, holdout_sites: set[int]) -> dict[str, Any]:
    df = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    df = df.dropna(subset=["site_client_id", "latitude", "longitude", "calendar_day"])
    all_sites = set(int(x) for x in df["site_client_id"].astype(int).unique())
    train_sites = all_sites - set(int(x) for x in holdout_sites)
    train = df[df["site_client_id"].astype(int).isin(train_sites)].copy()
    test = df[df["site_client_id"].astype(int).isin(holdout_sites)].copy()

    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = b._fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    # Drop noise sites from the train-only refit clustering.
    site_train = site_train[site_train["cluster_id_train"].fillna(-1).astype(int) != -1].copy()
    train, test = b.align_train_test_clusters_to_train_refit(train, test, site_train)
    centroids = b._cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )

    context_df = b._build_cluster_context(
        train,
        "dbscan_cluster_12km",
        b.CONTEXT_BASE_FEATURES_MONTHLY,
        include_target_aggs=True,
    )
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    monthly_series = _cluster_series_from_train(train, "dbscan_cluster_12km", "calendar_day", freq="MS")

    feature_candidates = (
        b.SITE_FEATURES_MONTHLY
        + b.TIME_LAG_FEATURES_MONTHLY
        + ["dbscan_cluster_12km"]
        + [c for c in train.columns if c.startswith("ctx_")]
    )
    feature_cols = [c for c in feature_candidates if c in train.columns and train[c].notna().any()]
    b.assert_less_than_no_weather_columns(feature_cols)

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    portable, metrics = b._train_ridge_portable(train, test, feature_cols)
    rf_bundle, rf_metrics = _train_rf_joblib_experiment(train, test, feature_cols)

    local_med = b._cluster_local_feature_medians(train, "dbscan_cluster_12km", feature_cols)
    context_records = b._context_records_with_local_medians(context_df, local_med)

    _save_json(out_dir / "wash_count_model_12km.portable.json", portable)
    joblib.dump(rf_bundle, out_dir / "wash_count_model_12km.rf.joblib")
    _save_json(out_dir / "training_metrics_12km.json", metrics)
    _save_json(out_dir / "training_metrics_rf_12km.json", rf_metrics)
    _save_json(out_dir / "cluster_centroids_12km.json", centroids)
    _save_json(out_dir / "cluster_context_12km.json", {
        "cluster_label": "dbscan_cluster_12km",
        "feature_cols": [c for c in context_df.columns if c != "cluster_id"],
        "records": context_records,
    })
    _save_json(out_dir / "cluster_monthly_series_12km.json", {"frequency": "MS", "series": {str(k): v for k, v in monthly_series.items()}})
    _save_json(out_dir / "feature_spec_12km.json", {
        "site_features_static": b.SITE_FEATURES_MONTHLY,
        "time_lag_features": b.TIME_LAG_FEATURES_MONTHLY,
        "cluster_context_features": [c for c in context_df.columns if c != "cluster_id"],
        "final_feature_order": feature_cols,
        "cluster_col": "dbscan_cluster_12km",
    })
    return {
        "n_train_sites": len(train_sites),
        "n_holdout_sites": len(holdout_sites),
        "ridge_metrics": metrics,
        "rf_metrics": rf_metrics,
    }


def _build_more_than_models(out_dir: Path, *, holdout_sites: set[int]) -> dict[str, Any]:
    master = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv", low_memory=False)
    more = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv", low_memory=False)
    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    more["calendar_day"] = pd.to_datetime(more["calendar_day"], errors="coerce")

    extra_from_more = [
        c for c in (
            ["dbscan_cluster_12km"]
            + b.TIME_LAG_FEATURES_DAILY
            + b.ANNUAL_WEATHER_FEATURES
            + ["region_enc", "state_enc", "costco_enc", "carwash_type_encoded"]
        )
        if c in more.columns and c not in master.columns
    ]
    key_cols = ["site_client_id", "calendar_day"]
    df = master.merge(more[key_cols + extra_from_more], on=key_cols, how="left")
    if "dbscan_cluster_12km" not in df.columns:
        raise RuntimeError(">2y: dbscan_cluster_12km missing after merge")

    df = df.dropna(subset=["site_client_id", "latitude", "longitude", "calendar_day"])
    all_sites = set(int(x) for x in df["site_client_id"].astype(int).unique())
    train_sites = all_sites - set(int(x) for x in holdout_sites)
    train = df[df["site_client_id"].astype(int).isin(train_sites)].copy()
    test = df[df["site_client_id"].astype(int).isin(holdout_sites)].copy()

    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = b._fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    # Drop noise sites from the train-only refit clustering.
    site_train = site_train[site_train["cluster_id_train"].fillna(-1).astype(int) != -1].copy()
    train, test = b.align_train_test_clusters_to_train_refit(train, test, site_train)
    centroids = b._cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )

    context_df = b._build_cluster_context(train, "dbscan_cluster_12km", b.CONTEXT_BASE_FEATURES)
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    monthly_series = _cluster_series_from_train(train, "dbscan_cluster_12km", "calendar_day", freq="MS")

    feature_candidates = (
        b.SITE_FEATURES_STATIC
        + b.ANNUAL_WEATHER_FEATURES
        + b.DAILY_WEATHER_FEATURES
        + b.TIME_LAG_FEATURES_DAILY
        + ["dbscan_cluster_12km"]
        + [c for c in train.columns if c.startswith("ctx_")]
    )
    feature_cols = [c for c in feature_candidates if c in train.columns and train[c].notna().any()]

    train = train.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])

    portable, metrics = b._train_ridge_portable(train, test, feature_cols)
    rf_bundle, rf_metrics = _train_rf_joblib_experiment(train, test, feature_cols)

    local_med = b._cluster_local_feature_medians(train, "dbscan_cluster_12km", feature_cols)
    context_records = b._context_records_with_local_medians(context_df, local_med)

    _save_json(out_dir / "wash_count_model_12km.portable.json", portable)
    joblib.dump(rf_bundle, out_dir / "wash_count_model_12km.rf.joblib")
    _save_json(out_dir / "training_metrics_12km.json", metrics)
    _save_json(out_dir / "training_metrics_rf_12km.json", rf_metrics)
    _save_json(out_dir / "cluster_centroids_12km.json", centroids)
    _save_json(out_dir / "cluster_context_12km.json", {
        "cluster_label": "dbscan_cluster_12km",
        "feature_cols": [c for c in context_df.columns if c != "cluster_id"],
        "records": context_records,
    })
    _save_json(out_dir / "cluster_monthly_series_12km.json", {"frequency": "MS", "series": {str(k): v for k, v in monthly_series.items()}})
    _save_json(out_dir / "feature_spec_12km.json", {
        "site_features_static": b.SITE_FEATURES_STATIC,
        "annual_weather_features": b.ANNUAL_WEATHER_FEATURES,
        "daily_weather_features": b.DAILY_WEATHER_FEATURES,
        "time_lag_features": b.TIME_LAG_FEATURES_DAILY,
        "cluster_context_features": [c for c in context_df.columns if c != "cluster_id"],
        "final_feature_order": feature_cols,
        "cluster_col": "dbscan_cluster_12km",
    })
    return {
        "n_train_sites": len(train_sites),
        "n_holdout_sites": len(holdout_sites),
        "ridge_metrics": metrics,
        "rf_metrics": rf_metrics,
    }


def _eval_site_holdout(
    *,
    run_root: Path,
    less_assets: dict[str, Any],
    more_assets: dict[str, Any],
    holdout_sites: set[int],
) -> dict[str, Any]:
    df_lt = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    df_gt = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
        low_memory=False,
    )

    df_lt = df_lt.dropna(subset=["site_client_id", "latitude", "longitude", "year_number"])
    df_gt = df_gt.dropna(subset=["site_client_id", "latitude", "longitude", "calendar_day"])

    # Precompute site metadata + actuals once (massive speedup vs per-site filtering).
    lt_meta = (
        df_lt[["site_client_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("site_client_id")
        .set_index("site_client_id")
    )
    lt_work = df_lt[["site_client_id", "year_number", TARGET]].copy()
    lt_work[TARGET] = pd.to_numeric(lt_work[TARGET], errors="coerce")
    lt_work["year_number"] = pd.to_numeric(lt_work["year_number"], errors="coerce")
    lt_work = lt_work.dropna(subset=["site_client_id", "year_number", TARGET])
    lt_sums = (
        lt_work.groupby(["site_client_id", "year_number"])[TARGET]
        .sum()
        .unstack(fill_value=0.0)
    )

    gt_meta = (
        df_gt[["site_client_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("site_client_id")
        .set_index("site_client_id")
    )
    gt_work = df_gt[["site_client_id", "calendar_day", TARGET]].copy()
    gt_work["calendar_day"] = pd.to_datetime(gt_work["calendar_day"], errors="coerce")
    gt_work[TARGET] = pd.to_numeric(gt_work[TARGET], errors="coerce")
    gt_work = gt_work.dropna(subset=["site_client_id", "calendar_day", TARGET])
    gt_work["year"] = gt_work["calendar_day"].dt.year
    gt_work = gt_work[gt_work["year"].isin([2024, 2025])]
    gt_sums = gt_work.groupby(["site_client_id", "year"])[TARGET].sum().unstack(fill_value=0.0)

    lt_sites = sorted(int(x) for x in holdout_sites if int(x) in set(df_lt["site_client_id"].astype(int).unique()))
    gt_sites = sorted(int(x) for x in holdout_sites if int(x) in set(df_gt["site_client_id"].astype(int).unique()))
    if MAX_HOLDOUT_SITES_EVAL is not None:
        if len(lt_sites) > MAX_HOLDOUT_SITES_EVAL:
            lt_sites = lt_sites[:MAX_HOLDOUT_SITES_EVAL]
        if len(gt_sites) > MAX_HOLDOUT_SITES_EVAL:
            gt_sites = gt_sites[:MAX_HOLDOUT_SITES_EVAL]

    out_rows: list[dict[str, Any]] = []
    by_variant: dict[str, Any] = {}

    for v in VARIANTS:
        lt_actual_y1: list[float] = []
        lt_pred_y1: list[float] = []
        lt_actual_y2: list[float] = []
        lt_pred_y2: list[float] = []

        gt_actual_2024: list[float] = []
        gt_pred_y3: list[float] = []
        gt_actual_2025: list[float] = []
        gt_pred_y4: list[float] = []

        for sid in lt_sites:
            if sid not in lt_meta.index:
                continue
            lat = float(lt_meta.loc[sid, "latitude"])
            lon = float(lt_meta.loc[sid, "longitude"])
            pred = _project_years_1_4(lat=lat, lon=lon, less_assets=less_assets, more_assets=more_assets, variant=v)
            act = {
                "actual_year_1": float(lt_sums.loc[sid, 1]) if sid in lt_sums.index and 1 in lt_sums.columns else 0.0,
                "actual_year_2": float(lt_sums.loc[sid, 2]) if sid in lt_sums.index and 2 in lt_sums.columns else 0.0,
            }
            lt_actual_y1.append(act["actual_year_1"])
            lt_pred_y1.append(pred["year_1"])
            lt_actual_y2.append(act["actual_year_2"])
            lt_pred_y2.append(pred["year_2"])
            out_rows.append({"variant": v.name, "cohort": "less_than_2yrs", "site_client_id": sid, **act, **{f"pred_year_{k[-1]}": pred[k] for k in ("year_1","year_2")}})

        for sid in gt_sites:
            if sid not in gt_meta.index:
                continue
            lat = float(gt_meta.loc[sid, "latitude"])
            lon = float(gt_meta.loc[sid, "longitude"])
            pred = _project_years_1_4(lat=lat, lon=lon, less_assets=less_assets, more_assets=more_assets, variant=v)
            act = {
                "actual_2024": float(gt_sums.loc[sid, 2024]) if sid in gt_sums.index and 2024 in gt_sums.columns else 0.0,
                "actual_2025": float(gt_sums.loc[sid, 2025]) if sid in gt_sums.index and 2025 in gt_sums.columns else 0.0,
            }
            gt_actual_2024.append(act["actual_2024"])
            gt_pred_y3.append(pred["year_3"])
            gt_actual_2025.append(act["actual_2025"])
            gt_pred_y4.append(pred["year_4"])
            out_rows.append({"variant": v.name, "cohort": "more_than_2yrs", "site_client_id": sid, **act, "pred_year_3": pred["year_3"], "pred_year_4": pred["year_4"]})

        by_variant[v.name] = {
            "variant": v.__dict__,
            "less_than_2yrs": {
                "n_sites": len(lt_sites),
                "year_1": _metric_block(lt_actual_y1, lt_pred_y1),
                "year_2": _metric_block(lt_actual_y2, lt_pred_y2),
                "combined_1_2": _metric_block(lt_actual_y1 + lt_actual_y2, lt_pred_y1 + lt_pred_y2),
            },
            "more_than_2yrs_mature_analogy": {
                "n_sites": len(gt_sites),
                "year_3_vs_2024": _metric_block(gt_actual_2024, gt_pred_y3),
                "year_4_vs_2025": _metric_block(gt_actual_2025, gt_pred_y4),
                "combined": _metric_block(gt_actual_2024 + gt_actual_2025, gt_pred_y3 + gt_pred_y4),
            },
        }

    def _rank_key(v: dict[str, Any]) -> float:
        a = v["less_than_2yrs"]["combined_1_2"].get("wape", float("inf"))
        b_ = v["more_than_2yrs_mature_analogy"]["combined"].get("wape", float("inf"))
        try:
            return float(a) + float(b_)
        except Exception:
            return float("inf")

    ranked = sorted(by_variant.values(), key=_rank_key)
    summary = {
        "ranking_rule": "Sort by (WAPE lt combined years 1-2) + (WAPE >2y mature analogy years 3-4). Lower is better.",
        "ranked_variants": [
            {
                "name": r["variant"]["name"],
                "score": round(_rank_key(r), 6),
                "lt_wape_1_2": r["less_than_2yrs"]["combined_1_2"].get("wape"),
                "gt_wape_3_4": r["more_than_2yrs_mature_analogy"]["combined"].get("wape"),
            }
            for r in ranked
        ],
    }

    return {
        "holdout_definition": {
            "type": "site_holdout_intersection",
            "holdout_frac": HOLDOUT_FRAC,
            "seed": RANDOM_SEED,
            "site_set": "intersection(less_than_2yrs, master_more_than_2yrs)",
            "max_holdout_sites_eval": MAX_HOLDOUT_SITES_EVAL,
            "noise_rule": "Train DBSCAN refit computed on train sites; training noise cluster (-1) sites removed from clustering inputs. Test sites always assigned to nearest non-noise centroid.",
        },
        "n_holdout_sites": {"less_than_2yrs": len(lt_sites), "more_than_2yrs": len(gt_sites)},
        "variants": by_variant,
        "summary": summary,
    }


def _write_report(run_root: Path, eval_blob: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Site-holdout 90/10 — model zoo report")
    lines.append("")
    lines.append(f"- Run: `{run_root.name}`")
    hd = eval_blob.get("holdout_definition") or {}
    lines.append(f"- Holdout: `{hd.get('type')}` frac={hd.get('holdout_frac')} seed={hd.get('seed')}")
    if hd.get("site_set"):
        lines.append(f"- Site set: `{hd.get('site_set')}`")
    if hd.get("max_holdout_sites_eval") is not None:
        lines.append(f"- Max holdout sites eval cap: `{hd.get('max_holdout_sites_eval')}`")
    lines.append("- TS methods: `arima`, `holt_winters`, `blend`, `meta` (auto-picks by rolling one-step MAE)")
    lines.append("")
    lines.append("## Overall ranking (lower is better)")
    for r in eval_blob["summary"]["ranked_variants"][:10]:
        lines.append(
            f"- `{r['name']}` score={r['score']}  lt_wape_1-2={r['lt_wape_1_2']}  gt_wape_3-4={r['gt_wape_3_4']}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `more_than_2yrs_mature_analogy` compares predicted operational Year 3/4 to actual calendar-year 2024/2025 totals of holdout mature sites (stage-aligned but not a true greenfield test).")
    lines.append("- `meta` uses only the cluster median monthly track to choose a TS method per forecast call; it does not train/save a separate TS model artifact.")
    (run_root / "results" / "REPORT.md").write_text("\n".join(lines) + "\n")


def _write_leaderboard(run_root: Path, eval_blob: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    score_map = {r["name"]: r["score"] for r in eval_blob["summary"]["ranked_variants"]}
    for name, v in eval_blob["variants"].items():
        lt = v["less_than_2yrs"]
        gt = v["more_than_2yrs_mature_analogy"]
        rows.append(
            {
                "variant": name,
                "score": score_map.get(name),
                "lt_wape_year_1": lt["year_1"].get("wape"),
                "lt_wape_year_2": lt["year_2"].get("wape"),
                "lt_wape_year_1_2": lt["combined_1_2"].get("wape"),
                "gt_wape_year_3": gt["year_3_vs_2024"].get("wape"),
                "gt_wape_year_4": gt["year_4_vs_2025"].get("wape"),
                "gt_wape_year_3_4": gt["combined"].get("wape"),
                "lt_n": lt["combined_1_2"].get("n"),
                "gt_n": gt["combined"].get("n"),
            }
        )
    df = pd.DataFrame(rows).sort_values(["score", "variant"], na_position="last")
    (run_root / "results" / "leaderboard.csv").write_text(df.to_csv(index=False))

    top = df.head(15).to_dict(orient="records")
    md: list[str] = []
    md.append("# Leaderboard")
    md.append("")
    md.append(f"- Run: `{run_root.name}`")
    hd = eval_blob.get("holdout_definition") or {}
    md.append(f"- Holdout: `{hd.get('type')}` frac={hd.get('holdout_frac')} seed={hd.get('seed')}")
    md.append("")
    md.append("| rank | variant | score | lt_wape(1-2) | gt_wape(3-4) | lt_n | gt_n |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(top, start=1):
        md.append(
            "| {rank} | `{variant}` | {score} | {lt} | {gt} | {ltn} | {gtn} |".format(
                rank=i,
                variant=r["variant"],
                score=r["score"],
                lt=r["lt_wape_year_1_2"],
                gt=r["gt_wape_year_3_4"],
                ltn=r["lt_n"],
                gtn=r["gt_n"],
            )
        )
    (run_root / "results" / "leaderboard.md").write_text("\n".join(md) + "\n")


def _write_holdout_sites(run_root: Path, holdout: set[int]) -> None:
    ids = sorted(int(x) for x in holdout)
    (run_root / "results" / "holdout_site_client_ids.json").write_text(json.dumps(ids, indent=2))

    lt = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    gt = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
        low_memory=False,
    )

    lt = lt.dropna(subset=["site_client_id"]).copy()
    gt = gt.dropna(subset=["site_client_id"]).copy()
    lt["site_client_id"] = lt["site_client_id"].astype(int)
    gt["site_client_id"] = gt["site_client_id"].astype(int)

    lt_addr_col = "address" if "address" in lt.columns else ("Address" if "Address" in lt.columns else None)
    lt_meta_cols = ["site_client_id", "latitude", "longitude"]
    if lt_addr_col:
        lt_meta_cols.append(lt_addr_col)
    lt_meta = lt[lt_meta_cols].drop_duplicates("site_client_id").set_index("site_client_id")

    # Build a readable address for >2y from available columns.
    gt_cols = ["site_client_id", "latitude", "longitude"]
    for c in ("street", "city", "state", "zip"):
        if c in gt.columns:
            gt_cols.append(c)
    gt_meta = gt[gt_cols].drop_duplicates("site_client_id").set_index("site_client_id")

    rows: list[dict[str, Any]] = []
    for sid in ids:
        r: dict[str, Any] = {"site_client_id": sid}
        if sid in lt_meta.index:
            r["lt_latitude"] = float(lt_meta.loc[sid, "latitude"]) if "latitude" in lt_meta.columns else None
            r["lt_longitude"] = float(lt_meta.loc[sid, "longitude"]) if "longitude" in lt_meta.columns else None
            if lt_addr_col and lt_addr_col in lt_meta.columns:
                r["lt_address"] = str(lt_meta.loc[sid, lt_addr_col])
        if sid in gt_meta.index:
            r["gt_latitude"] = float(gt_meta.loc[sid, "latitude"]) if "latitude" in gt_meta.columns else None
            r["gt_longitude"] = float(gt_meta.loc[sid, "longitude"]) if "longitude" in gt_meta.columns else None
            parts: list[str] = []
            for c in ("street", "city", "state", "zip"):
                if c in gt_meta.columns and pd.notna(gt_meta.loc[sid, c]):
                    parts.append(str(gt_meta.loc[sid, c]))
            if parts:
                r["gt_address"] = ", ".join(parts)
        rows.append(r)

    df = pd.DataFrame(rows)
    (run_root / "results" / "holdout_sites_intersection.csv").write_text(df.to_csv(index=False))


def _write_run_context(run_root: Path, build_summary: dict[str, Any], eval_blob: dict[str, Any]) -> None:
    hd = eval_blob.get("holdout_definition") or {}
    lines: list[str] = []
    lines.append("# Run context")
    lines.append("")
    lines.append(f"- Run id: `{build_summary.get('run_id')}`")
    if build_summary.get("repo_git_rev"):
        lines.append(f"- Git rev: `{build_summary.get('repo_git_rev')}`")
    lines.append(f"- Created (UTC): `{build_summary.get('created_utc')}`")
    lines.append("")
    lines.append("## What this run contains")
    lines.append("- Trained cohort models (level models + clustering artifacts):")
    lines.append(f"  - `<2y`: `{(run_root / 'models' / 'less_than').as_posix()}`")
    lines.append(f"  - `>2y`: `{(run_root / 'models' / 'more_than').as_posix()}`")
    lines.append("- Evaluation outputs:")
    lines.append(f"  - `{(run_root / 'results' / 'eval_site_holdout_90_10.json').as_posix()}`")
    lines.append(f"  - `{(run_root / 'results' / 'leaderboard.csv').as_posix()}`")
    lines.append(f"  - `{(run_root / 'results' / 'leaderboard.md').as_posix()}`")
    lines.append("")
    lines.append("## Evaluation protocol")
    lines.append(f"- Holdout type: `{hd.get('type')}`")
    lines.append(f"- Holdout fraction: `{hd.get('holdout_frac')}` seed=`{hd.get('seed')}`")
    if hd.get("site_set"):
        lines.append(f"- Site set for evaluation: `{hd.get('site_set')}`")
    if hd.get("noise_rule"):
        lines.append(f"- Noise handling: `{hd.get('noise_rule')}`")
    lines.append("")
    lines.append("## Variant meanings")
    lines.append("- Variant name format: `<level_model>_<ts_method>_<prefix/bridge setting>`")
    lines.append("  - `level_model`: `ridge` or `rf`")
    lines.append("  - `ts_method`: `arima`, `holt_winters`, `blend`, `meta`")
    lines.append("  - `prefix_bridge` means >2y TS is given the <2y 24-month prefix + optional bridge scaling")
    lines.append("  - `no_prefix` means >2y TS uses only the mature cluster series")
    lines.append("")
    best = (eval_blob.get("summary") or {}).get("ranked_variants", [None])[0] or {}
    if best.get("name"):
        lines.append("## Best result in this run")
        lines.append(
            f"- `{best.get('name')}` score={best.get('score')}  lt_wape(1-2)={best.get('lt_wape_1_2')}  gt_wape(3-4)={best.get('gt_wape_3_4')}"
        )
    (run_root / "results" / "RUN_CONTEXT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = EX_ROOT / "outputs_site_holdout_intersection" / run_id
    model_root = run_root / "models"
    results_root = run_root / "results"
    (model_root / "less_than").mkdir(parents=True, exist_ok=True)
    (model_root / "more_than").mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    build_summary = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"type": "site_holdout_intersection", "holdout_frac": HOLDOUT_FRAC, "seed": RANDOM_SEED},
        "repo_git_rev": _git_rev(),
        "rf_params": {
            "n_estimators": int(RF_N_ESTIMATORS),
            "max_depth": int(RF_MAX_DEPTH),
            "min_samples_leaf": int(RF_MIN_SAMPLES_LEAF),
        },
        "variants": [v.__dict__ for v in VARIANTS],
        "data_inputs": {
            "less_than_csv": "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
            "more_than_master_csv": "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
            "more_than_features_csv": "daily_data/daily-data-modelling/more_than-2yrs.csv",
        },
    }
    inter = _site_intersection()
    _, holdout = _site_split(inter, HOLDOUT_FRAC, RANDOM_SEED)
    build_summary["holdout_site_set"] = {
        "definition": "intersection(less_than_2yrs, master_more_than_2yrs)",
        "n_intersection_sites": int(len(inter)),
        "n_holdout_sites": int(len(holdout)),
        "max_holdout_sites_eval": MAX_HOLDOUT_SITES_EVAL,
    }
    build_summary["less_than"] = _build_less_than_models(model_root / "less_than", holdout_sites=holdout)
    build_summary["more_than"] = _build_more_than_models(model_root / "more_than", holdout_sites=holdout)
    _save_json(results_root / "build_summary.json", build_summary)

    less_assets = _load_assets(model_root / "less_than")
    more_assets = _load_assets(model_root / "more_than")

    eval_blob = _eval_site_holdout(
        run_root=run_root,
        less_assets=less_assets,
        more_assets=more_assets,
        holdout_sites=holdout,
    )
    _save_json(results_root / "eval_site_holdout_90_10.json", eval_blob)
    _write_report(run_root, eval_blob)
    _write_leaderboard(run_root, eval_blob)
    _write_holdout_sites(run_root, holdout)
    _write_run_context(run_root, build_summary, eval_blob)

    print(f"wrote {run_root}")
    print(f"report: {results_root / 'REPORT.md'}")
    print(f"eval:   {results_root / 'eval_site_holdout_90_10.json'}")
    print(f"leaderboard: {results_root / 'leaderboard.md'}")
    print(f"holdout sites: {results_root / 'holdout_sites_intersection.csv'}")
    print(f"context: {results_root / 'RUN_CONTEXT.md'}")


if __name__ == "__main__":
    main()
