"""Temporal train-2024 / test-2025 with time-series shape variants + 4y ARIMA outlook.

Compares end-to-end (level XGB + ARIMA cluster scaling) when the monthly TS *shape*
comes from:
  1) per-cluster history (baseline, same idea as run_time_split_end_to_end_2024_2025.py)
  2) cohort-wide mean of all cluster monthly medians (>2y and <2y separately)
  3) cross-cohort mean: average of the two cohort mean curves on a common month index

Also fits ARIMA on each mean curve and reports **illustrative** calendar-year sums
for 2025–2028 (48 months); there is no ground truth for 2026+ in this dataset slice.

Outputs: ../outputs_temporal_ts_mean_curves/<run_id>/results/report.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

EX_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = V2_DIR.parents[2]
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(REPO_ROOT))

import build_v2 as b  # noqa: E402
import project_site as ps  # noqa: E402

TARGET = b.TARGET


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _metric_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    ae = np.abs(err)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "actual_sum": round(float(np.sum(y)), 2),
        "pred_sum": round(float(np.sum(p)), 2),
        "bias_pct_of_actual_sum": round(float(np.sum(err) / max(np.sum(y), 1e-9)) * 100.0, 2),
        "mae": round(float(np.mean(ae)), 2),
        "rmse": round(_rmse(y, p), 2),
        "wape": round(_wape(y, p), 4),
        "r2": round(float(r2_score(y, p)), 4),
    }
    pos = y > 0
    if np.any(pos):
        ape = ae[pos] / y[pos]
        for thr in (0.10, 0.15, 0.20, 0.30):
            out[f"pct_within_{int(thr * 100)}pct"] = round(float(np.mean(ape <= thr)) * 100.0, 2)
    return out


def _month_start(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, errors="coerce")
    m = d.dt.to_period("M").dt.to_timestamp()
    return (m.dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthBegin(1)).astype("datetime64[ns]")


def _train_xgb(train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    from sklearn.pipeline import Pipeline as SkPipeline

    X = train[feature_cols].to_numpy(dtype=float)
    y = train[TARGET].to_numpy(dtype=float)
    return SkPipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "m",
                XGBRegressor(
                    n_estimators=600,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    ).fit(X, y)


def _mean_series_across_clusters(cluster_monthly: dict[int, pd.Series]) -> pd.Series:
    """Equal-weight mean of cluster median monthly series on the union month index."""
    nonempty = [s for s in cluster_monthly.values() if s is not None and len(s) > 0]
    if not nonempty:
        return pd.Series(dtype=float)
    idx = nonempty[0].index
    for s in nonempty[1:]:
        idx = idx.union(s.index)
    idx = idx.sort_values()
    rows = []
    for s in nonempty:
        rows.append(s.reindex(idx).astype(float).to_numpy())
    stacked = np.vstack(rows)
    return pd.Series(np.nanmean(stacked, axis=0), index=idx).sort_index()


def _cross_cohort_mean(s_gt: pd.Series, s_lt: pd.Series) -> pd.Series:
    idx = s_gt.index.union(s_lt.index).sort_values()
    a = s_gt.reindex(idx).astype(float)
    b = s_lt.reindex(idx).astype(float)
    return pd.concat([a, b], axis=1).mean(axis=1, skipna=True).dropna()


def _fc_for_test_year_from_series(s: pd.Series, method: str, test_year: int) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    train_end = pd.Timestamp(s.index.max())
    test_end = pd.Timestamp(f"{int(test_year)}-12-01")
    months_ahead = int((test_end.year - train_end.year) * 12 + (test_end.month - train_end.month))
    if months_ahead <= 0:
        return pd.Series(dtype=float)
    fc = ps._forecast(s, months_ahead, method)
    return fc[fc.index.year == int(test_year)]


def _predict_2025_uniform_shape(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    pipe: Pipeline,
    centroids: list[dict[str, Any]],
    ctx_map: dict[int, dict[str, Any]],
    cluster_col: str,
    date_col: str,
    ref_series: pd.Series,
    fc_2025: pd.Series,
    is_daily_level_model: bool,
) -> dict[str, Any]:
    """Same site loop as e2e, but TS shape + scale reference use ``ref_series`` / ``fc_2025`` for every site."""
    ref_tail = float(ref_series.tail(6).mean()) if len(ref_series) else float("nan")
    if not (np.isfinite(ref_tail) and ref_tail > 0):
        return {"n": 0, "note": "bad ref_series tail mean"}

    site_meta = (
        test_df[["site_client_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("site_client_id")
        .copy()
    )
    site_meta["site_client_id"] = site_meta["site_client_id"].astype(int)

    act = test_df[["site_client_id", date_col, TARGET]].copy()
    act["site_client_id"] = act["site_client_id"].astype(int)
    act[TARGET] = pd.to_numeric(act[TARGET], errors="coerce")
    act = act.dropna(subset=[date_col, TARGET])
    act["month"] = _month_start(act[date_col])
    act_m = act.groupby(["site_client_id", "month"], as_index=False)[TARGET].sum().rename(columns={TARGET: "actual"})

    preds: list[dict[str, Any]] = []
    for row in site_meta.to_dict(orient="records"):
        sid = int(row["site_client_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        nearest = ps._nearest_cluster(centroids, lat, lon)
        cid = int(nearest["cluster_id"])
        ctx_row = ctx_map.get(cid)
        feat_vec = ps._feature_vector({"cluster_col": cluster_col}, ctx_row, lat, lon, cid)
        vec = {c: feat_vec.get(c, np.nan) for c in feature_cols}
        X = np.array([[vec.get(c, np.nan) for c in feature_cols]], dtype=float)
        anchor_raw = float(pipe.predict(X)[0])
        anchor_monthly = anchor_raw * 30.0 if is_daily_level_model else anchor_raw
        scale = float(anchor_monthly / ref_tail) if np.isfinite(anchor_monthly) else 1.0
        for month, v in fc_2025.items():
            preds.append({"site_client_id": sid, "month": pd.Timestamp(month), "pred": float(max(float(v) * scale, 0.0))})

    pred_m = pd.DataFrame(preds)
    if pred_m.empty:
        return {"n": 0, "note": "empty preds"}
    merged = act_m.merge(pred_m, on=["site_client_id", "month"], how="inner")
    m = _metric_block(merged["actual"].to_numpy(), merged["pred"].to_numpy())
    m["n_sites"] = int(merged["site_client_id"].nunique())
    m["n_site_months"] = int(len(merged))
    return m


def _predict_2025_per_cluster(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    pipe: Pipeline,
    centroids: list[dict[str, Any]],
    ctx_map: dict[int, dict[str, Any]],
    cluster_col: str,
    date_col: str,
    cluster_monthly_series: dict[int, pd.Series],
    precomputed_fc: dict[tuple[int, str], pd.Series],
    ts_method: str,
    is_daily_level_model: bool,
) -> dict[str, Any]:
    site_meta = (
        test_df[["site_client_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("site_client_id")
        .copy()
    )
    site_meta["site_client_id"] = site_meta["site_client_id"].astype(int)

    act = test_df[["site_client_id", date_col, TARGET]].copy()
    act["site_client_id"] = act["site_client_id"].astype(int)
    act[TARGET] = pd.to_numeric(act[TARGET], errors="coerce")
    act = act.dropna(subset=[date_col, TARGET])
    act["month"] = _month_start(act[date_col])
    act_m = act.groupby(["site_client_id", "month"], as_index=False)[TARGET].sum().rename(columns={TARGET: "actual"})

    preds: list[dict[str, Any]] = []
    for row in site_meta.to_dict(orient="records"):
        sid = int(row["site_client_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        nearest = ps._nearest_cluster(centroids, lat, lon)
        cid = int(nearest["cluster_id"])
        ctx_row = ctx_map.get(cid)
        feat_vec = ps._feature_vector({"cluster_col": cluster_col}, ctx_row, lat, lon, cid)
        vec = {c: feat_vec.get(c, np.nan) for c in feature_cols}
        X = np.array([[vec.get(c, np.nan) for c in feature_cols]], dtype=float)
        anchor_raw = float(pipe.predict(X)[0])
        anchor_monthly = anchor_raw * 30.0 if is_daily_level_model else anchor_raw
        series = cluster_monthly_series.get(cid, pd.Series(dtype=float))
        cluster_level = float(series.tail(6).mean()) if len(series) else float("nan")
        scale = float(anchor_monthly / cluster_level) if np.isfinite(anchor_monthly) and cluster_level > 0 else 1.0
        fc0 = precomputed_fc.get((cid, ts_method))
        if fc0 is None or len(fc0) == 0:
            continue
        fc = fc0 * scale
        for month, v in fc.items():
            preds.append({"site_client_id": sid, "month": pd.Timestamp(month), "pred": float(max(v, 0.0))})

    pred_m = pd.DataFrame(preds)
    if pred_m.empty:
        return {"n": 0, "note": "empty preds"}
    merged = act_m.merge(pred_m, on=["site_client_id", "month"], how="inner")
    m = _metric_block(merged["actual"].to_numpy(), merged["pred"].to_numpy())
    m["n_sites"] = int(merged["site_client_id"].nunique())
    m["n_site_months"] = int(len(merged))
    return m


def _precompute_per_cluster(
    cluster_monthly_series: dict[int, pd.Series], methods: list[str], test_year: int
) -> dict[tuple[int, str], pd.Series]:
    out: dict[tuple[int, str], pd.Series] = {}
    for cid, s in cluster_monthly_series.items():
        if s is None or len(s) == 0:
            continue
        train_end = pd.Timestamp(s.index.max())
        test_end = pd.Timestamp(f"{int(test_year)}-12-01")
        months_ahead = int((test_end.year - train_end.year) * 12 + (test_end.month - train_end.month))
        if months_ahead <= 0:
            continue
        for m in methods:
            fc = ps._forecast(s, months_ahead, m)
            fc = fc[fc.index.year == int(test_year)]
            out[(int(cid), str(m))] = fc
    return out


def _yearly_sums_from_forecast(fc: pd.Series) -> dict[str, float]:
    if fc is None or len(fc) == 0:
        return {}
    s = fc.groupby(fc.index.year, sort=True).sum()
    return {str(int(y)): round(float(v), 2) for y, v in s.items()}


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = EX_ROOT / "outputs_temporal_ts_mean_curves" / run_id
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----- >2y -----
    master = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv", low_memory=False)
    more = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv", low_memory=False)
    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    more["calendar_day"] = pd.to_datetime(more["calendar_day"], errors="coerce")
    extra_from_more = [
        c
        for c in (
            ["dbscan_cluster_12km"]
            + b.TIME_LAG_FEATURES_DAILY
            + b.ANNUAL_WEATHER_FEATURES
            + ["region_enc", "state_enc", "costco_enc", "carwash_type_encoded"]
        )
        if c in more.columns and c not in master.columns
    ]
    key_cols = ["site_client_id", "calendar_day"]
    df_gt = master.merge(more[key_cols + extra_from_more], on=key_cols, how="left")
    df_gt = df_gt.dropna(subset=["site_client_id", "latitude", "longitude", "calendar_day"])
    df_gt[TARGET] = pd.to_numeric(df_gt[TARGET], errors="coerce")
    df_gt = df_gt.dropna(subset=[TARGET])
    train_gt = df_gt[df_gt["calendar_day"].dt.year == 2024].copy()
    test_gt = df_gt[df_gt["calendar_day"].dt.year == 2025].copy()

    site_train = (
        train_gt.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = b._fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    site_train = site_train[site_train["cluster_id_train"].fillna(-1).astype(int) != -1].copy()
    train_gt, test_gt = b.align_train_test_clusters_to_train_refit(train_gt, test_gt, site_train)

    centroids_blob = b._cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    centroids_gt = centroids_blob["centroids"]
    context_gt = b._build_cluster_context(train_gt, "dbscan_cluster_12km", b.CONTEXT_BASE_FEATURES)
    train_gt = train_gt.merge(context_gt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(
        columns=["cluster_id"]
    )
    test_gt = test_gt.merge(context_gt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(
        columns=["cluster_id"]
    )
    series_gt_raw = b._cluster_monthly_series(train_gt, "dbscan_cluster_12km", "calendar_day", freq="MS")
    series_gt = {int(k): ps._series_to_df(v) for k, v in series_gt_raw.items()}
    feature_cols_gt = ["latitude", "longitude", "dbscan_cluster_12km"] + [c for c in train_gt.columns if c.startswith("ctx_")]

    # ----- <2y -----
    df_lt = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    df_lt["month_number"] = pd.to_numeric(df_lt.get("month_number"), errors="coerce")
    df_lt["month_dt"] = pd.to_datetime(df_lt["calendar_day"], errors="coerce")
    df_lt.loc[df_lt["month_number"] >= 13, "month_dt"] = df_lt.loc[df_lt["month_number"] >= 13, "month_dt"] - pd.DateOffset(years=1)
    df_lt["month_dt"] = _month_start(df_lt["month_dt"])
    df_lt = df_lt.dropna(subset=["site_client_id", "latitude", "longitude", "month_dt"])
    df_lt[TARGET] = pd.to_numeric(df_lt[TARGET], errors="coerce")
    df_lt = df_lt.dropna(subset=[TARGET])
    train_lt = df_lt[df_lt["month_dt"].dt.year == 2024].copy()
    test_lt = df_lt[df_lt["month_dt"].dt.year == 2025].copy()

    site_train_lt = (
        train_lt.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters_lt = b._fit_dbscan(site_train_lt, "cluster_id_train")
    site_train_lt = site_train_lt.merge(site_clusters_lt, on="site_client_id", how="left")
    site_train_lt = site_train_lt[site_train_lt["cluster_id_train"].fillna(-1).astype(int) != -1].copy()
    train_lt, test_lt = b.align_train_test_clusters_to_train_refit(train_lt, test_lt, site_train_lt)

    centroids_lt_blob = b._cluster_centroids(
        site_train_lt.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    centroids_lt = centroids_lt_blob["centroids"]
    context_lt = b._build_cluster_context(
        train_lt, "dbscan_cluster_12km", b.CONTEXT_BASE_FEATURES_MONTHLY, include_target_aggs=True
    )
    train_lt = train_lt.merge(context_lt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(
        columns=["cluster_id"]
    )
    test_lt = test_lt.merge(context_lt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(
        columns=["cluster_id"]
    )
    series_lt_raw = b._cluster_monthly_series(train_lt, "dbscan_cluster_12km", "month_dt", freq="MS")
    series_lt = {int(k): ps._series_to_df(v) for k, v in series_lt_raw.items()}
    feature_cols_lt = ["latitude", "longitude", "dbscan_cluster_12km"] + [c for c in train_lt.columns if c.startswith("ctx_")]

    pipe_gt = _train_xgb(train_gt, feature_cols_gt)
    pipe_lt = _train_xgb(train_lt, feature_cols_lt)

    ctx_gt: dict[int, dict[str, Any]] = {}
    for r in context_gt.to_dict(orient="records"):
        try:
            ctx_gt[int(r["cluster_id"])] = r
        except Exception:
            continue
    ctx_lt: dict[int, dict[str, Any]] = {}
    for r in context_lt.to_dict(orient="records"):
        try:
            ctx_lt[int(r["cluster_id"])] = r
        except Exception:
            continue

    avg_gt = _mean_series_across_clusters(series_gt)
    avg_lt = _mean_series_across_clusters(series_lt)
    avg_cross = _cross_cohort_mean(avg_gt, avg_lt)

    fc_gt_per = _precompute_per_cluster(series_gt, ["arima"], 2025)
    fc_lt_per = _precompute_per_cluster(series_lt, ["arima"], 2025)

    fc_gt_avg = _fc_for_test_year_from_series(avg_gt, "arima", 2025)
    fc_lt_avg = _fc_for_test_year_from_series(avg_lt, "arima", 2025)
    fc_cross_avg = _fc_for_test_year_from_series(avg_cross, "arima", 2025)

    rows: dict[str, Any] = {
        "run_id": run_id,
        "definition": {
            "train": "2024 calendar rows (daily >2y, monthly <2y)",
            "test": "2025 monthly site totals",
            "level_model": "xgb",
            "ts": "arima(1,1,1) on monthly cluster medians or cohort-mean curves",
        },
        "cohort_mean_series_points": {
            ">2y_clusters": int(len(series_gt)),
            "<2y_clusters": int(len(series_lt)),
            "avg_gt_months": int(len(avg_gt)),
            "avg_lt_months": int(len(avg_lt)),
            "avg_cross_months": int(len(avg_cross)),
        },
        "test_2025_metrics": {},
    }

    # Per-cluster baseline
    m_gt_pc = _predict_2025_per_cluster(
        train_df=train_gt,
        test_df=test_gt,
        feature_cols=feature_cols_gt,
        pipe=pipe_gt,
        centroids=centroids_gt,
        ctx_map=ctx_gt,
        cluster_col="dbscan_cluster_12km",
        date_col="calendar_day",
        cluster_monthly_series=series_gt,
        precomputed_fc=fc_gt_per,
        ts_method="arima",
        is_daily_level_model=True,
    )
    m_lt_pc = _predict_2025_per_cluster(
        train_df=train_lt,
        test_df=test_lt,
        feature_cols=feature_cols_lt,
        pipe=pipe_lt,
        centroids=centroids_lt,
        ctx_map=ctx_lt,
        cluster_col="dbscan_cluster_12km",
        date_col="month_dt",
        cluster_monthly_series=series_lt,
        precomputed_fc=fc_lt_per,
        ts_method="arima",
        is_daily_level_model=False,
    )
    rows["test_2025_metrics"]["per_cluster_arima"] = {
        ">2y": m_gt_pc,
        "<2y": m_lt_pc,
        "score_wape_sum": round(float(m_gt_pc.get("wape", np.nan)) + float(m_lt_pc.get("wape", np.nan)), 6),
    }

    # Uniform cohort mean (>2y uses avg_gt, <2y uses avg_lt)
    m_gt_u1 = _predict_2025_uniform_shape(
        train_df=train_gt,
        test_df=test_gt,
        feature_cols=feature_cols_gt,
        pipe=pipe_gt,
        centroids=centroids_gt,
        ctx_map=ctx_gt,
        cluster_col="dbscan_cluster_12km",
        date_col="calendar_day",
        ref_series=avg_gt,
        fc_2025=fc_gt_avg,
        is_daily_level_model=True,
    )
    m_lt_u1 = _predict_2025_uniform_shape(
        train_df=train_lt,
        test_df=test_lt,
        feature_cols=feature_cols_lt,
        pipe=pipe_lt,
        centroids=centroids_lt,
        ctx_map=ctx_lt,
        cluster_col="dbscan_cluster_12km",
        date_col="month_dt",
        ref_series=avg_lt,
        fc_2025=fc_lt_avg,
        is_daily_level_model=False,
    )
    rows["test_2025_metrics"]["cohort_mean_ts_separate"] = {
        ">2y": m_gt_u1,
        "<2y": m_lt_u1,
        "score_wape_sum": round(float(m_gt_u1.get("wape", np.nan)) + float(m_lt_u1.get("wape", np.nan)), 6),
    }

    m_gt_x = _predict_2025_uniform_shape(
        train_df=train_gt,
        test_df=test_gt,
        feature_cols=feature_cols_gt,
        pipe=pipe_gt,
        centroids=centroids_gt,
        ctx_map=ctx_gt,
        cluster_col="dbscan_cluster_12km",
        date_col="calendar_day",
        ref_series=avg_cross,
        fc_2025=fc_cross_avg,
        is_daily_level_model=True,
    )
    m_lt_x = _predict_2025_uniform_shape(
        train_df=train_lt,
        test_df=test_lt,
        feature_cols=feature_cols_lt,
        pipe=pipe_lt,
        centroids=centroids_lt,
        ctx_map=ctx_lt,
        cluster_col="dbscan_cluster_12km",
        date_col="month_dt",
        ref_series=avg_cross,
        fc_2025=fc_cross_avg,
        is_daily_level_model=False,
    )
    rows["test_2025_metrics"]["cross_cohort_mean_ts"] = {
        ">2y": m_gt_x,
        "<2y": m_lt_x,
        "score_wape_sum": round(float(m_gt_x.get("wape", np.nan)) + float(m_lt_x.get("wape", np.nan)), 6),
    }

    # Illustrative 48-month ARIMA from end of each mean curve (unscaled cluster units).
    tail_ref = 6
    fc48_gt = ps._forecast(avg_gt, 48, "arima") if len(avg_gt) else pd.Series(dtype=float)
    fc48_lt = ps._forecast(avg_lt, 48, "arima") if len(avg_lt) else pd.Series(dtype=float)
    fc48_cross = ps._forecast(avg_cross, 48, "arima") if len(avg_cross) else pd.Series(dtype=float)
    rows["illustrative_arima_48m_unscaled"] = {
        "note": "Monthly sums in same units as cluster median series (not site-level). No labels after 2025.",
        "avg_gt_yearly_totals": _yearly_sums_from_forecast(fc48_gt),
        "avg_lt_yearly_totals": _yearly_sums_from_forecast(fc48_lt),
        "avg_cross_yearly_totals": _yearly_sums_from_forecast(fc48_cross),
        "avg_gt_level_ref_last6m_mean": round(float(avg_gt.tail(tail_ref).mean()), 4) if len(avg_gt) else None,
        "avg_lt_level_ref_last6m_mean": round(float(avg_lt.tail(tail_ref).mean()), 4) if len(avg_lt) else None,
        "avg_cross_level_ref_last6m_mean": round(float(avg_cross.tail(tail_ref).mean()), 4) if len(avg_cross) else None,
    }

    out_path = results_dir / "report.json"
    _save_json(out_path, rows)
    print(json.dumps(rows["test_2025_metrics"], indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
