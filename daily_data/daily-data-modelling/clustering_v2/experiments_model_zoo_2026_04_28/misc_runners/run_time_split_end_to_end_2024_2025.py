"""End-to-end time-split evaluation (train 2024 -> test 2025) for both cohorts.

>2y (daily data):
  - Train on 2024 daily rows: DBSCAN refit clusters, centroids, cluster context, cluster monthly median series,
    and a level model (ridge/rf/xgb).
  - Predict 2025 monthly totals per site using: (lat/lon -> nearest centroid -> anchor monthly level -> TS forecast 12m -> scale).
  - Score against actual 2025 monthly totals (summed from daily actuals).

<2y (monthly panel):
  - Train on 2024 monthly rows: DBSCAN refit clusters, centroids, and level model.
  - Predict 2025 monthly totals per site (12 months) and score vs actual 2025 monthly totals.

Per-site modeling: the **level model predicts a separate anchor per site** (from geo + cluster context).
The cluster monthly series only supplies a **shared seasonal shape**, scaled to that anchor — this matches
the greenfield design in project_site.py. WAPE pools site×month errors (standard for volume-weighted error).

TS methods include `prophet` (Facebook Prophet, if installed) and `meta` (rolling one-step MAE picks among
arima / holt_winters / blend / prophet).

This is a more "genuine" evaluation than the greenfield mature-analogy mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
warnings.filterwarnings("ignore")


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


def _train_ridge(train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    X = train[feature_cols].to_numpy(dtype=float)
    y = train[TARGET].to_numpy(dtype=float)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _train_rf(train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    from sklearn.ensemble import RandomForestRegressor

    X = train[feature_cols].to_numpy(dtype=float)
    y = train[TARGET].to_numpy(dtype=float)
    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "m",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_leaf=4,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _train_xgb(train: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    from xgboost import XGBRegressor

    X = train[feature_cols].to_numpy(dtype=float)
    y = train[TARGET].to_numpy(dtype=float)
    pipe = Pipeline(
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
    )
    pipe.fit(X, y)
    return pipe


@dataclass(frozen=True)
class Variant:
    name: str
    level_model: str  # ridge|rf|xgb
    ts_method: str  # arima|holt_winters|blend|prophet|meta


VARIANTS: list[Variant] = [
    Variant("ridge_arima", "ridge", "arima"),
    Variant("ridge_holt", "ridge", "holt_winters"),
    Variant("ridge_blend", "ridge", "blend"),
    Variant("rf_arima", "rf", "arima"),
    Variant("rf_holt", "rf", "holt_winters"),
    Variant("rf_blend", "rf", "blend"),
    Variant("xgb_arima", "xgb", "arima"),
    Variant("xgb_holt", "xgb", "holt_winters"),
    Variant("xgb_blend", "xgb", "blend"),
    Variant("ridge_prophet", "ridge", "prophet"),
    Variant("rf_prophet", "rf", "prophet"),
    Variant("xgb_prophet", "xgb", "prophet"),
    Variant("ridge_meta", "ridge", "meta"),
    Variant("rf_meta", "rf", "meta"),
    Variant("xgb_meta", "xgb", "meta"),
]


def _month_start(dt: pd.Series) -> pd.Series:
    # portable month-start in pandas without relying on MS period freq.
    d = pd.to_datetime(dt, errors="coerce")
    m = d.dt.to_period("M").dt.to_timestamp()
    return (m.dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthBegin(1)).astype("datetime64[ns]")


def _end_to_end_predict_2025_monthly(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str,
    date_col: str,
    variant: Variant,
    context_df: pd.DataFrame | None,
    centroids: list[dict[str, Any]],
    cluster_monthly_series: dict[int, pd.Series],
    is_daily_level_model: bool,
    test_year: int,
    precomputed_fc: dict[tuple[int, str], pd.Series],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if variant.level_model == "ridge":
        pipe = _train_ridge(train_df, feature_cols)
    elif variant.level_model == "rf":
        pipe = _train_rf(train_df, feature_cols)
    elif variant.level_model == "xgb":
        pipe = _train_xgb(train_df, feature_cols)
    else:
        raise ValueError(variant.level_model)

    # Prepare lookup maps.
    ctx_map: dict[int, dict[str, Any]] = {}
    if context_df is not None and "cluster_id" in context_df.columns:
        for r in context_df.to_dict(orient="records"):
            try:
                ctx_map[int(r["cluster_id"])] = r
            except Exception:
                continue

    site_meta = (
        test_df[["site_client_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("site_client_id")
        .copy()
    )
    site_meta["site_client_id"] = site_meta["site_client_id"].astype(int)

    # Actual 2025 monthly totals.
    act = test_df[["site_client_id", date_col, TARGET]].copy()
    act["site_client_id"] = act["site_client_id"].astype(int)
    act[TARGET] = pd.to_numeric(act[TARGET], errors="coerce")
    act = act.dropna(subset=[date_col, TARGET])
    act["month"] = _month_start(act[date_col])
    act_m = act.groupby(["site_client_id", "month"], as_index=False)[TARGET].sum().rename(columns={TARGET: "actual"})

    # Predict months in the test year for each site using precomputed per-cluster TS forecasts.
    preds: list[dict[str, Any]] = []
    for row in site_meta.to_dict(orient="records"):
        sid = int(row["site_client_id"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        nearest = ps._nearest_cluster(centroids, lat, lon)
        cid = int(nearest["cluster_id"])

        # Anchor monthly from level model using ctx row (cluster context from 2024 only).
        ctx_row = ctx_map.get(cid)
        feat_vec = ps._feature_vector(
            {"cluster_col": cluster_col},  # spec isn't used by _feature_vector beyond ctx values
            ctx_row,
            lat,
            lon,
            cid,
        )
        # The pipeline expects feature_cols; fill from ctx_row + geo + cluster id.
        vec = {c: feat_vec.get(c, np.nan) for c in feature_cols}
        X = np.array([[vec.get(c, np.nan) for c in feature_cols]], dtype=float)
        anchor_raw = float(pipe.predict(X)[0])
        anchor_monthly = anchor_raw * 30.0 if is_daily_level_model else anchor_raw

        series = cluster_monthly_series.get(cid, pd.Series(dtype=float))
        cluster_level = float(series.tail(6).mean()) if len(series) else float("nan")
        scale = float(anchor_monthly / cluster_level) if np.isfinite(anchor_monthly) and cluster_level > 0 else 1.0

        fc0 = precomputed_fc.get((cid, variant.ts_method))
        if fc0 is None or len(fc0) == 0:
            continue
        fc = fc0 * scale
        for month, v in fc.items():
            preds.append({"site_client_id": sid, "month": pd.Timestamp(month), "pred": float(max(v, 0.0))})

    pred_m = pd.DataFrame(preds)
    if pred_m.empty:
        return pred_m, {"n": 0, "note": "No predictions produced (missing cluster series or horizon)."}
    merged = act_m.merge(pred_m, on=["site_client_id", "month"], how="inner")
    metrics = _metric_block(merged["actual"].to_numpy(), merged["pred"].to_numpy())
    metrics["n_sites"] = int(merged["site_client_id"].nunique())
    metrics["n_site_months"] = int(len(merged))
    return merged, metrics


def _precompute_forecasts_for_test_year(
    *,
    cluster_monthly_series: dict[int, pd.Series],
    methods: list[str],
    test_year: int,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2024 / test 2025 end-to-end (greenfield-like). Use --only for a fast Prophet/meta comparison."
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated variant names (default: all 15). Example: xgb_arima,xgb_prophet,xgb_meta",
    )
    args = parser.parse_args()

    variants_to_run = list(VARIANTS)
    if args.only.strip():
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        variants_to_run = [v for v in VARIANTS if v.name in want]
        unknown = want - {v.name for v in VARIANTS}
        if unknown:
            raise SystemExit(f"Unknown variant names: {sorted(unknown)}")
        if not variants_to_run:
            raise SystemExit("--only matched nothing")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = EX_ROOT / "outputs_end_to_end_train2024_test2025" / run_id
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- >2y data: train 2024, test 2025 ----------------
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
    centroids = centroids_blob["centroids"]
    context_gt = b._build_cluster_context(train_gt, "dbscan_cluster_12km", b.CONTEXT_BASE_FEATURES)
    train_gt = train_gt.merge(context_gt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test_gt = test_gt.merge(context_gt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    series_gt_raw = b._cluster_monthly_series(train_gt, "dbscan_cluster_12km", "calendar_day", freq="MS")
    series_gt = {int(k): ps._series_to_df(v) for k, v in series_gt_raw.items()}

    # Restrict to "greenfield-available" features: geo + cluster id + 2024 peer context only.
    feature_cols_gt = ["latitude", "longitude", "dbscan_cluster_12km"] + [c for c in train_gt.columns if c.startswith("ctx_")]

    # ---------------- <2y data: train 2024, test 2025 (calendar_month epoch is correct) ----------------
    df_lt = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    # calendar_day is correct for months 1-12 (2024), but shifted +1y for months 13-24 (shows 2026 instead of 2025).
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
    context_lt = b._build_cluster_context(train_lt, "dbscan_cluster_12km", b.CONTEXT_BASE_FEATURES_MONTHLY, include_target_aggs=True)
    train_lt = train_lt.merge(context_lt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test_lt = test_lt.merge(context_lt, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    series_lt_raw = b._cluster_monthly_series(train_lt, "dbscan_cluster_12km", "month_dt", freq="MS")
    series_lt = {int(k): ps._series_to_df(v) for k, v in series_lt_raw.items()}

    feature_cols_lt = ["latitude", "longitude", "dbscan_cluster_12km"] + [c for c in train_lt.columns if c.startswith("ctx_")]

    out: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gt_train_days": int(len(train_gt)),
        "gt_test_days": int(len(test_gt)),
        "lt_train_rows": int(len(train_lt)),
        "lt_test_rows": int(len(test_lt)),
        "definition": {
            "gt": "Train 2024 daily rows, predict 2025 monthly site totals (end-to-end anchor+TS).",
            "lt": "Train 2024 monthly rows, predict 2025 monthly site totals (end-to-end anchor+TS).",
        },
        "variants": {},
        "variants_requested": [v.name for v in variants_to_run],
    }

    methods = sorted({v.ts_method for v in variants_to_run})
    fc_gt = _precompute_forecasts_for_test_year(cluster_monthly_series=series_gt, methods=methods, test_year=2025)
    fc_lt = _precompute_forecasts_for_test_year(cluster_monthly_series=series_lt, methods=methods, test_year=2025)

    # Evaluate variants for both cohorts.
    leaderboard_rows: list[dict[str, Any]] = []
    for v in variants_to_run:
        _, m_gt = _end_to_end_predict_2025_monthly(
            train_df=train_gt,
            test_df=test_gt,
            feature_cols=feature_cols_gt,
            cluster_col="dbscan_cluster_12km",
            date_col="calendar_day",
            variant=v,
            context_df=context_gt,
            centroids=centroids,
            cluster_monthly_series=series_gt,
            is_daily_level_model=True,
            test_year=2025,
            precomputed_fc=fc_gt,
        )
        _, m_lt = _end_to_end_predict_2025_monthly(
            train_df=train_lt,
            test_df=test_lt,
            feature_cols=feature_cols_lt,
            cluster_col="dbscan_cluster_12km",
            date_col="month_dt",
            variant=v,
            context_df=context_lt,
            centroids=centroids_lt,
            cluster_monthly_series=series_lt,
            is_daily_level_model=False,
            test_year=2025,
            precomputed_fc=fc_lt,
        )
        out["variants"][v.name] = {"gt_2025_monthly": m_gt, "lt_2025_monthly": m_lt}
        score = float(m_gt.get("wape", np.inf)) + float(m_lt.get("wape", np.inf))
        leaderboard_rows.append(
            {
                "variant": v.name,
                "level_model": v.level_model,
                "ts_method": v.ts_method,
                "score_wape_sum": round(score, 6),
                "gt_wape": m_gt.get("wape"),
                "lt_wape": m_lt.get("wape"),
                "gt_r2": m_gt.get("r2"),
                "lt_r2": m_lt.get("r2"),
                "gt_site_months": m_gt.get("n_site_months"),
                "lt_site_months": m_lt.get("n_site_months"),
            }
        )

    _save_json(results_dir / "time_split_e2e_2024_train_2025_test.json", out)
    df_lb = pd.DataFrame(leaderboard_rows).sort_values("score_wape_sum")
    (results_dir / "leaderboard.csv").write_text(df_lb.to_csv(index=False))

    md: list[str] = []
    md.append("# End-to-end time split leaderboard")
    md.append("")
    md.append("Definition:")
    md.append("- `<2y`: train 2024 monthly rows, test 2025 monthly totals (calendar_day for year2 is shifted in the CSV; corrected by -1 year).")
    md.append("- `>2y`: train 2024 daily rows, test 2025 monthly totals.")
    md.append("")
    md.append("Score = lt_wape(test year monthly) + gt_wape(test year monthly). Lower is better.")
    md.append("")
    md.append("| rank | variant | score | lt_wape | gt_wape | lt_r2 | gt_r2 |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(df_lb.to_dict(orient="records"), start=1):
        md.append(
            "| {rank} | `{variant}` | {score} | {lt} | {gt} | {ltr2} | {gtr2} |".format(
                rank=i,
                variant=r["variant"],
                score=r["score_wape_sum"],
                lt=r["lt_wape"],
                gt=r["gt_wape"],
                ltr2=r["lt_r2"],
                gtr2=r["gt_r2"],
            )
        )
    (results_dir / "leaderboard.md").write_text("\n".join(md) + "\n")
    print(f"wrote {run_root}")


if __name__ == "__main__":
    main()
