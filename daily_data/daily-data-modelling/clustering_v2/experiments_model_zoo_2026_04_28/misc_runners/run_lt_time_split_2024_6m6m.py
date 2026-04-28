"""<2y end-to-end time split inside 2024: train Jan-Jun, test Jul-Dec (6m/6m).

This is meant to provide a "genuine" evaluation for the <2y cohort even though
the dataset doesn't contain 2025 rows.

Pipeline evaluated (greenfield-like):
  - Fit DBSCAN 12km clusters on train-window sites (non-noise centroids only).
  - Assign each test site to nearest centroid.
  - Train a level model on train rows (geo + cluster id + 2024 peer context).
  - Build train-only cluster median monthly series and forecast next 6 months.
  - Scale TS forecast to the site anchor monthly level.
  - Score on site-month totals for the test window.
"""

from __future__ import annotations

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

warnings.filterwarnings("ignore")

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


def _month_start(ts: pd.Series) -> pd.Series:
    d = pd.to_datetime(ts, errors="coerce")
    m = d.dt.to_period("M").dt.to_timestamp()
    return (m.dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthBegin(1)).astype("datetime64[ns]")


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
    ts_method: str  # arima|holt_winters|blend


VARIANTS: list[Variant] = [
    Variant("ridge_arima", "ridge", "arima"),
    Variant("ridge_blend", "ridge", "blend"),
    Variant("ridge_holt", "ridge", "holt_winters"),
    Variant("rf_arima", "rf", "arima"),
    Variant("rf_blend", "rf", "blend"),
    Variant("rf_holt", "rf", "holt_winters"),
    Variant("xgb_arima", "xgb", "arima"),
    Variant("xgb_blend", "xgb", "blend"),
    Variant("xgb_holt", "xgb", "holt_winters"),
]


def _precompute_fc(
    *,
    series_by_cluster: dict[int, pd.Series],
    methods: list[str],
    horizon: int,
    test_months: set[pd.Timestamp],
) -> dict[tuple[int, str], pd.Series]:
    out: dict[tuple[int, str], pd.Series] = {}
    for cid, s in series_by_cluster.items():
        if s is None or len(s) == 0:
            continue
        for m in methods:
            fc = ps._forecast(s, horizon, m)
            fc = fc[fc.index.isin(test_months)]
            out[(int(cid), str(m))] = fc
    return out


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = EX_ROOT / "outputs_lt2y_within_2024_train6m_test6m" / run_id
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    # Use period_index for within-year split. (calendar_day is shifted for year2; year1 is fine.)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    df["period_index"] = pd.to_numeric(df["period_index"], errors="coerce")
    df = df.dropna(subset=["site_client_id", "latitude", "longitude", "calendar_day", "period_index"])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    # year_number==1 corresponds to the first 12 months. Use period_index <=12.
    df_2024 = df[df["period_index"] <= 12].copy()
    if df_2024.empty:
        raise SystemExit("No 2024 rows in <2y dataset.")

    # 6m/6m split inside the first year.
    train = df_2024[df_2024["period_index"] <= 6].copy()
    test = df_2024[(df_2024["period_index"] >= 7) & (df_2024["period_index"] <= 12)].copy()

    print(f"[lt 6m/6m] train_rows={len(train):,} test_rows={len(test):,} train_sites={train['site_client_id'].nunique():,} test_sites={test['site_client_id'].nunique():,}")

    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = b._fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    site_train = site_train[site_train["cluster_id_train"].fillna(-1).astype(int) != -1].copy()
    train, test = b.align_train_test_clusters_to_train_refit(train, test, site_train)

    centroids_blob = b._cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    centroids = centroids_blob["centroids"]

    context = b._build_cluster_context(
        train,
        "dbscan_cluster_12km",
        b.CONTEXT_BASE_FEATURES_MONTHLY,
        include_target_aggs=True,
    )
    train = train.merge(context, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    series_raw = b._cluster_monthly_series(train, "dbscan_cluster_12km", "calendar_day", freq="MS")
    series = {int(k): ps._series_to_df(v) for k, v in series_raw.items()}

    feature_cols = ["latitude", "longitude", "dbscan_cluster_12km"] + [c for c in train.columns if c.startswith("ctx_")]
    feature_cols = [c for c in feature_cols if c in train.columns]

    methods = sorted({v.ts_method for v in VARIANTS})
    test_months = set(pd.date_range("2024-07-01", periods=6, freq="MS"))
    fc_by_cluster = _precompute_fc(series_by_cluster=series, methods=methods, horizon=6, test_months=test_months)

    # Actual test monthly totals.
    act = test[["site_client_id", "calendar_day", TARGET]].copy()
    act["site_client_id"] = act["site_client_id"].astype(int)
    act["month"] = _month_start(act["calendar_day"])
    act_m = act.groupby(["site_client_id", "month"], as_index=False)[TARGET].sum().rename(columns={TARGET: "actual"})

    site_meta = (
        test[["site_client_id", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates("site_client_id")
        .copy()
    )
    site_meta["site_client_id"] = site_meta["site_client_id"].astype(int)

    ctx_map: dict[int, dict[str, Any]] = {}
    for r in context.to_dict(orient="records"):
        try:
            ctx_map[int(r["cluster_id"])] = r
        except Exception:
            continue

    out: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "Train 2024-01..2024-06; test 2024-07..2024-12; end-to-end anchor+cluster-TS using greenfield-like features.",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sites": int(train["site_client_id"].nunique()),
        "test_sites": int(test["site_client_id"].nunique()),
        "variants": {},
    }

    leaderboard: list[dict[str, Any]] = []
    for v in VARIANTS:
        if v.level_model == "ridge":
            pipe = _train_ridge(train, feature_cols)
        elif v.level_model == "rf":
            pipe = _train_rf(train, feature_cols)
        elif v.level_model == "xgb":
            pipe = _train_xgb(train, feature_cols)
        else:
            raise ValueError(v.level_model)

        preds: list[dict[str, Any]] = []
        for r in site_meta.to_dict(orient="records"):
            sid = int(r["site_client_id"])
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            nearest = ps._nearest_cluster(centroids, lat, lon)
            cid = int(nearest["cluster_id"])
            fc0 = fc_by_cluster.get((cid, v.ts_method))
            if fc0 is None or len(fc0) == 0:
                continue

            ctx_row = ctx_map.get(cid)
            feat_vec = ps._feature_vector({"cluster_col": "dbscan_cluster_12km"}, ctx_row, lat, lon, cid)
            x = np.array([[feat_vec.get(c, np.nan) for c in feature_cols]], dtype=float)
            anchor_monthly = float(pipe.predict(x)[0])

            s = series.get(cid, pd.Series(dtype=float))
            cluster_level = float(s.tail(3).mean()) if len(s) else float("nan")
            scale = float(anchor_monthly / cluster_level) if np.isfinite(anchor_monthly) and cluster_level > 0 else 1.0

            fc = fc0 * scale
            for m, val in fc.items():
                preds.append({"site_client_id": sid, "month": pd.Timestamp(m), "pred": float(max(val, 0.0))})

        pred_m = pd.DataFrame(preds)
        merged = act_m.merge(pred_m, on=["site_client_id", "month"], how="inner")
        met = _metric_block(merged["actual"].to_numpy(), merged["pred"].to_numpy())
        met["n_sites"] = int(merged["site_client_id"].nunique())
        met["n_site_months"] = int(len(merged))
        out["variants"][v.name] = met
        leaderboard.append(
            {
                "variant": v.name,
                "level_model": v.level_model,
                "ts_method": v.ts_method,
                "wape": met.get("wape"),
                "mae": met.get("mae"),
                "rmse": met.get("rmse"),
                "r2": met.get("r2"),
                "pct_within_20pct": met.get("pct_within_20pct"),
                "n_site_months": met.get("n_site_months"),
            }
        )

    _save_json(results_dir / "lt_time_split_2024_6m6m.json", out)
    df_lb = pd.DataFrame(leaderboard).sort_values("wape")
    (results_dir / "leaderboard.csv").write_text(df_lb.to_csv(index=False))

    md: list[str] = []
    md.append("# <2y 2024 time split (6m train / 6m test) — leaderboard")
    md.append("")
    md.append("- Train: 2024-01..2024-06")
    md.append("- Test:  2024-07..2024-12")
    md.append("")
    md.append("| rank | variant | wape | mae | rmse | r2 | pct_within_20pct | n_site_months |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(df_lb.to_dict(orient="records"), start=1):
        md.append(
            "| {rank} | `{variant}` | {wape} | {mae} | {rmse} | {r2} | {p20} | {n} |".format(
                rank=i,
                variant=r.get("variant"),
                wape=r.get("wape"),
                mae=r.get("mae"),
                rmse=r.get("rmse"),
                r2=r.get("r2"),
                p20=r.get("pct_within_20pct"),
                n=r.get("n_site_months"),
            )
        )
    (results_dir / "leaderboard.md").write_text("\n".join(md) + "\n")

    print(f"wrote {run_root}")


if __name__ == "__main__":
    main()
