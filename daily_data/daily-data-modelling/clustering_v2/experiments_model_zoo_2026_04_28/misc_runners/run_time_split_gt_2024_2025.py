"""More-than-2yrs genuine time-split evaluation: train on 2024, test on 2025.

This evaluates the *level model* for mature sites using a real temporal split,
instead of the greenfield "mature analogy" mapping Year3/4 to calendar years.

Outputs a versioned run folder under ../outputs_mature_level_train2024_test2025/<run_id>/ with:
  - models/more_than/* artifacts
  - results/time_split_gt_2024_train_2025_test.json
  - results/leaderboard.md
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


def _portable_from_ridge_pipe(pipe: Pipeline, feature_cols: list[str]) -> dict[str, Any]:
    imp: SimpleImputer = pipe.named_steps["imp"]
    sc: StandardScaler = pipe.named_steps["sc"]
    reg: Ridge = pipe.named_steps["m"]
    return {
        "feature_order": list(feature_cols),
        "imputer": {"strategy": "median", "statistics": imp.statistics_.tolist()},
        "scaler": {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
        "ridge": {
            "coef": reg.coef_.tolist(),
            "intercept": float(reg.intercept_),
            "alpha": float(reg.alpha),
        },
    }


@dataclass(frozen=True)
class LevelVariant:
    name: str


LEVEL_VARIANTS = [LevelVariant("ridge"), LevelVariant("rf"), LevelVariant("xgb")]


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


def _pred_monthly_totals(
    test: pd.DataFrame,
    yhat_daily: np.ndarray,
    *,
    date_col: str = "calendar_day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = test[["site_client_id", date_col, TARGET]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df["pred"] = np.asarray(yhat_daily, float)
    df = df.dropna(subset=[date_col, TARGET, "pred"])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp("M") - pd.offsets.MonthBegin(1)
    g = df.groupby(["site_client_id", "month"], as_index=False)[[TARGET, "pred"]].sum()
    return df, g


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = EX_ROOT / "outputs_mature_level_train2024_test2025" / run_id
    model_dir = run_root / "models" / "more_than"
    results_dir = run_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    df = master.merge(more[key_cols + extra_from_more], on=key_cols, how="left")
    if "dbscan_cluster_12km" not in df.columns:
        raise RuntimeError(">2y time split: dbscan_cluster_12km missing after merge")

    df = df.dropna(subset=["site_client_id", "latitude", "longitude", "calendar_day"])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    train = df[df["calendar_day"].dt.year == 2024].copy()
    test = df[df["calendar_day"].dt.year == 2025].copy()
    print(f"[gt time split] train_days={len(train):,} test_days={len(test):,} train_sites={train['site_client_id'].nunique():,} test_sites={test['site_client_id'].nunique():,}")

    site_train = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    site_clusters = b._fit_dbscan(site_train, "cluster_id_train")
    site_train = site_train.merge(site_clusters, on="site_client_id", how="left")
    site_train = site_train[site_train["cluster_id_train"].fillna(-1).astype(int) != -1].copy()
    train, test = b.align_train_test_clusters_to_train_refit(train, test, site_train)

    centroids = b._cluster_centroids(
        site_train.rename(columns={"cluster_id_train": "dbscan_cluster_12km"}),
        "dbscan_cluster_12km",
    )
    context_df = b._build_cluster_context(train, "dbscan_cluster_12km", b.CONTEXT_BASE_FEATURES)
    train = train.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test = test.merge(context_df, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])

    monthly_series = b._cluster_monthly_series(train, "dbscan_cluster_12km", "calendar_day", freq="MS")

    feature_candidates = (
        b.SITE_FEATURES_STATIC
        + b.ANNUAL_WEATHER_FEATURES
        + b.DAILY_WEATHER_FEATURES
        + b.TIME_LAG_FEATURES_DAILY
        + ["dbscan_cluster_12km"]
        + [c for c in train.columns if c.startswith("ctx_")]
    )
    feature_cols = [c for c in feature_candidates if c in train.columns and train[c].notna().any()]
    print(f"[gt time split] features={len(feature_cols)}")

    # Save shared cluster artifacts for this run.
    _save_json(model_dir / "cluster_centroids_12km.json", centroids)
    _save_json(model_dir / "cluster_monthly_series_12km.json", {"frequency": "MS", "series": {str(k): v for k, v in monthly_series.items()}})
    _save_json(model_dir / "feature_spec_12km.json", {
        "site_features_static": b.SITE_FEATURES_STATIC,
        "annual_weather_features": b.ANNUAL_WEATHER_FEATURES,
        "daily_weather_features": b.DAILY_WEATHER_FEATURES,
        "time_lag_features": b.TIME_LAG_FEATURES_DAILY,
        "cluster_context_features": [c for c in context_df.columns if c != "cluster_id"],
        "final_feature_order": feature_cols,
        "cluster_col": "dbscan_cluster_12km",
    })

    leaderboard_rows: list[dict[str, Any]] = []
    blobs: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"train_year": 2024, "test_year": 2025},
        "holdout_definition": "Temporal split; all 2025 days are test (no site holdout).",
        "train_sites": int(train["site_client_id"].nunique()),
        "test_sites": int(test["site_client_id"].nunique()),
        "n_train_days": int(len(train)),
        "n_test_days": int(len(test)),
        "variants": {},
    }

    for v in LEVEL_VARIANTS:
        if v.name == "ridge":
            pipe = _train_ridge(train, feature_cols)
            portable = _portable_from_ridge_pipe(pipe, feature_cols)
            _save_json(model_dir / "wash_count_model_12km.portable.json", portable)
        elif v.name == "rf":
            pipe = _train_rf(train, feature_cols)
            joblib.dump({"pipeline": pipe, "feature_order": feature_cols}, model_dir / "wash_count_model_12km.rf.joblib")
        elif v.name == "xgb":
            pipe = _train_xgb(train, feature_cols)
            joblib.dump({"pipeline": pipe, "feature_order": feature_cols}, model_dir / "wash_count_model_12km.xgb.joblib")
        else:
            raise ValueError(v.name)

        X_te = test[feature_cols].to_numpy(dtype=float)
        yhat = pipe.predict(X_te)
        _, monthly = _pred_monthly_totals(test, yhat, date_col="calendar_day")
        m = _metric_block(monthly[TARGET].to_numpy(), monthly["pred"].to_numpy())
        blobs["variants"][v.name] = {"monthly_site_totals": m}
        leaderboard_rows.append({"variant": v.name, **m})

    _save_json(results_dir / "time_split_gt_2024_train_2025_test.json", blobs)

    df_lb = pd.DataFrame(leaderboard_rows).sort_values("wape")
    md: list[str] = []
    md.append("# >2y time-split leaderboard (train 2024, test 2025)")
    md.append("")
    md.append(f"- Run: `{run_id}`")
    md.append(f"- Train days: `{len(train):,}`  Test days: `{len(test):,}`")
    md.append("")
    md.append("| rank | variant | wape | mae | rmse | r2 | pct_within_20pct |")
    md.append("|---:|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(df_lb.to_dict(orient="records"), start=1):
        md.append(
            "| {rank} | `{variant}` | {wape} | {mae} | {rmse} | {r2} | {p20} |".format(
                rank=i,
                variant=r.get("variant"),
                wape=r.get("wape"),
                mae=r.get("mae"),
                rmse=r.get("rmse"),
                r2=r.get("r2"),
                p20=r.get("pct_within_20pct"),
            )
        )
    (results_dir / "leaderboard.md").write_text("\n".join(md) + "\n")
    (results_dir / "leaderboard.csv").write_text(df_lb.to_csv(index=False))
    print(f"wrote {run_root}")


if __name__ == "__main__":
    main()
