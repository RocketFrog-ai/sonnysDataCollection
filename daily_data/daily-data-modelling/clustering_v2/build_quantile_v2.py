"""Quantile GBMs (q10/q50/q90) mirroring build_v2 cohorts. See APPROACH.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_v2 as bv


REPO_ROOT = bv.REPO_ROOT
V2_DIR = bv.V2_DIR
TARGET = bv.TARGET

OUT_MODELS_DIR = V2_DIR / "models_quantile"
OUT_RESULTS_DIR = V2_DIR / "results"

QUANTILES = {
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
}


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.abs(y_true).sum())
    if denom <= 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def _feature_filter(train_df: pd.DataFrame, feature_candidates: list[str]) -> list[str]:
    out: list[str] = []
    for c in feature_candidates:
        if c in train_df.columns and train_df[c].notna().any():
            out.append(c)
    return out


def _prepare_more_than() -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
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

    extra = [
        c
        for c in (
            ["dbscan_cluster_12km"]
            + bv.TIME_LAG_FEATURES_DAILY
            + bv.ANNUAL_WEATHER_FEATURES
            + ["region_enc", "state_enc", "costco_enc", "carwash_type_encoded"]
        )
        if c in more.columns and c not in master.columns
    ]

    df = master.merge(
        more[["site_client_id", "calendar_day"] + extra],
        on=["site_client_id", "calendar_day"],
        how="left",
    )

    split_desc = "train<2025-07-01, test>=2025-07-01"
    train = df[df["calendar_day"] < pd.Timestamp("2025-07-01")].copy().dropna(subset=[TARGET])
    test = df[df["calendar_day"] >= pd.Timestamp("2025-07-01")].copy().dropna(subset=[TARGET])

    st = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    st = st.merge(bv._fit_dbscan(st, "cluster_id_train"), on="site_client_id", how="left")
    train, test = bv.align_train_test_clusters_to_train_refit(train, test, st)

    ctx = bv._build_cluster_context(train, "dbscan_cluster_12km", bv.CONTEXT_BASE_FEATURES)
    train_ctx = train.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test_ctx = test.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    ctx_cols = [c for c in train_ctx.columns if c.startswith("ctx_")]

    feature_candidates = (
        bv.SITE_FEATURES_STATIC
        + bv.ANNUAL_WEATHER_FEATURES
        + bv.TIME_LAG_FEATURES_DAILY
        + ["dbscan_cluster_12km"]
        + bv.DAILY_WEATHER_FEATURES
        + ctx_cols
    )
    feature_cols = _feature_filter(train_ctx, feature_candidates)
    return train_ctx, test_ctx, feature_cols, split_desc


def _prepare_less_than() -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    df = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv",
        low_memory=False,
    )
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")

    split_desc = (
        "train period_index<=12 (year1); test period_index>=25 (year2); "
        "period_index skips 13-24 by construction — see APPROACH.md"
    )
    train = df[df["period_index"] <= 12].copy().dropna(subset=[TARGET])
    test = df[df["period_index"] >= 25].copy().dropna(subset=[TARGET])

    st = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    st = st.merge(bv._fit_dbscan(st, "cluster_id_train"), on="site_client_id", how="left")
    train, test = bv.align_train_test_clusters_to_train_refit(train, test, st)

    ctx = bv._build_cluster_context(
        train,
        "dbscan_cluster_12km",
        [],
        include_target_aggs=True,
    )
    train_ctx = train.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test_ctx = test.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    ctx_cols = [c for c in train_ctx.columns if c.startswith("ctx_")]

    feature_candidates = (
        bv.SITE_FEATURES_MONTHLY
        + bv.TIME_LAG_FEATURES_MONTHLY
        + ["dbscan_cluster_12km"]
        + ctx_cols
    )
    feature_cols = _feature_filter(train_ctx, feature_candidates)
    bv.assert_less_than_no_weather_columns(feature_cols)
    return train_ctx, test_ctx, feature_cols, split_desc


def _train_quantile_pipelines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
) -> dict[str, np.ndarray]:
    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    y_tr = train_df[TARGET].to_numpy(dtype=float)
    X_te = test_df[feature_cols].to_numpy(dtype=float)

    preds: dict[str, np.ndarray] = {}
    for name, q in QUANTILES.items():
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "m",
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=q,
                        max_iter=350,
                        learning_rate=0.05,
                        max_depth=6,
                        min_samples_leaf=40,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X_tr, y_tr)
        preds[name] = pipe.predict(X_te)

        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out_dir / f"{name}.joblib")

    return preds


def _ridge_baseline_pred(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    X_tr = train_df[feature_cols].to_numpy(dtype=float)
    y_tr = train_df[TARGET].to_numpy(dtype=float)
    X_te = test_df[feature_cols].to_numpy(dtype=float)

    pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=1.0, random_state=0)),
        ]
    )
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


def _evaluate(
    y_true: np.ndarray,
    q_preds: dict[str, np.ndarray],
    ridge_pred: np.ndarray,
) -> dict[str, Any]:
    q10 = q_preds["q10"]
    q50 = q_preds["q50"]
    q90 = q_preds["q90"]

    metrics_q50 = {
        "mae": float(mean_absolute_error(y_true, q50)),
        "rmse": _rmse(y_true, q50),
        "r2": float(r2_score(y_true, q50)),
        "wape": _wape(y_true, q50),
    }
    metrics_ridge = {
        "mae": float(mean_absolute_error(y_true, ridge_pred)),
        "rmse": _rmse(y_true, ridge_pred),
        "r2": float(r2_score(y_true, ridge_pred)),
        "wape": _wape(y_true, ridge_pred),
    }

    pinball = {
        "q10": _pinball_loss(y_true, q10, 0.10),
        "q50": _pinball_loss(y_true, q50, 0.50),
        "q90": _pinball_loss(y_true, q90, 0.90),
    }

    coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    interval_width = float(np.mean(q90 - q10))

    return {
        "q50_metrics": metrics_q50,
        "ridge_baseline_metrics": metrics_ridge,
        "pinball_loss": pinball,
        "interval_coverage_q10_q90": coverage,
        "interval_mean_width_q90_minus_q10": interval_width,
    }


def _run_cohort(
    cohort_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    split_desc: str,
    cohort_dir_name: str,
) -> dict[str, Any]:
    out_dir = OUT_MODELS_DIR / cohort_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_json(out_dir / "feature_order.json", feature_cols)

    q_preds = _train_quantile_pipelines(train_df, test_df, feature_cols, out_dir)
    ridge_pred = _ridge_baseline_pred(train_df, test_df, feature_cols)
    y_true = test_df[TARGET].to_numpy(dtype=float)

    eval_obj = _evaluate(y_true, q_preds, ridge_pred)
    eval_obj.update(
        {
            "cohort": cohort_key,
            "split": split_desc,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(len(feature_cols)),
            "feature_order_path": str((out_dir / "feature_order.json").relative_to(V2_DIR)),
            "model_paths": {
                "q10": str((out_dir / "q10.joblib").relative_to(V2_DIR)),
                "q50": str((out_dir / "q50.joblib").relative_to(V2_DIR)),
                "q90": str((out_dir / "q90.joblib").relative_to(V2_DIR)),
            },
        }
    )

    _save_json(out_dir / "quantile_metrics.json", eval_obj)
    return eval_obj


def main() -> None:
    train_mt, test_mt, feats_mt, split_mt = _prepare_more_than()
    train_lt, test_lt, feats_lt, split_lt = _prepare_less_than()

    more_summary = _run_cohort(
        cohort_key="more_than_2yrs",
        train_df=train_mt,
        test_df=test_mt,
        feature_cols=feats_mt,
        split_desc=split_mt,
        cohort_dir_name="more_than",
    )
    less_summary = _run_cohort(
        cohort_key="less_than_2yrs",
        train_df=train_lt,
        test_df=test_lt,
        feature_cols=feats_lt,
        split_desc=split_lt,
        cohort_dir_name="less_than",
    )

    overall = {
        "split_more_than_2yrs": split_mt,
        "split_less_than_2yrs": split_lt,
        "quantiles": {"q10": 0.10, "q50": 0.50, "q90": 0.90},
        "model_pipeline": {
            "imputer": "SimpleImputer(strategy='median')",
            "regressor": (
                "HistGradientBoostingRegressor("
                "loss='quantile', quantile=q, max_iter=350, learning_rate=0.05, "
                "max_depth=6, min_samples_leaf=40, random_state=42)"
            ),
        },
        "more_than_2yrs": more_summary,
        "less_than_2yrs": less_summary,
    }

    _save_json(OUT_RESULTS_DIR / "quantile_backtest_summary.json", overall)
    print(json.dumps(overall, indent=2, default=str))


if __name__ == "__main__":
    main()
