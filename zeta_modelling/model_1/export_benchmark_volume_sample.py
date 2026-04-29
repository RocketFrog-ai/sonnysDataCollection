"""
Build `data_1/model1_benchmark_volume_sample.csv`: aligned test-set actual vs predicted
monthly volumes across Model 1 benchmarks (for Streamlit volume plots).

By default this script caches trained models under `data_1/model_saves_1/`
and reuses them on subsequent runs. Use `--retrain` to force retraining.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from zeta_modelling.model_1.benchmarking import default_volume_sample_path
from zeta_modelling.model_1.phase2_deployable_multiplier import (
    add_base_features as dep_add_base,
    add_cluster_features as dep_add_cluster,
    fit_site_type_inference_model,
    infer_site_type,
)
from zeta_modelling.model_1.phase2_feature_engineering_and_model import build_features as p2_build_features
from zeta_modelling.model_1.phase2_no_lag_upgrade import (
    build_feature_table as p2u_build,
    split_train_test as p2u_split,
    train_eval as p2u_train_eval,
    tune_lightgbm as p2u_tune,
)
from zeta_modelling.model_1.phase2_site_profile_multiplier import (
    _unique_preserve_order,
    add_cluster_features as sp_add_cluster,
    add_lags_for_warm_model as sp_add_lags,
    add_lifecycle_curve_features as sp_add_life,
    add_site_behavior_profiles as sp_add_site,
    base_features as sp_base,
    train_test as sp_train_test,
)
from zeta_modelling.model_1.phase3_advanced_forecast import backtest_predictions_merge, load_artifacts


def _fit_abs_bundle(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    X_train = train_df[feature_cols].copy()
    y_train = train_df["monthly_volume"].astype(float)
    used_cat_cols = [c for c in categorical_cols if c in feature_cols]
    for c in used_cat_cols:
        X_train[c] = X_train[c].astype("category")
    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    model = LGBMRegressor(random_state=42, **params)
    model.fit(X_train, y_train, categorical_feature=used_cat_cols)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "categorical_cols": used_cat_cols,
        "fill_values": fill_values.to_dict(),
    }


def _predict_abs_bundle(bundle: dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    X = test_df[bundle["feature_cols"]].copy()
    for c in bundle["categorical_cols"]:
        X[c] = X[c].astype("category")
    fill_values = pd.Series(bundle["fill_values"], dtype=float)
    X = X.fillna(fill_values)
    return bundle["model"].predict(X)


def _fit_multiplier_bundle(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict[str, Any],
) -> dict[str, Any]:
    eps = 1e-6
    tr = train_df.copy()
    tr["target_multiplier"] = tr["monthly_volume"] / (tr["cluster_month_avg"] + eps)
    X_train = tr[feature_cols].copy()
    y_train = tr["target_multiplier"].astype(float)
    used_cat_cols = [c for c in categorical_cols if c in feature_cols]
    for c in used_cat_cols:
        X_train[c] = X_train[c].astype("category")
    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    model = LGBMRegressor(random_state=42, **params)
    model.fit(X_train, y_train, categorical_feature=used_cat_cols)
    return {
        "model": model,
        "feature_cols": feature_cols,
        "categorical_cols": used_cat_cols,
        "fill_values": fill_values.to_dict(),
    }


def _predict_multiplier_bundle(bundle: dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    eps = 1e-6
    X = test_df[bundle["feature_cols"]].copy()
    for c in bundle["categorical_cols"]:
        X[c] = X[c].astype("category")
    fill_values = pd.Series(bundle["fill_values"], dtype=float)
    X = X.fillna(fill_values)
    pred_mult = bundle["model"].predict(X)
    return pred_mult * (test_df["cluster_month_avg"].to_numpy(dtype=float) + eps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export aligned benchmark volume predictions for Model 1.")
    p.add_argument(
        "--input",
        type=Path,
        default=_REPO / "zeta_modelling" / "data_1" / "phase1_final_monthly_2024_2025.csv",
    )
    p.add_argument(
        "--artifacts",
        type=Path,
        default=_REPO / "zeta_modelling" / "model_1" / "phase3_artifacts.joblib",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to data_1/model1_benchmark_volume_sample.csv",
    )
    p.add_argument(
        "--model-save-dir",
        type=Path,
        default=_REPO / "zeta_modelling" / "data_1" / "model_saves_1",
    )
    p.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even when saved model bundles exist.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.output or default_volume_sample_path()
    model_save_dir = Path(args.model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values(["site_id", "date"]).reset_index(drop=True)

    keys = ["site_id", "date"]
    test_mask = raw["date"] >= pd.Timestamp("2025-01-01")
    out = raw.loc[test_mask, keys + ["monthly_volume"]].copy()
    out = out.rename(columns={"monthly_volume": "y_actual"})

    # --- Phase 2: feature engineering file (no lag / with lag) ---
    feat_p2 = p2_build_features(raw)
    train_p2 = feat_p2[feat_p2["date"] < pd.Timestamp("2025-01-01")].copy()
    test_p2 = feat_p2[feat_p2["date"] >= pd.Timestamp("2025-01-01")].copy()
    base_features = [
        "age_in_months",
        "real_age_months",
        "age_sq",
        "log_age",
        "is_early",
        "is_growth",
        "is_mature",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "month",
        "month_sin",
        "month_cos",
        "seasonality_factor",
        "latitude",
        "longitude",
        "lat_lon_interaction",
        "maturity_bucket",
    ]
    lag_features = ["lag_1", "lag_3", "rolling_mean_3"]
    cat_p2 = ["cluster_id", "maturity_bucket"]
    p2_no_lag_path = model_save_dir / "p2_no_lag.joblib"
    p2_with_lag_path = model_save_dir / "p2_with_lag.joblib"
    if p2_no_lag_path.exists() and not args.retrain:
        no_lag_bundle = joblib.load(p2_no_lag_path)
    else:
        no_lag_bundle = _fit_abs_bundle(
            train_p2,
            base_features,
            cat_p2,
            {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6},
        )
        joblib.dump(no_lag_bundle, p2_no_lag_path)
    if p2_with_lag_path.exists() and not args.retrain:
        with_lag_bundle = joblib.load(p2_with_lag_path)
    else:
        with_lag_bundle = _fit_abs_bundle(
            train_p2,
            base_features + lag_features,
            cat_p2,
            {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6},
        )
        joblib.dump(with_lag_bundle, p2_with_lag_path)
    pred_nl = _predict_abs_bundle(no_lag_bundle, test_p2)
    pred_lag = _predict_abs_bundle(with_lag_bundle, test_p2)
    m1 = test_p2[keys].copy()
    m1["pred_lightgbm_no_lag"] = pred_nl
    m1["pred_lightgbm_with_lag"] = pred_lag
    out = out.merge(m1, on=keys, how="left")

    # --- Phase 2: no-lag upgrade path ---
    feat_u = p2u_build(raw)
    tr_u, te_u = p2u_split(feat_u)
    cat_u = ["cluster_id", "maturity_bucket"]
    baseline_features = [
        "age_in_months",
        "real_age_months",
        "age_sq",
        "log_age",
        "is_early",
        "is_growth",
        "is_mature",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "month",
        "month_sin",
        "month_cos",
        "seasonality_factor",
        "latitude",
        "longitude",
        "lat_lon_interaction",
        "maturity_bucket",
    ]
    upgraded_features = baseline_features + [
        "cluster_lag_1",
        "cluster_lag_3",
        "cluster_growth_rate",
        "cluster_std",
        "cluster_rolling_mean",
        "age_cluster_interaction",
        "age_seasonality_interaction",
    ]
    warm_features = upgraded_features + ["lag_1", "lag_3", "rolling_mean_3"]
    p2u_path = model_save_dir / "p2_no_lag_upgrade.joblib"
    if p2u_path.exists() and not args.retrain:
        p2u_payload = joblib.load(p2u_path)
        pred_bl = p2u_payload["pred_bl"]
        pred_up = p2u_payload["pred_up"]
        pred_wm = p2u_payload["pred_wm"]
    else:
        baseline_params = {"num_leaves": 63, "max_depth": 6, "min_child_samples": 30, "learning_rate": 0.05}
        baseline_model, _, pred_bl = p2u_train_eval(tr_u, te_u, baseline_features, cat_u, baseline_params)
        best_params = p2u_tune(tr_u, upgraded_features, cat_u)
        upgraded_model, _, pred_up = p2u_train_eval(tr_u, te_u, upgraded_features, cat_u, best_params)
        warm_model, _, pred_wm = p2u_train_eval(tr_u, te_u, warm_features, cat_u, best_params)
        p2u_payload = {
            "baseline_model": baseline_model,
            "upgraded_model": upgraded_model,
            "warm_model": warm_model,
            "best_params": best_params,
            "pred_bl": pred_bl,
            "pred_up": pred_up,
            "pred_wm": pred_wm,
        }
        joblib.dump(p2u_payload, p2u_path)
    for label, pdf in (
        ("pred_p2_baseline_no_lag", pred_bl),
        ("pred_p2_upgraded_no_lag", pred_up),
        ("pred_p2_warm_lags", pred_wm),
    ):
        m = pdf[keys + ["pred"]].rename(columns={"pred": label})
        out = out.merge(m, on=keys, how="left")

    # --- Phase 2: deployable multiplier ---
    dep = dep_add_cluster(dep_add_base(raw))
    tr_d = dep[dep["date"] < pd.Timestamp("2025-01-01")].copy()
    te_d = dep[dep["date"] >= pd.Timestamp("2025-01-01")].copy()
    dep_path = model_save_dir / "p2_deployable_multiplier.joblib"
    dep_feats = [
        "age_in_months",
        "real_age_months",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "month",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
        "maturity_bucket",
        "site_type",
        "age_sq",
        "log_age",
        "is_early",
        "is_growth",
        "is_mature",
        "age_saturation",
        "growth_velocity",
        "distance_from_peak",
        "lat_lon_interaction",
    ]
    dep_feats = [c for c in dep_feats if c in tr_d.columns]
    if dep_path.exists() and not args.retrain:
        dep_payload = joblib.load(dep_path)
        clf = dep_payload["site_type_clf"]
        cluster_mode = dep_payload["cluster_mode"]
        dep_bundle = dep_payload["reg_bundle"]
    else:
        clf, cluster_mode = fit_site_type_inference_model(tr_d)
        tr_typed = tr_d.copy()
        tr_typed["site_type"] = infer_site_type(tr_typed, clf, cluster_mode)
        dep_bundle = _fit_multiplier_bundle(
            tr_typed,
            dep_feats,
            [c for c in ["cluster_id", "maturity_bucket", "site_type"] if c in tr_typed.columns],
            {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": -1,
                "min_child_samples": 30,
            },
        )
        joblib.dump(
            {"site_type_clf": clf, "cluster_mode": cluster_mode, "reg_bundle": dep_bundle},
            dep_path,
        )
    te_d = te_d.copy()
    te_d["site_type"] = infer_site_type(te_d, clf, cluster_mode)
    pred_dep = _predict_multiplier_bundle(dep_bundle, te_d)
    mdep = te_d[keys].copy()
    mdep["pred_deployable_multiplier"] = pred_dep
    out = out.merge(mdep, on=keys, how="left")

    # --- Phase 2: site profile multiplier ---
    sp = sp_base(raw)
    sp = sp_add_cluster(sp)
    sp = sp_add_site(sp)
    sp = sp_add_life(sp)
    sp = sp_add_lags(sp)
    tr_s, te_s = sp_train_test(sp)
    sp_path = model_save_dir / "p2_site_profile_multiplier.joblib"
    baseline_features_sp = [
        "age_in_months",
        "real_age_months",
        "age_sq",
        "log_age",
        "month",
        "month_sin",
        "month_cos",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "latitude",
        "longitude",
        "maturity_bucket",
    ]
    baseline_features_sp = [c for c in baseline_features_sp if c in sp.columns]
    baseline_features_sp = _unique_preserve_order(baseline_features_sp)
    if "seasonality_factor" in sp.columns and "seasonality_factor" not in baseline_features_sp:
        baseline_features_sp.insert(max(0, len(baseline_features_sp) - 3), "seasonality_factor")
    no_lag_sp = baseline_features_sp + [
        "site_avg_volume",
        "site_peak",
        "site_growth",
        "site_volatility",
        "site_type",
        "age_saturation",
        "growth_velocity",
        "distance_from_peak",
        "lat_lon_interaction",
    ]
    no_lag_sp = [c for c in no_lag_sp if c in sp.columns]
    no_lag_sp = _unique_preserve_order(no_lag_sp)
    cat_sp = [c for c in ["cluster_id", "maturity_bucket", "site_type"] if c in tr_s.columns]
    if sp_path.exists() and not args.retrain:
        sp_bundle = joblib.load(sp_path)
    else:
        sp_bundle = _fit_multiplier_bundle(
            tr_s,
            no_lag_sp,
            cat_sp,
            {
                "n_estimators": 1400,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "max_depth": -1,
                "min_child_samples": 30,
            },
        )
        joblib.dump(sp_bundle, sp_path)
    pred_sp = _predict_multiplier_bundle(sp_bundle, te_s)
    msp = te_s[keys].copy()
    msp["pred_site_profile_multiplier"] = pred_sp
    out = out.merge(msp, on=keys, how="left")

    # --- Phase 3: quantile forecaster P50 ---
    artifacts = load_artifacts(args.artifacts)
    p3 = backtest_predictions_merge(raw, artifacts, max_rows=None)
    mp3 = p3[keys + ["pred_p50"]].rename(columns={"pred_p50": "pred_phase3_p50"})
    out = out.merge(mp3, on=keys, how="left")

    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    for c in pred_cols:
        mae_c = mean_absolute_error(out["y_actual"], out[c])
        rmse_c = float(np.sqrt(mean_squared_error(out["y_actual"], out[c])))
        print(f"{c}: MAE={mae_c:.2f} RMSE={rmse_c:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    print(f"Saved/loaded model bundles in {model_save_dir}")


if __name__ == "__main__":
    main()
