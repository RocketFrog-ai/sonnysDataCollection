"""LightGBM training.

Modelling strategy
------------------
* Target = log1p(wash_count_total).
* Two models are trained:
    - "young" model on rows with site_age_months < MATURITY_MONTHS
    - "mature" model on rows with site_age_months >= MATURITY_MONTHS
  Splitting by maturity lets each booster specialise: the young model
  spends capacity learning ramp dynamics; the mature model focuses on
  market drift and seasonality.
* Per-site leave-out cross-validation: a site is either fully in train
  or fully in validation, so a "cold" site is genuinely cold at scoring
  time.
* We compare against a peer-anchor baseline (anchor_wash) so the model's
  lift can be reported honestly.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from . import config as C
from .features import FEATURE_COLS


def _fit_one(train_df: pd.DataFrame, val_df: pd.DataFrame, label: str) -> tuple[lgb.Booster, dict]:
    # Target = multiplicative residual on top of the peer anchor (in log space).
    # The booster only has to learn how a candidate site deviates from its
    # peer-typical trajectory, which is a much lower-variance target than the
    # raw log washes.
    dtrain = lgb.Dataset(train_df[FEATURE_COLS], label=train_df["y_residual"])
    dval = lgb.Dataset(val_df[FEATURE_COLS], label=val_df["y_residual"], reference=dtrain)
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=80,
        feature_fraction=0.80,
        bagging_fraction=0.80,
        bagging_freq=5,
        lambda_l2=2.0,
        verbosity=-1,
        seed=C.SEED,
    )
    booster = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    # Validate in original-scale washes so the metric is interpretable.
    pred_res = booster.predict(val_df[FEATURE_COLS], num_iteration=booster.best_iteration)
    pred_log = pred_res + val_df["y_anchor"].values
    pred_wash = np.expm1(pred_log)
    actual_wash = val_df["wash_count_total"].values
    val_rmse_wash = float(np.sqrt(np.mean((pred_wash - actual_wash) ** 2)))
    val_mae_wash = float(np.mean(np.abs(pred_wash - actual_wash)))
    return booster, {"label": label, "val_rmse_wash": val_rmse_wash,
                      "val_mae_wash": val_mae_wash,
                      "best_iter": booster.best_iteration}


def train_cohort_models(train_frame: pd.DataFrame, n_folds: int = 5) -> dict:
    """Train young- and mature-segment models with group K-fold CV.

    Returns dict with final boosters (trained on full data using mean of
    best_iter across folds) and per-fold metrics.
    """
    out = {}
    for seg_name, seg_mask in (
        ("young", train_frame["site_age_months"] < C.MATURITY_MONTHS),
        ("mature", train_frame["site_age_months"] >= C.MATURITY_MONTHS),
    ):
        seg = train_frame[seg_mask].dropna(subset=FEATURE_COLS + ["y"]).copy()
        if len(seg) < 200:
            print(f"[train] {seg_name}: only {len(seg)} rows — skipping")
            continue
        gkf = GroupKFold(n_splits=min(n_folds, seg["client_id_location_id"].nunique()))
        groups = seg["client_id_location_id"].values
        fold_metrics = []
        best_iters = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(seg, groups=groups)):
            tr = seg.iloc[tr_idx]; va = seg.iloc[va_idx]
            _, m = _fit_one(tr, va, f"{seg_name}_fold{fold}")
            fold_metrics.append(m)
            best_iters.append(m["best_iter"] or 500)
        # Final fit on all rows with averaged rounds (+10% safety)
        final_rounds = max(100, int(np.mean(best_iters) * 1.1))
        dtrain = lgb.Dataset(seg[FEATURE_COLS], label=seg["y_residual"])
        params = dict(
            objective="regression", metric="rmse",
            learning_rate=0.03, num_leaves=31, min_data_in_leaf=80,
            feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=5,
            lambda_l2=2.0, verbosity=-1, seed=C.SEED,
        )
        final_booster = lgb.train(params, dtrain, num_boost_round=final_rounds)
        # Calibration multiplier: log-target boosters predict the conditional
        # geometric mean. For unbiased point forecasts of washes we rescale
        # so that sum(predicted) == sum(actual) on the training fold.
        train_pred_res = final_booster.predict(seg[FEATURE_COLS])
        train_pred_wash = np.expm1(seg["y_anchor"].values + train_pred_res)
        cal = float(seg["wash_count_total"].sum() / max(train_pred_wash.sum(), 1.0))
        out[seg_name] = {
            "booster": final_booster,
            "fold_metrics": fold_metrics,
            "final_rounds": final_rounds,
            "calibration_multiplier": cal,
            "feature_importance": dict(zip(FEATURE_COLS,
                                           final_booster.feature_importance(importance_type="gain").tolist())),
            "n_rows": int(len(seg)),
        }
        avg_rmse = np.mean([m["val_rmse_wash"] for m in fold_metrics])
        avg_mae = np.mean([m["val_mae_wash"] for m in fold_metrics])
        avg_iter = np.mean(best_iters)
        print(f"[train] {seg_name}: rows={len(seg)} folds={len(fold_metrics)} "
              f"mean_val_rmse={avg_rmse:,.0f}  mean_val_mae={avg_mae:,.0f}  "
              f"avg_best_iter={avg_iter:.0f}  final_rounds={final_rounds}")
    return out


def predict_blend(models: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict for a frame using the appropriate cohort model per row.

    The booster predicts a *log-space residual* on top of the peer anchor.
    Reconstruct washes as ``expm1(log_anchor + residual)``.
    """
    pred_res = np.zeros(len(frame))
    young_mask = (frame["site_age_months"] < C.MATURITY_MONTHS).values
    mature_mask = ~young_mask
    if young_mask.any() and "young" in models:
        b = models["young"]["booster"]
        pred_res[young_mask] = b.predict(frame.loc[young_mask, FEATURE_COLS])
    if mature_mask.any() and "mature" in models:
        b = models["mature"]["booster"]
        pred_res[mature_mask] = b.predict(frame.loc[mature_mask, FEATURE_COLS])

    log_anchor = frame["log_anchor"].values
    pred_wash = np.expm1(log_anchor + pred_res)
    # Apply per-cohort calibration so predictions are unbiased in mean.
    cal_young = models.get("young", {}).get("calibration_multiplier", 1.0) or 1.0
    cal_mature = models.get("mature", {}).get("calibration_multiplier", 1.0) or 1.0
    pred_wash = np.where(young_mask, pred_wash * cal_young, pred_wash * cal_mature)
    bad = ~np.isfinite(pred_wash)
    if bad.any():
        pred_wash[bad] = np.expm1(pred_res[bad])
    return np.clip(pred_wash, 0, None)
