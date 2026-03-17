"""
Production-facing quantile predictor (v4).

The full implementation lives in
scoringmetric/approach2/v3/quantile_predictor_v4.py — it references training data
files relative to that location, so it stays there as the source of truth.

This module re-exports it under a stable production import path:

    from app.modelling.ds.quantile_predictor import QuantilePredictorV3

QuantilePredictorV3 is a backward-compat alias for QuantilePredictorV4.

Accuracy (5-fold CV, 482 sites, 4-class):
  v3 : exact 62.9%  |  within-1-quartile 97.9%  (CalibratedRandomForest)
  v4 : exact 63.5%  |  within-1-quartile 97.9%  (ExtraTrees, Optuna-tuned)
  * With tunnel_count + carwash_type (effective_capacity engineered feature).
    For brand-new sites without known tunnel count, exact accuracy ~37%.

Key improvements in v4 over v3:
  - ExtraTrees replaces CalibratedRandomForest (+0.6% exact CV accuracy)
    Random split thresholds in ExtraTrees reduce overfitting on 482-site dataset.
  - Optuna-tuned hyperparameters: n_estimators=600, max_depth=8, min_samples_leaf=5
    (v3 used max_depth=12 which overfit; depth=8 generalises better)
  - carwash_type_encoded now shown in feature_analysis output (display-only):
    it is intentionally excluded from the ML model because effective_capacity
    = tunnel_count × is_express already encodes it. Adding it directly causes
    multicollinearity and reduces accuracy by 0.9%.
  - Clearer report section explaining WHY carwash type is captured via
    effective_capacity (answers the "Express Tunnel → higher washes" question)
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scoringmetric.approach2.v3.quantile_predictor_v4 import (  # noqa: F401, E402
    QuantilePredictorV4,
    QuantilePredictorV3,  # backward-compat alias
    FEATURE_DIRECTIONS,
    FEATURE_SIGNAL,
    FEATURE_LABELS,
    QUANTILE_LABELS,
    QUANTILE_TIER_NAMES,
    SIGNAL_THRESHOLD,
    DISPLAY_ONLY_FEATURES,
)

__all__ = [
    "QuantilePredictorV4",
    "QuantilePredictorV3",
    "FEATURE_DIRECTIONS",
    "FEATURE_SIGNAL",
    "FEATURE_LABELS",
    "QUANTILE_LABELS",
    "QUANTILE_TIER_NAMES",
    "SIGNAL_THRESHOLD",
    "DISPLAY_ONLY_FEATURES",
]
