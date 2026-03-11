"""
Production-facing v3 quantile predictor.

The full implementation (QuantilePredictorV3) lives in
scoringmetric/approach2/v3/quantile_predictor_v3.py — it references training data
files relative to that location, so it stays there as the source of truth.

This module re-exports it under a stable production import path:

    from app.modelling.ds.quantile_predictor import QuantilePredictorV3

Accuracy (5-fold CV, 482 sites, 4-class):
  v3 : exact ~37–62%*  |  within-1-quartile ~74–97%*
  * Upper bound achieved when tunnel_count is provided (site capacity signal, r=0.89).
    For brand-new sites without known tunnel count, exact accuracy ~37%.

Key improvements over v2:
  - Site age (strongest predictor, Spearman r=-0.335)
  - Tunnel count proxy (r=+0.891 when available)
  - KNN imputation instead of global median
  - Calibrated RandomForest (isotonic regression) for reliable probabilities
  - Signal-validated feature directions (Spearman r on 481 common rows)
  - 4 engineered features: competition_quality, gas_station_draw,
    retail_proximity, weather_drive_score
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scoringmetric.approach2.v3.quantile_predictor_v3 import (  # noqa: F401, E402
    QuantilePredictorV3,
    FEATURE_DIRECTIONS,
    FEATURE_SIGNAL,
    FEATURE_LABELS,
    QUANTILE_LABELS,
    QUANTILE_TIER_NAMES,
    SIGNAL_THRESHOLD,
)

__all__ = [
    "QuantilePredictorV3",
    "FEATURE_DIRECTIONS",
    "FEATURE_SIGNAL",
    "FEATURE_LABELS",
    "QUANTILE_LABELS",
    "QUANTILE_TIER_NAMES",
    "SIGNAL_THRESHOLD",
]
