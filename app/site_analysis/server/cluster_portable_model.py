"""
Portable wash-count model (no sklearn pickle): Ridge pipeline coefficients only.
Loads JSON written by clustering/cluster_model_eval.py — works across sklearn versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def predict_wash_count(portable: Dict[str, Any], feature_values: Dict[str, Any]) -> float:
    feats: List[str] = portable["features"]
    x = np.array([feature_values.get(f, np.nan) for f in feats], dtype=np.float64)
    stat = np.asarray(portable["imputer_statistics"], dtype=np.float64)
    if x.shape[0] != stat.shape[0]:
        raise ValueError("Feature count mismatch vs portable model.")
    x = np.where(np.isnan(x), stat, x)
    mean = np.asarray(portable["scaler_mean"], dtype=np.float64)
    scale = np.asarray(portable["scaler_scale"], dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    xs = (x - mean) / scale
    coef = np.asarray(portable["ridge_coef"], dtype=np.float64)
    intercept = float(portable["ridge_intercept"])
    y = float(np.dot(coef, xs) + intercept)
    return max(0.0, y)


def load_portable(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def range_stats_dataframe(portable: Dict[str, Any]) -> pd.DataFrame:
    col = portable["cluster_col"]
    rows = portable["train_range_stats"]
    df = pd.DataFrame(rows)
    if col not in df.columns:
        raise ValueError(f"train_range_stats missing column {col}")
    return df
