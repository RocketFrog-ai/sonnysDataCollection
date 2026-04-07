"""Histogram shape labels and feature quantile bin helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


def _detect_shape(data: np.ndarray) -> str:
    if len(np.unique(data)) <= 6:
        return "discrete"
    skew_val = float(stats.skew(data))
    if skew_val > 0.5:
        return "right-skewed"
    elif skew_val < -0.5:
        return "left-skewed"
    return "symmetric"

def _quantile_boundaries(data: np.ndarray, n: int = 4) -> np.ndarray:
    pcts = np.linspace(0, 100, n + 1)
    return np.percentile(data, pcts)

def _assign_raw_quantile(value: float, boundaries: np.ndarray) -> int:
    for q in range(1, len(boundaries)):
        if value <= boundaries[q]:
            return q
    return len(boundaries) - 1

def _adj_quantile(raw_q: int, direction: str) -> int:
    if direction == "lower":
        return 5 - raw_q
    elif direction == "neutral":
        return raw_q
    return raw_q

def _next_better_boundary(boundaries: np.ndarray, adj_q: int, direction: str) -> Optional[float]:
    if adj_q >= 4:
        return None
    if direction == "higher":
        return float(boundaries[adj_q])
    elif direction == "lower":
        return float(boundaries[4 - adj_q])
    return None
