"""
Plain-English overall site verdict for /overall/{task_id}.

Uses weighted category scores and quantile outputs — not raw feature names.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# How we refer to each scoring bucket in user-facing text (not internal keys).
_CATEGORY_PLAIN: Dict[str, str] = {
    "Weather": "local weather pattern",
    "Competition": "competition",
    "Retail": "shopping proximity and food activity",
    "Gas": "gas-station pull",
}


def _format_int(n: Optional[float]) -> Optional[str]:
    if n is None:
        return None
    try:
        return f"{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return None


def _clean_volume_label(label: Optional[str]) -> Optional[str]:
    """Remove unit tails like 'cars/yr' since sentence already states unit."""
    if not label:
        return None
    cleaned = re.sub(r"\b(cars?/yr|cars?\s*per\s*year)\b", "", str(label), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -")
    return cleaned or None


def _category_pairs(category_scores: Optional[Dict[str, Any]]) -> List[Tuple[str, float]]:
    if not category_scores:
        return []
    out: List[Tuple[str, float]] = []
    for name, payload in category_scores.items():
        if not isinstance(payload, dict):
            continue
        sc = payload.get("score")
        if sc is None:
            continue
        try:
            out.append((str(name), float(sc)))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda x: -x[1])
    return out


def _category_balance_sentence(pairs: List[Tuple[str, float]]) -> str:
    """Simple positives/negatives from category scores."""
    if len(pairs) < 2:
        return ""
    top_score, bottom_score = pairs[0][1], pairs[-1][1]
    spread = top_score - bottom_score

    strong = _CATEGORY_PLAIN.get(pairs[0][0], pairs[0][0].lower())
    weak = _CATEGORY_PLAIN.get(pairs[-1][0], pairs[-1][0].lower())

    if spread < 10.0:
        return (
            " No single factor stands out too much; "
            "overall positives and risks look fairly balanced."
        )
    if len(pairs) == 2:
        return (
            f" It benefits from **{strong}**, "
            f"while **{weak}** is the key risk to watch."
        )

    second = _CATEGORY_PLAIN.get(pairs[1][0], pairs[1][0].lower())
    return (
        f" It benefits from **{strong}** and **{second}**, "
        f"which usually support stronger car-wash demand, but **{weak}** is the key risk to watch."
    )


def build_overall_site_analysis_verdict(
    *,
    predicted_tier: Optional[str],
    expected_volume_label: Optional[str],
    site_score: Optional[float],
    category_scores: Optional[Dict[str, Any]] = None,
    predicted_quantile: Optional[str] = None,
    quantile_probabilities: Optional[Dict[Any, float]] = None,
    weighted_volume_prediction: Optional[float] = None,
    operational_buffer: int = 20000,
) -> str:
    """
    Short, plain-English verdict with expected washes, +-20k range, and simple
    positives/negatives. Avoid technical labels and score jargon.
    """
    # Intentionally do not expose tier labels or site-score language.
    clean_label = _clean_volume_label(expected_volume_label)
    vol_part = (
        f"This site is expected to do about **{clean_label}** car washes per year."
        if clean_label
        else "This site does not have a volume forecast yet."
    )
    parts: List[str] = [vol_part]

    pairs = _category_pairs(category_scores)
    bal = _category_balance_sentence(pairs)
    if bal:
        parts.append(bal.strip())

    return " ".join(parts).strip()
