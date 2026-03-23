"""
Plain-English overall site verdict for /overall/{task_id}.

Uses weighted category scores and quantile outputs — not raw feature names.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# How we refer to each scoring bucket in user-facing text (not internal keys).
_CATEGORY_PLAIN: Dict[str, str] = {
    "Weather": "weather and seasons",
    "Competition": "competition nearby",
    "Retail": "nearby stores and food options",
    "Gas": "fuel-stop visibility and passing drivers",
}


def _norm_probability(p: Optional[float]) -> Optional[float]:
    if p is None:
        return None
    try:
        x = float(p)
    except (TypeError, ValueError):
        return None
    if x <= 1.0:
        return x * 100.0
    return x


def _predicted_band_probability(
    predicted_quantile: Optional[str],
    quantile_probabilities: Optional[Dict[Any, float]],
) -> Optional[float]:
    """Return 0–100 probability for the predicted band, if available."""
    if not quantile_probabilities or not predicted_quantile:
        return None
    pq = str(predicted_quantile).strip().upper()
    if not pq.startswith("Q"):
        pq = f"Q{pq}"
    raw = quantile_probabilities.get(pq)
    if raw is None:
        # Keys may be ints in some payloads
        try:
            qn = int(pq[1:])
            raw = quantile_probabilities.get(qn)
        except ValueError:
            raw = None
    return _norm_probability(raw)


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
    """
    Compare buckets using the same weighted category scores as site_score.
    Uses plain language — no feature names.
    """
    if len(pairs) < 2:
        return ""
    top_score = pairs[0][1]
    bottom_score = pairs[-1][1]
    spread = top_score - bottom_score
    if spread < 10.0:
        # All four buckets may not be present; mention generically.
        return (
            " Across the main areas we measure—weather, competition, nearby stores/food, "
            "and fuel-stop visibility—the picture is fairly balanced."
        )

    strong = _CATEGORY_PLAIN.get(pairs[0][0], pairs[0][0].lower())
    weak = _CATEGORY_PLAIN.get(pairs[-1][0], pairs[-1][0].lower())

    if len(pairs) == 2:
        return (
            f" Compared with other sites, **{strong}** score higher than **{weak}**."
        )

    second = _CATEGORY_PLAIN.get(pairs[1][0], pairs[1][0].lower())
    return (
        f" Compared with other sites, **{strong}** and **{second}** are the stronger parts of the picture, "
        f"while **{weak}** is the relative soft spot."
    )


def _confidence_sentence(
    predicted_quantile: Optional[str],
    quantile_probabilities: Optional[Dict[Any, float]],
) -> str:
    p = _predicted_band_probability(predicted_quantile, quantile_probabilities)
    if p is None:
        return ""
    if p >= 55.0:
        return f" The outlook mostly lines up with one volume band (about **{p:.0f}%** of the estimate)."
    if p >= 38.0:
        return " The outlook leans toward one volume band, but other bands are still plausible."
    return " Several volume bands are still in play—use the range as a guide, not a guarantee."


def build_overall_site_analysis_verdict(
    *,
    predicted_tier: Optional[str],
    expected_volume_label: Optional[str],
    site_score: Optional[float],
    category_scores: Optional[Dict[str, Any]] = None,
    predicted_quantile: Optional[str] = None,
    quantile_probabilities: Optional[Dict[Any, float]] = None,
) -> str:
    """
    Short markdown-friendly verdict: volume, score, category balance, confidence.

    Strengths/weakness lists from the quantile model often expose internal feature
    labels; we prefer category_scores (same weighting as site_score) for plain English.
    """
    # Intentionally do not expose tier labels in customer-facing text.
    vol_part = (
        f"This site is expected to do about **{expected_volume_label}** per year."
        if expected_volume_label
        else "This site does not have a volume forecast yet."
    )
    score_part = (
        f"Overall site score is **{site_score:.1f}/100**."
        if site_score is not None
        else ""
    )

    parts: List[str] = [p for p in [vol_part, score_part] if p]

    pairs = _category_pairs(category_scores)
    bal = _category_balance_sentence(pairs)
    if bal:
        parts.append(bal.strip())

    conf = _confidence_sentence(predicted_quantile, quantile_probabilities)
    if conf:
        parts.append(conf.strip())

    return " ".join(parts).strip()
