"""
Display mapping for v3 quantile (Q1–Q4) to UI category.

Used by the weather (and other) API to return quantile_score (percentile),
quantile (Q1/Q2/Q3/Q4), and category (Poor/Fair/Good/Strong) from the v3
feature_analysis (adjusted_percentile, wash_correlated_q).
"""

from typing import Optional

# Q1 → Poor, Q2 → Fair, Q3 → Good, Q4 → Strong (per product/quantile_report_v3)
QUANTILE_TO_CATEGORY = {
    1: "Poor",
    2: "Fair",
    3: "Good",
    4: "Strong",
}


def get_category_for_quantile(q: Optional[int]) -> Optional[str]:
    """Return display category for v3 wash_correlated_q (1–4)."""
    if q is None:
        return None
    return QUANTILE_TO_CATEGORY.get(int(q))
