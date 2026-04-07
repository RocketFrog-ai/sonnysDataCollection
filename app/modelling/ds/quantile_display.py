"""
Display mapping for wash tier quantile (Q1–Q4) to UI category.

Used by APIs to map `wash_correlated_q` to Poor / Fair / Good / Strong.
"""

from typing import Optional

# Q1 → Poor, Q2 → Fair, Q3 → Good, Q4 → Strong
QUANTILE_TO_CATEGORY = {
    1: "Poor",
    2: "Fair",
    3: "Good",
    4: "Strong",
}


def get_category_for_quantile(q: Optional[int]) -> Optional[str]:
    """Return display category for wash_correlated_q (1–4)."""
    if q is None:
        return None
    return QUANTILE_TO_CATEGORY.get(int(q))
