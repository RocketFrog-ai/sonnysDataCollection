"""
Analysis entry point: delegates to app.modelling.site_analysis.

Single-fetch flow: geocode → fetch all features (features/active) → quantile prediction (v3).
Same fetched data is used for feature_values (routes) and quantile_result.
"""

from app.modelling.site_analysis import (
    analyze_site_from_dict,
    run_site_analysis,
)

__all__ = [
    "analyze_site_from_dict",
    "run_site_analysis",
]
