# app/site_analysis/modelling/ai — narrative generation from quantile result (LLM agents).
# Feature-wise subfolders: weather/, gas/, retail/, competition/ (add as needed).
# Top-level narratives.py orchestrates and delegates to each feature.

from app.site_analysis.modelling.ai.narratives import (
    get_competition_narrative,
    get_feature_narratives,
    get_gas_narrative,
    get_overall_narrative,
    get_retail_narrative,
)

__all__ = [
    "get_competition_narrative",
    "get_feature_narratives",
    "get_gas_narrative",
    "get_overall_narrative",
    "get_retail_narrative",
]
