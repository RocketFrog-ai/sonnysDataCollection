# app/modelling/ai — narrative generation from quantile result (LLM agents).
# Stub implementations; wire LLM calls here for per-feature and overall summaries.

from app.modelling.ai.narratives import (
    get_feature_narratives,
    get_overall_narrative,
)

__all__ = [
    "get_feature_narratives",
    "get_overall_narrative",
]
