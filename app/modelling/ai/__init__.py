# app/modelling/ai — narrative generation from quantile result (LLM agents).
# Feature-wise subfolders: weather/, gas/, retail/, competition/ (add as needed).
# Top-level narratives.py orchestrates and delegates to each feature.

from app.modelling.ai.narratives import (
    get_feature_narratives,
    get_overall_narrative,
)

__all__ = [
    "get_feature_narratives",
    "get_overall_narrative",
]
