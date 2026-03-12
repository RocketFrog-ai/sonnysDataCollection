"""
Narrative orchestration: delegates to feature-wise subfolders (weather, gas, retail, competition).
Top-level entry point for get_feature_narratives and get_overall_narrative used by site_analysis and routes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Feature narrative modules (add gas, retail, competition when ready)
from app.modelling.ai.weather import (
    get_feature_narratives as get_weather_feature_narratives,
    get_insight as get_weather_insight,
    get_overall_narrative as get_weather_overall_narrative,
)


def get_feature_narratives(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build per-feature narrative entries from all enabled feature modules.
    Currently returns weather narratives only; extend by merging results from
    gas, retail, competition subfolders when added.
    """
    out: List[Dict[str, Any]] = []
    out.extend(
        get_weather_feature_narratives(quantile_result, feature_values)
    )
    # out.extend(gas_narratives.get_feature_narratives(quantile_result, feature_values))
    # out.extend(retail_narratives.get_feature_narratives(...))
    # out.extend(competition_narratives.get_feature_narratives(...))
    return out


def get_overall_narrative(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
    feature_narratives: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build overall narrative (insight, observation, conclusion).
    If feature_narratives is provided, reuses them to avoid duplicate LLM calls.
    Currently weather-only; extend to merge or segment by dimension when more features exist.
    """
    if feature_narratives is None:
        feature_narratives = get_feature_narratives(quantile_result, feature_values)
    # For now overall is weather-only (insight + observation + conclusion)
    insight = get_weather_insight(quantile_result, feature_narratives)
    overall = get_weather_overall_narrative(
        quantile_result, feature_narratives
    )
    return {
        "insight": insight,
        "observation": overall.get("observation"),
        "conclusion": overall.get("conclusion"),
    }
