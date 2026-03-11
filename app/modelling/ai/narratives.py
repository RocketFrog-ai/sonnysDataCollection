"""
Narrative generation from quantile result + feature values.

Stub implementations: return structured placeholders. Replace with LLM agents that take
quantile_result and feature_values and produce:
- Per-feature: summary, business_impact, impact_classification
- Overall: insight, observation, conclusion
"""

from __future__ import annotations

from typing import Any, Dict, List


def get_feature_narratives(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build per-feature narrative entries (summary, business_impact, impact_classification).
    Input: quantile_result from QuantilePredictorV3.analyze(), feature_values from site_analysis.
    Stub: returns one placeholder per feature in feature_analysis; replace with LLM agent.
    """
    feature_analysis = quantile_result.get("feature_analysis") or {}
    out: List[Dict[str, Any]] = []
    for feat_key, info in feature_analysis.items():
        out.append({
            "feature_key": feat_key,
            "label": info.get("label", feat_key),
            "value": info.get("value"),
            "percentile": info.get("adjusted_percentile"),
            "wash_q": info.get("wash_correlated_q"),
            "summary": None,  # LLM: e.g. "90% of sites generating 150k–350k washes..."
            "business_impact": None,  # LLM: e.g. "Frequent dirt triggers create strong recurring wash demand."
            "impact_classification": None,  # LLM: e.g. "Strong · 100–150 days"
        })
    return out


def get_overall_narrative(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build overall narrative (insight, observation, conclusion).
    Stub: returns placeholders; replace with LLM agent.
    """
    return {
        "insight": None,  # LLM: e.g. "Weather contributes ~20% to the site potential..."
        "observation": None,  # LLM: e.g. "Site ABC benefits from a well-rounded weather profile..."
        "conclusion": None,  # LLM: e.g. "The weather profile supports consistent year-round..."
    }
