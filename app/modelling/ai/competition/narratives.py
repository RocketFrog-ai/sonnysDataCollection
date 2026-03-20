"""Competition narrative generation."""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Dict, List, Optional

from app.modelling.ai.common import get_llm_text
from app.modelling.ai.competition.config import (
    COMPETITION_IMPACT_CLASSIFICATION_SUFFIX,
    COMPETITION_METRIC_DIRECTION,
    COMPETITION_METRIC_KEYS_ORDER,
    COMPETITION_METRIC_UNITS,
    COMPETITION_NARRATIVE_METRICS,
)
from app.modelling.ai.competition.prompts import (
    build_feature_summary_prompt,
    build_insight_prompt,
    build_overall_prompt,
)
from app.modelling.ds.quantile_display import get_category_for_quantile

logger = logging.getLogger(__name__)


def _feature_summary_agent(
    metric_key: str,
    display_name: str,
    subtitle: str,
    value: Optional[float],
    unit: str,
    category: Optional[str],
    percentile: Optional[float],
    dist_min: Optional[float],
    dist_max: Optional[float],
    quantile_label: Optional[str],
    direction: str = "higher",
) -> Dict[str, Optional[str]]:
    """LLM-generated dynamic summary for one competition metric."""
    prompt = build_feature_summary_prompt(
        display_name=display_name,
        subtitle=subtitle,
        value=value,
        unit=unit,
        category=category,
        percentile=percentile,
        dist_min=dist_min,
        dist_max=dist_max,
        quantile_label=quantile_label,
        direction=direction,
    )

    summary: Optional[str] = None
    try:
        summary = get_llm_text(prompt, max_new_tokens=256)
    except Exception as e:
        logger.warning("Competition feature summary LLM failed for %s: %s", metric_key, e)

    suffix = COMPETITION_IMPACT_CLASSIFICATION_SUFFIX.get(metric_key, "—")
    if category is not None and dist_min is not None and dist_max is not None:
        impact_classification = f"{category} · {dist_min:.1f}–{dist_max:.1f} {suffix}"
    elif category is not None:
        impact_classification = category
    else:
        impact_classification = None

    return {"summary": summary, "impact_classification": impact_classification}


def _insight_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Optional[str]:
    """Generate one competition insight paragraph."""
    prompt = build_insight_prompt(quantile_result, feature_narratives)
    try:
        text = get_llm_text(prompt, max_new_tokens=512)
        return text if text else None
    except Exception as e:
        logger.warning("Competition insight LLM failed: %s", e)
        return None


def _overall_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """Generate competition observation and conclusion."""
    prompt = build_overall_prompt(quantile_result, feature_narratives)
    out: Dict[str, Optional[str]] = {"observation": None, "conclusion": None}
    try:
        text = get_llm_text(prompt, max_new_tokens=512)
        if not text:
            return out
        obs_m = re.search(r"Observation:\s*(.+?)(?=\s*Conclusion:|$)", text, re.DOTALL | re.IGNORECASE)
        con_m = re.search(r"Conclusion:\s*(.+?)(?=\s*Observation:|$)", text, re.DOTALL | re.IGNORECASE)
        if obs_m:
            out["observation"] = obs_m.group(1).strip()
        if con_m:
            out["conclusion"] = con_m.group(1).strip()
    except Exception as e:
        logger.warning("Competition overall LLM failed: %s", e)
    return out


def get_feature_narratives(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Per-metric competition narratives (same-format count, distance to nearest, brand strength)."""
    feature_analysis = quantile_result.get("feature_analysis") or {}
    out: List[Dict[str, Any]] = []

    for metric_key in COMPETITION_METRIC_KEYS_ORDER:
        if metric_key not in COMPETITION_NARRATIVE_METRICS:
            continue
        display_name, subtitle, v3_key = COMPETITION_NARRATIVE_METRICS[metric_key]
        fa = feature_analysis.get(v3_key)
        if not fa:
            out.append({
                "feature_key": v3_key,
                "metric_key": metric_key,
                "label": display_name,
                "subtitle": subtitle,
                "value": None,
                "unit": COMPETITION_METRIC_UNITS.get(metric_key, ""),
                "category": None,
                "percentile": None,
                "wash_q": None,
                "summary": None,
                "impact_classification": None,
            })
            continue

        value = fa.get("value")
        if value is not None and hasattr(value, "__float__"):
            value = float(value)
        percentile = fa.get("adjusted_percentile")
        wash_q = fa.get("wash_correlated_q")
        feature_q = fa.get("feature_quantile_adj")
        category = fa.get("category") or get_category_for_quantile(wash_q) or get_category_for_quantile(feature_q)
        quantile_label = f"Q{int(wash_q)}" if wash_q is not None else (f"Q{int(feature_q)}" if feature_q is not None else None)
        dist_min = fa.get("dist_min")
        dist_max = fa.get("dist_max")
        unit = COMPETITION_METRIC_UNITS.get(metric_key, "—")
        direction = COMPETITION_METRIC_DIRECTION.get(metric_key, "higher")

        llm_out = _feature_summary_agent(
            metric_key=metric_key,
            display_name=display_name,
            subtitle=subtitle,
            value=value,
            unit=unit,
            category=category,
            percentile=percentile,
            dist_min=dist_min,
            dist_max=dist_max,
            quantile_label=quantile_label,
            direction=direction,
        )

        out.append({
            "feature_key": v3_key,
            "metric_key": metric_key,
            "label": display_name,
            "subtitle": subtitle,
            "value": value,
            "unit": unit,
            "category": category,
            "percentile": percentile,
            "wash_q": wash_q,
            "summary": llm_out.get("summary"),
            "impact_classification": llm_out.get("impact_classification"),
        })

    return out


def get_insight(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Optional[str]:
    return _insight_agent(quantile_result, feature_narratives)


def get_overall_narrative(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Any]:
    overall = _overall_agent(quantile_result, feature_narratives)
    return {"observation": overall.get("observation"), "conclusion": overall.get("conclusion")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competition narrative debug runner")
    parser.add_argument("--show-prompts", action="store_true", help="Print generated prompts")
    args = parser.parse_args()

    payload = {
        "quantile_result": {
            "predicted_wash_quantile": 2,
            "predicted_wash_quantile_label": "Q2",
            "predicted_wash_range": {"label": "130-170 washes/day"},
            "feature_analysis": {
                "competitors_count_4miles": {
                    "value": 6,
                    "adjusted_percentile": 63.0,
                    "wash_correlated_q": 3,
                    "feature_quantile_adj": 3,
                    "dist_min": 4,
                    "dist_max": 8,
                    "category": "Good",
                },
                "competitor_1_distance_miles": {
                    "value": 1.35,
                    "adjusted_percentile": 58.0,
                    "wash_correlated_q": 3,
                    "feature_quantile_adj": 3,
                    "dist_min": 0.8,
                    "dist_max": 1.9,
                    "category": "Good",
                },
                "competitor_1_google_rating": {
                    "value": 4.3,
                    "adjusted_percentile": 71.0,
                    "wash_correlated_q": 3,
                    "feature_quantile_adj": 3,
                    "dist_min": 4.0,
                    "dist_max": 4.5,
                    "category": "Good",
                },
                "competitor_1_rating_count": {
                    "value": 245,
                    "adjusted_percentile": 66.0,
                    "wash_correlated_q": 3,
                    "feature_quantile_adj": 3,
                    "dist_min": 120,
                    "dist_max": 310,
                    "category": "Good",
                },
                "competition_quality": {
                    "value": 0.74,
                    "adjusted_percentile": 69.0,
                    "wash_correlated_q": 3,
                    "feature_quantile_adj": 3,
                    "dist_min": 0.55,
                    "dist_max": 0.82,
                    "category": "Good",
                },
            },
        },
        "feature_values": {},
    }

    qr = payload.get("quantile_result", {})
    fv = payload.get("feature_values", {})

    if args.show_prompts:
        fa_map = qr.get("feature_analysis") or {}
        for metric_key in COMPETITION_METRIC_KEYS_ORDER:
            if metric_key not in COMPETITION_NARRATIVE_METRICS:
                continue
            display_name, subtitle, v3_key = COMPETITION_NARRATIVE_METRICS[metric_key]
            fa = fa_map.get(v3_key) or {}
            value = fa.get("value")
            if value is not None and hasattr(value, "__float__"):
                value = float(value)
            percentile = fa.get("adjusted_percentile")
            wash_q = fa.get("wash_correlated_q")
            feature_q = fa.get("feature_quantile_adj")
            category = fa.get("category") or get_category_for_quantile(wash_q) or get_category_for_quantile(feature_q)
            quantile_label = f"Q{int(wash_q)}" if wash_q is not None else (f"Q{int(feature_q)}" if feature_q is not None else None)
            print(f"\n--- FEATURE PROMPT: {metric_key} ---")
            print(
                build_feature_summary_prompt(
                    display_name=display_name,
                    subtitle=subtitle,
                    value=value,
                    unit=COMPETITION_METRIC_UNITS.get(metric_key, "—"),
                    category=category,
                    percentile=percentile,
                    dist_min=fa.get("dist_min"),
                    dist_max=fa.get("dist_max"),
                    quantile_label=quantile_label,
                    direction=COMPETITION_METRIC_DIRECTION.get(metric_key, "higher"),
                )
            )

    features = get_feature_narratives(qr, fv)
    if args.show_prompts:
        print("\n--- INSIGHT PROMPT ---")
        print(build_insight_prompt(qr, features))
        print("\n--- OVERALL PROMPT ---")
        print(build_overall_prompt(qr, features))
    print("Feature narratives:", features)
    print("Insight:", get_insight(qr, features))
    print("Overall:", get_overall_narrative(qr, features))
