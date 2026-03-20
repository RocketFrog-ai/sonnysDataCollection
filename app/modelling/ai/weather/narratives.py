"""Weather narrative generation."""

from __future__ import annotations

import argparse
import logging
import re
from typing import Any, Dict, List, Optional

from app.modelling.ai.common import get_llm_text
from app.modelling.ai.weather.config import (
    WEATHER_IMPACT_CLASSIFICATION_SUFFIX,
    WEATHER_METRIC_DIRECTION,
    WEATHER_METRIC_KEYS_ORDER,
    WEATHER_METRIC_UNITS,
    WEATHER_NARRATIVE_METRICS,
)
from app.modelling.ai.weather.prompts import (
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
    """Generate summary and impact label for one metric."""
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
        logger.warning("Feature summary LLM call failed for %s: %s", metric_key, e)

    suffix = WEATHER_IMPACT_CLASSIFICATION_SUFFIX.get(metric_key, "days")
    if category is not None and dist_min is not None and dist_max is not None:
        impact_classification = f"{category} · {dist_min:.0f}–{dist_max:.0f} {suffix}"
    elif category is not None:
        impact_classification = category
    else:
        impact_classification = None

    return {
        "summary": summary,
        "impact_classification": impact_classification,
    }


def _insight_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Optional[str]:
    """Generate one weather insight paragraph."""
    prompt = build_insight_prompt(quantile_result, feature_narratives)
    try:
        text = get_llm_text(prompt, max_new_tokens=512)
        if text:
            text = re.sub(r'^(?:\*\*)?Insight:(?:\*\*)?\s*', '', text.strip(), flags=re.IGNORECASE)
        return text if text else None
    except Exception as e:
        logger.warning("Insight LLM call failed: %s", e)
        return None


def _overall_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """Generate observation and conclusion for weather."""
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
        logger.warning("Overall narrative LLM call failed: %s", e)
    return out



def get_feature_narratives(
    quantile_result: Dict[str, Any],
    feature_values: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build per-feature weather narratives."""
    feature_analysis = quantile_result.get("feature_analysis") or {}
    out: List[Dict[str, Any]] = []

    for metric_key in WEATHER_METRIC_KEYS_ORDER:
        if metric_key not in WEATHER_NARRATIVE_METRICS:
            continue
        display_name, subtitle, v3_key = WEATHER_NARRATIVE_METRICS[metric_key]
        fa = feature_analysis.get(v3_key)
        if not fa:
            out.append({
                "feature_key": v3_key,
                "metric_key": metric_key,
                "label": display_name,
                "subtitle": subtitle,
                "value": None,
                "unit": WEATHER_METRIC_UNITS.get(metric_key, ""),
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

        direction = WEATHER_METRIC_DIRECTION.get(metric_key, "higher")

        raw_pct = fa.get("raw_percentile")

        if raw_pct is None:
            percentile = None
        else:
            percentile = float(raw_pct)
            if direction == "lower":
                percentile = 100.0 - percentile

        if percentile is None:
            metric_q = None
        elif percentile <= 25:
            metric_q = 1
        elif percentile <= 50:
            metric_q = 2
        elif percentile <= 75:
            metric_q = 3
        else:
            metric_q = 4

        q_bounds = fa.get("quantile_boundaries") or []
        dist_min = None
        dist_max = None
        try:
            if metric_q is not None and isinstance(q_bounds, list) and len(q_bounds) >= 5:
                q = int(metric_q)
                if direction == "higher":
                    dist_min = float(q_bounds[q - 1])
                    dist_max = float(q_bounds[q])
                else:
                    inv_q = 5 - q
                    dist_min = float(q_bounds[inv_q - 1])
                    dist_max = float(q_bounds[inv_q])
        except Exception:
            dist_min = fa.get("dist_min")
            dist_max = fa.get("dist_max")

        category = fa.get("category") or get_category_for_quantile(metric_q)
        quantile_label = f"Q{int(metric_q)}" if metric_q is not None else None

        wash_q = metric_q
        unit = WEATHER_METRIC_UNITS.get(metric_key, "days/year")

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
    """Generate one Insight paragraph from weather quantile results."""
    return _insight_agent(quantile_result, feature_narratives)


def get_overall_narrative(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate observation and conclusion from weather quantile results and business impacts."""
    overall = _overall_agent(quantile_result, feature_narratives)
    return {
        "observation": overall.get("observation"),
        "conclusion": overall.get("conclusion"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather narrative debug runner")
    parser.add_argument("--show-prompts", action="store_true", help="Print generated prompts")
    args = parser.parse_args()

    payload = {
        "quantile_result": {
            "predicted_wash_quantile": 3,
            "predicted_wash_quantile_label": "Q3",
            "predicted_wash_range": {"label": "180-220 washes/day"},
            "quantile_probabilities": {"Q1": 0.05, "Q2": 0.2, "Q3": 0.55, "Q4": 0.2},
            "feature_analysis": {
                "weather_rainy_days": {
                    "value": 126,
                    "raw_percentile": 68.0,
                    "quantile_boundaries": [60, 90, 115, 140, 185],
                },
                "weather_total_snowfall_cm": {
                    "value": 84,
                    "raw_percentile": 62.0,
                    "quantile_boundaries": [10, 35, 60, 95, 140],
                },
                "weather_days_pleasant_temp": {
                    "value": 205,
                    "raw_percentile": 72.0,
                    "quantile_boundaries": [120, 160, 190, 225, 270],
                },
                "weather_days_below_freezing": {
                    "value": 38,
                    "raw_percentile": 30.0,
                    "quantile_boundaries": [5, 20, 45, 70, 110],
                },
            },
        },
        "feature_values": {},
    }

    qr = payload.get("quantile_result", {})
    fv = payload.get("feature_values", {})

    if args.show_prompts:
        fa_map = qr.get("feature_analysis") or {}
        for metric_key in WEATHER_METRIC_KEYS_ORDER:
            if metric_key not in WEATHER_NARRATIVE_METRICS:
                continue
            display_name, subtitle, v3_key = WEATHER_NARRATIVE_METRICS[metric_key]
            fa = fa_map.get(v3_key) or {}
            value = fa.get("value")
            if value is not None and hasattr(value, "__float__"):
                value = float(value)
            direction = WEATHER_METRIC_DIRECTION.get(metric_key, "higher")
            raw_pct = fa.get("raw_percentile")
            percentile = None if raw_pct is None else float(raw_pct)
            if percentile is not None and direction == "lower":
                percentile = 100.0 - percentile
            if percentile is None:
                metric_q = None
            elif percentile <= 25:
                metric_q = 1
            elif percentile <= 50:
                metric_q = 2
            elif percentile <= 75:
                metric_q = 3
            else:
                metric_q = 4
            q_bounds = fa.get("quantile_boundaries") or []
            dist_min = None
            dist_max = None
            try:
                if metric_q is not None and isinstance(q_bounds, list) and len(q_bounds) >= 5:
                    q = int(metric_q)
                    if direction == "higher":
                        dist_min = float(q_bounds[q - 1])
                        dist_max = float(q_bounds[q])
                    else:
                        inv_q = 5 - q
                        dist_min = float(q_bounds[inv_q - 1])
                        dist_max = float(q_bounds[inv_q])
            except Exception:
                dist_min = fa.get("dist_min")
                dist_max = fa.get("dist_max")
            category = fa.get("category") or get_category_for_quantile(metric_q)
            quantile_label = f"Q{int(metric_q)}" if metric_q is not None else None
            print(f"\n--- FEATURE PROMPT: {metric_key} ---")
            print(
                build_feature_summary_prompt(
                    display_name=display_name,
                    subtitle=subtitle,
                    value=value,
                    unit=WEATHER_METRIC_UNITS.get(metric_key, "days/year"),
                    category=category,
                    percentile=percentile,
                    dist_min=dist_min,
                    dist_max=dist_max,
                    quantile_label=quantile_label,
                    direction=direction,
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
