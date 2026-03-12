"""
Weather narratives: per-metric summary (LLM), fixed business impact & impact classification,
insight and overall (observation, conclusion) agents.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.modelling.ai.common import extract_llm_text
from app.modelling.ai.weather.config import (
    WEATHER_IMPACT_CLASSIFICATION_SUFFIX,
    WEATHER_METRIC_BUSINESS_IMPACT,
    WEATHER_METRIC_KEYS_ORDER,
    WEATHER_METRIC_UNITS,
    WEATHER_NARRATIVE_METRICS,
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
) -> Dict[str, Optional[str]]:
    """
    Call local LLM only for the dynamic summary. Business impact and impact classification
    are fixed per metric (from config). Returns dict with summary, business_impact, impact_classification.
    """
    val_str = f"{value:.0f}" if value is not None and value == int(value) else (f"{value}" if value is not None else "N/A")
    pct_str = f"{percentile:.0f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.0f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.0f}" if dist_max is not None else "N/A"

    prompt = f"""You are a car wash site analyst. For the following weather metric, write one short summary sentence.

Metric: {display_name}
Subtitle: {subtitle}
Site value: {val_str} {unit}
Category (Poor/Fair/Good/Strong): {cat_str}
Percentile (vs. other sites): {pct_str}
Scale range (min–max for context): {min_str} – {max_str}
Quantile band: {quantile_label or 'N/A'}

Write a single sentence summary, e.g. "X% of sites generating Y–Z washes per year have a {display_name} of A–B {unit}. This site falls within that range."
Reply with only that sentence, no prefix or label."""

    summary: Optional[str] = None
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3)
        text = extract_llm_text(raw)
        if text:
            summary = text.strip()
    except Exception as e:
        logger.warning("Feature summary LLM call failed for %s: %s", metric_key, e)

    business_impact = WEATHER_METRIC_BUSINESS_IMPACT.get(metric_key)
    suffix = WEATHER_IMPACT_CLASSIFICATION_SUFFIX.get(metric_key, "days")
    if category is not None and dist_min is not None and dist_max is not None:
        impact_classification = f"{category} · {dist_min:.0f}–{dist_max:.0f} {suffix}"
    elif category is not None:
        impact_classification = category
    else:
        impact_classification = None

    return {
        "summary": summary,
        "business_impact": business_impact,
        "impact_classification": impact_classification,
    }


def _insight_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Collate the 4 weather quantile results and generate one Insight paragraph
    (e.g. "Weather contributes ~20% to the site potential. With a Weather Impact Score of 70%...").
    """
    pred_q = quantile_result.get("predicted_wash_quantile")
    pred_label = quantile_result.get("predicted_wash_quantile_label") or f"Q{pred_q}"
    proba = quantile_result.get("quantile_probabilities") or {}
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"

    lines = [
        "Weather metrics for this site:",
    ]
    for n in feature_narratives:
        name = n.get("label") or n.get("feature_key", "")
        val = n.get("value")
        unit = n.get("unit", "")
        cat = n.get("category") or n.get("wash_q")
        pct = n.get("percentile")
        line = f"- {name}: value={val} {unit}, category={cat}, percentile={pct}%"
        lines.append(line)
    lines.append(f"\nPredicted wash volume band: {pred_label} ({wash_range}). Probabilities: {proba}")
    lines.append("\nWrite one short paragraph (Insight) summarizing how weather contributes to site potential and what the combined weather profile means for wash demand. Start with something like 'Weather contributes ~X% to the site potential.' Keep it to 2-4 sentences.")

    prompt = "\n".join(lines)
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3)
        text = extract_llm_text(raw)
        return text if text else None
    except Exception as e:
        logger.warning("Insight LLM call failed: %s", e)
        return None


def _overall_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """
    From quantile results and per-feature business impacts, generate Observation and Conclusion.
    """
    business_impacts = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('business_impact') or 'N/A'}"
        for n in feature_narratives
    ]
    pred_label = quantile_result.get("predicted_wash_quantile_label") or "N/A"
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"

    prompt = f"""You are a car wash site analyst. Based on the following weather feature business impacts and predicted wash band, write two short paragraphs.

Predicted wash band: {pred_label} ({wash_range})

Per-feature business impacts:
{chr(10).join(business_impacts)}

Respond with exactly these two lines:
Observation: [2-3 sentences: how the site benefits from its weather profile, e.g. "Site benefits from a well-rounded weather profile. Dirt triggers are frequent enough to sustain recurring demand, comfortable days provide a long window for discretionary washes, and shutdown risk is minimal."]
Conclusion: [1-2 sentences: overall takeaway for weather-driven demand, e.g. "The weather profile supports consistent year-round wash activity with moderate seasonal variation, making this a favorable location from a weather-driven demand perspective."]"""

    out: Dict[str, Optional[str]] = {"observation": None, "conclusion": None}
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3)
        text = extract_llm_text(raw)
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
    """
    Build per-feature narrative entries for weather metrics.
    Returns list of dicts with feature_key (v3 key), label, value, unit, category, percentile,
    summary, business_impact, impact_classification.
    """
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
                "business_impact": None,
                "impact_classification": None,
            })
            continue

        value = fa.get("value")
        if value is not None and hasattr(value, "__float__"):
            value = float(value)
        percentile = fa.get("adjusted_percentile")
        wash_q = fa.get("wash_correlated_q")
        category = fa.get("category") or get_category_for_quantile(wash_q)
        quantile_label = f"Q{int(wash_q)}" if wash_q is not None else None
        dist_min = fa.get("dist_min")
        dist_max = fa.get("dist_max")
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
            "business_impact": llm_out.get("business_impact"),
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
