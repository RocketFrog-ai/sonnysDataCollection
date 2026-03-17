"""
Weather narratives: per-metric summary (LLM; value, percentile, quartile, category),
impact classification, insight and overall (observation, conclusion) agents.
Business impact is not returned; frontend handles it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.modelling.ai.common import extract_llm_text
from app.modelling.ai.weather.config import (
    WEATHER_IMPACT_CLASSIFICATION_SUFFIX,
    WEATHER_METRIC_DIRECTION,
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
    direction: str = "higher",
) -> Dict[str, Optional[str]]:
    """
    Call local LLM for a dynamic summary. No fixed template: use actual value, percentile,
    quartile, category, and direction (higher/lower better) to write a short rationale
    for car wash sites. Returns dict with summary, impact_classification.
    """
    val_str = f"{value:.0f}" if value is not None and value == int(value) else (f"{value}" if value is not None else "N/A")
    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.0f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.0f}" if dist_max is not None else "N/A"
    direction_note = "higher values are better for wash demand" if direction == "higher" else "lower values are better for wash demand"

    prompt = f"""You are a car wash site analyst. Write one or two short sentences in plain, non-technical English. Use the numbers given — do not use a fixed template.

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile (vs. other car wash sites in the dataset): {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: For this metric, {direction_note}.
Scale range in dataset: {min_str} – {max_str}

Write a dynamic rationale that:
- Explains what the percentile means using everyday words (e.g. "better than about X% of sites" or "worse than about X% of sites" depending on the direction).
- Mentions quartile and category (Q1–Q4, Poor/Fair/Good/Strong).
- Notes whether higher or lower is better for this metric so the reader understands the interpretation.
Avoid jargon (no 'distribution', 'correlation', 'quantile boundaries'). Reply with only the rationale, no prefix or label."""

    summary: Optional[str] = None
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3, max_new_tokens=256)
        text = extract_llm_text(raw)
        if text:
            summary = text.strip()
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
    """
    Collate the 4 weather quantile results and generate one Insight paragraph
    that explains how the combined weather profile affects wash demand and
    operational uptime, without forcing a fixed "% of site potential" line.
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
    lines.append(
        "\nWrite one short paragraph (Insight) explaining how the combined weather "
        "profile affects car wash demand and operations. Use the metrics above to "
        "highlight which aspects are strongest and which are weaker relative to other "
        "sites, and connect them to wash demand and operational uptime. Focus on "
        "qualitative strength/weakness language (e.g. 'strong tailwind', 'moderate drag') "
        "and do not claim that weather contributes a specific percentage of overall site "
        "potential. Keep it to 2-4 sentences."
    )

    prompt = "\n".join(lines)
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3, max_new_tokens=512)
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
    From quantile results and per-feature summaries, generate Observation and Conclusion.
    """
    feature_lines = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
        for n in feature_narratives
    ]
    pred_label = quantile_result.get("predicted_wash_quantile_label") or "N/A"
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"

    prompt = f"""You are a car wash site analyst. Based on the following weather feature business impacts and predicted wash band, write two short paragraphs.

Predicted wash band: {pred_label} ({wash_range})

Per-feature summaries (value, percentile, quartile, category):
{chr(10).join(feature_lines)}

Respond with exactly these two lines:
Observation: [2-3 sentences: how the site benefits from its weather profile, e.g. "Site benefits from a well-rounded weather profile. Dirt triggers are frequent enough to sustain recurring demand, comfortable days provide a long window for discretionary washes, and shutdown risk is minimal."]
Conclusion: [1-2 sentences: overall takeaway for weather-driven demand, e.g. "The weather profile supports consistent year-round wash activity with moderate seasonal variation, making this a favorable location from a weather-driven demand perspective."]"""

    out: Dict[str, Optional[str]] = {"observation": None, "conclusion": None}
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3, max_new_tokens=512)
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
    Returns list of dicts with feature_key, label, value, unit, category, percentile,
    wash_q (quartile), summary, impact_classification. No business_impact (frontend handles it).
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
                "impact_classification": None,
            })
            continue

        value = fa.get("value")
        if value is not None and hasattr(value, "__float__"):
            value = float(value)
        percentile = fa.get("adjusted_percentile")
        wash_q = fa.get("wash_correlated_q")
        # Some weather metrics (e.g. Days Below Freezing / Shutdown Risk) can be flagged low-signal
        # and therefore have wash_correlated_q=None. Still, we should provide a category label for
        # impact_classification and narrative readability. Fall back to feature's own adjusted quantile.
        q_for_category = wash_q if wash_q is not None else fa.get("feature_quantile_adj") or fa.get("feature_quantile_raw")
        category = fa.get("category") or get_category_for_quantile(q_for_category)
        quantile_label = f"Q{int(q_for_category)}" if q_for_category is not None else None
        dist_min = fa.get("dist_min")
        dist_max = fa.get("dist_max")
        unit = WEATHER_METRIC_UNITS.get(metric_key, "days/year")

        direction = WEATHER_METRIC_DIRECTION.get(metric_key, "higher")
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
