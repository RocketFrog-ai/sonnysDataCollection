"""
Competition narratives: per-metric summary (LLM), insight, observation for nearby car washes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.modelling.ai.common import extract_llm_text
from app.modelling.ai.competition.config import (
    COMPETITION_IMPACT_CLASSIFICATION_SUFFIX,
    COMPETITION_METRIC_DIRECTION,
    COMPETITION_METRIC_KEYS_ORDER,
    COMPETITION_METRIC_UNITS,
    COMPETITION_NARRATIVE_METRICS,
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
    val_str = f"{value:.1f}" if value is not None else "N/A"
    if value is not None and value == int(value):
        val_str = f"{int(value)}"
    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.1f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.1f}" if dist_max is not None else "N/A"
    if direction == "higher":
        direction_note = "higher values indicate stronger competitor presence (validated market)"
    elif direction == "lower":
        direction_note = "lower distance means closer competitor (more direct competition)"
    else:
        direction_note = "context-dependent for competitive intensity"

    prompt = f"""You are a car wash site analyst. Write one or two short sentences in plain, non-technical English. Refer to it as \"this site\" (not \"your site\"). Use the numbers given — no fixed template.

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile vs. other car wash sites: {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: {direction_note}.
Scale range in dataset: {min_str} – {max_str}

Write a dynamic rationale that states the value, explains the percentile in everyday words, and mentions quartile/category. Avoid jargon (no 'distribution', 'correlation'). Reply with only the rationale, no prefix or label."""

    summary: Optional[str] = None
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3, max_new_tokens=256)
        text = extract_llm_text(raw)
        if text:
            summary = text.strip()
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
    """One paragraph using quantile narrative for distance to nearest miles, Google rating, and user rating count."""
    pred_q = quantile_result.get("predicted_wash_quantile")
    pred_label = quantile_result.get("predicted_wash_quantile_label") or f"Q{pred_q}"
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"
    lines = ["Competition quantile metrics for this site (value, percentile, category):"]
    for n in feature_narratives:
        name = n.get("label") or n.get("feature_key", "")
        val = n.get("value")
        unit = n.get("unit", "")
        cat = n.get("category")
        pct = n.get("percentile")
        lines.append(f"- {name}: value={val} {unit}, percentile={pct}%, category={cat}")
    lines.append(f"\nPredicted wash volume band: {pred_label} ({wash_range}).")
    lines.append(
        "\nWrite one short paragraph (Insight) in plain English. Explain what the competition situation means for demand using: "
        "(1) distance to nearest competitor, (2) competitor rating, (3) competitor review count. "
        "Use the given percentiles/categories but describe them simply (e.g. 'better than most sites' or 'around average'). "
        "Reference the predicted wash band. Keep it to 2-4 sentences. Avoid technical jargon."
    )

    prompt = "\n".join(lines)
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3, max_new_tokens=512)
        text = extract_llm_text(raw)
        return text if text else None
    except Exception as e:
        logger.warning("Competition insight LLM failed: %s", e)
        return None


def _overall_agent(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    """Observation: total overall feature discussion (competitive picture for the site)."""
    feature_lines = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
        for n in feature_narratives
    ]
    prompt = (
        "Competition feature summaries for this car wash site:\n"
        + "\n".join(feature_lines)
        + "\n\nWrite an Observation paragraph (2-4 sentences): total overall feature discussion. "
        "Synthesize what the competition picture means for this site — competitive pressure, number of nearby car washes, distance to nearest, and brand strength (rating/reviews). "
        "Observation = overall feature takeaway, not raw quantile repetition. Prose only, no bullets. Reply with only the paragraph, no label."
    )
    out: Dict[str, Optional[str]] = {"observation": None}
    try:
        from app.utils.llm import local_llm as llm
        raw = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3, max_new_tokens=512)
        text = extract_llm_text(raw)
        if text:
            out["observation"] = text.strip()
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
    return {"observation": overall.get("observation"), "conclusion": None}
