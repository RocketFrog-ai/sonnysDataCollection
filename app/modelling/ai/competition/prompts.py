"""Prompt builders for competition narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modelling.ai.common import (
    format_forecast_snapshot_for_prompt,
    format_overall_dimension_context,
    format_plain_narrative_summaries,
)


def build_feature_summary_prompt(
    *,
    display_name: str,
    subtitle: str,
    value: Optional[float],
    unit: str,
    category: Optional[str],
    percentile: Optional[float],
    dist_min: Optional[float],
    dist_max: Optional[float],
    quantile_label: Optional[str],
    direction: str,
) -> str:
    val_str = f"{value:.1f}" if value is not None else "N/A"
    if value is not None and value == int(value):
        val_str = f"{int(value)}"
    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.1f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.1f}" if dist_max is not None else "N/A"
    if direction == "higher":
        direction_note = "higher values indicate stronger competitor presence"
    elif direction == "lower":
        direction_note = "lower distance means closer competitor"
    else:
        direction_note = "context-dependent for competitive intensity"

    return f"""You are a car wash site analyst. Write one or two short plain-English sentences about the local market environment.

IMPORTANT CONTEXT:
1. The site being analyzed is a NEW proposed development and has NO ratings or reviews of its own yet.
2. The values below describe the CLOSEST CAR WASH near to the site (the competitor).

Metric: {display_name} ({subtitle})
Market value: {val_str} {unit}
Percentile vs. other sites: {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: {direction_note}
Scale range: {min_str} – {max_str}

Explain how this neighboring wash metric reflects the local competitive intensity in everyday words. Reply with only the rationale."""


def build_insight_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    car_wash_type: Optional[str] = None,
) -> str:
    snap = format_forecast_snapshot_for_prompt(quantile_result)
    sums = format_plain_narrative_summaries(feature_narratives)
    if not sums:
        sums = (
            "(No short summaries available—describe the forecast in plain language only "
            "without naming internal metrics.)"
        )
    parts: List[str] = []
    if car_wash_type:
        parts.append(f"Site type: {car_wash_type}.")
    parts.extend(
        [
            "Forecast (use these facts; do not invent numbers):",
            snap,
            "",
            "Site context: This is a NEW proposed car wash development (no ratings or reviews yet).",
            "",
            sums,
            "",
            "Write one short Insight paragraph (2–4 sentences) about how the competition from the **closest car wash near to you** affects your future wash demand. "
            "Refer to neighbors as 'the closest car wash' or 'neighboring wash'. "
            "Never say 'your ratings' or 'your reviews' because the site is not built yet. "
            "Use everyday words only—no quartile codes, percentiles, or metric field names.",
        ]
    )
    return "\n".join(parts)


def build_overall_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    car_wash_type: Optional[str] = None,
) -> str:
    context_block = format_overall_dimension_context(quantile_result, feature_narratives)
    site_type_line = f"\nSite type: {car_wash_type}" if car_wash_type else ""

    return f"""
You are a car wash site analyst. Write a short, clear explanation in simple, everyday English that a layman can easily understand.

Refer to it as "this site" (not "your site").
{site_type_line}

Site context: This is a NEW proposed car wash development (no ratings or reviews yet).

Facts from the forecast and the **closest car wash near to you** (use only these numbers and ideas; do not invent):
{context_block}

Instructions:
- Write 2–3 short sentences (not too long, not too short)
- Combine all points into a smooth, natural explanation (do not just list them)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why the competition situation affects this site's car wash demand
- Use cause-and-effect reasoning (competition → demand)

Strict Rules:
- No jargon or technical terms (no quartiles, percentiles, “model features”, or variable names)
- Do not name metric titles from the bullet text; paraphrase the ideas only
- Avoid formal phrases like "indicates", "suggests", "positions", "accumulation"
- Avoid long or complex sentences
- Do NOT repeat the same idea
- Do NOT sound like a report

Style Guidance:
- Write like you are explaining to a normal person
- Keep it natural, smooth, and easy to follow
- Use simple connectors like "because", "so", "which means"
- Make it sound human and day-to-day conversational, never robotic or AI-generated
- Strictly refer to distance and ratings as belonging to the **closest car wash near to you** or the **neighboring wash**
- Do NOT use phrases like "your ratings" or "your popularity" for this site
- Do NOT use the word "rival"; use "closest car wash near to you" or "neighboring wash" instead

Output Format (STRICT):
Observation: <2–3 sentence explanation combining the factors>
Conclusion: <1 short sentence stating expected wash band in a natural way>

Example style (do not copy):
Observation: This site faces a balanced level of nearby competition, so customers have options, but your combination of distance and local reputation still supports strong demand. That keeps people choosing a wash without needing to over-discount.
Conclusion: Because of this, the site can expect around 130–170 washes per day.
"""
