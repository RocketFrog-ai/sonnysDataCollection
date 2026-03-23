"""Prompt builders for gas narratives."""

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
    val_str = f"{value:.2f}" if value is not None else "N/A"
    if value is not None and unit == "reviews":
        val_str = f"{int(value)}"
    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.2f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.2f}" if dist_max is not None else "N/A"
    direction_note = (
        "lower distance is better (closer gas station = more impulse potential)"
        if direction == "lower"
        else "higher values are better (stronger fuel traffic node)"
    )
    return f"""You are a car wash site analyst. Write one or two short plain-English sentences for this site.

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile vs. other sites: {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: {direction_note}
Scale range: {min_str} – {max_str}

Explain value + percentile in simple words and mention quartile/category. Reply with only the rationale."""


def build_insight_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    snap = format_forecast_snapshot_for_prompt(quantile_result)
    sums = format_plain_narrative_summaries(feature_narratives)
    if not sums:
        sums = (
            "(No short summaries available—describe the forecast in plain language only "
            "without naming internal metrics.)"
        )
    return "\n".join(
        [
            "Forecast (use these facts; do not invent numbers):",
            snap,
            "",
            sums,
            "",
            "Write one short Insight paragraph (2 sentences) on how nearby gas stations "
            "(distance, ratings, busy-ness) affect wash demand. "
            "Everyday words only—no quartile codes, percentiles, or metric field names.",
        ]
    )


def build_observation_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    context_block = format_overall_dimension_context(quantile_result, feature_narratives)

    return f"""
You are a car wash site analyst. Write a short, clear explanation in simple, everyday English that a layman can easily understand.

Refer to it as "this site" (not "your site").

Facts from the forecast and gas-station check (use only these numbers and ideas; do not invent):
{context_block}

Instructions:
- Write 2 short sentences (strictly 2; not too long, not too short)
- Combine all points into a smooth, natural explanation (do not just list them)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why the nearby gas ecosystem affects this site's car wash demand
- Use cause-and-effect reasoning (gas → demand)

Strict Rules:
- No jargon or technical terms (no quartiles, percentiles, “model features”, or variable names)
- Do not name metric titles from the bullet text; paraphrase the ideas only
- Avoid formal phrases like "indicates", "suggests", "positions", "accumulation"
- Avoid long or complex sentences
- Do NOT repeat the same idea
- Make it sound human and day-to-day conversational, never robotic or AI-generated

Output Format (STRICT):
Observation: <2 sentence explanation combining the factors>
"""


def build_conclusion_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    context_block = format_overall_dimension_context(quantile_result, feature_narratives)

    return f"""
You are a car wash site analyst. Write a very short, natural wrap-up in simple, everyday English.

Refer to it as "this site" (not "your site").

Facts from the forecast and gas-station check (use only these numbers and ideas; do not invent):
{context_block}

Instructions:
- Write 1–2 sentences max
- Explain the overall takeaway for car wash demand in a natural way
- Use simple cause-and-effect language
- Avoid jargon and long sentences; do not use quartile codes or metric field names
- Make it sound human and day-to-day conversational, never robotic or AI-generated

Output Format (STRICT):
Conclusion: <1 short sentence stating expected wash band in a natural way>
"""
