"""Prompt builders for retail narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modelling.ai.common import (
    format_forecast_snapshot_for_prompt,
    format_overall_dimension_context,
    format_plain_narrative_summaries,
)


def _is_out_of_range_costco(display_name: str, value: Optional[float]) -> bool:
    return (
        display_name == "Warehouse Club Distance"
        and value is not None
        and value >= 99
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
    car_wash_type: Optional[str] = None,
) -> str:
    if _is_out_of_range_costco(display_name, value):
        return """You are a car wash site analyst. Write one or two short plain-English sentences for this site.

Metric: Warehouse Club Distance
Site value: No warehouse club found within the configured search radius (encoded as out-of-range value).

Instructions:
- Do NOT treat the encoded value as literal miles.
- Clearly state that no nearby warehouse club was found in range.
- Explain what this means for local traffic in simple terms.
- Keep it natural, short, and conversational.

Reply with only the rationale."""

    val_str = f"{value:.1f}" if value is not None else "N/A"
    if value is not None and value == int(value):
        val_str = f"{int(value)}"
    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.1f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.1f}" if dist_max is not None else "N/A"
    direction_note = (
        "lower distance is better (closer anchor = more traffic)"
        if direction == "lower"
        else "higher count is better (more anchors = more traffic)"
    )
    return f"""You are a car wash site analyst. Write one or two short plain-English sentences for this site.

Car wash type: {car_wash_type or 'Unknown'}

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile vs. other sites: {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: {direction_note}
Scale range: {min_str} – {max_str}

Instructions:
- Explain value + percentile in simple words and mention quartile/category.
- If the car wash type is "Express Tunnel", do NOT say customers "shop while their car is washed". 
- Instead, describe it as a quick, convenient trip paired with their shopping (before or after).
- Reply with only the rationale."""


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
    return "\n".join(
        [
            "Forecast (use these facts; do not invent numbers):",
            snap,
            "",
            f"Car wash type: {car_wash_type or 'Unknown'}",
            "",
            sums,
            "",
            "Write one short Insight paragraph (2–4 sentences) on how shopping and food traffic "
            "around the site affect wash demand (warehouse clubs, big box, grocery, food). "
            "If the car wash type is 'Express Tunnel', never say characters 'shop while they wash'. "
            "Instead, emphasize trip-pairing convenience (washing before or after a shopping visit). "
            "Everyday words only—no quartile codes, percentiles, or metric field names.",
        ]
    )


def build_observation_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    car_wash_type: Optional[str] = None,
) -> str:
    context_block = format_overall_dimension_context(quantile_result, feature_narratives)

    return f"""
You are a car wash site analyst. Write a short, clear explanation in simple, everyday English that a layman can easily understand.

Refer to it as "this site" (not "your site").

Car wash operating model: {car_wash_type or 'Unknown'}

Facts from the forecast and retail check (use only these numbers and ideas; do not invent):
{context_block}

Instructions:
- Write 2–3 short sentences (not too long, not too short)
- Combine all points into a smooth, natural explanation (do not just list them)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why the retail ecosystem affects this site's car wash demand
- Use cause-and-effect reasoning (retail → demand)
- If the operating model is Express Tunnel, do NOT describe behavior as "drop-off" or "wash while shopping".
- Instead, describe it as a quick drive-through visit paired with a shopping trip (e.g., "stopping for a quick wash after getting groceries").

Strict Rules:
- No jargon or technical terms (no quartiles, percentiles, “model features”, or variable names)
- Do not name metric titles from the bullet text; paraphrase the ideas only
- Avoid formal phrases like "indicates", "suggests", "positions", "accumulation"
- Avoid long or complex sentences
- Do NOT repeat the same idea
- Make it sound human and day-to-day conversational, never robotic or AI-generated

Output Format (STRICT):
Observation: <2–3 sentence explanation combining the factors>
"""


def build_conclusion_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    car_wash_type: Optional[str] = None,
) -> str:
    context_block = format_overall_dimension_context(quantile_result, feature_narratives)

    return f"""
You are a car wash site analyst. Write a very short, natural wrap-up in simple, everyday English.

Refer to it as "this site" (not "your site").

Car wash operating model: {car_wash_type or 'Unknown'}

Facts from the forecast and retail check (use only these numbers and ideas; do not invent):
{context_block}

Instructions:
- Write 1–2 sentences max
- Explain the overall takeaway for car wash demand in a natural way
- Use simple cause-and-effect language
- Avoid jargon and long sentences; do not use quartile codes or metric field names
- Make it sound human and day-to-day conversational, never robotic or AI-generated
- If the operating model is Express Tunnel, avoid wording like "drop off the car".

Output Format (STRICT):
Conclusion: <1 short sentence stating expected wash band in a natural way>
"""
