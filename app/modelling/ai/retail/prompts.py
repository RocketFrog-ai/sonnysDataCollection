"""Prompt builders for retail narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modelling.ai.common import (
    format_forecast_snapshot_for_prompt,
    format_overall_dimension_context,
    format_plain_narrative_summaries,
)


_NEAR_MILE = 1.0   
_MID_MILE  = 3.0   
_COUNT_CAP = 5     


def _is_out_of_range_costco(display_name: str, value: Optional[float]) -> bool:
    return (
        display_name == "Warehouse Club Distance"
        and value is not None
        and value >= 99
    )


def _distance_band(metric_key: str, value: Optional[float]) -> Optional[str]:
    """
    Deterministic plain-English band for an anchor distance metric.
    Returns an authoritative context line the LLM MUST use as its first sentence basis.
    """
    distance_metrics = {"costco-distance", "walmart-distance", "target-distance"}
    if metric_key not in distance_metrics or value is None or value >= 99:
        return None
    if value <= _NEAR_MILE:
        return (
            f"Authoritative context (MUST use in first sentence): "
            f"This site is within {value:.2g} mile{'s' if value != 1 else ''} of this anchor — "
            f"that is very close and strong for pulling in traffic."
        )
    if value <= _MID_MILE:
        return (
            f"Authoritative context (MUST use in first sentence): "
            f"This site is {value:.1f} miles from this anchor — "
            f"a reasonable distance that still provides some traffic benefit, but not as strong as under 1 mile."
        )
    return (
        f"Authoritative context (MUST use in first sentence): "
        f"This site is {value:.1f} miles from this anchor — "
        f"far enough that direct foot-traffic pull from this location is limited."
    )


def _count_band(metric_key: str, value: Optional[float]) -> Optional[str]:
    """
    Deterministic plain-English band for a count metric.
    Caps display at _COUNT_CAP and returns an authoritative context line.
    """
    count_metrics = {"grocery-count", "food-joint-count"}
    if metric_key not in count_metrics or value is None:
        return None
    raw = int(value)
    capped = min(raw, _COUNT_CAP)
    if capped == 0:
        label = "none — no nearby activity boost from this category"
    elif capped <= 2:
        label = f"{capped} — limited presence nearby"
    else:
        label = f"{capped} — healthy concentration nearby"
    cap_note = f" (capped at {_COUNT_CAP} for narrative; actual fetched: {raw})" if raw > _COUNT_CAP else ""
    return (
        f"Authoritative context (MUST use in first sentence): "
        f"Count shown to user: {capped}{cap_note} — {label}."
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
    metric_key: str = "",
    car_wash_type: Optional[str] = None,
) -> str:
    if _is_out_of_range_costco(display_name, value):
        return """You are a car wash site analyst. Write one or two short plain-English sentences for this site.

Metric: Warehouse Club Distance
Site value: No warehouse club found within the configured search radius.

Instructions:
- Clearly state that no nearby warehouse club was found in range.
- Explain in simple terms what this means for local traffic.
- Keep it natural, short, and conversational.

Reply with only the rationale."""

    # Authoritative context line (distance band or count band)
    auth_line = _distance_band(metric_key, value) or _count_band(metric_key, value) or ""

    # Display value — for count metrics cap at _COUNT_CAP
    if metric_key in {"grocery-count", "food-joint-count"} and value is not None:
        display_val = min(int(value), _COUNT_CAP)
        val_str = str(display_val)
    elif value is not None and float(value) == int(float(value)):
        val_str = str(int(float(value)))
    else:
        val_str = f"{value:.1f}" if value is not None else "N/A"

    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.1f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.1f}" if dist_max is not None else "N/A"
    direction_note = (
        "closer (lower distance) is better — more traffic pull"
        if direction == "lower"
        else "more (higher count) is better — more traffic activity nearby"
    )

    auth_block = f"\n{auth_line}\n" if auth_line else ""

    return f"""You are a car wash site analyst. Write one or two short plain-English sentences for this site.

Car wash type: {car_wash_type or 'Unknown'}

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile vs. other sites: {pct_str}
Category: {cat_str}
Reference: {direction_note}
Scale range: {min_str} – {max_str}
{auth_block}
Instructions:
- Your FIRST sentence MUST reflect the authoritative context above (distance band or count band). Do not invent a different distance or count.
- Second sentence: explain what it means for wash demand in everyday words.
- If the car wash type is "Express Tunnel", describe quick trip-pairing (before/after a store visit) — not "shop while car is washed".
- No jargon, no quartile codes, no metric field names.

Reply with only the two sentences."""


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
            "Write one short Insight paragraph (2 sentences) on how shopping and food traffic "
            "around the site affect wash demand (warehouse clubs, big box, grocery, food). "
            "If the car wash type is 'Express Tunnel', never say characters 'shop while they wash'. "
            "Instead, emphasize trip-pairing convenience (washing before or after a shopping visit). "
            "Everyday words only—no quartile codes, percentiles, or metric field names.",
        ]
    )


# def build_observation_prompt(
#     quantile_result: Dict[str, Any],
#     feature_narratives: List[Dict[str, Any]],
#     car_wash_type: Optional[str] = None,
# ) -> str:
#     context_block = format_overall_dimension_context(quantile_result, feature_narratives)

#     return f"""
# You are a car wash site analyst. Write a short, clear explanation in simple, everyday English that a layman can easily understand.

# Refer to it as "this site" (not "your site").

# Car wash operating model: {car_wash_type or 'Unknown'}

# Facts from the forecast and retail check (use only these numbers and ideas; do not invent):
# {context_block}

# Instructions:
# - Write 2 short sentences (strictly 2; not too long, not too short)
# - Combine all points into a smooth, natural explanation (do not just list them)
# - Use simple, conversational language (avoid formal or report-like tone)
# - Clearly explain why the retail ecosystem affects this site's car wash demand
# - Use cause-and-effect reasoning (retail → demand)
# - If the operating model is Express Tunnel, do NOT describe behavior as "drop-off" or "wash while shopping".
# - Instead, describe it as a quick drive-through visit paired with a shopping trip (e.g., "stopping for a quick wash after getting groceries").

# Strict Rules:
# - No jargon or technical terms (no quartiles, percentiles, “model features”, or variable names)
# - Do not name metric titles from the bullet text; paraphrase the ideas only
# - Avoid formal phrases like "indicates", "suggests", "positions", "accumulation"
# - Avoid long or complex sentences
# - Do NOT repeat the same idea
# - Make it sound human and day-to-day conversational, never robotic or AI-generated

# Output Format (STRICT):
# Observation: <2 sentence explanation combining the factors>
# """

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
- Write 1 sentence for Pro (what in the retail environment supports or increases car wash demand)
- Write 1 sentence for Con (what in the retail environment limits or reduces car wash demand)
- Each sentence should be concise but meaningful (not too long, not too short)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why the retail ecosystem affects this site's car wash demand
- Use cause-and-effect reasoning (retail → demand)
- If the operating model is Express Tunnel, do NOT describe behavior as "drop-off" or "wash while shopping"
- Instead, describe it as a quick drive-through visit paired with a shopping trip (e.g., "stopping for a quick wash after getting groceries")

Strict Rules:
- No jargon or technical terms (no quartiles, percentiles, “model features”, or variable names)
- Do not name metric titles from the bullet text; paraphrase the ideas only
- Avoid formal phrases like "indicates", "suggests", "positions", "accumulation"
- Avoid long or complex sentences
- Do NOT repeat the same idea
- Do NOT sound like a report
- Make it sound human and natural, never robotic or AI-generated

Style Guidance:
- Write like you are explaining to a normal person
- Keep it natural, smooth, and easy to follow
- Use simple connectors like "because", "so", "which means"
- Make it sound human and professional (site analyst tone), not overly casual
- Avoid simplistic terms: do NOT use "this place", "the spot", "cars stay clean", "whenever they want"
- Use professional terms: "the site's environment", "customer flow patterns", "retail activity", "visit behavior"

Output Format (STRICT):
Pro: <1 sentence explaining the positive retail drivers>
Con: <1 sentence explaining the limiting retail factors>
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
