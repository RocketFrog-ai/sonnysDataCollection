"""Prompt builders for retail narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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
    pred_q = quantile_result.get("predicted_wash_quantile")
    pred_label = quantile_result.get("predicted_wash_quantile_label") or f"Q{pred_q}"
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"
    lines = ["Retail proximity metrics for this site (value, percentile, category):"]
    for n in feature_narratives:
        val = n.get("value")
        unit = n.get("unit", "")
        if (n.get("metric_key") == "costco-distance" or n.get("label") == "Warehouse Club Distance") and isinstance(val, (int, float)) and val >= 99:
            val_str = "No warehouse club found within configured search radius"
        else:
            val_str = "None found (estimated by model)" if val is None else f"{val} {unit}"
        lines.append(
            f"- {n.get('label') or n.get('feature_key', '')}: "
            f"value={val_str}, percentile={n.get('percentile')}%, category={n.get('category')}"
        )
    lines.append(f"\nPredicted wash volume band: {pred_label} ({wash_range}).")
    lines.append(
        "\nWrite one short Insight paragraph (2-4 sentences) on retail anchor impact "
        "(warehouse club, big box, grocery, food). Keep wording simple."
    )
    return "\n".join(lines)


def build_observation_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
    car_wash_type: Optional[str] = None,
) -> str:
    pred_label = (
        quantile_result.get("predicted_wash_quantile_label")
        or quantile_result.get("label")
        or (f"Q{quantile_result.get('predicted_wash_quantile')}" if quantile_result.get("predicted_wash_quantile") is not None else None)
        or "N/A"
    )
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label")
    wash_band = pred_label if not wash_range else f"{pred_label} ({wash_range})"

    feature_summaries = "\n".join(
        f"- {f.get('label', f.get('feature_key', ''))}: {f.get('summary')}"
        for f in feature_narratives
        if f.get("summary")
    )

    return f"""
You are a car wash site analyst. Write a short, clear explanation in simple, everyday English that a layman can easily understand.

Refer to it as "this site" (not "your site").

Predicted wash band: {wash_band}

Per-feature summaries:
{feature_summaries}

Car wash operating model: {car_wash_type or 'Unknown'}

Instructions:
- Write 2–3 short sentences (not too long, not too short)
- Combine all points into a smooth, natural explanation (do not just list them)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why the retail ecosystem affects this site's car wash demand
- Use cause-and-effect reasoning (retail → demand)
- If the operating model is Express Tunnel, do not describe behavior as "drop-off"; describe quick drive-through visits.

Strict Rules:
- No jargon or technical terms
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
    pred_label = (
        quantile_result.get("predicted_wash_quantile_label")
        or quantile_result.get("label")
        or (f"Q{quantile_result.get('predicted_wash_quantile')}" if quantile_result.get("predicted_wash_quantile") is not None else None)
        or "N/A"
    )
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label")
    wash_band = pred_label if not wash_range else f"{pred_label} ({wash_range})"

    feature_summaries = "\n".join(
        f"- {f.get('label', f.get('feature_key', ''))}: {f.get('summary')}"
        for f in feature_narratives
        if f.get("summary")
    )

    return f"""
You are a car wash site analyst. Write a very short, natural wrap-up in simple, everyday English.

Refer to it as "this site" (not "your site").

Predicted wash band: {wash_band}

Per-feature summaries:
{feature_summaries}

Car wash operating model: {car_wash_type or 'Unknown'}

Instructions:
- Write 1–2 sentences max
- Explain the overall takeaway for car wash demand in a natural way
- Use simple cause-and-effect language
- Avoid jargon and long sentences
- Make it sound human and day-to-day conversational, never robotic or AI-generated
- If the operating model is Express Tunnel, avoid wording like "drop off the car".

Output Format (STRICT):
Conclusion: <1 short sentence stating expected wash band in a natural way>
"""
