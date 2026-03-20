"""Prompt builders for competition narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


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

    return f"""You are a car wash site analyst. Write one or two short plain-English sentences about this site.

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile vs. other sites: {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: {direction_note}
Scale range: {min_str} – {max_str}

Explain value + percentile in everyday words and mention quartile/category. Reply with only the rationale."""


def build_insight_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    pred_q = quantile_result.get("predicted_wash_quantile")
    pred_label = quantile_result.get("predicted_wash_quantile_label") or f"Q{pred_q}"
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"
    lines = ["Competition metrics for this site (value, percentile, category):"]
    for n in feature_narratives:
        lines.append(
            f"- {n.get('label') or n.get('feature_key', '')}: "
            f"value={n.get('value')} {n.get('unit', '')}, "
            f"percentile={n.get('percentile')}%, category={n.get('category')}"
        )
    lines.append(f"\nPredicted wash volume band: {pred_label} ({wash_range}).")
    lines.append(
        "\nWrite one short Insight paragraph (2-4 sentences) explaining competition impact "
        "using distance, rating, and review count in plain English."
    )
    return "\n".join(lines)


def build_overall_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
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

Instructions:
- Write 2–3 short sentences (not too long, not too short)
- Combine all points into a smooth, natural explanation (do not just list them)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why the competition situation affects this site's car wash demand
- Use cause-and-effect reasoning (competition → demand)

Strict Rules:
- No jargon or technical terms
- Avoid formal phrases like "indicates", "suggests", "positions", "accumulation"
- Avoid long or complex sentences
- Do NOT repeat the same idea
- Do NOT sound like a report

Style Guidance:
- Write like you are explaining to a normal person
- Keep it natural, smooth, and easy to follow
- Use simple connectors like "because", "so", "which means"
- Make it sound human and day-to-day conversational, never robotic or AI-generated

Output Format (STRICT):
Observation: <2–3 sentence explanation combining the factors>
Conclusion: <1 short sentence stating expected wash band in a natural way>

Example style (do not copy):
Observation: This site faces a balanced level of nearby competition, so customers have options, but your combination of distance and local reputation still supports strong demand. That keeps people choosing a wash without needing to over-discount.
Conclusion: Because of this, the site can expect around 130–170 washes per day.
"""
