"""Prompt builders for weather narratives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.modelling.ai.common import (
    format_forecast_snapshot_for_prompt,
    format_overall_dimension_context,
    format_plain_narrative_summaries,
)


# def build_feature_summary_prompt(
#     *,
#     display_name: str,
#     subtitle: str,
#     value: Optional[float],
#     unit: str,
#     category: Optional[str],
#     percentile: Optional[float],
#     dist_min: Optional[float],
#     dist_max: Optional[float],
#     quantile_label: Optional[str],
#     direction: str,
# ) -> str:
#     val_str = (
#         f"{value:.0f}"
#         if value is not None and value == int(value)
#         else (f"{value}" if value is not None else "N/A")
#     )
#     pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
#     cat_str = category or "N/A"
#     min_str = f"{dist_min:.0f}" if dist_min is not None else "N/A"
#     max_str = f"{dist_max:.0f}" if dist_max is not None else "N/A"
#     direction_note = (
#         "higher values are better for wash demand"
#         if direction == "higher"
#         else "lower values are better for wash demand"
#     )
#     return f"""You are a car wash site analyst. Write one or two short sentences in plain, non-technical English. Refer to it as "this site" (not "your site"). Use the numbers given — do not use a fixed template.

# Metric: {display_name} ({subtitle})
# Site value: {val_str} {unit}
# Percentile (vs. other car wash sites in the dataset): {pct_str}
# Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
# Reference: For this metric, {direction_note}.
# Scale range in dataset: {min_str} – {max_str}

# Write a dynamic rationale that:
# - Explains what the percentile means using everyday words (e.g. "better than about X% of sites" or "worse than about X% of sites" depending on the direction).
# - Mentions quartile and category (Q1–Q4, Poor/Fair/Good/Strong).
# - Notes whether higher or lower is better for this metric.
# Percentile rule: interpret the percentile as "better than about X% of sites" on the good direction.
# Avoid jargon (no 'distribution', 'correlation', 'quantile boundaries'). Reply with only the rationale."""


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
    val_str = (
        f"{value:.0f}"
        if value is not None and value == int(value)
        else (f"{value}" if value is not None else "N/A")
    )
    pct_str = f"{percentile:.1f}%" if percentile is not None else "N/A"
    cat_str = category or "N/A"
    min_str = f"{dist_min:.0f}" if dist_min is not None else "N/A"
    max_str = f"{dist_max:.0f}" if dist_max is not None else "N/A"

    direction_note = (
        "higher values are better for wash demand"
        if direction == "higher"
        else "lower values are better for wash demand"
    )

    # Same percentile as feature quantile logic in weather/narratives.py (good-direction rank vs peers).
    if percentile is not None:
        perf_anchor = (
            f"Authoritative rank vs analyzed sites (use ONLY this number for “better than X%”; do not invent or round differently): "
            f"better than about {percentile:.1f}% of sites analyzed."
        )
    else:
        perf_anchor = (
            "No peer rank is available for this metric; do not claim a “better than X%” line. "
            "Describe the value and category only."
        )

    return f"""
You are a car wash site analyst. Write one or two short sentences in simple English.

Refer to it as "this site" (not "your site").

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Peer comparison (from model feature quantiles — same basis as the app): {pct_str}
Category: {cat_str}
Reference: {direction_note}
Range: {min_str} – {max_str}

{perf_anchor}

Metric Logic Hint:
- If display_name is "Dirt Creation Days": Rain creates road grime; demand spikes after the rain stops.
- If display_name is "Dirt Deposit Severity": Snow and salt create heavy dirt, driving seasonal volume.
- If display_name is "Comfortable Washing Days": Pleasant weather supports steady, spontaneous washing.
- If display_name is "Shutdown Risk Days": Freezing temperatures pose a risk to equipment and operations.

Instructions:
- FIRST sentence MUST state the peer comparison using ONLY the percentage above (when a rank is given). Do not use any other X%.
- Second sentence: why that matters for wash demand in professional (but plain) English.
- Use terms like "precipitation-driven demand", "operating window", or "dirt accumulation".
- Avoid simplistic phrasing like "dirty cars", "stay clean", or "happy customers".
- Avoid jargon like “distribution”, “correlation”, “quantile boundaries”.
- Do NOT repeat the full metric title twice.

Variability Rules (IMPORTANT):
- You may vary wording of the second sentence only; the X% in the first sentence must match the authoritative line above.
- Do NOT always start with "This site has"

Output Rules:
- Maximum 2 sentences
- No bullet points

Example shape (do not copy numbers):
- “This site is better than about 68% of sites analyzed … [why it matters].”
"""


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

    lines = [
        "Forecast (use these facts; do not invent numbers):",
        snap,
        "",
        sums,
        "",
        "Write one short paragraph (Insight) about how this weather picture affects wash demand and day-to-day operations. "
        "Highlight how specific climate patterns (like precipitation or temperature) drive or limit consistent volume. "
        "Use professional site-analysis language (e.g., 'climatic drivers', 'seasonal patterns', 'operating window'). "
        "Do NOT mention 'extra cleaning capacity', 'staffing', or 'maintenance'—focus only on demand. "
        "Keep it 2–4 sentences. No jargon: no quartile codes, percentiles, or metric field names.",
    ]
    return "\n".join(lines)


# def build_overall_prompt(
#     quantile_result: Dict[str, Any],
#     feature_narratives: List[Dict[str, Any]],
# ) -> str:
#     feature_lines = [
#         f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
#         for n in feature_narratives
#     ]
#     pred_label = quantile_result.get("predicted_wash_quantile_label") or "N/A"
#     wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"
#     return f"""You are a car wash site analyst. Use weather feature summaries and wash band to write:

# Predicted wash band: {pred_label} ({wash_range})

# Per-feature summaries:
# {chr(10).join(feature_lines)}

# Respond with exactly:
# Observation: [2-3 sentences]
# Conclusion: [1-2 sentences]"""


def build_overall_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    context_block = format_overall_dimension_context(quantile_result, feature_narratives)

    return f"""
You are a car wash site analyst. Write a short, clear explanation in simple, everyday English that a layman can easily understand.

Refer to it as "this site" (not "your site").

Facts from the forecast and weather check (use only these numbers and ideas; do not invent):
{context_block}

Instructions:
- Write 2–3 short sentences (not too long, not too short)
- Combine all points into a smooth, natural explanation (do not just list them)
- Use simple, conversational language (avoid formal or report-like tone)
- Clearly explain why this site gets this level of car wash demand
- Use cause-and-effect reasoning (weather → car wash demand)

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
- Make it sound human and professional (site analyst tone), not conversational like a child
- Avoid simplistic terms: do NOT use "this place", "the spot", "cars stay clean", "whenever they want"
- Use professional terms: "the site's environment", "climatic conditions", "precipitation patterns", "operating window"
- Do NOT mention "extra cleaning capacity", "staffing", or "internal maintenance"

Output Format (STRICT):
Observation: <2–3 sentence explanation combining the factors>
Conclusion: <1 short sentence stating expected wash band in a natural way>

Example style (do not copy):
Observation: This site benefits from a significant number of comfortable washing days and a lack of freezing temperatures, which provides a long and reliable operating window throughout the year. While limited precipitation reduces the frequency of dirt-driven demand spikes, the overall climatic stability supports steady and predictable customer volume.
Conclusion: Given these favorable weather conditions, the site can expect between 180 and 220 washes per day on average.
"""
