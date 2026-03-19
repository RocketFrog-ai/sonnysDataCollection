"""Prompt builders for weather narratives."""

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
    return f"""You are a car wash site analyst. Write one or two short sentences in plain, non-technical English. Refer to it as "this site" (not "your site"). Use the numbers given — do not use a fixed template.

Metric: {display_name} ({subtitle})
Site value: {val_str} {unit}
Percentile (vs. other car wash sites in the dataset): {pct_str}
Quartile: {quantile_label or 'N/A'} — Category: {cat_str}
Reference: For this metric, {direction_note}.
Scale range in dataset: {min_str} – {max_str}

Write a dynamic rationale that:
- Explains what the percentile means using everyday words (e.g. "better than about X% of sites" or "worse than about X% of sites" depending on the direction).
- Mentions quartile and category (Q1–Q4, Poor/Fair/Good/Strong).
- Notes whether higher or lower is better for this metric.
Percentile rule: interpret the percentile as "better than about X% of sites" on the good direction.
Avoid jargon (no 'distribution', 'correlation', 'quantile boundaries'). Reply with only the rationale."""


def build_insight_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    pred_q = quantile_result.get("predicted_wash_quantile")
    pred_label = quantile_result.get("predicted_wash_quantile_label") or f"Q{pred_q}"
    proba = quantile_result.get("quantile_probabilities") or {}
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"

    lines = ["Weather metrics for this site:"]
    for n in feature_narratives:
        name = n.get("label") or n.get("feature_key", "")
        val = n.get("value")
        unit = n.get("unit", "")
        cat = n.get("category") or n.get("wash_q")
        pct = n.get("percentile")
        lines.append(f"- {name}: value={val} {unit}, category={cat}, percentile={pct}%")
    lines.append(f"\nPredicted wash volume band: {pred_label} ({wash_range}). Probabilities: {proba}")
    lines.append(
        "\nWrite one short paragraph (Insight) about how this weather profile affects "
        "wash demand and operations. Highlight strengths and weaker points relative to "
        "other sites. Keep it 2-4 sentences and avoid technical jargon."
    )
    return "\n".join(lines)


def build_overall_prompt(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    feature_lines = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
        for n in feature_narratives
    ]
    pred_label = quantile_result.get("predicted_wash_quantile_label") or "N/A"
    wash_range = (quantile_result.get("predicted_wash_range") or {}).get("label") or "N/A"
    return f"""You are a car wash site analyst. Use weather feature summaries and wash band to write:

Predicted wash band: {pred_label} ({wash_range})

Per-feature summaries:
{chr(10).join(feature_lines)}

Respond with exactly:
Observation: [2-3 sentences]
Conclusion: [1-2 sentences]"""
