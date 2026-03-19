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


def build_overall_prompt(feature_narratives: List[Dict[str, Any]]) -> str:
    feature_lines = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
        for n in feature_narratives
    ]
    return (
        "Competition feature summaries:\n"
        + "\n".join(feature_lines)
        + "\n\nWrite an Observation paragraph (2-4 sentences) with the total competition picture."
    )
