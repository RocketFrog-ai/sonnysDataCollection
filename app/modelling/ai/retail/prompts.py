"""Prompt builders for retail narratives."""

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


def build_observation_prompt(feature_narratives: List[Dict[str, Any]]) -> str:
    feature_lines = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
        for n in feature_narratives
    ]
    return (
        "Retail feature summaries:\n"
        + "\n".join(feature_lines)
        + "\n\nWrite one Observation paragraph (2-4 sentences) with the overall retail ecosystem picture."
    )


def build_conclusion_prompt(feature_narratives: List[Dict[str, Any]]) -> str:
    feature_lines = [
        f"- {n.get('label', n.get('feature_key', ''))}: {n.get('summary') or 'N/A'}"
        for n in feature_narratives
    ]
    return (
        "Retail feature summaries:\n"
        + "\n".join(feature_lines)
        + "\n\nWrite a 1-2 sentence Conclusion on overall retail impact for car wash demand."
    )
