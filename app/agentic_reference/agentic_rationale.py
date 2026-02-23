"""
Agentic Rationale Pipeline (Local LLM)
========================================

Generates human-readable, business-oriented rationale for the
quantile profiling results using the project's local LLM.

The LLM receives structured profiling data (input values, IQR ranges,
dimension scores, strengths/weaknesses) and produces a dimension-by-
dimension analysis with verdict and actionable recommendations.

Falls back to deterministic rule-based rationale when the LLM is
unavailable.
"""

from __future__ import annotations

import json
from typing import Dict, List


# ── Human-readable labels for profiler features (no variable names in summaries) ──

PROFILER_FEATURE_LABELS: Dict[str, str] = {
    "count_of_target_5miles": "Target stores within 5 miles",
    "count_of_costco_5miles": "Costco stores within 5 miles",
    "count_of_walmart_5miles": "Walmart stores within 5 miles",
    "count_of_bestbuy_5miles": "Best Buy stores within 5 miles",
    "distance_from_nearest_target": "Distance to nearest Target",
    "distance_from_nearest_costco": "Distance to nearest Costco",
    "distance_from_nearest_walmart": "Distance to nearest Walmart",
    "distance_from_nearest_bestbuy": "Distance to nearest Best Buy",
    "Count of ChainXY VT - Building Supplies": "Building supplies stores (ChainXY)",
    "Count of ChainXY VT - Department Store": "Department stores (ChainXY)",
    "Count of ChainXY VT - Grocery": "Grocery stores (ChainXY)",
    "Count of ChainXY VT - Mass Merchant": "Mass merchant stores (ChainXY)",
    "Count of ChainXY VT - Real Estate Model": "Real estate model stores (ChainXY)",
    "Sum ChainXY": "Total ChainXY retail count",
    "competitors_count": "Car wash competitors nearby",
    "competitor_1_distance_miles": "Distance to nearest competitor",
    "competitor_1_google_user_rating_count": "Nearest competitor review count",
    "total_sunshine_hours": "Total sunshine hours",
    "days_pleasant_temp": "Pleasant temperature days",
    "total_precipitation_mm": "Total precipitation",
    "rainy_days": "Rainy days",
    "total_snowfall_cm": "Total snowfall",
    "snowy_days": "Snowy days",
    "days_below_freezing": "Days below freezing",
    "avg_daily_max_windspeed_ms": "Average daily max windspeed",
    "tunnel_length (in ft.)": "Tunnel length (ft)",
    "total_weekly_operational_hours": "Weekly operational hours",
}


def _profiler_feature_label(feat: str) -> str:
    """Human-readable label for profiler feature; avoid variable names in summaries."""
    return PROFILER_FEATURE_LABELS.get(feat, feat.replace("_", " ").title())


# ── Prompt construction ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior car wash site expert. You receive data about a candidate site and must explain its classification to a stakeholder.

GOAL:
Provide a clear, factual explanation of why the site fits its specific performance profile.

RULES:
1. NO JARGON: Do not use "IQR", "Fit Score", "Effect Size". Use "Typical Range" or "Profile Match".
2. EXPLAIN THE MATCH:
   - If a site is "Average", explain that its metrics align with our standard portfolio.
   - Example: "Traffic (15k) is consistent with our average-performing sites."
3. CONTEXTUALIZE:
   - Compare values to the *tier they match*.
   - If a value is low, say "Matches the low-volume profile."
4. VERDICT:
   - Conclude with a clear summary of the site's strengths and weaknesses relative to its tier.
"""


def _build_data_prompt(
    location_features: Dict[str, float],
    overall_prediction: Dict,
    dimension_results: Dict[str, Dict],
    dimension_strength: Dict[str, Dict],
    vote_result: Dict,
    strengths: List[str],
    weaknesses: List[str],
    neutrals: List[str],
) -> str:
    """Assemble the structured data the LLM will reason over."""
    parts: List[str] = []

    # Overall prediction
    p = overall_prediction
    ev = p["expected_volume"]
    parts.append("## Overall Quantile Prediction")
    parts.append(f"- Tier: **{p['predicted_category']}** (fit {p['fit_score']}%)")
    parts.append(
        f"- Expected volume: {ev['conservative']:,}–{ev['optimistic']:,} "
        f"cars/year (likely {ev['likely']:,})"
    )

    # Dimension results
    parts.append("\n## Dimension-Specific Results")
    for dim, res in dimension_results.items():
        pred = res.get("predicted", "N/A")
        fit = res.get("fit_score", 0)
        parts.append(f"\n### {dim} — {pred} (fit {fit:.1f}%)")
        if res.get("feature_details"):
            for feat, fd in res["feature_details"].items():
                val = fd["value"]
                bf = fd["best_fit"]
                rng = fd["ranges"].get(bf, {})
                q25, q50, q75 = rng.get("q25"), rng.get("q50"), rng.get("q75")
                if q25 is not None:
                    # Context relative to the MATCHED profile (Objective)
                    if val > q75:
                        pos_str = f"Higher than the typical range for {bf} sites"
                    elif val < q25:
                        pos_str = f"Lower than the typical range for {bf} sites"
                    else:
                        pos_str = f"Typical for {bf} sites"
                    
                    parts.append(
                        f"- {feat} = {val:,.1f} → Matches **{bf}** profile. "
                        f"(Context: {pos_str}. Typical {bf} range is {q25:,.1f}–{q75:,.1f})"
                    )

    # Dimension strength ranking
    parts.append("\n## Dimension Discriminatory Power (historical)")
    ranked = sorted(
        dimension_strength.items(),
        key=lambda x: x[1]["average_effect_size"],
        reverse=True,
    )
    for dim, info in ranked:
        parts.append(f"- {dim}: effect_size={info['average_effect_size']:.3f} (rank #{info['rank']})")

    # Summary
    parts.append(f"\n## Verdict Summary")
    parts.append(f"- Majority vote: **{vote_result['category']}** ({vote_result['agreement']}/{vote_result['total_dimensions']} dimensions)")
    parts.append(f"- Strengths: {', '.join(strengths) if strengths else 'None'}")
    parts.append(f"- Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}")
    parts.append(f"- Neutral: {', '.join(neutrals) if neutrals else 'None'}")

    parts.append(
        "\nProduce the full rationale report. Reference exact input values "
        "and IQR ranges. Give overall verdict and recommendations."
    )
    return "\n".join(parts)


# ── LLM-powered rationale ───────────────────────────────────────

def generate_rationale(
    location_features: Dict[str, float],
    overall_prediction: Dict,
    dimension_results: Dict[str, Dict],
    dimension_strength: Dict[str, Dict],
    vote_result: Dict,
    strengths: List[str],
    weaknesses: List[str],
    neutrals: List[str],
) -> str:
    """
    Call the local LLM to produce a narrative rationale report.
    Falls back to rule-based output on failure.
    """
    data_prompt = _build_data_prompt(
        location_features,
        overall_prediction,
        dimension_results,
        dimension_strength,
        vote_result,
        strengths,
        weaknesses,
        neutrals,
    )
    full_prompt = SYSTEM_PROMPT + "\n\n" + data_prompt

    try:
        from app.utils.llm import local_llm as llm

        response = llm.get_llm_response(
            full_prompt,
            reasoning_effort="low",
            temperature=0.3,
        )
        text = (response or {}).get("generated_text", "").strip()
        if text:
            return text
    except Exception as e:
        print(f"⚠ LLM rationale failed ({e}), using rule-based fallback")

    # Fallback
    return _generate_fallback_rationale(
        location_features,
        overall_prediction,
        dimension_results,
        dimension_strength,
        vote_result,
        strengths,
        weaknesses,
        neutrals,
    )


# ── Dimension-specific rationale (for summary endpoints) ────────

def generate_dimension_rationale(
    dimension_name: str,
    dimension_result: Dict,
    dimension_strength_info: Dict,
    overall_category: str,
) -> str:
    """
    Generate a focused rationale for a single dimension.
    Used by the per-modal summary endpoints.
    """
    pred = dimension_result.get("predicted", "N/A")
    fit = dimension_result.get("fit_score", 0)
    effect = dimension_strength_info.get("average_effect_size", 0)
    rank = dimension_strength_info.get("rank", "N/A")

    prompt = f"""You are determining the rationale for the **{dimension_name}** dimension.

Facts:
- Verdict: {pred}
- Confidence: {fit:.1f}% match
- Importance: Rank #{rank} (Impact: {effect:.2f})
- Overall Site: {overall_category}

Write a 2-sentence summary:
1. Explain why this dimension fits the {pred} profile.
2. Highlight any key features that drove this result.

Use only human-readable language. Do not use variable names, field names, or feature keys (e.g. do not write count_of_target_5miles or count_of_costco_5miles). Refer to metrics in plain English (e.g. "Target stores within 5 miles", "Costco stores within 5 miles").
"""
    if dimension_result.get("feature_details"):
        prompt += "\nFeature breakdown (use the labels below in your summary, not variable names):\n"
        for feat, fd in dimension_result["feature_details"].items():
            val = fd["value"]
            bf = fd["best_fit"]
            rng = fd["ranges"].get(bf, {})
            q25, q50, q75 = rng.get("q25"), rng.get("q50"), rng.get("q75")
            label = _profiler_feature_label(feat)
            if q25 is not None:
                if val > q75:
                    comparison = "Above typical range"
                elif val < q25:
                    comparison = "Below typical range"
                else:
                    comparison = "Within typical range"

                prompt += f"- {label}: {val:,.1f} ({comparison} for {bf} sites. Typical {bf} is {q25:,.1f}–{q75:,.1f})\n"

    prompt += "\nProvide a concise, data-backed executive summary in plain English. Be specific with numbers. Do not mention any variable or field names."

    try:
        from app.utils.llm import local_llm as llm

        response = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.2)
        text = (response or {}).get("generated_text", "").strip()
        if text:
            return text
    except Exception as e:
        print(f"⚠ LLM dimension rationale failed ({e})")

    # Fallback
    return _fallback_dimension_summary(dimension_name, dimension_result, dimension_strength_info)


def _fallback_dimension_summary(
    dimension_name: str,
    dimension_result: Dict,
    dimension_strength_info: Dict,
) -> str:
    """Deterministic single-dimension summary."""
    pred = dimension_result.get("predicted", "N/A")
    fit = dimension_result.get("fit_score", 0)
    n = dimension_result.get("features_scored", 0)
    rank = dimension_strength_info.get("rank", "?")

    icon = "✓" if "High" in pred else ("✗" if "Low" in pred else "○")
    lines = [
        f"{icon} **{dimension_name}**: {pred} (fit {fit:.1f}%, {n} features scored)",
        f"Discriminatory power: rank #{rank}.",
    ]

    if dimension_result.get("feature_details"):
        key_features = []
        for feat, fd in dimension_result["feature_details"].items():
            val = fd["value"]
            bf = fd["best_fit"]
            rng = fd["ranges"].get(bf, {})
            q25, q75 = rng.get("q25"), rng.get("q75")
            if q25 is not None:
                pos = "within" if q25 <= val <= q75 else ("below" if val < q25 else "above")
                key_features.append(f"{feat}={val:,.1f} ({pos} {bf} IQR [{q25:,.1f}–{q75:,.1f}])")
        if key_features:
            lines.append("Key features: " + "; ".join(key_features[:5]))

    return " ".join(lines)


# ── Fallback: full rule-based rationale ──────────────────────────

def _generate_fallback_rationale(
    location_features: Dict[str, float],
    overall_prediction: Dict,
    dimension_results: Dict[str, Dict],
    dimension_strength: Dict[str, Dict],
    vote_result: Dict,
    strengths: List[str],
    weaknesses: List[str],
    neutrals: List[str],
) -> str:
    """Full deterministic rationale when LLM is not available."""
    lines: List[str] = []
    lines.append("# 📊 Car Wash Site Profiling Report\n")

    # Overall
    p = overall_prediction
    ev = p["expected_volume"]
    lines.append("## Overall Classification\n")
    lines.append(f"**{p['predicted_category']}** (fit score: {p['fit_score']}%)\n")
    lines.append(
        f"Expected annual volume: **{ev['conservative']:,}** (conservative) — "
        f"**{ev['likely']:,}** (likely) — **{ev['optimistic']:,}** (optimistic)\n"
    )

    # Dimension breakdown
    lines.append("## Dimension-by-Dimension Analysis\n")
    for dim, res in dimension_results.items():
        pred = res.get("predicted", "N/A")
        fit = res.get("fit_score", 0)
        if pred == "Insufficient Data":
            continue

        icon = "✓" if "High" in pred else ("✗" if "Low" in pred else "○")
        lines.append(f"### {icon} {dim}: **{pred}** ({fit:.1f}%)\n")

        if res.get("feature_details"):
            for feat, fd in res["feature_details"].items():
                val = fd["value"]
                bf = fd["best_fit"]
                rng = fd["ranges"].get(bf, {})
                q25 = rng.get("q25")
                q50 = rng.get("q50")
                q75 = rng.get("q75")
                if q25 is not None:
                    pos = (
                        "within the IQR"
                        if q25 <= val <= q75
                        else ("below the IQR" if val < q25 else "above the IQR")
                    )
                    lines.append(
                        f"- **{feat}** = {val:,.1f} — {pos} of **{bf}** sites "
                        f"(Q25={q25:,.1f}, Median={q50:,.1f}, Q75={q75:,.1f})"
                    )
            lines.append("")

    # Verdict
    lines.append("## Overall Verdict\n")
    n_high = len(strengths)
    n_low = len(weaknesses)

    if n_high > n_low and n_low <= 1:
        verdict = "🟢 **STRONG PROCEED** — Majority of dimensions indicate High Performing"
    elif n_low > n_high:
        verdict = "🔴 **HIGH RISK** — Majority of dimensions indicate Low Performing"
    else:
        verdict = "🟡 **PROCEED WITH CAUTION** — Mixed signals across dimensions"

    lines.append(verdict)
    lines.append(f"\n- **{vote_result['agreement']}/{vote_result['total_dimensions']}** dimensions agree on **{vote_result['category']}**")
    lines.append(f"- Strengths: {', '.join(strengths) if strengths else 'None'}")
    lines.append(f"- Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}")

    # Recommendations
    lines.append("\n## Recommendations\n")
    if weaknesses:
        for w in weaknesses:
            lines.append(f"- ⚠ **{w}** scored Low — investigate mitigation strategies")
    if strengths:
        lines.append(f"- ✓ Leverage strong dimensions: {', '.join(strengths)}")
    lines.append("- Review competitive landscape on the ground before committing")

    return "\n".join(lines)
