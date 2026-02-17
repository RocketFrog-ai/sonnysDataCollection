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


# ── Prompt construction ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior car wash site analysis consultant. You receive \
structured quantile-profiling data for a candidate car wash location \
and must produce a clear, business-ready rationale report.

RULES:
1. STRICT ADHERENCE TO DATA: Never contradict the provided numerical comparisons. \
   If the prompt says "Value 200 > Median 100", do NOT say it is below.
2. For every dimension with data (Weather, Traffic, Competition, etc.), \
   state the key input values, the IQR range they fall in, and which \
   performance tier they match.
3. Use comparative language based on the provided "Position" (e.g., above Q75, below Q25).
   Example: "3,350 sunshine hours places this site in the **High Performing** \
   range (Q25=3,153, Q75=3,416), well above the median."
4. After the dimension analysis, give an OVERALL VERDICT:
   - STRONG PROCEED   — majority High, no critical weaknesses
   - PROCEED WITH CAUTION — mixed, or one deal-breaker dimension
   - HIGH RISK / REJECT  — majority Low or critical weaknesses
5. Include the expected volume range from the predicted tier.
6. End with 3–5 actionable recommendations specific to this site.
7. Be concise but thorough. Use markdown formatting.
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
                    # Explicitly calculate position for the LLM
                    if val > q75:
                        pos_str = "ABOVE Q75 (High)"
                    elif val < q25:
                        pos_str = "BELOW Q25 (Low)"
                    else:
                        pos_str = "WITHIN IQR (Average)"
                    
                    parts.append(
                        f"- {feat} = {val:,.1f} → **{bf}** "
                        f"(Position: {pos_str} | Q25={q25:,.1f}, Med={q50:,.1f}, Q75={q75:,.1f})"
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

    prompt = f"""You are a car wash site analyst. Provide a 2-3 sentence executive summary for the **{dimension_name}** dimension of a site profiling analysis.

Data:
- Dimension prediction: {pred} (fit score: {fit:.1f}%)
- This dimension ranks #{rank} in discriminatory power (effect size: {effect:.3f})
- Overall site classification: {overall_category}
"""
    if dimension_result.get("feature_details"):
        prompt += "\nFeature breakdown:\n"
        for feat, fd in dimension_result["feature_details"].items():
            val = fd["value"]
            bf = fd["best_fit"]
            rng = fd["ranges"].get(bf, {})
            q25, q50, q75 = rng.get("q25"), rng.get("q50"), rng.get("q75")
            if q25 is not None:
                # Explicitly calculate position
                if val > q75:
                    pos_str = "ABOVE Q75"
                elif val < q25:
                    pos_str = "BELOW Q25"
                else:
                    pos_str = "WITHIN IQR"

                prompt += f"- {feat} = {val:,.1f} → {bf} (Position: {pos_str} | Q25={q25:,.1f}, Med={q50:,.1f}, Q75={q75:,.1f})\n"

    prompt += "\nProvide a concise, data-backed executive summary. Be specific with numbers."

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
