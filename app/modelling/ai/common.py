"""Shared helpers for narrative generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def extract_llm_text(response: Dict[str, Any]) -> str:
    """Extract generated text from varied LLM response shapes."""
    if not response:
        return ""
    # Direct top-level text fields
    for field in ("generated_text", "content", "text", "output"):
        val = response.get(field)
        if val and isinstance(val, str):
            return val.strip()
    # OpenAI-style choices array
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content") or ""
            if content:
                return content.strip()
        # text-completion style
        text = choices[0].get("text") or ""
        if text:
            return text.strip()
    return ""


def get_llm_text(
    prompt: str,
    *,
    reasoning_effort: str = "low",
    temperature: float = 0.3,
    max_new_tokens: int = 256,
) -> Optional[str]:
    """Run local LLM and return stripped text if present."""
    from app.utils.llm import local_llm as llm

    raw = llm.get_llm_response(
        prompt,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    text = extract_llm_text(raw)
    return text.strip() if text else None


def load_input_payload(
    default_payload: Dict[str, Any],
    argv: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load test payload from --input JSON file or use defaults."""
    parser = argparse.ArgumentParser(description="Narrative module runner")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to JSON payload. If omitted, built-in sample is used.",
    )
    args = parser.parse_args(argv)
    if not args.input:
        return default_payload
    with args.input.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object.")
    return data


def format_plain_narrative_summaries(
    feature_narratives: List[Dict[str, Any]],
    *,
    heading: Optional[str] = None,
) -> str:
    """
    Join per-factor LLM summaries without metric titles or feature keys.
    Used in overall prompts so the model paraphrases ideas, not variable names.
    """
    summaries = [str(s).strip() for f in feature_narratives if (s := f.get("summary"))]
    if not summaries:
        return ""
    default_heading = (
        "What the analysis already says (rewrite in your own words; "
        "do not repeat metric titles, codes like Q1/Q4, or internal names):"
    )
    h = heading or default_heading
    bullet = "\n".join(f"- {s}" for s in summaries)
    return f"{h}\n{bullet}"


def format_forecast_snapshot_for_prompt(quantile_result: Dict[str, Any]) -> str:
    """
    Compact, plain-English forecast lines from the same quantile output as the app.
    """
    tier = quantile_result.get("predicted_wash_tier")
    pred_q = quantile_result.get("predicted_wash_quantile")
    pred_label = quantile_result.get("predicted_wash_quantile_label") or (
        f"Q{pred_q}" if pred_q is not None else None
    )
    wash_range = quantile_result.get("predicted_wash_range") or {}
    label = wash_range.get("label")
    proba = quantile_result.get("quantile_probabilities") or {}

    lines: List[str] = []
    if pred_label and label:
        lines.append(f"Volume band: {pred_label} ({label}).")
    elif pred_label:
        lines.append(f"Volume band: {pred_label}.")
    elif label:
        lines.append(f"Expected yearly wash volume: {label}.")
    if tier:
        lines.append(f"Outlook label: {tier}.")
    if pred_q is not None and proba:
        pt = proba.get(pred_q)
        if pt is None:
            pt = proba.get(str(pred_q))
        if pt is not None:
            try:
                pct = float(pt)
                if pct <= 1.0:
                    pct *= 100.0
                lines.append(f"About {pct:.0f}% of the estimate lines up with this band.")
            except (TypeError, ValueError):
                pass
    return "\n".join(lines)


def format_overall_dimension_context(
    quantile_result: Dict[str, Any],
    feature_narratives: List[Dict[str, Any]],
) -> str:
    """Forecast snapshot + plain summary bullets for dimension-level overall prompts."""
    snap = format_forecast_snapshot_for_prompt(quantile_result)
    sums = format_plain_narrative_summaries(feature_narratives)
    if not sums:
        sums = (
            "(No short factor summaries were returned—use only the forecast lines above; "
            "do not invent numbers.)"
        )
    return f"{snap}\n\n{sums}"


def get_car_wash_type_label(feature_values: Optional[Dict[str, Any]]) -> Optional[str]:
    """Derive display label for car wash type (Express Tunnel, Mobile, Hand Wash)."""
    if not feature_values:
        return None
    mapping = {
        1: "Express Tunnel",
        2: "Mobile",
        3: "Hand Wash",
    }
    encoded = feature_values.get("carwash_type_encoded")
    if encoded in mapping:
        return mapping[encoded]
    text = feature_values.get("type_of_car_wash")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None
