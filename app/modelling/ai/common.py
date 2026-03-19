"""Shared helpers for narrative generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


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
