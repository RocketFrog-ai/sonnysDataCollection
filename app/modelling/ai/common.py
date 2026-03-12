"""
Shared helpers for narrative generation (LLM response parsing, etc.).
Used by feature-wise narrative modules (weather, gas, retail, competition).
"""

from __future__ import annotations

from typing import Any, Dict


def extract_llm_text(response: Dict[str, Any]) -> str:
    """Extract generated text from local LLM response dict."""
    if not response:
        return ""
    text = response.get("generated_text") or response.get("content") or response.get("text")
    if text:
        return (text if isinstance(text, str) else "").strip()
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message", {})
        if isinstance(msg, dict):
            return (msg.get("content") or "").strip()
    return ""
