"""
Shared helpers for narrative generation (LLM response parsing, etc.).
Used by feature-wise narrative modules (weather, gas, retail, competition).
"""

from __future__ import annotations

from typing import Any, Dict


def extract_llm_text(response: Dict[str, Any]) -> str:
    """
    Extract generated text from the LLM server response dict.
    Handles multiple response shapes the server may return.
    """
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
