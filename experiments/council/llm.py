"""
Council-local Azure OpenAI chat client — a self-contained copy of the insights layer's Azure path
(`app/pnl_analysis/insights/llm.py`), so the council imports nothing from `app/`. **Azure endpoint ONLY.**

Reads `AZURE_OPENAI_ENDPOINT / _API_KEY / _API_VERSION / _MODEL_DEPLOYMENT_NAME` from the environment
(loading the repo-root `.env` if python-dotenv is present — same file `app.core.common` loads). `complete()`
raises `LLMUnavailable` when Azure is not configured/reachable, so callers degrade to a safe no-op.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

import requests


def _load_env() -> None:
    """Best-effort: load the repo-root .env so AZURE_OPENAI_* resolve even outside the app process."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")   # <repo>/.env
        load_dotenv()                                               # cwd override, like app.core.common
    except Exception:
        pass


_load_env()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "")


class LLMUnavailable(RuntimeError):
    """Azure is not configured or not reachable."""


def azure_available() -> bool:
    return bool(ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT)


def _extract_text(payload: dict) -> str:
    try:
        choices = payload.get("choices") or []
        if choices:
            return ((choices[0].get("message") or {}).get("content") or "").strip()
    except Exception:
        pass
    return ""


def complete(messages: List[dict], *, json_mode: bool = False, temperature: float = 0.2,
             max_tokens: int = 800) -> str:
    """One Azure chat completion (mirrors the working insights curl). Raises LLMUnavailable on any failure
    so the committee can treat a down backend as a safe no-op."""
    if not azure_available():
        raise LLMUnavailable("Azure OpenAI is not configured (AZURE_OPENAI_* env vars missing).")
    url = (f"{ENDPOINT.rstrip('/')}/openai/deployments/{DEPLOYMENT}/chat/completions"
           f"?api-version={API_VERSION}")
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    rf = {"response_format": {"type": "json_object"}} if json_mode else {}

    def _post(body: dict) -> requests.Response:
        return requests.post(url, headers=headers, json=body, timeout=60)

    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **rf}
    try:
        r = _post(body)
        # reasoning-style deployments reject `temperature` / want `max_completion_tokens` — retry once
        if r.status_code == 400 and any(k in r.text for k in ("max_completion_tokens", "temperature", "unsupported")):
            r = _post({"messages": messages, "max_completion_tokens": max_tokens, **rf})
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise LLMUnavailable(f"Azure request failed: {exc}") from exc
    text = _extract_text(r.json())
    if not text:
        raise LLMUnavailable("Azure returned an empty completion.")
    return text


def parse_json_lax(text: Optional[str]) -> Any:
    """Best-effort JSON parse: strip code fences, `json.loads`; else grab the first balanced {..}/[..].
    Returns `{}` on failure — a safe no-op for the committee (no messages, belief unchanged)."""
    if not text:
        return {}
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except Exception:
                continue
    return {}
