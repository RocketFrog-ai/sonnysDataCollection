"""
LLM client for the Key-Insights pipeline — a thin backend toggle.

`INSIGHTS_LLM_BACKEND` (env, default "azure") selects:
  • azure  — the verified Azure GPT-4o deployment, via a thin requests POST that mirrors the working
             curl (endpoint / deployment / api-version / api-key from app.utils.common).
  • local  — the internal LLM server, via app.utils.llm.local_llm.get_llm_text.

`complete(messages, backend=...)` returns the assistant text. It raises `LLMUnavailable` when the
chosen backend isn't configured/reachable, so the graph node can fall back to rule-based insights and
the panels always render. Secrets are read from env only — never hardcoded.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import List, Optional
from urllib.parse import urlparse

import requests

from app.utils import common as calib
from app.site_analysis.modelling.ai.common import extract_llm_text

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = os.getenv("INSIGHTS_LLM_BACKEND", "azure").strip().lower()


class LLMUnavailable(RuntimeError):
    """Raised when the selected LLM backend is not configured or not reachable."""


def resolve_backend(backend: Optional[str]) -> str:
    return (backend or os.getenv("INSIGHTS_LLM_BACKEND", "azure")).strip().lower()


# ─────────────────────────── availability guards ───────────────────────────
def azure_available() -> bool:
    return bool(calib.AZURE_OPENAI_ENDPOINT and calib.AZURE_OPENAI_API_KEY
                and calib.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME and calib.AZURE_OPENAI_API_VERSION)


def local_reachable(timeout: float = 2.0) -> bool:
    """Fast socket pre-check on the internal LLM host:port so a down endpoint never hangs the UI
    on the client's long retry timeout. Mirrors site_analysis_page._llm_reachable."""
    url = calib.LLM_REALTIME_URL or ""
    if not url:
        return False
    p = urlparse(url)
    host, port = p.hostname, (p.port or (443 if p.scheme == "https" else 80))
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def insights_llm_ready(backend: Optional[str] = None) -> bool:
    return azure_available() if resolve_backend(backend) == "azure" else local_reachable()


# ─────────────────────────── completion ───────────────────────────
def _azure_complete(messages: List[dict], max_tokens: int, temperature: float, json_mode: bool) -> str:
    if not azure_available():
        raise LLMUnavailable("Azure OpenAI is not configured (AZURE_OPENAI_* env vars missing).")
    url = (f"{calib.AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/"
           f"{calib.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME}/chat/completions"
           f"?api-version={calib.AZURE_OPENAI_API_VERSION}")
    headers = {"api-key": calib.AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    rf = {"response_format": {"type": "json_object"}} if json_mode else {}

    def _post(body: dict) -> requests.Response:
        return requests.post(url, headers=headers, json=body, timeout=60)

    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **rf}
    try:
        r = _post(body)
        # reasoning deployments reject `temperature` / want `max_completion_tokens` — retry once.
        if r.status_code == 400 and any(k in r.text for k in ("max_completion_tokens", "temperature", "unsupported")):
            r = _post({"messages": messages, "max_completion_tokens": max_tokens, **rf})
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise LLMUnavailable(f"Azure request failed: {exc}") from exc
    text = extract_llm_text(r.json())
    if not text:
        raise LLMUnavailable("Azure returned an empty completion.")
    return text


def _local_complete(messages: List[dict], max_tokens: int, temperature: float, json_mode: bool) -> str:
    if not local_reachable():
        raise LLMUnavailable("Local LLM server is not reachable (LLM_REALTIME_URL).")
    from app.utils.llm.local_llm import get_llm_text
    prompt = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
    if json_mode:
        prompt += "\n\nReturn ONLY the JSON object, with no prose or code fences."
    text = get_llm_text(prompt, temperature=temperature, max_new_tokens=min(max_tokens, 2048))
    if not text:
        raise LLMUnavailable("Local LLM returned an empty completion.")
    return text


def complete(messages: List[dict], *, backend: Optional[str] = None, max_tokens: int = 1500,
             temperature: float = 0.3, json_mode: bool = False) -> str:
    """Run a chat completion against the selected backend. Raises LLMUnavailable on failure."""
    be = resolve_backend(backend)
    if be == "local":
        return _local_complete(messages, max_tokens, temperature, json_mode)
    return _azure_complete(messages, max_tokens, temperature, json_mode)


def complete_cascade(messages: List[dict], *, backend: Optional[str] = None, max_tokens: int = 1500,
                     temperature: float = 0.3, json_mode: bool = False):
    """Try the chosen backend, then fall back to the OTHER one (azure<->local). No fixed-string fallback:
    if neither answers, raise LLMUnavailable. Returns (text, backend_used)."""
    primary = resolve_backend(backend)
    order = [primary, "local" if primary == "azure" else "azure"]
    errors = []
    for be in order:
        try:
            return complete(messages, backend=be, max_tokens=max_tokens,
                            temperature=temperature, json_mode=json_mode), be
        except LLMUnavailable as exc:
            errors.append(f"{be}: {exc}")
    raise LLMUnavailable("; ".join(errors))
