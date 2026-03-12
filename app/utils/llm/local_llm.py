import logging
import time
from typing import Any, Dict

import requests

from app.utils import common as calib

logger = logging.getLogger(__name__)

_REALTIME_URL: str = calib.LLM_REALTIME_URL
_BATCH_URL: str    = calib.LLM_BATCH_URL
_API_KEY: str      = calib.LOCAL_LLM_API_KEY

_MAX_TOKENS_HARD_LIMIT = 8192

MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = [2, 5, 10]

_TOKEN_BUDGET: Dict[str, int] = {
    "summary":     256,
    "insight":     512,
    "observation": 512,
    "conclusion":  256,
    "default":     512,
}


def _headers() -> Dict[str, str]:
    return {"x-api-key": _API_KEY, "Content-Type": "application/json"}


def _post(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """POST with retry on 429 / 5xx. Raises on 413 or after exhausted retries."""
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
            if resp.status_code == 413:
                raise ValueError(
                    f"LLM context too long (413). Reduce prompt length or max_new_tokens. "
                    f"max_new_tokens={payload.get('max_new_tokens')}"
                )
            if resp.status_code == 429:
                wait = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                logger.warning("LLM server busy (429); retrying in %ss (attempt %d/%d)", wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                last_exc = RuntimeError(f"LLM server busy (429) after {MAX_RETRIES} attempts")
                continue
            if resp.status_code >= 500:
                wait = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                logger.warning("LLM server error (%d); retrying in %ss (attempt %d/%d)", resp.status_code, wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                last_exc = RuntimeError(f"LLM server returned {resp.status_code}")
                continue
            resp.raise_for_status()
            return resp.json()
        except (ValueError, requests.exceptions.Timeout) as exc:
            raise
        except requests.exceptions.RequestException as exc:
            wait = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
            logger.warning("LLM request error: %s; retrying in %ss (attempt %d/%d)", exc, wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
            last_exc = exc
    raise RuntimeError(f"LLM request failed after {MAX_RETRIES} attempts: {last_exc}")


def get_llm_response(
    user_content: str,
    reasoning_effort: str = "low",
    temperature: float = 0.3,
    max_new_tokens: int = _TOKEN_BUDGET["default"],
    use_batch: bool = False,
) -> Dict[str, Any]:
    """Call internal LLM server (realtime by default, batch when use_batch=True)."""
    max_new_tokens = min(max_new_tokens, _MAX_TOKENS_HARD_LIMIT - 512)

    url = _BATCH_URL if use_batch else _REALTIME_URL
    if not url:
        raise RuntimeError(
            "LLM server URL not configured. Set LLM_BASE_URL (or LLM_REALTIME_URL / LLM_BATCH_URL) in env."
        )

    payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": user_content}],
        "reasoning": reasoning_effort,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    return _post(url, payload)


def get_llm_text(
    user_content: str,
    reasoning_effort: str = "low",
    temperature: float = 0.3,
    max_new_tokens: int = _TOKEN_BUDGET["default"],
    use_batch: bool = False,
) -> str:
    try:
        resp = get_llm_response(
            user_content,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_batch=use_batch,
        )
        return _extract_text(resp)
    except Exception as exc:
        logger.warning("get_llm_text failed: %s", exc)
        return ""


def _extract_text(response: Dict[str, Any]) -> str:
    """Extract generated text from any response shape the server may return."""
    if not response:
        return ""
    # Direct text fields
    for field in ("generated_text", "content", "text", "output"):
        val = response.get(field)
        if val and isinstance(val, str):
            return val.strip()
    # OpenAI-style choices
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
