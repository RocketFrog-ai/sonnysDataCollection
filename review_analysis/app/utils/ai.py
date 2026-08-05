"""LLM insights over a review selection (Azure OpenAI, gpt-4o).

The app is deterministic everywhere else; this is the one place that calls out
to a model, and it only ever *describes* reviews the reader is already looking
at. It never produces a metric — every number on screen still comes from
`metrics.py`, so a summary can be wrong in its wording without corrupting the
dashboard.

Shape of a call
---------------
A selection can be hundreds of reviews, which is more than fits in one useful
prompt, so this is map-reduce: reviews are chunked, each chunk is summarised
into themes with counts and quotes, and a second call merges those into the
final answer. One chunk skips nothing — it still runs the merge, so the output
shape is identical either way. The answer is JSON — headline, key points,
recommendations — so the page lays it out instead of hoping a markdown blob
renders well.

Answers are cached twice: in-process (`st.cache_data`) and on disk
(`data/ai_insights_cache.json`), keyed by a hash of the exact reviews
summarised. `scripts/precompute_insights.py` fills that file for the windows
the dashboard opens with, so pressing a tile's button is usually instant.

Credentials are read from `st.secrets` or the environment, never hardcoded:

    # .streamlit/secrets.toml  (gitignored)
    azure_openai_api_key = "..."
    azure_openai_endpoint = "https://<resource>.openai.azure.com"
    azure_openai_deployment = "gpt-4o"
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pandas as pd

try:
    import streamlit as st

    _cache = st.cache_data
except Exception:  # pragma: no cover - importable without Streamlit
    st = None

    def _cache(*a, **k):
        def deco(fn):
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

DEFAULT_ENDPOINT = "https://soneastus2proformaai.openai.azure.com"
DEFAULT_DEPLOYMENT = "gpt-4o"
API_VERSION = "2024-02-15-preview"

MAX_REVIEWS = 400          # ceiling on how many reviews one insight covers
CHUNK_SIZE = 45            # reviews per call in the map stage
MAX_CHARS_PER_REVIEW = 400
REQUEST_TIMEOUT = 90

CACHE_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ai_insights_cache.json"))
_CACHE_LOCK = threading.Lock()

_GROUND_RULES = (
    "You are a customer-experience analyst for a car-wash chain, reading its Google "
    "reviews.\n"
    "Ground rules:\n"
    "- Use ONLY the reviews given. Never invent an incident, a location or a number.\n"
    "- Counts must be counts of the reviews you were shown.\n"
    "- Quote a few words verbatim as evidence; never paraphrase inside quote marks.\n"
    "- Plain, specific language. No filler, no marketing tone."
)

MAP_PROMPT = _GROUND_RULES + (
    "\n\nSummarise this batch of reviews. Reply with JSON only:\n"
    '{"themes": [{"theme": "short label", "count": <int>, '
    '"detail": "one sentence", "evidence": ["short quote", ...]}]}\n'
    "Order themes by count, most common first. Merge near-duplicates."
)

REDUCE_PROMPT = _GROUND_RULES + (
    "\n\nYou are given theme summaries from several batches of the same selection, and "
    "a question. Merge them into one answer and reply with JSON only:\n"
    '{"headline": "one sentence answering the question directly",\n'
    ' "key_points": [{"point": "short label", "count": <int>, '
    '"detail": "1-2 sentences of what customers actually say", '
    '"evidence": ["short quote", ...]}],\n'
    ' "recommendations": [{"action": "what to do, concretely", '
    '"why": "the evidence it follows from"}]}\n'
    "Rules: 3-6 key points ordered by how many reviews mention them; counts summed "
    "across batches; 2-4 recommendations, each tied to a key point and specific enough "
    "to act on this week. If the reviews cannot answer the question, say so in the "
    "headline and return empty lists."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from st.secrets, then the environment, then a default."""
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception:  # no secrets.toml at all
            pass
    return os.getenv(name.upper(), default)


def is_configured() -> bool:
    return bool(_setting("azure_openai_api_key"))


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def review_digest(df: pd.DataFrame, text_col: str = "reviewText", rating_col: str = "rating",
                  date_col: str = "reviewDate", site_col: str = "site",
                  max_reviews: int = MAX_REVIEWS) -> tuple[str, ...]:
    """Format the reviews to send: the ones with text, newest first.

    Rating-only rows carry nothing for a language model to read, so they are
    dropped here rather than padding the prompt with empty strings.
    """
    if df is None or df.empty:
        return ()
    with_text = df[df[text_col].fillna("").str.strip() != ""]
    if with_text.empty:
        return ()

    # mergesort (stable): the cache key hashes the ordered tuple, and pandas'
    # default quicksort reorders ties differently depending on the incoming row
    # order, which would silently invalidate every precomputed answer.
    rows = with_text.sort_values(date_col, ascending=False, kind="mergesort").head(max_reviews)
    out = []
    for _, r in rows.iterrows():
        text = str(r[text_col]).strip().replace("\n", " ")
        if len(text) > MAX_CHARS_PER_REVIEW:
            text = text[:MAX_CHARS_PER_REVIEW] + "…"
        date = r[date_col].strftime("%b %Y") if pd.notna(r[date_col]) else "undated"
        out.append(f"[{int(r[rating_col])}★ · {r[site_col]} · {date}] {text}")
    return tuple(out)


def digest_coverage(df: pd.DataFrame, text_col: str = "reviewText",
                    max_reviews: int = MAX_REVIEWS) -> tuple[int, int]:
    """(reviews sent, reviews with text available) — for an honest caption."""
    if df is None or df.empty:
        return 0, 0
    available = int((df[text_col].fillna("").str.strip() != "").sum())
    return min(available, max_reviews), available


def digest_key(question: str, scope: str, reviews: tuple[str, ...]) -> str:
    """Stable id for one answer: the question, the scope and the exact reviews."""
    blob = "␟".join((question, scope, *reviews)).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _read_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def cache_get(key: str) -> dict | None:
    return _read_cache().get(key)


def cache_put(key: str, payload: dict) -> None:
    """Write one answer. Read-modify-write under a lock (single-process app)."""
    with _CACHE_LOCK:
        cache = _read_cache()
        cache[key] = payload
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, CACHE_PATH)


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------
MAX_ATTEMPTS = 3
RETRY_STATUS = {429, 500, 502, 503, 504}


def _chat(system: str, user: str, max_tokens: int = 1200) -> dict:
    """One chat completion that must return JSON.

    Retries the statuses Azure returns transiently (a 500 lost one scope of a
    precompute run); raises on anything else, or once the attempts run out.
    """
    key = _setting("azure_openai_api_key")
    if not key:
        raise RuntimeError("no API key configured")

    endpoint = (_setting("azure_openai_endpoint", DEFAULT_ENDPOINT) or "").rstrip("/")
    deployment = _setting("azure_openai_deployment", DEFAULT_DEPLOYMENT)
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={API_VERSION}"

    body = {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url, method="POST", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": key},
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                payload = json.load(response)
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                # Retrying an identical prompt at temperature 0.2 just truncates
                # again; fail loudly so the caller reports it instead of paying
                # for three identical calls.
                raise RuntimeError("model reply hit the token limit — reduce CHUNK_SIZE")
            return json.loads(choice["message"]["content"])
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == MAX_ATTEMPTS:
                raise
        time.sleep(2 ** attempt)  # 2s, 4s
    raise RuntimeError("unreachable")


def _chunks(reviews: tuple[str, ...], size: int = CHUNK_SIZE):
    for i in range(0, len(reviews), size):
        yield reviews[i:i + size]


def build_insight(question: str, scope: str, reviews: tuple[str, ...],
                  use_cache: bool = True) -> dict:
    """Map-reduce the reviews into {headline, key_points, recommendations}.

    Returns a dict carrying an `error` key instead of raising, so a failed call
    renders as a message on the page rather than a traceback.
    """
    if not reviews:
        return {"error": "No reviews with text in this selection — nothing to summarise."}

    # Cache first, key second: the precomputed answers ship with the repo, and
    # checking configuration ahead of the cache made all 18 unreachable on a
    # machine with no key.
    key = digest_key(question, scope, reviews)
    if use_cache:
        cached = cache_get(key)
        if cached:
            return cached

    if not is_configured():
        return {"error": "No Azure OpenAI key configured. Add `azure_openai_api_key` to "
                         "`.streamlit/secrets.toml` (or set AZURE_OPENAI_API_KEY)."}

    try:
        batches = list(_chunks(reviews))
        summaries = []
        for i, batch in enumerate(batches, start=1):
            got = _chat(MAP_PROMPT,
                        f"Selection: {scope}\nBatch {i} of {len(batches)} "
                        f"({len(batch)} reviews)\n\n" + "\n".join(batch))
            summaries.append(got.get("themes", got))

        merged = _chat(
            REDUCE_PROMPT,
            f"Question: {question}\nSelection: {scope}\n"
            f"Reviews summarised: {len(reviews)} across {len(batches)} batch(es).\n\n"
            + json.dumps(summaries, ensure_ascii=False),
            max_tokens=1600,
        )
    except urllib.error.HTTPError as exc:
        return {"error": f"Azure OpenAI returned {exc.code}: {exc.reason}"}
    except Exception as exc:  # network, timeout, malformed JSON
        return {"error": f"Could not complete the summary: {exc}"}

    answer = {
        "headline": str(merged.get("headline", "")).strip(),
        "key_points": merged.get("key_points", []) or [],
        "recommendations": merged.get("recommendations", []) or [],
        "n_reviews": len(reviews),
        "n_calls": len(batches) + 1,
        "question": question,
        "scope": scope,
    }
    cache_put(key, answer)
    return answer


@_cache(show_spinner=False, ttl=24 * 3600, max_entries=128)
def _build_memo(question: str, scope: str, reviews: tuple[str, ...]) -> dict:
    return build_insight(question, scope, reviews)


def _build_cached(question: str, scope: str, reviews: tuple[str, ...]) -> dict:
    """Memoised build, except for failures.

    A transient 502 must not be remembered for 24 hours: that would leave the
    tile permanently broken with no way to retry short of restarting the app.
    """
    answer = _build_memo(question, scope, reviews)
    if answer.get("error"):
        try:
            _build_memo.clear()
        except Exception:  # pragma: no cover - no-op cache outside Streamlit
            pass
    return answer


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
# The question each tile asks. The reader can follow up with their own.
DEFAULT_QUESTIONS = {
    "negative": "What are the main concerns customers raise in these negative reviews?",
    "worst": "What is going wrong at this location, according to its reviews?",
    "positive": "What do customers praise most in these positive reviews?",
    "best": "What is this location doing well, according to its reviews?",
    "sentiment": "What is driving sentiment here — what do customers complain about and praise?",
    "rating": "What separates the highest-rated locations from the lowest-rated ones?",
    "response": "What are these unanswered reviews asking for, and which need a reply first?",
    "reviews": "What are customers talking about in this period?",
    "volume": "What themes stand out across these reviews?",
}
DEFAULT_QUESTION = "What are the recurring themes in these reviews?"


def _demo_mode() -> bool:
    """Demo view hides provenance lines; the analyst view keeps them."""
    if st is None:
        return False
    from app.utils import theme  # local import: ai stays independent of the shell
    return theme.demo_mode()


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_answer(answer: dict) -> None:
    """Lay out one insight: headline, key points, recommendations."""
    if not answer:
        return
    if answer.get("error"):
        st.warning(answer["error"])
        return

    if answer.get("headline"):
        st.markdown(f'<div class="q-ai-headline">{_esc(answer["headline"])}</div>',
                    unsafe_allow_html=True)

    points = [p for p in (answer.get("key_points") or []) if isinstance(p, dict)]
    if points:
        rows = []
        for p in points:
            quotes = " ".join(f'<span class="q-ai-quote">“{_esc(q)}”</span>'
                              for q in (p.get("evidence") or [])[:3])
            count = p.get("count")
            chip = ""
            if isinstance(count, (int, float)):
                chip = (f'<span class="q-ai-count">{int(count)} review'
                        f'{"s" if int(count) != 1 else ""}</span>')
            rows.append(
                f'<div class="q-ai-point"><div class="q-ai-point-head">'
                f'<span class="q-ai-point-title">{_esc(p.get("point", ""))}</span>{chip}</div>'
                f'<div class="q-ai-point-detail">{_esc(p.get("detail", ""))}</div>'
                f'<div class="q-ai-evidence">{quotes}</div></div>'
            )
        st.markdown('<div class="q-ai-section">Key points</div>', unsafe_allow_html=True)
        st.markdown("".join(rows), unsafe_allow_html=True)

    recs = [r for r in (answer.get("recommendations") or []) if isinstance(r, dict)]
    if recs:
        items = "".join(
            f'<div class="q-ai-rec"><span class="q-ai-rec-n">{i}</span>'
            f'<div><div class="q-ai-rec-action">{_esc(r.get("action", ""))}</div>'
            f'<div class="q-ai-rec-why">{_esc(r.get("why", ""))}</div></div></div>'
            for i, r in enumerate(recs, start=1)
        )
        st.markdown('<div class="q-ai-section">Recommendations</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="q-ai-recs">{items}</div>', unsafe_allow_html=True)

    if not _demo_mode():
        st.caption(
            f"gpt-4o over {answer.get('n_reviews', 0)} reviews "
            f"({answer.get('n_calls', 1)} call{'s' if answer.get('n_calls', 1) != 1 else ''}) — "
            "it summarises review text and does not compute any figure on this page."
        )


def _dialog_body(scope_label: str, question: str, digest: tuple[str, ...], answer_key: str,
                 key: str) -> None:
    """The dialog is read-only: the tile's own question, answered."""
    st.caption(scope_label)
    render_answer(st.session_state.get(answer_key, {}))


if st is not None and hasattr(st, "dialog"):
    @st.dialog("✨ AI insights", width="large")
    def _insight_dialog(scope_label: str, question: str, digest: tuple[str, ...],
                        answer_key: str, key: str) -> None:
        _dialog_body(scope_label, question, digest, answer_key, key)
else:  # pragma: no cover - older Streamlit
    _insight_dialog = None


def insight_button(key: str, reviews, scope_label: str, focus: str | None = None,
                   text_col: str = "reviewText", rating_col: str = "rating",
                   date_col: str = "reviewDate", site_col: str = "site",
                   button_label: str = "✨ AI insights") -> None:
    """One-press insights for exactly this selection.

    Generation is bound to the press, not to rendering: this used to live in a
    popover whose body Streamlit evaluates on every rerun, which with nine
    tiles on screen would mean nine map-reduce jobs per page load. The answer
    is read-only -- each tile asks its own fixed question, which is what makes
    the precomputed cache hit.
    """
    if st is None:
        return

    answer_key = f"aia_{key}"
    question = DEFAULT_QUESTIONS.get(focus or "", DEFAULT_QUESTION)

    with st.container(key=f"qai_{key}"):
        pressed = st.button(button_label, key=f"aib_{key}",
                            help="Summarise these reviews with gpt-4o")
    if not pressed:
        return

    digest = review_digest(reviews, text_col, rating_col, date_col, site_col)
    with st.spinner("Reading the reviews..."):
        st.session_state[answer_key] = _build_cached(question, scope_label, digest)

    # The coverage note is display-only. Folding it into scope_label (as this
    # briefly did) changes the cache key, so every selection over MAX_REVIEWS
    # missed its precomputed answer and paid for a fresh map-reduce.
    sent, available = digest_coverage(reviews, text_col)
    shown_label = (f"{scope_label} · newest {sent:,} of {available:,} reviews with text"
                   if available > sent else scope_label)

    if _insight_dialog is not None:
        _insight_dialog(shown_label, question, digest, answer_key, key)
    else:  # pragma: no cover - fallback for older Streamlit
        _dialog_body(shown_label, question, digest, answer_key, key)
