"""LLM summaries of a review selection (Azure OpenAI, gpt-4o).

The app is deterministic everywhere else; this is the one place that calls out
to a model, and it only ever *describes* reviews the reader is already looking
at. It never produces a metric — every number on screen still comes from
`metrics.py`, so a summary can be wrong in its wording without corrupting the
dashboard.

Credentials are read from `st.secrets` or the environment, never hardcoded:

    # .streamlit/secrets.toml  (gitignored)
    azure_openai_api_key = "..."
    azure_openai_endpoint = "https://<resource>.openai.azure.com"
    azure_openai_deployment = "gpt-4o"
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd

try:
    import streamlit as st

    _cache = st.cache_data
except Exception:  # pragma: no cover
    st = None

    def _cache(*a, **k):
        def deco(fn):
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

DEFAULT_ENDPOINT = "https://soneastus2proformaai.openai.azure.com"
DEFAULT_DEPLOYMENT = "gpt-4o"
API_VERSION = "2024-02-15-preview"

MAX_REVIEWS = 60          # how many reviews get sent
MAX_CHARS_PER_REVIEW = 400
REQUEST_TIMEOUT = 60

SYSTEM_PROMPT = (
    "You are a customer-experience analyst for a car-wash chain. You are given a "
    "selection of that chain's Google reviews and a question about them.\n"
    "Rules:\n"
    "- Answer ONLY from the reviews provided. Never invent an incident, a location "
    "or a number.\n"
    "- Lead with the recurring themes, most common first, and say roughly how many "
    "of the reviews shown mention each.\n"
    "- Quote a short phrase (a few words) as evidence for each theme.\n"
    "- 3-6 bullets, one line each where possible. No preamble, no sign-off, no "
    "recommendations unless the question asks for them.\n"
    "- If the reviews do not support an answer, say so plainly."
)


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


def review_digest(df: pd.DataFrame, text_col: str, rating_col: str, date_col: str,
                  site_col: str, max_reviews: int = MAX_REVIEWS) -> tuple[str, ...]:
    """Format the reviews to send: the ones with text, newest first.

    Rating-only rows carry nothing for a language model to read, so they are
    dropped here rather than padding the prompt with empty strings.
    """
    with_text = df[df[text_col].fillna("").str.strip() != ""]
    if with_text.empty:
        return ()

    rows = with_text.sort_values(date_col, ascending=False).head(max_reviews)
    out = []
    for _, r in rows.iterrows():
        text = str(r[text_col]).strip().replace("\n", " ")
        if len(text) > MAX_CHARS_PER_REVIEW:
            text = text[:MAX_CHARS_PER_REVIEW] + "…"
        date = r[date_col].strftime("%b %Y") if pd.notna(r[date_col]) else "undated"
        out.append(f"[{int(r[rating_col])}★ · {r[site_col]} · {date}] {text}")
    return tuple(out)


@_cache(show_spinner=False, ttl=3600, max_entries=64)
def generate(question: str, scope: str, reviews: tuple[str, ...]) -> str:
    """Ask the model `question` about `reviews`. Cached per exact selection.

    Returns the answer, or a message beginning with a warning sign if the call
    could not be made — the caller renders whatever comes back, so a failure
    shows up as text on the page rather than a traceback.
    """
    key = _setting("azure_openai_api_key")
    if not key:
        return ("⚠️ No API key configured. Add `azure_openai_api_key` to "
                "`.streamlit/secrets.toml` (or set AZURE_OPENAI_API_KEY) and reload.")
    if not reviews:
        return "No reviews with text in this selection — nothing to summarise."

    endpoint = (_setting("azure_openai_endpoint", DEFAULT_ENDPOINT) or "").rstrip("/")
    deployment = _setting("azure_openai_deployment", DEFAULT_DEPLOYMENT)
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={API_VERSION}"

    body = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\n"
                f"Selection: {scope}\n"
                f"({len(reviews)} reviews with text are shown below; they are the most "
                f"recent in the selection.)\n\n" + "\n".join(reviews)
            )},
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }

    request = urllib.request.Request(
        url, method="POST", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": key},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.load(response)
        return payload["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as exc:
        return f"⚠️ Azure OpenAI returned {exc.code}: {exc.reason}"
    except Exception as exc:  # network down, timeout, unexpected shape
        return f"⚠️ Could not reach Azure OpenAI: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
# The question each tile's drill-down opens with. The reader can edit it.
DEFAULT_QUESTIONS = {
    "negative": "What are the main concerns customers raise in these negative reviews?",
    "worst": "What is going wrong at this location, according to its reviews?",
    "positive": "What do customers praise most in these positive reviews?",
    "best": "What is this location doing well, according to its reviews?",
    "sentiment": "What is driving sentiment here — what do people complain about and praise?",
    "rating": "What separates the highest-rated locations from the lowest-rated ones?",
    "response": "What are these unanswered reviews asking for, and which need a reply first?",
    "reviews": "What are customers talking about in this period?",
    "volume": "What themes stand out across these reviews?",
}
DEFAULT_QUESTION = "What are the recurring themes in these reviews?"


def _render_insight(scope_label: str, question: str, digest: tuple[str, ...],
                    answer_key: str, key: str) -> None:
    """Body of the insight panel: the answer, plus a follow-up box."""
    st.caption(f"{scope_label} · summarised from {len(digest)} reviews with text")
    answer = st.session_state.get(answer_key, "")

    follow_up = st.text_input("Ask a follow-up about these reviews", value=question,
                              key=f"aiq_{key}")
    if st.button("Ask", key=f"aiask_{key}"):
        with st.spinner("Reading the reviews..."):
            answer = generate(follow_up, scope_label, digest)
        st.session_state[answer_key] = answer

    # st.markdown on the raw answer, not wrapped in HTML: the model replies in
    # markdown, and inside an HTML div the ** and - render as literal characters.
    # The styled panel comes from the keyed container instead.
    with st.container(key=f"qaibox_{key}"):
        st.markdown(answer)
    st.caption("Generated by gpt-4o from the reviews above — it summarises text, it does not "
               "compute the figures on this page.")


if st is not None and hasattr(st, "dialog"):
    @st.dialog("✨ AI insights", width="large")
    def _insight_dialog(scope_label: str, question: str, digest: tuple[str, ...],
                        answer_key: str, key: str) -> None:
        _render_insight(scope_label, question, digest, answer_key, key)
else:  # pragma: no cover - older Streamlit
    _insight_dialog = None


def insight_button(key: str, reviews, scope_label: str, focus: str | None = None,
                   text_col: str = "reviewText", rating_col: str = "rating",
                   date_col: str = "reviewDate", site_col: str = "site",
                   button_label: str = "✨ AI insights") -> None:
    """A one-press '✨ AI insights' action for exactly this selection.

    Pressing it summarises immediately — no second Generate click — and opens
    the answer in a dialog with a follow-up box. Generation is deliberately
    tied to the press rather than to rendering: the popover this replaced had
    its body evaluated on every rerun, which with nine tiles on screen would
    have meant nine model calls per page load.

    `generate` is cached, so re-opening the same selection is instant and free.
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

    if not is_configured():
        st.warning("No Azure OpenAI key configured — add `azure_openai_api_key` to "
                   "`.streamlit/secrets.toml` and reload.")
        return

    digest = review_digest(reviews, text_col, rating_col, date_col, site_col)
    with st.spinner("Reading the reviews..."):
        st.session_state[answer_key] = generate(question, scope_label, digest)

    if _insight_dialog is not None:
        _insight_dialog(scope_label, question, digest, answer_key, key)
    else:  # fallback: render inline under the tile
        _render_insight(scope_label, question, digest, answer_key, key)
