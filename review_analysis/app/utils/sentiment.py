"""VADER-based sentiment classification for review text.

Reuses the sentiment approach from the exploratory notebook
(review_analysis/code/reviews_analysis (1).ipynb), which scores review text
with VADER's compound score (`SentimentIntensityAnalyzer().polarity_scores(text)["compound"]`).

The notebook itself never defines a 3-way positive/negative/neutral bucketer —
it works with the raw compound score directly — but the one classification
threshold it does use ad hoc (`sent < -.05` to flag "negative text" in the
stars-vs-text validation chart) matches VADER's own standard convention.
This module formalizes that into a reusable classifier using the standard
VADER thresholds:

    compound >= 0.05   -> "positive"
    compound <= -0.05  -> "negative"
    otherwise          -> "neutral"

Scope note: this module intentionally implements ONLY analyzer access,
single-text classification, and a dataframe helper. Replied-review-count,
negative keyword extraction, and other advanced analytics are out of scope
here — see review_analysis/docs/sentiment_notes.md for the deferred-work
plan.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Standard VADER compound-score thresholds (matches the notebook's own
# ad hoc negative cutoff of -0.05; see module docstring).
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

# Label returned for empty / missing / whitespace-only text. VADER would
# score empty string as compound == 0.0, which is indistinguishable from a
# genuinely neutral review of real text, so no-text reviews get their own
# label rather than being folded into "neutral".
NO_TEXT_LABEL = "no_text"


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return a cached VADER SentimentIntensityAnalyzer instance.

    Cached (not re-instantiated per call) since construction loads VADER's
    lexicon from disk. Plain functools.lru_cache, not st.cache_resource, so
    this module stays framework-agnostic and unit-testable outside Streamlit.
    """
    return SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> str:
    """Classify a single piece of review text as positive/negative/neutral.

    Uses VADER's compound score with the standard thresholds:
      - compound >= 0.05  -> "positive"
      - compound <= -0.05 -> "negative"
      - otherwise         -> "neutral"

    Missing/empty/whitespace-only text (None, NaN, "") returns "no_text"
    rather than "neutral", since VADER has nothing to score and a silent
    "neutral" would misleadingly imply the text was analyzed.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return NO_TEXT_LABEL

    text = str(text).strip()
    if not text:
        return NO_TEXT_LABEL

    compound = get_analyzer().polarity_scores(text)["compound"]
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def add_sentiment_column(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    out_col: str = "sentiment",
) -> pd.DataFrame:
    """Return a copy of df with a sentiment label column added.

    Applies classify_sentiment across df[text_col] and writes the result to
    df[out_col] ("positive" / "negative" / "neutral" / "no_text"). Does not
    mutate the input dataframe.

    Raises KeyError if text_col is not present in df.
    """
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' not found in dataframe columns: {list(df.columns)}")

    out = df.copy()
    out[out_col] = out[text_col].apply(classify_sentiment)
    return out
