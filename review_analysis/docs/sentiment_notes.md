# Sentiment classification — implementation notes (Agent 6)

Module: `review_analysis/app/utils/sentiment.py`

## Source / thresholds reused from the notebook

The reference notebook (`review_analysis/code/reviews_analysis (1).ipynb`)
uses VADER (`vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer`) to
score `reviewText` and stores the raw **compound** score in `df["sent"]`
(cell 1). It does **not** define a general positive/negative/neutral
classifier — every chart in the notebook works with the raw compound score
directly (means by star rating, scatter plots, etc.).

The one place the notebook does bucket by a threshold is the stars-vs-text
validation chart (cell 7), which flags "negative text" as `sent < -.05`.
That cutoff is exactly VADER's own standard convention, so this module
formalizes it into a full 3-way classifier using the standard VADER
thresholds:

| compound score | label |
|---|---|
| `>= 0.05` | `"positive"` |
| `<= -0.05` | `"negative"` |
| otherwise | `"neutral"` |

This keeps the app consistent with the notebook's own (implicit) convention
rather than inventing new cutoffs.

**Empty/missing text**: the notebook only scores rows where `has_text` is
True and leaves `sent` as `NaN` otherwise — it never calls VADER on empty
text. This module follows that spirit but returns an explicit label instead
of a silent gap: missing/empty/whitespace-only `reviewText` (None, NaN, `""`,
`"   "`) classifies as `"no_text"`, not `"neutral"`, so downstream code can
distinguish "we didn't have anything to score" from "VADER scored it as
neutral."

## Environment / dependency check

- `vaderSentiment` is **not** installed under the Homebrew `python3` on PATH
  (`/opt/homebrew/bin/python3`).
- It **is** already installed (v3.3.2) under the Miniconda base env
  (`/Users/lakshyatomar/miniconda3/bin/python3`), which is also the only env
  found with `streamlit` installed — this is almost certainly the env the
  app runs in.
- Action for Agent 7 (`requirements.txt`): add `vaderSentiment` regardless,
  since it's not guaranteed to exist in whatever env actually runs `streamlit
  run app/...` in a fresh setup. Pin: `vaderSentiment>=3.3.2` (or unpinned —
  no breaking API changes expected).

## Public API for integration

```python
from app.utils.sentiment import get_analyzer, classify_sentiment, add_sentiment_column

get_analyzer() -> SentimentIntensityAnalyzer
    # lru_cache'd singleton; not Streamlit-specific (works outside st context).

classify_sentiment(text: str) -> str
    # returns "positive" | "negative" | "neutral" | "no_text"

add_sentiment_column(
    df: pd.DataFrame,
    text_col: str = "reviewText",
    out_col: str = "sentiment",
) -> pd.DataFrame
    # returns a COPY of df with out_col added (does not mutate input).
    # raises KeyError if text_col is missing.
```

Agent 5 (review drilldown) should call `add_sentiment_column(reviews_df)`
once when loading review data, then filter/group on the `"sentiment"`
column (or call `classify_sentiment(text)` directly for one-off text, e.g. a
single expanded review row).

## Test results (real sample data)

Ran against `review_analysis/data/final_reviews.csv` (6,209 rows total,
1,740 with `reviewText` NaN) using the Miniconda `python3` (has both
`vaderSentiment` and `streamlit` installed):

- **8 random 5★ reviews with text** → all classified `"positive"`
  (e.g. *"The best car wash in town, with the nicest staff"*), except one
  2★ Apple-Pay-billing-complaint review that read `"positive"` overall
  because it closes with *"the staff is very nice and professional"* —
  this matches the notebook's own documented caveat (cell 8) that VADER
  occasionally disagrees with star rating on mixed-sentiment text.
- **2 rows with NaN `reviewText`** → both classified `"no_text"` as
  designed.
- **6 random 1–2★ reviews with text** → 5 of 6 classified `"negative"`
  (damage complaints, "no soap for mats", "highly unprofessional", etc.);
  1 of 6 classified `"positive"` despite 1★ rating — again a
  text/star-disagreement case (complains about damage but phrases it
  mildly), consistent with the notebook's finding that this happens for a
  small minority of reviews.
- **Edge cases**: `""`, `"   "`, `None`, `np.nan`, `float("nan")` all
  → `"no_text"`. Clearly positive/negative synthetic strings classified
  correctly. `add_sentiment_column` correctly raises `KeyError` when
  `text_col` doesn't exist. `get_analyzer()` returns the same cached
  instance across repeated calls.

Full pass — module behaves as intended on real data.

## Future implementation plan (deferred — NOT implemented here)

Out of scope for this module per requirements; noted for whoever picks
these up later:

- **Replied review count**: `df["ownerResponseText"].notna()` already gives
  a per-row boolean (the notebook computes this as `df["responded"]` in
  cell 1); a future metric module could aggregate `.sum()` /
  `.groupby("site")["responded"].mean()` for a "response rate" stat. No
  sentiment-specific logic needed — this is really a separate
  "owner-engagement" concern, not sentiment.
- **Negative-keyword extraction**: the notebook's theme-tagging (cell 13,
  regex keyword buckets like `staff/service`, `damage`, `price/value`) and
  distinctive-term mining (cell 17, log-odds of 1–2★ vs 4–5★ vocabulary,
  e.g. surfacing `refund`, `denied`, `accountability` as near-exclusive to
  low-star reviews) could be lifted into a
  `review_analysis/app/utils/keywords.py` module. Would take
  `add_sentiment_column`'s output (or the raw `rating` column) as input to
  split the corpus, then run the same regex/log-odds approach.
- **Other advanced analytics**: sentiment trend over time (`reviewDate`
  resampled), sentiment-by-location leaderboard (mirrors notebook cell 10
  but using `sentiment` instead of/alongside `rating`), and
  disagreement-flagging (star rating vs. sentiment mismatch, per notebook
  cell 7) are all straightforward extensions once `add_sentiment_column`
  is wired into the main review dataframe — but each was explicitly called
  out as out of scope for this task and is left for a later pass.
