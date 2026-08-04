"""Metric computation for the dashboard tiles and the drill-down table.

Presentation lives in `theme.py`, data cleaning in `data_loader.py`; this is
the layer in between — every number either page prints comes from here, so
the landing tile and the drill-down row for the same site cannot disagree.

Two conventions worth knowing before reading anything below:

*Current period.* The dataset ends mid-month (last review 2026-07-22), so
"current" means the latest calendar month **present in the data**, not the
wall-clock month, and "last month" is the calendar month before it. A
wall-clock definition would show an empty tile whenever the scrape is stale.

*Sentiment denominators.* 28% of reviews are rating-only (no text at all).
Those score `no_text`, never `neutral` — VADER scores an empty string as
0.0/neutral, which would make "no opinion recorded" indistinguishable from
"wrote something even-handed". Sentiment shares are therefore taken over
reviews **with text**; `n_scored` carries that denominator so a caller can
never accidentally divide by the full review count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # Streamlit is optional so this module stays importable/testable alone
    import streamlit as st

    _cache = st.cache_data
except Exception:  # pragma: no cover
    def _cache(*a, **k):
        def deco(fn):
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

from app.utils.data_loader import (
    DATE_COL,
    OWNER_RESPONSE_DATE_COL,
    OWNER_RESPONSE_TEXT_COL,
    RATING_COL,
    SITE_COL,
    TEXT_COL,
    BUSINESS_AVG_RATING_COL,
    BUSINESS_REVIEW_COUNT_COL,
    load_data,
)
from app.utils.sentiment import add_sentiment_scores

SCORE_COL = "sentiment_score"
LABEL_COL = "sentiment"
MONTH_COL = "review_month"
RESPONSE_HOURS_COL = "response_hours"


@_cache(show_spinner="Scoring review sentiment...")
def load_scored_data() -> pd.DataFrame:
    """Clean reviews + VADER score/label + month bucket + response latency.

    Scoring the full 6.2k-row file takes ~0.25s, so it is done once here and
    cached, rather than per-page on a filtered subset (which would make the
    same review's score depend on which filter produced it).
    """
    df = load_data()
    df = add_sentiment_scores(df, text_col=TEXT_COL, score_col=SCORE_COL, label_col=LABEL_COL)

    dates = df[DATE_COL]
    df[MONTH_COL] = dates.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")

    resp = df[OWNER_RESPONSE_DATE_COL]
    df[RESPONSE_HOURS_COL] = (resp - dates).dt.total_seconds() / 3600
    # Edited reviews carry the *edit* timestamp, so an owner reply can predate
    # the date we hold (docs/dataset_schema.md §5). Those negative latencies
    # are meaningless rather than fast -- drop them instead of reporting them.
    df.loc[df[RESPONSE_HOURS_COL] < 0, RESPONSE_HOURS_COL] = np.nan
    return df


def current_and_prior(df: pd.DataFrame) -> tuple[pd.Period | None, pd.Period | None]:
    """(latest month in the data, the calendar month before it)."""
    months = df[MONTH_COL].dropna()
    if months.empty:
        return None, None
    latest = months.max()
    return latest, latest - 1


def month_frame(df: pd.DataFrame, period: pd.Period | None) -> pd.DataFrame:
    if period is None:
        return df.iloc[0:0]
    return df[df[MONTH_COL] == period]


# Period selector options -> (window length in months, comparison caption).
# None means "everything", which has no prior window to compare against.
PERIOD_WINDOWS: dict[str, tuple[int | None, str]] = {
    "Current month": (1, "from last month"),
    "Last 3 months": (3, "from prior 3 months"),
    "Last 12 months": (12, "from prior 12 months"),
    "All time": (None, "no prior period"),
}


def window_frames(df: pd.DataFrame, choice: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Split reviews into (current window, prior window, window label, caption).

    The prior window is the same number of months immediately before the
    current one, so a 3-month tile compares like with like instead of against
    a single month.
    """
    months, caption = PERIOD_WINDOWS.get(choice, PERIOD_WINDOWS["Current month"])
    latest, _ = current_and_prior(df)
    if latest is None:
        return df.iloc[0:0], df.iloc[0:0], "no data", caption
    if months is None:
        first = df[MONTH_COL].dropna().min()
        return df, df.iloc[0:0], f"{first.strftime('%b %Y')} – {latest.strftime('%b %Y')}", caption

    cur_start = latest - (months - 1)
    prior_start, prior_end = cur_start - months, cur_start - 1
    in_cur = (df[MONTH_COL] >= cur_start) & (df[MONTH_COL] <= latest)
    in_prior = (df[MONTH_COL] >= prior_start) & (df[MONTH_COL] <= prior_end)
    label = (latest.strftime("%b %Y") if months == 1
             else f"{cur_start.strftime('%b %Y')} – {latest.strftime('%b %Y')}")
    return df[in_cur], df[in_prior], label, caption


def sentiment_counts(df: pd.DataFrame) -> dict:
    """Positive / neutral / negative counts plus the with-text denominator."""
    labels = df[LABEL_COL] if LABEL_COL in df.columns else pd.Series(dtype=str)
    counts = labels.value_counts()
    pos = int(counts.get("positive", 0))
    neu = int(counts.get("neutral", 0))
    neg = int(counts.get("negative", 0))
    scored = pos + neu + neg
    return {
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "no_text": int(counts.get("no_text", 0)),
        "n_scored": scored,
        "pct_positive": (pos / scored * 100) if scored else None,
        "pct_negative": (neg / scored * 100) if scored else None,
        # Net sentiment: share positive minus share negative, an NPS-shaped
        # number that moves when opinion flips rather than when volume grows.
        "net_sentiment": ((pos - neg) / scored * 100) if scored else None,
        "avg_score": (float(df[SCORE_COL].mean()) if scored else None),
    }


def period_metrics(df: pd.DataFrame) -> dict:
    """Every headline number for one slice of reviews (a month, or all time)."""
    n = len(df)
    if n == 0:
        return {
            "n_reviews": 0, "avg_rating": None, "pct_5_star": None, "n_1_2_star": 0,
            "pct_response": None, "median_response_hours": None, "n_sites": 0,
            "rating_counts": {r: 0 for r in range(1, 6)},
            **{k: (0 if k in {"positive", "neutral", "negative", "no_text", "n_scored"} else None)
               for k in ("positive", "neutral", "negative", "no_text", "n_scored",
                         "pct_positive", "pct_negative", "net_sentiment", "avg_score")},
        }

    ratings = pd.to_numeric(df[RATING_COL], errors="coerce")
    counts = ratings.value_counts()
    out = {
        "n_reviews": n,
        "avg_rating": float(ratings.mean()),
        "pct_5_star": float((ratings == 5).mean() * 100),
        "n_1_2_star": int((ratings <= 2).sum()),
        "pct_response": float(df[OWNER_RESPONSE_TEXT_COL].notna().mean() * 100),
        "median_response_hours": (
            float(df[RESPONSE_HOURS_COL].median())
            if df[RESPONSE_HOURS_COL].notna().any() else None
        ),
        "n_sites": int(df[SITE_COL].nunique()),
        "rating_counts": {r: int(counts.get(r, 0)) for r in range(1, 6)},
    }
    out.update(sentiment_counts(df))
    return out


def monthly_series(df: pd.DataFrame, months: int = 8) -> pd.DataFrame:
    """Per-month volume, rating and sentiment shares — the drill-down chart.

    Returns the last `months` calendar months with columns: period, label,
    n_reviews, avg_rating, pct_positive, pct_negative, net_sentiment.
    """
    if df.empty:
        return pd.DataFrame(columns=["period", "label", "n_reviews", "avg_rating",
                                     "pct_positive", "pct_negative", "net_sentiment"])

    rows = []
    for period, grp in df.groupby(MONTH_COL, observed=True):
        if pd.isna(period):
            continue
        sent = sentiment_counts(grp)
        rows.append({
            "period": period,
            "label": period.strftime("%b-%y"),
            "n_reviews": len(grp),
            "avg_rating": float(pd.to_numeric(grp[RATING_COL], errors="coerce").mean()),
            "pct_positive": sent["pct_positive"],
            "pct_negative": sent["pct_negative"],
            "net_sentiment": sent["net_sentiment"],
        })

    out = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    return out.tail(months).reset_index(drop=True)


ROLLUP_COLUMNS = [
    "n_reviews", "avg_rating", "google_rating", "google_review_count", "coverage_pct",
    "positive", "neutral", "negative", "no_text", "n_scored", "pct_positive",
    "pct_negative", "net_sentiment", "avg_score", "pct_response", "n_awaiting",
    "n_1_2_star", "last_review",
]


def _row_stats(grp: pd.DataFrame) -> dict:
    """The stats block behind one table row, whatever the row groups by.

    Used for both period rows and site rows so the two levels of the insights
    table can never drift apart: a month's total is literally the same
    function applied to a wider slice.

    `google_rating` is only meaningful when the slice is one site (Google
    publishes it per location), so it is left None when the group spans
    several -- rather than averaging 25 sites' published ratings into a
    number Google never reported.
    """
    ratings = pd.to_numeric(grp[RATING_COL], errors="coerce")
    sent = sentiment_counts(grp)

    one_site = grp[SITE_COL].nunique() == 1
    g_rating = g_count = None
    if one_site:
        gr = pd.to_numeric(grp[BUSINESS_AVG_RATING_COL], errors="coerce").dropna()
        gc = pd.to_numeric(grp[BUSINESS_REVIEW_COUNT_COL], errors="coerce").dropna()
        g_rating = float(gr.iloc[0]) if not gr.empty else None
        g_count = int(gc.iloc[0]) if not gc.empty else None

    return {
        "n_reviews": len(grp),
        "avg_rating": float(ratings.mean()),
        "google_rating": g_rating,
        "google_review_count": g_count,
        "coverage_pct": (len(grp) / g_count * 100) if g_count else None,
        "positive": sent["positive"],
        "neutral": sent["neutral"],
        "negative": sent["negative"],
        "no_text": sent["no_text"],
        "n_scored": sent["n_scored"],
        "pct_positive": sent["pct_positive"],
        "pct_negative": sent["pct_negative"],
        "net_sentiment": sent["net_sentiment"],
        "avg_score": sent["avg_score"],
        "pct_response": float(grp[OWNER_RESPONSE_TEXT_COL].notna().mean() * 100),
        "n_awaiting": int(grp[OWNER_RESPONSE_TEXT_COL].isna().sum()),
        "n_1_2_star": int((ratings <= 2).sum()),
        "last_review": grp[DATE_COL].max(),
    }


def site_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site.

    `avg_rating` is the mean of the scraped rows; `google_rating` is the
    site's published Google rating over all of its reviews. They are kept as
    separate columns rather than blended, because their denominators differ
    by up to 30x (see data_loader.get_locations).
    """
    if df.empty:
        return pd.DataFrame(columns=["site", *ROLLUP_COLUMNS])

    rows = [{"site": site, **_row_stats(grp)}
            for site, grp in df.groupby(SITE_COL, observed=True)]
    return pd.DataFrame(rows).sort_values("n_reviews", ascending=False).reset_index(drop=True)


# Granularity choices for the insights table / chart -> pandas period alias.
FREQ_CHOICES: dict[str, str] = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
_FREQ_FORMATS = {"M": "%b-%y", "Y": "%Y"}


def period_index(df: pd.DataFrame, granularity: str) -> pd.Series:
    """Period bucket per review at the requested granularity."""
    freq = FREQ_CHOICES.get(granularity, "M")
    naive = df[DATE_COL].dt.tz_convert("UTC").dt.tz_localize(None)
    return naive.dt.to_period(freq)


def period_label(period, granularity: str) -> str:
    """'Jan-26' / 'Q1-26' / '2026' for a period bucket."""
    freq = FREQ_CHOICES.get(granularity, "M")
    if freq == "Q":
        return f"Q{period.quarter}-{period.year % 100:02d}"
    return period.strftime(_FREQ_FORMATS[freq])


def period_table(df: pd.DataFrame, granularity: str = "Monthly",
                 last_n: int | None = 14) -> pd.DataFrame:
    """One row per calendar period — the top level of the insights table.

    `last_n` keeps the table to the most recent buckets (None = all). The
    default matches the chart above it, so the rows and the bars describe the
    same span.
    """
    if df.empty:
        return pd.DataFrame(columns=["period", "label", *ROLLUP_COLUMNS])

    work = df.assign(_period=period_index(df, granularity))
    rows = [{"period": period, "label": period_label(period, granularity), **_row_stats(grp)}
            for period, grp in work.groupby("_period", observed=True) if not pd.isna(period)]

    out = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    if last_n is not None:
        out = out.tail(last_n).reset_index(drop=True)
    return out


# Sort options offered inside an expanded site's review list. Each maps to
# (column, ascending); "sentiment" sorts on the raw compound score, which is
# the only ordering the coarse positive/neutral/negative label cannot express.
REVIEW_SORTS: dict[str, tuple[str, bool]] = {
    "Sentiment score — most positive": (SCORE_COL, False),
    "Sentiment score — most negative": (SCORE_COL, True),
    "Most recent": (DATE_COL, False),
    "Oldest first": (DATE_COL, True),
    "Rating — highest": (RATING_COL, False),
    "Rating — lowest": (RATING_COL, True),
}


def sort_reviews(df: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Sort a review frame by one of REVIEW_SORTS.

    `na_position="last"` matters: rating-only reviews have no sentiment score
    and must sink to the bottom of a sentiment sort rather than lead it.
    """
    col, ascending = REVIEW_SORTS.get(choice, (DATE_COL, False))
    return df.sort_values(col, ascending=ascending, na_position="last")
