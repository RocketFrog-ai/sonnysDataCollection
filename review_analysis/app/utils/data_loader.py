"""Data loading / cleaning utilities for the review-analysis Streamlit app.

Single source of truth for reading `data/final_reviews.csv` and turning it
into one clean DataFrame that every page works off. Pure pandas -- the only
optional Streamlit dependency is `st.cache_data`, imported defensively so
this module stays importable/testable outside Streamlit (see bottom of
file for a standalone `python3 app/utils/data_loader.py` smoke test).

Full column-by-column profiling, the location-key determination, and every
data-quality finding referenced in the comments below are written up in
`review_analysis/docs/dataset_schema.md` -- read that first if something
here looks surprising.

KEY FINDINGS THAT SHAPE THIS MODULE (see dataset_schema.md for evidence):

1. Real row count is 6,209, not the 7,897 raw lines in the file -- reviewText
   / ownerResponseText contain embedded newlines inside quoted CSV fields.
   Never use `wc -l` as a row-count proxy; pandas.read_csv parses this file
   cleanly with zero bad-line warnings.

2. `businessName` is constant ("Big Dan's Car Wash") -- this is ONE chain
   with 25 physical locations, not 25 businesses. `site` is the location
   dimension (e.g. "Big Dan's Rome"), and it maps 1:1 onto `address` with
   ZERO exceptions in either direction (verified: no site spans >1 address,
   no address is shared by >1 site). `site` is therefore the primary/composite
   location key -- LOCATION_KEY_COL below. `city` is NOT a valid location key
   on its own: three site pairs share a city (two Bradenton sites + a third
   Bradenton-city site on FL-64, two Woodstock sites, two Orlando sites).

3. `placeId` is null for exactly 5 of the 25 sites (Academy, Fountain Inn,
   Kissimmee OBT, Muscle Shoals, St Pete) -- all 5 are sourced entirely from
   the `own_crawler_*` pipelines rather than `v1_v2_scrape`, and correspondingly
   lack `businessAvgRating`/`businessReviewCount`/`isLocalGuide`/
   `reviewerReviewCount`/`likesCount`/`reviewUrl` too (same 377 rows null
   across all of those columns). This is a real, structural source-system
   gap, not randomly missing data -- do not impute a fake placeId.

4. `state` mixes full names ("Georgia", "Florida", "South Carolina") and
   abbreviations ("FL", "SC", "AL") for the SAME states, and this correlates
   perfectly with the placeId-null / own_crawler split above (the 5
   own_crawler sites use abbreviations; all v1_v2_scrape sites use full
   names). Normalized to full names by STATE_ABBR_TO_FULL below.

5. reviewId: there are NO genuine duplicate reviewId values among non-null
   reviewIds (6,084 non-null values, 6,084 unique). 125 rows have a null
   reviewId (all from own_crawler pipelines). IMPORTANT CORRECTION to an
   earlier team note: naively running `df['reviewId'].duplicated().sum()`
   reports 124 -- but that is a false positive caused by pandas treating
   NaN == NaN when comparing for duplicates, not real duplicate content.
   Blindly `drop_duplicates(subset='reviewId')` would silently discard 124
   legitimate, distinct reviews. This module dedupes reviewId only among
   non-null values (see `_dedupe` below).

6. One genuine duplicate WAS found by natural key (site + reviewerName +
   reviewDate + rating + reviewText): the same "Debi Madden" 5-star review
   at "Big Dan's Fountain Inn" appears twice -- once via own_crawler_api
   (null reviewId) and once via own_crawler_crawler (real reviewId). When
   this pattern occurs, the row WITH a reviewId is kept.

7. reviewDate mixes two ISO-8601 shapes: full timestamps with milliseconds
   ("2026-07-17T15:45:58.308Z", 5,712 rows, all v1_v2_scrape) and date-only
   strings ("2026-07-17", 497 rows, all own_crawler_*). Calling
   `pd.to_datetime(..., utc=True)` WITHOUT `format="mixed"` (or "ISO8601")
   silently turns every date-only row into NaT. This module always passes
   `format="mixed"`.

8. Caveat for trend analysis: reviewDate is Google's displayed review date,
   which reflects the EDIT time for reviews the reviewer later edited, not
   the original post time. Evidence: of 512 rows where ownerResponseDate
   precedes reviewDate (which should be impossible if reviewDate were the
   original post time), 502 (98%) have `reviewDateRelative` starting with
   "Edited ...". Treat reviewDate as the best available "effective" review
   date; day-level precision on older, edited reviews may not reflect the
   original post day. `reviewDateRelative` itself is not used for
   aggregation -- it is a redundant human string, not machine-parseable to
   an absolute date.

9. rating is a clean int 1-5, zero nulls. reviewText is null for 28.0% of
   raw rows -- these are legitimate rating-only reviews (Google allows
   posting a star rating with no text), not a data error; sentiment/text
   logic must treat them as "no text", not crash or coerce to empty string
   silently. Post-load (after the three dedup passes, 6,209 -> 6,087 rows)
   the figures are: reviewText null 1,700 (27.9%), ownerResponseText null
   604 -- a 90.5% owner reply rate.

10. Time-series caveat beyond note 8: the own_crawler_crawler rows carry the
   *scrape-batch* date rather than the review date (123 rows on 2026-06-22
   alone), and 12 of 25 sites have no rows at all before 2026-04. Monthly
   volume therefore tracks scrape coverage as much as review activity. See
   docs/dataset_schema.md section 9.
"""

from __future__ import annotations

import os

import pandas as pd

try:  # pragma: no cover - optional dependency for st.cache_data only
    import streamlit as st

    _cache_data = st.cache_data
except Exception:  # pragma: no cover - keep module usable without streamlit
    def _cache_data(func=None, **_kwargs):
        if func is not None:
            return func

        def _decorator(f):
            return f

        return _decorator


# ---------------------------------------------------------------------------
# Keep text columns on object dtype (pandas >= 3 defaults to Arrow-backed
# strings). This is not a style preference: with Arrow strings, the hash
# kernels behind groupby/value_counts run inside libarrow, and on macOS/arm64
# libarrow's mimalloc allocator segfaults (SIGSEGV in mi_thread_init) when it
# is first touched from one of Streamlit's script-runner worker threads --
# which is every rerun of this app. Object dtype keeps that work in NumPy and
# matches the dtypes the rest of this module was written against. Wrapped in
# a try/except so it is a no-op on pandas versions without the option.
# ---------------------------------------------------------------------------
try:
    pd.set_option("future.infer_string", False)
except Exception:  # pragma: no cover - option absent on older pandas
    pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "final_reviews.csv"))


# ---------------------------------------------------------------------------
# Logical field name -> actual CSV column name
# Downstream pages should import these constants instead of hardcoding
# column-name strings.
# ---------------------------------------------------------------------------

SITE_COL = "site"                              # per-location friendly label -- THE location key
BUSINESS_COL = "businessName"                  # constant across the whole dataset, not a location dim
ADDRESS_COL = "address"                        # maps 1:1 with site
CITY_COL = "city"                              # NOT unique per site (some cities host 2-3 sites)
STATE_COL = "state"                            # normalized to full state name by load_data()
POSTAL_COL = "postalCode"
PLACE_ID_COL = "placeId"                       # null for 5/25 sites (own_crawler source, see note 3)
CATEGORY_COL = "category"                      # stable per site; 2 distinct values overall
BUSINESS_AVG_RATING_COL = "businessAvgRating"  # null for the same 5 sites as placeId
BUSINESS_REVIEW_COUNT_COL = "businessReviewCount"
REVIEW_ID_COL = "reviewId"                     # null for 125 rows; NOT globally unique-safe w/o dropna
REVIEWER_NAME_COL = "reviewerName"
RATING_COL = "rating"                          # clean int 1-5, zero nulls
TEXT_COL = "reviewText"                        # null for 28.0% of rows (rating-only reviews)
DATE_COL = "reviewDate"                        # parsed to tz-aware UTC datetime64 by load_data()
DATE_RELATIVE_COL = "reviewDateRelative"       # human string e.g. "3 years ago" -- NOT for aggregation
LANGUAGE_COL = "language"
IS_LOCAL_GUIDE_COL = "isLocalGuide"
REVIEWER_REVIEW_COUNT_COL = "reviewerReviewCount"
LIKES_COL = "likesCount"
OWNER_RESPONSE_DATE_COL = "ownerResponseDate"  # parsed to tz-aware UTC datetime64 by load_data()
OWNER_RESPONSE_TEXT_COL = "ownerResponseText"  # null for 11.2% of rows -> 88.8% owner-reply rate
REVIEW_URL_COL = "reviewUrl"
DATA_SOURCE_COL = "dataSource"                 # 'v1_v2_scrape' | 'own_crawler_api' | 'own_crawler_crawler'

# Columns that together describe a location. `site` alone is the
# collision-free key (see note 2); the rest are descriptive attributes of
# that site included here for display/grouping convenience.
LOCATION_COLS = [SITE_COL, ADDRESS_COL, CITY_COL, STATE_COL, POSTAL_COL, PLACE_ID_COL]
LOCATION_KEY_COL = SITE_COL

# Logical-name -> actual-column-name map, for callers that prefer dict access
# over importing individual constants.
FIELD_MAP = {
    "rating": RATING_COL,
    "text": TEXT_COL,
    "date": DATE_COL,
    "date_relative": DATE_RELATIVE_COL,
    "site": SITE_COL,
    "business": BUSINESS_COL,
    "address": ADDRESS_COL,
    "city": CITY_COL,
    "state": STATE_COL,
    "postal_code": POSTAL_COL,
    "place_id": PLACE_ID_COL,
    "category": CATEGORY_COL,
    "review_id": REVIEW_ID_COL,
    "reviewer_name": REVIEWER_NAME_COL,
    "language": LANGUAGE_COL,
    "is_local_guide": IS_LOCAL_GUIDE_COL,
    "reviewer_review_count": REVIEWER_REVIEW_COUNT_COL,
    "likes_count": LIKES_COL,
    "owner_response_date": OWNER_RESPONSE_DATE_COL,
    "owner_response_text": OWNER_RESPONSE_TEXT_COL,
    "review_url": REVIEW_URL_COL,
    "data_source": DATA_SOURCE_COL,
    "business_avg_rating": BUSINESS_AVG_RATING_COL,
    "business_review_count": BUSINESS_REVIEW_COUNT_COL,
}

# Observed abbreviation -> full-name mapping (see note 4). Built only from
# abbreviations actually observed in the data -- do not extend speculatively.
STATE_ABBR_TO_FULL = {
    "FL": "Florida",
    "GA": "Georgia",
    "SC": "South Carolina",
    "AL": "Alabama",
}

# Columns stripped of leading/trailing whitespace during cleaning.
_STRIP_COLS = [
    SITE_COL, BUSINESS_COL, ADDRESS_COL, CITY_COL, STATE_COL, POSTAL_COL,
    PLACE_ID_COL, CATEGORY_COL, REVIEWER_NAME_COL, TEXT_COL,
    OWNER_RESPONSE_TEXT_COL, LANGUAGE_COL, DATA_SOURCE_COL,
]

# Natural key used to catch cross-pipeline duplicate reviews that don't share
# a reviewId (see note 6). reviewText is intentionally included -- it is the
# strongest signal that two rows describe the *same* review, not just the
# same reviewer posting twice on the same day.
_NATURAL_KEY_COLS = [SITE_COL, REVIEWER_NAME_COL, DATE_COL, RATING_COL, TEXT_COL]

# Date-blind key for the cross-scrape pass (see _dedupe_cross_scrape): the two
# own_crawler pipelines stamped the same review with different days, so the
# date has to be left out of the comparison for those copies to be caught.
_CROSS_SCRAPE_KEY = [SITE_COL, REVIEWER_NAME_COL, RATING_COL]
_RATING_ONLY_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in _STRIP_COLS:
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df


def _normalize_state(df: pd.DataFrame) -> pd.DataFrame:
    if STATE_COL in df.columns:
        df[STATE_COL] = df[STATE_COL].replace(STATE_ABBR_TO_FULL)
    return df


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # format="mixed" is required: reviewDate/ownerResponseDate mix full
    # timestamps with date-only strings; a naive to_datetime call silently
    # NaTs the date-only rows (see note 7).
    for col in (DATE_COL, OWNER_RESPONSE_DATE_COL):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if RATING_COL in df.columns:
        df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce").astype("Int64")
    if IS_LOCAL_GUIDE_COL in df.columns:
        df[IS_LOCAL_GUIDE_COL] = df[IS_LOCAL_GUIDE_COL].map(
            {"true": True, "True": True, "false": False, "False": False}
        ).astype("boolean")
    for col in (REVIEWER_REVIEW_COUNT_COL, LIKES_COL, BUSINESS_REVIEW_COUNT_COL):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # businessAvgRating is Google's own aggregate rating for the site. It was
    # documented as str->float but never actually coerced, so it stayed an
    # object column and any arithmetic on it (e.g. a review-count-weighted
    # chain rating) either raised or silently produced garbage. Float64
    # (nullable) because it is null for the 5 own_crawler sites.
    if BUSINESS_AVG_RATING_COL in df.columns:
        df[BUSINESS_AVG_RATING_COL] = pd.to_numeric(
            df[BUSINESS_AVG_RATING_COL], errors="coerce"
        ).astype("Float64")
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate reviews without discarding legitimate null-reviewId rows.

    Two passes, in order:
      1. reviewId dedup, restricted to non-null reviewId rows only. Doing
         this on the full column (including nulls) is a documented trap
         (see note 5): pandas treats NaN == NaN for duplicate detection, so
         a naive drop_duplicates would delete 124 distinct, real reviews
         that merely share a missing reviewId.
      2. Natural-key dedup (site, reviewerName, reviewDate, rating,
         reviewText) to catch the one confirmed cross-pipeline duplicate
         (note 6): same review scraped once via own_crawler_api (null
         reviewId) and once via own_crawler_crawler (real reviewId). Sort so
         the row WITH a reviewId sorts first, then keep='first'.
    """
    df = df.reset_index(drop=True)

    has_id = df[REVIEW_ID_COL].notna()
    id_dupe_mask = pd.Series(False, index=df.index)
    if has_id.any():
        id_dupe_mask.loc[has_id] = df.loc[has_id, REVIEW_ID_COL].duplicated(keep="first")
    df = df.loc[~id_dupe_mask]

    # Prefer rows that have a reviewId when a natural-key collision occurs.
    df = df.sort_values(by=REVIEW_ID_COL, key=lambda s: s.isna(), kind="stable")
    df = df.drop_duplicates(subset=_NATURAL_KEY_COLS, keep="first")

    df = _dedupe_cross_scrape(df)

    return df.sort_index().reset_index(drop=True)


def _dedupe_cross_scrape(df: pd.DataFrame) -> pd.DataFrame:
    """Third pass: one review captured by two scrapes on *different days*.

    The natural key includes `reviewDate`, so it only catches copies that share
    a timestamp. The own_crawler_api and own_crawler_crawler pipelines ran a day
    apart and stamped their own dates, so 124 rows survived it as visible
    duplicates -- the same reviewer, site, rating and word-for-word text
    rendering twice in the review list.

    Two rules, because the evidence differs:

    - **With text:** identical (site, reviewer, rating, text) is the same
      review, whatever the dates say. Google allows one review per person per
      place, and word-for-word repetition of a paragraph by the same named
      reviewer at the same site is not a coincidence.
    - **Rating-only:** there is no text to match on, so this only collapses
      copies within a week of each other (38 of the 43 candidates are 0-3 days
      apart -- scrape overlap). Pairs months apart are left alone rather than
      guessed at.
    """
    text = df[TEXT_COL].fillna("").str.strip().str.lower()
    has_text = text.str.len() > 0
    drop = pd.Series(False, index=df.index)

    keyed = df.loc[has_text, _CROSS_SCRAPE_KEY].assign(_text=text[has_text])
    drop.loc[has_text] = keyed.duplicated(keep="first")

    # Rating-only copies: same reviewer/site/rating within a week. Which copy
    # survives matters -- the own_crawler_api row is date-only (so it parses to
    # midnight and always sorts first) but carries no reviewId, no placeId and
    # no ownerResponseText, while the v1_v2_scrape row of the same day carries
    # all three. Keeping the earliest silently discarded 29 owner replies and
    # moved the all-time reply rate from 90.55% to 90.08%. Rank each group by
    # (has a reviewId, has an owner reply) first and keep that row, using the
    # date only to decide which rows are the same review.
    rating_only = df.loc[~has_text, [*_CROSS_SCRAPE_KEY, DATE_COL]].copy()
    rating_only["_richer"] = (df.loc[~has_text, REVIEW_ID_COL].notna().astype(int)
                              + df.loc[~has_text, OWNER_RESPONSE_TEXT_COL].notna().astype(int))
    rating_only = rating_only.sort_values(DATE_COL, kind="mergesort")

    prev_date = rating_only.groupby(_CROSS_SCRAPE_KEY, observed=True)[DATE_COL].shift(1)
    same_review = (rating_only[DATE_COL] - prev_date).dt.days.le(_RATING_ONLY_WINDOW_DAYS)
    # Number each run of same-review rows, counting *within* the key: a global
    # cumsum would be incremented by every other reviewer's rows in between, so
    # a pair's two rows landed in different runs and neither was deduped.
    starts = (~same_review.fillna(False)).astype(int)
    run_id = starts.groupby([rating_only[c] for c in _CROSS_SCRAPE_KEY], observed=True).cumsum()
    keep = (rating_only.assign(_run=run_id)
            .sort_values("_richer", ascending=False, kind="mergesort")
            .drop_duplicates(subset=[*_CROSS_SCRAPE_KEY, "_run"], keep="first").index)
    drop.loc[rating_only.index.difference(keep)] = True

    return df.loc[~drop]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@_cache_data(show_spinner="Loading reviews...")
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """Read the raw CSV and return one clean DataFrame -- the single source
    of truth for every page.

    Cleaning applied (see module docstring for the evidence behind each
    step): whitespace-strip key text/categorical columns, parse
    reviewDate/ownerResponseDate to tz-aware UTC datetimes (mixed-format
    safe), coerce rating/reviewerReviewCount/likesCount to nullable Int64,
    coerce isLocalGuide to nullable boolean, normalize `state` to full
    names, and dedupe (reviewId among non-null values, then natural-key for
    the one cross-pipeline duplicate pattern found in this dataset).

    This is a plain function decorated with `st.cache_data` when Streamlit
    is importable (falls back to a no-op decorator otherwise), so it is
    still directly callable/testable from plain Python.
    """
    df = pd.read_csv(path, dtype=str)
    df = _strip_whitespace(df)
    df = _normalize_state(df)
    df = _parse_dates(df)
    df = _coerce_types(df)
    df = _dedupe(df)
    return df


# Alias matching the field-name the original task brief asked for
# (`load_reviews(csv_path) -> pd.DataFrame`); identical behavior to
# load_data, kept so downstream code can use either name.
def load_reviews(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Alias for load_data(path=csv_path). See load_data for details."""
    return load_data(path=csv_path)


def get_filter_options(df: pd.DataFrame) -> dict:
    """Return choices for filter widgets."""
    date_series = df[DATE_COL].dropna()
    return {
        "sites": sorted(df[SITE_COL].dropna().unique().tolist()),
        "cities": sorted(df[CITY_COL].dropna().unique().tolist()),
        "states": sorted(df[STATE_COL].dropna().unique().tolist()),
        "categories": sorted(df[CATEGORY_COL].dropna().unique().tolist()),
        "date_min": date_series.min().date() if not date_series.empty else None,
        "date_max": date_series.max().date() if not date_series.empty else None,
        "rating_min": 1,
        "rating_max": 5,
    }


def filter_data(
    df: pd.DataFrame,
    date_range=None,
    sites=None,
    cities=None,
    states=None,
    categories=None,
    rating_range=None,
) -> pd.DataFrame:
    """Apply all given filters (None/empty = no-op for that filter).

    Returns a filtered copy; never mutates df in place.
    """
    out = df.copy()

    if date_range:
        start, end = date_range
        if start is not None:
            out = out[out[DATE_COL] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            # inclusive end-of-day
            end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            out = out[out[DATE_COL] < end_ts]

    if sites:
        out = out[out[SITE_COL].isin(sites)]
    if cities:
        out = out[out[CITY_COL].isin(cities)]
    if states:
        out = out[out[STATE_COL].isin(states)]
    if categories:
        out = out[out[CATEGORY_COL].isin(categories)]
    if rating_range:
        lo, hi = rating_range
        out = out[out[RATING_COL].between(lo, hi)]

    return out.reset_index(drop=True)


def get_kpis(df: pd.DataFrame) -> dict:
    """Summary numbers for landing-page tiles.

    Two different "average rating" numbers are returned on purpose, because
    the naive one is misleading and was previously the only one shown:

    - `avg_rating` — the mean over the scraped review rows in `df`. This is a
      *sample* mean and it silently weights each site by how many of its
      reviews we happened to scrape, not by how many it actually has. Rome
      contributes 912 rows, Muscle Shoals 17; Academy's 105 scraped reviews
      average a suspicious 5.00 against no Google baseline at all.
    - `avg_rating_weighted` — Google's own per-site rating weighted by
      Google's own per-site review count, i.e. the chain's real customer
      rating across every review that exists, not just the ones in this file.
      Available for the 20 sites that carry `businessAvgRating` (the 5
      own_crawler sites have none — see docs/dataset_schema.md §3).

    `avg_rating_display` is the one to put on a tile: the weighted figure
    when it can be computed, else the sample mean.
    """
    n = len(df)
    if n == 0:
        return {
            "total_reviews": 0, "avg_rating": None, "total_sites": 0,
            "pct_5_star": None, "pct_with_owner_response": None,
            "total_cities": 0, "avg_rating_weighted": None,
            "avg_rating_display": None, "google_review_total": 0,
            "coverage_pct": None, "sites_without_google_rating": 0,
        }

    sample_avg = float(df[RATING_COL].mean())

    per_site = df.groupby(SITE_COL, as_index=False).agg(
        g_rating=(BUSINESS_AVG_RATING_COL, "first"),
        g_count=(BUSINESS_REVIEW_COUNT_COL, "first"),
    )
    rated = per_site.dropna(subset=["g_rating", "g_count"])
    if not rated.empty and float(rated["g_count"].sum()) > 0:
        weighted = float((rated["g_rating"] * rated["g_count"]).sum() / rated["g_count"].sum())
        google_total = int(rated["g_count"].sum())
        # Coverage must compare like with like: `google_total` only covers the
        # sites that publish a count, so the numerator has to exclude rows from
        # the sites that do not (it read 41.5% against a true 39.1%).
        covered = df[SITE_COL].isin(rated[SITE_COL])
        scraped_in_scope = int(covered.sum())
    else:
        weighted, google_total, scraped_in_scope = None, 0, 0

    return {
        "total_reviews": n,
        "avg_rating": sample_avg,
        "avg_rating_weighted": weighted,
        "avg_rating_display": weighted if weighted is not None else sample_avg,
        "google_review_total": google_total,
        "coverage_pct": (float(scraped_in_scope / google_total * 100) if google_total else None),
        "sites_without_google_rating": int(len(per_site) - len(rated)),
        "total_sites": int(df[SITE_COL].nunique()),
        "pct_5_star": float((df[RATING_COL] == 5).mean() * 100),
        "pct_with_owner_response": float(df[OWNER_RESPONSE_TEXT_COL].notna().mean() * 100),
        "total_cities": int(df[CITY_COL].nunique()),
    }


def get_locations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique location (`site`), with review counts and stats.

    Columns: site, address, city, state, postalCode, placeId, category,
    review_count, avg_rating, pct_5_star, pct_with_owner_response,
    first_review_date, last_review_date, google_avg_rating,
    google_review_count, coverage_pct, avg_rating_display, n_1_2_star.

    Rating columns, and which to display (same trap as get_kpis):
    `avg_rating` is the mean of the *scraped* rows only, so it is a sample
    mean whose reliability tracks `coverage_pct` (scraped rows / Google's own
    review count). `google_avg_rating` is Google's published site rating over
    every review the site has. `avg_rating_display` prefers Google's figure
    and falls back to the sample mean for the 5 sites Google data is missing
    for.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            SITE_COL, ADDRESS_COL, CITY_COL, STATE_COL, POSTAL_COL, PLACE_ID_COL,
            CATEGORY_COL, "review_count", "avg_rating", "pct_5_star",
            "pct_with_owner_response", "first_review_date", "last_review_date",
            "google_avg_rating", "google_review_count", "coverage_pct",
            "avg_rating_display", "n_1_2_star",
        ])

    grouped = df.groupby(SITE_COL, as_index=False).agg(
        address=(ADDRESS_COL, "first"),
        city=(CITY_COL, "first"),
        state=(STATE_COL, "first"),
        postalCode=(POSTAL_COL, "first"),
        placeId=(PLACE_ID_COL, "first"),
        category=(CATEGORY_COL, "first"),
        review_count=(RATING_COL, "count"),
        avg_rating=(RATING_COL, "mean"),
        pct_5_star=(RATING_COL, lambda s: (s == 5).mean() * 100),
        n_1_2_star=(RATING_COL, lambda s: int((s <= 2).sum())),
        google_avg_rating=(BUSINESS_AVG_RATING_COL, "first"),
        google_review_count=(BUSINESS_REVIEW_COUNT_COL, "first"),
        first_review_date=(DATE_COL, "min"),
        last_review_date=(DATE_COL, "max"),
    )
    resp_rate = (
        df.groupby(SITE_COL)[OWNER_RESPONSE_TEXT_COL]
        .apply(lambda s: s.notna().mean() * 100)
        .rename("pct_with_owner_response")
    )
    grouped = grouped.merge(resp_rate, left_on=SITE_COL, right_index=True)

    g_count = pd.to_numeric(grouped["google_review_count"], errors="coerce")
    grouped["coverage_pct"] = (grouped["review_count"] / g_count * 100).where(g_count > 0)
    g_rating = pd.to_numeric(grouped["google_avg_rating"], errors="coerce")
    grouped["avg_rating_display"] = g_rating.fillna(
        pd.to_numeric(grouped["avg_rating"], errors="coerce")
    ).astype(float)

    return grouped.sort_values("review_count", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # Self-contained validation against the real CSV. Run directly:
    #   python3 app/utils/data_loader.py
    df = load_data()
    print(f"Loaded rows: {len(df)}")
    print(f"Unique sites: {df[SITE_COL].nunique()}")
    print(f"State values after normalization: {sorted(df[STATE_COL].unique())}")
    print(f"reviewDate dtype: {df[DATE_COL].dtype}, "
          f"range: {df[DATE_COL].min()} -> {df[DATE_COL].max()}")
    print(f"rating dtype: {df[RATING_COL].dtype}, range: "
          f"{df[RATING_COL].min()}-{df[RATING_COL].max()}")
    print()
    print("KPIs:", get_kpis(df))
    print()
    print("Locations (top 5 by review_count):")
    print(get_locations(df).head(5).to_string(index=False))
    print()
    opts = get_filter_options(df)
    print("Filter options:", {k: (v if not isinstance(v, list) else f"{len(v)} values") for k, v in opts.items()})
