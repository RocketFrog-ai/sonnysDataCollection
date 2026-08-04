"""Time aggregation utilities for review trend analysis.

Pure pandas functions -- no Streamlit dependency (Streamlit is only
referenced in type hints/docstrings, never imported).

These functions expect a DataFrame with:
  - a datetime column (already parsed to pandas datetime64; see the
    date-parsing gotcha note below) identified by `date_col`
  - a numeric rating column identified by `rating_col`

DATE PARSING GOTCHA (found while validating against
data/final_reviews.csv, see __main__ block below):
    The raw `reviewDate` column mixes two ISO 8601 shapes:
      - full timestamps: "2026-07-17T15:45:58.308Z"
      - date-only strings: "2026-07-17"
    Calling `pd.to_datetime(col, utc=True, errors="coerce")` WITHOUT an
    explicit format silently turns every date-only row into NaT (~8% of
    rows, 497/6209 in this dataset) because pandas' format inference
    locks onto the first (full-timestamp) shape it sees and rejects the
    shorter ones instead of falling back. Fix: pass
    `format="ISO8601"` (or `format="mixed"`) to `pd.to_datetime`, which
    parses both shapes correctly with zero failures. Agent 2's
    data_loader.py should do this when producing DATE_COL.

    Also note: `wc -l` on the CSV reports 7897 lines, but the actual
    row count is 6209 -- reviewText/ownerResponseText contain embedded
    newlines inside quoted CSV fields, which inflates the raw line
    count. `pd.read_csv` handles this correctly (no bad-line warnings);
    just don't use `wc -l` as a proxy for row count on this file.

Functions in this module assume `date_col` is already a proper
datetime64 dtype (tz-aware or naive, both handled) -- they do not
re-parse strings themselves, since that's data_loader's job.
"""

from __future__ import annotations

import pandas as pd

# Frequency aliases accepted by aggregate_trend / pandas Grouper
FREQ_MONTH = "M"
FREQ_DAY = "D"


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Return a copy of df with date_col guaranteed to be datetime64.

    If the column is not already parsed, falls back to
    format="ISO8601" (see module docstring for why the naive
    pd.to_datetime call is unsafe on this dataset).
    """
    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
        out[date_col] = pd.to_datetime(
            out[date_col], utc=True, errors="coerce", format="ISO8601"
        )
    return out


def aggregate_trend(
    df: pd.DataFrame,
    date_col: str,
    rating_col: str,
    freq: str = "M",
) -> pd.DataFrame:
    """Aggregate reviews into time buckets of arbitrary frequency.

    Parameters
    ----------
    df : DataFrame containing at least date_col and rating_col.
    date_col : name of the datetime column.
    rating_col : name of the numeric rating column.
    freq : pandas offset alias, e.g. "M" (calendar month), "D" (day),
        "W" (week). Defaults to monthly.

    Returns
    -------
    DataFrame sorted by period ascending, with columns:
        period        : pandas Period (or Timestamp) marking the bucket
        period_label   : human-readable string label for display
        review_count  : number of reviews in that period
        avg_rating    : mean rating in that period (NaN if none)
        pct_change_count : % change in review_count vs previous period
        pct_change_rating : percentage-point change in avg_rating vs
                             previous period
        trend_direction   : "up" / "down" / "flat" / "n/a" based on
                             review_count movement vs previous period
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "period_label",
                "review_count",
                "avg_rating",
                "pct_change_count",
                "pct_change_rating",
                "trend_direction",
            ]
        )

    work = _ensure_datetime(df, date_col)
    work = work.dropna(subset=[date_col])

    # tz-aware datetimes need tz-naive Period conversion; drop tz for grouping
    dt_series = work[date_col]
    if isinstance(dt_series.dtype, pd.DatetimeTZDtype):
        dt_series = dt_series.dt.tz_convert("UTC").dt.tz_localize(None)

    periods = dt_series.dt.to_period(freq)
    grouped = work.assign(_period=periods).groupby("_period", observed=True)[rating_col]

    agg = grouped.agg(review_count="count", avg_rating="mean").reset_index()
    agg = agg.rename(columns={"_period": "period"})
    agg = agg.sort_values("period").reset_index(drop=True)

    if freq.upper().startswith("M"):
        agg["period_label"] = agg["period"].dt.strftime("%Y-%m")
    elif freq.upper().startswith("D"):
        agg["period_label"] = agg["period"].dt.strftime("%Y-%m-%d")
    elif freq.upper().startswith("W"):
        agg["period_label"] = agg["period"].astype(str)
    else:
        agg["period_label"] = agg["period"].astype(str)

    agg = compute_growth_indicators(agg)
    return agg


def aggregate_by_month(
    df: pd.DataFrame, date_col: str, rating_col: str
) -> pd.DataFrame:
    """Month-level aggregation: review count and avg rating per calendar month."""
    return aggregate_trend(df, date_col, rating_col, freq=FREQ_MONTH)


def aggregate_by_day(
    df: pd.DataFrame, date_col: str, rating_col: str
) -> pd.DataFrame:
    """Day-level aggregation: review count and avg rating per calendar day."""
    return aggregate_trend(df, date_col, rating_col, freq=FREQ_DAY)


def compute_growth_indicators(agg: pd.DataFrame) -> pd.DataFrame:
    """Add period-over-period growth/decline indicators to an aggregate table.

    Expects `agg` to already be sorted ascending by period and to contain
    `review_count` and `avg_rating` columns (as produced by
    aggregate_trend). Adds/overwrites:
        pct_change_count  : % change in review_count vs previous period
                             (NaN for the first period, inf-safe: 0-count
                             previous periods produce NaN rather than inf)
        pct_change_rating : percentage-point change in avg_rating vs
                             previous period
        trend_direction   : "up" / "down" / "flat" / "n/a"
    """
    out = agg.copy()
    prev_count = out["review_count"].shift(1)
    pct = (out["review_count"] - prev_count) / prev_count * 100
    out["pct_change_count"] = pct.replace([float("inf"), float("-inf")], pd.NA)

    out["pct_change_rating"] = out["avg_rating"] - out["avg_rating"].shift(1)

    def _direction(row) -> str:
        if pd.isna(row["pct_change_count"]):
            return "n/a"
        if row["pct_change_count"] > 0:
            return "up"
        if row["pct_change_count"] < 0:
            return "down"
        return "flat"

    out["trend_direction"] = out.apply(_direction, axis=1)
    return out


def latest_period_summary(agg: pd.DataFrame) -> dict:
    """Convenience helper: pull out the most recent period's stats as a dict.

    Useful for rendering a headline metric (st.metric) with a delta.
    Returns {} if agg is empty.
    """
    if agg.empty:
        return {}
    last = agg.iloc[-1]
    return {
        "period_label": last["period_label"],
        "review_count": int(last["review_count"]),
        "avg_rating": float(last["avg_rating"]) if pd.notna(last["avg_rating"]) else None,
        "pct_change_count": (
            float(last["pct_change_count"]) if pd.notna(last["pct_change_count"]) else None
        ),
        "pct_change_rating": (
            float(last["pct_change_rating"]) if pd.notna(last["pct_change_rating"]) else None
        ),
        "trend_direction": last["trend_direction"],
    }


if __name__ == "__main__":
    # Self-contained validation against the real CSV. Run directly:
    #   python3 app/utils/time_utils.py
    # Does NOT depend on data_loader.py (parses reviewDate itself).
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "..", "..", "data", "final_reviews.csv")

    raw = pd.read_csv(csv_path)
    print(f"raw rows read by pandas: {len(raw)}")

    raw["reviewDate"] = pd.to_datetime(
        raw["reviewDate"], utc=True, errors="coerce", format="ISO8601"
    )
    print(f"rows with unparseable reviewDate: {raw['reviewDate'].isna().sum()}")
    print(f"date range: {raw['reviewDate'].min()} -> {raw['reviewDate'].max()}")

    monthly = aggregate_by_month(raw, "reviewDate", "rating")
    print("\n--- monthly aggregate (tail) ---")
    print(
        monthly[
            [
                "period_label",
                "review_count",
                "avg_rating",
                "pct_change_count",
                "pct_change_rating",
                "trend_direction",
            ]
        ].tail(8).to_string(index=False)
    )
    print(f"\nmonthly buckets total: {len(monthly)}")
    print(f"sum of review_count across months: {monthly['review_count'].sum()} "
          f"(should equal parseable row count: {raw['reviewDate'].notna().sum()})")

    daily = aggregate_by_day(raw, "reviewDate", "rating")
    print("\n--- daily aggregate (tail) ---")
    print(
        daily[["period_label", "review_count", "avg_rating", "trend_direction"]]
        .tail(5)
        .to_string(index=False)
    )
    print(f"\ndaily buckets total: {len(daily)}")

    print("\n--- latest_period_summary (monthly) ---")
    print(latest_period_summary(monthly))
