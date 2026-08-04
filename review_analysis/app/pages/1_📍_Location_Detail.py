"""Location Detail page.

Features 2, 3, 4 from the requirements: location filters, month/day trend
charts with growth/decline cues, and a searchable review drill-down split
into Positive / Negative / Neutral (VADER) sections, sorted by recency by
default.

Pre-filtered by st.session_state["selected_location"] (a `site` string) set
by a tile click on the landing page (app/Home.py); also usable standalone via
the site/city filter widgets below.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.data_loader import (
    ADDRESS_COL,
    CATEGORY_COL,
    CITY_COL,
    DATE_COL,
    OWNER_RESPONSE_TEXT_COL,
    PLACE_ID_COL,
    RATING_COL,
    REVIEWER_NAME_COL,
    SITE_COL,
    STATE_COL,
    TEXT_COL,
    filter_data,
    load_data,
)
from app.utils.sentiment import add_sentiment_column
from app.utils.time_utils import aggregate_by_day, aggregate_by_month, latest_period_summary

st.set_page_config(page_title="Location Detail", page_icon="📍", layout="wide")

st.page_link("Home.py", label="← Back to dashboard")
st.title("Location Detail")

df = load_data()

# --------------------------------------------------------------------------
# Feature 3: location filters
# --------------------------------------------------------------------------
all_sites = sorted(df[SITE_COL].dropna().unique().tolist())
selected_from_landing = st.session_state.get("selected_location")

with st.expander("Location filters", expanded=True):
    filter_mode = st.radio("Filter by", options=["Site", "City"], horizontal=True, key="loc_filter_mode")

    if filter_mode == "Site":
        default_sites = [selected_from_landing] if selected_from_landing in all_sites else all_sites[:1]
        chosen_sites = st.multiselect("Site(s)", options=all_sites, default=default_sites, key="loc_filter_sites")
        filtered_df = filter_data(df, sites=chosen_sites) if chosen_sites else df.iloc[0:0]
    else:
        all_cities = sorted(df[CITY_COL].dropna().unique().tolist())
        default_city = (
            [df.loc[df[SITE_COL] == selected_from_landing, CITY_COL].iloc[0]]
            if selected_from_landing in all_sites else all_cities[:1]
        )
        chosen_cities = st.multiselect("City(ies)", options=all_cities, default=default_city, key="loc_filter_cities")
        filtered_df = filter_data(df, cities=chosen_cities) if chosen_cities else df.iloc[0:0]

if filtered_df.empty:
    st.info("No location selected (or the selection matched no reviews). Pick a site or city above.")
    st.stop()

st.caption("Showing: " + ", ".join(sorted(filtered_df[SITE_COL].dropna().unique().tolist())))

# --------------------------------------------------------------------------
# Summary stats
# --------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total reviews", f"{len(filtered_df):,}")
col2.metric("Average rating", f"{filtered_df[RATING_COL].mean():.2f}")
col3.metric("5-star share", f"{(filtered_df[RATING_COL] == 5).mean() * 100:.0f}%")
col4.metric("1-2 star share", f"{(filtered_df[RATING_COL] <= 2).mean() * 100:.0f}%")

with st.expander("Location details"):
    meta_cols = [c for c in [ADDRESS_COL, CITY_COL, STATE_COL, CATEGORY_COL, PLACE_ID_COL] if c in filtered_df.columns]
    st.dataframe(
        filtered_df[[SITE_COL, *meta_cols]].drop_duplicates().reset_index(drop=True),
        width="stretch", hide_index=True,
    )

# --------------------------------------------------------------------------
# Feature 2: trends (month/day toggle, growth/decline)
# --------------------------------------------------------------------------
st.subheader("Review Trends")

granularity = st.radio("Granularity", options=["Month", "Day"], index=0, horizontal=True, key="trend_granularity")
agg = aggregate_by_month(filtered_df, DATE_COL, RATING_COL) if granularity == "Month" else aggregate_by_day(filtered_df, DATE_COL, RATING_COL)

if agg.empty:
    st.info("No dated reviews available to build a trend.")
else:
    summary = latest_period_summary(agg)
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.metric(
            label=f"Reviews ({summary['period_label']})",
            value=summary["review_count"],
            delta=f"{summary['pct_change_count']:.1f}% vs prior period" if summary["pct_change_count"] is not None else None,
        )
    with tcol2:
        st.metric(
            label=f"Avg Rating ({summary['period_label']})",
            value=f"{summary['avg_rating']:.2f}" if summary["avg_rating"] is not None else "n/a",
            delta=f"{summary['pct_change_rating']:+.2f} vs prior period" if summary["pct_change_rating"] is not None else None,
        )

    bar_colors = agg["trend_direction"].map(
        {"up": "#0ca30c", "down": "#d03b3b", "flat": "#898781", "n/a": "#898781"}
    ).fillna("#898781")
    fig = go.Figure()
    fig.add_bar(x=agg["period_label"], y=agg["review_count"], name="Review Count", marker_color=bar_colors, yaxis="y1")
    fig.add_trace(go.Scatter(x=agg["period_label"], y=agg["avg_rating"], name="Avg Rating",
                              mode="lines+markers", line=dict(color="#256abf", width=2), yaxis="y2"))
    fig.update_layout(
        xaxis=dict(title="Period", type="category"),
        yaxis=dict(title="Review Count", side="left"),
        yaxis2=dict(title="Avg Rating", side="right", overlaying="y", range=[1, 5]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fcfcfb", paper_bgcolor="#fcfcfb",
        margin=dict(l=10, r=10, t=40, b=10), height=420,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Show underlying aggregate table"):
        st.dataframe(
            agg[["period_label", "review_count", "avg_rating", "pct_change_count", "pct_change_rating", "trend_direction"]],
            width="stretch", hide_index=True,
        )

# --------------------------------------------------------------------------
# Feature 4: review list / drilldown with VADER sentiment
# --------------------------------------------------------------------------
st.subheader("Reviews")

search_col, sort_col = st.columns([3, 1])
with search_col:
    keyword = st.text_input("Search review text", value="", placeholder="e.g. clean, friendly, wait time...", key="review_search_keyword")
with sort_col:
    sort_choice = st.selectbox("Sort by", options=["Most recent", "Oldest first", "Highest rating", "Lowest rating"], index=0, key="review_sort_choice")

reviews_df = filtered_df.copy()
if keyword.strip():
    reviews_df = reviews_df[reviews_df[TEXT_COL].fillna("").str.contains(keyword, case=False, na=False)]

sort_map = {
    "Most recent": (DATE_COL, False),
    "Oldest first": (DATE_COL, True),
    "Highest rating": (RATING_COL, False),
    "Lowest rating": (RATING_COL, True),
}
sort_col_name, sort_ascending = sort_map[sort_choice]
reviews_df = reviews_df.sort_values(sort_col_name, ascending=sort_ascending, na_position="last")

# VADER sentiment (Feature: Sentiment Analysis) -- run on the already
# filtered/searched set, never the full 6,208-row dataset, per the caching
# guidance in docs/architecture.md.
reviews_df = add_sentiment_column(reviews_df, text_col=TEXT_COL, out_col="sentiment")
# "no_text" (rating-only reviews with nothing for VADER to score) folded
# into Neutral -- this UI only has three tabs.
reviews_df["sentiment_bucket"] = reviews_df["sentiment"].replace({"no_text": "neutral"})


def render_review_card(row: pd.Series) -> None:
    rating = int(row[RATING_COL])
    stars = "★" * rating + "☆" * (5 - rating)
    date_val = row.get(DATE_COL)
    date_str = date_val.strftime("%b %d, %Y") if pd.notna(date_val) else "Unknown date"
    reviewer = row.get(REVIEWER_NAME_COL) or "Anonymous"

    st.markdown(f"**{reviewer}** &nbsp;&nbsp; {stars} &nbsp;&nbsp; *{date_str}*")
    text = row.get(TEXT_COL)
    st.write(text) if pd.notna(text) and str(text).strip() else st.caption("(No review text)")

    owner_resp = row.get(OWNER_RESPONSE_TEXT_COL)
    if pd.notna(owner_resp) and str(owner_resp).strip():
        st.markdown(f"> **Owner response:** {owner_resp}")
    st.divider()


tab_pos, tab_neg, tab_neu = st.tabs([
    f"Positive ({(reviews_df['sentiment_bucket'] == 'positive').sum()})",
    f"Negative ({(reviews_df['sentiment_bucket'] == 'negative').sum()})",
    f"Neutral ({(reviews_df['sentiment_bucket'] == 'neutral').sum()})",
])

for tab, label in ((tab_pos, "positive"), (tab_neg, "negative"), (tab_neu, "neutral")):
    with tab:
        subset = reviews_df[reviews_df["sentiment_bucket"] == label]
        if subset.empty:
            st.caption(f"No {label} reviews match the current filters.")
        for _, row in subset.iterrows():
            render_review_card(row)
