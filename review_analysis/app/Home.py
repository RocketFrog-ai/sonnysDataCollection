"""Review Analytics — landing dashboard (entrypoint).

Run with: streamlit run app/Home.py

Laid out to match the product dashboard: fixed top bar + icon rail (drawn by
app/utils/theme.py), page heading, an "All Sites" tab strip, a location /
period filter row, and a 3x3 grid of KPI tiles. Every tile is one click
target that drills through to the site breakdown.

NOTE: this file is intentionally NOT named app.py. `app/` is also a Python
package (it contains app/utils/), and naming the entrypoint script the same
as the package it lives in creates a real import collision under
`streamlit run` (Streamlit registers the running script as module `app`,
shadowing the `app` package, so `from app.utils... import` fails with
"'app' is not a package"). Keeping the entrypoint filename distinct from
the package name avoids this.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.utils import reviews_ui as RU
from app.utils import theme as T
from app.utils.data_loader import SITE_COL, get_kpis
from app.utils.metrics import (
    MONTH_COL,
    PERIOD_WINDOWS,
    load_scored_data,
    monthly_series,
    period_metrics,
    site_table,
    window_frames,
)

DETAIL_PAGE = "pages/1_📍_Location_Detail.py"
INSIGHTS_PAGE = "pages/2_📊_Insights.py"
SESSION_SITES = "selected_sites"
SESSION_PERIOD = "selected_period"

T.setup_page("Review Analytics")

RATING_MINI_COLORS = {5: T.MINT, 4: T.CREAM, 3: T.YELLOW, 2: T.PURPLE, 1: T.CORAL}


def drill_to(page: str = DETAIL_PAGE, *, sort: str | None = None, asc: bool = False,
             open_site: str | None = None, open_period: str | None = None,
             review_sort: str | None = None, review_filter: str | None = None) -> None:
    """Open a drill-down already set up to answer the tile's question.

    Every tile lands somewhere specific rather than dumping the reader on a
    default table: the negative-reviews tile opens the worst site with its
    reviews filtered to negative and sorted most-negative-first, the response
    tile sorts by reply rate ascending, and so on. The dashboard's period
    selection travels with the click so the drill-down shows the same window.
    """
    st.session_state["detail_period"] = period_choice
    st.session_state["ins_period"] = period_choice

    if page == DETAIL_PAGE:
        if sort:
            st.session_state["table_sort_col"] = sort
            st.session_state["table_sort_asc"] = asc
        st.session_state["open_site"] = open_site
        if open_site:
            k = RU.safe_key(open_site)
            if review_sort:
                st.session_state[f"sort_{k}"] = review_sort
            if review_filter:
                st.session_state[f"filt_{k}"] = review_filter
    else:
        if sort:
            st.session_state["ins_sort_col"] = sort
            st.session_state["ins_sort_asc"] = asc
        st.session_state["ins_open_period"] = open_period
    st.switch_page(page)


# ---------------------------------------------------------------------------
# Heading + tabs
# ---------------------------------------------------------------------------
df_all = load_scored_data()
all_sites = sorted(df_all[SITE_COL].dropna().unique().tolist())

T.page_heading("Reviews", "Big Dan's Car Wash")
T.tab_strip([
    ("All Sites", True, None),
    ("Insights", False, INSIGHTS_PAGE),
    ("Site Breakdown", False, DETAIL_PAGE),
])

# ---------------------------------------------------------------------------
# Filter row — location popover on the left, period pill on the right
# ---------------------------------------------------------------------------
chosen_sites = st.session_state.get(SESSION_SITES) or all_sites
chosen_sites = [s for s in chosen_sites if s in all_sites] or all_sites

period_choice = st.session_state.get(SESSION_PERIOD, "Current month")

left, spacer, right = st.columns([2.4, 6.4, 1.7])
with left:
    with st.container(key="qfilter_loc"):
        with st.popover(f"Location ({len(chosen_sites)})"):
            picked = st.multiselect(
                "Sites", options=all_sites, default=chosen_sites,
                key="home_site_filter", label_visibility="collapsed",
            )
            if picked and picked != chosen_sites:
                st.session_state[SESSION_SITES] = picked
                st.rerun()
            elif not picked:
                st.caption("Nothing selected — showing all 25 locations.")
with right:
    with st.container(key="qfilter_period"):
        with st.popover(f"🗓  {period_choice}"):
            chosen_period = st.radio(
                "Period", options=list(PERIOD_WINDOWS),
                index=list(PERIOD_WINDOWS).index(period_choice),
                key="home_period_radio", label_visibility="collapsed",
            )
            if chosen_period != period_choice:
                st.session_state[SESSION_PERIOD] = chosen_period
                st.rerun()

df = df_all[df_all[SITE_COL].isin(chosen_sites)]
cur, prior, window_label, caption = window_frames(df, period_choice)

m_cur, m_prior = period_metrics(cur), period_metrics(prior)
monthly = monthly_series(df, months=8)
kpis = get_kpis(df)

st.markdown(
    f'<div class="q-note" style="margin:-6px 0 14px;">Showing <b>{window_label}</b> · '
    f'{len(chosen_sites)} of {len(all_sites)} locations · {m_cur["n_reviews"]:,} reviews in period '
    f'· {len(df):,} reviews in current selection</div>',
    unsafe_allow_html=True,
)

if m_cur["n_reviews"] == 0:
    st.warning("No reviews in the selected period for these locations.")
    st.stop()


def delta(cur_val, prior_val, higher_is_better: bool = True) -> str:
    return T.delta_html(T.pct_change(cur_val, prior_val), caption, higher_is_better)


# ---------------------------------------------------------------------------
# Tile definitions — 3 x 3, in reading order
# ---------------------------------------------------------------------------
days_in_window = max(1, (cur["reviewDate"].max() - cur["reviewDate"].min()).days + 1)
rating_mix = [(m_cur["rating_counts"][r], RATING_MINI_COLORS[r]) for r in (5, 4, 3, 2, 1)]
month_counts = monthly["n_reviews"].tolist() if not monthly.empty else [0]
month_neg = [
    (p or 0) / 100 * n for p, n in zip(monthly.get("pct_negative", []), monthly.get("n_reviews", []))
] if not monthly.empty else [0]
month_pos = [
    (p or 0) / 100 * n for p, n in zip(monthly.get("pct_positive", []), monthly.get("n_reviews", []))
] if not monthly.empty else [0]

# Best / worst location, over sites with enough scored reviews to mean
# anything. Without the floor a site with 13 reviews and no complaints wins
# "best sentiment" at a perfect 100% every single month, which is noise, not
# a leaderboard. If no site clears the bar (short windows, few locations
# selected), fall back to the largest sample available.
MIN_SCORED_FOR_RANK = 20
sites_tbl = site_table(cur)
eligible = sites_tbl[sites_tbl["n_scored"] >= MIN_SCORED_FOR_RANK]
if len(eligible) < 3 and not sites_tbl.empty:
    # A one-month window rarely gives 25 sites 20 scored reviews each. Rather
    # than rank on a single qualifying site (which would make "best" and
    # "needs attention" the same tile), fall back to the largest samples
    # available so the two ends are at least different sites.
    eligible = sites_tbl.nlargest(min(3, len(sites_tbl)), "n_scored")
ranked = eligible.sort_values("net_sentiment")
worst = ranked.iloc[0] if not ranked.empty else None
best = ranked.iloc[-1] if len(ranked) > 1 else None
chain_net = m_cur["net_sentiment"]


def gap_line(row) -> str:
    """'x.x pts above/below the chain average' for the leaderboard tiles."""
    if row is None or chain_net is None or pd.isna(row["net_sentiment"]):
        return T.delta_html(None, "net sentiment")
    gap = row["net_sentiment"] - chain_net
    color = T.UP if gap >= 0 else T.DOWN
    word = "above" if gap >= 0 else "below"
    return (f'<div class="qcard-delta"><b style="color:{color}">{abs(gap):.1f} pts</b>'
            f'&nbsp; {word} the chain average</div>')

resp_hours = m_cur["median_response_hours"]
resp_str = (f"{resp_hours:.0f}h" if resp_hours is not None and resp_hours < 72
            else (f"{resp_hours / 24:.1f}d" if resp_hours is not None else "--"))

TILES = [
    dict(
        key="reviews",
        title=f"New Reviews ({window_label})",
        value=T.fmt_compact(m_cur["n_reviews"]),
        delta=delta(m_cur["n_reviews"], m_prior["n_reviews"]),
        subs=[(f"{m_cur['n_sites']}", "Locations reporting"),
              (f"{m_cur['n_reviews'] / days_in_window:.1f}", "Reviews per day")],
        chart=T.mini_stack(rating_mix),
    ),
    dict(
        key="rating",
        title="Average Rating",
        value=f"{m_cur['avg_rating']:.2f}",
        delta=delta(m_cur["avg_rating"], m_prior["avg_rating"]),
        # The tile's big number is this period's scraped mean; the sub-stat is
        # Google's own rating weighted by each site's true review count, which
        # is the chain's real customer rating (see data_loader.get_kpis).
        subs=[(f"{kpis['avg_rating_display']:.2f} ★", "Chain rating, all reviews"),
              (f"{m_cur['pct_5_star']:.0f}%", "5-star share")],
        # Truncated scale (4.0-5.0): every month sits above 4.4, so a
        # zero-based mini chart would be five identical bars.
        chart=T.mini_bars(monthly["avg_rating"].tolist() if not monthly.empty else [0],
                          color=T.PURPLE, baseline=4.0),
    ),
    dict(
        key="sentiment",
        title="Review Sentiment",
        value=(f"{m_cur['net_sentiment']:.2f}%" if m_cur["net_sentiment"] is not None else "--"),
        delta=delta(m_cur["net_sentiment"], m_prior["net_sentiment"]),
        subs=[(f"{m_cur['positive']:,} / {m_cur['negative']:,}", "Positive / negative"),
              (f"{m_cur['avg_score']:+.2f}" if m_cur["avg_score"] is not None else "--",
               "Avg VADER score")],
        chart=T.mini_stack([(m_cur["positive"], T.POS_COLOR),
                            (m_cur["neutral"], T.NEU_COLOR),
                            (m_cur["negative"], T.NEG_COLOR)]),
    ),
    dict(
        key="positive",
        title="Positive Reviews",
        value=T.fmt_compact(m_cur["positive"]),
        delta=delta(m_cur["positive"], m_prior["positive"]),
        subs=[(f"{m_cur['pct_positive']:.1f}%" if m_cur["pct_positive"] is not None else "--",
               "Of reviews with text"),
              (f"{m_cur['no_text']:,}", "Rating-only, unscored")],
        chart=T.mini_bars(month_pos, color=T.POS_COLOR),
    ),
    dict(
        key="negative",
        title="Negative Reviews",
        value=T.fmt_compact(m_cur["negative"]),
        delta=delta(m_cur["negative"], m_prior["negative"], higher_is_better=False),
        subs=[(f"{m_cur['n_1_2_star']:,}", "1–2 star reviews"),
              (f"{m_cur['pct_negative']:.1f}%" if m_cur["pct_negative"] is not None else "--",
               "Of reviews with text")],
        chart=T.mini_bars(month_neg, color=T.NEG_COLOR),
    ),
    dict(
        key="response",
        title="Owner Response Rate",
        value=f"{m_cur['pct_response']:.2f}%",
        delta=delta(m_cur["pct_response"], m_prior["pct_response"]),
        subs=[(resp_str, "Median time to reply"),
              (f"{m_cur['n_reviews'] - round(m_cur['n_reviews'] * m_cur['pct_response'] / 100):,}",
               "Awaiting a reply")],
        chart=T.mini_stack([(m_cur["pct_response"], T.MINT),
                            (100 - m_cur["pct_response"], T.CORAL)]),
    ),
    dict(
        key="volume",
        title="Review Volume Trend",
        value=T.fmt_compact(sum(month_counts)),
        delta=T.delta_html(
            T.pct_change(month_counts[-1] if month_counts else None,
                         month_counts[-2] if len(month_counts) > 1 else None),
            "latest month vs prior",
        ),
        subs=[(f"{len(monthly)}", "Months shown"),
              (f"{max(month_counts):,.0f}", "Best month")],
        chart=T.mini_bars(month_counts, color=T.BLUE),
    ),
    dict(
        key="best",
        title="Best Sentiment Location",
        value=(f"{best['net_sentiment']:.1f}%" if best is not None else "--"),
        delta=gap_line(best),
        subs=[(str(best["site"]).replace("Big Dan's ", "") if best is not None else "--",
               f"{int(best['n_scored']):,} scored reviews" if best is not None else ""),
              (f"{best['avg_rating']:.2f} ★" if best is not None else "--", "Avg rating")],
        chart=T.mini_stack([(best["positive"], T.POS_COLOR), (best["neutral"], T.NEU_COLOR),
                            (best["negative"], T.NEG_COLOR)] if best is not None else []),
    ),
    dict(
        key="worst",
        title="Needs Attention",
        value=(f"{worst['net_sentiment']:.1f}%" if worst is not None else "--"),
        delta=gap_line(worst),
        subs=[(str(worst["site"]).replace("Big Dan's ", "") if worst is not None else "--",
               f"{int(worst['negative']):,} negative reviews" if worst is not None else ""),
              (f"{worst['avg_rating']:.2f} ★" if worst is not None else "--", "Avg rating")],
        chart=T.mini_stack([(worst["positive"], T.POS_COLOR), (worst["neutral"], T.NEU_COLOR),
                            (worst["negative"], T.NEG_COLOR)] if worst is not None else []),
    ),
]

# Where each tile goes when clicked, and how that view is set up on arrival.
MOST_NEGATIVE = "Sentiment score — most negative"
MOST_POSITIVE = "Sentiment score — most positive"
latest_period = str(cur[MONTH_COL].max()) if not cur.empty else None
best_site = str(best["site"]) if best is not None else None
worst_site = str(worst["site"]) if worst is not None else None

DESTINATIONS = {
    "reviews": (dict(page=INSIGHTS_PAGE, sort="n_reviews", open_period=latest_period),
                "Open this period in Insights"),
    "rating": (dict(sort="avg_rating"), "Rank locations by rating"),
    "sentiment": (dict(sort="net_sentiment"), "Rank locations by net sentiment"),
    "positive": (dict(sort="positive"), "Rank locations by positive reviews"),
    "negative": (dict(sort="negative", open_site=worst_site, review_filter="Negative",
                      review_sort=MOST_NEGATIVE),
                 "Read the negative reviews, worst site first"),
    "response": (dict(sort="pct_response", asc=True), "Locations with the slowest replies first"),
    "volume": (dict(page=INSIGHTS_PAGE, sort="label", open_period=latest_period),
               "Open the month-by-month table"),
    "best": (dict(sort="net_sentiment", open_site=best_site, review_sort=MOST_POSITIVE),
             f"Read {best_site}'s reviews" if best_site else "Open the site breakdown"),
    "worst": (dict(sort="net_sentiment", asc=True, open_site=worst_site,
                   review_filter="Negative", review_sort=MOST_NEGATIVE),
              f"Read {worst_site}'s negative reviews" if worst_site else "Open the site breakdown"),
}

for row_start in range(0, len(TILES), 3):
    cols = st.columns(3, gap="medium")
    for col, tile in zip(cols, TILES[row_start:row_start + 3]):
        with col:
            target, tip = DESTINATIONS.get(tile["key"], ({}, "Open the site breakdown"))
            T.clickable_card(
                tile["key"],
                T.kpi_card_html(tile["title"], tile["value"], tile["delta"],
                                tile["chart"], tile["subs"]),
                on_click=(lambda t=target: drill_to(**t)),
                help_text=tip,
            )
    st.write("")

st.markdown(
    '<div class="q-note" style="margin-top:6px;">Sentiment is VADER compound score on review '
    'text; rating-only reviews are excluded from sentiment shares. Every tile is clickable — it '
    'opens the drill-down that explains it, already sorted and filtered.</div>',
    unsafe_allow_html=True,
)
