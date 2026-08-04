"""Insights page — the period drill-down.

Same shape as the product's drill-down: one chart card with a granularity
selector on top, then a table whose rows are calendar periods. Expanding a
period lists the sites that reported in it; expanding a site lists that
site's reviews for that period, ordered by whichever sort is chosen
(sentiment score, date, rating).

Three levels, one definition of the numbers: a period row and a site row are
both `metrics._row_stats` over a slice, so a month's total is exactly the sum
of what its sites show.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils import reviews_ui as RU
from app.utils import theme as T
from app.utils.data_loader import SITE_COL
from app.utils.metrics import (
    FREQ_CHOICES,
    PERIOD_WINDOWS,
    load_scored_data,
    period_index,
    period_label,
    period_metrics,
    period_table,
    site_table,
    window_frames,
)

SESSION_SITES = "selected_sites"

T.setup_page("Insights", page_icon="📊")

df_all = load_scored_data()
all_sites = sorted(df_all[SITE_COL].dropna().unique().tolist())

T.page_heading("Reviews", "Big Dan's Car Wash")
T.tab_strip([
    ("All Sites", False, "Home.py"),
    ("Insights", True, None),
    ("Site Breakdown", False, "pages/1_📍_Location_Detail.py"),
])

# ---------------------------------------------------------------------------
# Filter row
# ---------------------------------------------------------------------------
chosen = st.session_state.get(SESSION_SITES) or all_sites
chosen = [s for s in chosen if s in all_sites] or all_sites

left, _mid, right = st.columns([2.4, 6.4, 1.7])
with left:
    with st.container(key="qfilter_loc3"):
        with st.popover(f"Location ({len(chosen)})"):
            picked = st.multiselect("Sites", options=all_sites, default=chosen,
                                    key="ins_site_filter", label_visibility="collapsed")
            if picked and picked != chosen:
                st.session_state[SESSION_SITES] = picked
                st.rerun()
with right:
    period_choice = st.session_state.get("ins_period", "Last 12 months")
    with st.container(key="qfilter_period3"):
        with st.popover(f"🗓  {period_choice}"):
            picked_period = st.radio("Period", options=list(PERIOD_WINDOWS),
                                     index=list(PERIOD_WINDOWS).index(period_choice),
                                     key="ins_period_radio", label_visibility="collapsed")
            if picked_period != period_choice:
                st.session_state["ins_period"] = picked_period
                st.rerun()

df = df_all[df_all[SITE_COL].isin(chosen)]
cur, _prior, window_label, _caption = window_frames(df, period_choice)
if cur.empty:
    st.warning("No reviews in the selected period for these locations.")
    st.stop()

m = period_metrics(cur)
net_txt = f' · net sentiment {m["net_sentiment"]:.1f}%' if m["net_sentiment"] is not None else ""
st.markdown(
    f'<div class="q-note" style="margin:-6px 0 16px;">{window_label} · {m["n_reviews"]:,} reviews '
    f'· {len(chosen)} locations{net_txt}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chart card — sentiment split by period, with trend lines
# ---------------------------------------------------------------------------
with T.panel_start("inschart"):
    head_l, head_r = st.columns([6, 1.3])
    with head_l:
        st.markdown('<div class="q-panel-title">Sentiment &amp; Response Rate By Period</div>',
                    unsafe_allow_html=True)
    with head_r:
        granularity = st.selectbox(
            "Granularity", options=list(FREQ_CHOICES),
            index=list(FREQ_CHOICES).index(st.session_state.get("ins_granularity", "Monthly")),
            key="ins_granularity", label_visibility="collapsed",
        )
    rows = period_table(cur, granularity, last_n=14)

    if rows.empty:
        st.info("Not enough dated reviews to plot a trend.")
    else:
        x = rows["label"].tolist()
        pos_pct = [v if pd.notna(v) else None for v in rows["pct_positive"]]
        neg_pct = [v if pd.notna(v) else None for v in rows["pct_negative"]]

        fig = go.Figure()
        fig.add_bar(x=x, y=neg_pct, name="Negative Rate", marker_color=T.ORANGE)
        fig.add_bar(x=x, y=pos_pct, name="Positive Rate", marker_color=T.BLUE)

        def trendline(y_vals, color):
            """Least-squares fit across the plotted periods."""
            ys = pd.Series(y_vals, dtype="float64")
            valid = ys.notna()
            if valid.sum() < 2:
                return
            idx = pd.Series(range(len(ys)), dtype="float64")
            slope, intercept = np.polyfit(idx[valid], ys[valid], 1)
            fig.add_scatter(x=x, y=(slope * idx + intercept).tolist(), mode="lines",
                            line=dict(color=color, width=2, dash="dash"),
                            showlegend=False, hoverinfo="skip")

        trendline(neg_pct, "#f7d3a6")
        trendline(pos_pct, "#c3dcf5")

        fig.update_layout(
            barmode="group", bargap=0.55, bargroupgap=0.03,
            height=420, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=18, b=10), font=dict(color=T.MUTED, size=13),
            xaxis=dict(type="category", showgrid=False, linecolor=T.LINE,
                       tickfont=dict(color=T.INK_SOFT)),
            yaxis=dict(title=dict(text="Percentage", font=dict(color=T.MUTED)), showgrid=True,
                       gridcolor="#eef1f6", zeroline=False, rangemode="tozero"),
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0.02,
                        font=dict(color=T.INK_SOFT, size=13)),
        )
        # A one- or two-period window would otherwise render as a couple of
        # enormous bars, since Plotly divides the plot area between categories.
        if len(x) < 5:
            fig.update_traces(selector=dict(type="bar"), width=0.18)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.write("")

# ---------------------------------------------------------------------------
# Hierarchical table: period -> site -> reviews
# ---------------------------------------------------------------------------
COLUMNS = [
    ("label", "Period", 2.6),
    ("n_reviews", "Reviews (#)", 1.2),
    ("avg_rating", "Avg Rating", 1.2),
    ("positive", "Positive (#)", 1.2),
    ("neutral", "Neutral (#)", 1.1),
    ("negative", "Negative (#)", 1.2),
    ("no_text", "Rating-only (#)", 1.4),
    ("net_sentiment", "Net Sentiment", 1.3),
    ("pct_response", "Response Rate", 1.3),
]
WIDTHS = [w for _, _, w in COLUMNS]

st.session_state.setdefault("ins_sort_col", "label")
st.session_state.setdefault("ins_sort_asc", False)


def row_values(row, name: str, indent: int = 0):
    """Cells for a period row or a site row — identical columns either way."""
    return [
        (name, "name"),
        (f"{int(row['n_reviews']):,}", "num"),
        (f"{row['avg_rating']:.2f} ★", "strong"),
        (f"{int(row['positive']):,}", "num"),
        (f"{int(row['neutral']):,}", "num"),
        (f"{int(row['negative']):,}", "num"),
        (f"{int(row['no_text']):,}", "num"),
        (f"{row['net_sentiment']:.1f}%" if pd.notna(row["net_sentiment"]) else None, "strong"),
        (f"{row['pct_response']:.1f}%", "num"),
    ]


with T.panel_start("instable"):
    st.markdown('<div class="q-panel-title">Reviews By Period</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-note" style="margin-bottom:8px;">Click a column to sort. Expand a '
                'period to see its locations, then a location to read its reviews.</div>',
                unsafe_allow_html=True)

    RU.sort_header(COLUMNS, "ins_sort_col", "ins_sort_asc")

    sort_col = st.session_state["ins_sort_col"]
    sort_asc = st.session_state["ins_sort_asc"]
    # "Period" sorts chronologically, not alphabetically ("Feb-26" < "Jan-26").
    table = rows.sort_values("period" if sort_col == "label" else sort_col,
                             ascending=sort_asc, na_position="last")

    open_period = st.session_state.get("ins_open_period")
    open_site = st.session_state.get("ins_open_site")
    period_col = period_index(cur, granularity)

    for _, prow in table.iterrows():
        pkey = str(prow["period"])
        is_open = open_period == pkey

        def toggle_period(pkey=pkey):
            st.session_state["ins_open_period"] = None if st.session_state.get(
                "ins_open_period") == pkey else pkey
            st.session_state["ins_open_site"] = None

        RU.table_row(f"p_{pkey}", row_values(prow, prow["label"]), WIDTHS,
                     is_open, toggle_period, highlight=is_open)

        if not is_open:
            continue

        # --- level 2: the sites that reported in this period -----------------
        period_df = cur[period_col == prow["period"]]
        sites = site_table(period_df)
        sites = sites.sort_values(sort_col if sort_col in sites.columns else "n_reviews",
                                  ascending=sort_asc if sort_col in sites.columns else False,
                                  na_position="last")

        for _, srow in sites.iterrows():
            site = str(srow["site"])
            skey = f"{pkey}|{site}"
            site_open = open_site == skey

            def toggle_site(skey=skey):
                st.session_state["ins_open_site"] = None if st.session_state.get(
                    "ins_open_site") == skey else skey

            RU.table_row(f"s_{skey}", row_values(srow, site), WIDTHS,
                         site_open, toggle_site, indent=26)

            if site_open:
                # --- level 3: the reviews themselves ------------------------
                sub = period_df[period_df[SITE_COL] == site]
                coverage = (f" · site captured {int(srow['n_reviews']):,} of "
                            f"{int(srow['google_review_count']):,} Google reviews all-time"
                            if pd.notna(srow.get("google_review_count")) else "")
                RU.render_reviews(
                    skey, sub,
                    caption=f"<b>{site}</b> · {prow['label']} · "
                            f"{int(srow['n_reviews']):,} reviews{coverage}",
                )
                st.write("")

st.write("")
if st.button("← Back to dashboard", key="ins_back"):
    st.switch_page("Home.py")
