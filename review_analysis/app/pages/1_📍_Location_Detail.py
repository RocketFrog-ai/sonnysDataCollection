"""Site breakdown / drill-down page.

Mirrors the product's drill-down layout: a monthly sentiment chart with
trend lines, then a sortable table with one row per site. Expanding a row
reveals that site's individual reviews, ordered by whichever sort the user
picks — including the raw VADER compound score, which is the only ordering
that ranks "furious" above "mildly annoyed".

Reached from a tile on app/Home.py (which passes the location selection
through st.session_state["selected_sites"]), or usable standalone.
"""

from __future__ import annotations

import html
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
from app.utils.data_loader import ADDRESS_COL, SITE_COL
from app.utils.metrics import (
    PERIOD_WINDOWS,
    load_scored_data,
    monthly_series,
    period_metrics,
    site_table,
    window_frames,
)

SESSION_SITES = "selected_sites"

T.setup_page("Site Breakdown", page_icon="📍")

df_all = load_scored_data()
all_sites = sorted(df_all[SITE_COL].dropna().unique().tolist())

T.page_heading("Reviews", "Big Dan's Car Wash")
T.tab_strip([
    ("All Sites", False, "Home.py"),
    ("Insights", False, "pages/2_📊_Insights.py"),
    ("Site Breakdown", True, None),
])

# ---------------------------------------------------------------------------
# Filter row
# ---------------------------------------------------------------------------
chosen = st.session_state.get(SESSION_SITES) or all_sites
chosen = [s for s in chosen if s in all_sites] or all_sites

left, _mid, right = st.columns([2.2, 6, 1.9])
with left:
    with st.container(key="qfilter_loc2"):
        with st.popover(f"Location ({len(chosen)})"):
            picked = st.multiselect(
                "Sites", options=all_sites, default=chosen,
                key="detail_site_filter", label_visibility="collapsed",
            )
            if picked and picked != chosen:
                st.session_state[SESSION_SITES] = picked
                st.rerun()
with right:
    period_choice = st.session_state.get("detail_period", "All time")
    with st.container(key="qfilter_period2"):
        with st.popover(f"🗓  {period_choice}"):
            picked_period = st.radio(
                "Period", options=list(PERIOD_WINDOWS),
                index=list(PERIOD_WINDOWS).index(period_choice),
                key="detail_period_radio", label_visibility="collapsed",
            )
            if picked_period != period_choice:
                st.session_state["detail_period"] = picked_period
                st.rerun()

df = df_all[df_all[SITE_COL].isin(chosen)]
cur, prior, window_label, caption = window_frames(df, period_choice)
if cur.empty:
    st.warning("No reviews in the selected period for these locations.")
    st.stop()

m = period_metrics(cur)
net_txt = f' · net sentiment {m["net_sentiment"]:.1f}%' if m["net_sentiment"] is not None else ""
st.markdown(
    f'<div class="q-note" style="margin:-6px 0 16px;">{window_label} · {m["n_reviews"]:,} reviews '
    f'· {len(chosen)} locations{net_txt}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chart panel: monthly sentiment split, with trend lines
# ---------------------------------------------------------------------------
with T.panel_start("chart"):
    head_l, head_r = st.columns([6, 1.3])
    with head_l:
        st.markdown('<div class="q-panel-title">Sentiment &amp; Rating By Month</div>',
                    unsafe_allow_html=True)
    with head_r:
        chart_metric = st.selectbox(
            "Metric", options=["Sentiment split (%)", "Volume & rating"],
            index=0, key="chart_metric", label_visibility="collapsed",
        )

    series = monthly_series(cur, months=14)
    if series.empty:
        st.info("Not enough dated reviews to plot a trend.")
    else:
        x = series["label"].tolist()
        fig = go.Figure()

        def trendline(y_vals, color, name):
            """Straight least-squares fit across the plotted months."""
            ys = pd.Series(y_vals, dtype="float64")
            valid = ys.notna()
            if valid.sum() < 2:
                return
            idx = pd.Series(range(len(ys)), dtype="float64")
            slope, intercept = np.polyfit(idx[valid], ys[valid], 1)
            fig.add_trace(go.Scatter(
                x=x, y=(slope * idx + intercept).tolist(), mode="lines", name=name,
                line=dict(color=color, width=2, dash="dash"), showlegend=False,
                hoverinfo="skip",
            ))

        if chart_metric == "Sentiment split (%)":
            neg = series["pct_negative"].tolist()
            pos = series["pct_positive"].tolist()
            fig.add_bar(x=x, y=neg, name="Negative %", marker_color=T.ORANGE)
            fig.add_bar(x=x, y=pos, name="Positive %", marker_color=T.BLUE)
            trendline(neg, "#f7d3a6", "Negative trend")
            trendline(pos, "#c3dcf5", "Positive trend")
            y_title = "Percentage"
        else:
            counts = series["n_reviews"].tolist()
            ratings = series["avg_rating"].tolist()
            fig.add_bar(x=x, y=counts, name="Reviews", marker_color=T.BLUE, yaxis="y")
            fig.add_trace(go.Scatter(
                x=x, y=ratings, name="Avg rating", mode="lines+markers", yaxis="y2",
                line=dict(color=T.ORANGE, width=2.5),
            ))
            fig.update_layout(yaxis2=dict(title="Avg rating", overlaying="y", side="right",
                                          range=[1, 5], showgrid=False))
            y_title = "Reviews"

        fig.update_layout(
            barmode="group", bargap=0.55, bargroupgap=0.03,
            height=420, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=18, b=10),
            font=dict(color=T.MUTED, size=13),
            xaxis=dict(type="category", showgrid=False, linecolor=T.LINE,
                       tickfont=dict(color=T.INK_SOFT)),
            yaxis=dict(title=dict(text=y_title, font=dict(color=T.MUTED)), showgrid=True,
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
# Site table with expandable rows
# ---------------------------------------------------------------------------
COLUMNS = [
    ("site", "Site", 3.4),
    ("n_reviews", "Reviews (#)", 1.2),
    ("avg_rating", "Avg Rating", 1.2),
    ("google_rating", "Google Rating", 1.3),
    ("positive", "Positive (#)", 1.2),
    ("neutral", "Neutral (#)", 1.1),
    ("negative", "Negative (#)", 1.2),
    ("net_sentiment", "Net Sentiment", 1.4),
    ("pct_response", "Response Rate", 1.4),
]

table = site_table(cur)
st.session_state.setdefault("table_sort_col", "n_reviews")
st.session_state.setdefault("table_sort_asc", False)

WIDTHS = [w for _, _, w in COLUMNS]

with T.panel_start("table"):
    st.markdown('<div class="q-panel-title">Reviews By Site</div>', unsafe_allow_html=True)
    st.markdown('<div class="q-note" style="margin-bottom:8px;">Click a column to sort; expand a '
                'site to read its reviews.</div>', unsafe_allow_html=True)

    RU.sort_header(COLUMNS, "table_sort_col", "table_sort_asc")

    sort_key = st.session_state["table_sort_col"]
    sort_asc = st.session_state["table_sort_asc"]
    table = table.sort_values(sort_key, ascending=sort_asc, na_position="last")
    open_site = st.session_state.get("open_site")

    for _, row in table.iterrows():
        site = str(row["site"])
        is_open = open_site == site

        def toggle(site=site):
            st.session_state["open_site"] = None if st.session_state.get("open_site") == site else site

        values = [
            (html.escape(site), "name"),
            (f"{int(row['n_reviews']):,}", "num"),
            (f"{row['avg_rating']:.2f} \u2605", "strong"),
            (f"{row['google_rating']:.1f} \u2605" if pd.notna(row["google_rating"]) else None, "num"),
            (f"{int(row['positive']):,}", "num"),
            (f"{int(row['neutral']):,}", "num"),
            (f"{int(row['negative']):,}", "num"),
            (f"{row['net_sentiment']:.1f}%" if pd.notna(row["net_sentiment"]) else None, "strong"),
            (f"{row['pct_response']:.1f}%", "num"),
        ]
        RU.table_row(f"site_{site}", values, WIDTHS, is_open, toggle, highlight=is_open)

        if is_open:
            sub = cur[cur[SITE_COL] == site]
            addr = sub[ADDRESS_COL].dropna().iloc[0] if sub[ADDRESS_COL].notna().any() else ""
            # Coverage can exceed 100%: Google's own review count is a snapshot
            # taken at scrape time and drifts from the rows actually collected.
            coverage = (f" \u00b7 {int(row['n_reviews']):,} reviews captured, "
                        f"Google reports {int(row['google_review_count']):,} "
                        f"({row['coverage_pct']:.0f}%)"
                        if pd.notna(row.get("coverage_pct")) else
                        " \u00b7 no Google review count published for this site")
            RU.render_reviews(site, sub, caption=f"{html.escape(str(addr))}{coverage}")
            st.write("")

st.write("")
# A button + switch_page rather than st.page_link: page_link renders the
# hidden sidebar nav entry, and the shell hides that nav entirely.
if st.button("← Back to dashboard", key="back_home"):
    st.switch_page("Home.py")
