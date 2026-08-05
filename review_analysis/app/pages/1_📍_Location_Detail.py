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

from app.utils import ai as AI
from app.utils import reviews_ui as RU
from app.utils import theme as T
from app.utils.data_loader import ADDRESS_COL, RATING_COL, SITE_COL
from app.utils.metrics import (
    MONTH_COL,
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
demo_view = T.demo_mode()
if demo_view:
    # A tile's drill-down is standalone in demo mode: no sibling tabs, just the
    # way back to the dashboard.
    with st.container(key="qback"):
        if st.button("← Back to dashboard", key="back_home_top"):
            st.switch_page("Home.py")
else:
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
lens = st.session_state.get(RU.DEFAULT_FILTER_KEY, "All")
ai_reviews = RU.apply_filter(cur, lens)
# The leaderboard tiles are about one location, so their insight is too — and
# this is the label scripts/precompute_insights.py keys its cache on.
ai_site = (st.session_state.get("open_site")
           if st.session_state.get("rv_focus") in {"best", "worst"} else None)
if ai_site:
    ai_reviews = ai_reviews[ai_reviews[SITE_COL] == ai_site]
scope_label = (f"{window_label} · {ai_site or f'{len(chosen)} location(s)'}"
               + (f" · {lens.lower()} reviews only" if lens != "All" else ""))
net_txt = f' · net sentiment {m["net_sentiment"]:.1f}%' if m["net_sentiment"] is not None else ""
st.markdown(
    f'<div class="q-note" style="margin:-6px 0 16px;">{window_label} · {m["n_reviews"]:,} reviews '
    f'· {len(chosen)} locations{net_txt}</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chart panel: monthly sentiment split, with trend lines
# ---------------------------------------------------------------------------
CHART_MODES = ["Sentiment split (%)", "Volume & rating", "Avg rating by location"]

with T.panel_start("chart"):
    head_l, head_r = st.columns([6, 1.6])
    with head_l:
        title = ("Average Rating By Location"
                 if st.session_state.get("chart_metric") == "Avg rating by location"
                 else "Sentiment &amp; Rating By Month")
        show_rating_chart = st.session_state.get("chart_metric") == "Avg rating by location"
        st.markdown(f'<div class="q-panel-title">{title}</div>', unsafe_allow_html=True)
    with head_r:
        # Canonical value in a plain key; the widget is keyed separately and
        # seeded from it. Changing the period calls st.rerun() before this
        # selectbox is reached, and Streamlit drops the state of widgets a run
        # did not render -- which silently reset the chart to "Sentiment split".
        mode_value = st.session_state.get("chart_metric", CHART_MODES[0])
        if mode_value not in CHART_MODES:
            mode_value = CHART_MODES[0]
        chart_metric = st.selectbox(
            "Metric", options=CHART_MODES, index=CHART_MODES.index(mode_value),
            key="chart_mode_widget", label_visibility="collapsed",
        )
        st.session_state["chart_metric"] = chart_metric

    if chart_metric == "Avg rating by location":
        # Per-location ratings, not a time series: the dashboard's rating tile
        # lands here, and the question it raises is "which locations", not
        # "which months". Google's own published rating rides alongside as
        # markers, since the bars are only the reviews this file captured.
        # Horizontal bars: 25 location names cannot be read on a vertical
        # axis without rotating them, and rotated labels forced the value
        # labels to rotate too. Sorted worst-at-the-bottom so the eye lands on
        # the leaders first, with Google's own rating as a diamond alongside.
        rank = site_table(cur).sort_values("avg_rating", ascending=True)
        names = [str(s).replace("Big Dan's ", "") for s in rank["site"]]
        idx = list(range(len(rank)))
        values = [float(v) for v in rank["avg_rating"]]
        chain = float(pd.to_numeric(cur[RATING_COL], errors="coerce").mean())

        # Ratings cluster between ~4.2 and 5.0, so a 0-5 axis draws 24
        # near-identical full-width bars. Zoom to the pack and call out anything
        # that falls below the floor rather than clipping it silently.
        floor = 4.0 if float(rank["avg_rating"].median()) > 4.4 else 0.0
        below = [(n_, v) for n_, v in zip(names, values) if v < floor]

        fig = go.Figure()
        fig.add_bar(
            x=values, y=idx, orientation="h", width=0.68,
            name="Average rating (this selection)", cliponaxis=False,
            # Two tones, not two colours: behind-average locations recede
            # instead of shouting, and the dotted line says where the bar is.
            marker=dict(color=[T.BLUE if v >= chain else "#a9c3e4" for v in values],
                        cornerradius=4),
            text=[f"{v:.2f}" for v in values], textposition="outside",
            textfont=dict(color=T.INK, size=11.5),
            hovertemplate="%{customdata}: %{x:.2f} ★<extra></extra>", customdata=names,
        )

        google = rank["google_rating"].tolist()
        if rank["google_rating"].notna().any():
            # An invisible-ish chip carries the hover; the logo itself is drawn
            # on top as a layout image, so the mark is Google's own rather than
            # a diamond the reader has to decode from a legend.
            fig.add_trace(go.Scatter(
                x=google, y=idx, mode="markers", name="Google rating",
                marker=dict(symbol="circle", size=19, color="#ffffff",
                            line=dict(color="#e3e6ec", width=1)),
                hovertemplate="%{customdata} on Google: %{x:.1f} ★<extra></extra>",
                customdata=names,
            ))
            for i, g in enumerate(google):
                if g is None or pd.isna(g):
                    continue
                fig.add_layout_image(dict(
                    source=T.GOOGLE_LOGO_URI, xref="x", yref="y", x=float(g), y=i,
                    sizex=0.030, sizey=0.62, xanchor="center", yanchor="middle",
                    layer="above", sizing="contain",
                ))

        fig.add_vline(x=chain, line=dict(color=T.MUTED, width=1, dash="dot"))
        fig.add_annotation(yref="paper", y=1.03, x=chain, xanchor="center", yanchor="bottom",
                           text=f"avg {chain:.2f}", showarrow=False,
                           font=dict(color=T.MUTED, size=11.5))
        for i, v in enumerate(values):
            if v < floor:
                fig.add_annotation(x=floor, y=i, xanchor="left", yanchor="middle",
                                   text=f"  {v:.2f} — below the axis", showarrow=False,
                                   font=dict(color=T.NEG_COLOR, size=11))

        fig.update_layout(
            height=max(340, 23 * len(rank) + 96), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            margin=dict(l=10, r=34, t=44, b=8), font=dict(color=T.MUTED, size=12),
            bargap=0.32,
            yaxis=dict(tickmode="array", tickvals=idx, ticktext=names, showgrid=False,
                       tickfont=dict(color=T.INK_SOFT, size=11.5), linecolor=T.LINE,
                       ticklabelstandoff=8),
            xaxis=dict(title=dict(text="Average rating", font=dict(color=T.MUTED)),
                       range=[floor, 5.08], showgrid=True, gridcolor="#eef1f6",
                       zeroline=False, dtick=0.2, tickfont=dict(size=11)),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
                        font=dict(color=T.INK_SOFT, size=12.5)),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="q-note">Bars are the mean of the reviews captured for this selection; '
            f'<img src="{T.GOOGLE_LOGO_URI}" style="height:13px;vertical-align:-2px;"> is the '
            'rating Google publishes over every review it holds. Paler bars sit below the '
            'selection average.'
            + (f' Axis starts at {floor:.1f}; ' + ", ".join(f"{n_} ({v:.2f})" for n_, v in below)
               + " sits below it." if below else "")
            + '</div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="q-panel-title" style="font-size:20px;">'
                    'Sentiment &amp; Rating By Month</div>', unsafe_allow_html=True)

    # The rating view shows this trend too -- both readings of "how are we
    # rated" are useful, so the rating tile gets location ranking *and* history.
    if True:
        trend_mode = "Volume & rating" if chart_metric == "Avg rating by location" else chart_metric
        # The chart follows the selection. A one-month window would draw a
        # single bar, so it falls back to twelve months of context -- and says
        # so, rather than quietly showing months the filter excludes.
        chart_extended = cur[MONTH_COL].nunique() < 3
        series = monthly_series(df if chart_extended else cur, months=12)
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

            window_labels = (set(monthly_series(cur, months=14)["label"])
                             if chart_extended else set(x))
            opacity = [1.0 if lbl in window_labels else 0.32 for lbl in x]

            if trend_mode == "Sentiment split (%)":
                neg = series["pct_negative"].tolist()
                pos = series["pct_positive"].tolist()
                neu = [max(0.0, 100 - (p or 0) - (n or 0)) for p, n in zip(pos, neg)]
                fig.add_bar(x=x, y=neg, name="Negative %", marker_color=T.ORANGE,
                            marker_opacity=opacity)
                fig.add_bar(x=x, y=pos, name="Positive %", marker_color=T.BLUE,
                            marker_opacity=opacity)
                fig.add_bar(x=x, y=neu, name="Neutral %", marker_color=T.NEU_COLOR,
                            marker_opacity=opacity)
                trendline(neg, "#f7d3a6", "Negative trend")
                trendline(pos, "#c3dcf5", "Positive trend")
                y_title = "Percentage"
            else:
                counts = series["n_reviews"].tolist()
                ratings = series["avg_rating"].tolist()
                fig.add_bar(x=x, y=counts, name="Reviews", marker_color=T.BLUE, yaxis="y",
                            marker_opacity=opacity)
                fig.add_trace(go.Scatter(
                    x=x, y=ratings, name="Avg rating", mode="lines+markers", yaxis="y2",
                    line=dict(color=T.ORANGE, width=2.5),
                ))
                fig.update_layout(yaxis2=dict(title="Avg rating", overlaying="y", side="right",
                                              range=[1, 5], showgrid=False))
                y_title = "Reviews"

            fig.update_layout(
                barmode="stack", bargap=0.42,
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
            # A short window would otherwise render as a couple of enormous blocks:
            # Plotly splits the axis between however many categories exist, and bar
            # `width` is in those axis units, so narrowing the bars alone does
            # nothing. Widen the axis to a notional six slots and centre the real
            # ones in it.
            if len(x) < 6:
                pad = (6 - len(x)) / 2
                fig.update_traces(selector=dict(type="bar"), width=0.3)
                fig.update_xaxes(range=[-0.5 - pad, len(x) - 0.5 + pad])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            if chart_extended:
                st.markdown(
                    f'<div class="q-note">Showing 12 months for context; '
                    f'<b>{window_label}</b> — your selection — is the solid bar, and the '
                    f'table below covers it alone.</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Site table with expandable rows
# ---------------------------------------------------------------------------
# Every column the table can show, and how to render its cell. A tile only
# ever shows the handful that answer its own question -- the full nine-column
# grid repeated for 25 locations is the cognitive load the tiles exist to
# avoid.
COLUMN_DEFS = {
    "site":          ("Location", 3.4, lambda r: (html.escape(str(r["site"])), "name")),
    "n_reviews":     ("Reviews (#)", 1.3, lambda r: (f"{int(r['n_reviews']):,}", "num")),
    "avg_rating":    ("Avg Rating", 1.3,
                      lambda r: (f"{r['avg_rating']:.2f} \u2605", "strong")),
    "google_rating": ("Google Rating", 1.4,
                      lambda r: (f"{r['google_rating']:.1f} \u2605"
                                 if pd.notna(r["google_rating"]) else None, "num")),
    "positive":      ("Positive (#)", 1.3, lambda r: (f"{int(r['positive']):,}", "strong")),
    "neutral":       ("Neutral (#)", 1.2, lambda r: (f"{int(r['neutral']):,}", "num")),
    "negative":      ("Negative (#)", 1.3, lambda r: (f"{int(r['negative']):,}", "strong")),
    "pct_positive":  ("Positive %", 1.3,
                      lambda r: (f"{r['pct_positive']:.1f}%"
                                 if pd.notna(r["pct_positive"]) else None, "num")),
    "pct_negative":  ("Negative %", 1.3,
                      lambda r: (f"{r['pct_negative']:.1f}%"
                                 if pd.notna(r["pct_negative"]) else None, "num")),
    "net_sentiment": ("Net Sentiment", 1.5,
                      lambda r: (f"{r['net_sentiment']:.1f}%"
                                 if pd.notna(r["net_sentiment"]) else None, "strong")),
    "pct_response":  ("Response Rate", 1.5, lambda r: (f"{r['pct_response']:.1f}%", "num")),
    "n_awaiting":    ("Awaiting Reply (#)", 1.7,
                      lambda r: (f"{int(r['n_awaiting']):,}", "strong")),
    "n_1_2_star":    ("1\u20132 Star (#)", 1.4, lambda r: (f"{int(r['n_1_2_star']):,}", "num")),
}

# Which columns each tile's drill-down shows, and what to call the table.
# (columns, table title, default sort, "hide rows where this is 0")
FOCUS_VIEWS = {
    "negative":  (["site", "negative", "pct_negative", "n_reviews"],
                  "Negative Reviews By Location", "negative", "negative"),
    "worst":     (["site", "negative", "net_sentiment", "n_reviews"],
                  "Negative Reviews By Location", "negative", "negative"),
    "positive":  (["site", "positive", "pct_positive", "n_reviews"],
                  "Positive Reviews By Location", "positive", "positive"),
    "best":      (["site", "positive", "net_sentiment", "n_reviews"],
                  "Positive Reviews By Location", "positive", "positive"),
    "sentiment": (["site", "net_sentiment", "positive", "negative"],
                  "Sentiment By Location", "net_sentiment", None),
    "rating":    (["site", "avg_rating", "google_rating", "n_1_2_star", "n_reviews"],
                  "Rating By Location", "avg_rating", None),
    "response":  (["site", "pct_response", "n_awaiting", "n_reviews"],
                  "Owner Responses By Location", "n_awaiting", "n_awaiting"),
}
FULL_VIEW = (["site", "n_reviews", "avg_rating", "google_rating", "positive",
              "neutral", "negative", "net_sentiment", "pct_response"],
             "Reviews By Location", "n_reviews", None)

focus = st.session_state.get("rv_focus")
column_keys, table_title, default_sort, nonzero_col = FOCUS_VIEWS.get(focus, FULL_VIEW)
COLUMNS = [(k, COLUMN_DEFS[k][0], COLUMN_DEFS[k][1]) for k in column_keys]
WIDTHS = [w for _, _, w in COLUMNS]

table = site_table(cur)
n_locations = len(table)
# A "negative reviews" view listing 14 locations with zero negative reviews is
# noise; drop the rows that have nothing of what the tile asked about.
if nonzero_col:
    table = table[table[nonzero_col].fillna(0) > 0]
st.session_state.setdefault("table_sort_col", default_sort)
st.session_state.setdefault("table_sort_asc", False)
# A tile can arrive sorted by a column this view does not show.
if st.session_state["table_sort_col"] not in column_keys:
    st.session_state["table_sort_col"] = default_sort

with T.panel_start("table"):
    head_l, head_r = st.columns([6, 1.6])
    with head_l:
        st.markdown(f'<div class="q-panel-title">{table_title}</div>', unsafe_allow_html=True)
        shown_note = (f"{len(table)} of {n_locations} locations have any"
                      if nonzero_col else f"{len(table)} locations")
        st.markdown(f'<div class="q-note" style="margin-bottom:8px;">{shown_note} · click a '
                    'column to sort; expand a location to read its reviews.</div>',
                    unsafe_allow_html=True)
    with head_r:
        AI.insight_button(f"detail_{focus or 'all'}", ai_reviews, scope_label, focus)

    RU.review_controls(bool(st.session_state.get("open_site")))
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

        values = [COLUMN_DEFS[k][2](row) for k in column_keys]
        RU.table_row(f"site_{site}", values, WIDTHS, is_open, toggle, highlight=is_open)

        if is_open:
            sub = cur[cur[SITE_COL] == site]
            addr = sub[ADDRESS_COL].dropna().iloc[0] if sub[ADDRESS_COL].notna().any() else ""
            # Coverage can exceed 100%: Google's own review count is a snapshot
            # taken at scrape time and drifts from the rows actually collected.
            # Google's count is all-time, so a share of it only means anything
            # when the window is too; on a one-month view it read "3 reviews
            # captured, Google reports 907 (0%)".
            if pd.isna(row.get("google_review_count")):
                coverage = " \u00b7 no Google review count published for this site"
            elif period_choice == "All time":
                coverage = (f" \u00b7 {int(row['n_reviews']):,} reviews captured, "
                            f"Google reports {int(row['google_review_count']):,} "
                            f"({row['coverage_pct']:.0f}%)")
            else:
                coverage = (f" \u00b7 {int(row['n_reviews']):,} reviews in this period "
                            f"\u00b7 Google reports {int(row['google_review_count']):,} all-time")
            RU.render_reviews(site, sub, caption=f"{html.escape(str(addr))}{coverage}")
            st.write("")


