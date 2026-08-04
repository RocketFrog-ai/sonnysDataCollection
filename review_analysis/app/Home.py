"""Review Analytics — landing dashboard (entrypoint).

Run with: streamlit run app/Home.py

NOTE: this file is intentionally NOT named app.py. `app/` is also a Python
package (it contains app/utils/), and naming the entrypoint script the same
as the package it lives in creates a real import collision under
`streamlit run` (Streamlit registers the running script as module `app`,
shadowing the `app` package, so `from app.utils... import` fails with
"'app' is not a package"). Keeping the entrypoint filename distinct from
the package name avoids this.

Feature 1 (Dashboard & Ratings Overview): KPI tiles, rating distribution,
review-count-by-location chart, and a clickable tile grid (one tile per
`site` — this dataset is a single car-wash chain with 25 physical
locations; `site` is the location key, see docs/dataset_schema.md).
Clicking a tile's "View details" button navigates to the Location Detail
page, pre-filtered to that site.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils.data_loader import (
    CATEGORY_COL,
    CITY_COL,
    RATING_COL,
    SITE_COL,
    STATE_COL,
    get_kpis,
    get_locations,
    load_data,
)

st.set_page_config(page_title="Review Analytics Dashboard", page_icon="🚗", layout="wide")

SESSION_KEY_SELECTED_LOCATION = "selected_location"
DETAIL_PAGE_PATH = "pages/1_📍_Location_Detail.py"

# House style: white chart surface, status colors for rating tiers, one
# sequential blue for plain magnitude bars.
CHART_SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SEQUENTIAL_BLUE = "#256abf"

STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_LABELS = {"good": "Excellent", "warning": "Good", "serious": "Needs attention", "critical": "Critical"}
RATING_BUCKET_COLORS = {1: STATUS_COLORS["critical"], 2: STATUS_COLORS["serious"], 3: STATUS_COLORS["warning"],
                         4: "#5aa8e0", 5: STATUS_COLORS["good"]}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .kpi-tile {{
            background-color: {CHART_SURFACE}; border: 1px solid {GRIDLINE};
            border-radius: 10px; padding: 16px 18px;
        }}
        .kpi-label {{
            color: {TEXT_MUTED}; font-size: 0.8rem; text-transform: uppercase;
            letter-spacing: 0.03em; margin-bottom: 4px;
        }}
        .kpi-value {{ color: {TEXT_PRIMARY}; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }}
        .loc-card {{
            border: 1px solid {GRIDLINE}; border-radius: 10px; padding: 14px 16px;
            background-color: {CHART_SURFACE}; margin-bottom: 6px;
        }}
        .loc-name {{ color: {TEXT_PRIMARY}; font-weight: 600; font-size: 1.0rem; margin-bottom: 2px; }}
        .loc-sub {{ color: {TEXT_MUTED}; font-size: 0.78rem; margin-bottom: 8px; }}
        .loc-stars {{ color: #eda100; font-size: 1.0rem; letter-spacing: 1px; }}
        .loc-rating-num {{ color: {TEXT_PRIMARY}; font-weight: 600; font-size: 0.95rem; }}
        .loc-count {{ color: {TEXT_SECONDARY}; font-size: 0.82rem; }}
        .status-badge {{
            display: inline-block; padding: 2px 9px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 600; color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def rating_tier(avg_rating: float) -> str:
    if pd.isna(avg_rating):
        return "warning"
    if avg_rating >= 4.5:
        return "good"
    if avg_rating >= 3.5:
        return "warning"
    if avg_rating >= 2.5:
        return "serious"
    return "critical"


def star_string(avg_rating: float, max_stars: int = 5) -> str:
    if pd.isna(avg_rating):
        return "—"
    filled = max(0, min(max_stars, int(round(avg_rating))))
    return "★" * filled + "☆" * (max_stars - filled)


def build_rating_distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df[RATING_COL].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[f"{b}★" for b in counts.index],
            y=counts.values,
            marker=dict(color=[RATING_BUCKET_COLORS[b] for b in counts.index]),
            text=[f"{c:,}" for c in counts.values],
            textposition="outside",
            hovertemplate="%{x}: %{y:,} reviews<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Reviews by star rating</b>", font=dict(color=TEXT_PRIMARY, size=16)),
        plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE, font=dict(color=TEXT_SECONDARY),
        margin=dict(l=40, r=20, t=50, b=40), height=340, showlegend=False,
        xaxis=dict(showgrid=False, linecolor=GRIDLINE, tickfont=dict(color=TEXT_SECONDARY)),
        yaxis=dict(showgrid=True, gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=TEXT_MUTED)),
    )
    return fig


def build_reviews_per_location_chart(locations: pd.DataFrame) -> go.Figure:
    rows = locations.sort_values("review_count")
    fig = go.Figure(
        go.Bar(
            x=rows["review_count"], y=rows[SITE_COL], orientation="h",
            marker=dict(color=SEQUENTIAL_BLUE),
            text=[f"{c:,}" for c in rows["review_count"]], textposition="outside",
            hovertemplate="%{y}: %{x:,} reviews<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="<b>Review count by location</b>", font=dict(color=TEXT_PRIMARY, size=16)),
        plot_bgcolor=CHART_SURFACE, paper_bgcolor=CHART_SURFACE, font=dict(color=TEXT_SECONDARY),
        margin=dict(l=10, r=40, t=50, b=40), height=max(340, 26 * len(rows)), showlegend=False,
        xaxis=dict(showgrid=True, gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(showgrid=False, linecolor=GRIDLINE, tickfont=dict(color=TEXT_SECONDARY)),
    )
    return fig


def render_kpi_row(kpis: dict) -> None:
    cols = st.columns(5)
    tiles = [
        ("Total reviews", f"{kpis['total_reviews']:,}"),
        ("Overall avg rating", f"{kpis['avg_rating']:.2f} ★" if kpis["avg_rating"] is not None else "—"),
        ("Locations", f"{kpis['total_sites']:,}"),
        ("5-star share", f"{kpis['pct_5_star']:.0f}%" if kpis["pct_5_star"] is not None else "—"),
        ("Owner response rate", f"{kpis['pct_with_owner_response']:.0f}%" if kpis["pct_with_owner_response"] is not None else "—"),
    ]
    for col, (label, value) in zip(cols, tiles):
        with col:
            st.markdown(
                f'<div class="kpi-tile"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>',
                unsafe_allow_html=True,
            )


def navigate_to_detail(site_value: str) -> None:
    st.session_state[SESSION_KEY_SELECTED_LOCATION] = site_value
    st.switch_page(DETAIL_PAGE_PATH)


def render_location_tile(loc: pd.Series) -> None:
    site = loc[SITE_COL]
    location_line = ", ".join([v for v in (loc.get(CITY_COL), loc.get(STATE_COL)) if isinstance(v, str) and v])
    avg_rating = loc["avg_rating"]
    tier = rating_tier(avg_rating)

    with st.container():
        st.markdown(
            f"""<div class="loc-card">
                    <div class="loc-name">{site}</div>
                    <div class="loc-sub">{location_line or loc.get(CATEGORY_COL, "")}</div>
                    <div><span class="loc-stars">{star_string(avg_rating)}</span>
                         <span class="loc-rating-num">&nbsp;{avg_rating:.2f}</span></div>
                    <div class="loc-count">{int(loc['review_count']):,} reviews</div>
                    <div style="margin-top:6px;">
                        <span class="status-badge" style="background-color:{STATUS_COLORS[tier]};">{STATUS_LABELS[tier]}</span>
                    </div>
                </div>""",
            unsafe_allow_html=True,
        )
        if st.button("View details", key=f"loc_btn_{site}", width="stretch"):
            navigate_to_detail(site)


def render_location_grid(locations: pd.DataFrame, n_cols: int = 4) -> None:
    st.subheader("Locations")
    rows = locations.to_dict("records")
    for i in range(0, len(rows), n_cols):
        cols = st.columns(n_cols)
        for col, loc in zip(cols, rows[i:i + n_cols]):
            with col:
                render_location_tile(pd.Series(loc))


def main() -> None:
    inject_css()
    st.title("Big Dan's Car Wash — Review Analytics")
    st.caption(
        "Overview across all 25 locations (`site`). Click a location tile to drill "
        "into its trends and individual reviews."
    )

    df = load_data()
    kpis = get_kpis(df)
    locations = get_locations(df)

    render_kpi_row(kpis)

    st.write("")
    chart_col1, chart_col2 = st.columns([1, 1.4])
    with chart_col1:
        st.plotly_chart(build_rating_distribution_chart(df), width="stretch")
    with chart_col2:
        st.plotly_chart(build_reviews_per_location_chart(locations), width="stretch")

    st.write("")
    render_location_grid(locations, n_cols=4)


main()
