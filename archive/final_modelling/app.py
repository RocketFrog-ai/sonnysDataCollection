"""Streamlit app: predict a new car-wash site's KPIs from its cluster neighbours.

Drop a pin on the USA map (inside a shaded cluster box, where data exists and
sites move in sync). The app finds neighbours within the 20 km radius and
predicts 5 KPI trajectories as a distance-weighted blend of those neighbours,
shown against the neighbours' own curves.
"""
from __future__ import annotations

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from src import config as C
from src.data_loader import load_all
from src.neighbours import load_cluster_bboxes
from src.predict import predict_site

st.set_page_config(page_title="Car-Wash Site Predictor", layout="wide")

PRED_COLOR = "#E74C3C"
NBR_LINE = "rgba(130,130,130,0.35)"
USA_CENTER = [39.5, -98.35]


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data…")
def get_dataset():
    return load_all()


@st.cache_data
def get_bboxes():
    try:
        return load_cluster_bboxes()
    except FileNotFoundError:
        return []


@st.cache_data
def get_eval():
    try:
        return pd.read_csv(C.EVAL_PATH)
    except FileNotFoundError:
        return None


ds = get_dataset()
bboxes = get_bboxes()


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
def build_map(pin=None, neighbours=None, buffer_km=C.BUFFER_KM, tiles="CartoDB Voyager"):
    m = folium.Map(location=USA_CENTER, zoom_start=5, tiles=tiles, control_scale=True)

    # cluster footprints: centroid circles matching the radius the prediction
    # actually uses (a min 1 km dot so tight clusters stay visible).
    for b in bboxes:
        folium.Circle(
            [b["centroid_lat"], b["centroid_lon"]],
            radius=max(b["radius_km"], 1.0) * 1000,
            color="#2E86DE", weight=1, fill=True, fill_opacity=0.12,
            tooltip=(
                f"cluster {b['cluster_id']} · {b['n_sites']} sites · "
                f"{b['dominant_state']} · r={b['radius_km']:.1f} km"
            ),
        ).add_to(m)

    if pin is not None:
        lat, lon = pin
        folium.Circle(
            [lat, lon], radius=buffer_km * 1000, color=PRED_COLOR,
            weight=2, fill=True, fill_opacity=0.06,
            tooltip=f"{buffer_km:.0f} km capture radius",
        ).add_to(m)
        folium.Marker(
            [lat, lon], tooltip="New site",
            icon=folium.Icon(color="red", icon="location-pin", prefix="fa"),
        ).add_to(m)
        if neighbours is not None:
            for _, nb in neighbours.iterrows():
                folium.CircleMarker(
                    [nb.lat, nb.lon], radius=5, color="#27AE60",
                    fill=True, fill_opacity=0.9,
                    tooltip=f"{nb.site_uid} · {nb.dist_km:.1f} km · {nb.client}",
                ).add_to(m)
    return m


# --------------------------------------------------------------------------- #
# Cluster-site share view: clickable cluster map (left) drives the pies (right).
# Isolated in a fragment so a marker click reruns only this block.
# --------------------------------------------------------------------------- #
def _share_pie(membership, retail, title, unit):
    fig = go.Figure(go.Pie(
        labels=["Membership", "Retail"],
        values=[float(membership), float(retail)],
        hole=0.45, sort=False,
        marker=dict(colors=[PRED_COLOR, "#2E86DE"]),
        textinfo="label+percent",
        hovertemplate=f"%{{label}}<br>{unit}<br>%{{percent}}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=300,
        margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
    )
    return fig


def _share_area(x, membership_pct, retail_pct, title, unit):
    """Month-on-month 100%-stacked area: membership band + retail band."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=membership_pct, name="Membership", mode="lines",
        stackgroup="one", line=dict(width=0.5, color=PRED_COLOR),
        fillcolor="rgba(231,76,60,0.65)",
        hovertemplate=f"Membership %{{y:.1f}}% of {unit}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=retail_pct, name="Retail", mode="lines",
        stackgroup="one", line=dict(width=0.5, color="#2E86DE"),
        fillcolor="rgba(46,134,222,0.65)",
        hovertemplate=f"Retail %{{y:.1f}}% of {unit}<extra></extra>"))
    fig.update_layout(
        title=title, height=300, hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(range=[0, 100], title="% share"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


@st.fragment
def render_share_pies(bd, panels, month_axis, avail, neighbours, pin):
    nb = (
        neighbours[neighbours["site_uid"].isin(avail)]
        .dropna(subset=["lat", "lon"]).reset_index(drop=True)
    )
    if nb.empty:
        st.info("No locatable cluster sites with share data here.")
        return

    # Selected site: last map click (persisted) falling back to the nearest one.
    state = st.session_state.get("cluster_map")
    sel = st.session_state.get("share_sel_uid")
    pts = (state or {}).get("selection", {}).get("points") if state else None
    if pts:
        cd = pts[0].get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is not None:
            sel = cd
    if sel not in set(nb["site_uid"]):
        sel = nb["site_uid"].iloc[0]
    st.session_state["share_sel_uid"] = sel

    left, right = st.columns([1, 1])

    # --- left: clickable cluster map (Plotly, resizes cleanly in columns) ---
    with left:
        span = max(nb["lat"].max() - nb["lat"].min(),
                   nb["lon"].max() - nb["lon"].min()) + 0.02
        zoom = float(min(13, max(8, 9.7 - (span - 0.1) * 6)))

        cmap = go.Figure()
        cmap.add_trace(go.Scattermapbox(
            lat=nb["lat"], lon=nb["lon"], mode="markers",
            marker=dict(
                size=[18 if u == sel else 12 for u in nb["site_uid"]],
                color=[PRED_COLOR if u == sel else "#27AE60" for u in nb["site_uid"]],
            ),
            customdata=nb["site_uid"].to_numpy().reshape(-1, 1),
            text=nb["site_uid"] + " · " + nb["dist_km"].round(1).astype(str) + " km",
            hovertemplate="%{text}<extra></extra>", name="cluster sites",
        ))
        cmap.add_trace(go.Scattermapbox(
            lat=[pin[0]], lon=[pin[1]], mode="markers",
            marker=dict(size=15, color="#FF2D2D"),
            text=["New site"], hoverinfo="text", name="new site",
        ))
        cmap.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=nb["lat"].mean(), lon=nb["lon"].mean()),
                zoom=zoom,
            ),
            margin=dict(l=0, r=0, t=0, b=0), height=620, showlegend=False,
            # keep pan/zoom while clicking within a cluster, but reset to the
            # default view when the pin moves to a new location
            uirevision=f"{pin[0]:.3f},{pin[1]:.3f}",
        )
        st.plotly_chart(
            cmap, use_container_width=True, key="cluster_map",
            on_select="rerun", selection_mode="points",
        )
        st.caption("Click a green dot to inspect that cluster site")

    # --- right: share view for the selected site ---
    with right:
        st.markdown(f"#### {sel}")
        view = st.radio(
            "View", ["Totals (pie)", "Month-on-month (lines)"],
            horizontal=True, key="share_view",
            help="Totals = volume-weighted share pooled over all months "
                 "(sum of $ / washes, not an average of monthly %). "
                 "Month-on-month = each month's own share.",
        )

        if view.startswith("Totals"):
            if sel in bd["wash"].index:
                w = bd["wash"].loc[sel]
                st.plotly_chart(
                    _share_pie(w["membership_wash"], w["retail_wash"],
                               "Wash share — all months", "%{value:,.0f} washes"),
                    width="stretch")
            if sel in bd["sales"].index:
                s = bd["sales"].loc[sel]
                st.plotly_chart(
                    _share_pie(s["membership_sales"], s["retail_sales"],
                               "Sales share — all months", "$%{value:,.0f}"),
                    width="stretch")
        else:
            x = month_axis.to_timestamp()
            if panels and sel in panels["membership_pct_wash"].index:
                st.plotly_chart(
                    _share_area(
                        x, panels["membership_pct_wash"].loc[sel],
                        panels["retail_pct_wash"].loc[sel],
                        "Wash share — month on month", "washes"),
                    width="stretch")
            if panels and sel in panels["membership_pct_sales"].index:
                st.plotly_chart(
                    _share_area(
                        x, panels["membership_pct_sales"].loc[sel],
                        panels["retail_pct_sales"].loc[sel],
                        "Sales share — month on month", "sales"),
                    width="stretch")


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.header("Controls")
agg = "idw" if st.sidebar.radio(
    "Neighbour weighting", ["Inverse-distance (IDW)", "Simple mean"],
    help="A new site has no history, so neighbours are weighted by distance only.",
) == "Inverse-distance (IDW)" else "mean"

view = st.sidebar.radio(
    "Plot view", ["Real units", "Normalized shape (0–1)"],
    help="Real units = predicted magnitude. Normalized = trend shape / sync only.",
)
normalized = view.startswith("Normalized")

buffer_km = C.BUFFER_KM

TILES = {
    "Dark": "CartoDB dark_matter",
    "Streets (city names)": "CartoDB Voyager",
    "OpenStreetMap": "OpenStreetMap",
    "Minimal light": "CartoDB positron",
}
tile_label = st.sidebar.selectbox("Map style", list(TILES), index=0)
tiles = TILES[tile_label]


# --------------------------------------------------------------------------- #
# Map (full-width hero)
# --------------------------------------------------------------------------- #
fmap = build_map(
    pin=st.session_state.get("pin"),
    neighbours=st.session_state.get("neighbours"),
    buffer_km=buffer_km, tiles=tiles,
)
out = st_folium(fmap, height=680, use_container_width=True,
                returned_objects=["last_clicked"])
click = out.get("last_clicked")
if click:
    new_pin = (round(click["lat"], 5), round(click["lng"], 5))
    if new_pin != st.session_state.get("pin"):
        st.session_state["pin"] = new_pin
        st.rerun()


# --------------------------------------------------------------------------- #
# Prediction (full-width, below the map)
# --------------------------------------------------------------------------- #
pin = st.session_state.get("pin")

if pin is not None:
    lat, lon = pin
    res = predict_site(
        lat, lon, panels=ds.panels, sites=ds.sites,
        month_axis=ds.month_axis, agg=agg, buffer_km=buffer_km,
    )
    st.session_state["neighbours"] = res.neighbours

    st.subheader(f"📍 New site at ({lat:.4f}, {lon:.4f})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Neighbours", res.meta["n_neighbours"])
    c2.metric("Mean distance", f"{res.meta['mean_dist_km']:.1f} km")
    c3.metric("Weighting", "IDW" if agg == "idw" else "Mean")
    for w in res.meta["warnings"]:
        st.warning(w)

    with st.expander("Neighbour sites used", expanded=False):
        st.dataframe(
            res.neighbours[["site_uid", "client", "state", "dist_km"]]
            .round({"dist_km": 1}),
            hide_index=True, width="stretch",
        )

    preds = res.normalized() if normalized else res.predictions
    trajs = res.neighbour_trajectories_normalized() if normalized else res.neighbour_trajectories
    x = ds.month_axis.to_timestamp()

    st.markdown("### Predicted trajectories vs neighbours")
    st.caption("Grey = neighbour sites · Red = predicted new site")
    cols = st.columns(2)
    for i, kpi in enumerate(C.TARGET_KPIS):
        fig = go.Figure()
        for uid, row in trajs[kpi].iterrows():
            fig.add_trace(go.Scatter(
                x=x, y=row.values, mode="lines",
                line=dict(color=NBR_LINE, width=1),
                name=str(uid), showlegend=False,
                hovertemplate=f"<b>{uid}</b><br>%{{x|%b %Y}} · %{{y:,.1f}}<extra></extra>"))
        pser = preds[kpi]
        fig.add_trace(go.Scatter(
            x=x, y=pser.values, mode="lines+markers",
            line=dict(color=PRED_COLOR, width=3),
            marker=dict(size=4), name="Predicted"))

        # Label the final predicted value (last non-NaN point), whatever it is.
        valid = pser.dropna()
        title = C.KPI_LABELS[kpi]
        if not valid.empty:
            last_v = valid.iloc[-1]
            last_x = x[pser.index.get_loc(valid.index[-1])]
            if normalized:
                last_txt = f"{last_v:.2f}"
            elif kpi in C.RATIO_KPIS:
                last_txt = f"${last_v:,.2f}"
            else:
                last_txt = f"{last_v:,.0f}"
            title = f"{C.KPI_LABELS[kpi]} · {last_txt}"
            fig.add_trace(go.Scatter(
                x=[last_x], y=[last_v], mode="markers+text",
                marker=dict(color=PRED_COLOR, size=9),
                text=[last_txt], textposition="top center",
                textfont=dict(color=PRED_COLOR, size=12),
                showlegend=False, hoverinfo="skip"))
        fig.update_layout(
            title=title, height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title=("normalized" if normalized else "value"),
            showlegend=False,
        )
        cols[i % 2].plotly_chart(fig, width="stretch")

    # ----------------------------------------------------------------------- #
    # Membership vs retail share — per cluster site (pick from a dropdown).
    # Aggregated at the plan level, so each is a clean 2-way membership/retail
    # pie: wash share from wash counts, sales share from summed dollars.
    # Totals are over all available months for the chosen site.
    # ----------------------------------------------------------------------- #
    if getattr(ds, "pct_breakdowns", None):
        bd = ds.pct_breakdowns
        nbr_uids = res.neighbours["site_uid"].tolist()
        avail = [u for u in nbr_uids if u in bd["wash"].index or u in bd["sales"].index]

        st.markdown("### Membership vs retail share — cluster sites")
        if not avail:
            st.info("None of the cluster sites at this location carry share data.")
        else:
            st.caption(
                f"{len(avail)} of {res.meta['n_neighbours']} cluster sites carry "
                "share data · pick a site on the map, then choose Totals or "
                "Month-on-month on the right"
            )
            # Fragment: clicking a site on the cluster map reruns ONLY this
            # block, so the page doesn't scroll back up to the main map.
            render_share_pies(
                bd, ds.pct_panels, ds.month_axis, avail, res.neighbours, pin)
