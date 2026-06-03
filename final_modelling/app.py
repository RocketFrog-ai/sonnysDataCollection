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

    # cluster footprints: where neighbours exist and co-move
    for b in bboxes:
        folium.Rectangle(
            bounds=[[b["lat_min"], b["lon_min"]], [b["lat_max"], b["lon_max"]]],
            color="#2E86DE", weight=1, fill=True, fill_opacity=0.12,
            tooltip=f"cluster {b['cluster_id']} · {b['n_sites']} sites · {b['dominant_state']}",
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

buffer_km = st.sidebar.slider("Cluster radius (km)", 5, 50, int(C.BUFFER_KM), 5)

TILES = {
    "Streets (city names)": "CartoDB Voyager",
    "OpenStreetMap": "OpenStreetMap",
    "Minimal light": "CartoDB positron",
    "Dark": "CartoDB dark_matter",
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
                name=uid, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=preds[kpi].values, mode="lines+markers",
            line=dict(color=PRED_COLOR, width=3),
            marker=dict(size=4), name="Predicted"))
        fig.update_layout(
            title=C.KPI_LABELS[kpi], height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title=("normalized" if normalized else "value"),
            showlegend=False,
        )
        cols[i % 2].plotly_chart(fig, width="stretch")
