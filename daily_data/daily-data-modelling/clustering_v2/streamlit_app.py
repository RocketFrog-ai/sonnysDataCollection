"""
Streamlit UI for clustering_v2 greenfield projections.

Calendar-year Ridge/RF charts reuse Matplotlib helpers from ``project_site``; maps and panels use Plotly.
Training panels are ``data_paths.LESS_THAN_CLUSTERING_READY_CSV`` and ``MASTER_MORE_THAN_2YRS_CSV``
(``daily_data/daily-data-modelling/{less_than-2yrs-clustering-ready,master_more_than-2yrs}.csv``).

  cd daily_data/daily-data-modelling/clustering_v2
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

V2_DIR = Path(__file__).resolve().parent

# Default operating expense ($ / operational year): Y1 $1M; Y2 +25%; Y3 +15%; Y4 +10% (each vs prior year).
OPEX_DEFAULT_Y1 = 1_000_000.0
OPEX_DEFAULT_Y2 = OPEX_DEFAULT_Y1 * 1.25
OPEX_DEFAULT_Y3 = OPEX_DEFAULT_Y2 * 1.15
OPEX_DEFAULT_Y4 = OPEX_DEFAULT_Y3 * 1.10

if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from data_paths import (  # noqa: E402 — clustering_v2 training CSVs (same as build_v2)
    LESS_THAN_CLUSTERING_READY_CSV,
    MASTER_MORE_THAN_2YRS_CSV,
    MODELLING_DATA_DIR,
)

import project_site as ps  # noqa: E402
import project_site_quantile as psq  # noqa: E402

PATH_LESS_THAN = LESS_THAN_CLUSTERING_READY_CSV
PATH_MORE_THAN = MASTER_MORE_THAN_2YRS_CSV
MODELLING_DIR = MODELLING_DATA_DIR


def _quantile_models_available() -> bool:
    root = V2_DIR / "models_quantile"
    for cohort in ("less_than", "more_than"):
        d = root / cohort
        for fn in ("q10.joblib", "q50.joblib", "q90.joblib", "feature_order.json"):
            if not (d / fn).is_file():
                return False
    return True


def _haversine_vec_m(lat0: float, lon0: float, lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    r = 6371.0088
    la1 = np.radians(lat0)
    lo1 = np.radians(lon0)
    la2 = np.radians(lat_arr.astype(float))
    lo2 = np.radians(lon_arr.astype(float))
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2.0) ** 2
    return (2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))).astype(float)


@st.cache_data(show_spinner=True)
def _load_less_than_panel() -> pd.DataFrame:
    if not PATH_LESS_THAN.is_file():
        raise FileNotFoundError(str(PATH_LESS_THAN))
    return pd.read_csv(
        PATH_LESS_THAN,
        usecols=[
            "site_client_id",
            "latitude",
            "longitude",
            "calendar_day",
            "wash_count_total",
            "address",
            "dbscan_cluster_12km",
        ],
    )


@st.cache_data(show_spinner=True)
def _load_more_than_panel() -> pd.DataFrame:
    if not PATH_MORE_THAN.is_file():
        raise FileNotFoundError(str(PATH_MORE_THAN))
    return pd.read_csv(
        PATH_MORE_THAN,
        usecols=["site_client_id", "latitude", "longitude", "calendar_day", "wash_count_total", "Address"],
    )


# Peers must lie within this great-circle distance of the **projected** site (new location).
MAX_PEER_KM_FROM_PROJECTION = 20.0
# Safety cap for very dense markets (table + chart); sorted nearest-first.
MAX_PEERS_LIST_DISPLAY = 150

# Plotly USA centroid map: selection is read at the start of ``main()`` (before lat/lon widgets).
USA_CENTROID_PLOT_KEY = "usa_centroid_plot"


def _lat_lon_from_plotly_usa_map_state(state: Any) -> tuple[float, float] | None:
    """Extract one lat/lon from Streamlit ``st.plotly_chart(..., on_select='rerun')`` widget state."""
    if state is None:
        return None
    try:
        sel = state["selection"]
    except (KeyError, TypeError):
        return None
    pts = sel.get("points") if hasattr(sel, "get") else sel["points"]
    if not pts:
        return None
    lats: list[float] = []
    lons: list[float] = []
    for p0 in pts:
        lat = p0.get("lat")
        lon = p0.get("lon")
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))
    if not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _sync_lat_lon_pick(nlat: float, nlon: float) -> None:
    """Set site coordinates from a map or table pick (clears synthetic random-demo address)."""
    olat = float(st.session_state.lat)
    olon = float(st.session_state.lon)
    if abs(nlat - olat) < 1e-9 and abs(nlon - olon) < 1e-9:
        return
    st.session_state.lat = nlat
    st.session_state.lon = nlon
    st.session_state["synth_addr"] = None
    st.session_state.pop("dual_meta", None)


def _cluster_centroid_from_cohort_block(
    block: dict[str, Any] | None,
) -> tuple[int | None, float | None, float | None]:
    if not block or "error" in block:
        return None, None, None
    cl = block.get("cluster") or {}
    try:
        cid = int(cl["cluster_id"])
        c_lat = float(cl["centroid_lat"])
        c_lon = float(cl["centroid_lon"])
    except (KeyError, TypeError, ValueError):
        return None, None, None
    return cid, c_lat, c_lon


def _group_sites_geo(panel: pd.DataFrame, addr_col: str) -> pd.DataFrame:
    return (
        panel.groupby("site_client_id", sort=False)
        .agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            addr=(addr_col, "first"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["lat", "lon"])
    )


def _peer_sites_from_grouped_within_projection(
    g: pd.DataFrame,
    c_lat: float,
    c_lon: float,
    p_lat: float,
    p_lon: float,
) -> tuple[list[dict[str, Any]], int]:
    """
    Training sites in ``g`` within MAX_PEER_KM_FROM_PROJECTION of the projected point,
    sorted by distance to the projection (nearest first). ``distance_km`` is still
    distance to the cohort **centroid** (for comparison to the DBSCAN assignment).
    Returns (peer dicts, total count before display cap).
    """
    if g.empty:
        return [], 0
    lat = g["lat"].to_numpy(dtype=float, copy=False)
    lon = g["lon"].to_numpy(dtype=float, copy=False)
    d_proj = _haversine_vec_m(p_lat, p_lon, lat, lon)
    d_cent = _haversine_vec_m(c_lat, c_lon, lat, lon)
    ok = d_proj <= (MAX_PEER_KM_FROM_PROJECTION + 1e-6)
    idx = np.flatnonzero(ok)
    n_matched = int(idx.size)
    if n_matched == 0:
        return [], 0
    idx = idx[np.argsort(d_proj[idx])]
    if idx.size > MAX_PEERS_LIST_DISPLAY:
        idx = idx[:MAX_PEERS_LIST_DISPLAY]
    out: list[dict[str, Any]] = []
    for jj in idx:
        jj = int(jj)
        sid = float(g.index[jj])
        row = g.iloc[jj]
        out.append(
            {
                "site_client_id": sid,
                "distance_km": float(d_cent[jj]),
                "km_from_projection": float(d_proj[jj]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "address": str(row["addr"]) if pd.notna(row["addr"]) else "",
            }
        )
    return out, n_matched


def _peer_sites_gt_within_projection(
    panel: pd.DataFrame,
    addr_col: str,
    c_lat: float,
    c_lon: float,
    p_lat: float,
    p_lon: float,
) -> tuple[list[dict[str, Any]], int]:
    """>2y training sites within MAX_PEER_KM_FROM_PROJECTION of the projected site (daily CSV has no cluster id)."""
    g = _group_sites_geo(panel, addr_col)
    return _peer_sites_from_grouped_within_projection(g, c_lat, c_lon, p_lat, p_lon)


def _peer_sites_lt_same_cluster(
    panel: pd.DataFrame,
    cluster_id: int,
    c_lat: float,
    c_lon: float,
    addr_col: str,
    p_lat: float,
    p_lon: float,
) -> tuple[list[dict[str, Any]], int]:
    col = "dbscan_cluster_12km"
    if col not in panel.columns:
        g = _group_sites_geo(panel, addr_col)
        return _peer_sites_from_grouped_within_projection(g, c_lat, c_lon, p_lat, p_lon)
    cc = pd.to_numeric(panel[col], errors="coerce")
    sub = panel.loc[cc.fillna(-999).astype(int) == int(cluster_id)].copy()
    if sub.empty:
        g = _group_sites_geo(panel, addr_col)
        return _peer_sites_from_grouped_within_projection(g, c_lat, c_lon, p_lat, p_lon)
    g = (
        sub.groupby("site_client_id", sort=False)
        .agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            addr=(addr_col, "first"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["lat", "lon"])
    )
    return _peer_sites_from_grouped_within_projection(g, c_lat, c_lon, p_lat, p_lon)


def _peer_rows_with_washes(panel: pd.DataFrame, peers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in peers:
        w = _wash_totals_2024_2025(panel, p["site_client_id"])
        rows.append({**p, "w2024": w[2024], "w2025": w[2025]})
    return rows


def _wash_totals_2024_2025(panel: pd.DataFrame, site_client_id: float) -> dict[int, float]:
    s = panel.loc[panel["site_client_id"].astype(float) == float(site_client_id)].copy()
    if s.empty:
        return {2024: 0.0, 2025: 0.0}
    s["cy"] = pd.to_datetime(s["calendar_day"], errors="coerce").dt.year
    s = s.loc[s["cy"].isin([2024, 2025])]
    sums = s.groupby("cy", sort=False)["wash_count_total"].sum()
    out = {2024: 0.0, 2025: 0.0}
    for y in (2024, 2025):
        if y in sums.index and pd.notna(sums.loc[y]):
            out[y] = float(max(sums.loc[y], 0.0))
    return out


def _fig_cluster_peers_map(
    qlat: float,
    qlon: float,
    cent_lt: tuple[float, float] | None,
    cent_gt: tuple[float, float] | None,
    peers_lt: list[dict[str, Any]],
    peers_gt: list[dict[str, Any]],
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=[qlat],
            lon=[qlon],
            mode="markers+text",
            text=["Projected"],
            textposition="top center",
            name="Projected site",
            marker=dict(size=16, color="#dc2626", symbol="star", line=dict(width=1, color="white")),
            hovertemplate="Projected site<br>lat=%{lat:.5f}, lon=%{lon:.5f}<extra></extra>",
        )
    )
    if cent_lt is not None:
        clt_lat, clt_lon = cent_lt
        fig.add_trace(
            go.Scattergeo(
                lat=[clt_lat],
                lon=[clt_lon],
                mode="markers+text",
                text=["<2y cl"],
                textposition="bottom center",
                name="<2y cluster centroid",
                marker=dict(size=14, color="#1d4ed8", symbol="x", line=dict(width=1, color="#172554")),
                hovertemplate="<2y centroid<br>%{lat:.5f}, %{lon:.5f}<extra></extra>",
            )
        )
    if cent_gt is not None:
        cgt_lat, cgt_lon = cent_gt
        fig.add_trace(
            go.Scattergeo(
                lat=[cgt_lat],
                lon=[cgt_lon],
                mode="markers+text",
                text=[">2y cl"],
                textposition="top center",
                name=">2y cluster centroid",
                marker=dict(size=14, color="#166534", symbol="x", line=dict(width=1, color="#14532d")),
                hovertemplate=">2y centroid<br>%{lat:.5f}, %{lon:.5f}<extra></extra>",
            )
        )
    if peers_lt:
        fig.add_trace(
            go.Scattergeo(
                lat=[p["lat"] for p in peers_lt],
                lon=[p["lon"] for p in peers_lt],
                mode="markers",
                name=f"<2y cluster peers (n={len(peers_lt)})",
                marker=dict(size=8, color="#3b82f6", opacity=0.75, line=dict(width=0.5, color="white")),
                text=[f"id {int(p['site_client_id'])}" for p in peers_lt],
                hovertemplate=(
                    "%{text}<br>from projection: %{customdata[0]:.2f} km · to <2y centroid: %{customdata[1]:.2f} km"
                    "<br>%{lat:.5f}, %{lon:.5f}<extra></extra>"
                ),
                customdata=list(
                    zip([p["km_from_projection"] for p in peers_lt], [p["distance_km"] for p in peers_lt])
                ),
            )
        )
    if peers_gt:
        fig.add_trace(
            go.Scattergeo(
                lat=[p["lat"] for p in peers_gt],
                lon=[p["lon"] for p in peers_gt],
                mode="markers",
                name=f">2y centroid peers (n={len(peers_gt)})",
                marker=dict(size=8, color="#22c55e", opacity=0.75, line=dict(width=0.5, color="white")),
                text=[f"id {int(p['site_client_id'])}" for p in peers_gt],
                hovertemplate=(
                    "%{text}<br>from projection: %{customdata[0]:.2f} km · to >2y centroid: %{customdata[1]:.2f} km"
                    "<br>%{lat:.5f}, %{lon:.5f}<extra></extra>"
                ),
                customdata=list(
                    zip([p["km_from_projection"] for p in peers_gt], [p["distance_km"] for p in peers_gt])
                ),
            )
        )
    lats = [qlat]
    lons = [qlon]
    if cent_lt is not None:
        lats.append(float(cent_lt[0]))
        lons.append(float(cent_lt[1]))
    if cent_gt is not None:
        lats.append(float(cent_gt[0]))
        lons.append(float(cent_gt[1]))
    lats.extend(p["lat"] for p in peers_lt)
    lons.extend(p["lon"] for p in peers_lt)
    lats.extend(p["lat"] for p in peers_gt)
    lons.extend(p["lon"] for p in peers_gt)
    pad = 0.55
    lat0, lat1 = min(lats) - pad, max(lats) + pad
    lon0, lon1 = min(lons) - pad, max(lons) + pad
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Projected site, assigned centroids, and peer training sites",
            font=dict(size=16, color="#0f172a"),
        ),
        height=480,
        margin=dict(t=56, b=16, l=16, r=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11),
        ),
        geo=dict(
            scope="usa",
            projection_type="mercator",
            showland=True,
            landcolor="#f8fafc",
            coastlinecolor="#94a3b8",
            lataxis_range=[lat0, lat1],
            lonaxis_range=[lon0, lon1],
            countrycolor="#cbd5e1",
            showcountries=True,
        ),
    )
    return fig


def _nice_y_axis_params(values: list[float], *, include_zero: bool = False) -> dict[str, Any]:
    """Readable y-range and dtick; scales per dataset so tight bands get usable bar height."""
    arr = np.array([float(x) for x in values if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return {"range": [0.0, 1.0], "dtick": 0.2}

    raw_lo, raw_hi = float(arr.min()), float(arr.max())
    data_span_raw = max(raw_hi - raw_lo, max(abs(raw_hi), 1.0) * 1e-9)

    if include_zero:
        peak = max(raw_hi, 0.0)
        if peak <= 0:
            return {"range": [0.0, 1.0], "dtick": 0.2}
        lo, hi = 0.0, peak
        mag = max(peak, 1.0)
        pad_top = max(mag * 0.06, data_span_raw * 0.12, 1.0)
        vmin, vmax = 0.0, peak + pad_top
    else:
        lo, hi = raw_lo, raw_hi
        mag = max(abs(hi), abs(lo), 1.0)
        span_data = max(hi - lo, mag * 1e-6)
        # When points sit in a narrow band, widen the *data* window slightly so bars
        # are not pixel-identical across unrelated charts that happen to be close in level.
        span = max(span_data, mag * 0.04)
        extra = span - span_data
        lo_e = lo - 0.5 * extra
        hi_e = hi + 0.5 * extra
        pad = max(span * 0.1, mag * 0.018)
        vmin, vmax = lo_e - pad, hi_e + pad

    span_e = max(vmax - vmin, 1e-9)
    base_steps = [1, 2, 5, 10, 20, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    step: float | None = None
    for s in base_steps:
        n = span_e / float(s)
        if 4.0 <= n <= 11.0:
            step = float(s)
            break
    if step is None:
        for s in reversed(base_steps):
            if span_e / float(s) >= 3.5:
                step = float(s)
                break
    if step is None:
        step = float(base_steps[0])
    if span_e >= 500 and step < 50:
        step = 50.0

    y0 = math.floor(vmin / step) * step
    y1 = math.ceil(vmax / step) * step
    content_span = max(data_span_raw, mag * 0.02, 1e-9)
    if y1 <= y0 + step * 2:
        widened = y0 + step * 4
        if widened - y0 <= max(5 * step, content_span * 3.5 + 4 * step):
            y1 = widened

    # Prefer 5-unit ticks when the band is modest (wash counts ~tens–hundreds, or similar $).
    band = raw_hi - raw_lo
    if (
        25 <= raw_hi <= 2500
        and 25 <= raw_lo <= 2500
        and band <= max(80.0, 0.22 * mag)
        and step < 5
    ):
        step = 5.0
        y0 = math.floor(vmin / step) * step
        y1 = math.ceil(vmax / step) * step
        if y1 <= y0 + step * 2:
            w2 = y0 + step * 4
            if w2 - y0 <= max(5 * step, content_span * 3.5 + 4 * step):
                y1 = w2

    if include_zero:
        y0 = min(y0, 0.0)
        if y1 <= y0 + step * 1.5:
            y1 = y0 + step * 3

    return {"range": [float(y0), float(y1)], "dtick": float(step)}


def _fig_cluster_peer_yearly_bars(rows: list[dict[str, Any]], title: str) -> go.Figure:
    """Grouped 2024 vs 2025 bars per peer site (same x order as input)."""
    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text="No peer sites in this cohort",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(template="plotly_white", title=dict(text=_wrap_title_html(title), x=0.5, xanchor="center"))
        return fig
    labels = [f"{int(r['site_client_id'])}<br>{r['km_from_projection']:.1f} km proj" for r in rows]
    y24 = [float(r["w2024"]) for r in rows]
    y25 = [float(r["w2025"]) for r in rows]
    vals = y24 + y25
    y_ax = _nice_y_axis_params(vals, include_zero=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="2024",
            x=labels,
            y=y24,
            marker_color="#475569",
            opacity=0.88,
            text=[f"{int(v):,}" for v in y24],
            textposition="outside",
            textfont=dict(size=9, color="#0f172a"),
            cliponaxis=False,
            hovertemplate="site %{x}<br>2024: %{y:,.0f} washes<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="2025",
            x=labels,
            y=y25,
            marker_color="#0d9488",
            opacity=0.88,
            text=[f"{int(v):,}" for v in y25],
            textposition="outside",
            textfont=dict(size=9, color="#0f172a"),
            cliponaxis=False,
            hovertemplate="site %{x}<br>2025: %{y:,.0f} washes<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f1f5f9",
        barmode="group",
        bargap=0.18,
        bargroupgap=0.06,
        title=dict(text=_wrap_title_html(title), x=0.5, xanchor="center", font=dict(size=14, color="#0f172a")),
        yaxis=dict(
            title="wash_count_total (calendar-year sum)",
            range=y_ax["range"],
            dtick=y_ax["dtick"],
            showgrid=True,
            gridcolor="#cbd5e1",
            griddash="dot",
            tickfont=dict(size=11, color="#334155"),
        ),
        xaxis=dict(tickfont=dict(size=10, color="#334155")),
        height=int(min(920, max(400, 64 + 14 * len(rows)))),
        margin=dict(t=88, l=64, r=28, b=int(min(280, 72 + 5 * len(rows)))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
    )
    return fig


def _calendar_wash_totals(resp: dict[str, Any]) -> list[float] | None:
    cy = resp.get("calendar_year_washes") or {}
    keys = ("year_1", "year_2", "year_3", "year_4")
    if not all(k in cy for k in keys):
        return None
    return [float(cy[k]) for k in keys]


def _effective_dollar_per_wash(prices: list[float], pcts: list[float]) -> float:
    """Blended revenue per wash: Σ price_i × (pct_i / 100)."""
    return float(sum(p * max(c, 0.0) / 100.0 for p, c in zip(prices, pcts)))


def _fig_revenue_vs_opex_bars(
    year_labels: list[str],
    revenue: list[float],
    opex: list[float],
    title: str,
) -> go.Figure:
    y_ax = _nice_y_axis_params(
        [float(x) for x in revenue] + [float(x) for x in opex],
        include_zero=True,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Revenue ($) bars",
            x=year_labels,
            y=revenue,
            marker_color="#0284c7",
            opacity=0.55,
            width=0.45,
            text=[f"${v:,.0f}" for v in revenue],
            textposition="outside",
            textfont=dict(size=11, color="#0f172a"),
            cliponaxis=False,
            hovertemplate="%{x}<br>Revenue $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Expense ($) bars",
            x=year_labels,
            y=opex,
            marker_color="#ea580c",
            opacity=0.55,
            width=0.45,
            text=[f"${v:,.0f}" for v in opex],
            textposition="outside",
            textfont=dict(size=11, color="#0f172a"),
            cliponaxis=False,
            hovertemplate="%{x}<br>Expense $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Revenue trace",
            x=year_labels,
            y=revenue,
            mode="lines+markers",
            line=dict(color="#0c4a6e", width=2.5),
            marker=dict(size=9, color="#0c4a6e", symbol="circle", line=dict(width=1, color="white")),
            hovertemplate="%{x}<br>Revenue $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Expense trace",
            x=year_labels,
            y=opex,
            mode="lines+markers",
            line=dict(color="#9a3412", width=2.5),
            marker=dict(size=9, color="#9a3412", symbol="diamond", line=dict(width=1, color="white")),
            hovertemplate="%{x}<br>Expense $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        barmode="overlay",
        bargap=0.08,
        title=dict(text=_wrap_title_html(title), x=0.5, xanchor="center", font=dict(size=14, color="#0f172a")),
        yaxis=dict(
            title="Dollars ($)",
            range=y_ax["range"],
            dtick=y_ax["dtick"],
            showgrid=True,
            gridcolor="#e2e8f0",
            tickfont=dict(size=11, color="#334155"),
        ),
        xaxis=dict(title="Operational year (projection)"),
        height=440,
        margin=dict(t=88, l=56, r=24, b=52),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
    )
    return fig


def _load_centroids(cohort: str) -> list[dict[str, Any]]:
    p = ps.MODELS_DIR / cohort / "cluster_centroids_12km.json"
    return json.loads(p.read_text())["centroids"]


def _destination_point(lat: float, lon: float, dist_km: float, bearing_rad: float) -> tuple[float, float]:
    r_earth = 6371.0088
    lat1, lon1 = math.radians(lat), math.radians(lon)
    d_r = dist_km / r_earth
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(bearing_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing_rad) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _random_point_in_disk(lat0: float, lon0: float, radius_km: float) -> tuple[float, float]:
    ang = random.random() * 2 * math.pi
    rad = radius_km * math.sqrt(random.random())
    return _destination_point(lat0, lon0, rad, ang)


@st.cache_data
def _cohort_overlap_pairs(radius_km: float = 12.0) -> tuple[list[dict], list[dict], list[tuple[dict, dict, float]]]:
    lt = _load_centroids("less_than")
    gt = _load_centroids("more_than")
    pairs: list[tuple[dict, dict, float]] = []
    max_d = 2 * radius_km
    for a in lt:
        for b in gt:
            d = ps._haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= max_d + 1e-6:
                pairs.append((a, b, d))
    return lt, gt, pairs


def _sample_dual_cohort_point(
    pairs: list[tuple[dict, dict, float]],
    radius_km: float = 12.0,
    max_trials: int = 8000,
) -> tuple[float, float, dict[str, Any]] | None:
    if not pairs:
        return None
    for _ in range(max_trials):
        lt_c, gt_c, _ = random.choice(pairs)
        lat, lon = _random_point_in_disk(lt_c["lat"], lt_c["lon"], radius_km)
        if ps._haversine_km(lat, lon, gt_c["lat"], gt_c["lon"]) <= radius_km:
            meta = {
                "less_than_cluster_id": int(lt_c["cluster_id"]),
                "more_than_cluster_id": int(gt_c["cluster_id"]),
                "less_than_centroid": (lt_c["lat"], lt_c["lon"]),
                "more_than_centroid": (gt_c["lat"], gt_c["lon"]),
            }
            return lat, lon, meta
    return None


def _synthetic_address(lat: float, lon: float, meta: dict[str, Any]) -> str:
    n = random.randint(100, 9999)
    return (
        f"{n} Demo Wash Ln, Synthetic USA "
        f"(<2y cl {meta['less_than_cluster_id']}, >2y cl {meta['more_than_cluster_id']}; "
        f"{lat:.5f}, {lon:.5f})"
    )


def _wrap_title_html(title: str, first_line_max: int = 54) -> str:
    """Break long titles so Plotly does not truncate; use HTML line breaks."""
    t = title.strip()
    if len(t) <= first_line_max:
        return t
    sep = " — "
    if sep in t and t.index(sep) < first_line_max + 8:
        a, b = t.split(sep, 1)
        return f"{a}{sep}<br>{b.strip()}"
    cut = t.rfind(" ", 12, first_line_max)
    if cut == -1:
        cut = first_line_max
    return t[:cut] + "<br>" + t[cut:].lstrip()


def _build_usa_centroid_map(lt: list[dict], gt: list[dict], site: tuple[float, float] | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lat=[c["lat"] for c in lt],
            lon=[c["lon"] for c in lt],
            mode="markers",
            marker=dict(size=7, color="#2563eb", opacity=0.75, line=dict(width=0.5, color="white")),
            name="<2y centroids",
            text=[f"<2y cl {c['cluster_id']} (n={c.get('size', '?')})" for c in lt],
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scattergeo(
            lat=[c["lat"] for c in gt],
            lon=[c["lon"] for c in gt],
            mode="markers",
            marker=dict(size=7, color="#16a34a", opacity=0.75, line=dict(width=0.5, color="white")),
            name=">2y centroids",
            text=[f">2y cl {c['cluster_id']} (n={c.get('size', '?')})" for c in gt],
            hoverinfo="text",
        )
    )
    if site is not None:
        fig.add_trace(
            go.Scattergeo(
                lat=[site[0]],
                lon=[site[1]],
                mode="markers",
                marker=dict(size=14, color="#dc2626", symbol="star", line=dict(width=1, color="white")),
                name="Selected site",
                text=["Selected / projected site"],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        template="plotly_white",
        title=dict(text="12 km DBSCAN train centroids (both cohorts)", font=dict(size=16, color="#0f172a")),
        height=560,
        margin=dict(l=0, r=0, t=64, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e2e8f0",
            font=dict(size=13, color="#0f172a"),
        ),
        font=dict(size=13, color="#0f172a"),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="#f1f5f9",
            coastlinecolor="#94a3b8",
            lakecolor="#e2e8f0",
            showlakes=True,
        ),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Clustering V2 projections", layout="wide")
    if "lat" not in st.session_state:
        st.session_state.lat = 33.95
    if "lon" not in st.session_state:
        st.session_state.lon = -84.55
    # ``st.button`` is only True on the click frame; widget reruns (e.g. revenue radio) must still show results.
    if "tab_proj_ran_once" not in st.session_state:
        st.session_state.tab_proj_ran_once = False
    if "sample_r_km" not in st.session_state:
        st.session_state.sample_r_km = 12.0
    st.title("Clustering V2 — wash projections")

    # Sidebar (part 1): controls that do not include lat/lon widgets — the USA map runs in the main
    # area next so Plotly selection can update ``lat``/``lon`` before those ``number_input``s (part 2).
    with st.sidebar:
        st.header("Inputs")
        method = st.selectbox("Time series method", ["arima", "holt_winters", "blend"], index=0)
        allow_far = st.checkbox(
            "Allow nearest centroid > 20 km",
            value=False,
            help="Matches `--allow-distant-nearest-cluster` in project_site.py",
        )
        use_address = st.radio("Location mode", ["Lat / lon", "Address (geocode)"], index=0)

    # Random sampler runs **before** the USA map so the red star and lat/lon stay in sync on the same rerun.
    with st.sidebar:
        st.divider()
        st.subheader("Random demo site")
        st.slider(
            "Max distance to each cohort centroid (km)",
            min_value=6.0,
            max_value=20.0,
            step=1.0,
            key="sample_r_km",
            help="Sampled point lies in the intersection of this-radius disks around one <2y and one >2y centroid.",
        )
        _sr = float(st.session_state.sample_r_km)
        if st.button("Sample random dual-cohort location"):
            _lt, _gt, _pairs = _cohort_overlap_pairs(_sr)
            sp = _sample_dual_cohort_point(_pairs, radius_km=_sr)
            if sp is None:
                st.error("Could not sample a dual-cohort point (no overlapping centroid pairs?).")
            else:
                rlat, rlon, meta = sp
                st.session_state.lat = float(rlat)
                st.session_state.lon = float(rlon)
                st.session_state["synth_addr"] = _synthetic_address(rlat, rlon, meta)
                st.session_state["dual_meta"] = meta
                # Drop Plotly selection so a stale map pick cannot overwrite the new sample on this or the next run.
                st.session_state.pop(USA_CENTROID_PLOT_KEY, None)
                st.success("Updated lat/lon. Use **Lat / lon** mode and **Run projections**.")
        if "synth_addr" in st.session_state:
            st.text(st.session_state["synth_addr"])

    sample_r = float(st.session_state.sample_r_km)
    lt, gt, pairs = _cohort_overlap_pairs(sample_r)
    rf_ok = ps._rf_models_available()

    tab_map, tab_proj = st.tabs(["USA map — centroids", "Projections"])

    lat_cur = float(st.session_state.lat)
    lon_cur = float(st.session_state.lon)
    site: tuple[float, float] | None = None
    if use_address == "Lat / lon":
        site = (lat_cur, lon_cur)
    with tab_map:
        _map_event = st.plotly_chart(
            _build_usa_centroid_map(lt, gt, site),
            use_container_width=True,
            key=USA_CENTROID_PLOT_KEY,
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )
        _picked = _lat_lon_from_plotly_usa_map_state(_map_event) or _lat_lon_from_plotly_usa_map_state(
            st.session_state.get(USA_CENTROID_PLOT_KEY)
        )
        if _picked is not None:
            _sync_lat_lon_pick(_picked[0], _picked[1])
        st.info(
            f"{len(lt)} <2y centroids (blue), {len(gt)} >2y centroids (green). "
            f"At radius {sample_r:.0f} km: {len(pairs)} centroid pairs with centers ≤{2 * sample_r:.0f} km apart."
        )

    with st.sidebar:
        st.divider()
        st.number_input("Latitude", format="%.6f", key="lat")
        st.number_input("Longitude", format="%.6f", key="lon")
        address = st.text_input(
            "Address",
            placeholder="7021 Executive Center Dr, Brentwood, TN 37027",
            key="addr_field",
        )

        st.divider()
        st.subheader("Wash packages & expenses")
        n_tiers = int(
            st.number_input(
                "Number of package tiers",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key="n_wash_tiers",
            )
        )
        _tier_prices = [20.0, 15.0, 12.0, 10.0, 8.0, 7.0]
        _tier_pcts = [60.0, 40.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(n_tiers):
            c1, cx, c2 = st.columns([2.1, 0.5, 2.1])
            with c1:
                st.number_input(
                    f"Tier {i + 1} price ($)",
                    min_value=0.0,
                    value=float(_tier_prices[i]),
                    step=0.5,
                    key=f"wash_price_{i}",
                )
            with cx:
                st.markdown("**×**")
            with c2:
                st.number_input(
                    f"Tier {i + 1} % of users",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(_tier_pcts[i]),
                    step=1.0,
                    key=f"wash_pct_{i}",
                )
        st.markdown("**Operating expense ($ per operational year)**")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.number_input(
                "Year 1", min_value=0.0, value=OPEX_DEFAULT_Y1, step=5000.0, key="opex_y1"
            )
            st.number_input(
                "Year 3", min_value=0.0, value=OPEX_DEFAULT_Y3, step=5000.0, key="opex_y3"
            )
        with ex2:
            st.number_input(
                "Year 2", min_value=0.0, value=OPEX_DEFAULT_Y2, step=5000.0, key="opex_y2"
            )
            st.number_input(
                "Year 4", min_value=0.0, value=OPEX_DEFAULT_Y4, step=5000.0, key="opex_y4"
            )

        st.divider()
        run = st.button("Run projections", type="primary")

    lat_cur = float(st.session_state.lat)
    lon_cur = float(st.session_state.lon)

    with tab_proj:
        if not run and not st.session_state.tab_proj_ran_once:
            st.write("Configure the sidebar and click **Run projections**.")
            return

        try:
            if use_address == "Address (geocode)":
                if not address.strip():
                    st.error("Enter an address or switch to Lat/lon.")
                    return
                lat_in, lon_in, addr_res = None, None, address.strip()
            else:
                lat_in, lon_in, addr_res = lat_cur, lon_cur, st.session_state.get("synth_addr")
        except (TypeError, ValueError):
            st.error("Invalid latitude or longitude.")
            return

        if run:
            st.session_state.tab_proj_ran_once = True

        cap_kw = {"allow_nearest_cluster_beyond_distance_cap": allow_far}
        loc_title = addr_res or (f"{lat_in:.5f}, {lon_in:.5f}" if lat_in is not None else address)

        with st.spinner("Running projections…"):
            if rf_ok:
                rr0 = ps.run_projection(
                    addr_res if use_address == "Address (geocode)" else None,
                    lat_in,
                    lon_in,
                    method,
                    use_opening_prefix_for_mature_forecast=False,
                    bridge_opening_to_mature_when_prefix=True,
                    level_model="ridge",
                    **cap_kw,
                )
                rr1 = ps.run_projection(
                    addr_res if use_address == "Address (geocode)" else None,
                    lat_in,
                    lon_in,
                    method,
                    use_opening_prefix_for_mature_forecast=True,
                    bridge_opening_to_mature_when_prefix=True,
                    level_model="ridge",
                    **cap_kw,
                )
                rf0 = ps.run_projection(
                    addr_res if use_address == "Address (geocode)" else None,
                    lat_in,
                    lon_in,
                    method,
                    use_opening_prefix_for_mature_forecast=False,
                    bridge_opening_to_mature_when_prefix=True,
                    level_model="rf",
                    **cap_kw,
                )
                rf1 = ps.run_projection(
                    addr_res if use_address == "Address (geocode)" else None,
                    lat_in,
                    lon_in,
                    method,
                    use_opening_prefix_for_mature_forecast=True,
                    bridge_opening_to_mature_when_prefix=True,
                    level_model="rf",
                    **cap_kw,
                )
            else:
                rr0 = ps.run_projection(
                    addr_res if use_address == "Address (geocode)" else None,
                    lat_in,
                    lon_in,
                    method,
                    use_opening_prefix_for_mature_forecast=False,
                    bridge_opening_to_mature_when_prefix=True,
                    level_model="ridge",
                    **cap_kw,
                )
                rr1 = ps.run_projection(
                    addr_res if use_address == "Address (geocode)" else None,
                    lat_in,
                    lon_in,
                    method,
                    use_opening_prefix_for_mature_forecast=True,
                    bridge_opening_to_mature_when_prefix=True,
                    level_model="ridge",
                    **cap_kw,
                )
                rf0 = rf1 = None

        lt_e = "error" in (rr1.get("less_than_2yrs") or {})
        gt_e = "error" in (rr1.get("more_than_2yrs") or {})
        cap_resp = rr0 if (lt_e or gt_e) else rr1
        st.subheader(ps._nearest_cluster_caption(cap_resp))

        if rf_ok and rf0 is not None and rf1 is not None:
            pan = (
                ("Ridge | No <2y TS prefix", rr0),
                ("Ridge | <2y prefix + 24→25 bridge", rr1),
                ("RF | No <2y TS prefix", rf0),
                ("RF | <2y prefix + 24→25 bridge", rf1),
            )
            fig_m = ps._plot_ridge_rf_four_panels(pan, None, loc_title, method, return_figure=True)
            if fig_m is not None:
                st.pyplot(fig_m)
                plt.close(fig_m)
        else:
            st.warning(
                "RF checkpoints not found under `models/*/wash_count_model_12km.rf.joblib`. "
                "Showing Ridge-only comparison (`--plot-two-way`). Run `python build_v2.py` in this folder to train RF."
            )
            pan2 = (
                ("No <2y TS prefix", rr0),
                ("<2y prefix + 24→25 bridge (default)", rr1),
            )
            fig_m = ps._plot_compare_panels(pan2, None, loc_title, method, return_figure=True)
            if fig_m is not None:
                st.pyplot(fig_m)
                plt.close(fig_m)

        st.divider()
        st.subheader("V2 quantile calendar years (q10–q50–q90)")
        st.caption(
            "Quantile bars use **models_quantile/** (q10/q50/q90 from `build_quantile_v2.py`—histogram gradient boosting), "
            "not the Ridge or RF **wash_count** heads in **models/**. TS method (ARIMA / Holt / blend) is shared with the charts above."
        )
        if not _quantile_models_available():
            st.warning(
                "Quantile bundle not found. Expected `clustering_v2/models_quantile/{less_than,more_than}/` "
                "with `q10.joblib`, `q50.joblib`, `q90.joblib`, `feature_order.json`. Run `python build_quantile_v2.py`."
            )
        else:
            inp_q = rr1.get("input") or {}
            try:
                qlat = float(inp_q["lat"])
                qlon = float(inp_q["lon"])
            except (KeyError, TypeError, ValueError):
                qlat, qlon = lat_cur, lon_cur
            q_addr = inp_q.get("address")
            try:
                q_no = psq.build_quantile_projection_response(
                    qlat,
                    qlon,
                    method,
                    q_addr,
                    use_opening_prefix_for_mature_forecast=False,
                    bridge_opening_to_mature_when_prefix=True,
                    **cap_kw,
                )
                q_yes = psq.build_quantile_projection_response(
                    qlat,
                    qlon,
                    method,
                    q_addr,
                    use_opening_prefix_for_mature_forecast=True,
                    bridge_opening_to_mature_when_prefix=True,
                    **cap_kw,
                )
            except Exception as exc:
                st.error(f"Quantile projection failed: {exc}")
            else:
                fig_q = psq._plot_quantile_compare_panels(
                    (
                        ("No <2y q50 TS prefix", q_no),
                        ("<2y q50 prefix + 24→25 bridge (default)", q_yes),
                    ),
                    None,
                    loc_title,
                    method,
                    return_figure=True,
                )
                if fig_q is not None:
                    st.pyplot(fig_q)
                    plt.close(fig_q)
                with st.expander("Quantile response JSON (with q50 prefix)"):
                    st.json(q_yes)

        st.divider()
        st.subheader("Revenue vs operating expenses")
        year_labs = ["Year 1", "Year 2", "Year 3", "Year 4"]
        rev_opts: list[tuple[str, dict[str, Any]]] = [
            ("Ridge + <2y prefix + bridge", rr1),
            ("Ridge no prefix", rr0),
        ]
        if rf_ok and rf1 is not None:
            rev_opts.extend(
                [
                    ("RF + <2y prefix + bridge", rf1),
                    ("RF no prefix", rf0),
                ]
            )
        rev_labels = [a for a, _ in rev_opts]
        rev_by_label = dict(rev_opts)
        prev_pick = st.session_state.get("revenue_wash_scenario")
        if prev_pick is not None and prev_pick not in rev_by_label:
            del st.session_state["revenue_wash_scenario"]
        pick = st.radio(
            "Which projection's calendar-year wash counts drive revenue?",
            rev_labels,
            horizontal=True,
            key="revenue_wash_scenario",
        )
        rev_resp = rev_by_label[pick]
        washes_rev = _calendar_wash_totals(rev_resp)
        nt = int(st.session_state.get("n_wash_tiers", 2))
        prices = [float(st.session_state.get(f"wash_price_{i}", 0.0)) for i in range(nt)]
        pcts = [float(st.session_state.get(f"wash_pct_{i}", 0.0)) for i in range(nt)]
        blend = _effective_dollar_per_wash(prices, pcts)
        pct_sum = float(sum(pcts))
        if abs(pct_sum - 100.0) > 0.51:
            st.caption(
                f"Tier **% of users** sum to **{pct_sum:.1f}%** (not 100). "
                "Blend still uses Σ (price × %/100) exactly as entered."
            )
        opex = [
            float(st.session_state.get("opex_y1", OPEX_DEFAULT_Y1)),
            float(st.session_state.get("opex_y2", OPEX_DEFAULT_Y2)),
            float(st.session_state.get("opex_y3", OPEX_DEFAULT_Y3)),
            float(st.session_state.get("opex_y4", OPEX_DEFAULT_Y4)),
        ]
        if washes_rev is None:
            st.info("Selected scenario has no `calendar_year_washes`; cannot compute revenue.")
        else:
            revenue = [w * blend for w in washes_rev]
            net = [r - e for r, e in zip(revenue, opex)]
            st.dataframe(
                pd.DataFrame(
                    {
                        "Year": year_labs,
                        "Wash count": [int(round(w)) for w in washes_rev],
                        "Blended $/wash": [round(blend, 4)] * 4,
                        "Revenue ($)": [round(r, 2) for r in revenue],
                        "Expense ($)": [round(e, 2) for e in opex],
                        "Net ($)": [round(n, 2) for n in net],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.plotly_chart(
                _fig_revenue_vs_opex_bars(
                    year_labs,
                    revenue,
                    opex,
                    f"Revenue vs expense (grouped bars) — {pick} — {loc_title}",
                ),
                use_container_width=True,
                key="rev_opex_" + hashlib.md5(pick.encode("utf-8")).hexdigest()[:16],
            )

        with st.expander("Why Year 3 can equal Year 4 (Ridge & RF, prefix path)"):
            st.markdown(
                """
**Year 3 vs Year 4 with `<2y` prefix:** Calendar years come from `project_site._append_calendar_year_washes_ridge`:
Years 1–2 sum the `<2y` monthly opening forecast; Years 3–4 sum months **25–36** and **37–48** of the `>2y` mature
monthly track (after the optional opening→mature bridge). With prefix, that mature track is an **ARIMA(1,1,1)**
forecast on **cluster history plus the 24 opening months** (`_series_for_mature_forecast_with_opening_context` +
`_forecast`). The fitted model often **levels off** to an almost constant monthly wash rate, so the two successive
12-month blocks can have **nearly identical sums** (Ridge vs RF can match closely too because they share the **same**
time-series path; only the **level anchor** differs). This is expected dynamics, not a wrong index into
`monthly_projection_mature_25_48`.

**Chart style:** The 2×2 / 2-panel figures use **shared y-axes** (`sharey=True`) like the CLI PNGs, so small
year-to-year differences can look flat; use the **printed values** above each bar for exact totals.
                """
            )

        st.divider()
        st.subheader("Assigned clusters — peer wash totals (2024 vs 2025)")
        st.caption(
            "**DBSCAN 12 km** (`cluster_centroids_12km.json`): the nearest cohort cluster comes from **projected lat/lon only** "
            "in `project_site._nearest_cluster` — **identical for Ridge and RF** (`level_model` only changes the wash head, "
            "not cluster assignment). This section reads **`cluster` from the Ridge + prefix response** (`rr1`); RF’s "
            "clusters match for the same location. Use the RF JSON expander below to inspect RF wash outputs."
        )
        peer_ctx = rr1

        inp = peer_ctx.get("input") or {}
        try:
            p_lat = float(inp["lat"])
            p_lon = float(inp["lon"])
        except (KeyError, TypeError, ValueError):
            p_lat = float(lat_in) if lat_in is not None else float("nan")
            p_lon = float(lon_in) if lon_in is not None else float("nan")
        if not (np.isfinite(p_lat) and np.isfinite(p_lon)):
            st.warning("Could not read projected lat/lon from the selected run; skipping cluster peer panel.")
        else:
            try:
                dflt = _load_less_than_panel()
                dfgt = _load_more_than_panel()
            except FileNotFoundError as exc:
                st.warning(f"Could not load panel CSV: {exc}")
            else:
                lt_block = peer_ctx.get("less_than_2yrs") or {}
                gt_block = peer_ctx.get("more_than_2yrs") or {}
                cid_lt, clt_lat, clt_lon = _cluster_centroid_from_cohort_block(lt_block)
                cid_gt, cgt_lat, cgt_lon = _cluster_centroid_from_cohort_block(gt_block)
                if cid_lt is None and cid_gt is None:
                    st.info("No cluster centroids on this run (cohort errors); peer panel skipped.")
                else:
                    peers_lt: list[dict[str, Any]] = []
                    peers_gt: list[dict[str, Any]] = []
                    n_lt_matched = n_gt_matched = 0
                    if cid_lt is not None and clt_lat is not None and clt_lon is not None:
                        peers_lt, n_lt_matched = _peer_sites_lt_same_cluster(
                            dflt, cid_lt, clt_lat, clt_lon, "address", p_lat, p_lon
                        )
                    if cid_gt is not None and cgt_lat is not None and cgt_lon is not None:
                        peers_gt, n_gt_matched = _peer_sites_gt_within_projection(
                            dfgt, "Address", cgt_lat, cgt_lon, p_lat, p_lon
                        )
                    rows_lt = _peer_rows_with_washes(dflt, peers_lt)
                    rows_gt = _peer_rows_with_washes(dfgt, peers_gt)
                    cent_lt = (clt_lat, clt_lon) if cid_lt is not None and clt_lat is not None and clt_lon is not None else None
                    cent_gt = (cgt_lat, cgt_lon) if cid_gt is not None and cgt_lat is not None and cgt_lon is not None else None
                    if cent_lt is not None or cent_gt is not None or peers_lt or peers_gt:
                        st.plotly_chart(
                            _fig_cluster_peers_map(p_lat, p_lon, cent_lt, cent_gt, peers_lt, peers_gt),
                            use_container_width=True,
                        )
                    st.caption(
                        f"Only training sites **≤ {MAX_PEER_KM_FROM_PROJECTION:.0f} km** (great circle) from the **projected** "
                        "lat/lon are listed. **<2y:** same `dbscan_cluster_12km` as the model; **>2y:** daily CSV has no cluster "
                        "column—peers are all panel sites in that radius (sort: nearest to projection first). "
                        "The number in **“cluster 69”** is the **cluster id**, not a peer count."
                    )
                    if n_lt_matched > len(peers_lt):
                        st.caption(
                            f"<2y: **{n_lt_matched}** sites match the filter; showing the **{len(peers_lt)}** nearest to the "
                            f"projection (display cap {MAX_PEERS_LIST_DISPLAY})."
                        )
                    if n_gt_matched > len(peers_gt):
                        st.caption(
                            f">2y: **{n_gt_matched}** sites within {MAX_PEER_KM_FROM_PROJECTION:.0f} km; showing the "
                            f"**{len(peers_gt)}** nearest to the projection (display cap {MAX_PEERS_LIST_DISPLAY})."
                        )
                    ctab1, ctab2 = st.columns(2)
                    with ctab1:
                        if cid_lt is not None:
                            st.markdown(
                                f"**<2y DBSCAN cluster id {cid_lt}** — {len(rows_lt)} site(s) in table "
                                f"(≤{MAX_PEER_KM_FROM_PROJECTION:.0f} km from projection)"
                            )
                        if rows_lt:
                            df_show = pd.DataFrame(
                                [
                                    {
                                        "site_client_id": f"{int(r['site_client_id'])}",
                                        "km from projection": f"{r['km_from_projection']:.2f}",
                                        "km to <2y centroid": f"{r['distance_km']:.2f}",
                                        "2024": int(round(r["w2024"])),
                                        "2025": int(round(r["w2025"])),
                                    }
                                    for r in rows_lt
                                ]
                            )
                            st.dataframe(df_show, use_container_width=True, hide_index=True)
                        elif cid_lt is not None:
                            st.warning(
                                f"No <2y training sites in this cluster within {MAX_PEER_KM_FROM_PROJECTION:.0f} km of the projection."
                            )
                    with ctab2:
                        if cid_gt is not None:
                            st.markdown(
                                f"**>2y centroid cluster id {cid_gt}** — {len(rows_gt)} site(s) in table "
                                f"(≤{MAX_PEER_KM_FROM_PROJECTION:.0f} km from projection)"
                            )
                        if rows_gt:
                            df_show_g = pd.DataFrame(
                                [
                                    {
                                        "site_client_id": f"{int(r['site_client_id'])}",
                                        "km from projection": f"{r['km_from_projection']:.2f}",
                                        "km to >2y centroid": f"{r['distance_km']:.2f}",
                                        "2024": int(round(r["w2024"])),
                                        "2025": int(round(r["w2025"])),
                                    }
                                    for r in rows_gt
                                ]
                            )
                            st.dataframe(df_show_g, use_container_width=True, hide_index=True)
                        elif cid_gt is not None:
                            st.warning(
                                f"No >2y training sites within {MAX_PEER_KM_FROM_PROJECTION:.0f} km of the projection."
                            )
                    if rows_lt:
                        st.plotly_chart(
                            _fig_cluster_peer_yearly_bars(
                                rows_lt,
                                (
                                    f"<2y cluster id {cid_lt} — ≤{MAX_PEER_KM_FROM_PROJECTION:.0f} km from projection — "
                                    f"{len(rows_lt)} site(s) — wash totals 2024/2025 — {loc_title}"
                                ),
                            ),
                            use_container_width=True,
                        )
                    if rows_gt:
                        st.plotly_chart(
                            _fig_cluster_peer_yearly_bars(
                                rows_gt,
                                (
                                    f">2y (centroid cluster id {cid_gt}) — ≤{MAX_PEER_KM_FROM_PROJECTION:.0f} km from projection — "
                                    f"{len(rows_gt)} site(s) — wash totals 2024/2025 — {loc_title}"
                                ),
                            ),
                            use_container_width=True,
                        )

        with st.expander("Raw JSON (Ridge + prefix run)"):
            st.json(rr1)
        if rf_ok and rf1 is not None:
            with st.expander("Raw JSON (RF + prefix run)"):
                st.json(rf1)


if __name__ == "__main__":
    main()
