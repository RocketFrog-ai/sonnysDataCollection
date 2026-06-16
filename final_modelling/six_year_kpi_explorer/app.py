"""Streamlit app: 6-year site KPI explorer.

Drop a pin anywhere on the USA map (or pick an existing site in the sidebar).
The app finds every site within RADIUS_KM and draws each of the KPI
trajectories over the full Jan-2020 → mid-2026 window against:
  • each neighbour's own curve (grey)
  • the pin's actual curve, when the pin sits on an existing site (blue)

Only sites with a full 6 years of data (a record in each of 2020-2025) are shown.

Run:  streamlit run six_year_app.py
"""
from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

DATA_PATH = "data/main-ds.csv"   # KPIs + lat/lon in one file (no separate coords join)
FULL_YEARS = range(2020, 2026)   # "full 6 years" = a record in each of 2020-2025
RADIUS_KM = 20.0            # same neighbourhood radius proven to co-move (app.py)
IDW_EPS_KM = 1e-3           # avoids div-by-zero for a coincident neighbour
EARTH_KM = 6371.0088

SEL_COLOR = "#3498DB"       # selected site actual
PRED_COLOR = "#E74C3C"      # IDW neighbour blend
NBR_LINE = "rgba(130,130,130,0.40)"

# Sustainability quadrant thresholds (on total revenue, annual 2020-2025):
SUS_GROWTH_FLOOR = -4.0     # %/yr trend ≥ this → "overall increasing" (±tolerance)
SUS_BUMP_MAX = 15.0         # detrended volatility ≤ this % → "smooth, not bumpy"
SUS_GREEN = "#27AE60"
SUS_RED = "#C0392B"

# The 8 reported columns (matching the region/state pivot the user verified):
#   raw sums  : ret_revenue (E), ret_wash_count (F), mem_wash_count (H),
#               mem_purchase_count (I), mem_revenue / Membership Revenue (J)
#   derived   : avg_package (G) = J / I,  asp_wash_mem (K) = J / H,
#               asp_wash_ret (L) = E / F
# Each metric: (column, label, unit) where unit is "$" or "count".
METRICS = [
    ("ret_revenue",        "Retail revenue ($)",                    "$"),
    ("ret_wash_count",     "Retail wash count",                     "count"),
    ("avg_package",        "Avg package amount ($)  ·  mem_rev ÷ purchases", "$"),
    ("mem_wash_count",     "Membership wash count",                 "count"),
    ("mem_purchase_count", "Memberships purchased",                 "count"),
    ("mem_revenue",        "Membership revenue ($)",                "$"),
    ("asp_wash_mem",       "ASP per wash — membership ($)  ·  mem_rev ÷ mem washes", "$"),
    ("asp_wash_ret",       "ASP per wash — retail ($)  ·  ret_rev ÷ ret washes",     "$"),
]

st.set_page_config(page_title="6-Year Site KPI Explorer", layout="wide")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)   # lat/lon already inline

    # keep only sites with a record in every one of the FULL_YEARS ("full 6 years")
    have = (
        df.groupby(["client_id", "site_id"])["year"]
        .agg(lambda s: set(FULL_YEARS).issubset(set(s)))
    )
    full = have[have].index
    df = df.set_index(["client_id", "site_id"]).loc[full].reset_index()

    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df["uid"] = df.client_id + " · " + df.site_id.astype(str)
    # derived metrics (user-verified formulas); guard div-by-zero -> NaN
    df["avg_package"] = df.mem_revenue / df.mem_purchase_count.replace(0, np.nan)
    df["asp_wash_mem"] = df.mem_revenue / df.mem_wash_count.replace(0, np.nan)
    df["asp_wash_ret"] = df.ret_revenue / df.ret_wash_count.replace(0, np.nan)
    df["total_rev"] = df.mem_revenue + df.ret_revenue
    return df


@st.cache_data
def site_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site with coordinates and a display name."""
    return (
        df.groupby(["client_id", "site_id"], as_index=False)
        .agg(
            uid=("uid", "first"),
            client_name=("client_name", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            region=("region", "first"),
        )
    )


@st.cache_data
def month_axis(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.date_range(df.date.min(), df.date.max(), freq="MS")


@st.cache_data
def metric_panel(df: pd.DataFrame, col: str, _axis: pd.DatetimeIndex) -> pd.DataFrame:
    """site uid × month matrix for one metric, reindexed to the full axis."""
    p = df.pivot_table(index="uid", columns="date", values=col, aggfunc="mean")
    return p.reindex(columns=_axis)


@st.cache_data
def market_totals(df: pd.DataFrame, uids: tuple, _axis: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly wash counts summed across the given cluster of site uids."""
    g = (
        df[df.uid.isin(uids)]
        .groupby("date")[["mem_wash_count", "ret_wash_count"]]
        .sum()
        .reindex(_axis)
    )
    g["total_wash_count"] = g.mem_wash_count + g.ret_wash_count
    return g


@st.cache_data
def sustainability(df: pd.DataFrame) -> pd.DataFrame:
    """Per-site revenue sustainability from annual (2020-2025) total revenue.

    growth = linear-trend slope as % of mean revenue per year (overall direction)
    bumpy  = std of detrended residuals as % of mean revenue (year-to-year jitter)
    sustainable = growth ≥ floor (overall increasing, ±tolerance) AND bumpy ≤ max.
    """
    a = (
        df[df.year.between(2020, 2025)]
        .groupby(["uid", "year"]).total_rev.sum()
        .unstack("year").reindex(columns=range(2020, 2026)).dropna()
    )
    cname = df.groupby("uid").client_name.first()
    yrs = np.arange(a.shape[1])
    rows = []
    for uid, r in a.iterrows():
        y = r.to_numpy(dtype=float)
        m = y.mean()
        if m <= 0:
            continue
        slope, intc = np.polyfit(yrs, y, 1)
        fit = slope * yrs + intc
        rows.append((uid, slope / m * 100.0, np.std(y - fit) / m * 100.0, cname.get(uid)))
    s = pd.DataFrame(rows, columns=["uid", "growth", "bumpy", "client_name"]).set_index("uid")
    s["sustainable"] = (s.growth >= SUS_GROWTH_FLOOR) & (s.bumpy <= SUS_BUMP_MAX)
    return s


def haversine_km(lat0, lon0, lat, lon) -> np.ndarray:
    r1, n1 = np.radians(lat0), np.radians(lon0)
    r2, n2 = np.radians(lat), np.radians(lon)
    dlat, dlon = r2 - r1, n2 - n1
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return EARTH_KM * 2 * np.arcsin(np.sqrt(a))


df = load_data()
sites = site_table(df)
axis = month_axis(df)
panels = {col: metric_panel(df, col, axis) for col, *_ in METRICS}

X_MIN, X_MAX = axis.min(), axis.max()


# --------------------------------------------------------------------------- #
# Sidebar — controls (+ an optional existing-site picker that snaps the pin)
# --------------------------------------------------------------------------- #
st.sidebar.header("Controls")
st.sidebar.caption("Drop a pin anywhere on the map, or snap it to an existing "
                   "site here.")

# apply a pending map-click site-snap *before* the widgets are instantiated
if "pending_sel" in st.session_state:
    pc, ps = st.session_state.pop("pending_sel")
    st.session_state["sel_client"] = pc
    st.session_state["sel_site"] = ps

clients = sorted(sites["client_id"].unique())
client = st.sidebar.selectbox("Client", clients, key="sel_client")

client_sites = sorted(sites.loc[sites.client_id == client, "site_id"].unique())
if st.session_state.get("sel_site") not in client_sites:
    st.session_state["sel_site"] = client_sites[0]
site_id = st.sidebar.selectbox("Site", client_sites, key="sel_site")

show_actual = st.sidebar.checkbox(
    "Show pin's actual curve", value=True,
    help="Only applies when the pin sits on an existing site.",
)
zoom_to = st.sidebar.checkbox(
    "Zoom map to the pin", value=True,
    help="Uncheck to see the full national footprint.",
)

sel = sites[(sites.client_id == client) & (sites.site_id == site_id)].iloc[0]
sel_uid = sel.uid
sel_lat, sel_lon = float(sel.lat), float(sel.lon)

# --- Resolve the active pin ------------------------------------------------ #
# Changing the sidebar site snaps the pin onto that site (so its actual shows);
# a map click drops a free pin — a hypothetical NEW site — anywhere.
cur_sel = (client, site_id)
if st.session_state.get("base_sel") != cur_sel:
    st.session_state["base_sel"] = cur_sel
    st.session_state["pin"] = (sel_lat, sel_lon)
    st.session_state["pin_uid"] = sel_uid
pin_lat, pin_lon = st.session_state.get("pin", (sel_lat, sel_lon))
pin_uid = st.session_state.get("pin_uid", sel_uid)
is_new_pin = pin_uid is None

st.sidebar.markdown(
    f"**Pin:** {pin_lat:.4f}, {pin_lon:.4f}  \n"
    f"**Window:** {X_MIN:%b %Y} – {X_MAX:%b %Y}"
)

# neighbours within RADIUS_KM of the pin (excluding the pinned site itself)
sites = sites.copy()
sites["dist_km"] = haversine_km(pin_lat, pin_lon, sites.lat.values, sites.lon.values)
neighbours = sites[
    (sites.dist_km > 0) & (sites.dist_km <= RADIUS_KM) & (sites.uid != pin_uid)
].sort_values("dist_km")
nbr_uids = neighbours["uid"].tolist()
nbr_dist = neighbours["dist_km"].to_numpy()


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📈 6-Year Site KPI Explorer")
if is_new_pin:
    who = f"📍 New pin @ ({pin_lat:.4f}, {pin_lon:.4f})"
else:
    cname = "" if pd.isna(sel.client_name) else f"{sel.client_name} — "
    who = f"{cname}{pin_uid}"
_pin_region = sel.region if (not is_new_pin and "region" in sites.columns
                             and pd.notna(getattr(sel, "region", None))) else None
_region_tag = f"  ·  📍 {_pin_region}" if _pin_region else ""
st.caption(
    f"{who}  ·  {len(neighbours)} neighbour(s) within {RADIUS_KM:.0f} km{_region_tag}"
)


# --------------------------------------------------------------------------- #
# Map (hero) — drop a pin anywhere; prediction = blend of its neighbours
# --------------------------------------------------------------------------- #
st.subheader("📍 Drop a pin — explore a site and its neighbours")
st.caption("Click anywhere on the map. The red ring is the 20 km capture radius; "
           "green dots are the neighbours shown below.")

col_map, col_list = st.columns([3, 2])
with col_map:
    fmap = folium.Map(
        location=[pin_lat, pin_lon],
        zoom_start=9 if zoom_to else 4,
        tiles="CartoDB dark_matter", control_scale=True,
    )
    # faint national footprint (non-neighbour sites)
    nbr_set = set(nbr_uids)
    for r in sites.itertuples():
        if r.uid == pin_uid or r.uid in nbr_set:
            continue
        folium.CircleMarker(
            [r.lat, r.lon], radius=2, color="#888888", weight=0,
            fill=True, fill_opacity=0.35,
        ).add_to(fmap)
    # capture radius
    folium.Circle(
        [pin_lat, pin_lon], radius=RADIUS_KM * 1000, color=PRED_COLOR,
        weight=2, fill=True, fill_opacity=0.05,
        tooltip=f"{RADIUS_KM:.0f} km capture radius",
    ).add_to(fmap)
    # neighbours
    for r in neighbours.itertuples():
        folium.CircleMarker(
            [r.lat, r.lon], radius=5, color="#27AE60", fill=True,
            fill_opacity=0.9, tooltip=f"{r.uid} · {r.dist_km:.1f} km",
        ).add_to(fmap)
    # the pin
    folium.Marker(
        [pin_lat, pin_lon],
        tooltip="New pin" if is_new_pin else str(pin_uid),
        icon=folium.Icon(color="red", icon="location-pin", prefix="fa"),
    ).add_to(fmap)

    out = st_folium(
        fmap, height=560, use_container_width=True,
        returned_objects=["last_clicked"], key="folium_map",
    )
    click = out.get("last_clicked") if out else None
    if click:
        new = (round(click["lat"], 5), round(click["lng"], 5))
        if new != (pin_lat, pin_lon):
            d = haversine_km(new[0], new[1], sites.lat.values, sites.lon.values)
            j = int(np.argmin(d))
            # snap to an existing site if the click lands almost on top of one
            st.session_state["pin"] = new
            st.session_state["pin_uid"] = (
                sites.iloc[j].uid if d[j] <= 0.3 else None
            )
            st.rerun()

with col_list:
    if neighbours.empty:
        st.info("No sites within 20 km of this pin — move it to a populated area.")
    else:
        st.dataframe(
            neighbours.assign(
                site=lambda d: d.uid,
                distance_km=lambda d: d.dist_km.round(1),
            )[["site", "distance_km"]].reset_index(drop=True),
            use_container_width=True, height=560,
        )


# --------------------------------------------------------------------------- #
# KPI trajectory charts — neighbours (grey) + actual (blue) + blend (red)
# --------------------------------------------------------------------------- #
def trajectory_figure(col: str, label: str, unit: str) -> go.Figure:
    panel = panels[col]
    vfmt = "$%{y:,.2f}" if unit == "$" else "%{y:,.0f}"
    fig = go.Figure()

    # neighbours first (faded), so the actual/blend lines draw on top
    for uid, dkm in zip(nbr_uids, nbr_dist):
        if uid not in panel.index:
            continue
        fig.add_trace(go.Scatter(
            x=axis, y=panel.loc[uid].values, mode="lines",
            line=dict(color=NBR_LINE, width=1), showlegend=False,
            name=str(uid),
            hovertemplate=f"<b>{uid}</b> ({dkm:.1f} km)<br>"
                          f"%{{x|%b %Y}} · {vfmt}<extra></extra>",
        ))

    # pin's actual — only when the pin sits on an existing site
    if show_actual and not is_new_pin and pin_uid in panel.index:
        fig.add_trace(go.Scatter(
            x=axis, y=panel.loc[pin_uid].values, mode="lines",
            line=dict(color=SEL_COLOR, width=2.5), name="Actual (pin site)",
            hovertemplate=f"<b>Actual</b><br>%{{x|%b %Y}} · {vfmt}<extra></extra>",
        ))

    fig.update_layout(
        title=label, height=320, hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="$" if unit == "$" else "count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28),
    )
    fig.update_xaxes(range=[X_MIN, X_MAX], dtick="M6", tickformat="%b %Y")
    return fig


# --------------------------------------------------------------------------- #
# Market growth — total wash count across the pin's cluster (20 km)
# --------------------------------------------------------------------------- #
# cluster = the pinned site (if it sits on a real site) + its neighbours
cluster_uids = list(nbr_uids)
if not is_new_pin and pin_uid is not None:
    cluster_uids.append(pin_uid)

# region of the cluster (US census region: South / West / Midwest / Northeast).
# Sites within 20 km almost always share one region; if not, label them all.
_cluster_regions = (
    sites.loc[sites.uid.isin(cluster_uids), "region"].dropna().unique().tolist()
)
cluster_region = " / ".join(sorted(_cluster_regions)) if _cluster_regions else "—"

st.subheader(f"📊 Market growth — total wash count (this cluster · {cluster_region})")
st.caption(
    f"Region: **{cluster_region}**  ·  "
    f"Membership, retail, and total monthly wash counts summed across the "
    f"{len(cluster_uids)} site(s) within {RADIUS_KM:.0f} km of the pin — "
    f"is this local market growing?"
)

MKT_SERIES = [
    ("total_wash_count", "Total wash count", "#F1C40F"),
    ("ret_wash_count", "Retail wash count", "#3498DB"),
    ("mem_wash_count", "Membership wash count", "#27AE60"),
]

if not cluster_uids:
    st.info("No sites within 20 km of this pin — move it to a populated area.")
else:
    mkt = market_totals(df, tuple(sorted(cluster_uids)), axis)

    # monthly totals — absolute wash counts, all three series
    gfig = go.Figure()
    for col, lbl, color in MKT_SERIES:
        gfig.add_trace(go.Scatter(
            x=mkt.index, y=mkt[col].values, mode="lines",
            line=dict(color=color, width=3 if col == "total_wash_count" else 2),
            name=lbl,
            hovertemplate=f"<b>{lbl}</b><br>%{{x|%b %Y}} · %{{y:,.0f}}<extra></extra>",
        ))
    gfig.update_layout(
        height=360, hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="wash count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    gfig.update_xaxes(range=[X_MIN, X_MAX], dtick="M6", tickformat="%b %Y")
    st.plotly_chart(gfig, use_container_width=True, key="market_growth")

    # year-on-year % change — full calendar years only (2026 is partial)
    annual = (
        mkt.assign(year=mkt.index.year)
        .groupby("year")[[c for c, *_ in MKT_SERIES]]
        .sum()
        .loc[lambda d: d.index.isin(list(FULL_YEARS))]
    )
    yoy = annual.pct_change().mul(100).dropna(how="all")

    st.markdown("**Year-on-year % change** (each full year vs the prior)")
    yfig = go.Figure()
    for col, lbl, color in MKT_SERIES:
        yfig.add_trace(go.Bar(
            x=yoy.index, y=yoy[col].values, name=lbl, marker_color=color,
            hovertemplate=f"<b>{lbl}</b><br>%{{x}} · %{{y:+.1f}}%<extra></extra>",
        ))
    yfig.update_layout(
        barmode="group", height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="% change YoY",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    yfig.add_hline(y=0, line_width=1, line_color="#888888")
    yfig.update_xaxes(dtick=1)
    st.plotly_chart(yfig, use_container_width=True, key="market_yoy")


st.subheader("Monthly metrics (6 yr) — neighbours · actual")
st.caption(
    "Grey = neighbours · "
    + ("(no actual — new pin)" if is_new_pin else "Blue = pin-site actual")
)

cols = st.columns(2)
for i, (col, label, unit) in enumerate(METRICS):
    cols[i % 2].plotly_chart(
        trajectory_figure(col, label, unit),
        use_container_width=True, key=f"kpi_{col}",
    )


# --------------------------------------------------------------------------- #
# Sustainability quadrant — total revenue, all full-6-year sites (click to drill)
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("🌱 Revenue sustainability — all full-6-year sites")
st.caption(
    "Each dot is a site (total revenue = membership + retail).  "
    f"**Y = growth trend** (%/yr — higher is growing).  "
    f"**X = bumpiness** (detrended year-to-year volatility % — lower is smoother).  "
    f"Green = sustainable: growth ≥ {SUS_GROWTH_FLOOR:.0f}%/yr (overall increasing, "
    f"±tolerance) **and** bumpiness ≤ {SUS_BUMP_MAX:.0f}% (not bumpy).  "
    "Click a dot to see that site's 6-year revenue journey."
)

sus = sustainability(df)
rev_panel = metric_panel(df, "total_rev", axis)
n_green = int(sus.sustainable.sum())
st.caption(f"**{n_green} / {len(sus)}** sites are sustainable "
           f"({n_green / len(sus) * 100:.0f}%).")

sfig = go.Figure()
for flag, color, name in [(True, SUS_GREEN, "Sustainable"),
                          (False, SUS_RED, "Not sustainable")]:
    d = sus[sus.sustainable == flag]
    sfig.add_trace(go.Scattergl(
        x=d.bumpy, y=d.growth, mode="markers", name=name,
        marker=dict(color=color, size=9, opacity=0.6),
        customdata=np.stack([d.index.to_numpy(),
                             d.client_name.fillna("").to_numpy()], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                      "growth %{y:+.1f}%/yr · bumpy %{x:.1f}%<extra></extra>",
    ))
sfig.add_vline(x=SUS_BUMP_MAX, line_dash="dash", line_color="#444")
sfig.add_hline(y=SUS_GROWTH_FLOOR, line_dash="dash", line_color="#444")
sfig.add_annotation(
    x=sus.bumpy.min(), y=sus.growth.max(),
    text="SUSTAINABLE<br>smooth + growing", showarrow=False,
    xanchor="left", yanchor="top", align="left",
    font=dict(color=SUS_GREEN, size=14, family="Arial Black"),
)
sfig.update_layout(
    height=560, margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="revenue bumpiness  (detrended volatility, %)",
    yaxis_title="revenue growth trend  (%/yr)",
    legend=dict(orientation="h", yanchor="bottom", y=-0.16),
)
event = st.plotly_chart(
    sfig, use_container_width=True, key="sus_scatter",
    on_select="rerun", selection_mode="points",
)

pts = event.selection.points if event and event.selection else []
if pts:
    sel_uid = pts[0]["customdata"][0]
    row = sus.loc[sel_uid]
    cn = "" if pd.isna(row.client_name) else f"{row.client_name} — "
    st.markdown(
        f"**{cn}{sel_uid}**  ·  growth {row.growth:+.1f}%/yr  ·  "
        f"bumpiness {row.bumpy:.1f}%  ·  "
        f"{'🌱 sustainable' if row.sustainable else '⚠️ not sustainable'}"
    )
    jfig = go.Figure(go.Scatter(
        x=axis, y=rev_panel.loc[sel_uid].values, mode="lines+markers",
        line=dict(color=SUS_GREEN if row.sustainable else SUS_RED, width=2.5),
        marker=dict(size=3),
        hovertemplate="%{x|%b %Y} · $%{y:,.0f}<extra></extra>",
    ))
    jfig.update_layout(
        height=340, margin=dict(l=10, r=10, t=36, b=10),
        title="6-year total revenue journey", yaxis_title="total revenue ($)",
    )
    jfig.update_xaxes(range=[X_MIN, X_MAX], dtick="M6", tickformat="%b %Y")
    st.plotly_chart(jfig, use_container_width=True, key="sus_journey")
else:
    st.info("Click a dot above to see that site's 6-year total-revenue journey.")


# --------------------------------------------------------------------------- #
# Animated demo — how the X (bumpiness) and Y (growth) axes are derived
# --------------------------------------------------------------------------- #
st.markdown("---")
st.subheader("🎬 How the two axes are built — animated demo")
st.caption(
    "One real site's 6 yearly total-revenue points become the **Y (growth)** "
    "and **X (bumpiness)** used in the scatter above. Press ▶ Play to step through it."
)

# pick an illustrative sustainable site (smooth + clearly growing)
_cand = sus[sus.sustainable].sort_values("bumpy")
ex_uid = _cand.index[len(_cand) // 4] if len(_cand) else sus.index[0]
_yv = (df[df.uid == ex_uid].groupby("year").total_rev.sum()
       .reindex(range(2020, 2026)))
yr_lbl = [str(y) for y in _yv.index]
yv = _yv.to_numpy(dtype=float)
xi = np.arange(len(yv))
_slope, _intc = np.polyfit(xi, yv, 1)
fit = _slope * xi + _intc
mrev = yv.mean()
ex_growth = _slope / mrev * 100.0
ex_bumpy = np.std(yv - fit) / mrev * 100.0

# residual segments (vertical gaps from each point to the trend line)
rx, ry = [], []
for lbl, a_, f_ in zip(yr_lbl, yv, fit):
    rx += [lbl, lbl, None]
    ry += [a_, f_, None]

# 3 fixed traces: points (always), trend line, residuals
pts_tr = go.Scatter(x=yr_lbl, y=yv, mode="markers+lines", name="actual revenue",
                    marker=dict(size=13, color="#2C3E50"),
                    line=dict(color="rgba(44,62,80,0.25)", width=1))
trend_tr = go.Scatter(x=yr_lbl, y=fit, mode="lines", name="trend line (slope = growth)",
                      line=dict(color=SUS_GREEN, width=4), visible=False)
resid_tr = go.Scatter(x=rx, y=ry, mode="lines", name="residuals (gap to trend)",
                      line=dict(color="#FF5A4D", width=3, dash="dot"), visible=False)

_formula = (f"<b>Y · growth</b> = slope ÷ mean revenue = <b>{ex_growth:+.1f}%/yr</b><br>"
            f"<b>X · bumpiness</b> = std(residuals) ÷ mean revenue = <b>{ex_bumpy:.1f}%</b>")
_box = dict(xref="paper", yref="paper", x=0.02, y=0.97, xanchor="left", yanchor="top",
            align="left", showarrow=False, bordercolor="#FFFFFF", borderwidth=1,
            borderpad=10, bgcolor="rgba(20,24,28,0.92)",
            font=dict(size=15, color="#FFFFFF"))

frames = [
    go.Frame(name="1", traces=[1, 2],
             data=[go.Scatter(visible=False), go.Scatter(visible=False)],
             layout=go.Layout(title="Step 1 · Plot the 6 yearly total-revenue points")),
    go.Frame(name="2", traces=[1, 2],
             data=[go.Scatter(visible=True), go.Scatter(visible=False)],
             layout=go.Layout(title="Step 2 · Fit the trend line — its slope is the GROWTH (Y)")),
    go.Frame(name="3", traces=[1, 2],
             data=[go.Scatter(visible=True), go.Scatter(visible=True)],
             layout=go.Layout(title="Step 3 · Residuals = vertical gap from each point to the line")),
    go.Frame(name="4", traces=[1, 2],
             data=[go.Scatter(visible=True), go.Scatter(visible=True)],
             layout=go.Layout(
                 title="Step 4 · Y = growth from the slope · X = bumpiness from the residual spread",
                 annotations=[{**_box, "text": _formula}])),
]

play = dict(
    type="buttons", showactive=False, x=0.0, y=-0.18, xanchor="left",
    buttons=[
        dict(label="▶ Play", method="animate",
             args=[None, dict(frame=dict(duration=1400, redraw=True),
                              fromcurrent=True, transition=dict(duration=500))]),
        dict(label="⟲ Reset", method="animate",
             args=[["1"], dict(frame=dict(duration=0, redraw=True), mode="immediate")]),
    ],
)
slider = dict(active=0, x=0.18, len=0.8, y=-0.16,
              currentvalue=dict(prefix="Step "),
              steps=[dict(label=f.name, method="animate",
                          args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                               mode="immediate")]) for f in frames])

demo = go.Figure(data=[pts_tr, trend_tr, resid_tr], frames=frames)
demo.update_layout(
    title=dict(text="Step 1 · Plot the 6 yearly total-revenue points",
               x=0.0, xanchor="left", y=0.97, font=dict(size=18)),
    height=560, margin=dict(l=10, r=10, t=110, b=80),
    yaxis_title="total revenue ($)", xaxis_title="year",
    updatemenus=[play], sliders=[slider],
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0.0,
                bgcolor="rgba(0,0,0,0)"),
)
cn_ex = "" if pd.isna(sus.loc[ex_uid].client_name) else f"{sus.loc[ex_uid].client_name} — "
st.caption(f"Example site: **{cn_ex}{ex_uid}**")
st.plotly_chart(demo, use_container_width=True, key="axis_demo")
