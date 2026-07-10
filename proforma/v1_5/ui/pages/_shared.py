"""
Shared constants and data/geo helpers used by more than one Local Market Explorer page
(the Pinpoint-forecast page, the Explore-markets page, and app.py's own dispatcher).

Extracted from app.py during the UI page-split refactor (behavior-preserving code motion —
see proforma/v1_5/ui/app.py for the entrypoint and mode dispatch).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import folium

from proforma.v1_5.models import coldstart as cm

HERE = Path(__file__).resolve().parents[1]  # pages/_shared.py -> proforma/v1_5/ui (one dir deeper than app.py)
_V1_5 = HERE.parent                         # proforma/v1_5
_PROFORMA = HERE.parents[1]                 # proforma
CSV = _PROFORMA / "data" / "panel" / "main-data-v2-stitched.csv"
TYPES_CSV = _PROFORMA / "data" / "ref" / "site_carwash_types.csv"
ARTIFACTS = _V1_5 / "artifacts"
EARTH_KM = 6371.0088
EXPRESS_TYPE = "Express Tunnel"          # the "express only" filter keeps just this primary_carwash_type
EXPRESS_MIN_MONTHS = 30                   # express mode also requires ≥30 monthly records → richer history
# ── corrupted-ASP floor (a real data-feed drop: revenue decays to ~0 while wash_count stays normal) ──
# A site-month is implausible when it has material volume but a near-zero implied price. Drop these rows
# BEFORE pooling the cluster ASP so the bad sites can't halve the $/wash. Wash-weighting means dropping
# cheap rows only ever pulls the ratio toward the healthy majority — never inflates it.
ASP_MIN_WASH = 200       # only judge rows with material volume (≥200 washes) — ignore thin/noisy months
ASP_FLOOR_MEM = 4.0      # $/membership-wash below this @ ≥200 washes ⇒ corrupt (healthy median ~$11)
ASP_FLOOR_RET = 5.0      # $/retail-wash below this @ ≥200 washes ⇒ corrupt (healthy median ~$16)


@st.cache_data(show_spinner=False)
def load_carwash_types():
    """site_key -> primary_carwash_type, keyed (like main-ds) on client_id::site_id. Sites that were 'Unknown'
    and later resolved to Express are already written into site_carwash_types.csv as 'Express Tunnel'."""
    t = pd.read_csv(TYPES_CSV, low_memory=False)
    t["site_key"] = t.client_id.astype(str) + "::" + t.site_id.astype(str)
    t = t.dropna(subset=["primary_carwash_type"]).drop_duplicates("site_key", keep="first")
    return t.set_index("site_key").primary_carwash_type


@st.cache_data(show_spinner="Loading & clustering sites…")
def load_data(express_only=False):
    raw = pd.read_csv(CSV, low_memory=False)
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=1))
    raw["op_start"] = pd.to_datetime(raw["operational_start"], format="%m-%Y", errors="coerce")
    raw["site_key"] = raw.client_id.astype(str) + "::" + raw.site_id.astype(str)

    df = raw.copy()
    asp_r = np.where(df.ret_wash_count > 0, df.ret_revenue / df.ret_wash_count, np.nan)
    asp_m = np.where(df.mem_wash_count > 0, df.mem_revenue / df.mem_wash_count, np.nan)
    df.loc[asp_r > 200, "ret_revenue"] = np.nan
    df.loc[asp_m > 200, "mem_revenue"] = np.nan
    df["tot_wash_count"] = df.mem_wash_count + df.ret_wash_count
    df["tot_revenue"] = df[["mem_revenue", "ret_revenue"]].sum(axis=1, min_count=1)
    df["mem_share_wash"] = np.where(df.tot_wash_count > 0, df.mem_wash_count / df.tot_wash_count, np.nan)

    # car-wash type from the classifier output; "express only" drops Flex / full-service / etc. up front, so the
    # market/cluster/KPI views and the forecast's wash trajectory + level anchor operate on express sites only.
    # (The P&L OPEX% curve — scoped by state/region — and the global campaign popover still read all sites.)
    types = load_carwash_types()
    df["carwash_type"] = df.site_key.map(types)
    if express_only:
        df = df[df.carwash_type == EXPRESS_TYPE].copy()

    site = (
        df.groupby("site_key")
        .agg(client_name=("client_name", "first"), lat=("lat", "first"), lon=("lon", "first"),
             state=("state", "first"), region=("region", "first"), op_start=("op_start", "first"),
             first_obs=("date", "min"), last_obs=("date", "max"), n_obs=("date", "size"))
        .reset_index()
    )
    site["carwash_type"] = site.site_key.map(types)
    site["is_express"] = site.carwash_type.eq(EXPRESS_TYPE)
    site["left_censored"] = site.op_start <= pd.Timestamp("2020-01-01")
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)

    # express only: keep just sites with a richer history (≥30 monthly records) so series, clusters and
    # forecasts are well-grounded — drops thin/young express sites before clustering
    if express_only:
        rich_keys = set(site.site_key[site.n_obs >= EXPRESS_MIN_MONTHS])
        site = site[site.site_key.isin(rich_keys)].reset_index(drop=True)
        df = df[df.site_key.isin(rich_keys)].copy()

    # density-aware "local market" clustering (adaptive 10/20km — won the bake-off vs fixed 20km; see coldstart_forecast.ipynb)
    site["cluster"] = cm.assign_clusters(site, "adaptive")
    return df, site


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def neighbourhood(site, lat, lon, radius_km):
    """Sites within `radius_km` of a free (lat, lon) pin point. Tags entrants relative to the market."""
    d = haversine_km(lat, lon, site.lat.values, site.lon.values)
    nb = site.loc[(d <= radius_km) & site.has_coords].copy()
    nb["dist_km"] = d[(d <= radius_km) & site.has_coords.values]
    nb = nb.sort_values("op_start")
    earliest = nb.op_start.min()
    # an "entrant" opened after the market's earliest site AND is a genuinely-observed opening
    nb["is_entrant"] = (~nb.left_censored) & (nb.op_start > earliest)
    return nb.reset_index(drop=True)


def add_cluster_regions(fmap, site, plat, plon, max_km, color="#3b7dd8", fill_opacity=0.12):
    """Shade each site-cluster within `max_km` of (plat, plon) as a light, borderless circle (same
    colour for all) hugging the cluster's own footprint (centroid → farthest member, min 2 / cap 20 km)."""
    area = site[site.has_coords & (site.cluster >= 0)].copy()
    # drop junk coordinates (near-zero / out-of-range)
    area = area[area.lat.between(-89, 89) & area.lon.between(-179, 179)
                & (area.lat.abs() > 1e-3) & (area.lon.abs() > 1e-3)]
    area["d"] = haversine_km(plat, plon, area.lat.values, area.lon.values)
    area = area[area.d <= max_km]
    for cid, cg in area.groupby("cluster"):
        clat, clon = float(cg.lat.mean()), float(cg.lon.mean())
        spread = float(haversine_km(clat, clon, cg.lat.values, cg.lon.values).max())   # cluster's own radius
        r_km = min(max(spread + 1.0, 2.0), 20.0)
        folium.Circle([clat, clon], radius=r_km * 1000, weight=0,
                      fill=True, fill_color=color, fill_opacity=fill_opacity,
                      tooltip=f"cluster {int(cid)} · {len(cg)} sites · ~{r_km:.0f} km").add_to(fmap)


def add_all_site_dots(fmap, sites, color="#00E5FF", edge="#0086c3"):
    """Every site in the passed `sites` frame (the active, possibly express-filtered, universe) as a bright,
    identifiable dot — regardless of distance, so the whole universe is on the map and you can pan/zoom to any
    market. Junk coords (near-zero / out-of-range) dropped. itertuples + canvas-friendly markers keep ~2k dots fast."""
    s = sites[sites.has_coords]
    s = s[s.lat.between(-89, 89) & s.lon.between(-179, 179)
          & (s.lat.abs() > 1e-3) & (s.lon.abs() > 1e-3)]
    for r in s.itertuples(index=False):
        folium.CircleMarker([r.lat, r.lon], radius=4, color=edge, fill=True, fill_color=color,
                            fill_opacity=0.95, weight=1, tooltip=str(r.client_name)).add_to(fmap)


def interesting_pins(site):
    """Sites that sit in a multi-site market with ≥1 genuine entrant — good random picks."""
    geo = site[site.has_coords & (site.cluster >= 0)]
    good = []
    for _, g in geo.groupby("cluster"):
        if len(g) >= 2 and ((~g.left_censored) & (g.op_start > g.op_start.min())).any():
            good += g.site_key.tolist()
    return good or geo.site_key.tolist()


def deseason_pct_change(df, incumbent_key, metric, entry_date, pre=(-6, -1), post=(1, 12)):
    """Deseasonalized % change (post vs pre) for one incumbent around an entry date."""
    s = df.loc[df.site_key == incumbent_key].set_index("date")[metric].sort_index()
    if s.empty:
        return np.nan
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    moy = s.index.month
    factor = pd.Series(s.values, index=moy).groupby(level=0).transform("mean") / np.nanmean(s.values)
    des = pd.Series(s.values / factor.values, index=s.index)
    k = (des.index.year - entry_date.year) * 12 + (des.index.month - entry_date.month)
    o = pd.Series(des.values, index=k)
    a = o[(o.index >= pre[0]) & (o.index <= pre[1])].mean()
    b = o[(o.index >= post[0]) & (o.index <= post[1])].mean()
    if not np.isfinite(a) or a == 0:
        return np.nan
    return (b - a) / a * 100


@st.cache_data(show_spinner=False)
def pick_default_pin(_site, _df, _pins):
    """First showcase: a clean local market — a handful of incumbents with measurable response to 1–3 new entrants.
    Prefer moderate, geographically tight clusters over the chained mega-clusters."""
    csz = _site.groupby("cluster").size()
    cand = _site[_site.site_key.isin(_pins)].copy()
    cand["csz"] = cand.cluster.map(csz)
    cand = cand[(cand.csz >= 3) & (cand.csz <= 12)].sort_values(["csz", "op_start"])
    fallback = None
    for k in cand.site_key.head(150):
        _ks = _site.loc[_site.site_key == k].iloc[0]
        nbf = neighbourhood(_site, _ks.lat, _ks.lon, 20)
        n_ent = int(nbf.is_entrant.sum())
        if n_ent == 0:
            continue
        ed = nbf[nbf.is_entrant].sort_values("op_start").op_start.iloc[-1]
        chs = [deseason_pct_change(_df, s, "ret_wash_count", ed) for s in nbf.loc[~nbf.is_entrant, "site_key"]]
        chs = [c for c in chs if np.isfinite(c)]
        if fallback is None and chs:
            fallback = k
        # clean "few incumbents + a new entrant" story with a believable (non-closure) impact
        if 1 <= n_ent <= 3 and len(chs) >= 2 and -40 <= float(np.median(chs)) <= 15:
            return k
    return fallback or (_pins[0] if _pins else _site.site_key.iloc[0])


def anon_names(site_df, keys):
    """site_key -> 'Site N' ordered by opening date (earliest = Site 1) — for the anonymized client demo."""
    sub = site_df[site_df.site_key.isin(list(keys))].sort_values("op_start")
    return {k: f"Site {i + 1}" for i, k in enumerate(sub.site_key)}


@st.cache_resource(show_spinner="Loading cold-start model…")
def get_model():
    return cm.load()


GRAN_OPTS = {"Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}
GRAN_RULE = {"M": "MS", "Q": "QS", "Y": "YS"}      # resample to period START so x stays a real date
GRAN_STEP = {"M": 1, "Q": 3, "Y": 12}              # months per bucket for months-since-open series
GRAN_UNIT = {"M": "month", "Q": "quarter", "Y": "year"}


def gran_picker(key, label="Window", default="Monthly"):
    """Per-plot Monthly/Quarterly/Yearly selector rendered right above its chart. Returns 'M' | 'Q' | 'Y'."""
    return GRAN_OPTS[st.radio(label, list(GRAN_OPTS), horizontal=True,
                              index=list(GRAN_OPTS).index(default), key=key)]


def rs_dates(obj, gran, how="sum"):
    """Resample a datetime-indexed Series/DataFrame to period-START buckets (Q/Y). Monthly = unchanged.
    sum uses min_count=1 so all-NaN periods stay NaN (a real data gap) instead of becoming a fake 0.
    Drops an incomplete TRAILING period (e.g. a partial current year) so the line doesn't crater at the end."""
    if gran == "M" or obj is None or len(obj) == 0:
        return obj
    r = obj.resample(GRAN_RULE[gran])
    out = r.sum(min_count=1) if how == "sum" else getattr(r, how)()
    cnt = r.size()
    if len(out) > 1 and int(cnt.iloc[-1]) < GRAN_STEP[gran]:
        out = out.iloc[:-1]
    return out


def gran_date_tickformat(gran):
    """d3 tick format for a real-date x-axis at the chosen granularity."""
    return {"M": "%b %Y", "Q": "%b %Y", "Y": "%Y"}[gran]
