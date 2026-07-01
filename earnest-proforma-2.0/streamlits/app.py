"""
Local Market Explorer — car-wash clusters, new-site entry & ramp-up.

Pick a pin on the map (or 🎲 Random). The app finds its neighbours within a radius
(the local market), draws each site's wash-count time series, and highlights the
NEW site(s) that entered the market in a distinct colour — so you can watch the new
site ramp up and see how the incumbents' series respond before/after the opening.

Run:   cd earnest-proforma-2.0/streamlits && streamlit run app.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import folium
from streamlit_folium import st_folium
import coldstart_model as cm
import site_visual_page as svp
try:                                                          # tunnel-length → CAPEX model (learned from proforma data)
    import tunnel_capex as tcx
    _CAPEX_OK = True
except Exception:                                             # pragma: no cover
    tcx = None
    _CAPEX_OK = False

# ── make the repo-root `app.*` package importable from streamlits/ (the local app.py would otherwise
#    shadow it). Mirrors site_analysis_page.py: register `app` as a namespace package at <repo>/app. ──
import importlib.machinery
import importlib.util
import os
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_APP_DIR = _REPO_ROOT / "app"
if _APP_DIR.is_dir() and ("app" not in sys.modules or not hasattr(sys.modules.get("app"), "__path__")):
    _app_mod = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("app", None, is_package=True))
    _app_mod.__path__ = [str(_APP_DIR)]
    sys.modules["app"] = _app_mod

try:                                                          # keep the dashboard alive if the package can't import
    from app.pnl_analysis.insights.graph import market_insights
    from app.pnl_analysis.insights.llm import insights_llm_ready
    _INSIGHTS_OK = True
except Exception as _insights_imp_err:                        # pragma: no cover
    market_insights = None
    def insights_llm_ready(*_a, **_k):
        return False
    _INSIGHTS_OK = False

# ── POC (kept deliberately separate from the grounded pipeline above): raw-GPT market analysis from the
#    pin's location alone, no data fed in. Its own module/prompt; reuses only the shared Azure transport. ──
try:
    from app.pnl_analysis.insights.location_poc import (location_market_analysis, pollinate_analysis,
                                                        competition_scale_analysis)
    _LOC_POC_OK = True
except Exception:                                             # pragma: no cover
    location_market_analysis = None
    pollinate_analysis = None
    competition_scale_analysis = None
    _LOC_POC_OK = False

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "main-ds.csv"
TYPES_CSV = HERE.parent / "data" / "site_carwash_types.csv"
ARTIFACTS = HERE.parent / "notebooks" / "artifacts"
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

# ───────────────────────────── data ─────────────────────────────
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


def _drop_corrupt_asp_rows(rec):
    """Drop site-months whose revenue feed collapsed to ~0 while wash_count stayed normal (a data-feed drop,
    not a real price). Row is bad if it has ≥ASP_MIN_WASH washes AND an implied $/wash below the floor.
    Self-contained per-row predicate (no per-site baseline). Returns (filtered_rec, n_dropped)."""
    if rec.empty:
        return rec, 0
    mw, rw = rec.mem_wash_count.replace(0, np.nan), rec.ret_wash_count.replace(0, np.nan)
    bad = (((rec.mem_wash_count >= ASP_MIN_WASH) & (rec.mem_revenue / mw < ASP_FLOOR_MEM))
           | ((rec.ret_wash_count >= ASP_MIN_WASH) & (rec.ret_revenue / rw < ASP_FLOOR_RET))).fillna(False)
    return rec[~bad], int(bad.sum())


@st.cache_data(show_spinner=False)
def global_healthy_asp(express_only=False):
    """Wash-weighted cluster-ASP fallback from ALL healthy site-months (after the corrupt-row floor),
    used when every in-radius neighbour is corrupt — far better than the flat $30/$15 defaults.
    Returns (cl_mem_pp, purch_per_wash, cl_ret)."""
    df, _ = load_data(express_only)
    rec, _ = _drop_corrupt_asp_rows(df)
    mm = rec.dropna(subset=["mem_revenue"]); rr = rec.dropna(subset=["ret_revenue"])
    mp, mw = mm.mem_purchase_count.sum(), mm.mem_wash_count.sum()
    cl_mem_pp = float(mm.mem_revenue.sum() / mp) if mp > 0 else 30.0
    ppw = float(mp / mw) if mw > 0 else 0.33
    cl_ret = float(rr.ret_revenue.sum() / rr.ret_wash_count.sum()) if rr.ret_wash_count.sum() > 0 else 15.0
    return cl_mem_pp, ppw, cl_ret


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


PNL = HERE.parent / "data" / "opex-data.csv"


PNL_EXCLUDE = {"alpinecarwash_000087"}   # sites kept OUT of the P&L analysis (matched on client_id)


@st.cache_data(show_spinner="Loading P&L…")
def load_pnl():
    """Per-location-year P&L from opex-data.csv: SUM the sub-monthly report rows into an annual
    operating expense / income per site, keep near-full years only. Returns one row per (location, state, year)."""
    p = pd.read_csv(PNL, low_memory=False)
    p = p[~p.client_id.astype(str).isin(PNL_EXCLUDE)]      # drop excluded sites before any aggregation
    g = (p.groupby(["location_name", "state", "year"])
         .agg(opex=("total_expenses", "sum"), income=("total_income", "sum"), cogs=("cogs", "sum"),
              months=("month", "nunique"), asp_mem=("ASP_mem", "median"), asp_ret=("ASP_ret", "median"),
              lat=("lat", "first"), lon=("lon", "first"))
         .reset_index())
    return g[(g.months >= 11) & (g.year.between(2022, 2025))].copy()


def regional_opex(pnl, art, state, region, min_sites=5):
    """Average annual operating expense per site, by YEAR, for the pin's STATE (if ≥min_sites of P&L data there)
    else its REGION else all sites. Returns (year-table, scope-label)."""
    s2r = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
           .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
    p = pnl.copy(); p["region"] = p.state.map(s2r)
    sub, scope = p[p.state == state], f"state {state}"
    if sub.location_name.nunique() < min_sites:
        rsub = p[p.region == region]
        sub, scope = (rsub, f"region {region}") if len(rsub) else (p, "all sites (no local P&L)")
    yo = (sub.groupby("year").agg(opex=("opex", "mean"), income=("income", "mean"), n=("location_name", "nunique"),
                                  asp_mem=("asp_mem", "median"), asp_ret=("asp_ret", "median")).reset_index())
    return yo, scope


@st.cache_data(show_spinner="Loading monthly P&L…")
def load_pnl_monthly():
    """MONTHLY P&L per location: opex = SUM of the sub-monthly report rows; washes/revenue = the monthly snapshot."""
    p = pd.read_csv(PNL, low_memory=False)
    p = p[~p.client_id.astype(str).isin(PNL_EXCLUDE)]
    m = (p.groupby(["location_name", "state", "year", "month"])
         .agg(opex=("total_expenses", "sum"), income=("total_income", "sum"),
              mem_wash=("mem_wash_count", "first"), ret_wash=("ret_wash_count", "first"),
              lat=("lat", "first"), lon=("lon", "first")).reset_index())
    m = m[m.year.between(2022, 2025)].copy()
    m["date"] = pd.to_datetime(dict(year=m.year, month=m.month, day=1))
    first = m.groupby("location_name").date.transform("min")          # age 0 = the site's FIRST P&L row (year,month), not created_date
    m["age"] = (m.date.dt.year - first.dt.year) * 12 + (m.date.dt.month - first.dt.month)
    m["wash"] = m.mem_wash.fillna(0) + m.ret_wash.fillna(0)
    return m


def opex_per_wash(pm, art, lat, lon, state, region, min_sites=5):
    """MATURE (age 18–30) monthly operating-expense **per wash** ($) — the settled cost level. Scoped to the pin's
    state (≥min_sites of P&L data), else its region (≥min_sites), else all sites. Returns ($/wash, scope label)."""
    s2r = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
           .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
    d = pm[(pm.wash > 0) & (pm.opex > 0) & (pm.age.between(18, 30))].copy(); d["region"] = d.state.map(s2r)
    sub, sc = d[d.state == state], f"state {state}"
    if sub.location_name.nunique() < min_sites:
        r = d[d.region == region]
        sub, sc = (r, f"region {region}") if r.location_name.nunique() >= min_sites else (d, "all sites")
    return float((sub.opex / sub.wash).median()), sc


def opex_ramp(pm, art, state=None, region=None, min_sites=8):
    """LEARNED new-site opex lifecycle from the P&L, REGION-SPECIFIC where supported.
    age 0 = the site's FIRST P&L row (year,month) — NOT created_date. For each site, opex(age) ÷ its OWN mature
    (mo 18–30) opex, then the median across sites by age — scoped to the pin's STATE (≥min_sites), else REGION,
    else ALL sites. New sites run HOT early (setup/marketing/ramp) then settle to ~1× by ~year 1.
    NOTE: the P&L only spans ~33 months, so the learned curve ends at `hage` (the last age with support). The caller
    EXTENDS months hage+1..60 with the forecast wash volume (opex ≈ $/wash × forecast washes), not a flat line.
    Returns (ramp[0..60] normalized so mature=1, scope-label, hage = last age with real P&L support)."""
    s2r = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
           .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
    d = pm.copy(); d["region"] = d.state.map(s2r)
    last = d.groupby("location_name").date.transform("max")            # drop each site's LAST month — partial-period
    d = d[d.date < last]                                               # export artifact (near-zero opex at the tail)
    d = d[(d.age >= 0) & (d.age <= 42) & (d.opex > 0)]
    mat = d[d.age.between(18, 30)].groupby("location_name").opex.mean(); mat = mat[mat > 0]
    d = d[d.location_name.isin(mat.index)].copy(); d["rel"] = d.opex / d.location_name.map(mat)

    def _curve(sub):
        sup = sub.groupby("age").location_name.nunique()
        med = sub.groupby("age").rel.median()[sup >= max(4, min_sites // 2)]   # keep only ages with support
        med = med[med.index <= 30]                                            # trust only ≤ mo30 (end of mature window);
        if med.empty:                                                         # mo31+ is thin/partial-export → forecast it
            return None
        arr = np.full(61, np.nan)
        for a in med.index:
            if 0 <= a <= 60 and np.isfinite(med[a]) and med[a] > 0:
                arr[a] = med[a]
        s = pd.Series(arr).interpolate(limit_area="inside").rolling(3, center=True, min_periods=1).mean()
        asym = float(np.nanmean(s.values[18:31])) or 1.0
        hage = int(med.index.max())                                    # last age with real P&L support
        s.iloc[hage + 1:] = asym                                       # placeholder beyond data (caller extends w/ forecast)
        s = s.fillna(asym)
        return np.clip((s / asym).to_numpy(), 0.5, 3.0), hage

    sub, scope = d[d.state == state], f"state {state}"
    if sub.location_name.nunique() < min_sites:
        r = d[d.region == region]
        sub, scope = (r, f"region {region}") if r.location_name.nunique() >= min_sites else (d, "all sites")
    res = _curve(sub)
    if res is None:
        res, scope = _curve(d), "all sites"
    ramp, hage = res
    return ramp, scope, hage


def opex_pct_curve(months, hot_end=9, mat_start=30, hot_hi=0.62, hot_lo=0.55, mat=0.45):
    """Opex as a % of revenue for the forecast P&L orange line.
    HOT launch: ~50–62% of revenue over the first ~hot_end months (declines gently 0.62→0.55,
    staying inside the 50–62% band), then a smooth ease down to a mature ~40–50% (settles at
    `mat`) by month `mat_start`, held flat thereafter."""
    m = np.asarray(months, float)
    pct = np.full(m.shape, float(mat))
    hot = m <= hot_end
    pct[hot] = hot_hi + (hot_lo - hot_hi) * (m[hot] / max(hot_end, 1))     # 50–62% band, first ~8–10 months
    mid = (m > hot_end) & (m < mat_start)
    f = (m[mid] - hot_end) / (mat_start - hot_end)
    f = 0.5 - 0.5 * np.cos(np.pi * f)                                      # smoothstep ease
    pct[mid] = hot_lo + (mat - hot_lo) * f                                 # glide down into the 40–50% band
    return pct


def opex_pct_fit(pm, art, state=None, region=None, min_sites=8, max_age=42):
    """EMPIRICAL operating-expense ratio (opex ÷ income, %) by months since inception — the data behind the
    modelled orange opex line, mirroring the reference 'total_expense_pct — normalized by month of inception'.
    For each scoped site we take its monthly opex%, then the MEDIAN and the 25–75 percentile band across sites
    by age (age 0 = the site's FIRST P&L row). Scoped to the pin's STATE (≥min_sites), else REGION, else ALL.
    Returns (ages, median, q25, q75, support_n, scope) or None if there's not enough history."""
    s2r = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
           .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
    d = pm.copy(); d["region"] = d.state.map(s2r)
    last = d.groupby("location_name").date.transform("max")            # drop each site's LAST month — partial-period export artifact
    d = d[d.date < last]
    d = d[(d.age >= 0) & (d.age <= max_age) & (d.opex > 0) & (d.income > 0)].copy()
    d["pct"] = 100.0 * d.opex / d.income
    d = d[d.pct.between(0, 500)]                                        # drop export garbage (near-zero income → absurd %)

    def _fit(sub):
        if sub.location_name.nunique() < min_sites:
            return None
        sup = sub.groupby("age").location_name.nunique()
        keep = sup[sup >= max(4, min_sites // 2)].index                # keep only ages with enough sites
        g = sub[sub.age.isin(keep)].groupby("age").pct
        med = g.median()
        if med.empty:
            return None
        return med, g.quantile(0.25), g.quantile(0.75), sup.reindex(med.index)

    res = _fit(d[d.state == state]); scope = f"state {state}"
    if res is None:
        res = _fit(d[d.region == region]); scope = f"region {region}"
    if res is None:
        res = _fit(d); scope = "all sites"
    if res is None:
        return None
    med, q25, q75, sup = res
    return med.index.to_numpy(), med.to_numpy(), q25.to_numpy(), q75.to_numpy(), sup.to_numpy(), scope


def opex_pct_curve_fit(pm, art, state=None, region=None, months=None, min_sites=8):
    """Opex-as-%-of-revenue curve FIT to the empirical P&L pattern (median opex ÷ income by month since
    inception): a HOT launch decaying to a mature level. Fits  mature + (hot − mature)·exp(−age/τ)  to the
    scoped per-age median (support-weighted, so thin/noisy ages don't drag it), then evaluates over `months`
    — naturally PROPAGATING forward to month 60, asymptoting to the mature level past the data.
    Falls back to the hand-set opex_pct_curve if there isn't enough P&L history. Returns a fraction array."""
    if months is None:
        months = np.arange(0, 61)
    months = np.asarray(months, float)
    fit = opex_pct_fit(pm, art, state, region, min_sites=min_sites, max_age=30)
    if fit is None:
        return opex_pct_curve(months)
    age, med, _q25, _q75, sup, _scope = fit
    age = age.astype(float)
    y = med.astype(float) / 100.0                                      # opex% as a fraction of revenue
    w = np.sqrt(np.clip(sup.astype(float), 1, None))                   # weight ages by how many sites support them

    mat0 = float(np.average(y[age >= 18], weights=w[age >= 18])) if (age >= 18).any() else float(np.median(y))
    hot0 = float(y[age <= 2].mean()) if (age <= 2).any() else float(y[0])
    lo, hi = (0.45, 0.25, 1.0), (1.6, 0.70, 36.0)
    p0 = [min(max(hot0, lo[0]), hi[0]), min(max(mat0, lo[1]), hi[1]), 6.0]   # clip guess into the bounds

    def _decay(a, hot, mat, tau):
        return mat + (hot - mat) * np.exp(-a / np.maximum(tau, 1e-6))

    try:
        from scipy.optimize import curve_fit
        popt, _ = curve_fit(_decay, age, y, p0=p0, sigma=1.0 / w, absolute_sigma=False,
                            maxfev=10000, bounds=(lo, hi))
    except Exception:
        popt = p0
    return np.clip(_decay(months, *popt), 0.30, 1.5)


# ── expense plan: fit user OPEX% (% of sales/yr) onto the learned pattern + spread CAPEX$ (mirrors app/pnl_analysis) ──
def year_slices(n_months, n_years=5):
    """Partition month indices 0..n_months-1 into `n_years` yearly buckets of 12, the LAST bucket absorbing any
    remainder (so a 61-month horizon folds month 60 into year 5, not a 1-month stub year 6)."""
    out = []
    for y in range(n_years):
        start = 12 * y
        if start >= n_months:
            break
        out.append(slice(start, n_months if y == n_years - 1 else min(12 * y + 12, n_months)))
    return out


def fit_opex_pct_to_targets(shape, revenue, opex_pct_by_year, opex_growth_pct=0.0):
    """Re-scale the learned monthly opex% `shape` so each YEAR's $-weighted opex% (= annual opex$ ÷ annual sales$,
    sales = ASP × washes) equals the user's per-year target %, while keeping the within-year hot→mature shape.
    Years past the last supplied inherit the last target escalated by opex_growth_pct/yr.
    Returns (opex_pct[fraction, len=months], {year: target_fraction})."""
    shape, revenue = np.asarray(shape, float), np.asarray(revenue, float)
    targets = {int(y): v / 100.0 for y, v in (opex_pct_by_year or {}).items() if v is not None}
    last_year = max(targets) if targets else None
    out = shape.copy()
    year_targets = {}
    for y, sl in enumerate(year_slices(len(shape))):
        seg, seg_rev = shape[sl], revenue[sl]
        wts = seg_rev if float(np.nansum(seg_rev)) > 0 else np.ones_like(seg)
        wmean = float(np.average(seg, weights=wts)) or 1.0
        yr = y + 1
        if yr in targets:
            tgt = targets[yr]
        elif last_year is not None:
            tgt = targets[last_year] * (1 + opex_growth_pct / 100.0) ** (yr - last_year)
        else:
            tgt = wmean
        year_targets[yr] = tgt
        out[sl] = seg * (tgt / wmean)
    return np.clip(out, 0.0, 3.0), year_targets


def spread_capex(n_months, capex_by_year):
    """CAPEX $ per year spread evenly across that year's months → a monthly $ array."""
    out = np.zeros(n_months)
    for y, sl in enumerate(year_slices(n_months)):
        amt = (capex_by_year or {}).get(y + 1)
        if amt:
            out[sl] = amt / max(sl.stop - sl.start, 1)
    return out


@st.cache_data(show_spinner=False)
def capex_band_table():
    """Tunnel-length → CAPEX band table (median/mean/n per band), learned from the proforma data."""
    return tcx.capex_band_table() if _CAPEX_OK else pd.DataFrame()


@st.cache_data(show_spinner=False)
def capex_builds():
    """Per-build (tunnel length, total CAPEX) points for the scatter."""
    return tcx.load_builds() if _CAPEX_OK else pd.DataFrame(columns=["tlen", "capex"])


@st.cache_data(show_spinner=False)
def capex_fit():
    """(slope $/m, intercept $, corr, n) of CAPEX ~ tunnel length, for the analysis caption."""
    return tcx.fit() if _CAPEX_OK else (None, None, None, 0)


def per_year_to_monthly(n_months, by_year, default):
    """Expand a {year: value} map to a length-n_months array — each year's months get that year's value
    (a step series), missing years fall back to `default`. Used for the year-wise ASP ($/wash)."""
    out = np.full(n_months, float(default))
    for y, sl in enumerate(year_slices(n_months)):
        v = (by_year or {}).get(y + 1)
        if v is not None:
            out[sl] = float(v)
    return out


# ── per-plot time-granularity (Monthly / Quarterly / Yearly window, summed into period totals) ──
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


def agg_months(y, gran, how="sum", start=0):
    """Aggregate a months-since-open array (positions = months 0,1,2,…) into Q/Y buckets.
    Keeps x in MONTH units (each bucket plotted at its start month) so vlines/vrects in months still align.
    Returns (x_months, y_agg)."""
    y = np.asarray(y, dtype=float)
    if gran == "M":
        return np.arange(start, start + len(y)), y
    step = GRAN_STEP[gran]
    xs, ys = [], []
    for i in range(0, len(y), step):
        chunk = y[i:i + step]
        if len(chunk) < step:          # drop the incomplete trailing bucket (e.g. lone month 60) — no fake end-crash
            break
        xs.append(start + i)
        ys.append(np.nansum(chunk) if how == "sum" else np.nanmean(chunk))
    return np.array(xs), np.array(ys)


def gran_xaxes_months(fig, gran, xb, noun="open"):
    """Relabel a months-since-open x-axis to quarter/year buckets (data x stays in months so vlines still align):
    Monthly → 'months since …'; Quarterly → Q1,Q2,…; Yearly → Yr1,Yr2,…"""
    if gran == "M":
        fig.update_xaxes(title_text=f"months since {noun}")
        return
    step, u = GRAN_STEP[gran], {"Q": "Q", "Y": "Yr"}[gran]
    fig.update_xaxes(title_text=f"{GRAN_UNIT[gran]}s since {noun}",
                     tickvals=list(xb), ticktext=[f"{u}{int(i) // step + 1}" for i in xb])


def gran_date_tickformat(gran):
    """d3 tick format for a real-date x-axis at the chosen granularity."""
    return {"M": "%b %Y", "Q": "%b %Y", "Y": "%Y"}[gran]


def opex_trend_hist(pnl, art, state, region, min_sites=5):
    """Median per-site YoY opex growth for the pin's scope — the historical opex 'pattern' (context for the
    cost-growth slider). NOTE: on this data it's strongly negative & noisy (likely a reporting artifact), so it's
    shown, not used as the default. Returns a fraction/yr."""
    s2r = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
           .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
    a = pnl.copy(); a["region"] = a.state.map(s2r)
    a = a.sort_values(["location_name", "year"]); a["prev"] = a.groupby("location_name").opex.shift(1)
    a["yoy"] = a.opex / a["prev"] - 1
    sub = a[a.state == state]
    if sub.location_name.nunique() < min_sites:
        r = a[a.region == region]; sub = r if len(r) else a
    yoy = sub.yoy.replace([np.inf, -np.inf], np.nan).dropna()
    return float(yoy.median()) if len(yoy) else 0.0


def asp_refs(pnl, art, lat, lon, state, region):
    """Two reference ASPs to mark on the sliders: OVERALL (all P&L sites) and CLUSTER/LOCAL (P&L sites ≤25 km of
    the pin, falling back to state → region → overall when too few). Returns mem/ret for each + a scope label."""
    ov_mem, ov_ret = float(pnl.asp_mem.median()), float(pnl.asp_ret.median())
    loc = (pnl.groupby("location_name")
           .agg(lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                asp_mem=("asp_mem", "median"), asp_ret=("asp_ret", "median")).reset_index().dropna(subset=["lat", "lon"]))
    loc["d"] = haversine_km(lat, lon, loc.lat.values, loc.lon.values)
    near = loc[loc.d <= 25]
    if len(near) >= 2:
        sub, sc = near, f"cluster ≤25 km · {len(near)} sites"
    elif (loc.state == state).sum() >= 3:
        sub, sc = loc[loc.state == state], f"state {state}"
    else:
        s2r = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
               .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
        loc["region"] = loc.state.map(s2r); rg = loc[loc.region == region]
        sub, sc = (rg, f"region {region}") if len(rg) else (loc, "all sites")
    return dict(ov_mem=ov_mem, ov_ret=ov_ret, cl_mem=float(sub.asp_mem.median()),
                cl_ret=float(sub.asp_ret.median()), scope=sc)


TAU_Y = 2.0   # post-maturity growth SATURATES over ~2 yr (sites plateau ~24 mo empirically); see _sat_years


def _sat_years(years):
    """Saturating 'effective years' of drift: ≈ linear near 0, asymptotes to TAU_Y. So a trend applies at full
    strength early then DECELERATES — a booming market ramps then plateaus instead of compounding forever. This is
    the principled replacement for the old ±% rate clamp (growth can't be extrapolated unbounded — booms saturate)."""
    return TAU_Y * (1.0 - np.exp(-np.asarray(years, dtype=float) / TAU_Y))


def _robust_slope(arr):
    """Per-month log-growth slope (Theil-Sen on 6-mo-smoothed log level, last ~30 mo) + its SE. SE inflated by √6
    for the smoothing autocorrelation (effective N ≈ K/6). Returns (slope, se) or None if the series is too short."""
    arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
    if len(arr) < 18:
        return None
    sm = pd.Series(arr).rolling(6, min_periods=3).mean().dropna().to_numpy()
    K = min(30, len(sm))
    if K < 8:
        return None
    yv = np.log(np.clip(sm[-K:], 1.0, None)); xv = np.arange(K, dtype=float)
    try:
        from scipy.stats import theilslopes
        sl, _, lo, hi = theilslopes(yv, xv)                # robust median slope + 95% CI
    except Exception:
        sl = float(np.polyfit(xv, yv, 1)[0]); lo = hi = sl
    se = max((hi - lo) / (2 * 1.96), 1e-9) * np.sqrt(6.0)
    return float(sl), float(se)


def _shrink_annualize(sl, se):
    """Shrink a per-month slope toward 0 by its signal-to-noise t²/(1+t²), annualize → (g, g_lo, g_hi). When the
    trend isn't distinguishable from flat (wide CI) the central collapses to ~0; the band is the honest slope CI.
    Loose ±40%/yr SANITY rail only stops a degenerate series exploding — it is NOT the old [-5%,+8%] clamp."""
    SANE = lambda r: float(np.clip(r, -0.40, 0.40))
    t = abs(sl) / se
    sl_c = sl * (t * t / (1.0 + t * t))
    return SANE(np.exp(sl_c * 12) - 1), SANE(np.exp((sl - 1.96 * se) * 12) - 1), SANE(np.exp((sl + 1.96 * se) * 12) - 1)


def robust_growth(arr):
    """Annual growth (central + data-based CI band) for ONE series — no hand-set ±clamp."""
    g = _robust_slope(arr)
    return _shrink_annualize(*g) if g else (0.0, 0.0, 0.0)


def market_trend(piv):
    """COMPOSITION-ROBUST market trend from a date×site pivot. Each site's OWN robust slope (immune to sites
    entering/leaving an average — that lurch faked +trends in 1–2-site markets), pooled by MEDIAN; pooled SE from
    the between-site spread and within-site error. Returns (g, g_lo, g_hi). Data-driven, no growth clamp."""
    slopes, ses = [], []
    for col in getattr(piv, "columns", []):
        r = _robust_slope(piv[col].dropna().to_numpy())
        if r:
            slopes.append(r[0]); ses.append(r[1])
    if not slopes:
        return 0.0, 0.0, 0.0
    slopes = np.asarray(slopes); ses = np.asarray(ses); n = len(slopes)
    sl = float(np.median(slopes))
    between = float(np.std(slopes, ddof=1)) if n >= 2 else 0.0
    se = max(np.hypot(between, float(np.median(ses))) / np.sqrt(n), 1e-9)
    return _shrink_annualize(sl, se)


def forecast_series(s, H, g=None):
    """SMOOTH 5-yr expected-trend forecast (no repeating seasonal photocopy). Starts exactly at the last actual
    value and blends over ~a quarter into a trend line at the recent deseasonalized LEVEL, growing at a ROBUST
    annual rate (`robust_growth` — Theil-Sen, follows the true long-run direction, not a spike-fooled ratio).
    Predicting 5 yrs of month-by-month seasonal wiggle isn't meaningful; the history keeps its real seasonality."""
    s = pd.Series(s).astype(float).dropna()
    n = len(s)
    if n == 0:
        return np.zeros(H)
    arr = s.to_numpy()
    last = float(arr[-1])
    level = float(arr[-min(12, n):].mean())                    # deseasonalized recent level
    if g is None:
        g = robust_growth(arr)[0]
    t = np.arange(1, H + 1)
    trend = level * (1 + g) ** _sat_years(t / 12.0)            # smooth trend at the robust rate, SATURATING (boom decelerates)
    w = np.exp(-(t - 1) / 3.0)                                 # blend: start at last actual, converge to trend (~quarter)
    return np.clip(last * w + trend * (1 - w), 0, None)


# ── campaign signal — what the operators' P&L actually shows (event study on opex-data.csv) ──
# A "campaign" = a promotional OPEX spike. HONEST read: the apparent 12-month "lift" in the raw event
# study is mostly the site's own organic ramp (≈half of campaign sites are <1yr old & ramping). Once each
# site's trend is removed (detrend + difference-in-differences), the clean incremental effect is a SHORT
# (~1–6 month) retail→MEMBERSHIP CONVERSION — biggest where there's retail headroom (low membership share),
# best ROI in dense markets. The promo OPEX is front-loaded (hot launch month, short tail).
CAMP_OPEX_TAIL = [1.33, 1.17, 1.10, 1.05, 1.03, 1.02]    # opex multiplier during the campaign window (the spend)


def campaign_conv_pct(mem_share):
    """Membership-wash lift a campaign delivers, scaled by the site's membership share (= retail headroom).
    From the timing analysis: low-share sites convert the most retail customers into members."""
    if mem_share < 0.65:
        return 0.30          # lots of retail to convert (data: +36% lift; tempered for the new-site case)
    if mem_share < 0.78:
        return 0.14
    return 0.07              # already mostly members → little headroom


def campaign_effect(launch, mem_share, intensity=1.0, window=6, horizon=61):
    """Per-month multipliers (membership washes, retail washes, opex) for a SHORT campaign. The effect is a
    retail→membership conversion that ramps in over ~2 months, holds through `window` months, then fades
    (members partly stick, ~12-mo half-life). Membership washes lift by the share-scaled amount; retail
    gives up ~half as many washes (converts wash more often). OPEX carries the promo spend over the window."""
    lift = campaign_conv_pct(mem_share) * intensity
    mem, ret, opx = np.ones(horizon), np.ones(horizon), np.ones(horizon)
    for t in range(horizon):
        k = t - launch
        if k < 0:
            continue
        if k < window:
            ramp = min(1.0, (k + 1) / 2.0)                          # conversion ramps in over ~2 months
            mem[t] = 1 + lift * ramp
            ret[t] = 1 - 0.5 * lift * ramp                         # members wash more → retail falls ~half as much
            opx[t] = 1 + (CAMP_OPEX_TAIL[k] - 1) * intensity if k < len(CAMP_OPEX_TAIL) else 1.0
        else:
            f = 0.5 ** ((k - window) / 12.0)                       # membership base partly sticks, slowly fades
            mem[t] = 1 + lift * f
            ret[t] = 1 - 0.5 * lift * f
    return mem, ret, opx


@st.cache_data(show_spinner="Detecting campaigns…")
def campaign_months_by_site():
    """site_key -> list of campaign month timestamps (real promo OPEX spikes) from opex-data.csv.
    Spike = true_opex (cogs+expenses) > median+3·MAD AND > 1.3× trailing-6mo median; interior months only."""
    p = pd.read_csv(PNL, low_memory=False)
    p["site_key"] = p.client_id.astype(str) + "::" + p.site_id.astype(str)
    p["date"] = pd.to_datetime(p["report_date"]).dt.to_period("M").dt.to_timestamp()   # month start (matches main-ds)
    p["true_opex"] = p.cogs.fillna(0) + p.expenses.fillna(0)
    # one row per (site, month): keep the real financial row, dropping all-zero artifact duplicates
    p = p.sort_values("total_income").drop_duplicates(["site_key", "date"], keep="last")
    out = {}
    for sk, g in p.sort_values("date").groupby("site_key"):
        s = g.set_index("date")["true_opex"]
        if len(s) < 12:
            continue
        med = s.median(); mad = 1.4826 * (s - med).abs().median()
        troll = s.shift(1).rolling(6, min_periods=4).median()
        spike = (s > med + 3 * mad) & (s > 1.3 * troll)
        cutoff = s.index[-3]                                        # need a post-window → drop last 3 months
        dates = [d for d in s.index[spike.fillna(False)] if d <= cutoff]
        if dates:
            out[sk] = dates
    return out


@st.cache_data(show_spinner=False)
def _campaign_data():
    """`data` for the book_v4 campaign plots — opex-data.csv with site_key = client_id::site_id."""
    d = pd.read_csv(PNL, low_memory=False)
    d["site_key"] = d.client_id.astype(str) + "::" + d.site_id.astype(str)
    return d


@st.cache_data(show_spinner=False)
def _campaigns_df():
    """`campaigns_df` for the book_v4 plots — detect OPEX spikes (true_opex > 1.2× trailing-6mo mean) and
    cluster consecutive spike months (gap ≤ 1) into campaigns (site_key / campaign_start / duration_months)."""
    data = _campaign_data()
    sub = data.sort_values(["site_key", "report_date"]).copy()
    sub["report_date"] = pd.to_datetime(sub["report_date"])
    sub["true_opex"] = sub["cogs"] + sub["expenses"]
    sub["opex_baseline"] = (sub.groupby("site_key")["true_opex"]
                            .transform(lambda s: s.shift(1).rolling(6, min_periods=4).mean()))
    sub["opex_vs_baseline"] = sub["true_opex"] / sub["opex_baseline"]
    spikes = sub[sub["opex_vs_baseline"] > 1.2].copy()
    records = []
    for site_key, grp in spikes.sort_values("report_date").groupby("site_key"):
        rows = grp.reset_index(drop=True); i = 0
        while i < len(rows):
            start_date = rows.loc[i, "report_date"]; months = [rows.loc[i, "report_date"]]; j = i + 1
            while j < len(rows):
                gap = ((rows.loc[j, "report_date"].year - rows.loc[j-1, "report_date"].year) * 12 +
                       (rows.loc[j, "report_date"].month - rows.loc[j-1, "report_date"].month))
                if gap <= 1:
                    months.append(rows.loc[j, "report_date"]); j += 1
                else:
                    break
            records.append({"site_key": site_key, "campaign_start": start_date, "duration_months": len(months)}); i = j
    return pd.DataFrame(records)


def render_campaign_snapshot():
    """OPEX / Revenue / Profit / Mem Purchases — 3 separate plots, exactly as in book_v4.ipynb."""
    data = _campaign_data()
    campaigns_df = _campaigns_df().copy()

    # ── OPEX / Revenue / Profit / Mem Purchases — 3 separate plots ───────────────  [book_v4.ipynb]
    _rd = data.copy()
    _rd["report_date"] = pd.to_datetime(_rd["report_date"])
    _rd["true_opex"]   = _rd["cogs"] + _rd["expenses"]
    _rd_by_site = {sk: grp.sort_values("report_date")
                   for sk, grp in _rd.groupby("site_key")}

    def reclassify(d):
        if d == 1: return "1 month"
        if d == 2: return "2 months"
        return "3+ months"

    campaigns_df["dur_bucket2"] = campaigns_df["duration_months"].apply(reclassify)

    _records = []
    for _, camp in campaigns_df.iterrows():
        sk     = camp["site_key"]
        anchor = pd.to_datetime(camp["campaign_start"])
        bucket = camp["dur_bucket2"]
        dur    = int(camp["duration_months"])
        if sk not in _rd_by_site:
            continue
        ts = _rd_by_site[sk].copy()
        ts["mfs"] = (
            (ts["report_date"].dt.year  - anchor.year)  * 12 +
            (ts["report_date"].dt.month - anchor.month)
        )
        for _, row in ts[(ts["mfs"] >= -3) & (ts["mfs"] <= 6)].iterrows():
            _records.append({
                "bucket":           bucket,
                "duration":         dur,
                "mfs":              int(row["mfs"]),
                "opex":             row["true_opex"],
                "revenue":          row["total_income"],
                "profit":           row["total_income"] - row["true_opex"],
                "mem_purchases":    row["mem_purchase_count"],
            })

    snap_df2 = pd.DataFrame(_records)

    OPEX_COLOR    = "#4FC3F7"
    REV_COLOR     = "#FFA726"
    PROFIT_COLOR  = "#66BB6A"
    MEM_COLOR     = "#CE93D8"   # purple — membership purchases
    CAMP_SHADE    = "#FFF9C4"

    BUCKET_CONFIG = {
        "1 month":   {"camp_months": [0],       "title": "1-Month Campaigns"},
        "2 months":  {"camp_months": [0, 1],    "title": "2-Month Campaigns"},
        "3+ months": {"camp_months": [0, 1, 2], "title": "3+ Month Campaigns"},
    }

    for bucket, cfg in BUCKET_CONFIG.items():
        sub = snap_df2[snap_df2["bucket"] == bucket]
        if len(sub) == 0:
            continue

        agg = (
            sub.groupby("mfs")[["opex", "revenue", "profit", "mem_purchases"]]
            .median()
            .reset_index()
        )
        n_camps     = campaigns_df[campaigns_df["dur_bucket2"] == bucket].shape[0]
        camp_months = cfg["camp_months"]
        month_list  = list(agg["mfs"])

        x_labels = []
        for m in month_list:
            if m in camp_months:
                x_labels.append(f"T={m}<br><b>📍 Campaign</b>")
            else:
                x_labels.append(f"T={m}")

        fig = go.Figure()

        # Shade campaign months
        for cm in camp_months:
            if cm in month_list:
                pos = month_list.index(cm)
                fig.add_vrect(
                    x0=pos - 0.5, x1=pos + 0.5,
                    fillcolor=CAMP_SHADE, opacity=0.55,
                    layer="below", line_width=0,
                )

        fig.add_trace(go.Bar(
            name="OPEX", x=x_labels, y=agg["opex"],
            marker_color=OPEX_COLOR,
            hovertemplate="T=%{x}<br>OPEX: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Revenue", x=x_labels, y=agg["revenue"],
            marker_color=REV_COLOR,
            hovertemplate="T=%{x}<br>Revenue: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Profit (Rev − OPEX)", x=x_labels, y=agg["profit"],
            marker_color=PROFIT_COLOR,
            hovertemplate="T=%{x}<br>Profit: $%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Membership Purchases", x=x_labels, y=agg["mem_purchases"],
            marker_color=MEM_COLOR,
            hovertemplate="T=%{x}<br>Mem Purchases: %{y:,.0f}<extra></extra>",
        ))

        camp_label = (
            "T=0 = campaign month"
            if len(camp_months) == 1
            else f"T={camp_months[0]}–T={camp_months[-1]} = campaign months (yellow)"
        )

        fig.update_layout(
            barmode="group",
            title=dict(
                text=(
                    f"<b>OPEX · Revenue · Profit · Membership Purchases — {cfg['title']}</b>"
                    f"<br><sub>n={n_camps} campaigns · {camp_label} · "
                    "median values · T=-3 to T=-1 = pre-campaign baseline</sub>"
                ),
                x=0.02, xanchor="left",
            ),
            xaxis=dict(
                title="Month Offset from Campaign Start",
                showgrid=False, showline=True, linecolor="#CCC",
            ),
            yaxis=dict(
                title="Median Value",
                tickformat=",.0f",
                gridcolor="#EEE",
            ),
            legend=dict(orientation="h", x=0.02, y=1.01, xanchor="left"),
            plot_bgcolor="white",
            height=470,
            margin=dict(l=90, r=40, t=120, b=60),
            bargap=0.18,
            bargroupgap=0.04,
        )
        st.plotly_chart(fig, width="stretch", theme=None, key=f"camp_snap_{bucket}")


def campaign_cluster_panel(df, site, lat, lon, radius, demo=False):
    """Real per-site line charts for the local-market cluster, with detected campaign months marked — so
    you can SEE the retail→membership conversion in actual nearby sites (the evidence behind the model)."""
    g = site[site.has_coords].copy()
    g["d"] = haversine_km(lat, lon, g.lat.values, g.lon.values)
    keys = g[g.d <= radius].nsmallest(8, "d").site_key.tolist()       # the local-market sites (cap 8 for legibility)
    if not keys:
        return
    st.divider()
    st.subheader("Real campaigns in this local market — the evidence")
    camp = campaign_months_by_site()
    metric_opt = st.radio("Series", ["Membership share of washes", "Membership washes", "Retail washes"],
                          horizontal=True, key="camp_cluster_metric")
    col = {"Membership share of washes": "mem_share_wash", "Membership washes": "mem_wash_count",
           "Retail washes": "ret_wash_count"}[metric_opt]
    gcl = gran_picker("gran_cluster")
    how = "mean" if col == "mem_share_wash" else "sum"            # share is a rate → average; washes → sum
    label = (anon_names(site, set(keys)) if demo
             else {k: str(site.loc[site.site_key == k, "client_name"].iloc[0])[:22] for k in keys})
    PAL = ["#2E86DE", "#16a085", "#8e44ad", "#e67e22", "#27ae60", "#c0392b", "#2c3e50", "#f39c12"]
    fig = go.Figure()
    for i, k in enumerate(keys):
        s = df[df.site_key == k].set_index("date")[col].sort_index()
        if s.dropna().empty:
            continue
        s = rs_dates(s.dropna(), gcl, how)
        c = PAL[i % len(PAL)]
        fmt = "%{y:.0%}" if col == "mem_share_wash" else "%{y:,.0f}"
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=label.get(k, "site"),
                                 line=dict(color=c, width=2),
                                 hovertemplate=f"<b>{label.get(k,'site')}</b><br>%{{x|%b %Y}} · {fmt}<extra></extra>"))
        for dt in camp.get(k, []):                                    # mark this site's real promo OPEX spikes
            fig.add_vline(x=dt, line=dict(color=c, width=1.5, dash="dot"), opacity=0.6)
    if col == "mem_share_wash":
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=400, template="plotly_white", hovermode="closest",
                      xaxis_title="date", yaxis_title=(metric_opt if how == "mean" else f"{metric_opt} / {GRAN_UNIT[gcl]}"),
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.22, font=dict(size=10)))
    fig.update_xaxes(tickformat=gran_date_tickformat(gcl))
    st.plotly_chart(fig, width="stretch", key="camp_cluster_chart")


def _pnl_figure(rev_m, opex_m, capex_m, totexp_m, net_m, gran, *, title,
                camp_on=False, rev_base=None, c_launch=None, win=6):
    """Build the 5-year P&L figure (revenue · stacked OPEX+CAPEX · total expenses · net income). Reused so the
    baseline and the campaign scenario render as two IDENTICAL plots. `camp_on` overlays the no-campaign revenue
    line and shades the promo window."""
    xb, rev_a = agg_months(rev_m, gran)
    _, opex_a = agg_months(opex_m, gran)
    _, capex_a = agg_months(capex_m, gran)
    _, totexp_a = agg_months(totexp_m, gran)
    _, net_a = agg_months(net_m, gran)
    f = go.Figure()
    f.add_trace(go.Scatter(x=xb, y=opex_a, name="OPEX", mode="lines", line=dict(color="#e67e22", width=0.5),
                           stackgroup="exp", fillcolor="rgba(230,126,34,0.55)",
                           hovertemplate="m%{x} · $%{y:,.0f}<extra>OPEX</extra>"))
    f.add_trace(go.Scatter(x=xb, y=capex_a, name="CAPEX", mode="lines", line=dict(color="#8e44ad", width=0.5),
                           stackgroup="exp", fillcolor="rgba(142,68,173,0.45)",
                           hovertemplate="m%{x} · $%{y:,.0f}<extra>CAPEX</extra>"))
    f.add_trace(go.Scatter(x=xb, y=totexp_a, name="total expenses", mode="lines", line=dict(color="#c0392b", width=2),
                           hovertemplate="m%{x} · $%{y:,.0f}<extra>total expenses</extra>"))
    f.add_trace(go.Scatter(x=xb, y=rev_a, name="revenue (sales)", mode="lines", line=dict(color="#16a085", width=3),
                           hovertemplate="m%{x} · $%{y:,.0f}<extra>revenue</extra>"))
    f.add_trace(go.Scatter(x=xb, y=net_a, name="net income", mode="lines", line=dict(color="#0a84ff", width=2.6),
                           hovertemplate="m%{x} · $%{y:,.0f}<extra>net</extra>"))
    f.add_hline(y=0, line=dict(color="#9aa6b2", width=1, dash="dot"))
    if camp_on and rev_base is not None:
        _, revb_a = agg_months(rev_base, gran)
        f.add_trace(go.Scatter(x=xb, y=revb_a, name="revenue (no campaign)", mode="lines",
                               line=dict(color="#9aa6b2", width=1.4, dash="dot"),
                               hovertemplate="m%{x} · $%{y:,.0f}<extra>no campaign</extra>"))
        f.add_vrect(x0=c_launch, x1=min(60, c_launch + win), fillcolor="rgba(230,25,75,0.08)", line_width=0,
                    annotation_text="campaign", annotation_position="top left")
        f.add_vline(x=c_launch, line=dict(color="#c0392b", dash="dash", width=1.2))
    f.update_layout(title=title, height=440, template="plotly_white", hovermode="x unified",
                    yaxis_title=f"$ / {GRAN_UNIT[gran]}", margin=dict(l=10, r=10, t=44, b=10),
                    legend=dict(orientation="h", y=-0.24))
    gran_xaxes_months(f, gran, xb, noun="opening")
    return f


def drop_pin_ui(df, site, art, demo=False, express_only=False):
    st.title("📍 Pinpoint forecast")
    st.caption("Drop a pin inside a market for a grounded **5-year forecast of a new car-wash site** — "
               "wash volume, P&L, the build CAPEX for your tunnel size, and whether to run a launch campaign.")
    if express_only:
        st.caption("🚿 **Express-only mode** — local-market trends, the cluster gate and the level anchor all use "
                   "Express Tunnel sites only.")
    brand = None; op_label = None                         # set by the operator selector (Model 4) in the sidebar below
    # the point is chosen in Explore-markets (or by clicking the map below). Radius is FIXED at the cluster
    # training radius — clusters are trained at ≤20 km, so it can't be widened here — and no smoothing.
    radius = 20
    smooth = 1
    if "pin" not in st.session_state:
        r = art["sites_rl"].sample(1, random_state=1).iloc[0]
        st.session_state.pin = (float(r.lat), float(r.lon))
    with st.sidebar:
        ov = st.number_input("Plateau override — total washes/mo (0 = use model)", min_value=0, value=0, step=500)
        # Separate, self-contained level strategies (each leave-one-out backtested; WAPE shown). The ramp-up →
        # mature → plateau trajectory is the SAME underneath — the models differ only in how the plateau level is set.
        # leakage-free LOO backtest (1223 sites, ≥24mo): Model 2 WAPE 43.6 · Model 3 40.2 (cold-start) ·
        # Model 4 ~34.1 WAPE / 27.7 medAPE when the operator is KNOWN (uses that operator's avg mature level, brand_loo).
        # Model 1 (no local anchor) was DROPPED — it under-predicts wash count badly (bias 0.72; 29% of sites fall
        # below their cluster-neighbour minimum). All kept models anchor on the local matured level. Labels keep
        # their original numbers so the WAPE references above stay meaningful.
        MODEL_STRATEGIES = {
            "Model 2": dict(local_anchor=True, model_kind="lgb", use_operator=False),
            "Model 3": dict(local_anchor=True, model_kind="et", use_operator=False),
            "Model 4": dict(local_anchor=True, model_kind="et", use_operator=True),
        }
        strat = MODEL_STRATEGIES[st.radio("Model", list(MODEL_STRATEGIES), index=1,    # default Model 3 (best cold-start)
                                          help="Plateau-level strategy (LOO WAPE): M2 43.6%, M3 40.2% "
                                               "(cold-start, no operator). M4 ≈ 34% — operator-based; pick the operator below.")]
        anchor_on = strat["local_anchor"]
        model_kind = strat["model_kind"]
        # Model 4: operator-based — pick a known operator; the model uses that operator's avg mature level (brand_loo)
        if strat["use_operator"] and not demo:
            _rl = art["sites_rl"][["client_id", "client_name"]].drop_duplicates("client_id")
            _bn = art["brand_n"]; _known = set(art["brand_mean"].keys())
            _rl = _rl[_rl.client_id.isin(_known) & _rl.client_name.notna()].copy()
            _rl["lab"] = _rl.apply(lambda r: f"{r.client_name} · {int(_bn.get(r.client_id, 0))} sites", axis=1)
            _id_by = dict(zip(_rl.lab, _rl.client_id))
            op_label = st.selectbox("Operator (brand)", ["(none — new operator)"] + sorted(_id_by),
                                    help="Use a known operator's track record (their avg mature level across all their "
                                         "sites). Leave as 'new operator' for a brand with no history (cold-start).")
            if op_label and op_label != "(none — new operator)":
                brand = _id_by.get(op_label)
        elif strat["use_operator"] and demo:
            st.caption("Operator selection is hidden in client-demo mode.")
        gm = st.slider("Yr 3–5 membership — extra on top of per-site trend (%/yr)", -15, 25, 0)
        gr = st.slider("Yr 3–5 retail — extra on top of per-site trend (%/yr)", -20, 15, 0)
    lat, lon = st.session_state.pin
    # Learn this LOCAL MARKET's trends PER COMPONENT (membership vs retail behave very differently, and differ by
    # cluster) from the neighbours' own series — the new site tracks them after maturity and the market forecast uses
    # them. Data-driven (robust Theil-Sen slope), not a single blended filter.
    _nb = site[site.has_coords]
    _d = haversine_km(lat, lon, _nb.lat.values, _nb.lon.values)
    _keys = _nb.site_key[(_d <= radius) & (_d > 1e-6)].tolist()
    if _keys:
        _sub = df[df.site_key.isin(_keys)]
        _pm = _sub.pivot_table(index="date", columns="site_key", values="mem_wash_count")
        _pr = _sub.pivot_table(index="date", columns="site_key", values="ret_wash_count")
        mem_g, mem_lo, mem_hi = market_trend(_pm)               # per-site median membership trend + CI band (composition-robust)
        ret_g, ret_lo, ret_hi = market_trend(_pr)               # per-site median retail trend + CI band
    else:
        mem_g = mem_lo = mem_hi = ret_g = ret_lo = ret_hi = 0.0
    # a dropped pin must sit inside a real local market (an existing cluster within the radius). Otherwise the
    # "forecast" would just be a region/national prior with no local grounding (region is inherited from the
    # nearest site, even if far) — so we refuse to predict and let the user move the pin into a cluster.
    _cl = site[site.has_coords & (site.cluster >= 0)]
    _cld = haversine_km(lat, lon, _cl.lat.values, _cl.lon.values)
    if not bool((_cld <= radius).any()):
        st.warning(f"This point is **outside any cluster** — no clustered sites within {radius} km, so there's no "
                   f"local market to ground a forecast. Drop the pin inside/near a cluster (blue dots) or widen the radius.")
        st.subheader("📍 Pick a location inside a cluster — blue dots are clustered sites")
        fmap = folium.Map(location=[lat, lon], zoom_start=10, tiles="cartodbpositron", control_scale=True)
        if not demo:
            _near = _cl.assign(d=_cld)
            for _, s in _near[_near.d <= 80].iterrows():
                folium.CircleMarker([s.lat, s.lon], radius=3, color="#3b7dd8", fill=True, fill_color="#3b7dd8",
                                    fill_opacity=0.7, weight=0,
                                    tooltip=f"{s.client_name} · market #{int(s.cluster)} · {s.d:.1f} km").add_to(fmap)
        folium.Circle([lat, lon], radius=radius * 1000, color="#c0392b", weight=2, fill=False).add_to(fmap)
        folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="star"), tooltip="📍 your point").add_to(fmap)
        m = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
        lc = (m or {}).get("last_clicked")
        if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(lat, 5), round(lon, 5)):
            st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
        return
    # express mode: anchor the plateau level on express neighbours only (cluster ∩ matured ∩ express)
    anchor_keys = set(site.site_key) if express_only else None
    traj, info = cm.predict_site(lat, lon, brand=brand, plateau_override=(ov or None),
                                 annual_mem_growth=mem_g + gm / 100, annual_ret_change=ret_g + gr / 100,
                                 mem_growth_band=(mem_lo + gm / 100, mem_hi + gm / 100),
                                 ret_change_band=(ret_lo + gr / 100, ret_hi + gr / 100), art=art,
                                 local_anchor=anchor_on, anchor_keys=anchor_keys, model_kind=model_kind)
    g = traj.set_index("month")
    # ── big, full-width interactive map: pan/zoom is smooth (no rerun) → click to drop the pin ──
    # EVERY site within the radius is a neighbour the model actually uses (local trend + level anchor) — including
    # singletons that aren't assigned to a cluster. Show them all (not just clustered) so the map matches the model.
    _nb = site[site.has_coords].copy()
    _nb["d"] = haversine_km(lat, lon, _nb.lat.values, _nb.lon.values)
    used = _nb[_nb.d <= radius].sort_values("d")               # neighbours used (within the radius)
    fmap = folium.Map(location=[lat, lon], zoom_start=10,          # same zoom as the Explore-markets map
                      tiles="cartodbpositron", control_scale=True, prefer_canvas=True)
    if demo:
        # confidential demo: no site dots, no exact pin — a soft shaded red region marks the chosen area
        for rad_km, fop in [(radius, 0.08), (radius * 0.55, 0.12), (radius * 0.28, 0.18)]:
            folium.Circle([lat, lon], radius=rad_km * 1000, color="#c0392b", weight=0,
                          fill=True, fill_color="#e6194B", fill_opacity=fop).add_to(fmap)
    else:
        add_all_site_dots(fmap, site)                          # ALL sites, bright (full footprint, same as Explore)
        for _, s in used.iterrows():                           # highlight the in-radius neighbours the model actually uses
            folium.CircleMarker([s.lat, s.lon], radius=6, color="#b34700", fill=True, fill_color="#ff7f0e",
                                fill_opacity=0.95, weight=1,
                                tooltip=f"{s.client_name} · {s.d:.1f} km · {int(s.n_obs)} mo of data").add_to(fmap)
        folium.Circle([lat, lon], radius=radius * 1000, color="#c0392b", weight=2, fill=False).add_to(fmap)
        folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="star"), tooltip="📍 your new site").add_to(fmap)
    # return ONLY last_clicked → panning/zooming no longer triggers a rerun (fixes the dimming-on-zoom)
    m = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
    lc = (m or {}).get("last_clicked")
    if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(lat, 5), round(lon, 5)):
        st.session_state.pin = (lc["lat"], lc["lng"])
        st.rerun()
    st.caption(f"🟠 **{len(used)} sites within {radius} km** feed the local trend; the level anchor uses "
               f"**{info.get('n_local_mature', 0)} matured** of them (needs ≥3, else it falls back to the global model). "
               f"🔵 all other sites · ⭕ {radius} km · ⭐ your pin. **Click to drop the pin.**")

    # ── 📊 Forecast at a glance — headline KPIs straight off the trajectory ──
    _plat = info.get("plateau_med")
    _peak = float(np.nanmax(g.total_med.to_numpy())) if len(g) else None
    _mm5 = float(np.nansum(g.mem_med.reindex(range(36, 61)))) if "mem_med" in g else 0.0
    _rr5 = float(np.nansum(g.ret_med.reindex(range(36, 61)))) if "ret_med" in g else 0.0
    _msat = (_mm5 / (_mm5 + _rr5)) if (_mm5 + _rr5) > 0 else None
    st.markdown("##### 📊 Forecast at a glance")
    _k1, _k2, _k3, _k4 = st.columns(4)
    _k1.metric("Plateau washes / mo", f"{_plat:,.0f}" if _plat else "—",
               help="Mature monthly wash level the forecast settles at (years 4–5).")
    _k2.metric("Peak washes / mo", f"{_peak:,.0f}" if _peak else "—",
               help="Highest monthly volume across the 5-year trajectory.")
    _k3.metric("Membership @ maturity", f"{_msat:.0%}" if _msat is not None else "—",
               help="Membership share of washes at maturity — higher = stickier, recurring demand.")
    _k4.metric("Sites in market", f"{len(used)}", help=f"Existing sites within {radius} km feeding the local trend.")

    st.divider(); st.subheader("📈 Total local-market wash count — history + 5-year forecast")
    gmk = gran_picker("gran_market")
    today = pd.Timestamp(df.date.max())
    H = 60
    fdates = pd.date_range(today + pd.DateOffset(months=1), periods=H, freq="MS")
    _tj = traj.set_index("month")
    new_traj = _tj["total_med"].reindex(range(H)).to_numpy()                           # the new entrant's own journey
    new_lo = _tj["total_lo"].reindex(range(H)).to_numpy(); new_hi = _tj["total_hi"].reindex(range(H)).to_numpy()
    nbk = site[site.has_coords].copy(); nbk["d"] = haversine_km(lat, lon, nbk.lat.values, nbk.lon.values)
    nbk = nbk[(nbk.d <= radius) & (nbk.d > 1e-6)]
    fig = go.Figure()
    if len(nbk):
        keys = nbk.site_key.tolist()
        comp = df[df.site_key.isin(keys)].groupby("date")[["mem_wash_count", "ret_wash_count"]].sum()
        idx = pd.date_range(comp.index.min(), today, freq="MS")
        hist_mem = comp["mem_wash_count"].reindex(idx); hist_ret = comp["ret_wash_count"].reindex(idx)
        hist = hist_mem.add(hist_ret, fill_value=0)                                   # total = membership + retail
        hist_disp = hist.rolling(smooth, center=True, min_periods=1).mean() if (smooth and smooth > 1) else hist   # honor the smoothing slider
        base_mem = forecast_series(hist_mem, H, g=mem_g)                              # membership forecast (central trend)
        base_ret = forecast_series(hist_ret, H, g=ret_g)                              # retail forecast (central trend)
        base_fc = base_mem + base_ret                                                 # market total WITHOUT the new entrant
        # data-based confidence band: re-forecast at the trend's lo/hi CI rates → a noisy market self-widens
        base_mem_lo = forecast_series(hist_mem, H, g=mem_lo); base_mem_hi = forecast_series(hist_mem, H, g=mem_hi)
        base_ret_lo = forecast_series(hist_ret, H, g=ret_lo); base_ret_hi = forecast_series(hist_ret, H, g=ret_hi)
        # cannibalization hits RETAIL (LEARNED a·exp(-d/L) for this region), phased over the 1st year
        rec = (df[df.site_key.isin(keys)].sort_values("date").groupby("site_key").tail(12)
               .groupby("site_key").agg(ret=("ret_wash_count", "mean")).join(nbk.set_index("site_key").d))
        cp = cm.cannib_params(art, lat, lon)                                          # LEARNED a·exp(-d/L) for this region
        cannib_full = float((cm._cannib_ret(rec.d.values, cp) * rec.ret.values).sum())
        phase = np.minimum(1.0, np.arange(1, H + 1) / 12.0)
        with_fc = np.clip(base_mem + np.clip(base_ret - cannib_full * phase, 0, None) + new_traj, 0, None)
        with_lo = np.clip(base_mem_lo + np.clip(base_ret_lo - cannib_full * phase, 0, None) + new_lo, 0, None)
        with_hi = np.clip(base_mem_hi + np.clip(base_ret_hi - cannib_full * phase, 0, None) + new_hi, 0, None)
        MKT = "#0a84ff"    # bright blue — one colour for the market total: solid history -> dotted forecast
        hist_s = rs_dates(hist_disp.dropna(), gmk)                                  # history summed into the chosen window
        _fc = lambda a: rs_dates(pd.Series(np.asarray(a, float), index=fdates), gmk)
        wfc, bfc, whi, wlo, ntr = _fc(with_fc), _fc(base_fc), _fc(with_hi), _fc(with_lo), _fc(new_traj)
        last, last_x = float(hist_s.iloc[-1]), hist_s.index[-1]                     # connect forecast to history endpoint
        fig.add_trace(go.Scatter(x=[last_x] + list(whi.index) + list(whi.index[::-1]) + [last_x],
                                 y=[last] + list(whi.values) + list(wlo.values[::-1]) + [last],
                                 fill="toself", fillcolor="rgba(10,132,255,0.12)", line=dict(width=0),
                                 name="forecast band (trend CI)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist_s.index, y=hist_s.values, line=dict(color=MKT, width=3.2), name="market total — actual history"))
        fig.add_trace(go.Scatter(x=[last_x] + list(wfc.index), y=[last] + list(wfc.values), line=dict(color=MKT, width=3.2, dash="dot"), name="market total — forecast (with new site)"))
        fig.add_trace(go.Scatter(x=[last_x] + list(bfc.index), y=[last] + list(bfc.values), line=dict(color="#9aa6b2", width=1.6, dash="dot"), name="market without the new site"))
        fig.add_trace(go.Scatter(x=ntr.index, y=ntr.values, line=dict(color="#ff375f", width=3), name="🆕 new entrant — its own journey"))
        fig.add_vline(x=today, line=dict(color="#c0392b", dash="dash", width=1.5))
        fig.add_annotation(x=today, yref="paper", y=1.03, text="new site opens", showarrow=False, font=dict(color="#c0392b", size=11))
        fig.update_layout(height=480, template="plotly_white", hovermode="x unified", xaxis_title="date",
                          yaxis_title=f"market total washes / {GRAN_UNIT[gmk]}", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28))
        fig.update_xaxes(tickformat=gran_date_tickformat(gmk))
        st.plotly_chart(fig, width="stretch")
    else:
        ntr = rs_dates(pd.Series(np.asarray(new_traj, float), index=fdates), gmk)
        fig.add_trace(go.Scatter(x=ntr.index, y=ntr.values, line=dict(color="#e6194B", width=3), name="🆕 new site"))
        fig.update_layout(height=420, template="plotly_white", xaxis_title="date", yaxis_title=f"washes / {GRAN_UNIT[gmk]}",
                          margin=dict(l=10, r=10, t=20, b=10))
        fig.update_xaxes(tickformat=gran_date_tickformat(gmk))
        st.plotly_chart(fig, width="stretch")
        st.info(f"No existing sites within {radius} km — a fresh market, so the chart shows only the new site's own 5-year journey.")

    # the new site's own 5-year trajectory (after the market view)
    st.divider(); st.subheader("Predicted 5-year trajectory")
    _ml = "Model 3 (ExtraTrees)" if model_kind == "et" else "Model 2"   # both use the local anchor; level model differs
    if ov:
        st.caption(f"🔧 Plateau **manually overridden** to {int(ov):,}/mo (model ignored).")
    elif info.get("brand_known"):
        st.caption(f"🏢 **Model 4 (operator):** using **{op_label}** — that operator's avg mature level "
                   f"({art['brand_mean'][brand]:,.0f}/mo) → plateau **{info['plateau_med']:,.0f}/mo**. "
                   f"(Operator-known is the most accurate setup: ~34% WAPE vs ~40% cold-start.)")
    elif strat.get("use_operator"):                                      # Model 4 chosen but no operator selected
        st.caption("🏢 **Model 4 (operator)** — no operator selected, so this falls back to the cold-start anchor "
                   "(= Model 3). Pick an operator in the sidebar for the operator-based forecast.")
    elif anchor_on and info.get("proxy_used"):
        _lvl = "ExtraTrees level" if model_kind == "et" else "model's operator-level slot"
        st.caption(
            f"🔧 **{_ml}:** {info['n_local_mature']} matured sites ≤20 km (median {info['anchor_level']:,.0f}/mo) "
            f"feed the {_lvl} → plateau **{info['plateau_med']:,.0f}/mo** "
            f"(vs {info['model_plateau']:,.0f}/mo without it).")
    elif anchor_on and info.get("n_local_mature", 0) > 0:
        st.caption(
            f"🔧 **{_ml}:** {info['n_local_mature']} local matured sites are too varied "
            f"(CoV {info['local_cov']:.1f}) to anchor reliably — falling back to the base model "
            f"(**{info['plateau_med']:,.0f}/mo**).")
    elif anchor_on:
        _none = "no **express** matured sites" if express_only else "no matured sites"
        st.caption(f"🔧 **{_ml}** on, but {_none} within 20 km — falls back to the base model (no local anchor) here.")
    else:                                                        # unreachable now (all kept models anchor); defensive
        st.caption(f"🔧 **Base model (no anchor):** plateau **{info['plateau_med']:,.0f}/mo**.")
    gtr = gran_picker("gran_traj")
    xb, tot = agg_months(g.total_med.to_numpy(), gtr)
    _, hi = agg_months(g.total_hi.to_numpy(), gtr); _, lo = agg_months(g.total_lo.to_numpy(), gtr)
    _, memv = agg_months(g.mem_med.to_numpy(), gtr); _, retv = agg_months(g.ret_med.to_numpy(), gtr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(xb) + list(xb[::-1]), y=list(hi) + list(lo[::-1]),
                             fill="toself", fillcolor="rgba(41,128,185,0.15)", line=dict(width=0), name="P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xb, y=tot, line=dict(color="#2980b9", width=3), name="total"))
    fig.add_trace(go.Scatter(x=xb, y=memv, line=dict(color="#16a085", width=2), name="membership"))
    fig.add_trace(go.Scatter(x=xb, y=retv, line=dict(color="#c0392b", width=2), name="retail"))
    fig.update_layout(height=400, yaxis_title=f"washes / {GRAN_UNIT[gtr]}", hovermode="x unified",
                      template="plotly_white", margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    gran_xaxes_months(fig, gtr, xb)
    st.plotly_chart(fig, width="stretch")

    # ───────── P&L: revenue (membership purchases × $/purchase + retail washes × $/wash) vs expenses ─────────
    st.divider(); st.subheader("💰 P&L — revenue vs expenses (OPEX + CAPEX)")
    # CLUSTER ASP from the dense operational data (main-ds): the ≤radius km neighbours' last 12 months — a real cluster
    # figure. Membership ASP = revenue ÷ PURCHASES (per-membership price); retail ASP = revenue ÷ washes. The forecast
    # projects membership WASHES, so we convert to purchases with the cluster purchases-per-wash ratio before pricing.
    _nbk = site[site.has_coords].copy()
    _nbk["d"] = haversine_km(lat, lon, _nbk.lat.values, _nbk.lon.values)
    _ck = _nbk.site_key[(_nbk.d <= radius) & (_nbk.d > 1e-6)].tolist()
    _rec = df[df.site_key.isin(_ck)].sort_values("date").groupby("site_key").tail(12) if _ck else df.iloc[:0]
    # Drop corrupted site-months (revenue feed dropped to ~0 with washes intact) BEFORE pooling — else a couple
    # of bad neighbours halve the cluster $/wash and the forecast revenue with it. See _drop_corrupt_asp_rows.
    _rec, _n_drop = _drop_corrupt_asp_rows(_rec)
    _mm, _rr = _rec.dropna(subset=["mem_revenue"]), _rec.dropna(subset=["ret_revenue"])
    _mp, _mw = _mm.mem_purchase_count.sum(), _mm.mem_wash_count.sum()
    # Fallback chain: clean cluster pool → if a side is empty (all neighbours corrupt / no market) use the
    # GLOBAL healthy-median ASP (never the flat $30/$15, which can be worse than a degraded-but-real figure).
    _g_mem_pp, _g_ppw, _g_ret = global_healthy_asp(express_only)
    cl_mem_pp = float(_mm.mem_revenue.sum() / _mp) if _mp > 0 else _g_mem_pp     # cluster $/membership PURCHASE (the new ASP)
    purch_per_wash = float(_mp / _mw) if _mw > 0 else _g_ppw                     # cluster membership purchases per membership wash
    cl_ret = float(_rr.ret_revenue.sum() / _rr.ret_wash_count.sum()) if _rr.ret_wash_count.sum() > 0 else _g_ret
    if not _ck:
        asp_scope = "global healthy avg (no neighbours)"
    elif _mp <= 0 and _rr.ret_wash_count.sum() <= 0:
        asp_scope = f"global healthy avg ({len(_ck)} neighbours all corrupt)"
    else:
        asp_scope = f"cluster ≤{radius} km · {len(_ck)} sites" + (f" · {_n_drop} corrupt site-mo excluded" if _n_drop else "")
    tj = traj.set_index("month"); months = np.arange(0, 61)
    mem = tj["mem_med"].reindex(months).fillna(0.0).to_numpy()                  # projected membership WASHES
    ret = tj["ret_med"].reindex(months).fillna(0.0).to_numpy()                  # projected retail washes
    mem_purch = mem * purch_per_wash                                            # → projected membership PURCHASES (revenue basis)

    # ── cost assumptions: one editable Year × {ASP, OPEX %, CAPEX} grid ──
    opex_shape = opex_pct_curve_fit(load_pnl_monthly(), art, info.get("state"), info.get("region"), months)
    _yslices = year_slices(len(months))
    _rev0 = mem_purch * cl_mem_pp + ret * cl_ret                                # cluster-blend revenue (for the OPEX default weighting)
    def _learned_pct(sl):                                                       # learned opex% per year → the OPEX defaults (Y4–5 extrapolated)
        seg, sr = opex_shape[sl], _rev0[sl]
        w = sr if float(np.nansum(sr)) > 0 else np.ones_like(seg)
        return int(min(150, max(0, round(100 * float(np.average(seg, weights=w))))))
    # ── 🏗️ Expected tunnel length → build CAPEX (learned from 187 real proforma builds) ──
    st.markdown("##### 🏗️ Expected tunnel length → build CAPEX")
    # 💡 Recommended tunnel length from the forecast's PEAK monthly washes (same proxy as the explore-markets plot):
    #   peak-month washes ÷ 250 (25 operating days × 10 h/day → peak cars/hour) ÷ 3.2 + 10 → tunnel metres.
    _fc_tot = np.asarray(mem, float) + np.asarray(ret, float)                   # forecast TOTAL washes/mo (60-mo trajectory)
    _peak_fc = float(np.nanmax(_fc_tot)) if _fc_tot.size and np.isfinite(np.nanmax(_fc_tot)) else 0.0
    rec_len = (_peak_fc / 250 / 3.2 + 10) if _peak_fc > 0 else None
    if rec_len:
        st.info(f"💡 **Recommended tunnel length ≈ {rec_len:.0f} m** — from your forecast peak of "
                f"**{_peak_fc:,.0f} washes/mo** (peak ÷ 250 ÷ 3.2 + 10). Rough proxy — used to pre-pick the band below.")
    _ctab = capex_band_table()
    if _CAPEX_OK and len(_ctab):
        _opts = _ctab.band.tolist()
        _slope, _intr, _corr, _nfit = capex_fit()
        # pre-select the band that CONTAINS the recommended length (fallback 35–40 m)
        _rec_band = next((row.band for _, row in _ctab.iterrows() if rec_len and row.lo <= rec_len < row.hi), None)
        _def_ix = (_opts.index(_rec_band) if _rec_band in _opts
                   else (_opts.index("35–40 m") if "35–40 m" in _opts else len(_opts) // 2))

        def _capex_opt_label(b):
            r = _ctab[_ctab.band == b].iloc[0]
            return f"{b}  ·  ≈ ${r['median'] / 1e6:.1f}M" + (f"  (n={int(r['n'])})" if r["n"] else "")

        tlen_band = st.selectbox("Expected tunnel length", _opts, index=_def_ix, format_func=_capex_opt_label,
                                 key=f"tlen_band_{round(lat, 3)}_{round(lon, 3)}",
                                 help="Pre-set to the band matching the recommended length above; change it to override. "
                                      "Sets the build CAPEX from the median total investment of real builds at that length.")
        _band_med = float(_ctab[_ctab.band == tlen_band]["median"].iloc[0])
        build_capex = float(st.number_input(
            "Build CAPEX ($) — auto-set from tunnel length (editable)", min_value=0,
            value=int(round(_band_med)), step=50_000, format="%d", key=f"build_capex_{tlen_band}",
            help="Total build/acquisition investment for this tunnel size, spread evenly across the 5-year P&L. "
                 "Edit to override the data-driven default."))
        _cap_cap = (f"Build CAPEX **${build_capex:,.0f}** for a **{tlen_band}** tunnel "
                    f"(spread evenly over 5 years ≈ **${build_capex / 60:,.0f}/mo**).")
        if _slope:
            _cap_cap += f" Across {_nfit} real builds, CAPEX rises ~${_slope:,.0f}/m (corr {_corr:.2f})."
        st.caption(_cap_cap.replace("$", "\\$"))                                 # escape $ so Streamlit doesn't render LaTeX
        with st.expander("📊 How tunnel length drives CAPEX (real proforma builds)"):
            _b = capex_builds()

            def _bx(lo, hi):                                                     # band marker at its real median length
                seg = _b[(_b.tlen >= lo) & (_b.tlen < hi)] if len(_b) else _b
                return float(seg.tlen.median()) if len(seg) else (lo + min(hi, lo + 10)) / 2
            if len(_b):
                _cfig = go.Figure()
                _cfig.add_trace(go.Scatter(x=_b.tlen, y=_b.capex, mode="markers", name="real builds",
                                           marker=dict(size=6, color="#9ecae1", opacity=0.55),
                                           hovertemplate="%{x:.0f} m · $%{y:,.0f}<extra></extra>"))
                _cx = [_bx(r.lo, r.hi) for r in _ctab.itertuples()]
                _cfig.add_trace(go.Scatter(x=_cx, y=_ctab["median"], mode="lines+markers", name="band median",
                                           line=dict(color="#2171b5", width=2),
                                           hovertemplate="band median $%{y:,.0f}<extra></extra>"))
                _sel = _ctab[_ctab.band == tlen_band].iloc[0]
                _cfig.add_trace(go.Scatter(x=[_bx(_sel.lo, _sel.hi)], y=[build_capex], mode="markers",
                                           name="your choice", marker=dict(size=16, color="#e6194B", symbol="star"),
                                           hovertemplate="your build · $%{y:,.0f}<extra></extra>"))
                _cfig.update_layout(height=340, template="plotly_white", margin=dict(l=8, r=8, t=10, b=10),
                                    legend=dict(orientation="h", y=-0.2),
                                    xaxis_title="tunnel length (m)", yaxis_title="total build CAPEX ($)")
                st.plotly_chart(_cfig, width="stretch", key="capex_scatter")
            _show = _ctab.assign(**{"median CAPEX": _ctab["median"].map(lambda v: f"${v:,.0f}"),
                                    "mean CAPEX": _ctab["mean"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "—")})
            st.dataframe(_show[["band", "n", "median CAPEX", "mean CAPEX"]].rename(columns={"band": "tunnel length"}),
                         hide_index=True, width="stretch")
            st.caption("CAPEX = `project_cost_total_investment` from builds with a known tunnel length & non-zero "
                       "investment. The median is the auto-set default; band medians are kept non-decreasing.")
    else:
        build_capex = 0.0
        st.caption("_Tunnel-length CAPEX model unavailable — enter CAPEX manually in the grid below._")

    with st.expander("⚙️ Cost assumptions — by year", expanded=True):
        _plan0 = pd.DataFrame({
            "Year": [f"Year {i + 1}" for i in range(5)],
            "Mem ASP ($/purchase)": [round(min(cl_mem_pp, 100.0), 1)] * 5,
            "Retail ASP ($/wash)": [round(min(cl_ret, 60.0), 1)] * 5,
            "OPEX (% of sales)": [_learned_pct(sl) for sl in _yslices],
            "Extra CAPEX ($)": [0] * 5,
        })
        plan = st.data_editor(
            _plan0, hide_index=True, width="stretch", num_rows="fixed", disabled=["Year"], key="pnl_plan",
            column_config={
                "Mem ASP ($/purchase)": st.column_config.NumberColumn(
                    min_value=1.0, max_value=100.0, step=0.5, format="$%.1f",
                    help=f"Membership revenue per PURCHASE for the year. Cluster avg ${cl_mem_pp:.1f} ({asp_scope})."),
                "Retail ASP ($/wash)": st.column_config.NumberColumn(
                    min_value=1.0, max_value=60.0, step=0.5, format="$%.1f",
                    help=f"Retail revenue per WASH for the year. Cluster avg ${cl_ret:.1f} ({asp_scope})."),
                "OPEX (% of sales)": st.column_config.NumberColumn(
                    min_value=0, max_value=150, step=1, format="%d%%",
                    help="Operating expense as % of sales; fitted onto the learned hot→mature pattern. Defaults follow the learned curve (Years 4–5 extrapolated)."),
                "Extra CAPEX ($)": st.column_config.NumberColumn(
                    min_value=0, step=25000, format="$%d",
                    help="ADDITIONAL CAPEX that year (e.g. refurb), on TOP of the tunnel-length build CAPEX above; "
                         "spread across that year's months."),
            },
        )
        mem_asp_in = {i + 1: float(plan["Mem ASP ($/purchase)"].iloc[i]) for i in range(5)}
        ret_asp_in = {i + 1: float(plan["Retail ASP ($/wash)"].iloc[i]) for i in range(5)}
        opex_in = {i + 1: float(plan["OPEX (% of sales)"].iloc[i]) for i in range(5)}
        extra_capex = {i + 1: float(plan["Extra CAPEX ($)"].iloc[i]) for i in range(5) if float(plan["Extra CAPEX ($)"].iloc[i])}
    capex_in = dict(extra_capex)
    if build_capex:                                                            # tunnel-length build CAPEX spread evenly over 5 years
        for _y in range(1, 6):
            capex_in[_y] = capex_in.get(_y, 0.0) + build_capex / 5
    asp_mem_pp = per_year_to_monthly(len(months), mem_asp_in, cl_mem_pp)        # $/purchase per month (membership)
    asp_ret = per_year_to_monthly(len(months), ret_asp_in, cl_ret)             # $/wash per month (retail)
    rev_base = mem_purch * asp_mem_pp + ret * asp_ret                          # sales = mem purchases × $/purchase + retail washes × $/wash

    # ── baseline expenses (no campaign): OPEX %/yr fitted onto the learned hot→mature shape; CAPEX spread per year ──
    opex_pct, opex_tgts = fit_opex_pct_to_targets(opex_shape, rev_base, opex_in)
    opex_base_m = opex_pct * rev_base                                  # OPEX $ (no campaign)
    capex_m = spread_capex(len(months), capex_in)                     # CAPEX $ spread across each year
    total_exp_base = opex_base_m + capex_m
    net_base = rev_base - total_exp_base

    gpnl = gran_picker("gran_pnl")
    st.plotly_chart(_pnl_figure(rev_base, opex_base_m, capex_m, total_exp_base, net_base, gpnl,
                                title="P&L — revenue vs expenses (OPEX + CAPEX) · net income, 5-year"), width="stretch")
    # Breakeven = cash PAYBACK: months until cumulative OPERATING profit (sales − OPEX) recovers the FULL upfront
    # build + extra CAPEX. The P&L line amortizes CAPEX over 5y for the income-statement view, but payback must
    # treat it as spent upfront — else it reports breakeven (e.g. month 15) while ~75% of the build is unrecovered.
    _oper = np.nan_to_num(rev_base - opex_base_m)                     # operating profit, CAPEX NOT amortized here
    _capex_tot = float(np.nansum(capex_m))
    _cum_oper = np.cumsum(_oper)
    _be = (int(np.argmax(_cum_oper >= _capex_tot)) + 1) if (_cum_oper >= _capex_tot).any() else None  # 1-indexed
    _sales5 = float(np.nansum(rev_base)); _opex5 = float(np.nansum(opex_base_m))
    _capex5 = float(np.nansum(capex_m)); _net5 = float(np.nansum(net_base))
    st.markdown("**5-year totals**")
    _t1, _t2, _t3, _t4, _t5 = st.columns(5)
    _t1.metric("Sales", f"${_sales5 / 1e6:.2f}M")
    _t2.metric("OPEX", f"${_opex5 / 1e6:.2f}M", help="Operating expenses (excl. CAPEX) over 5 years.")
    _t3.metric("CAPEX", f"${_capex5 / 1e6:.2f}M", help="Build investment, spread evenly over the 5 years.")
    _t4.metric("Net income", f"${_net5 / 1e6:.2f}M", delta=(f"{_net5 / _sales5:.0%} margin" if _sales5 > 0 else None))
    _t5.metric("Breakeven", (f"month {_be}" if _be is not None else "—"),
               help=("Cash payback: month cumulative operating profit (sales − OPEX) recovers the full upfront "
                     f"build CAPEX (${_capex_tot/1e6:.2f}M)." if _be is not None
                     else "Operating profit doesn't recover the build CAPEX within 5 years."))
    st.caption((f"ROI (5-yr net ÷ CAPEX) **{_net5 / _capex5:.0%}** · " if _capex5 > 0 else "")
               + "OPEX %/yr (fitted): " + ", ".join(f"Y{y}={t * 100:.0f}%" for y, t in opex_tgts.items()) + ".")

    # ════════════ CAMPAIGN ANALYSIS — self-contained, separate from the baseline P&L above ════════════
    st.divider()
    st.header("🎯 Campaign analysis — should this site run a launch promotion?")
    st.caption("A promo converts retail visits into membership and steals some neighbour share. Everything below is "
               "the **campaign scenario** — the P&L above stays the campaign-free baseline.")
    ms = float(info.get("mem_share", 0.6))                       # fallback if the trajectory is empty
    # the LOCAL MARKET incumbents: established (≥2yr) sites within the radius — do they prove membership works?
    _loc = site[site.has_coords].copy(); _loc["d"] = haversine_km(lat, lon, _loc.lat.values, _loc.lon.values)
    _inc = _loc[(_loc.d <= radius) & (_loc.d > 1e-6) & (_loc.n_obs >= 24)]
    n_inc = len(_inc)
    _r12 = df[df.site_key.isin(_inc.site_key)].sort_values("date").groupby("site_key").tail(12)
    # neighbours' membership share = MEDIAN of each incumbent's own recent 12-mo share (every site counts equally)
    _per = _r12.groupby("site_key").apply(
        lambda gg: gg.mem_wash_count.fillna(0).sum()
        / max(1.0, gg.mem_wash_count.fillna(0).sum() + gg.ret_wash_count.fillna(0).sum()), include_groups=False)
    nb_ms = float(_per.median()) if len(_per) else float("nan")
    # THIS SITE's predicted membership = what its 5-yr trajectory SETTLES at (plateau months 36–60).
    _plm, _plr = float(np.nansum(mem[36:61])), float(np.nansum(ret[36:61]))
    if _plm + _plr > 0:
        ms = _plm / (_plm + _plr)
    conv = campaign_conv_pct(ms)
    # verdict — only recommend when the market is PROVEN: good established incumbents + membership works + share to take
    if n_inc < 2:
        ok, verdict = False, (f"🔴 **Not recommended.** Only **{n_inc}** established incumbent(s) within {radius} km — the "
                              f"market is unproven, and in the data **77% of promos captured no share**. Choose a denser market.")
    elif not np.isfinite(nb_ms) or nb_ms < 0.45:
        ok, verdict = False, (f"🔴 **Not recommended.** Neighbours' membership share is **{nb_ms:.0%}** (low) — the membership "
                              f"model isn't proven here, so converted customers are unlikely to stick.")
    elif nb_ms >= 0.82:
        ok, verdict = False, (f"🟠 **Marginal.** Neighbours are **{nb_ms:.0%}** membership — the market is near-saturated, "
                              f"so there's little retail left to convert or steal.")
    else:
        ok, verdict = True, (f"🟢 **Recommended.** **{n_inc}** established incumbents within {radius} km at **{nb_ms:.0%}** "
                             f"membership.")
    cA, cB, cC = st.columns(3)
    cA.metric("Neighbours' membership share", f"{nb_ms:.0%}" if np.isfinite(nb_ms) else "—",
              help="MEDIAN of each established (≥2yr) incumbent's own recent 12-mo membership share (every site counts equally) — does membership work here?")
    cB.metric("Established incumbents", n_inc, help=f"Sites with ≥2 years of history within {radius} km — proof the market is viable & competitive.")
    cC.metric("This site's predicted membership", f"{ms:.0%}",
              help="Membership share its 5-year trajectory settles at (predicted membership ÷ total washes at maturity) — the same split shown in the trajectory plot. Lower = more retail headroom to convert.")
    (st.success if verdict.startswith("🟢") else st.warning if verdict.startswith("🟠") else st.error)(verdict)
    with st.expander("Apply a campaign and see the P&L change", expanded=False):
        camp_on = st.checkbox("Apply campaign", value=False)
        cc1, cc2 = st.columns(2)
        c_launch = cc1.slider("Launch — month after opening", 1, 48, 13, disabled=not camp_on)
        c_int = cc2.slider("Intensity (× typical campaign)", 0.5, 1.5, 1.0, 0.1, disabled=not camp_on,
                           help="Scales the conversion lift and the promo spend. 1.0 = a typical observed campaign.")
    WIN = 6                                                            # campaign impact window (months)
    mem_mult, ret_mult, opex_mult_c = (campaign_effect(c_launch, ms, c_int, window=WIN) if camp_on
                                       else (np.ones(61), np.ones(61), np.ones(61)))
    if camp_on:                                                        # additional info: the real OPEX/Rev/Profit/Mem breakdown
        with st.popover("Additional info — what a campaign does to OPEX / Revenue / Profit / Membership", width=1000):
            render_campaign_snapshot()
    # campaign scenario: revenue shifts retail→membership; OPEX carries the promo spend over the window; CAPEX unchanged
    rev_m = mem_purch * mem_mult * asp_mem_pp + ret * ret_mult * asp_ret
    opex_m = opex_base_m * opex_mult_c
    total_exp_m = opex_m + capex_m
    net_m = rev_m - total_exp_m

    # second copy of the SAME P&L plot — now the campaign scenario (overlays the no-campaign revenue when applied)
    gpnl_c = gran_picker("gran_pnl_camp")
    st.plotly_chart(_pnl_figure(rev_m, opex_m, capex_m, total_exp_m, net_m, gpnl_c,
                                title="P&L WITH campaign — revenue vs expenses (OPEX + CAPEX) · net income, 5-year",
                                camp_on=camp_on, rev_base=rev_base, c_launch=c_launch, win=WIN), width="stretch")
    if camp_on:
        _net5c = float(np.nansum(net_m)); _dnet = _net5c - _net5
        st.caption(f"Campaign 5-yr net **\\${_net5c / 1e6:.2f}M** vs baseline **\\${_net5 / 1e6:.2f}M** "
                   f"(**{'+' if _dnet >= 0 else '−'}\\${abs(_dnet) / 1e6:.2f}M** from the promo).")
    else:
        st.caption("_Tick **Apply campaign** in the expander above to overlay the promo scenario on this plot._")

    # ── eating the market: your site vs EACH incumbent, each time-series-forecast forward 5 years ──
    if n_inc >= 1:
        st.markdown("##### 📈 Eating the market — your site vs each incumbent (5-year)")
        geat = gran_picker("gran_eat")
        new_base = mem + ret                                           # new site's washes/mo (no campaign)
        new_camp = mem * mem_mult + ret * ret_mult                     # with the campaign's conversion lift
        steal_peak = 0.06 * min(1.0, n_inc / 4.0) * (c_int if camp_on else 1.0)   # share theft scales with density
        steal = np.zeros(61)
        if camp_on:
            for t in range(61):
                k = t - c_launch
                if 0 <= k < WIN:
                    steal[t] = steal_peak * min(1.0, (k + 1) / 2.0)
                elif k >= WIN:
                    steal[t] = steal_peak * 0.5 ** ((k - WIN) / 12.0)  # incumbents recover as the promo fades
        # rank incumbents by recent volume; forecast the largest few forward (cap for legibility)
        _rank = (_r12.assign(w=_r12.mem_wash_count.fillna(0) + _r12.ret_wash_count.fillna(0))
                 .groupby("site_key").w.mean().sort_values(ascending=False))
        shown = _rank.index[:6].tolist()
        IPAL = ["#5b8db8", "#8e44ad", "#e67e22", "#2c3e50", "#7f8c8d", "#9b59b6"]
        lbl = (anon_names(site, set(shown)) if demo
               else {k: str(site.loc[site.site_key == k, "client_name"].iloc[0])[:18] for k in shown})
        xb, nb_a = agg_months(new_base, geat)
        sf = go.Figure()
        for i, k in enumerate(shown):                                  # each incumbent = its own forward forecast
            hist = df[df.site_key == k].set_index("date").sort_index()["tot_wash_count"].dropna()
            if hist.empty:
                continue
            yb = np.concatenate([[float(hist.iloc[-1])], forecast_series(hist, 60)])   # m0 = last actual, m1..60 = forecast
            col = IPAL[i % len(IPAL)]
            nm = lbl.get(k, "incumbent")
            _, yb_a = agg_months(yb, geat)
            sf.add_trace(go.Scatter(x=xb, y=yb_a, name=nm, mode="lines", line=dict(color=col, width=1.8, dash="dot"),
                                    hovertemplate=f"m%{{x}} · %{{y:,.0f}}<extra>{nm} (expected)</extra>"))
            if camp_on:                                                # drifts DOWN as your promo steals its retail
                _, ybc_a = agg_months(yb * (1 - steal), geat)
                sf.add_trace(go.Scatter(x=xb, y=ybc_a, name=f"{nm} (campaign)", mode="lines",
                                        line=dict(color=col, width=2.4), showlegend=False,
                                        hovertemplate=f"m%{{x}} · %{{y:,.0f}}<extra>{nm} (with campaign)</extra>"))
        # your site (green): dotted = expected · solid = with campaign (drifts up)
        sf.add_trace(go.Scatter(x=xb, y=nb_a, name="your site — expected", mode="lines",
                                line=dict(color="#16a085", width=2.5, dash="dot"),
                                hovertemplate="m%{x} · %{y:,.0f}<extra>your site (expected)</extra>"))
        if camp_on:
            _, nc_a = agg_months(new_camp, geat)
            sf.add_trace(go.Scatter(x=xb, y=nc_a, name="your site — with campaign", mode="lines",
                                    line=dict(color="#16a085", width=3.5),
                                    hovertemplate="m%{x} · %{y:,.0f}<extra>your site (with campaign)</extra>"))
            sf.add_vrect(x0=c_launch, x1=min(60, c_launch + WIN), fillcolor="rgba(230,25,75,0.08)", line_width=0,
                         annotation_text="campaign", annotation_position="top left")
            sf.add_vline(x=c_launch, line=dict(color="#c0392b", dash="dash", width=1.2))
        sf.update_layout(height=400, template="plotly_white", hovermode="x unified",
                         yaxis_title=f"washes / {GRAN_UNIT[geat]}", margin=dict(l=10, r=10, t=20, b=10),
                         legend=dict(orientation="h", y=-0.3, font=dict(size=10)))
        gran_xaxes_months(sf, geat, xb, noun="opening")
        st.plotly_chart(sf, width="stretch")
        if len(shown) < n_inc:
            st.caption(f"Showing the {len(shown)} largest of {n_inc} incumbents (each forecast forward individually).")

    campaign_cluster_panel(df, site, lat, lon, radius, demo)


# ───────────────────────────── UI ─────────────────────────────
st.set_page_config(page_title="Local Market Explorer", layout="wide")
# ── light global polish: card-style metrics, tidier expanders/headers (no logic, pure CSS) ──
st.markdown("""
<style>
/* theme-agnostic: semi-transparent grey tints work on BOTH light and dark backgrounds (no hardcoded white) */
[data-testid="stMetric"]{background:rgba(128,128,128,.10);border:1px solid rgba(128,128,128,.22);
  border-radius:12px;padding:10px 16px;}
[data-testid="stMetricLabel"]{opacity:.8;}
[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:600;}
[data-testid="stMetricDelta"]{font-size:.82rem;}
div[data-testid="stExpander"] details{border:1px solid rgba(128,128,128,.22);border-radius:12px;}
h1{letter-spacing:-.02em;} h2,h3{letter-spacing:-.01em;padding-top:.15rem;}
hr{margin:.7rem 0;}
/* readable AI summaries — generous line-height, spaced headings & lists so a long LLM report scans cleanly */
[data-testid="stMarkdownContainer"] p{line-height:1.58;margin:.35rem 0;}
[data-testid="stMarkdownContainer"] li{line-height:1.5;margin:.14rem 0;}
[data-testid="stMarkdownContainer"] ul,[data-testid="stMarkdownContainer"] ol{margin:.25rem 0 .6rem;}
[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{margin:.85rem 0 .3rem;font-weight:650;}
[data-testid="stMarkdownContainer"] h4{font-size:1.02rem;}
/* bordered summary cards: rounded, soft tint, comfortable padding */
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px;}
[data-testid="stVerticalBlockBorderWrapper"]>div{padding:.2rem .25rem;}
</style>
""", unsafe_allow_html=True)
MODES = ["🛰️ Sitewise", "🗺️ Explore markets", "📍 Pinpoint forecast"]
with st.sidebar:
    st.header("Controls")
    demo = st.toggle("👔 Client demo (anonymized)", value=False,
                     help="Hide identities: sites become 'Site 1, 2, 3…' by opening order, operators become "
                          "'Operator N', and the map shows a shaded red region instead of exact dots.")
    express_only = st.toggle("🚿 Express-only sites", value=False,
                     help="Restrict markets, clusters, KPIs and the forecast's wash trajectory / level anchor to "
                          "Express Tunnel car washes (drops Flex, full-service, self-serve, unknown, etc.). The "
                          "P&L cost curves and the global campaign-evidence popover still use all sites.")
# load AFTER the express toggle so the whole app (clusters included) reruns on the filtered universe
df, site = load_data(express_only)
pins = interesting_pins(site)
with st.sidebar:
    # mode switcher — segmented control (acts like app tabs)
    app_mode = st.segmented_control("Mode", MODES, default=MODES[0], key="app_mode") or MODES[0]
    if app_mode.startswith("🗺️"):
        # distance + smoothing apply to Explore only (Forecast is fixed to the trained cluster radius)
        st.divider()
        radius = st.slider("Neighbour radius (km)", 2, 40, 20, 1, key="radius")
        smooth = st.select_slider("Smoothing (months)", [1, 2, 3, 4, 6], value=1, key="smooth")
if app_mode.startswith("📍"):
    drop_pin_ui(df, site, get_model(), demo, express_only)
    st.stop()
if app_mode.startswith("🛰️"):
    if "pin" not in st.session_state:                        # seed the SAME default point Explore uses
        _k0 = pick_default_pin(site, df, tuple(pins))
        _s0 = site.loc[site.site_key == _k0].iloc[0]
        st.session_state.pin = (float(_s0.lat), float(_s0.lon))
    svp.render(demo)
    st.stop()

st.title("PROFORMA DEMO")
if express_only:
    st.caption(f"🚿 **Express-only sites** — every market, cluster, KPI and forecast below uses only the "
               f"{len(site):,} Express Tunnel sites with ≥{EXPRESS_MIN_MONTHS} months of history.")

with st.sidebar:
    max_sites = 10  # cap on sites shown; the pin and all new entrants are always kept
    hl_client = None
    if not demo:                                          # highlight one operator's whole footprint (hidden in demo)
        _clients = sorted(site.client_name.dropna().unique())
        _sel = st.selectbox("Highlight operator / brand", ["(none)"] + _clients, index=0)
        hl_client = None if _sel == "(none)" else _sel
    st.divider()
    if "pin" not in st.session_state:
        _k0 = pick_default_pin(site, df, tuple(pins))
        _s0 = site.loc[site.site_key == _k0].iloc[0]
        st.session_state.pin = (float(_s0.lat), float(_s0.lon))       # pin is a free (lat, lon) point
    # freedom: drop a pin anywhere by typing coordinates → shows whatever sites fall in the radius
    with st.expander("📍 Or type any location (lat, lon)"):
        _plat, _plon = st.session_state.pin
        ilat = st.number_input("Latitude", value=float(_plat), format="%.4f", key="ex_lat")
        ilon = st.number_input("Longitude", value=float(_plon), format="%.4f", key="ex_lon")
        if st.button("Drop pin here", width="stretch"):
            st.session_state.pin = (float(ilat), float(ilon))
            st.rerun()

pin = st.session_state.pin                                            # (lat, lon) free point
plat, plon = pin
# Explore: rich-history sites only — ≥30 monthly records (the SAME floor as express) → no half-drawn lines, and
# we don't dot thin/young sites you can't actually analyze
MIN_MONTHS = EXPRESS_MIN_MONTHS
site_rich = site[site.n_obs >= MIN_MONTHS]
nb_full = neighbourhood(site_rich, plat, plon, radius)
if nb_full.empty:
    st.warning(f"No sites with ≥{MIN_MONTHS} months of data within {radius} km of this pin — drop it elsewhere or widen the radius.")
    # still show the map so you can see where the data actually is and move the pin there
    st.subheader("Map")
    fmap = folium.Map(location=[plat, plon], zoom_start=9, tiles="cartodbpositron", prefer_canvas=True)
    folium.Circle([plat, plon], radius=radius * 1000, color="#999", weight=1, fill=True, fill_opacity=0.05).add_to(fmap)
    if demo:
        add_cluster_regions(fmap, site, plat, plon, max_km=50)
    else:
        add_all_site_dots(fmap, site_rich)                                # rich-history sites (≥30 mo) on the map, pan anywhere
        if hl_client:
            for _, s in site[site.has_coords & (site.client_name == hl_client)].iterrows():
                folium.CircleMarker([s.lat, s.lon], radius=6, color="#d4a500", fill=True, fill_color="#ffd000",
                                    fill_opacity=0.95, weight=2, tooltip=f"{s.client_name} (operator)").add_to(fmap)
    folium.Marker([plat, plon], icon=folium.Icon(color="black", icon="star"), tooltip="📍 pin").add_to(fmap)
    mp = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
    lc = (mp or {}).get("last_clicked")
    if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(plat, 5), round(plon, 5)):
        st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
    st.caption((f"Dots = every express site" if express_only else "Dots = every site")
               + f" with ≥{MIN_MONTHS} months of history — click anywhere on the map to move the pin there.")
    st.stop()
# cap clutter: always keep every entrant, fill the rest with the nearest incumbents
keep = nb_full[nb_full.is_entrant]
n_inc = max(0, max_sites - len(keep))
inc = nb_full[~nb_full.is_entrant].nsmallest(n_inc, "dist_km")
nb = pd.concat([keep, inc]).drop_duplicates("site_key").sort_values("op_start").reset_index(drop=True)
entrants = nb[nb.is_entrant]
# focal new site = the newest entrant (fallback to the nearest site to the pin)
focal_key = (entrants.sort_values("op_start").site_key.iloc[-1] if len(entrants)
             else nb.sort_values("dist_km").site_key.iloc[0])

_dom = set(nb_full.site_key)
demo_label = anon_names(site, _dom) if demo else {}                    # site_key -> "Site N" by opening order
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pin", "your pin" if demo else f"{plat:.3f}, {plon:.3f}")
c2.metric("Sites in market", len(nb_full), help=f"within {radius} km; showing {len(nb)}")
c3.metric("New entrants", int(nb_full.is_entrant.sum()))
c4.metric("Local market", f"≤{radius} km")

# ── map, full width left-to-right ──
st.subheader("Map")
if hl_client and not demo:                                # an operator is highlighted → USA-level view of its footprint
    fmap = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron", prefer_canvas=True)
else:
    fmap = folium.Map(location=[plat, plon], zoom_start=10, tiles="cartodbpositron", prefer_canvas=True)
if demo:
    # confidential demo: no site dots / no exact pin — nearby cluster regions (≤200 km) shaded by colour,
    # with a soft red blur marking the chosen local market on top
    add_cluster_regions(fmap, site, plat, plon, max_km=200)
    for rad_km, fop in [(radius, 0.08), (radius * 0.55, 0.12), (radius * 0.28, 0.18)]:
        folium.Circle([plat, plon], radius=rad_km * 1000, color="#c0392b", weight=0,
                      fill=True, fill_color="#e6194B", fill_opacity=fop).add_to(fmap)
    mp = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
    lc = (mp or {}).get("last_clicked")           # click anywhere → drop the pin right there
    if lc:
        st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
else:
    folium.Circle([plat, plon], radius=radius * 1000, color="#999", weight=1, fill=True,
                  fill_opacity=0.05).add_to(fmap)
    # rich-history sites (≥30 mo) as light dots — full footprint, all distances — so you can pan to any market
    # (thin/young sites are intentionally NOT dotted: they can't be analyzed and would just be clutter)
    add_all_site_dots(fmap, site_rich)
    for _, s in nb.iterrows():
        if s.site_key == focal_key:
            color, rad = "#e6194B", 9
        elif s.is_entrant:
            color, rad = "#f58231", 7
        else:
            color, rad = "#5b8db8", 6
        folium.CircleMarker(
            [s.lat, s.lon], radius=rad, color=color, fill=True, fill_color=color, fill_opacity=0.9, weight=2,
            tooltip=f"{s.client_name} · opened {s.op_start:%b %Y} · {s.dist_km:.1f} km"
                    + (" · NEW" if s.is_entrant else "")).add_to(fmap)
    if hl_client:                                            # mark every site of the chosen operator in a separate colour
        for _, s in site[site.has_coords & (site.client_name == hl_client)].iterrows():
            folium.CircleMarker([s.lat, s.lon], radius=6, color="#d4a500", fill=True, fill_color="#ffd000",
                                fill_opacity=0.95, weight=2, tooltip=f"{s.client_name} (operator)").add_to(fmap)
    folium.Marker([plat, plon], icon=folium.Icon(color="black", icon="star"),
                  tooltip="📍 pin").add_to(fmap)
    mp = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
    lc = (mp or {}).get("last_clicked")                          # click ANYWHERE -> drop the pin there
    if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(plat, 5), round(plon, 5)):
        st.session_state.pin = (lc["lat"], lc["lng"]); st.rerun()
    if hl_client:
        st.caption(f"🟡 yellow = every site operated by **{hl_client}**.")

# ───────────────────────── cluster-wise KPI panels ─────────────────────────
# The 6 KPIs (retail/membership wash & revenue + the two ASPs), summed across the whole
# cluster each month — the same set shown in final_modelling/six_year_app.py, but cluster-level.
st.divider()
ckeys = nb_full.site_key.tolist()
cdesc = f"this local market · {len(ckeys)} sites" if demo else f"local market ≤{radius} km · {len(ckeys)} sites"
st.subheader(f"Local-market KPIs over time — {cdesc}")
sub = df[df.site_key.isin(ckeys)].copy()
# Null corrupted revenue (feed dropped to ~0 with washes intact) so the revenue/ASP charts and the AI
# insights below aren't deflated by it. Same floor as the P&L block; washes & purchases are left intact.
_subr = (sub.ret_wash_count >= ASP_MIN_WASH) & (sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan) < ASP_FLOOR_RET)
_subm = (sub.mem_wash_count >= ASP_MIN_WASH) & (sub.mem_revenue / sub.mem_wash_count.replace(0, np.nan) < ASP_FLOOR_MEM)
sub.loc[_subr.fillna(False), "ret_revenue"] = np.nan
sub.loc[_subm.fillna(False), "mem_revenue"] = np.nan
sub["tot_revenue"] = sub[["mem_revenue", "ret_revenue"]].sum(axis=1, min_count=1)
sub["asp_ret"] = sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan)        # retail ASP = revenue ÷ retail washes
sub["asp_mem"] = sub.mem_revenue / sub.mem_purchase_count.replace(0, np.nan)    # membership ASP = revenue ÷ membership PURCHASES
# one figure per group (washes → revenue → ASP). Grounded Key Insights are shown ONCE as a card up top
# (the Analysis → Key Insights view), not repeated under each chart.
GROUPS = [
    ("Washes", [("tot_wash_count", "Total washes", "count"), ("ret_wash_count", "Retail washes", "count"), ("mem_wash_count", "Membership washes", "count")]),
    ("Revenue", [("tot_revenue", "Total revenue ($)", "$"), ("ret_revenue", "Retail revenue ($)", "$"), ("mem_revenue", "Membership revenue ($)", "$")]),
    ("ASPs", [("asp_ret", "ASP per wash — retail ($)", "$"), ("asp_mem", "ASP per membership — membership ($)", "$")]),
]
name_of = demo_label if demo else site.set_index("site_key").client_name.to_dict()
PALETTE = ["#2E86DE", "#16a085", "#8e44ad", "#e67e22", "#27ae60", "#2980b9", "#c0392b", "#d35400", "#7f8c8d",
           "#2c3e50", "#1abc9c", "#9b59b6", "#34495e", "#f39c12", "#3498db", "#e74c3c", "#95a5a6", "#0a84ff"]
order = [k for k in ckeys if k != focal_key] + ([focal_key] if focal_key in ckeys else [])   # draw the focal site LAST so it sits on top
gframes = {}                                                                   # even monthly grid per site (reused across groups)
for k in order:
    g = sub[sub.site_key == k].set_index("date").sort_index()
    gframes[k] = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS")) if len(g) else g

# ── AI Key Insights — 2-node pipeline (compute_metrics -> generate_insights), one read-out per group ──
# Computed once per market on demand (the button) and stored by market signature, so flipping a group's
# granularity radio re-renders instantly without another LLM call.
insights_backend = os.getenv("INSIGHTS_LLM_BACKEND", "azure").strip().lower()
imeta = nb_full[[c for c in ["site_key", "op_start", "dist_km", "is_entrant", "left_censored"] if c in nb_full.columns]].copy()
if "left_censored" not in imeta.columns:                                       # safety: pull from `site` if dropped
    imeta = imeta.merge(site[["site_key", "left_censored"]], on="site_key", how="left")
imeta["name"] = imeta.site_key.map(name_of)                                    # demo-safe ("Site N") names
isig = (tuple(sorted(ckeys)), str(focal_key), int(radius), bool(demo), insights_backend)


@st.cache_data(show_spinner=False, ttl=3600)
def _market_insights_cached(_sig, _sub, _meta, focal_key, backend):
    """Cached per market signature `_sig` (the big frames are underscore-prefixed so they aren't hashed)."""
    return market_insights(_sub, _meta, focal_key, backend=backend)["insights"]


# ── Summaries are prepared AUTOMATICALLY on pin/market change, then VIEWED via the dropdown ──
# No "Generate" click: when a market is chosen we eagerly compute and cache every LLM-backed summary we can
# support for that market, so switching the dropdown only renders already-prepared output.
insights_store = st.session_state.setdefault("insights_store", {})
loc_poc_store = st.session_state.setdefault("loc_poc_store", {})
pollinate_store = st.session_state.setdefault("pollinate_store", {})
compete_store = st.session_state.setdefault("compete_store", {})
loc_sig = (round(plat, 5), round(plon, 5), int(radius))


@st.cache_data(show_spinner=False, ttl=3600)
def _location_poc_cached(_sig, lat, lon, radius_km, backend):
    """Location-only LLM summary, cached per (rounded location, radius, backend)."""
    return location_market_analysis(lat, lon, radius_km=radius_km, backend=backend)


@st.cache_data(show_spinner=False, ttl=3600)
def _pollinate_cached(_sig, _qual_text, _quant, _comp, lat, lon, radius_km, backend):
    """Fusion of the summaries, cached per market signature `_sig` (text/dict args underscore-prefixed → not hashed).
    `_comp` is the competition-scale read (or None) folded in as the competitive-saturation dimension."""
    return pollinate_analysis(_qual_text, _quant, lat=lat, lon=lon, radius_km=radius_km, competition=_comp,
                              backend=backend)


@st.cache_data(show_spinner=False, ttl=3600)
def _compete_cached(_sig, lat, lon, radius_km, known_sites, backend):
    """LLM competitive-saturation estimate — client footprint vs total landscape, cached per (location, radius, known set)."""
    return competition_scale_analysis(lat, lon, known_sites=list(known_sites), radius_km=radius_km, backend=backend)


MODE_KEY = "✨ Key Insights — grounded in this market's data"
MODE_DIRECT = "🌍 Direct LLM summary — location only, no data"
MODE_POLLINATE = "🔀 Pollinated — location commentary × data insights"
MODE_COMPETE = "🏁 Competitive saturation — client footprint vs the trade area"
_modes = []
if _INSIGHTS_OK:
    _modes.append(MODE_KEY)
if (not demo) and _LOC_POC_OK:                                # Direct/Pollinated/Competition reveal the city → not in demo
    _modes.append(MODE_DIRECT)
    if _INSIGHTS_OK:
        _modes.append(MODE_POLLINATE)
    _modes.append(MODE_COMPETE)
if not _modes:
    _modes = [MODE_KEY]

# names of the car washes we actually have data for in this radius (real names — this mode reveals the city)
_known_names = tuple(sorted(name_of.get(k, str(k)) for k in ckeys)) if not demo else tuple()

# Eager precompute (cached → runs once per new market, then served instantly on every later rerun).
_llm_ready = insights_llm_ready(insights_backend)
if not _llm_ready:
    st.caption(f"⚠️ `{insights_backend}` LLM endpoint unavailable — summaries can't be prepared right now.")
else:
    if _INSIGHTS_OK and isig not in insights_store:
        try:
            with st.spinner("Preparing grounded Key Insights for this market…"):
                insights_store[isig] = _market_insights_cached(isig, sub, imeta, focal_key, insights_backend)
        except Exception as e:
            st.caption(f"_Key Insights couldn't be prepared: {e}_")
    if not demo and _LOC_POC_OK:
        if loc_sig not in loc_poc_store:
            try:
                with st.spinner("Preparing the location-only summary…"):
                    loc_poc_store[loc_sig] = _location_poc_cached(loc_sig, plat, plon, int(radius),
                                                                  insights_backend)
            except Exception as e:
                st.caption(f"_Location summary couldn't be prepared: {e}_")
        _qual = loc_poc_store.get(loc_sig)
        _quant = insights_store.get(isig)
        if _qual and _quant:
            _ckey = (loc_sig, _known_names)
            if _ckey not in compete_store:
                try:
                    with st.spinner("Sizing the competitive set near this pin…"):
                        compete_store[_ckey] = _compete_cached(_ckey, plat, plon, int(radius),
                                                               _known_names, insights_backend)
                except Exception as e:
                    st.caption(f"_Competition estimate couldn't be prepared: {e}_")
            _out_c = compete_store.get(_ckey)
            if _out_c and isig not in pollinate_store:
                try:
                    with st.spinner("Combining location commentary × grounded data × competition…"):
                        pollinate_store[isig] = _pollinate_cached(
                            isig, _qual["text"], _quant, _out_c, plat, plon, int(radius), insights_backend
                        )
                except Exception as e:
                    st.caption(f"_Pollinated summary couldn't be prepared: {e}_")

gen_mode = st.selectbox("Analysis — pick a view (all summaries prepare automatically when you choose a pin)", _modes,
                        key="analysis_mode")
group_insights = insights_store.get(isig, {})                 # the per-chart loop below renders this

if gen_mode == MODE_DIRECT:
    _out = loc_poc_store.get(loc_sig)
    if _out:
        with st.container(border=True):
            st.markdown("#### 🌍 Location-only market read")
            st.markdown(_out["text"])
            st.caption(f"From location alone via `{_out['backend']}` — no operating data was sent.")
            with st.expander("🔎 Exact prompt sent to the LLM"):
                st.code(_out["prompt"], language="text")
elif gen_mode == MODE_POLLINATE:
    _qual, _quant = loc_poc_store.get(loc_sig), insights_store.get(isig)
    if _qual and _quant:
        _ckey = (loc_sig, _known_names)
        _out_c = compete_store.get(_ckey)
        _out = pollinate_store.get(isig)
        if _out:
            with st.container(border=True):
                st.markdown("#### 🔀 Pollinated read — location × market data × competition")
                st.markdown(_out["text"])
                st.caption(f"Fuses the location commentary, the grounded data insights and the competitive-"
                           f"saturation read via `{_out['backend']}`.")
                with st.expander("🔎 The inputs + the pollination prompt"):
                    st.markdown("**(A) Location-only commentary**")
                    st.markdown(_qual["text"])
                    st.markdown("**Pollination prompt sent:**")
                    st.code(_out["prompt"], language="text")
elif gen_mode == MODE_COMPETE:
    _ckey = (loc_sig, _known_names)
    _out = compete_store.get(_ckey)
    if _out:
        d = _out.get("data") or {}
        n_known = _out["known_count"]
        exp = d.get("estimated_express_tunnels") or {}
        tot = d.get("estimated_total_carwashes") or {}
        share = d.get("estimated_client_share") or {}
        st.markdown("#### 🏁 Competitive landscape — the client's footprint vs the whole trade area")
        _c1, _c2, _c3, _c4 = st.columns(4)
        _c1.metric(f"Client's own sites (≤{int(radius)} km)", n_known,
                   help="Express car washes this operator runs in the trade area.")
        if exp:
            _c2.metric("Est. express tunnels (total)", f"{exp.get('low','?')}–{exp.get('high','?')}")
        if tot:
            _c3.metric("Est. all car washes", f"{tot.get('low','?')}–{tot.get('high','?')}")
        if share:
            _c4.metric("Client share of express", f"{share.get('low','?')}–{share.get('high','?')}%")
        # coverage multiple — the factor to scale competitive pressure by (constructive, NOT a "gap")
        _se = _out.get("scale_express")
        if _se and exp and n_known > 0:
            st.info(f"📐 **Coverage scale.** The client runs **{n_known}** of an estimated **{exp.get('low','?')}–"
                    f"{exp.get('high','?')}** express tunnels in this trade area — so the true competitive set is roughly "
                    f"**~{_se.get('low','?')}×–{_se.get('high','?')}×** the client's own footprint. Scale competitive "
                    f"pressure accordingly. Express supply density: **{d.get('saturation','—')}**.")
        elif n_known == 0:
            st.info("The client runs no sites of their own in this radius — the table below is the competitive "
                    "landscape they'd be entering.")
        # ── competitors, as a typed table (express tunnels first, then other types; sorted by threat) ──
        _comps = [c for c in (d.get("competitors") or []) if isinstance(c, dict)]
        if _comps:
            _TYPE_ORD = {"Express tunnel": 0, "In-bay automatic": 1, "Self-serve": 2, "Other": 3}

            def _norm_type(t):
                s = (t or "").strip().lower()
                if any(k in s for k in ("express", "tunnel", "conveyor")):
                    return "Express tunnel"
                if any(k in s for k in ("in-bay", "in bay", "iba", "automatic")):
                    return "In-bay automatic"
                if "self" in s:
                    return "Self-serve"
                return "Other"
            _rows = [{"Competitor": c.get("name", "?"), "Type": _norm_type(c.get("type")),
                      "Scale": c.get("scale", ""), "Threat": c.get("threat", ""), "Notes": c.get("note", "")}
                     for c in _comps]
            _cdf = pd.DataFrame(_rows)
            _cdf["_t"] = _cdf["Type"].map(_TYPE_ORD).fillna(9)
            _cdf["_th"] = _cdf["Threat"].map({"High": 0, "Medium": 1, "Low": 2}).fillna(9)
            _cdf = _cdf.sort_values(["_t", "_th"]).drop(columns=["_t", "_th"])
            _bytype = _cdf["Type"].value_counts()
            st.markdown("**Competitors by type:** " + " · ".join(f"{k} **{v}**" for k, v in _bytype.items()))
            st.dataframe(_cdf, width="stretch", hide_index=True)
        # ── richer competitive read ──
        if d.get("client_position"):
            st.markdown(f"**Client's competitive position:** {d['client_position']}")
        _g1, _g2 = st.columns(2)
        with _g1:
            if d.get("competitive_intensity"):
                st.markdown(f"**Competitive intensity:** {d['competitive_intensity']}")
            if d.get("headroom"):
                st.markdown(f"**Headroom for a new build:** {d['headroom']}")
        with _g2:
            if d.get("pricing_positioning"):
                st.markdown(f"**Pricing / positioning:** {d['pricing_positioning']}")
            if d.get("expansion_signals"):
                st.markdown(f"**Expansion signals:** {d['expansion_signals']}")
        _rk = d.get("client_sites_recognized") or []
        if _rk:
            st.caption(f"LLM recognized of the client's sites: {', '.join(map(str, _rk))}")
        st.caption(f"Confidence: **{d.get('confidence','—')}** · via `{_out['backend']}`. {d.get('reasoning','')}")
        with st.expander("🔎 Prompt + raw JSON"):
            st.code(_out["prompt"], language="text")
            st.code(_out.get("raw", ""), language="json")
elif gen_mode == MODE_KEY:
    overall_insight = group_insights.get("Washes") if group_insights else None
    if overall_insight:
        with st.container(border=True):
            st.markdown("#### ✨ Key Insights — grounded in this market's data")
            st.markdown(overall_insight)

for gi, (gname, panels) in enumerate(GROUPS):
    gk = gran_picker(f"gran_kpi_{gname}")
    gk_how = "mean" if gname == "ASPs" else "sum"                 # ASP is a per-wash rate → average; washes/$ → sum
    gfig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels], horizontal_spacing=0.06)
    for si, k in enumerate(order):
        g = gframes[k]
        is_focal = (k == focal_key)
        color = "#e6194B" if is_focal else PALETTE[si % len(PALETTE)]
        nm = (str(name_of.get(k, "?"))[:18]) + (" 🆕" if is_focal else "")
        for i, (c, lbl, unit) in enumerate(panels):
            ya = rs_dates(g[c], gk, gk_how)
            if gk == "M" and smooth and smooth > 1:
                ya = ya.rolling(smooth, center=True, min_periods=1).mean()   # smoothing slider (monthly view only)
            vfmt = "$%{y:,.2f}" if unit == "$" else "%{y:,.0f}"
            gfig.add_trace(go.Scatter(x=ya.index, y=ya.values, mode="lines", name=nm, legendgroup=k, showlegend=(gi == 0 and i == 0),
                                      line=dict(color=color, width=3 if is_focal else 1.4), opacity=1.0 if is_focal else 0.7,
                                      hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}} · {vfmt}<extra></extra>"),
                           row=1, col=i + 1)
    gfig.update_layout(height=340, template="plotly_white", margin=dict(l=8, r=8, t=44, b=10),
                       hovermode="closest", legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
    if gk == "Y":
        gfig.update_xaxes(dtick="M12", tickformat="%Y")
    else:
        gfig.update_xaxes(tickformat=gran_date_tickformat(gk))
    st.plotly_chart(gfig, width="stretch", key=f"kpi_{gname}")
    st.divider()

# ───────────────────────── Tunnel-length proxy (estimated metres) ─────────────────────────
# Proxy for tunnel LENGTH in metres, from peak monthly volume:
#   peak-month total washes ÷ 25 operating days ÷ 10 hours/day ÷ 3.2 cars/hr per metre ≈ tunnel metres.
# Toggle Operator-wise (an operator's sites here collapsed to one bar, median length) vs Site-wise.
# Horizontal bars + 10-m range bands keep it readable. Demo-safe: `name_of` ("Site N") / "Operator N".
st.subheader("Tunnel length proxy — estimated metres")
tl_group = st.radio("Group by", ["Operator", "Site"], horizontal=True, key="tl_group",
                    help="Operator-wise collapses an operator's sites in this market into one bar "
                         "(median tunnel length); Site-wise shows every site.")
st.caption("**peak-month washes ÷ 25 days ÷ 10 hours ÷ 3.2** ≈ tunnel length (m), in 10-m range bands "
           "(0–10 · 10–20 · 20–30 · 30–40 · 40m+). Further right / darker = longer; 🆕 = the new site.")
_BANDS = [("0–10m", "#c6dbef"), ("10–20m", "#9ecae1"), ("20–30m", "#6baed6"), ("30–40m", "#3182bd"), ("40m+", "#08519c")]
def _bandlabel(m):
    return _BANDS[min(int(m // 10), 4)][0]                                   # 0–9→band0 … ≥40→band4 (40m+)
def _dedupe(labels):                                                         # unique, readable y-axis names
    seen, out = {}, []
    for l in labels:
        seen[l] = seen.get(l, 0) + 1
        out.append(l if seen[l] == 1 else f"{l} ({seen[l]})")
    return out

_peak = sub.groupby("site_key")["tot_wash_count"].max()                      # peak month total washes per site
_site_m = (_peak / 25 / 10 / 3.2).replace([np.inf, -np.inf], np.nan).dropna()   # ÷25d ÷10h ÷3.2 → tunnel metres
_site_m = _site_m[_site_m.index.isin(ckeys)]
if len(_site_m):
    base = pd.DataFrame({"site_key": _site_m.index, "metres": _site_m.values})
    base["peak"] = base.site_key.map(_peak)
    base["is_focal"] = base.site_key == focal_key
    if tl_group == "Operator":                                              # collapse each operator's sites → one bar
        base["oid"] = base.site_key.str.split("::").str[0]
        agg = (base.groupby("oid")
               .agg(metres=("metres", "median"), peak=("peak", "max"), is_focal=("is_focal", "any"),
                    n=("site_key", "size"), k=("site_key", "first")).reset_index())
        if demo:                                                            # demo-safe anonymous operator labels
            agg = agg.sort_values("metres").reset_index(drop=True)
            agg["label"] = [f"Operator {i + 1}" for i in range(len(agg))]
        else:
            agg["label"] = agg["k"].map(lambda x: str(name_of.get(x, "?"))[:26])
        plot_df = agg[["label", "metres", "peak", "is_focal", "n"]].copy()
    else:
        base["label"] = base.site_key.map(lambda x: str(name_of.get(x, "?"))[:26])
        plot_df = base.assign(n=1)[["label", "metres", "peak", "is_focal", "n"]].copy()
    plot_df["label"] = _dedupe(plot_df["label"].tolist())                   # disambiguate same-name sites/operators
    plot_df.loc[plot_df.is_focal, "label"] = plot_df.loc[plot_df.is_focal, "label"] + " 🆕"
    plot_df = plot_df.sort_values("metres").reset_index(drop=True)          # ascending → longest at the TOP (h-bars)
    tlfig = go.Figure()
    for _bl, _bc in _BANDS:                                                 # one trace per band → discrete legend
        d = plot_df[plot_df.metres.map(_bandlabel) == _bl]
        if d.empty:
            continue
        tlfig.add_trace(go.Bar(
            orientation="h", y=d.label, x=d.metres, name=_bl, marker_color=_bc,
            marker_line_color=["#e6194B" if f else "rgba(0,0,0,0)" for f in d.is_focal],
            marker_line_width=[3 if f else 0 for f in d.is_focal],
            customdata=np.stack([d.peak.values, d.n.values], axis=-1),
            text=[f"{m:.0f} m" for m in d.metres], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>~%{x:.0f} m tunnel (proxy) · " + _bl +
                          "<br>peak month %{customdata[0]:,.0f} washes · %{customdata[1]:.0f} site(s)<extra></extra>"))
    tlfig.update_layout(height=max(260, 42 * len(plot_df) + 120), template="plotly_white", barmode="overlay",
                        bargap=0.35, margin=dict(l=8, r=8, t=10, b=10),
                        legend=dict(orientation="h", y=-0.18, title="range band"),
                        xaxis_title="estimated tunnel length (m)", yaxis_title=None)
    tlfig.update_yaxes(categoryorder="array", categoryarray=plot_df.label.tolist())
    tlfig.update_xaxes(dtick=10, ticksuffix="m", rangemode="tozero")
    for _xv in (10, 20, 30, 40):                                            # range-band boundary gridlines
        tlfig.add_vline(x=_xv, line_dash="dot", line_color="#cccccc", line_width=1)
    st.plotly_chart(tlfig, width="stretch", key="tunnel_length")
    _foc = plot_df[plot_df.is_focal]
    if len(_foc):
        _fr = _foc.iloc[-1]
        _extra = f", median across {int(_fr.n)} sites" if tl_group == "Operator" and _fr.n > 1 else ""
        st.caption(f"🆕 The new site ≈ **{_fr.metres:.0f} m** ({_bandlabel(_fr.metres)} band{_extra}).")
else:
    st.caption("_No wash data available for this market to estimate tunnel length._")
