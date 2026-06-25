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

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "main-ds.csv"
TYPES_CSV = HERE.parent / "data" / "site_carwash_types.csv"
ARTIFACTS = HERE.parent / "notebooks" / "artifacts"
EARTH_KM = 6371.0088
EXPRESS_TYPE = "Express Tunnel"          # the "express only" filter keeps just this primary_carwash_type
EXPRESS_MIN_MONTHS = 30                   # express mode also requires ≥30 monthly records → richer history

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


def drop_pin_ui(df, site, art, demo=False, express_only=False):
    st.title("Forecasting for a new site")
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
        # leakage-free LOO backtest (1223 sites, ≥24mo): Model 1 WAPE 46.5 · Model 2 43.6 · Model 3 40.2 (cold-start) ·
        # Model 4 ~34.1 WAPE / 27.7 medAPE when the operator is KNOWN (uses that operator's avg mature level, brand_loo).
        MODEL_STRATEGIES = {
            "Model 1": dict(local_anchor=False, model_kind="lgb", use_operator=False),
            "Model 2": dict(local_anchor=True, model_kind="lgb", use_operator=False),
            "Model 3": dict(local_anchor=True, model_kind="et", use_operator=False),
            "Model 4": dict(local_anchor=True, model_kind="et", use_operator=True),
        }
        strat = MODEL_STRATEGIES[st.radio("Model", list(MODEL_STRATEGIES), index=2,
                                          help="Plateau-level strategy (LOO WAPE): M1 46.5%, M2 43.6%, M3 40.2% "
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
    gco = site[site.has_coords & (site.cluster >= 0)]              # only clustered sites (no orange standalones)
    fmap = folium.Map(location=[lat, lon], zoom_start=10,          # same zoom as the Explore-markets map
                      tiles="cartodbpositron", control_scale=True)
    if demo:
        # confidential demo: no site dots, no exact pin — a soft shaded red region marks the chosen area
        for rad_km, fop in [(radius, 0.08), (radius * 0.55, 0.12), (radius * 0.28, 0.18)]:
            folium.Circle([lat, lon], radius=rad_km * 1000, color="#c0392b", weight=0,
                          fill=True, fill_color="#e6194B", fill_opacity=fop).add_to(fmap)
    else:
        for _, s in gco.iterrows():
            folium.CircleMarker([s.lat, s.lon], radius=3, color="#3b7dd8", fill=True, fill_color="#3b7dd8",
                                fill_opacity=0.7, weight=0,
                                tooltip=f"{s.client_name} · market #{int(s.cluster)}").add_to(fmap)
        folium.Circle([lat, lon], radius=radius * 1000, color="#c0392b", weight=2, fill=False).add_to(fmap)
        folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="star"), tooltip="📍 your new site").add_to(fmap)
    # return ONLY last_clicked → panning/zooming no longer triggers a rerun (fixes the dimming-on-zoom)
    m = st_folium(fmap, height=500, use_container_width=True, returned_objects=["last_clicked"])
    lc = (m or {}).get("last_clicked")
    if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(lat, 5), round(lon, 5)):
        st.session_state.pin = (lc["lat"], lc["lng"])
        st.rerun()
    st.caption(f"🔵 clustered sites · ⭕ {radius} km radius · ⭐ your pin. "
               "**Click to drop the pin.** For exact coordinates use the lat/lon boxes in the sidebar.")

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
        st.caption(f"🔧 **{_ml}** on, but {_none} within 20 km — same as Model 1 here.")
    else:
        st.caption(f"🔧 **Model 1:** plateau **{info['plateau_med']:,.0f}/mo**.")
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
    _mm, _rr = _rec.dropna(subset=["mem_revenue"]), _rec.dropna(subset=["ret_revenue"])
    _mp, _mw = _mm.mem_purchase_count.sum(), _mm.mem_wash_count.sum()
    cl_mem_pp = float(_mm.mem_revenue.sum() / _mp) if _mp > 0 else 30.0         # cluster $/membership PURCHASE (the new ASP)
    purch_per_wash = float(_mp / _mw) if _mw > 0 else 0.33                      # cluster membership purchases per membership wash
    cl_ret = float(_rr.ret_revenue.sum() / _rr.ret_wash_count.sum()) if _rr.ret_wash_count.sum() > 0 else 15.0
    asp_scope = f"cluster ≤{radius} km · {len(_ck)} sites" if _ck else "default (no neighbours)"
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
    with st.expander("⚙️ Cost assumptions — by year", expanded=True):
        _plan0 = pd.DataFrame({
            "Year": [f"Year {i + 1}" for i in range(5)],
            "Mem ASP ($/purchase)": [round(min(cl_mem_pp, 100.0), 1)] * 5,
            "Retail ASP ($/wash)": [round(min(cl_ret, 60.0), 1)] * 5,
            "OPEX (% of sales)": [_learned_pct(sl) for sl in _yslices],
            "CAPEX ($)": [0] * 5,
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
                "CAPEX ($)": st.column_config.NumberColumn(
                    min_value=0, step=25000, format="$%d",
                    help="Build-out / equipment $ spent that year, added into expenses and spread across its months."),
            },
        )
        mem_asp_in = {i + 1: float(plan["Mem ASP ($/purchase)"].iloc[i]) for i in range(5)}
        ret_asp_in = {i + 1: float(plan["Retail ASP ($/wash)"].iloc[i]) for i in range(5)}
        opex_in = {i + 1: float(plan["OPEX (% of sales)"].iloc[i]) for i in range(5)}
        capex_in = {i + 1: float(plan["CAPEX ($)"].iloc[i]) for i in range(5) if float(plan["CAPEX ($)"].iloc[i])}
    asp_mem_pp = per_year_to_monthly(len(months), mem_asp_in, cl_mem_pp)        # $/purchase per month (membership)
    asp_ret = per_year_to_monthly(len(months), ret_asp_in, cl_ret)             # $/wash per month (retail)
    rev_base = mem_purch * asp_mem_pp + ret * asp_ret                          # sales = mem purchases × $/purchase + retail washes × $/wash

    # ── CAMPAIGN — promo that converts retail→membership and eats neighbours' share (opex-data.csv + book_v4) ──
    st.markdown("##### 🎯 Campaign — should this site run a promotion?")
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
    # THIS SITE's predicted membership = what its 5-yr trajectory SETTLES at (plateau months 36–60): predicted
    # membership washes ÷ total washes — read straight off the trajectory plot, not the raw model parameter.
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
    st.markdown(verdict)
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

    rev_m = mem_purch * mem_mult * asp_mem_pp + ret * ret_mult * asp_ret   # campaign shifts the mix retail→membership (purchases scale with mem washes)
    # OPEX line = the user's per-year % of sales, fitted onto the learned hot→mature pattern (shape kept, level set
    # to each year's target). CAPEX = $ per year spread across its months. Expenses = OPEX + CAPEX; net = sales − both.
    opex_pct, opex_tgts = fit_opex_pct_to_targets(opex_shape, rev_base, opex_in)
    opex_base_m = opex_pct * rev_base                                  # OPEX $ (no campaign)
    opex_m = opex_base_m * opex_mult_c                                 # + the promo spend over the campaign window
    capex_m = spread_capex(len(months), capex_in)                     # CAPEX $ spread across each year
    total_exp_base = opex_base_m + capex_m
    total_exp_m = opex_m + capex_m
    net_m = rev_m - total_exp_m

    gpnl = gran_picker("gran_pnl")
    xb, rev_a = agg_months(rev_m, gpnl)
    _, opex_a = agg_months(opex_m, gpnl)
    _, capex_a = agg_months(capex_m, gpnl)
    _, totexp_a = agg_months(total_exp_m, gpnl)
    _, net_a = agg_months(net_m, gpnl)
    f2 = go.Figure()
    # expense composition: OPEX + CAPEX stacked → the top of the filled stack IS total expenses
    f2.add_trace(go.Scatter(x=xb, y=opex_a, name="OPEX", mode="lines", line=dict(color="#e67e22", width=0.5),
                            stackgroup="exp", fillcolor="rgba(230,126,34,0.55)",
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>OPEX</extra>"))
    f2.add_trace(go.Scatter(x=xb, y=capex_a, name="CAPEX", mode="lines", line=dict(color="#8e44ad", width=0.5),
                            stackgroup="exp", fillcolor="rgba(142,68,173,0.45)",
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>CAPEX</extra>"))
    f2.add_trace(go.Scatter(x=xb, y=totexp_a, name="total expenses", mode="lines", line=dict(color="#c0392b", width=2),
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>total expenses</extra>"))
    f2.add_trace(go.Scatter(x=xb, y=rev_a, name="revenue (sales)", mode="lines", line=dict(color="#16a085", width=3),
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>revenue</extra>"))
    f2.add_trace(go.Scatter(x=xb, y=net_a, name="net income", mode="lines", line=dict(color="#0a84ff", width=2.6),
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>net</extra>"))
    f2.add_hline(y=0, line=dict(color="#9aa6b2", width=1, dash="dot"))
    if camp_on:
        _, revb_a = agg_months(rev_base, gpnl)
        f2.add_trace(go.Scatter(x=xb, y=revb_a, name="revenue (no campaign)", mode="lines",
                                line=dict(color="#9aa6b2", width=1.4, dash="dot"), hovertemplate="m%{x} · $%{y:,.0f}<extra>no campaign</extra>"))
        f2.add_vrect(x0=c_launch, x1=min(60, c_launch + WIN), fillcolor="rgba(230,25,75,0.08)", line_width=0,
                     annotation_text="campaign", annotation_position="top left")
        f2.add_vline(x=c_launch, line=dict(color="#c0392b", dash="dash", width=1.2))
    f2.update_layout(title="P&L — revenue vs expenses (OPEX + CAPEX) · net income, 5-year", height=440,
                     template="plotly_white", hovermode="x unified",
                     yaxis_title=f"$ / {GRAN_UNIT[gpnl]}", margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=-0.24))
    gran_xaxes_months(f2, gpnl, xb, noun="opening")
    st.plotly_chart(f2, width="stretch")
    _cum = np.cumsum(np.nan_to_num(net_m)); _be = int(np.argmax(_cum > 0)) if (_cum > 0).any() else None
    st.caption(f"5-yr totals — sales **${np.nansum(rev_base):,.0f}** · OPEX **${np.nansum(opex_base_m):,.0f}** · "
               f"CAPEX **${np.nansum(capex_m):,.0f}** · net **${np.nansum(rev_base - total_exp_base):,.0f}**"
               + (f" · breakeven ~month **{_be}**." if _be is not None else " · no breakeven within 5 yrs.")
               + "  OPEX %/yr (fitted): " + ", ".join(f"Y{y}={t * 100:.0f}%" for y, t in opex_tgts.items()) + ".")

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
MODES = ["🗺️ Explore markets", "📍 Drop-a-pin forecast", "🛰️ Site analysis (visual) · beta"]
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
sub["asp_ret"] = sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan)        # retail ASP = revenue ÷ retail washes
sub["asp_mem"] = sub.mem_revenue / sub.mem_purchase_count.replace(0, np.nan)    # membership ASP = revenue ÷ membership PURCHASES
# one figure per group (washes → revenue → ASP); each followed by a "Key insights" note
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
    st.markdown(f"**Key insights — {gname}**")
    st.caption("_Work in progress._")
    st.divider()
