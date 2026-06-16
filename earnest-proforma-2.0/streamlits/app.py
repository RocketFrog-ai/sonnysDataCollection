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

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "main-ds.csv"
ARTIFACTS = HERE.parent / "notebooks" / "artifacts"
EARTH_KM = 6371.0088

METRICS = {
    "Total washes": "tot_wash_count",
    "Membership washes": "mem_wash_count",
    "Retail washes": "ret_wash_count",
    "Total revenue ($)": "tot_revenue",
    "Membership share of washes": "mem_share_wash",
}

# ───────────────────────────── data ─────────────────────────────
@st.cache_data(show_spinner="Loading & clustering sites…")
def load_data():
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

    site = (
        df.groupby("site_key")
        .agg(client_name=("client_name", "first"), lat=("lat", "first"), lon=("lon", "first"),
             state=("state", "first"), region=("region", "first"), op_start=("op_start", "first"),
             first_obs=("date", "min"), last_obs=("date", "max"), n_obs=("date", "size"))
        .reset_index()
    )
    site["left_censored"] = site.op_start <= pd.Timestamp("2020-01-01")
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)

    # density-aware "local market" clustering (adaptive 10/20km — won the bake-off vs fixed 20km; see coldstart_forecast.ipynb)
    site["cluster"] = cm.assign_clusters(site, "adaptive")
    return df, site


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def neighbourhood(site, pin_key, radius_km):
    """Sites within `radius_km` of the pin (incl. the pin). Tags roles relative to the market."""
    p = site.loc[site.site_key == pin_key].iloc[0]
    d = haversine_km(p.lat, p.lon, site.lat.values, site.lon.values)
    nb = site.loc[(d <= radius_km) & site.has_coords].copy()
    nb["dist_km"] = d[(d <= radius_km) & site.has_coords.values]
    nb = nb.sort_values("op_start")
    earliest = nb.op_start.min()
    # an "entrant" opened after the market's earliest site AND is a genuinely-observed opening
    nb["is_entrant"] = (~nb.left_censored) & (nb.op_start > earliest)
    nb["is_pin"] = nb.site_key == pin_key
    return nb.reset_index(drop=True)


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
        nbf = neighbourhood(_site, k, 20)
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


# ───────────────────────────── figure ─────────────────────────────
INCUMBENT_COLORS = ["#5b8db8", "#7fb27f", "#9e86c9", "#69a7a0", "#b0926a", "#8a9bb0"]


def anon_names(site_df, keys):
    """site_key -> 'Site N' ordered by opening date (earliest = Site 1) — for the anonymized client demo."""
    sub = site_df[site_df.site_key.isin(list(keys))].sort_values("op_start")
    return {k: f"Site {i + 1}" for i, k in enumerate(sub.site_key)}


def anon_operators(art):
    """client_id -> 'Operator N' (stable order) — hides operator/brand identity in the client demo."""
    return {c: f"Operator {i + 1}" for i, c in enumerate(sorted(art["brand_mean"].keys()))}


def build_figure(df, nb, metric, metric_label, focal_key, x_mode, smooth, label_of=None):
    fig = go.Figure()
    inc_i = 0
    series = {}
    for _, s in nb.iterrows():
        ts = df.loc[df.site_key == s.site_key].set_index("date")[metric].sort_index()
        ts = ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="MS"))
        if smooth and smooth > 1:
            ts = ts.rolling(smooth, center=True, min_periods=1).mean()
        if x_mode == "Months since each site opened":
            x = (ts.index.year - s.op_start.year) * 12 + (ts.index.month - s.op_start.month)
        else:
            x = ts.index
        series[s.site_key] = (x, ts.values, s)

    # draw incumbents first (muted), then entrants, then focal on top
    def role(s):
        if s.site_key == focal_key: return "focal"
        if s.is_entrant: return "entrant"
        return "incumbent"
    order = sorted(nb.itertuples(), key=lambda s: {"incumbent": 0, "entrant": 1, "focal": 2}[role(s)])
    for s in order:
        x, y, meta = series[s.site_key]
        hovname = label_of.get(meta.site_key, "Site") if label_of else meta.client_name
        nm = hovname if label_of else f"{meta.client_name[:22]} ({meta.op_start:%b %Y})"
        r = role(s)
        if r == "focal":
            color, width, op, dash = "#e6194B", 4.0, 1.0, None
            nm = "🆕 " + nm + "  ← NEW SITE"
        elif r == "entrant":
            color, width, op, dash = "#f58231", 2.6, 0.95, None
            nm = "new: " + nm
        else:
            color, width, op, dash = INCUMBENT_COLORS[inc_i % len(INCUMBENT_COLORS)], 1.8, 0.55, None
            inc_i += 1
        fig.add_trace(go.Scatter(
            x=list(x), y=list(y), name=nm, mode="lines",
            line=dict(color=color, width=width, dash=dash), opacity=op,
            hovertemplate=f"<b>{hovname}</b><br>{meta.dist_km:.1f} km from pin<br>"
                          f"%{{x}}<br>{metric_label}: %{{y:,.0f}}<extra></extra>"))

    # entry markers
    if x_mode == "Calendar date":
        for _, s in nb[nb.is_entrant].iterrows():
            c = "#e6194B" if s.site_key == focal_key else "#f58231"
            fig.add_vline(x=s.op_start, line=dict(color=c, width=1.5, dash="dash"), opacity=0.6)
        fig.add_annotation(x=nb.loc[nb.site_key == focal_key, "op_start"].iloc[0], yref="paper", y=1.02,
                           text="NEW site opens", showarrow=False, font=dict(color="#e6194B", size=11))
        xtitle = "date"
    else:
        fig.add_vline(x=0, line=dict(color="#888", width=1.5, dash="dash"))
        xtitle = "months since each site opened"
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
                      xaxis_title=xtitle, yaxis_title=metric_label, template="plotly_white")
    return fig


@st.cache_resource(show_spinner="Loading cold-start model…")
def get_model():
    return cm.load()


PNL = HERE.parent / "data" / "pnl_operational.xlsx"


PNL_EXCLUDE = {"alpinecarwash_000087"}   # sites kept OUT of the P&L analysis (per user) — Alpine Wash, site 1


@st.cache_data(show_spinner="Loading P&L…")
def load_pnl():
    """Per-location-year P&L from pnl_operational.xlsx: SUM the sub-monthly report rows into an annual
    operating expense / income per site, keep near-full years only. Returns one row per (location, state, year)."""
    p = pd.read_excel(PNL)
    p = p[~p.subdomain.astype(str).isin(PNL_EXCLUDE)]      # drop excluded sites before any aggregation
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
    p = pd.read_excel(PNL)
    p = p[~p.subdomain.astype(str).isin(PNL_EXCLUDE)]
    m = (p.groupby(["location_name", "state", "year", "month"])
         .agg(opex=("total_expenses", "sum"), mem_wash=("mem_wash_count", "first"), ret_wash=("ret_wash_count", "first"),
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


@st.cache_data(show_spinner="Loading site opex…")
def pnl_opex_by_site():
    """Monthly operating expense keyed by main-ds site_key (subdomain::site_id) — for the anchor-journey view."""
    p = pd.read_excel(PNL)
    p = p[~p.subdomain.astype(str).isin(PNL_EXCLUDE)]
    p["site_key"] = p.subdomain.astype(str) + "::" + p.site_id.astype(str)
    g = p.groupby(["site_key", "year", "month"]).agg(opex=("total_expenses", "sum")).reset_index()
    g["date"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
    return g[["site_key", "date", "opex"]]


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


def drop_pin_ui(df, site, art, demo=False):
    st.title("📍 Drop-a-pin forecast — expected 5-year car-wash counts")
    st.caption("Click a spot on the map (or type coordinates), pick the operator, and get the **new site's** expected "
               "monthly washes for 5 years + the **impact on existing neighbours**. The ramp shape is well-learned; the "
               "plateau level is the uncertain part (~28% median error with a known operator) — bands and the override reflect that.")
    name_of = art["sites_rl"].drop_duplicates("client_id").set_index("client_id").client_name.to_dict()
    op_anon = anon_operators(art) if demo else {}
    if "drop" not in st.session_state:
        r = art["sites_rl"].sample(1, random_state=1).iloc[0]
        st.session_state.drop = (float(r.lat), float(r.lon))
    with st.sidebar:
        clients = sorted(art["brand_mean"].keys())
        op = st.selectbox("Operator / brand", ["(unknown / new operator)"] + clients,
                          format_func=lambda c: c if c.startswith("(")
                          else (f"{op_anon[c]} ({int(art['brand_n'].get(c, 0))} sites)" if demo
                                else f"{str(name_of.get(c, c))[:24]} ({int(art['brand_n'].get(c, 0))} sites)"))
        brand = None if op.startswith("(") else op
        lat = st.number_input("Latitude", value=float(st.session_state.drop[0]), format="%.5f")
        lon = st.number_input("Longitude", value=float(st.session_state.drop[1]), format="%.5f")
        if (round(lat, 5), round(lon, 5)) != (round(st.session_state.drop[0], 5), round(st.session_state.drop[1], 5)):
            st.session_state.drop = (lat, lon); st.rerun()
        ov = st.number_input("Plateau override — total washes/mo (0 = use model)", min_value=0, value=0, step=500)
        gm = st.slider("Yr 3–5 membership — extra on top of per-site trend (%/yr)", -15, 25, 0)
        gr = st.slider("Yr 3–5 retail — extra on top of per-site trend (%/yr)", -20, 15, 0)
    lat, lon = st.session_state.drop
    # Learn this LOCAL MARKET's trends PER COMPONENT (membership vs retail behave very differently, and differ by
    # cluster) from the neighbours' own series — the new site tracks them after maturity and the market forecast uses
    # them. Data-driven (robust Theil-Sen slope), not a single blended filter.
    _nb = site[site.has_coords]
    _d = haversine_km(lat, lon, _nb.lat.values, _nb.lon.values)
    _keys = _nb.site_key[(_d <= 20) & (_d > 1e-6)].tolist()
    if _keys:
        _sub = df[df.site_key.isin(_keys)]
        _pm = _sub.pivot_table(index="date", columns="site_key", values="mem_wash_count")
        _pr = _sub.pivot_table(index="date", columns="site_key", values="ret_wash_count")
        mem_g, mem_lo, mem_hi = market_trend(_pm)               # per-site median membership trend + CI band (composition-robust)
        ret_g, ret_lo, ret_hi = market_trend(_pr)               # per-site median retail trend + CI band
    else:
        mem_g = mem_lo = mem_hi = ret_g = ret_lo = ret_hi = 0.0
    traj, info = cm.predict_site(lat, lon, brand=brand, plateau_override=(ov or None),
                                 annual_mem_growth=mem_g + gm / 100, annual_ret_change=ret_g + gr / 100,
                                 mem_growth_band=(mem_lo + gm / 100, mem_hi + gm / 100),
                                 ret_change_band=(ret_lo + gr / 100, ret_hi + gr / 100), art=art)
    g = traj.set_index("month")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plateau (total/mo)", f"{info['plateau_med']:,.0f}")
    c2.metric("P10–P90 range", f"{info['plateau_lo']:,.0f}–{info['plateau_hi']:,.0f}")
    c3.metric("Mem-share @ maturity", f"{info['mem_share']:.0%}")
    c4.metric("Neighbours ≤20 km", info["n_neighbours_20km"])
    if not info["brand_known"]:
        st.info("No operator selected → the plateau uses only the local-market / region prior (wider band). "
                "Pick an operator for a sharper estimate (operator is the strongest predictor of site scale).")
    # ── big, full-width interactive map: pan/zoom is smooth (no rerun) → click to drop the pin ──
    st.subheader("📍 Pick a location — pan & zoom freely, then click to drop the pin")
    gco = site[site.has_coords]
    if "drop_zoom" not in st.session_state:
        st.session_state.drop_zoom = 4
    fmap = folium.Map(location=[lat, lon], zoom_start=st.session_state.drop_zoom,
                      tiles="cartodbpositron", control_scale=True)
    if demo:
        # confidential demo: no site dots, no exact pin — a soft shaded red region marks the chosen area
        for rad_km, fop in [(20, 0.08), (11, 0.12), (5, 0.18)]:
            folium.Circle([lat, lon], radius=rad_km * 1000, color="#c0392b", weight=0,
                          fill=True, fill_color="#e6194B", fill_opacity=fop).add_to(fmap)
    else:
        for _, s in gco.iterrows():
            standalone = s.cluster < 0
            col = "#e67e22" if standalone else "#3b7dd8"
            folium.CircleMarker([s.lat, s.lon], radius=3, color=col, fill=True, fill_color=col, fill_opacity=0.7, weight=0,
                                tooltip=f"{s.client_name}" + (" · standalone (out of cluster)" if standalone else f" · market #{int(s.cluster)}")).add_to(fmap)
        folium.Circle([lat, lon], radius=20000, color="#c0392b", weight=2, fill=False).add_to(fmap)
        folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="star"), tooltip="📍 your new site").add_to(fmap)
    # return ONLY last_clicked → panning/zooming no longer triggers a rerun (fixes the dimming-on-zoom)
    m = st_folium(fmap, height=600, use_container_width=True, returned_objects=["last_clicked"])
    lc = (m or {}).get("last_clicked")
    if lc and (round(lc["lat"], 5), round(lc["lng"], 5)) != (round(lat, 5), round(lon, 5)):
        st.session_state.drop = (lc["lat"], lc["lng"])
        st.session_state.drop_zoom = max(st.session_state.drop_zoom, 10)   # zoom in around the placed pin
        st.rerun()
    st.caption("🔵 in-market · 🟠 standalone (out of cluster) · ⭕ 20 km radius · ⭐ your pin. "
               "**Pan/zoom is smooth now (no reload) — click to drop the pin.** For exact coordinates use the lat/lon boxes in the sidebar.")

    st.subheader("Predicted 5-year trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(g.index) + list(g.index[::-1]), y=list(g.total_hi) + list(g.total_lo[::-1]),
                             fill="toself", fillcolor="rgba(41,128,185,0.15)", line=dict(width=0), name="P10–P90", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.index, y=g.total_med, line=dict(color="#2980b9", width=3), name="total"))
    fig.add_trace(go.Scatter(x=g.index, y=g.mem_med, line=dict(color="#16a085", width=2), name="membership"))
    fig.add_trace(go.Scatter(x=g.index, y=g.ret_med, line=dict(color="#c0392b", width=2), name="retail"))
    fig.update_layout(height=400, xaxis_title="months since open", yaxis_title="washes / month", hovermode="x unified",
                      template="plotly_white", margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    yr = [g.loc[m, "total_med"] for m in [12, 36, 60] if m in g.index]
    st.caption(f"Expected total washes/mo — year 1 ≈ {yr[0]:,.0f}, year 3 ≈ {yr[1]:,.0f}, year 5 ≈ {yr[2]:,.0f}. "
               f"**Ramps over ~2–3 yr, then plateaus** (ramp from **{info['ramp_source']}**). After maturity it drifts at "
               f"this market's trend — membership ({mem_g:+.0%}/yr), retail ({ret_g:+.0%}/yr), the **median of each "
               f"neighbour's own slope** (robust to sites entering/leaving the average) — but that drift **saturates over "
               f"~2 yr**, so even a booming market ramps then levels off instead of compounding forever. The **P10–P90 band "
               f"fans out** with the trend's Theil-Sen CI on top of the plateau range — a noisy market widens itself; "
               f"there is **no rate clamp**. Sliders adjust on top.")
    st.divider(); st.subheader("📈 Total local-market wash count — history + 5-year forecast")
    today = pd.Timestamp(df.date.max())
    H = 60
    fdates = pd.date_range(today + pd.DateOffset(months=1), periods=H, freq="MS")
    _tj = traj.set_index("month")
    new_traj = _tj["total_med"].reindex(range(H)).to_numpy()                           # the new entrant's own journey
    new_lo = _tj["total_lo"].reindex(range(H)).to_numpy(); new_hi = _tj["total_hi"].reindex(range(H)).to_numpy()
    nbk = site[site.has_coords].copy(); nbk["d"] = haversine_km(lat, lon, nbk.lat.values, nbk.lon.values)
    nbk = nbk[(nbk.d <= 20) & (nbk.d > 1e-6)]
    fig = go.Figure()
    if len(nbk):
        keys = nbk.site_key.tolist()
        comp = df[df.site_key.isin(keys)].groupby("date")[["mem_wash_count", "ret_wash_count"]].sum()
        idx = pd.date_range(comp.index.min(), today, freq="MS")
        hist_mem = comp["mem_wash_count"].reindex(idx); hist_ret = comp["ret_wash_count"].reindex(idx)
        hist = hist_mem.add(hist_ret, fill_value=0)                                   # total = membership + retail
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
        last = float(hist.dropna().iloc[-1])
        MKT = "#0a84ff"    # bright blue — one colour for the market total: solid history -> dotted forecast
        fig.add_trace(go.Scatter(x=[today] + list(fdates) + list(fdates[::-1]) + [today],
                                 y=[last] + list(with_hi) + list(with_lo[::-1]) + [last],
                                 fill="toself", fillcolor="rgba(10,132,255,0.12)", line=dict(width=0),
                                 name="forecast band (trend CI)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, line=dict(color=MKT, width=3.2), name="market total — actual history"))
        fig.add_trace(go.Scatter(x=[today] + list(fdates), y=[last] + list(with_fc), line=dict(color=MKT, width=3.2, dash="dot"), name="market total — forecast (with new site)"))
        fig.add_trace(go.Scatter(x=[today] + list(fdates), y=[last] + list(base_fc), line=dict(color="#9aa6b2", width=1.6, dash="dot"), name="market without the new site"))
        fig.add_trace(go.Scatter(x=fdates, y=new_traj, line=dict(color="#ff375f", width=3), name="🆕 new entrant — its own journey"))
        fig.add_vline(x=today, line=dict(color="#c0392b", dash="dash", width=1.5))
        fig.add_annotation(x=today, yref="paper", y=1.03, text="new site opens", showarrow=False, font=dict(color="#c0392b", size=11))
        fig.update_layout(height=480, template="plotly_white", hovermode="x unified", xaxis_title="date",
                          yaxis_title="market total washes / month", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.28))
        st.plotly_chart(fig, width="stretch")
        net = float(with_fc[-1] - base_fc[-1])
        _clo, _chi = float(with_lo[-1]), float(with_hi[-1])
        st.caption(f"**Blue** = total market washes (Σ of all ≤20 km sites): **solid = actual history → dotted = 5-yr forecast**, "
                   f"with the **shaded band = trend confidence interval** (Theil-Sen slope CI — wider when this market's "
                   f"history is noisier; **not a clamp**). The forecast carries each existing site forward at the market's "
                   f"trend — membership ({mem_g:+.0%}/yr) and retail ({ret_g:+.0%}/yr), the median of each site's own slope — "
                   f"**saturating over ~2 yr** (no unbounded boom), then adds the entrant's ramp and subtracts cannibalization "
                   f"({'fallback' if cp.get('fallback') else 'data-fit'} {cp['a']*100:.0f}%·e^(−d/{cp['L']:.1f} km)). "
                   f"**Faint grey dotted** = market without the new site · **Red** = the entrant's own journey. "
                   f"Net market change at year 5 ≈ **{net:+,.0f}** washes/mo (band **{_clo:,.0f}–{_chi:,.0f}**).")
    else:
        fig.add_trace(go.Scatter(x=fdates, y=new_traj, line=dict(color="#e6194B", width=3), name="🆕 new site"))
        fig.update_layout(height=420, template="plotly_white", xaxis_title="date", yaxis_title="washes / month",
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.info("No existing sites within 20 km — a fresh market, so the chart shows only the new site's own 5-year journey.")

    # ───────── P&L: expected revenue (forecast × ASP) vs regional operating expense ─────────
    st.divider(); st.subheader("💰 P&L — expected revenue vs operating expense")
    pnl = load_pnl()
    yo, scope = regional_opex(pnl, art, info.get("state"), info.get("region"))
    # CLUSTER ASP from the dense operational data (main-ds): the ≤20 km neighbours' last 12 months — a real cluster figure
    _nbk = site[site.has_coords].copy()
    _nbk["d"] = haversine_km(lat, lon, _nbk.lat.values, _nbk.lon.values)
    _ck = _nbk.site_key[(_nbk.d <= 20) & (_nbk.d > 1e-6)].tolist()
    _rec = df[df.site_key.isin(_ck)].sort_values("date").groupby("site_key").tail(12) if _ck else df.iloc[:0]
    _mm, _rr = _rec.dropna(subset=["mem_revenue"]), _rec.dropna(subset=["ret_revenue"])
    cl_mem = float(_mm.mem_revenue.sum() / _mm.mem_wash_count.sum()) if _mm.mem_wash_count.sum() > 0 else 10.0
    cl_ret = float(_rr.ret_revenue.sum() / _rr.ret_wash_count.sum()) if _rr.ret_wash_count.sum() > 0 else 15.0
    asp_scope = f"cluster ≤20 km · {len(_ck)} sites" if _ck else "default (no neighbours)"
    s_mix = float(info.get("mem_share", 0.6))                                   # this site's membership share of washes
    asp_blend = s_mix * cl_mem + (1 - s_mix) * cl_ret                           # cluster average BLENDED $/wash
    asp = st.slider(f"ASP ($/wash) — cluster avg ${asp_blend:.1f}", 1.0, 30.0, round(asp_blend, 1), 0.1,
                    help=f"Blended average selling price per wash. Defaults to the {asp_scope} average (${asp_blend:.1f}/wash).")
    k_asp = asp / max(asp_blend, 1e-9)                                          # one slider scales price; keep the data's mem/retail split
    asp_mem, asp_ret = cl_mem * k_asp, cl_ret * k_asp
    st.caption(f"Slider starts at the **{asp_scope}** average (**${asp_blend:.1f}/wash**, from the operational data). "
               f"Membership/retail split (${cl_mem:.1f} / ${cl_ret:.1f}) is kept from the data and scaled together.")
    tj = traj.set_index("month")
    months = np.arange(0, 61)
    mem = tj["mem_med"].reindex(months).fillna(0.0).to_numpy()
    ret = tj["ret_med"].reindex(months).fillna(0.0).to_numpy()
    # opex = LEARNED new-site ramp (shape, REGION-scoped) × this site's mature level (mature $/wash for the SCOPE × plateau)
    pm_m = load_pnl_monthly()
    ramp_o, ramp_scope, ramp_hage = opex_ramp(pm_m, art, info.get("state"), info.get("region"))
    opw_mat, opw_scope = opex_per_wash(pm_m, art, lat, lon, info.get("state"), info.get("region"))
    mature_opex = opw_mat * float(info["plateau_med"])                            # this site's settled monthly opex $
    # BEYOND the ~33-mo P&L horizon: don't flat-line — drive opex with the forecast wash volume (opex ≈ $/wash × washes),
    # so years 3–5 track the volume forecast's secular drift instead of a frozen line.
    wt = mem + ret
    H = min(int(ramp_hage), 60)
    base = wt[H] if (H < len(wt) and wt[H] > 0) else float(np.nanmedian(wt[max(0, H - 3):H + 1]) or 0)
    if base > 0:
        for t in range(H + 1, 61):
            ramp_o[t] = ramp_o[H] * (wt[t] / base)                                # opex follows forecast volume past the data
        ramp_o = np.clip(ramp_o, 0.3, 3.5)
    g_hist = opex_trend_hist(pnl, art, info.get("state"), info.get("region"))     # past opex YoY (noisy → shown, not defaulted)
    og = st.slider("Opex cost growth (%/yr)", -10, 15, 0,
                   help=f"Escalates opex over the 5 years (on top of the learned new-site ramp). Past P&L opex trend ≈ "
                        f"{g_hist*100:+.0f}%/yr but it's noisy & likely a reporting artifact, so the default is flat — set ~+3% for inflation.")

    # ── optional promotion window (data-grounded: an ASP dip lifts volume ~0.64×, persisting ~3 mo) ──
    with st.expander("🎟️ Promotion window — discount ASP for a period (optional)", expanded=False):
        promo_on = st.checkbox("Run a promotion", value=False)
        q1, q2, q3, q4 = st.columns(4)
        p_start = q1.slider("Start (month after opening)", 1, 56, 13, disabled=not promo_on)
        p_dur = q2.slider("Duration (months)", 1, 12, 3, disabled=not promo_on)
        p_cut = q3.slider("ASP discount (%)", 0, 50, 15, disabled=not promo_on)
        p_opex = q4.slider("Extra promo opex (% of opex)", 0, 60, 10, disabled=not promo_on,
                           help="Marketing/labour cost of the promo. The data didn't show an automatic opex bump, so set it here.")
    ELAST = 0.64    # learned: a ~13% ASP dip drove ~+8.6% volume over the following quarter
    asp_mult = np.ones(61); vol_mult = np.ones(61); opex_x = np.zeros(61); end = -1
    if promo_on and p_cut > 0:
        end = min(60, p_start + p_dur - 1)
        lift = ELAST * (p_cut / 100.0)
        for t in range(61):
            if p_start <= t <= end:
                asp_mult[t] = 1 - p_cut / 100.0; vol_mult[t] = 1 + lift; opex_x[t] = p_opex / 100.0
            elif t > end:
                vol_mult[t] = 1 + lift * 0.5 ** ((t - end) / 3.0)      # lift persists, ~3-month half-life

    rev_m = mem * vol_mult * asp_mem * asp_mult + ret * vol_mult * asp_ret * asp_mult
    rev_base = mem * asp_mem + ret * asp_ret
    opex_grow = (1 + og / 100.0) ** (months / 12.0)                    # cost escalation (slider) over the horizon
    opex_m = mature_opex * ramp_o[:61] * (1 + opex_x) * opex_grow      # LEARNED ramp (hot early) × promo cost × growth
    net_m = rev_m - opex_m
    ys = lambda a, k: float(np.asarray(a)[(k - 1) * 12 + 1: k * 12 + 1].sum())

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=months, y=rev_m, name="revenue", mode="lines", line=dict(color="#16a085", width=3),
                            fill="tozeroy", fillcolor="rgba(22,160,133,0.12)", hovertemplate="m%{x} · $%{y:,.0f}<extra>revenue</extra>"))
    f2.add_trace(go.Scatter(x=months, y=opex_m, name="operating expense", mode="lines", line=dict(color="#e67e22", width=2.5),
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>opex</extra>"))
    f2.add_trace(go.Scatter(x=months, y=net_m, name="net operating income", mode="lines", line=dict(color="#0a84ff", width=2, dash="dot"),
                            hovertemplate="m%{x} · $%{y:,.0f}<extra>net</extra>"))
    if promo_on and end >= 0:
        f2.add_trace(go.Scatter(x=months, y=rev_base, name="revenue (no promo)", mode="lines",
                                line=dict(color="#9aa6b2", width=1.4, dash="dot"), hovertemplate="m%{x} · $%{y:,.0f}<extra>no promo</extra>"))
        f2.add_vrect(x0=p_start, x1=end, fillcolor="rgba(230,25,75,0.10)", line_width=0,
                     annotation_text="promo", annotation_position="top left")
    f2.update_layout(title="Monthly P&L — revenue vs operating expense (5-year forecast)", height=380,
                     template="plotly_white", hovermode="x unified", xaxis_title="months since opening",
                     yaxis_title="$ per month", margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(f2, width="stretch")
    pn = f" · the promo changes 5-yr revenue by **${float(np.sum(rev_m) - np.sum(rev_base)):+,.0f}**" if (promo_on and end >= 0) else ""
    st.caption(f"**Revenue** = predicted washes × your ASP. **Operating expense = a LEARNED new-site ramp ({ramp_scope})** — "
               f"opex runs hot early (~{ramp_o[1]:.1f}× at opening: setup/marketing) and settles to its mature level "
               f"(${mature_opex:,.0f}/mo = ${opw_mat:.2f}/wash [{opw_scope}] × plateau){f', escalated {og:+d}%/yr' if og else ''} by ~year 1, "
               f"from the P&L. The ramp is learned by **months since the site's first P&L row**, scoped to **{ramp_scope}**. "
               f"The P&L only spans ~33 months (learned through month **{ramp_hage}**); **beyond that, opex is forecast from the "
               f"projected wash volume** (opex ≈ $/wash × forecast washes), not flat-lined — add the growth slider for inflation. "
               f"**Net** = revenue − opex (a new site loses money early, as expected). "
               f"Year-1 net ≈ **${ys(net_m,1):,.0f}**, year-5 net ≈ **${ys(net_m,5):,.0f}**.{pn} "
               f"Promo effect is **from the data** (a ~13% ASP dip drove ~+9% volume that quarter, persisting ~3 mo; elasticity ≈ {ELAST}); "
               f"the opex bump is your input (the data didn't show an automatic one).")

    # historical context — region/state operating expense, year over year (source of the $/wash rate)
    f1 = go.Figure()
    f1.add_trace(go.Bar(x=yo.year.astype(int).astype(str), y=yo.opex, marker_color="#e67e22",
                        text=[f"${v/1e3:,.0f}k" for v in yo.opex], textposition="outside",
                        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
    f1.update_layout(title=f"Context · avg operating expense / site — {scope} (history)", height=300,
                     template="plotly_white", yaxis_title="$ per year", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(f1, width="stretch")
    st.caption(f"Historical **annual operating expense per site** in **{scope}** ({int(yo.n.max()) if len(yo) else 0} sites), "
               f"year over year — from `pnl_operational.xlsx`.")


def anchor_journey_ui(df, site, demo=False):
    st.title("⚓ Anchor journey — how a site captured its market")
    st.caption("Clusters that have P&L (opex) data. The **anchor ★** = the cluster's biggest site by total washes. "
               "Watch how it ramped, priced (the discount-then-raise play), spent on opex, and took share from neighbours.")
    opex = pnl_opex_by_site(); pnl_keys = set(opex.site_key.unique())
    tot = df.groupby("site_key").tot_wash_count.sum()
    cl = site[site.cluster >= 0].copy(); cl["tw"] = cl.site_key.map(tot).fillna(0)
    cl["client_id"] = cl.site_key.str.split("::").str[0]
    min3 = []
    for c, d in cl.groupby("cluster"):
        if len(d) < 2:
            continue
        npnl = len(set(d.site_key) & pnl_keys)
        if npnl < 3:                                        # keep only clusters with >=3 sites that have opex + metrics
            continue
        d = d.sort_values("tw", ascending=False); ak = d.iloc[0].site_key
        share = float(d.iloc[0].tw / d.tw.sum()) if d.tw.sum() > 0 else 0.0
        # geographic label for the cluster: dominant census region + the state(s) it spans
        regs = d.region.dropna().astype(str); sts = sorted(d.state.dropna().astype(str).unique())
        reg = regs.mode().iloc[0] if len(regs) else "—"
        slbl = ", ".join(sts[:3]) + ("…" if len(sts) > 3 else "")
        geo = f"{reg} · {slbl}" if slbl else reg
        min3.append((int(c), ak, len(d), share, npnl, int(d.client_id.nunique()), geo))
    cands = sorted(min3, key=lambda t: (-t[4], -t[3]))      # most P&L sites first, then anchor dominance
    if not cands:
        st.info("No cluster has ≥3 sites with opex + metrics."); return

    def _label(t):
        c, ak, n, sh, npnl, nop, geo = t
        who = f"market {c}" if demo else f"#{c} · {str(site.loc[site.site_key==ak,'client_name'].iloc[0])[:22]}"
        return f"{who} · {geo} · {n} sites · anchor {sh:.0%} share · {npnl}/{n} w/ P&L · {nop} operator(s)"
    sel = st.selectbox(f"Choose a cluster — {len(cands)} available (most P&L sites first)", cands, format_func=_label)
    c, anchor, n, sh, npnl, nop, geo = sel
    ck = site[site.cluster == c].sort_values("op_start"); keys = ck.site_key.tolist()
    lab = anon_names(site, keys) if demo else {k: str(site.loc[site.site_key == k, "client_name"].iloc[0])[:18] for k in keys}
    smooth = st.select_slider("Smoothing (months)", [1, 2, 3, 6], value=1)
    an = ck.loc[ck.site_key == anchor].iloc[0]
    mcol = st.columns(4)
    mcol[0].metric("Anchor", lab[anchor] + " ★")
    mcol[1].metric("Anchor market share", f"{sh:.0%}")
    mcol[2].metric("Opened", an.op_start.strftime("%b %Y") if pd.notna(an.op_start) else "—")
    mcol[3].metric("Region", geo)

    def _sm(s):
        return s.rolling(smooth, center=True, min_periods=1).mean() if smooth and smooth > 1 else s
    PAL = ["#2E86DE", "#8e44ad", "#27ae60", "#2980b9", "#7f8c8d", "#9b59b6", "#34495e", "#e67e22"]
    # STABLE colour per site (by cluster order) → a site is the SAME colour on every plot, even when the
    # opex plot drops sites that have no P&L. Anchor is always red.
    cmap = {k: PAL[i % len(PAL)] for i, k in enumerate(x for x in keys if x != anchor)}
    cmap[anchor] = "#e6194B"
    # ONE common month-on-month axis for the whole cluster → every site lines up month-for-month
    _ad = df[df.site_key.isin(keys)].date
    cidx = pd.date_range(_ad.min(), _ad.max(), freq="MS")
    frames = {k: df[df.site_key == k].set_index("date").sort_index().reindex(cidx) for k in keys}
    opx = {}
    for k in keys:
        o = opex[opex.site_key == k].set_index("date").opex.sort_index()
        if len(o):
            opx[k] = o.reindex(cidx)

    xpos = np.arange(1, len(cidx) + 1)                              # sequential month: 1, 2, 3, … to the end
    cmon = [d.strftime("%b %Y") for d in cidx]                      # calendar label kept for the hover
    xdt = 6 if len(cidx) > 30 else 3

    # opex spikes on the cluster-total opex → shade the SAME month windows on EVERY plot
    spike_bands = []
    if opx:
        otot = _sm(pd.concat(opx.values(), axis=1).sum(axis=1, min_count=1).reindex(cidx))
        if otot.notna().sum() >= 6:
            med = otot.median(); mad = (otot - med).abs().median()
            thr = (med + 3 * 1.4826 * mad) if mad > 0 else med * 1.5
            flags = ((otot > thr) & (otot > 1.25 * med)).fillna(False).values
            lo_b, hi_b = 0.5, len(cidx) + 0.5
            i = 0
            while i < len(flags):                                    # merge consecutive spike months into one band
                if flags[i]:
                    j = i
                    while j + 1 < len(flags) and flags[j + 1]:
                        j += 1
                    # pad 3 months back and 3 months ahead of the spike, clamped to the axis
                    spike_bands.append((max(lo_b, i + 0.5 - 3), min(hi_b, j + 1.5 + 3))); i = j + 1
                else:
                    i += 1

    def _plot(seriesmap, title, ytitle, key, money=False):
        f = go.Figure()
        vfmt = "$%{y:,.0f}" if money else "%{y:,.0f}"
        for x0, x1 in spike_bands:                                   # opex-spike shading, shared across all plots
            f.add_vrect(x0=x0, x1=x1, fillcolor="#e6194B", opacity=0.10, line_width=0, layer="below")
        for k in [x for x in keys if x != anchor] + [anchor]:
            if k not in seriesmap:
                continue
            y = _sm(seriesmap[k]).reindex(cidx); is_a = (k == anchor)
            f.add_trace(go.Scatter(x=xpos, y=y.values, name=lab[k] + (" ★" if is_a else ""), mode="lines",
                                   line=dict(color=cmap[k], width=3.4 if is_a else 1.8),
                                   opacity=1.0 if is_a else 0.7, customdata=cmon,
                                   hovertemplate="month %{x} · %{customdata} · " + vfmt + f"<extra>{lab[k]}</extra>"))
        f.update_layout(title=title, height=300, template="plotly_white", yaxis_title=ytitle,
                        margin=dict(l=8, r=8, t=38, b=8), legend=dict(orientation="h", y=-0.3, font=dict(size=9)),
                        xaxis=dict(title="month", dtick=xdt, range=[0.5, len(cidx) + 0.5]))
        st.plotly_chart(f, width="stretch", key=key)

    def _col(name):
        return {k: frames[k][name] for k in keys if len(frames[k])}

    def _asp(rev, wash):
        return {k: frames[k][rev] / frames[k][wash].replace(0, np.nan) for k in keys if len(frames[k])}

    st.markdown("**≥3 sites here have P&L — compare the anchor ★ (bold red) against each neighbour on revenue, price & spend:**")
    r = st.columns(2)
    with r[0]: _plot(_col("tot_revenue"), "Total revenue", "$ / mo", "aj_totrev", money=True)
    with r[1]: _plot(opx, "Operating expense", "$ / mo", "aj_opex", money=True)
    r = st.columns(2)
    with r[0]: _plot(_asp("mem_revenue", "mem_wash_count"), "Membership ASP", "$ / wash", "aj_am", money=True)
    with r[1]: _plot(_asp("ret_revenue", "ret_wash_count"), "Retail ASP", "$ / wash", "aj_ar", money=True)
    r = st.columns(2)
    with r[0]: _plot(_col("tot_wash_count"), "Total wash count", "washes / mo", "aj_tot")
    with r[1]: _plot(_col("mem_wash_count"), "Membership wash count", "washes / mo", "aj_mw")
    r = st.columns(2)
    with r[0]: _plot(_col("ret_wash_count"), "Retail wash count", "washes / mo", "aj_rw")
    with r[1]: _plot(_col("mem_revenue"), "Membership revenue", "$ / mo", "aj_mr", money=True)
    r = st.columns(2)
    with r[0]: _plot(_col("ret_revenue"), "Retail revenue", "$ / mo", "aj_rr", money=True)
    st.caption("Each line = a site in the cluster (★ = anchor). Washes / ASP / revenue from `main-ds.csv`, opex from "
               "`pnl_operational.xlsx`. **Red shaded bands = months the cluster's opex spiked (± 3 months around it)** — "
               "the same window is marked on every plot so you can see what revenue/price/volume did before, during and "
               "after the spend surge. "
               "**The cannibalization story:** the anchor typically **undercuts on ASP** and/or "
               "**out-spends on opex**, so its washes & revenue climb while the neighbours' fall.")


# ───────────────────────────── UI ─────────────────────────────
st.set_page_config(page_title="Local Market Explorer", layout="wide")
df, site = load_data()
pins = interesting_pins(site)

with st.sidebar:
    st.header("Controls")
    demo = st.toggle("👔 Client demo (anonymized)", value=False,
                     help="Hide identities: sites become 'Site 1, 2, 3…' by opening order, operators become "
                          "'Operator N', and the map shows a shaded red region instead of exact dots.")
    app_mode = st.radio("Mode", ["🗺️ Explore markets", "📍 Drop-a-pin forecast", "⚓ Anchor journey"], index=0)
if app_mode.startswith("📍"):
    drop_pin_ui(df, site, get_model(), demo)
    st.stop()
if app_mode.startswith("⚓"):
    anchor_journey_ui(df, site, demo)
    st.stop()

st.title("🚗 Local Market Explorer — new-site entry & ramp-up")
st.caption("Pick a pin → see its neighbours within the radius (the local market). The **NEW** entrant is drawn in **red**; "
           "the dashed line marks its opening, so you can watch it ramp up while the incumbents' series react.")

with st.sidebar:
    metric_label = st.selectbox("Metric", list(METRICS), index=0)
    metric = METRICS[metric_label]
    radius = st.slider("Neighbour radius (km)", 2, 40, 20, 1)
    x_mode = st.radio("X-axis", ["Calendar date", "Months since each site opened"], index=0,
                      help="Calendar shows the entry event & neighbour response; months-since-open overlays ramp shapes.")
    smooth = st.select_slider("Smoothing (months)", [1, 2, 3, 4, 6], value=1)
    max_sites = st.slider("Max sites shown", 4, 25, 10, help="Caps clutter in dense markets; the pin and all new entrants are always shown.")
    st.divider()
    if "pin" not in st.session_state:
        st.session_state.pin = pick_default_pin(site, df, tuple(pins))
    if st.button("🎲 Random pin", width="stretch"):
        st.session_state.pin = pins[np.random.randint(len(pins))]
    # cluster / site pickers
    multi = site[(site.cluster >= 0)].copy()
    cl_sizes = multi.groupby("cluster").size()
    cur_cluster = int(site.loc[site.site_key == st.session_state.pin, "cluster"].iloc[0])
    cl_opts = sorted(cl_sizes.index, key=lambda c: -cl_sizes[c])
    sel_cl = st.selectbox("Jump to cluster (by size)", cl_opts,
                          index=cl_opts.index(cur_cluster) if cur_cluster in cl_opts else 0,
                          format_func=lambda c: f"cluster {c} · {cl_sizes[c]} sites")
    cl_sites = site[site.cluster == sel_cl].sort_values("op_start")
    clrank = {k: i + 1 for i, k in enumerate(cl_sites.site_key)}        # Site N by opening order (demo labels)
    sel_site = st.selectbox("Pin (site)", cl_sites.site_key.tolist(),
                            index=(cl_sites.site_key.tolist().index(st.session_state.pin)
                                   if st.session_state.pin in cl_sites.site_key.tolist() else 0),
                            format_func=lambda k: (f"Site {clrank[k]}" if demo
                                                   else f"{site.loc[site.site_key==k,'client_name'].iloc[0][:26]} "
                                                        f"({site.loc[site.site_key==k,'op_start'].iloc[0]:%b %Y})"))
    if sel_site != st.session_state.pin:
        st.session_state.pin = sel_site
    # freedom: drop a pin anywhere by typing coordinates → jumps to the nearest site's market
    with st.expander("📍 Or type any location (lat, lon)"):
        _cur = site.loc[site.site_key == st.session_state.pin].iloc[0]
        ilat = st.number_input("Latitude", value=float(_cur.lat), format="%.4f", key="ex_lat")
        ilon = st.number_input("Longitude", value=float(_cur.lon), format="%.4f", key="ex_lon")
        if st.button("Go to nearest site", width="stretch"):
            _g = site[site.has_coords]
            st.session_state.pin = _g.iloc[int(np.argmin(haversine_km(ilat, ilon, _g.lat.values, _g.lon.values)))].site_key
            st.rerun()

pin = st.session_state.pin
nb_full = neighbourhood(site, pin, radius)
# cap clutter: always keep the pin + every entrant, fill the rest with nearest incumbents
keep = nb_full[nb_full.is_pin | nb_full.is_entrant]
n_inc = max(0, max_sites - len(keep))
inc = nb_full[~nb_full.is_pin & ~nb_full.is_entrant].nsmallest(n_inc, "dist_km")
nb = pd.concat([keep, inc]).drop_duplicates("site_key").sort_values("op_start").reset_index(drop=True)
entrants = nb[nb.is_entrant]
# focal new site = the newest entrant (fallback to pin)
focal_key = entrants.sort_values("op_start").site_key.iloc[-1] if len(entrants) else pin

p = site.loc[site.site_key == pin].iloc[0]
_dom = set(nb_full.site_key) | (set(site[site.cluster == p.cluster].site_key) if p.cluster >= 0 else set())
demo_label = anon_names(site, _dom) if demo else {}                    # site_key -> "Site N" by opening order
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pin", demo_label.get(pin, "Site") if demo else p.client_name[:20])
c2.metric("Sites in market", len(nb_full), help=f"within {radius} km; showing {len(nb)}")
c3.metric("New entrants", int(nb_full.is_entrant.sum()))
c4.metric("Local market", ("this market" if demo else ("standalone" if p.cluster < 0 else f"#{p.cluster}")))

left, right = st.columns([1, 1.4])
with left:
    st.subheader("Map")
    fmap = folium.Map(location=[p.lat, p.lon], zoom_start=10, tiles="cartodbpositron")
    if demo:
        # confidential demo: no site dots / no exact pin — a soft shaded red region marks the local market
        for rad_km, fop in [(radius, 0.08), (radius * 0.55, 0.12), (radius * 0.28, 0.18)]:
            folium.Circle([p.lat, p.lon], radius=rad_km * 1000, color="#c0392b", weight=0,
                          fill=True, fill_color="#e6194B", fill_opacity=fop).add_to(fmap)
        mp = st_folium(fmap, height=430, use_container_width=True, returned_objects=["last_clicked"])
        lc = (mp or {}).get("last_clicked")           # click anywhere → snap to the nearest (hidden) site's market
        if lc:
            gco = site[site.has_coords]
            nk = gco.iloc[int(np.argmin(haversine_km(lc["lat"], lc["lng"], gco.lat.values, gco.lon.values)))].site_key
            if nk != pin:
                st.session_state.pin = nk; st.rerun()
    else:
        folium.Circle([p.lat, p.lon], radius=radius * 1000, color="#999", weight=1, fill=True,
                      fill_opacity=0.05).add_to(fmap)
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
                        + (" · NEW" if s.is_entrant else ""),
                popup=s.site_key).add_to(fmap)
        folium.Marker([p.lat, p.lon], icon=folium.Icon(color="black", icon="star"),
                      tooltip="📍 pin").add_to(fmap)
        mp = st_folium(fmap, height=430, use_container_width=True,
                       returned_objects=["last_object_clicked_popup", "last_clicked"])
        clicked = (mp or {}).get("last_object_clicked_popup")        # click a marker -> exact repin
        if clicked and clicked in set(site.site_key) and clicked != pin:
            st.session_state.pin = clicked; st.rerun()
        lc = (mp or {}).get("last_clicked")                          # click ANYWHERE -> snap to the nearest site
        if lc:
            _g = site[site.has_coords]
            nk = _g.iloc[int(np.argmin(haversine_km(lc["lat"], lc["lng"], _g.lat.values, _g.lon.values)))].site_key
            if nk != pin:
                st.session_state.pin = nk; st.rerun()

with right:
    st.subheader(f"{metric_label} vs {'date' if x_mode=='Calendar date' else 'months since open'}")
    fig = build_figure(df, nb, metric, metric_label, focal_key, x_mode, smooth, label_of=(demo_label if demo else None))
    st.plotly_chart(fig, width="stretch")

# ───────────────────────── cluster-wise KPI panels ─────────────────────────
# The 6 KPIs (retail/membership wash & revenue + the two ASPs), summed across the whole
# cluster each month — the same set shown in final_modelling/six_year_app.py, but cluster-level.
st.divider()
if int(p.cluster) >= 0:
    ckeys = site[site.cluster == p.cluster].site_key.tolist()
    cdesc = f"this local market · {len(ckeys)} sites" if demo else f"cluster #{int(p.cluster)} · {len(ckeys)} sites"
else:
    ckeys = nb_full.site_key.tolist()
    cdesc = f"this local market · {len(ckeys)} sites" if demo else f"local market ≤{radius} km · {len(ckeys)} sites (pin is standalone)"
st.subheader(f"📊 Cluster KPIs over time — {cdesc}")
sub = df[df.site_key.isin(ckeys)].copy()
sub["asp_ret"] = sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan)   # per-SITE ASP = its revenue ÷ its washes
sub["asp_mem"] = sub.mem_revenue / sub.mem_wash_count.replace(0, np.nan)
PANELS = [("ret_wash_count", "Retail wash count", "count"), ("mem_wash_count", "Membership wash count", "count"),
          ("ret_revenue", "Retail revenue ($)", "$"), ("mem_revenue", "Membership revenue ($)", "$"),
          ("asp_ret", "ASP per wash — retail ($)", "$"), ("asp_mem", "ASP per wash — membership ($)", "$")]
name_of = demo_label if demo else site.set_index("site_key").client_name.to_dict()
PALETTE = ["#2E86DE", "#16a085", "#8e44ad", "#e67e22", "#27ae60", "#2980b9", "#c0392b", "#d35400", "#7f8c8d",
           "#2c3e50", "#1abc9c", "#9b59b6", "#34495e", "#f39c12", "#3498db", "#e74c3c", "#95a5a6", "#0a84ff"]
kfig = make_subplots(rows=2, cols=3, subplot_titles=[m[1] for m in PANELS],
                     vertical_spacing=0.14, horizontal_spacing=0.06)
order = [k for k in ckeys if k != pin] + ([pin] if pin in ckeys else [])   # draw the pin LAST so it sits on top
for si, k in enumerate(order):
    g = sub[sub.site_key == k].set_index("date").sort_index()
    if len(g):
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS"))   # even monthly grid for smoothing
    is_pin = (k == pin)
    color = "#e6194B" if is_pin else PALETTE[si % len(PALETTE)]
    nm = (str(name_of.get(k, "?"))[:18]) + (" ⭐" if is_pin else "")
    for i, (c, lbl, unit) in enumerate(PANELS):
        y = g[c].rolling(smooth, center=True, min_periods=1).mean() if (smooth and smooth > 1) else g[c]   # match the slider
        vfmt = "$%{y:,.2f}" if unit == "$" else "%{y:,.0f}"
        kfig.add_trace(go.Scatter(x=g.index, y=y, mode="lines", name=nm, legendgroup=k, showlegend=(i == 0),
                                  line=dict(color=color, width=3 if is_pin else 1.4), opacity=1.0 if is_pin else 0.7,
                                  hovertemplate=f"<b>{nm}</b><br>%{{x|%b %Y}} · {vfmt}<extra></extra>"),
                       row=i // 3 + 1, col=i % 3 + 1)
kfig.update_layout(height=620, template="plotly_white", margin=dict(l=8, r=8, t=44, b=10),
                   hovermode="closest", legend=dict(orientation="h", y=-0.12, font=dict(size=10)))
kfig.update_xaxes(dtick="M12", tickformat="%Y")
st.plotly_chart(kfig, width="stretch", key="cluster_kpis")
st.caption(f"**One line per site** in the cluster (⭐ = your pin, bold red) across all 6 KPIs — wash counts, revenue, and "
           f"each site's ASP (its revenue ÷ its washes). Click a site in the legend to isolate it across every panel. "
           f"{'Smoothed to a **'+str(smooth)+'-mo** rolling average (sidebar slider).' if smooth and smooth > 1 else 'Raw monthly values (set the smoothing slider to average).'}")

st.divider()
st.caption("Roles: 🔴 red = newest entrant · 🟠 orange = other new entrants · 🔵 blue = incumbents (already open). "
           "**Click anywhere on the map**, type coordinates in the sidebar, pick a site, or hit 🎲 — any of them jumps to that market.")
