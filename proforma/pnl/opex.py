"""Shared opex / ASP scope helpers — the P&L math used by BOTH the API (app/pnl_analysis/modelling/pnl.py)
and the Streamlit UI (proforma/ui/panels/_pinpoint_forecast.py). ONE copy; see docs/DIVERGENCES.md §1.

  regional_opex / opex_per_wash / opex_ramp / opex_pct_fit / opex_pct_curve_fit — the learned opex ramp
  and %-of-revenue curve, scoped state->region->all. opex_trend_hist — historical YoY opex growth (context).
  asp_refs — overall + cluster reference ASPs for the price slider. _drop_corrupt_asp_rows / global_healthy_asp
  — the corrupted-ASP floor and the global healthy fallback.

PURE module: numpy / pandas only, NO streamlit. State->region and the P&L loaders come from proforma.pnl.data.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from proforma.pnl import data as D
from proforma.pnl.data import haversine_km


# ── corrupted-ASP floor (mirror of app.py): drop site-months where the revenue feed dropped to ~0 while
# wash_count stayed normal (a data-feed drop, not a real price) BEFORE pooling the cluster ASP. ──
ASP_MIN_WASH = 200       # only judge rows with material volume (>=200 washes)
ASP_FLOOR_MEM = 4.0      # $/membership-wash below this @ >=200 washes => corrupt (healthy median ~$11)
ASP_FLOOR_RET = 5.0      # $/retail-wash below this @ >=200 washes => corrupt (healthy median ~$16)


def _drop_corrupt_asp_rows(rec):
    """Drop site-months whose revenue collapsed to ~0 while wash_count stayed normal. Row is bad if it has
    >=ASP_MIN_WASH washes AND an implied $/wash below the floor. Returns (filtered_rec, n_dropped)."""
    if rec.empty:
        return rec, 0
    mw, rw = rec.mem_wash_count.replace(0, np.nan), rec.ret_wash_count.replace(0, np.nan)
    bad = (((rec.mem_wash_count >= ASP_MIN_WASH) & (rec.mem_revenue / mw < ASP_FLOOR_MEM))
           | ((rec.ret_wash_count >= ASP_MIN_WASH) & (rec.ret_revenue / rw < ASP_FLOOR_RET))).fillna(False)
    return rec[~bad], int(bad.sum())


_GLOBAL_ASP_CACHE: Dict[str, float] = {}


def global_healthy_asp(df):
    """Wash-weighted cluster-ASP fallback from ALL healthy site-months (after the corrupt-row floor),
    used when every in-radius neighbour is corrupt. Returns (cl_mem_pp, purch_per_wash, cl_ret) —
    membership priced per PURCHASE plus purchases-per-wash, so a caller can price either basis
    (wash-basis cl_mem = cl_mem_pp * purch_per_wash). Cached on first call."""
    if not _GLOBAL_ASP_CACHE:
        rec, _ = _drop_corrupt_asp_rows(df)
        mm = rec.dropna(subset=["mem_revenue"]); rr = rec.dropna(subset=["ret_revenue"])
        mp, mw = mm.mem_purchase_count.sum(), mm.mem_wash_count.sum()
        _GLOBAL_ASP_CACHE["mem_pp"] = float(mm.mem_revenue.sum() / mp) if mp > 0 else 30.0
        _GLOBAL_ASP_CACHE["ppw"] = float(mp / mw) if mw > 0 else 0.33
        _GLOBAL_ASP_CACHE["ret"] = float(rr.ret_revenue.sum() / rr.ret_wash_count.sum()) if rr.ret_wash_count.sum() > 0 else 15.0
    return _GLOBAL_ASP_CACHE["mem_pp"], _GLOBAL_ASP_CACHE["ppw"], _GLOBAL_ASP_CACHE["ret"]


# ─────────────────────────── opex / asp scope helpers (ported verbatim from app.py) ───────────────────────────
def regional_opex(pnl, art, state, region, min_sites=5):
    """Average annual operating expense per site, by YEAR, for the pin's STATE (if >=min_sites of P&L data
    there) else its REGION else all sites. Returns (year-table, scope-label)."""
    s2r = D.state_to_region(art)
    p = pnl.copy(); p["region"] = p.state.map(s2r)
    sub, scope = p[p.state == state], f"state {state}"
    if sub.location_name.nunique() < min_sites:
        rsub = p[p.region == region]
        sub, scope = (rsub, f"region {region}") if len(rsub) else (p, "all sites (no local P&L)")
    yo = (sub.groupby("year").agg(opex=("opex", "mean"), income=("income", "mean"), n=("location_name", "nunique"),
                                  asp_mem=("asp_mem", "median"), asp_ret=("asp_ret", "median")).reset_index())
    return yo, scope


def opex_per_wash(pm, art, lat, lon, state, region, min_sites=5):
    """MATURE (age 18–30) monthly operating-expense **per wash** ($) — the settled cost level. Scoped to the
    pin's state (>=min_sites of P&L data), else its region (>=min_sites), else all sites. Returns ($/wash, scope)."""
    s2r = D.state_to_region(art)
    d = pm[(pm.wash > 0) & (pm.opex > 0) & (pm.age.between(18, 30))].copy(); d["region"] = d.state.map(s2r)
    sub, sc = d[d.state == state], f"state {state}"
    if sub.location_name.nunique() < min_sites:
        r = d[d.region == region]
        sub, sc = (r, f"region {region}") if r.location_name.nunique() >= min_sites else (d, "all sites")
    return float((sub.opex / sub.wash).median()), sc


def opex_ramp(pm, art, state=None, region=None, min_sites=8):
    """LEARNED new-site opex lifecycle from the P&L, REGION-SPECIFIC where supported.
    age 0 = the site's FIRST P&L row (year,month) — NOT created_date. For each site, opex(age) ÷ its OWN mature
    (mo 18–30) opex, then the median across sites by age — scoped to the pin's STATE (>=min_sites), else REGION,
    else ALL sites. New sites run HOT early (setup/marketing/ramp) then settle to ~1× by ~year 1.
    NOTE: the P&L only spans ~33 months, so the learned curve ends at `hage` (the last age with support). The caller
    EXTENDS months hage+1..60 with the forecast wash volume (opex ≈ $/wash × forecast washes), not a flat line.
    Returns (ramp[0..60] normalized so mature=1, scope-label, hage = last age with real P&L support)."""
    s2r = D.state_to_region(art)
    d = pm.copy(); d["region"] = d.state.map(s2r)
    last = d.groupby("location_name").date.transform("max")            # drop each site's LAST month — partial-period
    d = d[d.date < last]                                               # export artifact (near-zero opex at the tail)
    d = d[(d.age >= 0) & (d.age <= 42) & (d.opex > 0)]
    mat = d[d.age.between(18, 30)].groupby("location_name").opex.mean(); mat = mat[mat > 0]
    d = d[d.location_name.isin(mat.index)].copy(); d["rel"] = d.opex / d.location_name.map(mat)

    def _curve(sub):
        sup = sub.groupby("age").location_name.nunique()
        med = sub.groupby("age").rel.median()[sup >= max(4, min_sites // 2)]   # keep only ages with support
        med = med[med.index <= 30]                                            # trust only <= mo30 (end of mature window);
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


def opex_pct_fit(pm, art, state=None, region=None, min_sites=8, max_age=30):
    """EMPIRICAL operating-expense ratio (opex ÷ income, %) by months since inception — the %-of-revenue opex
    PATTERN. For each scoped site we take its monthly opex%, then the MEDIAN and 25–75 band across sites by age
    (age 0 = the site's FIRST P&L row), scoped state→region→all. Mirrors app.opex_pct_fit.
    Returns (ages, median, q25, q75, support_n, scope) or None if there isn't enough history."""
    s2r = D.state_to_region(art)
    d = pm.copy(); d["region"] = d.state.map(s2r)
    last = d.groupby("location_name").date.transform("max")            # drop each site's partial last month
    d = d[d.date < last]
    d = d[(d.age >= 0) & (d.age <= max_age) & (d.opex > 0) & (d.income > 0)].copy()
    d["pct"] = 100.0 * d.opex / d.income
    d = d[d.pct.between(0, 500)]                                        # drop export garbage (near-zero income → absurd %)

    def _fit(sub):
        if sub.location_name.nunique() < min_sites:
            return None
        sup = sub.groupby("age").location_name.nunique()
        keep = sup[sup >= max(4, min_sites // 2)].index
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


def _synthetic_opex_pct(months):
    """Fallback opex %-of-revenue curve when there's too little P&L to fit: hot ~60% easing to a mature ~45%."""
    m = np.asarray(months, float)
    return np.clip(0.45 + 0.17 * np.exp(-m / 6.0), 0.30, 1.5)


def opex_pct_curve_fit(pm, art, state=None, region=None, months=None, min_sites=8):
    """Opex %-of-revenue curve FIT to the empirical pattern: mature + (hot − mature)·exp(−age/τ), support-weighted
    so thin/noisy ages don't drag it, evaluated over `months` (PROPAGATED flat to the mature level past the data).
    Mirrors app.opex_pct_curve_fit. Returns (curve_fraction[len(months)], scope_label, is_learned)."""
    if months is None:
        months = np.arange(0, 61)
    months = np.asarray(months, float)
    fit = opex_pct_fit(pm, art, state, region, min_sites=min_sites, max_age=30)
    if fit is None:
        return _synthetic_opex_pct(months), "synthetic (insufficient P&L)", False
    age, med, _q25, _q75, sup, scope = fit
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
    return np.clip(_decay(months, *popt), 0.30, 1.5), scope, True


def opex_trend_hist(pnl, art, state, region, min_sites=5):
    """Median per-site YoY opex growth for the pin's scope — the historical opex 'pattern' (context for the
    cost-growth slider). NOTE: on this data it's strongly negative & noisy (likely a reporting artifact), so it's
    shown, not used as the default. Returns a fraction/yr."""
    s2r = D.state_to_region(art)
    a = pnl.copy(); a["region"] = a.state.map(s2r)
    a = a.sort_values(["location_name", "year"]); a["prev"] = a.groupby("location_name").opex.shift(1)
    a["yoy"] = a.opex / a["prev"] - 1
    sub = a[a.state == state]
    if sub.location_name.nunique() < min_sites:
        r = a[a.region == region]; sub = r if len(r) else a
    yoy = sub.yoy.replace([np.inf, -np.inf], np.nan).dropna()
    return float(yoy.median()) if len(yoy) else 0.0


def asp_refs(pnl, art, lat, lon, state, region):
    """Two reference ASPs to mark on the sliders: OVERALL (all P&L sites) and CLUSTER/LOCAL (P&L sites <=25 km of
    the pin, falling back to state → region → overall when too few). Returns mem/ret for each + a scope label."""
    ov_mem, ov_ret = float(pnl.asp_mem.median()), float(pnl.asp_ret.median())
    loc = (pnl.groupby("location_name")
           .agg(lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
                asp_mem=("asp_mem", "median"), asp_ret=("asp_ret", "median")).reset_index().dropna(subset=["lat", "lon"]))
    loc["d"] = haversine_km(lat, lon, loc.lat.values, loc.lon.values)
    near = loc[loc.d <= 25]
    if len(near) >= 2:
        sub, sc = near, f"cluster <=25 km · {len(near)} sites"
    elif (loc.state == state).sum() >= 3:
        sub, sc = loc[loc.state == state], f"state {state}"
    else:
        s2r = D.state_to_region(art)
        loc["region"] = loc.state.map(s2r); rg = loc[loc.region == region]
        sub, sc = (rg, f"region {region}") if len(rg) else (loc, "all sites")
    return dict(ov_mem=ov_mem, ov_ret=ov_ret, cl_mem=float(sub.asp_mem.median()),
                cl_ret=float(sub.asp_ret.median()), scope=sc)
