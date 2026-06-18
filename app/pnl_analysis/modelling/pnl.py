"""
P&L engine for the Forecast tab — expected revenue (forecast washes × ASP) vs operating expense.

Backs the Forecast tab's "💰 P&L — expected revenue vs operating expense" section. Every helper here
is ported verbatim from the Streamlit reference (earnest-proforma-2.0/streamlits/app.py); the only
change is that each one builds the state→region map via `D.state_to_region(art)` instead of inline, and
reads the operating P&L through `D.load_pnl_annual()` / `D.load_pnl_monthly()` (same data as app.py's
`load_pnl()` / `load_pnl_monthly()`).

  • regional_opex(...)     — average annual opex/income per site by year, scoped state→region→all.
  • opex_per_wash(...)     — mature (age 18–30) monthly opex per wash ($), scoped state→region→all.
  • opex_ramp(...)         — learned new-site opex lifecycle (normalized so mature=1), region-scoped.
  • opex_trend_hist(...)   — median per-site YoY opex growth for the scope (context, not a default).
  • asp_refs(...)          — overall + cluster/local reference ASPs (mem/ret) for the price slider.
  • pnl_forecast(...)      — the orchestrator: reproduces the drop_pin_ui P&L math end-to-end and
                             returns one JSON-serializable dict (floats/ints/None — never numpy/NaN).

PURE module: numpy / pandas only, NO streamlit / plotly / folium. Where the reference reads a value off
a chart or slider, it is exposed here as a function argument with the same default.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.pnl_analysis.modelling import data as D
from app.pnl_analysis.modelling.data import haversine_km
from app.pnl_analysis.modelling.market import compute_trajectory
from app.pnl_analysis.modelling.campaign import campaign_effect, campaign_conv_pct


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


# ─────────────────────────── helpers ───────────────────────────
def _f(x):
    """numpy/NaN-safe float coercion for JSON output: returns a Python float, or None for NaN/inf/None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ─────────────────────────── orchestrator ───────────────────────────
def pnl_forecast(lat: float, lon: float, brand: Optional[str] = None,
                 plateau_override: Optional[float] = None, mem_growth_pct: float = 0.0,
                 ret_growth_pct: float = 0.0, asp_override: Optional[float] = None,
                 opex_growth_pct: float = 0.0, campaign_on: bool = False, campaign_launch: int = 13,
                 campaign_intensity: float = 1.0, window: int = 6,
                 horizon_months: int = 60) -> Dict[str, Any]:
    """Forecast-tab P&L: expected monthly revenue (forecast washes × ASP) vs operating expense over 5 years,
    with an optional retail→membership campaign overlay. Reproduces the math in app.py's drop_pin_ui P&L block.

    Args mirror the reference's chart/slider reads: `asp_override` is the ASP ($/wash) slider (None → cluster
    blended avg), `opex_growth_pct` is the opex cost-growth slider (%/yr), and `campaign_*` / `window` are the
    campaign controls. Returns a JSON-serializable dict (floats/ints/None — no numpy scalars, no NaN).
    """
    radius = 20.0  # the cold-start trajectory & ASP/incumbent neighbourhood radius (compute_trajectory default)

    df, site = D.load_panel()
    art = D.load_model()
    pnl = D.load_pnl_annual()
    pm_monthly = D.load_pnl_monthly()

    # 1) the new site's 5-year monthly trajectory + the trends/info it used
    traj, info, trends = compute_trajectory(lat, lon, brand, plateau_override,
                                            mem_growth_pct, ret_growth_pct, horizon_months)
    state, region = info.get("state"), info.get("region")
    yo, opex_scope = regional_opex(pnl, art, state, region)

    tj = traj.set_index("month")
    months = np.arange(0, 61)
    mem = tj["mem_med"].reindex(months).fillna(0.0).to_numpy()
    ret = tj["ret_med"].reindex(months).fillna(0.0).to_numpy()

    # 2) CLUSTER ASP from the dense operational data (main-ds): the <=radius km neighbours' last 12 months
    _nbk = site[site.has_coords].copy()
    _nbk["d"] = haversine_km(lat, lon, _nbk.lat.values, _nbk.lon.values)
    _ck = _nbk.site_key[(_nbk.d <= radius) & (_nbk.d > 1e-6)].tolist()
    _rec = df[df.site_key.isin(_ck)].sort_values("date").groupby("site_key").tail(12) if _ck else df.iloc[:0]
    _mm, _rr = _rec.dropna(subset=["mem_revenue"]), _rec.dropna(subset=["ret_revenue"])
    cl_mem = float(_mm.mem_revenue.sum() / _mm.mem_wash_count.sum()) if _mm.mem_wash_count.sum() > 0 else 10.0
    cl_ret = float(_rr.ret_revenue.sum() / _rr.ret_wash_count.sum()) if _rr.ret_wash_count.sum() > 0 else 15.0
    asp_scope = f"cluster <={radius:.0f} km · {len(_ck)} sites" if _ck else "default (no neighbours)"
    s_mix = float(info.get("mem_share", 0.6))                                   # this site's membership share of washes
    asp_blend = s_mix * cl_mem + (1 - s_mix) * cl_ret                           # cluster average BLENDED $/wash

    # reference ASPs + historical opex YoY (context, exposed in the output)
    refs = asp_refs(pnl, art, lat, lon, state, region)
    g_hist = opex_trend_hist(pnl, art, state, region)

    # 3) the ASP slider scales price; keep the data's mem/retail split
    asp = float(asp_override) if asp_override is not None else asp_blend
    k_asp = asp / max(asp_blend, 1e-9)
    asp_mem, asp_ret = cl_mem * k_asp, cl_ret * k_asp

    # 4) opex = LEARNED new-site ramp (shape, REGION-scoped) × this site's mature level (mature $/wash × plateau)
    ramp_o, ramp_scope, ramp_hage = opex_ramp(pm_monthly, art, state, region)
    opw_mat, opw_scope = opex_per_wash(pm_monthly, art, lat, lon, state, region)
    mature_opex = opw_mat * float(info["plateau_med"])                          # this site's settled monthly opex $
    # BEYOND the ~33-mo P&L horizon: don't flat-line — drive opex with the forecast wash volume.
    wt = mem + ret
    H = min(int(ramp_hage), 60)
    base = wt[H] if (H < len(wt) and wt[H] > 0) else float(np.nanmedian(wt[max(0, H - 3):H + 1]) or 0)
    if base > 0:
        for t in range(H + 1, 61):
            ramp_o[t] = ramp_o[H] * (wt[t] / base)                             # opex follows forecast volume past the data
        ramp_o = np.clip(ramp_o, 0.3, 3.5)

    # 6a) THIS SITE's settled membership share (plateau months 36–60) — what the trajectory settles at
    ms = float(info.get("mem_share", 0.6))                                     # fallback if the trajectory is empty
    _plm, _plr = float(np.nansum(mem[36:61])), float(np.nansum(ret[36:61]))
    if _plm + _plr > 0:
        ms = _plm / (_plm + _plr)
    conv = campaign_conv_pct(ms)

    # 5) cost escalation (slider) over the horizon × the learned ramp
    opex_grow = (1 + opex_growth_pct / 100.0) ** (months / 12.0)
    opex_base_m = mature_opex * ramp_o[:61] * opex_grow                        # LEARNED ramp (hot early) × growth — no campaign

    # 6b) campaign overlay
    if campaign_on:
        mem_mult, ret_mult, opex_mult_c = campaign_effect(campaign_launch, ms, campaign_intensity, window=window)
    else:
        mem_mult, ret_mult, opex_mult_c = np.ones(61), np.ones(61), np.ones(61)

    rev_base = mem * asp_mem + ret * asp_ret                                   # revenue without the campaign
    rev_m = mem * mem_mult * asp_mem + ret * ret_mult * asp_ret                # campaign shifts the wash mix retail→membership
    opex_m = opex_base_m * opex_mult_c                                         # + the promo spend over the window
    net_m = rev_m - opex_m
    net_base = rev_base - opex_base_m

    # 5-year totals + breakeven (first month cumulative net turns positive)
    total_revenue_5yr = float(np.nansum(rev_m))
    total_opex_5yr = float(np.nansum(opex_m))
    net_5yr = float(np.nansum(net_m))
    cum = np.cumsum(np.nan_to_num(net_m, nan=0.0))
    pos = np.where(cum > 0)[0]
    breakeven_month = int(pos[0]) if len(pos) else None

    return {
        "lat": _f(lat), "lon": _f(lon), "brand": brand, "horizon_months": int(horizon_months),
        "months": [int(m) for m in months],
        "asp": {
            "mem": _f(asp_mem), "ret": _f(asp_ret), "blend": _f(asp_blend), "used": _f(asp),
            "scope": asp_scope,
            "refs": {"ov_mem": _f(refs["ov_mem"]), "ov_ret": _f(refs["ov_ret"]),
                     "cl_mem": _f(refs["cl_mem"]), "cl_ret": _f(refs["cl_ret"]), "scope": refs["scope"]},
        },
        "opex": {
            "mature_opex": _f(mature_opex), "ramp_scope": ramp_scope, "opw_scope": opw_scope,
            "ramp_hage": int(ramp_hage), "hist_yoy": _f(g_hist),
        },
        "scopes": {"opex": opex_scope},
        "series": {
            "revenue_base": [_f(v) for v in rev_base],
            "revenue": [_f(v) for v in rev_m],
            "opex_base": [_f(v) for v in opex_base_m],
            "opex": [_f(v) for v in opex_m],
            "net": [_f(v) for v in net_m],
            "net_base": [_f(v) for v in net_base],
        },
        "campaign": {
            "applied": bool(campaign_on), "launch": int(campaign_launch),
            "intensity": _f(campaign_intensity), "window": int(window),
            "conv_pct": _f(conv), "mem_share_settled": _f(ms),
        },
        "summary": {
            "plateau_med": _f(info.get("plateau_med")), "mem_share": _f(info.get("mem_share")),
            "state": None if state is None else str(state), "region": None if region is None else str(region),
            "total_revenue_5yr": _f(total_revenue_5yr), "total_opex_5yr": _f(total_opex_5yr),
            "net_5yr": _f(net_5yr), "breakeven_month": breakeven_month,
        },
    }
