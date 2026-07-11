"""
P&L engine for the Forecast tab — expected revenue (forecast washes × ASP) vs operating expense.

Backs the Forecast tab's "💰 P&L — expected revenue vs operating expense" section. Every helper here
is ported verbatim from the Streamlit reference (proforma/ui/app.py); the only
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

from proforma.pnl import data as D
from proforma.pnl.data import haversine_km
from app.pnl_analysis.modelling.market import age_periods, compute_trajectory
from proforma.pnl.campaign import campaign_effect, campaign_conv_pct

from proforma.pnl.opex import (  # shared opex/asp helpers (one copy; see docs/DIVERGENCES.md §1)
    ASP_MIN_WASH, ASP_FLOOR_MEM, ASP_FLOOR_RET,
    _drop_corrupt_asp_rows, global_healthy_asp, regional_opex, opex_per_wash, opex_ramp,
    opex_pct_fit, _synthetic_opex_pct, opex_pct_curve_fit, opex_trend_hist, asp_refs,
)



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
                 horizon_months: int = 60, express_only: bool = False) -> Dict[str, Any]:
    """Forecast-tab P&L: expected monthly revenue (forecast washes × ASP) vs operating expense over 5 years,
    with an optional retail→membership campaign overlay. Reproduces the math in app.py's drop_pin_ui P&L block.

    Args mirror the reference's chart/slider reads: `asp_override` is the ASP ($/wash) slider (None → cluster
    blended avg), `opex_growth_pct` is the opex cost-growth slider (%/yr), and `campaign_*` / `window` are the
    campaign controls. Returns a JSON-serializable dict (floats/ints/None — no numpy scalars, no NaN).
    """
    radius = 20.0  # the cold-start trajectory & ASP/incumbent neighbourhood radius (compute_trajectory default)

    df, site = D.load_panel(express_only)
    art = D.load_model()
    pnl = D.load_pnl_annual()
    pm_monthly = D.load_pnl_monthly()

    # 1) the new site's 5-year monthly trajectory + the trends/info it used
    traj, info, trends = compute_trajectory(lat, lon, brand, plateau_override,
                                            mem_growth_pct, ret_growth_pct, horizon_months,
                                            express_only=express_only)
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
    # Drop corrupted site-months (revenue feed dropped to ~0 with washes intact) BEFORE pooling — else a couple
    # of bad neighbours halve the cluster $/wash and the forecast revenue. Mirrors app.py drop_pin_ui.
    _rec, _n_drop = _drop_corrupt_asp_rows(_rec)
    _mm, _rr = _rec.dropna(subset=["mem_revenue"]), _rec.dropna(subset=["ret_revenue"])
    _g_mem_pp, _g_ppw, _g_ret = global_healthy_asp(df)                          # global healthy fallback (never flat $0-ish)
    _g_mem = _g_mem_pp * _g_ppw                                                 # wash-basis membership ASP
    cl_mem = float(_mm.mem_revenue.sum() / _mm.mem_wash_count.sum()) if _mm.mem_wash_count.sum() > 0 else _g_mem
    cl_ret = float(_rr.ret_revenue.sum() / _rr.ret_wash_count.sum()) if _rr.ret_wash_count.sum() > 0 else _g_ret
    if not _ck:
        asp_scope = "global healthy avg (no neighbours)"
    elif _mm.mem_wash_count.sum() <= 0 and _rr.ret_wash_count.sum() <= 0:
        asp_scope = f"global healthy avg ({len(_ck)} neighbours all corrupt)"
    else:
        asp_scope = f"cluster <={radius:.0f} km · {len(_ck)} sites" + (f" · {_n_drop} corrupt site-mo excluded" if _n_drop else "")
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
        "years": age_periods(months)[0], "quarters": age_periods(months)[1],
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


# ─────────────────────────── expense plan (user OPEX% + CAPEX$ → opex / capex / combined lines) ───────────────────────────
def _year_slices(n_months: int, n_years: int = 5):
    """Partition month indices 0..n_months-1 into `n_years` yearly buckets of 12, the LAST bucket absorbing any
    remainder (so a 61-month horizon folds month 60 into year 5, not a 1-month stub year 6). Returns slices."""
    out = []
    for y in range(n_years):
        start = 12 * y
        if start >= n_months:
            break
        stop = n_months if y == n_years - 1 else min(12 * y + 12, n_months)
        out.append(slice(start, stop))
    return out


def _per_year_to_monthly(n_months: int, by_year: Dict[int, float], default: float) -> np.ndarray:
    """Expand a {year: value} map to a length-n_months array — each year's months get that year's value (a step
    series), missing years fall back to `default`. Used for the year-wise ASP ($/wash)."""
    out = np.full(n_months, float(default))
    for y, sl in enumerate(_year_slices(n_months)):
        v = (by_year or {}).get(y + 1)
        if v is not None:
            out[sl] = float(v)
    return out


def expense_plan(lat: float, lon: float, brand: Optional[str] = None,
                 plateau_override: Optional[float] = None, mem_growth_pct: float = 0.0,
                 ret_growth_pct: float = 0.0, asp_override: Optional[float] = None,
                 asp_by_year: Optional[Dict[int, float]] = None,
                 opex_pct_by_year: Optional[Dict[int, float]] = None,
                 capex_by_year: Optional[Dict[int, float]] = None,
                 opex_growth_pct: float = 0.0, horizon_months: int = 60,
                 express_only: bool = False) -> Dict[str, Any]:
    """User-driven expense plan over the 5-year horizon: monthly OPEX, CAPEX and combined expenses.

    OPEX is given as a % of revenue PER YEAR ({1: 55, 2: 48, 3: 45}) and is FITTED onto the LEARNED new-site opex
    pattern (hot early, easing to mature): the learned monthly shape is preserved WITHIN each year while it is
    re-scaled so that year's average opex% equals the user's target. Years past the last supplied one inherit the
    last target escalated by `opex_growth_pct`/yr. OPEX$ = opex% × revenue, where revenue = the pin's forecast
    washes × ASP. CAPEX is a $ amount PER YEAR ({1: 500000, …}), spread evenly across that year's 12 months.
    Returns a JSON-serializable dict with the opex / capex / combined_expenses monthly lines + annual rollups.
    """
    opex_pct_by_year = {int(k): float(v) for k, v in (opex_pct_by_year or {}).items()}
    capex_by_year = {int(k): float(v) for k, v in (capex_by_year or {}).items()}
    asp_by_year = {int(k): float(v) for k, v in (asp_by_year or {}).items()}

    df, site = D.load_panel(express_only)
    art = D.load_model()
    pm_monthly = D.load_pnl_monthly()

    # 1) the new site's 5-year monthly wash trajectory (mem/ret) — drives revenue
    traj, info, _trends = compute_trajectory(lat, lon, brand, plateau_override,
                                             mem_growth_pct, ret_growth_pct, horizon_months,
                                             express_only=express_only)
    state, region = info.get("state"), info.get("region")
    tj = traj.set_index("month")
    months = np.arange(0, 61)
    mem = tj["mem_med"].reindex(months).fillna(0.0).to_numpy()
    ret = tj["ret_med"].reindex(months).fillna(0.0).to_numpy()

    # 2) CLUSTER ASP (≤20 km neighbours' last 12 months) — same as pnl_forecast; the ASP input scales it
    radius = 20.0
    _nbk = site[site.has_coords].copy()
    _nbk["d"] = haversine_km(lat, lon, _nbk.lat.values, _nbk.lon.values)
    _ck = _nbk.site_key[(_nbk.d <= radius) & (_nbk.d > 1e-6)].tolist()
    _rec = df[df.site_key.isin(_ck)].sort_values("date").groupby("site_key").tail(12) if _ck else df.iloc[:0]
    _mm, _rr = _rec.dropna(subset=["mem_revenue"]), _rec.dropna(subset=["ret_revenue"])
    cl_mem = float(_mm.mem_revenue.sum() / _mm.mem_wash_count.sum()) if _mm.mem_wash_count.sum() > 0 else 10.0
    cl_ret = float(_rr.ret_revenue.sum() / _rr.ret_wash_count.sum()) if _rr.ret_wash_count.sum() > 0 else 15.0
    s_mix = float(info.get("mem_share", 0.6))
    asp_blend = s_mix * cl_mem + (1 - s_mix) * cl_ret
    # ASP: a year-wise {year: $/wash} map takes precedence, else the scalar override, else the cluster blend.
    if asp_by_year:
        asp_month = _per_year_to_monthly(len(months), asp_by_year, asp_blend)
    else:
        asp_month = np.full(len(months), float(asp_override) if asp_override is not None else asp_blend)
    k_asp = asp_month / max(asp_blend, 1e-9)                                   # scale the cluster mem/retail split per month
    asp_mem, asp_ret = cl_mem * k_asp, cl_ret * k_asp                          # monthly $/wash arrays (mem & retail)
    revenue = mem * asp_mem + ret * asp_ret
    asp_used = float(np.nanmean(asp_month))                                    # representative scalar for the output

    # 3) the LEARNED monthly opex% shape (fraction of revenue), hot early → mature
    shape, shape_scope, is_learned = opex_pct_curve_fit(pm_monthly, art, state, region, months)
    shape = np.asarray(shape, float)

    # 4) FIT the user's per-year opex% targets onto the learned shape: keep the within-year shape, but re-scale
    #    each year so its $-weighted opex% (= annual opex$ ÷ annual sales$) equals the target. Sales = ASP × washes.
    #    Years past the last supplied → last target escalated by opex_growth_pct/yr.
    yslices = _year_slices(len(months))
    targets = {y: v / 100.0 for y, v in opex_pct_by_year.items()}          # percent → fraction
    last_year = max(targets) if targets else None
    opex_pct = shape.copy()
    year_targets: Dict[int, Optional[float]] = {}
    for y, sl in enumerate(yslices):
        seg, seg_rev = shape[sl], revenue[sl]
        # revenue-WEIGHTED mean of the learned shape, so scaling it hits the target as a $-ratio:
        #   year opex$ / year sales$ = Σ(seg·f·rev)/Σrev = f·wmean = target.
        wts = seg_rev if float(np.nansum(seg_rev)) > 0 else np.ones_like(seg)
        seg_wmean = float(np.average(seg, weights=wts)) or 1.0
        yr = y + 1
        if yr in targets:
            tgt = targets[yr]
        elif last_year is not None:
            tgt = targets[last_year] * (1 + opex_growth_pct / 100.0) ** (yr - last_year)
        else:
            tgt = seg_wmean                                                # no targets → learned shape as-is
        year_targets[yr] = tgt
        opex_pct[sl] = seg * (tgt / seg_wmean)
    opex_pct = np.clip(opex_pct, 0.0, 3.0)
    opex_dollars = opex_pct * revenue

    # 5) CAPEX $ per year → spread evenly across that year's months
    capex_dollars = np.zeros(len(months))
    for y, sl in enumerate(yslices):
        amt = capex_by_year.get(y + 1)
        if amt:
            capex_dollars[sl] = amt / max(sl.stop - sl.start, 1)

    combined = opex_dollars + capex_dollars
    net = revenue - combined

    def _ser(a):
        return [_f(v) for v in a]

    def _annual(a):
        return [_f(float(np.nansum(a[sl]))) for sl in yslices]

    return {
        "lat": _f(lat), "lon": _f(lon), "brand": brand, "horizon_months": int(horizon_months),
        "months": [int(m) for m in months],
        "years": age_periods(months)[0], "quarters": age_periods(months)[1],
        "asp": {"blend": _f(asp_blend), "used_avg": _f(asp_used),
                "by_year": {str(y): _f(v) for y, v in asp_by_year.items()} or None,
                "monthly": _ser(asp_month),
                "scope": f"cluster ≤{radius:.0f} km · {len(_ck)} sites" if _ck else "default (no neighbours)"},
        "opex_pattern": {
            "scope": shape_scope, "is_learned": bool(is_learned),
            "opex_pct_used": _ser(opex_pct),                              # the fitted monthly opex% (fraction)
            "year_targets_pct": {str(y): _f((t or 0) * 100.0) for y, t in year_targets.items()},
        },
        "inputs": {
            "opex_pct_by_year": {str(k): _f(v) for k, v in opex_pct_by_year.items()},
            "capex_by_year": {str(k): _f(v) for k, v in capex_by_year.items()},
            "asp_by_year": {str(k): _f(v) for k, v in asp_by_year.items()},
            "opex_growth_pct": _f(opex_growth_pct), "asp_used_avg": _f(asp_used),
        },
        "series": {
            "revenue": _ser(revenue),
            "opex": _ser(opex_dollars),                                   # the OPEX line
            "capex": _ser(capex_dollars),                                 # the CAPEX line
            "combined_expenses": _ser(combined),                          # the COMBINED line (opex + capex)
            "net": _ser(net),
        },
        "annual": {
            "year": list(range(1, len(yslices) + 1)),
            "revenue": _annual(revenue), "opex": _annual(opex_dollars),
            "capex": _annual(capex_dollars), "combined_expenses": _annual(combined),
            "net": _annual(net),
        },
        "summary": {
            "total_revenue": _f(float(np.nansum(revenue))),
            "total_opex": _f(float(np.nansum(opex_dollars))),
            "total_capex": _f(float(np.nansum(capex_dollars))),
            "total_expenses": _f(float(np.nansum(combined))),
            "net": _f(float(np.nansum(net))),
            "state": None if state is None else str(state),
            "region": None if region is None else str(region),
        },
    }
