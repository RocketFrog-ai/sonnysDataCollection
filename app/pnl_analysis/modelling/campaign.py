"""
Campaign engine for the Forecast tab — the "should this site run a promotion?" decision and its evidence.

Backs four Streamlit sections from proforma/ui/app.py (drop_pin_ui):

  campaign_verdict(...)        — the "🎯 Campaign — should this site run a promotion?" block: established
                                 incumbents, neighbours' membership share, this site's predicted membership,
                                 and the recommend / marginal / not-recommended ladder (lines ~882-920).
  eating_the_market(...)       — the "📈 Eating the market" chart data: your site (base vs with-campaign) and
                                 each top incumbent forecast forward, drifting down as your promo steals share
                                 (lines ~959-1010).
  campaign_snapshot()          — the book_v4 OPEX / Revenue / Profit / Membership-purchases snapshot popover:
                                 median values by month-offset for the 1 / 2 / 3+ month campaign buckets
                                 (render_campaign_snapshot, lines ~508-651, sans plotting).
  local_campaign_evidence(...) — the "Real campaigns in this local market" panel: the nearest in-radius sites'
                                 monthly series for a metric with each site's detected promo months marked
                                 (campaign_cluster_panel, lines ~654-689, sans plotting).

The conversion-lift primitives `CAMP_OPEX_TAIL`, `campaign_conv_pct`, `campaign_effect` are ported verbatim
and re-exported here (pnl.py imports them too). PURE module: no streamlit / plotly / folium — every function
returns plain JSON-serializable dicts/lists (floats not numpy scalars, None for NaN). `campaign_effect` keeps
returning numpy arrays (its callers — pnl.py and this module — multiply them into float series first).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from proforma.pnl import data as D
from proforma.pnl.data import haversine_km
from app.pnl_analysis.modelling.market import age_periods, compute_trajectory
from proforma.pnl.trend import forecast_series
from proforma.pnl.campaign import (  # shared campaign helpers (one copy; docs/DIVERGENCES.md §1)
    CAMP_OPEX_TAIL, campaign_conv_pct, campaign_effect, campaign_months_by_site, _campaigns_df,
)




# ─────────────────────────── 1) campaign verdict ───────────────────────────
def campaign_verdict(lat: float, lon: float, radius_km: float = 20.0, brand: Optional[str] = None,
                     plateau_override: Optional[float] = None, mem_growth_pct: float = 0.0,
                     ret_growth_pct: float = 0.0, express_only: bool = False) -> Dict[str, Any]:
    """The "🎯 Campaign — should this site run a promotion?" decision (app.py lines ~882-920).

    Established incumbents = in-radius sites with ≥2yr history (n_obs ≥ 24). Neighbours' membership share =
    the MEDIAN of each incumbent's own recent-12mo membership share (every site counts equally). This site's
    predicted membership = its 5-yr trajectory's settled split (mem[36:61] / (mem+ret)[36:61]). Verdict ladder:
    <2 incumbents → not recommended; nb share <45% → not recommended; ≥82% → marginal; else recommended.
    """
    df, site = D.load_panel(express_only)

    traj, info, _ = compute_trajectory(lat, lon, brand=brand, plateau_override=plateau_override,
                                       mem_growth_pct=mem_growth_pct, ret_growth_pct=ret_growth_pct,
                                       horizon_months=60, radius_km=radius_km)
    ms = float(info.get("mem_share", 0.6))                          # fallback if the trajectory is empty

    # the LOCAL MARKET incumbents: established (≥2yr) sites within the radius
    _loc = site[site.has_coords].copy()
    _loc["d"] = haversine_km(lat, lon, _loc.lat.values, _loc.lon.values)
    _inc = _loc[(_loc.d <= radius_km) & (_loc.d > 1e-6) & (_loc.n_obs >= 24)]
    n_inc = len(_inc)
    _r12 = df[df.site_key.isin(_inc.site_key)].sort_values("date").groupby("site_key").tail(12)
    # neighbours' membership share = MEDIAN of each incumbent's own recent 12-mo share (every site counts equally)
    _per = _r12.groupby("site_key").apply(
        lambda gg: gg.mem_wash_count.fillna(0).sum()
        / max(1.0, gg.mem_wash_count.fillna(0).sum() + gg.ret_wash_count.fillna(0).sum()), include_groups=False)
    nb_ms = float(_per.median()) if len(_per) else float("nan")

    # THIS SITE's predicted membership = what its 5-yr trajectory SETTLES at (plateau months 36–60)
    tj = traj.set_index("month")
    months = np.arange(0, 61)
    mem = tj["mem_med"].reindex(months).fillna(0.0).to_numpy()
    ret = tj["ret_med"].reindex(months).fillna(0.0).to_numpy()
    _plm, _plr = float(np.nansum(mem[36:61])), float(np.nansum(ret[36:61]))
    if _plm + _plr > 0:
        ms = _plm / (_plm + _plr)
    conv = campaign_conv_pct(ms)

    # verdict — only recommend when the market is PROVEN: good established incumbents + membership works + share to take
    if n_inc < 2:
        ok, level = False, "not_recommended"
        verdict = (f"Not recommended. Only {n_inc} established incumbent(s) within {radius_km:g} km — the "
                   f"market is unproven, and in the data 77% of promos captured no share. Choose a denser market.")
    elif not np.isfinite(nb_ms) or nb_ms < 0.45:
        ok, level = False, "not_recommended"
        verdict = (f"Not recommended. Neighbours' membership share is {_pct(nb_ms)} (low) — the membership "
                   f"model isn't proven here, so converted customers are unlikely to stick.")
    elif nb_ms >= 0.82:
        ok, level = False, "marginal"
        verdict = (f"Marginal. Neighbours are {_pct(nb_ms)} membership — the market is near-saturated, "
                   f"so there's little retail left to convert or steal.")
    else:
        ok, level = True, "recommended"
        verdict = (f"Recommended. {n_inc} established incumbents within {radius_km:g} km at {_pct(nb_ms)} "
                   f"membership.")

    return {
        "ok": bool(ok),
        "verdict_level": level,
        "verdict_text": verdict,
        "neighbours_mem_share": (float(nb_ms) if np.isfinite(nb_ms) else None),
        "n_established_incumbents": int(n_inc),
        "this_site_mem_share": float(ms),
        "conv_pct": float(conv),
        "radius_km": float(radius_km),
    }


# ─────────────────────────── 2) eating the market ───────────────────────────
def eating_the_market(lat: float, lon: float, radius_km: float = 20.0, brand: Optional[str] = None,
                      plateau_override: Optional[float] = None, mem_growth_pct: float = 0.0,
                      ret_growth_pct: float = 0.0, campaign_on: bool = False, campaign_launch: int = 13,
                      campaign_intensity: float = 1.0, window: int = 6, max_incumbents: int = 6,
                      express_only: bool = False) -> Dict[str, Any]:
    """The "📈 Eating the market" chart data (app.py lines ~959-1010): your site (base vs with-campaign) and
    the top `max_incumbents` incumbents each forecast forward 5 years, drifting down as your promo steals share.

    `campaign_launch` / `campaign_intensity` / `window` are the campaign sliders (defaults match the UI).
    steal_peak = 0.06 · min(1, n_inc/4) · (intensity if campaign on else 1); phased over `window`, then recovering.
    Your site's with-campaign curve uses the same settled mem-share `ms` the verdict reads off the trajectory.
    """
    df, site = D.load_panel(express_only)
    traj, info, _ = compute_trajectory(lat, lon, brand=brand, plateau_override=plateau_override,
                                       mem_growth_pct=mem_growth_pct, ret_growth_pct=ret_growth_pct,
                                       horizon_months=60, radius_km=radius_km)
    tj = traj.set_index("month")
    months = np.arange(0, 61)
    mem = tj["mem_med"].reindex(months).fillna(0.0).to_numpy()
    ret = tj["ret_med"].reindex(months).fillna(0.0).to_numpy()

    # this site's settled membership share (months 36–60) — same as the verdict block
    ms = float(info.get("mem_share", 0.6))
    _plm, _plr = float(np.nansum(mem[36:61])), float(np.nansum(ret[36:61]))
    if _plm + _plr > 0:
        ms = _plm / (_plm + _plr)

    # established (≥2yr) incumbents in radius
    _loc = site[site.has_coords].copy()
    _loc["d"] = haversine_km(lat, lon, _loc.lat.values, _loc.lon.values)
    _inc = _loc[(_loc.d <= radius_km) & (_loc.d > 1e-6) & (_loc.n_obs >= 24)]
    n_inc = len(_inc)
    _r12 = df[df.site_key.isin(_inc.site_key)].sort_values("date").groupby("site_key").tail(12)

    # campaign multipliers (mem/ret) over the horizon — drives your site's with-campaign curve
    if campaign_on:
        mem_mult, ret_mult, _ = campaign_effect(campaign_launch, ms, campaign_intensity, window=window)
    else:
        mem_mult, ret_mult = np.ones(61), np.ones(61)

    new_base = mem + ret                                            # new site's washes/mo (no campaign)
    new_camp = mem * mem_mult + ret * ret_mult                      # with the campaign's conversion lift

    steal_peak = 0.06 * min(1.0, n_inc / 4.0) * (campaign_intensity if campaign_on else 1.0)
    steal = np.zeros(61)
    if campaign_on:
        for t in range(61):
            k = t - campaign_launch
            if 0 <= k < window:
                steal[t] = steal_peak * min(1.0, (k + 1) / 2.0)
            elif k >= window:
                steal[t] = steal_peak * 0.5 ** ((k - window) / 12.0)   # incumbents recover as the promo fades

    # rank incumbents by recent 12-mo volume; forecast the largest few forward (cap for legibility)
    _rank = (_r12.assign(w=_r12.mem_wash_count.fillna(0) + _r12.ret_wash_count.fillna(0))
             .groupby("site_key").w.mean().sort_values(ascending=False))
    shown = _rank.index[:max_incumbents].tolist()

    incumbents: List[Dict[str, Any]] = []
    for k in shown:                                                 # each incumbent = its own forward forecast
        hist = df[df.site_key == k].set_index("date").sort_index()["tot_wash_count"].dropna()
        if hist.empty:
            continue
        yb = np.concatenate([[float(hist.iloc[-1])], forecast_series(hist, 60)])   # m0 = last actual, m1..60 = forecast
        name = str(site.loc[site.site_key == k, "client_name"].iloc[0])
        incumbents.append({
            "site_key": str(k),
            "name": name,
            "expected": [float(v) for v in yb],
            "with_campaign": [float(v) for v in (yb * (1 - steal))],
        })

    return {
        "months": [int(m) for m in months],
        "years": age_periods(months)[0], "quarters": age_periods(months)[1],
        "your_site": {
            "base": [float(v) for v in new_base],
            "with_campaign": [float(v) for v in new_camp],
        },
        "incumbents": incumbents,
        "n_incumbents": int(n_inc),
        "n_shown": len(incumbents),
        "steal_peak": float(steal_peak),
        "campaign": {
            "applied": bool(campaign_on),
            "launch": int(campaign_launch),
            "intensity": float(campaign_intensity),
            "window": int(window),
        },
    }


# ─────────────────────────── 3) campaign snapshot (book_v4 popover) ───────────────────────────
def campaign_snapshot() -> Dict[str, Any]:
    """The book_v4 OPEX / Revenue / Profit / Membership-purchases snapshot (render_campaign_snapshot,
    app.py lines ~508-651) WITHOUT plotting: for each duration bucket (1 / 2 / 3+ months) the MEDIAN
    OPEX / Revenue / Profit / Mem-purchases by month-offset (mfs −3..6) plus n_campaigns and camp_months."""
    data = D.load_campaign_panel()
    campaigns_df = _campaigns_df().copy()

    buckets_out: List[Dict[str, Any]] = []
    if campaigns_df.empty:
        return {"buckets": buckets_out}

    _rd = data.copy()
    _rd["report_date"] = pd.to_datetime(_rd["report_date"])
    _rd["true_opex"] = _rd["cogs"] + _rd["expenses"]
    _rd_by_site = {sk: grp.sort_values("report_date") for sk, grp in _rd.groupby("site_key")}

    def reclassify(d):
        if d == 1:
            return "1 month"
        if d == 2:
            return "2 months"
        return "3+ months"

    campaigns_df["dur_bucket2"] = campaigns_df["duration_months"].apply(reclassify)

    _records: List[Dict[str, Any]] = []
    for _, camp in campaigns_df.iterrows():
        sk = camp["site_key"]
        anchor = pd.to_datetime(camp["campaign_start"])
        bucket = camp["dur_bucket2"]
        dur = int(camp["duration_months"])
        if sk not in _rd_by_site:
            continue
        ts = _rd_by_site[sk].copy()
        ts["mfs"] = (
            (ts["report_date"].dt.year - anchor.year) * 12 +
            (ts["report_date"].dt.month - anchor.month)
        )
        for _, row in ts[(ts["mfs"] >= -3) & (ts["mfs"] <= 6)].iterrows():
            _records.append({
                "bucket": bucket,
                "duration": dur,
                "mfs": int(row["mfs"]),
                "opex": row["true_opex"],
                "revenue": row["total_income"],
                "profit": row["total_income"] - row["true_opex"],
                "mem_purchases": row["mem_purchase_count"],
            })

    snap_df2 = pd.DataFrame(_records)

    BUCKET_CONFIG = {
        "1 month":   {"camp_months": [0],       "title": "1-Month Campaigns"},
        "2 months":  {"camp_months": [0, 1],    "title": "2-Month Campaigns"},
        "3+ months": {"camp_months": [0, 1, 2], "title": "3+ Month Campaigns"},
    }

    for bucket, cfg in BUCKET_CONFIG.items():
        sub = snap_df2[snap_df2["bucket"] == bucket] if len(snap_df2) else snap_df2
        if len(sub) == 0:
            continue
        agg = (sub.groupby("mfs")[["opex", "revenue", "profit", "mem_purchases"]]
               .median()
               .reset_index())
        n_camps = int(campaigns_df[campaigns_df["dur_bucket2"] == bucket].shape[0])
        buckets_out.append({
            "bucket": bucket,
            "title": cfg["title"],
            "n_campaigns": n_camps,
            "camp_months": [int(m) for m in cfg["camp_months"]],
            "mfs": [int(m) for m in agg["mfs"]],
            "opex": [_num(v) for v in agg["opex"]],
            "revenue": [_num(v) for v in agg["revenue"]],
            "profit": [_num(v) for v in agg["profit"]],
            "mem_purchases": [_num(v) for v in agg["mem_purchases"]],
        })

    return {"buckets": buckets_out}


# ─────────────────────────── 4) local campaign evidence ───────────────────────────
_METRIC_COLS = {"mem_share_wash", "mem_wash_count", "ret_wash_count"}


def local_campaign_evidence(lat: float, lon: float, radius_km: float = 20.0,
                            metric: str = "mem_share_wash", max_sites: int = 8,
                            demo: bool = False, express_only: bool = False) -> Dict[str, Any]:
    """The "Real campaigns in this local market" evidence panel (campaign_cluster_panel, app.py ~654-689):
    the up-to-`max_sites` nearest in-radius sites' monthly series for `metric` (one of mem_share_wash |
    mem_wash_count | ret_wash_count) plus each site's detected campaign months. `demo` anonymizes names to
    "Site N" by opening order."""
    col = metric if metric in _METRIC_COLS else "mem_share_wash"
    df, site = D.load_panel(express_only)

    g = site[site.has_coords].copy()
    g["d"] = haversine_km(lat, lon, g.lat.values, g.lon.values)
    keys = g[g.d <= radius_km].nsmallest(max_sites, "d").site_key.tolist()       # the local-market sites (cap for legibility)

    out_sites: List[Dict[str, Any]] = []
    if not keys:
        return {"lat": float(lat), "lon": float(lon), "radius_km": float(radius_km), "metric": col, "sites": out_sites}

    camp = campaign_months_by_site()
    # demo: anonymized "Site N" by opening order (matches app.anon_names); else truncated client_name ([:22])
    if demo:
        sub = site[site.site_key.isin(keys)].sort_values("op_start")
        label = {k: f"Site {i + 1}" for i, k in enumerate(sub.site_key)}
    else:
        label = {k: str(site.loc[site.site_key == k, "client_name"].iloc[0])[:22] for k in keys}

    for k in keys:
        s = df[df.site_key == k].set_index("date")[col].sort_index()
        if s.dropna().empty:
            continue
        out_sites.append({
            "site_key": str(k),
            "name": label.get(k, "site"),
            "x": [d.strftime("%Y-%m-%d") for d in s.index],
            "y": [_num(v) for v in s.values],
            "campaign_months": list(camp.get(str(k), [])),         # ISO date strings of detected promo spikes
        })

    return {"lat": float(lat), "lon": float(lon), "radius_km": float(radius_km), "metric": col, "sites": out_sites}


# ─────────────────────────── helpers ───────────────────────────
def _num(v) -> Optional[float]:
    """float() a value, mapping NaN/inf/None to None (no numpy scalars, no NaN in JSON output)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _pct(x: float) -> str:
    """Render a fraction as a whole-percent string, matching the app's '{x:.0%}' (e.g. 0.63 -> '63%').
    NaN/inf-safe: an unknown share renders 'n/a' (the NaN verdict branch reaches here when no incumbent
    has recent wash data) instead of raising on round(NaN)."""
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{round(x * 100)}%"
