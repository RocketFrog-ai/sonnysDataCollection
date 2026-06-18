"""
Local-market engine for the Explore-markets + Forecast backend.

Plot-ready data for the Streamlit app's first two modes (earnest-proforma-2.0/streamlits/app.py):

  explore_market(...)        — header counts + role-tagged map markers + per-site monthly series for one
                               metric, for every site within `radius_km` of the pin (the line explorer).
  explore_market_kpis(...)   — the 6 grouped per-site KPI series (washes ×3, revenue ×3, ASP ×2) for the
                               whole local market — the "Local-market KPIs over time" panels.
  compute_trajectory(...)    — the shared cold-start 5-yr trajectory for a pin (used by pinpoint_forecast,
                               pnl and campaign so every Forecast-tab section agrees on the same curve).
  pinpoint_forecast(...)     — the new site's trajectory + the local market's history-plus-forecast total.
  list_brands(...)           — operator/brand dropdown values for the Forecast tab.

All heavy artifacts come from `data.py` (loaded once, shared) and the trend math from `trend.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.pnl_analysis.modelling import data as D
from app.pnl_analysis.modelling.data import cm, haversine_km
from app.pnl_analysis.modelling.trend import forecast_series, market_trend

# role -> marker colour (mirrors the Explore-markets map in app.py)
ROLE_COLOR = {"focal": "#e6194B", "entrant": "#f58231", "incumbent": "#5b8db8"}
MIN_MONTHS_RICH = 36   # Explore "rich-history" filter — ≥3 full years, so no half-drawn KPI lines

# the 6 KPIs grouped into 3 figures (col, label, unit) — matches the GROUPS in app.py
KPI_GROUPS: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("Washes", [("tot_wash_count", "Total washes", "count"), ("ret_wash_count", "Retail washes", "count"),
                ("mem_wash_count", "Membership washes", "count")]),
    ("Revenue", [("tot_revenue", "Total revenue ($)", "$"), ("ret_revenue", "Retail revenue ($)", "$"),
                 ("mem_revenue", "Membership revenue ($)", "$")]),
    ("ASPs", [("asp_ret", "ASP per wash — retail ($)", "$"), ("asp_mem", "ASP per wash — membership ($)", "$")]),
]


# ─────────────────────────── neighbourhood helpers ───────────────────────────
def _neighbourhood(site: pd.DataFrame, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    """Sites within `radius_km` of the pin, with `dist_km` and an `is_entrant` tag (opened after the
    market's earliest site AND a genuinely-observed, non-left-censored opening)."""
    d = haversine_km(lat, lon, site.lat.values, site.lon.values)
    in_radius = (d <= radius_km) & site.has_coords.values
    nb = site.loc[in_radius].copy()
    nb["dist_km"] = d[in_radius]
    nb = nb.sort_values("op_start")
    if len(nb):
        earliest = nb.op_start.min()
        nb["is_entrant"] = (~nb.left_censored) & (nb.op_start > earliest)
    else:
        nb["is_entrant"] = False
    return nb.reset_index(drop=True)


def _focal_key(nb: pd.DataFrame) -> Optional[str]:
    """The focal new site = the newest entrant, falling back to the nearest site to the pin."""
    if nb.empty:
        return None
    ent = nb[nb.is_entrant]
    if len(ent):
        return str(ent.sort_values("op_start").site_key.iloc[-1])
    return str(nb.sort_values("dist_km").site_key.iloc[0])


def _cluster_regions(site: pd.DataFrame, plat: float, plon: float, max_km: float) -> List[Dict[str, Any]]:
    """Shaded local-market circles (demo mode): each clustered footprint within `max_km` of the pin,
    hugging its own spread (centroid → farthest member, min 2 / cap 20 km). Mirrors add_cluster_regions."""
    area = site[site.has_coords & (site.cluster >= 0)].copy()
    area = area[area.lat.between(-89, 89) & area.lon.between(-179, 179)
                & (area.lat.abs() > 1e-3) & (area.lon.abs() > 1e-3)]
    area["d"] = haversine_km(plat, plon, area.lat.values, area.lon.values)
    area = area[area.d <= max_km]
    out = []
    for cid, cg in area.groupby("cluster"):
        clat, clon = float(cg.lat.mean()), float(cg.lon.mean())
        spread = float(haversine_km(clat, clon, cg.lat.values, cg.lon.values).max())
        r_km = min(max(spread + 1.0, 2.0), 20.0)
        out.append({"cluster": int(cid), "lat": clat, "lon": clon,
                    "radius_km": round(r_km, 2), "n_sites": int(len(cg))})
    return out


# ─────────────────────────── tab 1a: explore market (the map + header counts) ───────────────────────────
def explore_market(lat: float, lon: float, radius_km: float = 20.0, max_sites: int = 10,
                   min_months: int = MIN_MONTHS_RICH, operator: Optional[str] = None,
                   demo: bool = False) -> Dict[str, Any]:
    """The Explore-markets MAP + header counts for the pin's local market (NO time series — that's the
    job of explore_market_kpis, the 8 KPI panels).

    Returns the header metrics (sites-in-market, new-entrants) and the map layers:
      • markers          — the in-market sites, role-tagged focal / entrant / incumbent (the "cluster points").
      • reference_dots   — geographic reference: other rich-history sites ≤50 km of the pin (outside the market).
      • operator_footprint — when an `operator` (client_name) is highlighted, ALL of its sites nationwide.
      • cluster_regions  — demo mode only: shaded local-market footprints instead of exact dots.
    `min_months` keeps rich-history sites (≥3 yrs, matching the Explore view — that's why the count is small).
    """
    df, site = D.load_panel()
    pool = site[site.n_obs >= min_months] if min_months > 1 else site

    nb_full = _neighbourhood(pool, lat, lon, radius_km)
    n_in_market = len(nb_full)
    n_entrants = int(nb_full.is_entrant.sum()) if n_in_market else 0
    focal_key = _focal_key(nb_full)

    # cap markers for legibility: keep every entrant, fill the rest with the nearest incumbents
    entrants = nb_full[nb_full.is_entrant]
    n_inc = max(0, max_sites - len(entrants))
    inc = nb_full[~nb_full.is_entrant].nsmallest(n_inc, "dist_km")
    shown = pd.concat([entrants, inc]).drop_duplicates("site_key").sort_values("op_start")

    markers: List[Dict[str, Any]] = []
    for _, s in shown.iterrows():
        role = "focal" if s.site_key == focal_key else ("entrant" if s.is_entrant else "incumbent")
        markers.append({
            "site_key": s.site_key, "name": None if demo else s.client_name,
            "lat": float(s.lat), "lon": float(s.lon), "dist_km": round(float(s.dist_km), 2),
            "op_start": s.op_start.strftime("%Y-%m") if pd.notna(s.op_start) else None,
            "role": role, "is_entrant": bool(s.is_entrant), "color": ROLE_COLOR[role],
        })

    # geographic reference: rich-history sites ≤50 km of the pin that are NOT already in-market markers
    pool_ref = site[site.n_obs >= MIN_MONTHS_RICH]
    dref = haversine_km(lat, lon, pool_ref.lat.values, pool_ref.lon.values)
    keep_ref = (dref <= 50) & pool_ref.has_coords.values
    ref = pool_ref.loc[keep_ref].copy(); ref["d"] = dref[keep_ref]
    shown_keys = set(shown.site_key)
    reference_dots = [] if demo else [
        {"site_key": r.site_key, "name": r.client_name, "lat": float(r.lat), "lon": float(r.lon),
         "dist_km": round(float(r.d), 2)}
        for _, r in ref.iterrows() if r.site_key not in shown_keys
    ]

    operator_footprint = []
    if operator and not demo:
        of = site[site.has_coords & (site.client_name == operator)]
        operator_footprint = [{"site_key": s.site_key, "name": s.client_name,
                               "lat": float(s.lat), "lon": float(s.lon)} for _, s in of.iterrows()]

    return {
        "lat": lat, "lon": lon, "radius_km": radius_km, "min_months": min_months,
        "n_sites_in_market": int(n_in_market), "n_entrants": n_entrants, "n_shown": len(markers),
        "focal_site_key": focal_key, "operator": operator,
        "map": {
            "center": {"lat": lat, "lon": lon},
            "radius_km": radius_km,
            "markers": markers,
            "reference_dots": reference_dots,
            "cluster_regions": _cluster_regions(site, lat, lon, max_km=200) if demo else [],
            "operator_footprint": operator_footprint,
        },
    }


# ─────────────────────────── tab 1b: local-market KPI panels ───────────────────────────
def explore_market_kpis(lat: float, lon: float, radius_km: float, smoothing: int,
                        min_months: int = MIN_MONTHS_RICH, demo: bool = False) -> Dict[str, Any]:
    """The 6 grouped per-site KPI series (Washes ×3 / Revenue ×3 / ASP ×2) for the whole local market —
    every in-radius site with ≥`min_months` of history. Mirrors the "Local-market KPIs over time" panels."""
    df, site = D.load_panel()
    pool = site[site.n_obs >= min_months] if min_months > 1 else site
    nb_full = _neighbourhood(pool, lat, lon, radius_km)
    focal_key = _focal_key(nb_full)
    ckeys = nb_full.site_key.tolist()

    sub = df[df.site_key.isin(ckeys)].copy()
    sub["asp_ret"] = sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan)
    sub["asp_mem"] = sub.mem_revenue / sub.mem_wash_count.replace(0, np.nan)

    # anonymized "Site N" labels by opening order (demo), else client_name
    order_for_label = nb_full.sort_values("op_start").site_key.tolist()
    anon = {k: f"Site {i + 1}" for i, k in enumerate(order_for_label)}
    name_of = {k: (anon[k] if demo else str(site.loc[site.site_key == k, "client_name"].iloc[0])) for k in ckeys}

    # even monthly grid per site, reused across groups; focal drawn last (legend order)
    order = [k for k in ckeys if k != focal_key] + ([focal_key] if focal_key in ckeys else [])
    entrant_of = dict(zip(nb_full.site_key, nb_full.is_entrant))
    gframes: Dict[str, pd.DataFrame] = {}
    for k in order:
        g = sub[sub.site_key == k].set_index("date").sort_index()
        gframes[k] = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS")) if len(g) else g

    sites_meta = [
        {"site_key": k, "name": name_of.get(k), "is_focal": k == focal_key,
         "is_entrant": bool(entrant_of.get(k, False)),
         "dist_km": round(float(nb_full.loc[nb_full.site_key == k, "dist_km"].iloc[0]), 2),
         "op_start": (nb_full.loc[nb_full.site_key == k, "op_start"].iloc[0].strftime("%Y-%m")
                      if pd.notna(nb_full.loc[nb_full.site_key == k, "op_start"].iloc[0]) else None)}
        for k in order
    ]

    groups: List[Dict[str, Any]] = []
    for gname, panels in KPI_GROUPS:
        gpanels = []
        for col, label, unit in panels:
            pser = []
            for k in order:
                g = gframes[k]
                if not len(g) or col not in g:
                    continue
                y = g[col].rolling(smoothing, center=True, min_periods=1).mean() if (smoothing and smoothing > 1) else g[col]
                if y.dropna().empty:
                    continue
                pser.append({"site_key": k, "name": name_of.get(k), "is_focal": k == focal_key,
                             "is_entrant": bool(entrant_of.get(k, False)),
                             "x": [d.strftime("%Y-%m-%d") for d in g.index],
                             "y": [None if pd.isna(v) else float(v) for v in y.values]})
            gpanels.append({"col": col, "label": label, "unit": unit, "series": pser})
        groups.append({"name": gname, "panels": gpanels})

    return {
        "lat": lat, "lon": lon, "radius_km": radius_km, "smoothing": smoothing, "min_months": min_months,
        "n_sites": len(ckeys), "focal_site_key": focal_key,
        "sites": sites_meta, "groups": groups,
    }


# ─────────────────────────── shared trajectory ───────────────────────────
def compute_trajectory(lat: float, lon: float, brand: Optional[str] = None,
                       plateau_override: Optional[float] = None, mem_growth_pct: float = 0.0,
                       ret_growth_pct: float = 0.0, horizon_months: int = 60,
                       radius_km: float = 20.0) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """The cold-start 5-yr monthly trajectory for a pin + the local-market per-component trends it used.

    Learns this market's membership / retail trends from the ≤`radius_km` neighbours' own series
    (composition-robust), then runs the cold-start model with those drifts plus the user's extra sliders.
    Returns (traj_df, info, trends). `trends` carries the RAW market trend (no slider) + the applied
    slider fractions, so callers can reuse them consistently (e.g. the market baseline forecast uses the
    raw trend while the entrant's trajectory uses raw+slider). Shared by pinpoint_forecast / pnl / campaign.
    """
    df, site = D.load_panel()
    art = D.load_model()
    gm, gr = mem_growth_pct / 100.0, ret_growth_pct / 100.0

    nb = site[site.has_coords]
    d = haversine_km(lat, lon, nb.lat.values, nb.lon.values)
    keys = nb.site_key[(d <= radius_km) & (d > 1e-6)].tolist()
    if keys:
        sub = df[df.site_key.isin(keys)]
        pm = sub.pivot_table(index="date", columns="site_key", values="mem_wash_count")
        pr = sub.pivot_table(index="date", columns="site_key", values="ret_wash_count")
        mem_g, mem_lo, mem_hi = market_trend(pm)
        ret_g, ret_lo, ret_hi = market_trend(pr)
    else:
        mem_g = mem_lo = mem_hi = ret_g = ret_lo = ret_hi = 0.0

    traj, info = cm.predict_site(
        lat, lon, brand=brand, plateau_override=(plateau_override or None),
        annual_mem_growth=mem_g + gm, annual_ret_change=ret_g + gr,
        mem_growth_band=(mem_lo + gm, mem_hi + gm), ret_change_band=(ret_lo + gr, ret_hi + gr),
        horizon=horizon_months, art=art,
    )
    trends = {"mem_g": mem_g, "mem_lo": mem_lo, "mem_hi": mem_hi,
              "ret_g": ret_g, "ret_lo": ret_lo, "ret_hi": ret_hi,
              "gm": gm, "gr": gr, "neighbour_keys": keys}
    return traj, info, trends


# ─────────────────────────── tab 2a: the new site's 5-year trajectory ───────────────────────────
def pinpoint_forecast(lat: float, lon: float, brand: Optional[str], plateau_override: Optional[float],
                      mem_growth_pct: float, ret_growth_pct: float, horizon_months: int) -> Dict[str, Any]:
    """The NEW SITE's own 5-year monthly trajectory (the "Predicted 5-year trajectory" chart) + the summary
    KPI cards. Total / membership / retail washes with a P10–P90 band.

    `mem_growth_pct` / `ret_growth_pct` are extra yr3-5 drift (%/yr) on top of the market's own trend.
    The whole-market history+forecast plot is a SEPARATE endpoint — see `market_forecast`.
    """
    traj, info, trends = compute_trajectory(lat, lon, brand, plateau_override,
                                            mem_growth_pct, ret_growth_pct, horizon_months)
    mem_g, ret_g = trends["mem_g"], trends["ret_g"]
    g = traj.set_index("month")
    return {
        "lat": lat, "lon": lon, "brand": brand,
        "summary": {
            "plateau_med": float(info["plateau_med"]), "plateau_lo": float(info["plateau_lo"]),
            "plateau_hi": float(info["plateau_hi"]), "mem_share": float(info["mem_share"]),
            "n_neighbours_20km": int(info["n_neighbours_20km"]), "brand_known": bool(info["brand_known"]),
            "ramp_source": info["ramp_source"], "region": info.get("region"), "state": info.get("state"),
            "mem_growth": float(mem_g + trends["gm"]), "ret_growth": float(ret_g + trends["gr"]),
        },
        "trajectory": {
            "months": [int(m) for m in g.index],
            "total_med": [float(v) for v in g.total_med], "total_lo": [float(v) for v in g.total_lo],
            "total_hi": [float(v) for v in g.total_hi],
            "mem_med": [float(v) for v in g.mem_med], "ret_med": [float(v) for v in g.ret_med],
        },
    }


# ─────────────────────────── tab 2b: total local-market wash count (history + forecast) ───────────────────────────
def market_forecast(lat: float, lon: float, brand: Optional[str], plateau_override: Optional[float],
                    mem_growth_pct: float, ret_growth_pct: float, horizon_months: int) -> Dict[str, Any]:
    """The TOTAL LOCAL-MARKET wash count: actual history + 5-year forecast (the "Total local-market wash
    count" growth plot). The forecast carries every neighbour forward at the local trend, subtracts the new
    site's (distance-learned, retail) cannibalization phased over the first year, and adds the entrant's own
    journey. Returns history + four forecast series; the cannibalization params themselves are internal.
    """
    df, site = D.load_panel()
    art = D.load_model()
    traj, info, trends = compute_trajectory(lat, lon, brand, plateau_override,
                                            mem_growth_pct, ret_growth_pct, horizon_months)
    keys = trends["neighbour_keys"]
    mem_g, mem_lo, mem_hi = trends["mem_g"], trends["mem_lo"], trends["mem_hi"]
    ret_g, ret_lo, ret_hi = trends["ret_g"], trends["ret_lo"], trends["ret_hi"]

    g = traj.set_index("month")
    today = pd.Timestamp(df.date.max())
    H = horizon_months
    fdates = pd.date_range(today + pd.DateOffset(months=1), periods=H, freq="MS")
    new_traj = g["total_med"].reindex(range(H)).to_numpy()
    new_lo = g["total_lo"].reindex(range(H)).to_numpy()
    new_hi = g["total_hi"].reindex(range(H)).to_numpy()
    cp = cm.cannib_params(art, lat, lon)

    out: Dict[str, Any] = {
        "lat": lat, "lon": lon, "brand": brand,
        "open_date": (today + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
    }

    if keys:
        nb = site[site.has_coords]
        comp = df[df.site_key.isin(keys)].groupby("date")[["mem_wash_count", "ret_wash_count"]].sum()
        idx = pd.date_range(comp.index.min(), today, freq="MS")
        hist_mem = comp["mem_wash_count"].reindex(idx)
        hist_ret = comp["ret_wash_count"].reindex(idx)
        hist = hist_mem.add(hist_ret, fill_value=0)

        base_mem = forecast_series(hist_mem, H, g=mem_g)
        base_ret = forecast_series(hist_ret, H, g=ret_g)
        base_fc = base_mem + base_ret
        base_mem_lo = forecast_series(hist_mem, H, g=mem_lo); base_mem_hi = forecast_series(hist_mem, H, g=mem_hi)
        base_ret_lo = forecast_series(hist_ret, H, g=ret_lo); base_ret_hi = forecast_series(hist_ret, H, g=ret_hi)

        nb_keyed = nb.set_index("site_key").loc[keys]
        dist_by_key = pd.Series(haversine_km(lat, lon, nb_keyed.lat.values, nb_keyed.lon.values), index=keys)
        rec_ret = (df[df.site_key.isin(keys)].sort_values("date").groupby("site_key").tail(12)
                   .groupby("site_key")["ret_wash_count"].mean())
        cannib_full = float((cm._cannib_ret(dist_by_key.reindex(rec_ret.index).values, cp) * rec_ret.values).sum())
        phase = np.minimum(1.0, np.arange(1, H + 1) / 12.0)

        with_fc = np.clip(base_mem + np.clip(base_ret - cannib_full * phase, 0, None) + new_traj, 0, None)
        with_lo = np.clip(base_mem_lo + np.clip(base_ret_lo - cannib_full * phase, 0, None) + new_lo, 0, None)
        with_hi = np.clip(base_mem_hi + np.clip(base_ret_hi - cannib_full * phase, 0, None) + new_hi, 0, None)

        out.update({
            "has_neighbours": True,
            "history": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                        "values": [None if pd.isna(v) else float(v) for v in hist.values]},
            "forecast": {"dates": [d.strftime("%Y-%m-%d") for d in fdates],
                         "with_new_site": [float(v) for v in with_fc],
                         "without_new_site": [float(v) for v in base_fc],
                         "band_lo": [float(v) for v in with_lo], "band_hi": [float(v) for v in with_hi],
                         "new_entrant_journey": [float(v) for v in new_traj]},
            "net_change_year5": float(with_fc[-1] - base_fc[-1]),
        })
    else:
        out.update({
            "has_neighbours": False,
            "history": {"dates": [], "values": []},
            "forecast": {"dates": [d.strftime("%Y-%m-%d") for d in fdates],
                         "with_new_site": [float(v) for v in new_traj], "without_new_site": [0.0] * H,
                         "band_lo": [float(v) for v in new_lo], "band_hi": [float(v) for v in new_hi],
                         "new_entrant_journey": [float(v) for v in new_traj]},
            "net_change_year5": float(new_traj[-1]),
        })
    return out


# ─────────────────────────── brand / operator lookups ───────────────────────────
def list_brands() -> Dict[str, Any]:
    """Operator/brand dropdown values for the Forecast tab. `client_id` is the value the model keys on
    (its leave-one-out mature volume is the strongest plateau predictor); `client_name` is the label."""
    df, site = D.load_panel()
    art = D.load_model()
    known = set(str(k) for k in art.get("brand_mean", {}).keys())
    g = (site.dropna(subset=["client_id"]).groupby("client_id")
         .agg(client_name=("client_name", "first"), n_sites=("site_key", "nunique")).reset_index())
    brands = [{"client_id": str(r.client_id), "client_name": r.client_name,
               "n_sites": int(r.n_sites), "model_known": str(r.client_id) in known}
              for _, r in g.sort_values("n_sites", ascending=False).iterrows()]
    return {"n_brands": len(brands), "brands": brands}


def list_operators() -> Dict[str, Any]:
    """Operator/brand NAMES for the Explore-markets highlight dropdown (client_name, alphabetical)."""
    df, site = D.load_panel()
    names = sorted(site.client_name.dropna().unique().tolist())
    return {"n_operators": len(names), "operators": names}
