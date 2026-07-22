"""
Local-market engine for the Explore-markets + Forecast backend.

Plot-ready data for the Streamlit app's first two modes (proforma/ui/app.py):

  explore_market(...)        — header counts + role-tagged map markers + per-site monthly series for one
                               metric, for every site within `radius_km` of the pin (the line explorer).
  explore_market_kpis(...)   — the 9 grouped per-site KPI series (washes ×3, revenue ×3, ASP ×2,
                               membership purchases ×1 — always last) for the whole local market —
                               the "Local-market KPIs over time" panels.
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

from proforma.pnl import data as D
from proforma.pnl.data import cm, haversine_km, rank_sites as _rank_sites
from proforma.pnl.demographics import annual_pop_growth, pin_demographics
from proforma.pnl.trend import forecast_series, market_trend

# role -> marker colour (mirrors the Explore-markets map in app.py)
ROLE_COLOR = {"focal": "#e6194B", "entrant": "#f58231", "incumbent": "#5b8db8"}
MIN_MONTHS_RICH = 36   # Explore "rich-history" filter — ≥3 full years, so no half-drawn KPI lines

# the 9 KPIs grouped into 4 figures (col, label, unit) — matches the GROUPS in the Streamlit explore panel.
# Membership purchases stays the LAST group (frontend renders the array in order).
def age_periods(months, n_years: int = 5) -> Tuple[List[int], List[int]]:
    """Relative (year, quarter) labels for an AGE-indexed forecast series where month 0 = the site's
    open month. `year` is the 1-based 12-month block (1..`n_years`) and `quarter` is 1..4 WITHIN that
    year — so the frontend can filter e.g. "Year 2 / Q3". The trailing partial month past the horizon
    (e.g. index 60 of a 61-point 5-year series) folds into the LAST year/Q4, matching `_year_slices`
    and the annual rollups. Returns parallel int lists aligned to `months`."""
    ms = [int(m) for m in months]
    years = [min(m // 12 + 1, n_years) for m in ms]
    quarters = [min((m - (y - 1) * 12) // 3 + 1, 4) for m, y in zip(ms, years)]
    return years, quarters


def calendar_periods(dates) -> Tuple[List[int], List[int]]:
    """Calendar (year, quarter) labels for a DATE-indexed series. `dates` are 'YYYY-MM-DD' strings (or
    Timestamps). `quarter` is the calendar quarter 1..4. Returns parallel int lists aligned to `dates`."""
    ts = pd.to_datetime(list(dates))
    years = [int(d.year) for d in ts]
    quarters = [int((d.month - 1) // 3 + 1) for d in ts]
    return years, quarters


KPI_GROUPS: List[Tuple[str, List[Tuple[str, str, str]]]] = [
    ("Washes", [("tot_wash_count", "Total washes", "count"), ("ret_wash_count", "Retail washes", "count"),
                ("mem_wash_count", "Membership washes", "count")]),
    ("Revenue", [("tot_revenue", "Total revenue ($)", "$"), ("ret_revenue", "Retail revenue ($)", "$"),
                 ("mem_revenue", "Membership revenue ($)", "$")]),
    ("ASPs", [("asp_ret", "ASP per retail wash ($)", "$"), ("asp_mem", "ASP per membership wash ($)", "$")]),
    ("Membership purchases", [("mem_purchase_count", "Membership purchases", "count")]),
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
                   demo: bool = False, express_only: bool = False) -> Dict[str, Any]:
    """The Explore-markets MAP + header counts for the pin's local market (NO time series — that's the
    job of explore_market_kpis, the 9 KPI panels).

    Returns the header metrics (sites-in-market, new-entrants) and the map layers:
      • markers          — the in-market sites, role-tagged focal / entrant / incumbent (the "cluster points").
      • reference_dots   — geographic reference: other rich-history sites ≤50 km of the pin (outside the market).
      • operator_footprint — when an `operator` (client_name) is highlighted, ALL of its sites nationwide.
      • cluster_regions  — demo mode only: shaded local-market footprints instead of exact dots.
    `min_months` keeps rich-history sites (≥3 yrs, matching the Explore view — that's why the count is small).
    """
    df, site = D.load_panel(express_only)
    pool = site[site.n_obs >= min_months] if min_months > 1 else site

    nb_full = _neighbourhood(pool, lat, lon, radius_km)
    n_in_market = len(nb_full)
    n_entrants = int(nb_full.is_entrant.sum()) if n_in_market else 0
    focal_key = _focal_key(nb_full)

    # cap markers for legibility: keep every entrant, fill the rest with the nearest incumbents.
    # Markers come back in the shared cross-endpoint rank order (see _rank_sites) with an explicit
    # `rank`, so the dashboard can slice a consistent "top N" against the KPI panels.
    entrants = nb_full[nb_full.is_entrant]
    n_inc = max(0, max_sites - len(entrants))
    inc = nb_full[~nb_full.is_entrant].nsmallest(n_inc, "dist_km")
    _, rank_of = _rank_sites(df, nb_full, focal_key)
    shown = pd.concat([entrants, inc]).drop_duplicates("site_key")
    shown = shown.iloc[shown.site_key.map(rank_of).argsort(kind="mergesort")]

    markers: List[Dict[str, Any]] = []
    for _, s in shown.iterrows():
        role = "focal" if s.site_key == focal_key else ("entrant" if s.is_entrant else "incumbent")
        markers.append({
            "site_key": s.site_key, "name": None if demo else s.client_name,
            "lat": float(s.lat), "lon": float(s.lon), "dist_km": round(float(s.dist_km), 2),
            "op_start": s.op_start.strftime("%Y-%m") if pd.notna(s.op_start) else None,
            "role": role, "is_entrant": bool(s.is_entrant), "color": ROLE_COLOR[role],
            "rank": rank_of[s.site_key], "n_months": int(s.n_obs),
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
                        min_months: int = MIN_MONTHS_RICH, demo: bool = False,
                        express_only: bool = False) -> Dict[str, Any]:
    """The 9 grouped per-site KPI series (Washes ×3 / Revenue ×3 / ASP ×2 / Membership purchases ×1,
    always the LAST group) for the whole local market — every in-radius site with ≥`min_months` of
    history. Mirrors the "Local-market KPIs over time" panels. Membership ASP divides by PURCHASES
    (one purchase funds many washes), matching the Streamlit chart."""
    df, site = D.load_panel(express_only)
    pool = site[site.n_obs >= min_months] if min_months > 1 else site
    nb_full = _neighbourhood(pool, lat, lon, radius_km)
    focal_key = _focal_key(nb_full)
    ckeys = nb_full.site_key.tolist()

    sub = df[df.site_key.isin(ckeys)].copy()
    sub["asp_ret"] = sub.ret_revenue / sub.ret_wash_count.replace(0, np.nan)        # retail ASP = revenue ÷ retail washes
    sub["asp_mem"] = sub.mem_revenue / sub.mem_purchase_count.replace(0, np.nan)    # membership ASP = revenue ÷ PURCHASES (matches the Streamlit chart)

    # anonymized "Site N" labels by opening order (demo), else client_name
    order_for_label = nb_full.sort_values("op_start").site_key.tolist()
    anon = {k: f"Site {i + 1}" for i, k in enumerate(order_for_label)}
    name_of = {k: (anon[k] if demo else str(site.loc[site.site_key == k, "client_name"].iloc[0])) for k in ckeys}

    # even monthly grid per site, reused across groups. Sites and series come back in the shared
    # cross-endpoint rank order (focal first, then longest/largest history — see _rank_sites) with an
    # explicit `rank`, so the dashboard can show a consistent "top N + show more" across every panel.
    order, rank_of = _rank_sites(df, nb_full, focal_key)
    entrant_of = dict(zip(nb_full.site_key, nb_full.is_entrant))
    n_obs_of = dict(zip(nb_full.site_key, nb_full.n_obs))
    gframes: Dict[str, pd.DataFrame] = {}
    for k in order:
        g = sub[sub.site_key == k].set_index("date").sort_index()
        gframes[k] = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="MS")) if len(g) else g

    sites_meta = [
        {"site_key": k, "name": name_of.get(k), "is_focal": k == focal_key,
         "is_entrant": bool(entrant_of.get(k, False)),
         "rank": rank_of[k], "n_months": int(n_obs_of[k]),
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
                             "is_entrant": bool(entrant_of.get(k, False)), "rank": rank_of[k],
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
                       radius_km: float = 20.0,
                       model_kind: str = "et",
                       express_only: bool = False,
                       use_super: bool = False, open_year: Optional[int] = None,
                       pay_stations: Optional[str] = None, vacuum_slots: Optional[str] = None,
                       lot_type: Optional[str] = None, traffic_count: Optional[float] = None,
                       factors: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """The cold-start 5-yr monthly trajectory for a pin + the local-market per-component trends it used.

    Learns this market's membership / retail trends from the ≤`radius_km` neighbours' own series
    (composition-robust), then runs the cold-start model with those drifts plus the user's extra sliders.
    Returns (traj_df, info, trends). `trends` carries the RAW market trend (no slider) + the applied
    slider fractions, so callers can reuse them consistently (e.g. the market baseline forecast uses the
    raw trend while the entrant's trajectory uses raw+slider). Shared by pinpoint_forecast / pnl / campaign.

    `use_super` switches the level to Model 5 (SUPER, mirroring the Streamlit sidebar): the calibrated
    ridge level when all 4 site inputs (pay_stations, vacuum_slots, lot_type, traffic_count) are given,
    else the pin-only calibration — plus per-operating-year debias, and the bounded score-sheet factor
    multiplier when `factors` ({factor: option} — areaProfile, nearestCompetition, weeklyHoursCategory,
    siteAccessibility, entranceStackUpArea, visibility, trafficSpeed; ridge-covered ones skipped) is
    supplied. A manual `plateau_override` wins outright and skips the whole layer, same as the UI.
    """
    df, site = D.load_panel(express_only)
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

    # express only: restrict the model's LOCAL-MATURED level anchor to the express sites, exactly as
    # _pinpoint_forecast.drop_pin_ui does (`anchor_keys = set(site.site_key) if express_only else None`).
    # The LightGBM neighbour features stay on the full site set so they remain on the training
    # distribution; only the anchor is scoped. None => the model's default behaviour, unchanged.
    anchor_keys = set(site.site_key) if express_only else None

    traj, info = cm.predict_site(
        lat, lon, brand=brand, plateau_override=(plateau_override or None),
        annual_mem_growth=mem_g + gm, annual_ret_change=ret_g + gr,
        mem_growth_band=(mem_lo + gm, mem_hi + gm), ret_change_band=(ret_lo + gr, ret_hi + gr),
        horizon=horizon_months, art=art, model_kind=model_kind,   # default "et" = Model 3 (most accurate)
        anchor_keys=anchor_keys,
    )
    # Model 5 (SUPER): calibrated level + per-year debias + optional score-sheet factor multiplier.
    # A manual plateau override wins outright (skip the layer), exactly as the Streamlit sidebar does.
    if use_super and not plateau_override:
        from proforma.models import super_ensemble as se
        traj, info = se.apply_super(
            traj, info, open_year=int(open_year) if open_year else int(pd.Timestamp.now().year),
            pay_stations=pay_stations, vacuum_slots=vacuum_slots,
            lot_type=lot_type, traffic_count=traffic_count, extra_factors=factors)
    trends = {"mem_g": mem_g, "mem_lo": mem_lo, "mem_hi": mem_hi,
              "ret_g": ret_g, "ret_lo": ret_lo, "ret_hi": ret_hi,
              "gm": gm, "gr": gr, "neighbour_keys": keys}
    return traj, info, trends


# ─────────────────────────── tab 2a: the new site's 5-year trajectory ───────────────────────────
def pinpoint_forecast(lat: float, lon: float, brand: Optional[str], plateau_override: Optional[float],
                      mem_growth_pct: float, ret_growth_pct: float, horizon_months: int,
                      express_only: bool = False,
                      use_super: bool = False, open_year: Optional[int] = None,
                      pay_stations: Optional[str] = None, vacuum_slots: Optional[str] = None,
                      lot_type: Optional[str] = None, traffic_count: Optional[float] = None,
                      factors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The NEW SITE's own 5-year monthly trajectory (the "Predicted 5-year trajectory" chart) + the summary
    KPI cards. Total / membership / retail washes with a P10–P90 band.

    `mem_growth_pct` / `ret_growth_pct` are extra yr3-5 drift (%/yr) on top of the market's own trend.
    `use_super` (+ the 4 site inputs, `open_year`, and the score-sheet `factors`) switches the level to
    Model 5 — see compute_trajectory; the summary then gains model/level provenance + the itemised
    `factor_adjustment`. The whole-market history+forecast plot is a SEPARATE endpoint — see
    `market_forecast`.
    """
    traj, info, trends = compute_trajectory(lat, lon, brand, plateau_override,
                                            mem_growth_pct, ret_growth_pct, horizon_months,
                                            express_only=express_only,
                                            use_super=use_super, open_year=open_year,
                                            pay_stations=pay_stations, vacuum_slots=vacuum_slots,
                                            lot_type=lot_type, traffic_count=traffic_count,
                                            factors=factors)
    mem_g, ret_g = trends["mem_g"], trends["ret_g"]
    g = traj.set_index("month")
    tj_years, tj_quarters = age_periods(g.index)
    summary: Dict[str, Any] = {
        "plateau_med": float(info["plateau_med"]), "plateau_lo": float(info["plateau_lo"]),
        "plateau_hi": float(info["plateau_hi"]), "mem_share": float(info["mem_share"]),
        "n_neighbours_20km": int(info["n_neighbours_20km"]), "brand_known": bool(info["brand_known"]),
        "ramp_source": info["ramp_source"], "region": info.get("region"), "state": info.get("state"),
        "mem_growth": float(mem_g + trends["gm"]), "ret_growth": float(ret_g + trends["gr"]),
    }
    if use_super:
        summary.update({
            "model": "super" if info.get("level_source") else "coldstart_m3",  # override skips the layer
            "level_source": info.get("level_source"),          # "user_inputs" | "pin_only" | None
            "super_level": (float(info["super_level"]) if info.get("super_level") is not None else None),
            "factor_adjustment": info.get("factor_adjustment"),  # itemised score-sheet multiplier, or None
        })
    return {
        "lat": lat, "lon": lon, "brand": brand,
        "summary": summary,
        "trajectory": {
            "months": [int(m) for m in g.index],
            "years": tj_years, "quarters": tj_quarters,
            "total_med": [float(v) for v in g.total_med], "total_lo": [float(v) for v in g.total_lo],
            "total_hi": [float(v) for v in g.total_hi],
            "mem_med": [float(v) for v in g.mem_med], "ret_med": [float(v) for v in g.ret_med],
        },
    }


# ─────────────────────────── tab 2b: total local-market wash count (history + forecast) ───────────────────────────
_PLACES_CACHE: Dict[Tuple[float, float], Tuple[float, List[dict]]] = {}
_PLACES_TTL_S = 3600.0   # 1 h, same freshness as the 🛰️ Sitewise wash layer (st.cache_data ttl)


def _places_car_washes(lat: float, lon: float) -> List[dict]:
    """Real car washes (any type) within ~5 mi of the pin, nearest first: [{name, lat, lng, m}].
    Reuses the existing Places(New) wrapper (`app.core.places.search_nearby` — the Competition
    Coverage source) with the same 5 mi ring the 🛰️ Sitewise wash layer shows. [] when no
    GOOGLE_MAPS_API_KEY is configured or the call fails — treated as "no ground truth", never an
    error. In-process cache per ~110 m pin, 1 h TTL — so one 3/6/12-mi radius sweep bills one
    Places call, but a long-lived server never serves a stale wash list."""
    import time
    from app.core import common as calib
    key = calib.GOOGLE_MAPS_API_KEY or ""
    if not key:
        return []
    ck = (round(lat, 3), round(lon, 3))
    hit = _PLACES_CACHE.get(ck)
    if hit is not None and (time.monotonic() - hit[0]) < _PLACES_TTL_S:
        return hit[1]
    out: List[dict] = []
    try:
        from app.core.places.search_nearby import find_nearby_places
        resp = find_nearby_places(key, lat, lon, radius_miles=5.0, included_types=["car_wash"],
                                  max_results=20, rank_preference="DISTANCE") or {}
        for p in (resp.get("places") or []):
            loc = p.get("location") or {}
            plat, plon = loc.get("latitude"), loc.get("longitude")
            if plat is None or plon is None:
                continue
            m = float(haversine_km(lat, lon, np.array([plat]), np.array([plon]))[0] * 1000.0)
            out.append({"name": (p.get("displayName") or {}).get("text") or "—",
                        "lat": plat, "lng": plon, "m": round(m)})
        out.sort(key=lambda x: x["m"])
    except Exception:
        out = []
    _PLACES_CACHE[ck] = (time.monotonic(), out)
    return out


def _resolve_market_multiplier(lat: float, lon: float, n_own: int, radius_km: float,
                               market_multiplier: Optional[float],
                               own_coords: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """The overall-market ÷ Sonny's-only multiplier for the 🌐 true-market view, sized for THIS radius.

    Resolution ladder (mirrors the Streamlit 🌐 expander):
      1. a caller-supplied `market_multiplier` wins outright (deterministic, cache-friendly);
      2. else ONE small express-only LLM call (`market_scale_multiplier`) — grounded on the REAL car
         washes Google Places sees near the pin (the same ~5 mi layer the 🛰️ Sitewise tab shows; the
         prompt tells the model to extrapolate that observed set to the full `radius_km`), with the
         radius-scoped Sonny's site count;
      3. else (no LLM configured / it fails) the Places-count ratio (Sonny's + unseen washes) ÷ Sonny's —
         a lower bound, source "places_ratio";
      4. else ×1 with source "unavailable".
    Always returns {multiplier, source, confidence, total_express_tunnels, n_places_washes,
    n_places_non_sonnys}. `own_coords` (lat/lon of the in-radius Sonny's sites) is what separates a
    Places wash we already track (<0.25 km from a panel site) from an unseen competitor."""
    gw = _places_car_washes(lat, lon)                  # [] when no key / API down — never fatal
    gw_new = gw
    if len(gw) and own_coords is not None and len(own_coords):
        gw_new = [w for w in gw
                  if float(haversine_km(w["lat"], w["lng"],
                                        own_coords.lat.values, own_coords.lon.values).min()) > 0.25]
    grounding = {"n_places_washes": len(gw), "n_places_non_sonnys": len(gw_new)}

    if market_multiplier is not None:
        return {"multiplier": float(min(max(market_multiplier, 1.0), 25.0)), "source": "user",
                "confidence": None, "total_express_tunnels": None, **grounding}
    try:
        import os as _os
        from app.pnl_analysis.insights.llm import insights_llm_ready as _llm_ready
        from app.pnl_analysis.insights.location_poc import market_scale_multiplier as _mult_llm
        _backend = _os.getenv("INSIGHTS_LLM_BACKEND", "azure").strip().lower()
        if _llm_ready(_backend):
            _nearby = [{"name": w.get("name"), "distance_miles": round((w.get("m") or 0) / 1609.34, 1)}
                       for w in gw]
            ai = _mult_llm(lat, lon, n_client_sites=max(n_own, 1), radius_km=radius_km,
                           nearby_washes=_nearby, backend=_backend)
            if ai.get("multiplier"):
                return {"multiplier": float(ai["multiplier"]), "source": "llm",
                        "confidence": ai.get("confidence"),
                        "total_express_tunnels": ai.get("total"), **grounding}
    except Exception:                                  # LLM must never break the deterministic forecast
        pass
    if n_own > 0 and len(gw_new):                      # Places-count ratio — a lower bound (~5 mi ring)
        return {"multiplier": float(min((n_own + len(gw_new)) / n_own, 25.0)), "source": "places_ratio",
                "confidence": None, "total_express_tunnels": None, **grounding}
    return {"multiplier": 1.0, "source": "unavailable", "confidence": None,
            "total_express_tunnels": None, **grounding}


_MILE_KM = 1.609344
MARKET_RADII_MILES = (3, 6, 12)   # the distance filter the frontend offers for the market plot


def market_forecast(lat: float, lon: float, brand: Optional[str], plateau_override: Optional[float],
                    mem_growth_pct: float, ret_growth_pct: float, horizon_months: int,
                    express_only: bool = False, use_competitors: bool = False,
                    market_multiplier: Optional[float] = None,
                    use_population_growth: bool = False,
                    radius_miles: Optional[float] = None,
                    use_super: bool = False, open_year: Optional[int] = None,
                    pay_stations: Optional[str] = None, vacuum_slots: Optional[str] = None,
                    lot_type: Optional[str] = None, traffic_count: Optional[float] = None,
                    factors: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The TOTAL LOCAL-MARKET wash count: actual history + 5-year forecast (the "Total local-market wash
    count" growth plot). The forecast carries every neighbour forward at the local trend, subtracts the new
    site's (distance-learned, retail) cannibalization phased over the first year, and adds the entrant's own
    journey. Returns history + four forecast series; the cannibalization params themselves are internal.

    🌐 True-market view (all opt-in; defaults leave the response byte-identical):
      • `use_competitors` — adds a SECOND "overall market" trajectory (history + forecast) = the Sonny's-only
        lines × an express-tunnel multiplier, so the chart shows BOTH the Sonny's-clients market and the full
        competitive market; the gap between them is the non-Sonny's volume — the opportunity a new entrant
        can capture. The multiplier comes from `market_multiplier` when supplied (deterministic), else one
        small express-only LLM call (the same one the Streamlit 🌐 expander makes; NOTE: that path is
        non-deterministic, and every request computes LIVE — the DB response cache was removed 2026-07-22 —
        so it is re-sized on each call; pass `market_multiplier` for a stable number).
      • `use_population_growth` — compounds the trade area's projected '25→'30 population growth (council
        site-wise CSV, nearest covered site at 3→6→9 mi — same source as /site-factors) onto every market
        forecast line. The entrant's own journey is NEVER scaled by either factor.
    When either flag is on, the response gains a `market_view` block carrying the multiplier provenance, the
    population record used and `population_per_wash_site` = population ÷ (Sonny's sites + non-Sonny's sites
    + 1 for the new entrant) — the headline people-per-tunnel saturation metric.

    `radius_miles` (3 / 6 / 12) filters the LOCAL MARKET by distance: every plotted number — the history
    sum, the neighbour trends, the forecast lines, cannibalization, the multiplier's site count — is
    computed over the sites within that ring of the pin. None (the default) keeps the legacy 20 km market,
    byte-identical to before. The response echoes `radius_miles` / `radius_km` / `n_sites_in_market`.
    """
    df, site = D.load_panel(express_only)
    art = D.load_model()
    radius_km = float(radius_miles) * _MILE_KM if radius_miles else 20.0
    traj, info, trends = compute_trajectory(lat, lon, brand, plateau_override,
                                            mem_growth_pct, ret_growth_pct, horizon_months,
                                            radius_km=radius_km,
                                            express_only=express_only,
                                            use_super=use_super, open_year=open_year,
                                            pay_stations=pay_stations, vacuum_slots=vacuum_slots,
                                            lot_type=lot_type, traffic_count=traffic_count,
                                            factors=factors)
    keys = trends["neighbour_keys"]
    mem_g, mem_lo, mem_hi = trends["mem_g"], trends["mem_lo"], trends["mem_hi"]
    ret_g, ret_lo, ret_hi = trends["ret_g"], trends["ret_lo"], trends["ret_hi"]

    g = traj.set_index("month")
    _mc = df.groupby("date").size()                                          # robust "now": last month with real coverage
    today = pd.Timestamp(_mc[_mc >= 0.5 * _mc.median()].index.max())          # ignore sparse straggler months (e.g. a lone future row)
    if keys:                                                                  # anchor to THIS local market's last real month so the
        _lm = (df[df.site_key.isin(keys)]                                     # forecast is CONTIGUOUS with history — no None-tail gap
               .groupby("date")[["mem_wash_count", "ret_wash_count"]].sum().sum(axis=1))  # and no month-shifted seasonality
        _lm = _lm[_lm > 0]
        if len(_lm):
            today = min(today, pd.Timestamp(_lm.index.max()))
    H = horizon_months
    fstart = today + pd.DateOffset(months=1)                                  # forecast starts the month AFTER the last data (contiguous)
    fdates = pd.date_range(fstart, periods=H, freq="MS")
    # the NEW SITE opens next CALENDAR month from today (real clock) — never before the forecast starts.
    _now_m = pd.Timestamp.now().normalize().replace(day=1)
    open_date = max(_now_m + pd.DateOffset(months=1), fstart)
    open_off = (open_date.year - fstart.year) * 12 + (open_date.month - fstart.month)   # months from forecast start to opening
    # place the entrant's own journey at its OPENING month within the horizon (0 before it opens)
    _mt = g["total_med"].reindex(range(H)).to_numpy()
    _ml = g["total_lo"].reindex(range(H)).to_numpy()
    _mh = g["total_hi"].reindex(range(H)).to_numpy()
    _n = max(0, H - open_off)
    new_traj = np.zeros(H); new_lo = np.zeros(H); new_hi = np.zeros(H)
    new_traj[open_off:open_off + _n] = _mt[:_n]
    new_lo[open_off:open_off + _n] = _ml[:_n]
    new_hi[open_off:open_off + _n] = _mh[:_n]
    ent_dates = fdates[open_off:]                                             # the new site's OWN timeline (starts at open_date)
    ent_years, ent_quarters = calendar_periods(ent_dates)
    cp = cm.cannib_params(art, lat, lon)

    out: Dict[str, Any] = {
        "lat": lat, "lon": lon, "brand": brand,
        "open_date": open_date.strftime("%Y-%m-%d"),
    }
    if radius_miles:                                   # echo the distance filter behind every plotted number
        out.update({"radius_miles": float(radius_miles), "radius_km": radius_km,
                    "n_sites_in_market": len(keys)})   # (absent on the legacy 20 km default — golden-safe)

    # ── 🌐 true-market view (opt-in): the multiplier, the trade-area demographics and the headline
    #    population-per-tunnel metric. _k / pop_g stay (1.0, 0.0) unless the flags ask otherwise, so the
    #    default response is untouched. ──
    _k, pop_g = 1.0, 0.0
    if use_competitors or use_population_growth:
        n_own = len(keys)
        dem = pin_demographics(lat, lon)
        pop = dem.get("population") if dem.get("found") else None
        pop_growth = dem.get("growth_2030_2025") if dem.get("found") else None
        pop_g_annual = annual_pop_growth(pop_growth) if pop_growth is not None else None
        if use_population_growth and pop_g_annual is not None:
            pop_g = pop_g_annual
        _own_xy = site.loc[site.site_key.isin(keys), ["lat", "lon"]] if keys else None
        mult_info = (_resolve_market_multiplier(lat, lon, n_own, radius_km, market_multiplier,
                                                own_coords=_own_xy)
                     if (use_competitors and n_own > 0) else
                     {"multiplier": 1.0, "source": ("no_neighbours" if use_competitors else "off"),
                      "confidence": None, "total_express_tunnels": None})
        if use_competitors:
            _k = float(mult_info["multiplier"])
        n_non = max(0, int(round(n_own * _k)) - n_own)             # implied non-Sonny's express tunnels
        out["market_view"] = {
            "use_competitors": use_competitors, "use_population_growth": use_population_growth,
            "multiplier": _k, "multiplier_source": mult_info["source"],
            "multiplier_confidence": mult_info["confidence"],
            "total_express_tunnels": mult_info["total_express_tunnels"],
            # Places grounding behind the multiplier (absent counts = 0: no key / no washes seen)
            "n_places_washes": mult_info.get("n_places_washes", 0),
            "n_places_non_sonnys": mult_info.get("n_places_non_sonnys", 0),
            "n_sonnys_sites": n_own, "n_non_sonnys_sites": n_non,
            "population": pop,
            "population_growth_2030_2025": pop_growth,
            "population_growth_annual": pop_g_annual,
            "population_growth_applied": bool(use_population_growth and pop_g),
            "population_source": ({"radius_used_miles": dem.get("radius_used_miles"),
                                   **(dem.get("match") or {})} if dem.get("found") else None),
            # the whiteboard metric: people per wash site once the new entrant opens
            "population_per_wash_site": (float(pop) / (n_own + n_non + 1) if pop else None),
        }

    if keys:
        nb = site[site.has_coords]
        comp = df[df.site_key.isin(keys)].groupby("date")[["mem_wash_count", "ret_wash_count"]].sum()
        idx = pd.date_range(comp.index.min(), today, freq="MS")
        hist_mem = comp["mem_wash_count"].reindex(idx)
        hist_ret = comp["ret_wash_count"].reindex(idx)
        hist = hist_mem.add(hist_ret, fill_value=0)

        base_mem = forecast_series(hist_mem, H, g=mem_g, seasonal=True)
        base_ret = forecast_series(hist_ret, H, g=ret_g, seasonal=True)
        base_fc = base_mem + base_ret
        base_mem_lo = forecast_series(hist_mem, H, g=mem_lo, seasonal=True); base_mem_hi = forecast_series(hist_mem, H, g=mem_hi, seasonal=True)
        base_ret_lo = forecast_series(hist_ret, H, g=ret_lo, seasonal=True); base_ret_hi = forecast_series(hist_ret, H, g=ret_hi, seasonal=True)
        if pop_g:                                                             # 🌐 demographic drift: projected population
            _demog = (1.0 + pop_g) ** (np.arange(1, H + 1) / 12.0)            # growth compounds over the forecast years
            base_mem, base_ret = base_mem * _demog, base_ret * _demog
            base_mem_lo, base_mem_hi = base_mem_lo * _demog, base_mem_hi * _demog
            base_ret_lo, base_ret_hi = base_ret_lo * _demog, base_ret_hi * _demog
            base_fc = base_mem + base_ret

        nb_keyed = nb.set_index("site_key").loc[keys]
        dist_by_key = pd.Series(haversine_km(lat, lon, nb_keyed.lat.values, nb_keyed.lon.values), index=keys)
        rec_ret = (df[df.site_key.isin(keys)].sort_values("date").groupby("site_key").tail(12)
                   .groupby("site_key")["ret_wash_count"].mean())
        cannib_full = float((cm._cannib_ret(dist_by_key.reindex(rec_ret.index).values, cp) * rec_ret.values).sum())
        phase = np.clip((np.arange(H) - open_off + 1) / 12.0, 0.0, 1.0)   # cannibalization ramps from the OPENING month (0 before)

        with_fc = np.clip(base_mem + np.clip(base_ret - cannib_full * phase, 0, None) + new_traj, 0, None)
        with_lo = np.clip(base_mem_lo + np.clip(base_ret_lo - cannib_full * phase, 0, None) + new_lo, 0, None)
        with_hi = np.clip(base_mem_hi + np.clip(base_ret_hi - cannib_full * phase, 0, None) + new_hi, 0, None)

        # 🌐 overall market (Sonny's + non-Sonny's): the Sonny's-only history/forecast scaled ×k — the unseen
        # express competitors lose retail to the entrant too, so cannibalization scales with the market. The
        # entrant's own journey is added once, NEVER scaled. The gap between these lines and the Sonny's-only
        # ones is the non-Sonny's volume — the market opportunity the frontend shades.
        if _k > 1.0:
            mkt_hist = hist * _k
            mkt_base = base_fc * _k
            mkt_with = np.clip(base_mem * _k + np.clip(base_ret * _k - cannib_full * _k * phase, 0, None)
                               + new_traj, 0, None)

        hist_years, hist_quarters = calendar_periods(hist.index)
        fc_years, fc_quarters = calendar_periods(fdates)
        out.update({
            "has_neighbours": True,
            "history": {"dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                        "years": hist_years, "quarters": hist_quarters,
                        "values": [None if pd.isna(v) else float(v) for v in hist.values]},
            "forecast": {"dates": [d.strftime("%Y-%m-%d") for d in fdates],
                         "years": fc_years, "quarters": fc_quarters,
                         # market baseline: seamless dotted continuation of history (starts the month after the last data)
                         "without_new_site": [float(v) for v in base_fc],
                         # with-new-site total + its CI band only EXIST from the opening month (null before open_date)
                         "with_new_site": [None if i < open_off else float(with_fc[i]) for i in range(H)],
                         "band_lo": [None if i < open_off else float(with_lo[i]) for i in range(H)],
                         "band_hi": [None if i < open_off else float(with_hi[i]) for i in range(H)],
                         # the entrant's own journey as a self-contained dated series, starting at open_date
                         "new_entrant_journey": [float(new_traj[i]) for i in range(open_off, H)],
                         "new_entrant_dates": [d.strftime("%Y-%m-%d") for d in ent_dates],
                         "new_entrant_years": ent_years, "new_entrant_quarters": ent_quarters},
            "net_change_year5": float(with_fc[-1] - base_fc[-1]),
        })
        if _k > 1.0:                                   # 🌐 the second, overall-market trajectory (opt-in)
            out["history"]["market_values"] = [None if pd.isna(v) else float(v) for v in mkt_hist.values]
            out["forecast"]["market_without_new_site"] = [float(v) for v in mkt_base]
            out["forecast"]["market_with_new_site"] = [None if i < open_off else float(mkt_with[i])
                                                       for i in range(H)]
    else:
        fc_years, fc_quarters = calendar_periods(fdates)
        out.update({
            "has_neighbours": False,
            "history": {"dates": [], "years": [], "quarters": [], "values": []},
            "forecast": {"dates": [d.strftime("%Y-%m-%d") for d in fdates],
                         "years": fc_years, "quarters": fc_quarters,
                         "without_new_site": [0.0] * H,
                         "with_new_site": [None if i < open_off else float(new_traj[i]) for i in range(H)],
                         "band_lo": [None if i < open_off else float(new_lo[i]) for i in range(H)],
                         "band_hi": [None if i < open_off else float(new_hi[i]) for i in range(H)],
                         "new_entrant_journey": [float(new_traj[i]) for i in range(open_off, H)],
                         "new_entrant_dates": [d.strftime("%Y-%m-%d") for d in ent_dates],
                         "new_entrant_years": ent_years, "new_entrant_quarters": ent_quarters},
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
