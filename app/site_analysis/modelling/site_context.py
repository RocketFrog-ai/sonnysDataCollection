"""
Synchronous lat/lon site-analysis — the FastAPI counterpart to the Streamlit "🔎 Site analysis" page.

Given a (lat, lon) pin (the same one Explore markets & Forecast use) this pulls live location
context around the site in ONE synchronous call (no celery, no task polling):
  • weather / climate     (Open-Meteo)
  • competing car washes  (Google Places, ≤4 mi driving)
  • retail anchors        (Google Places, ≤3 mi: warehouse clubs, big box, grocery, food)
  • gas stations          (Google Places, ≤3 mi driving)

It then assembles the exact same per-dimension data the page shows — headline metrics, map markers,
and rule-based insights (plus optional internal-LLM write-ups) — into a single JSON-serializable dict
that the backend can serve directly. This is the lat/lon counterpart to the async, celery-wrapped
app/site_analysis/modelling/site_analysis.py (which keys on a geocoded address).

The maths, thresholds, windows and fallbacks are ported verbatim from
earnest-proforma-2.0/streamlits/site_analysis_page.py — see render()/fetch_features()/build_markers()
and the rule_* functions there. The only changes are: no streamlit/folium/plotly, no @st.cache_data,
and all numbers coerced to plain float/int/None (never NaN) so the result is JSON-serializable.
"""
from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.utils import common as calib
from app.site_analysis.features.active.nearbyGasStations.get_nearby_gas_stations import get_nearby_gas_stations
from app.site_analysis.features.active.nearbyRetailers.get_nearby_retail_anchors import get_nearby_retail_anchors
from app.site_analysis.features.active.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
from app.site_analysis.features.active.weather.open_meteo import fetch_climate_for_site, get_default_weather_range
from app.site_analysis.server.config import (
    GAS_RADIUS_FAR_MILES,
    RETAIL_RADIUS_FAR_MILES,
    WEATHER_METRIC_CONFIG,
    WEATHER_METRIC_DISPLAY,
    get_weather_metric_value_from_climate,
    is_high_traffic_gas_brand,
)

COMPETITOR_RADIUS_MILES = 4.0
METERS_PER_MILE = 1609.34


# ─────────────────────────── helpers (JSON-safe coercion) ───────────────────────────
def _f(x: Any) -> Optional[float]:
    """Coerce to a plain float, mapping None/NaN/non-numeric to None (never emit NaN)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _i(x: Any) -> Optional[int]:
    """Coerce to a plain int, mapping None/NaN/non-numeric to None."""
    v = _f(x)
    return None if v is None else int(v)


# ───────────────────────── data layer (app/site_analysis) ─────────────────────────
def fetch_features(lat: float, lon: float) -> dict:
    """Parallel external fetch around (lat, lon): {climate, gas_stations, retail_anchors, competitors_data}.

    Mirrors app/site_analysis fetch_all_features, calling the feature fetchers directly. Synchronous
    (one ThreadPoolExecutor, blocks until all four finish). Each fetcher degrades to an empty
    container on missing API key or failure, so the page never errors on partial data.
    """
    start_date, end_date = get_default_weather_range()
    api_key = calib.GOOGLE_MAPS_API_KEY or ""

    def _climate():
        out = fetch_climate_for_site(lat, lon, start_date=start_date, end_date=end_date)
        return out if out and not out.get("error") else {}

    def _gas():
        if not api_key:
            return []
        try:
            return get_nearby_gas_stations(api_key, lat, lon, radius_miles=GAS_RADIUS_FAR_MILES,
                                           max_results=20, fetch_place_details=False) or []
        except Exception:
            return []

    def _retail():
        if not api_key:
            return {}
        try:
            return get_nearby_retail_anchors(api_key, lat, lon, radius_miles=RETAIL_RADIUS_FAR_MILES) or {}
        except Exception:
            return {}

    def _competitors():
        if not api_key:
            return {}
        try:
            return get_nearby_competitors(api_key, lat, lon, radius_miles=COMPETITOR_RADIUS_MILES,
                                          fetch_place_details=False) or {}
        except Exception:
            return {}

    out = {"climate": {}, "gas_stations": [], "retail_anchors": {}, "competitors_data": {}}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_climate): "climate", ex.submit(_gas): "gas_stations",
                ex.submit(_retail): "retail_anchors", ex.submit(_competitors): "competitors_data"}
        for fut in as_completed(futs):
            try:
                out[futs[fut]] = fut.result()
            except Exception:
                pass
    return out


def _llm_reachable(timeout: float = 2.0) -> bool:
    """Fast socket pre-check on the internal LLM host:port — so requesting AI summaries never hangs
    on the client's long retry timeout when the endpoint is down (e.g. off-network). Returns False
    when no LLM URL is configured or the host/port can't be opened within `timeout` seconds."""
    url = calib.LLM_REALTIME_URL or ""
    if not url:
        return False
    p = urlparse(url)
    host, port = p.hostname, (p.port or (443 if p.scheme == "https" else 80))
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ai_summaries(fetched: dict) -> dict:
    """Internal-LLM write-ups per dimension. Returns {} when the LLM is unreachable (fast pre-check).

    The AI summaries module is imported lazily inside the reachable branch (mirrors the reference) so
    that requesting AI never imports/initialises the LLM client when the endpoint is down.
    """
    if not _llm_reachable():
        return {}
    from app.site_analysis.modelling.ai import (
        summarize_weather, summarize_competition, summarize_retail, summarize_gas,
    )
    return {
        "weather": summarize_weather(fetched.get("climate") or {}),
        "competition": summarize_competition(fetched.get("competitors_data") or {}),
        "retail": summarize_retail(fetched.get("retail_anchors") or {}),
        "gas": summarize_gas(fetched.get("gas_stations") or []),
    }


def build_markers(lat, lon, fetched, address=None):
    """Origin + gas + competitor + retail markers (plain dicts; lat/lon already on the fetched items).

    Backs the map on the page. `address` is the origin pin label (passed None in demo mode to hide it).
    """
    markers = [{"category": "origin", "name": "Site", "lat": _f(lat), "lon": _f(lon),
                "distance_miles": 0.0, "address": address}]
    for s in (fetched.get("gas_stations") or []):
        if s.get("latitude") is not None and s.get("longitude") is not None:
            markers.append({"category": "gas_station", "name": s.get("name"),
                            "lat": _f(s["latitude"]), "lon": _f(s["longitude"]),
                            "distance_miles": _f(s.get("distance_miles")), "rating": _f(s.get("rating")),
                            "address": s.get("address"),
                            "high_traffic": bool(is_high_traffic_gas_brand(s.get("name")))})
    for c in ((fetched.get("competitors_data") or {}).get("competitors") or []):
        if c.get("latitude") is not None and c.get("longitude") is not None:
            markers.append({"category": "car_wash", "name": c.get("name"),
                            "lat": _f(c["latitude"]), "lon": _f(c["longitude"]),
                            "distance_miles": _f(c.get("distance_miles")), "rating": _f(c.get("rating")),
                            "user_rating_count": _i(c.get("user_rating_count")),
                            "primary_type": c.get("primary_type"), "address": c.get("address")})
    for a in ((fetched.get("retail_anchors") or {}).get("anchors") or []):
        if a.get("latitude") is not None and a.get("longitude") is not None:
            markers.append({"category": "retail", "name": a.get("name"), "type": a.get("type"),
                            "lat": _f(a["latitude"]), "lon": _f(a["longitude"]),
                            "distance_miles": _f(a.get("distance_miles")), "address": a.get("address")})
    return markers


# ───────────────────────── rule-based insights (always available) ─────────────────────────
def rule_weather(climate: dict) -> dict:
    """Rule-based weather read — backs the Weather tab's insight card (ported verbatim)."""
    climate = climate or {}
    if not climate:
        return {}
    rainy = climate.get("rainy_days")
    snow = climate.get("total_snowfall_cm")
    pleasant = climate.get("days_pleasant_temp")
    freeze = climate.get("days_below_freezing")
    bits = []
    if rainy is not None:
        bits.append(f"{int(rainy)} rainy days/yr")
    if snow is not None and snow >= 1:
        bits.append(f"{snow:.0f} cm snow/yr")
    if pleasant is not None:
        bits.append(f"{int(pleasant)} comfortable days/yr")
    if freeze is not None:
        bits.append(f"{int(freeze)} freezing days/yr")
    insight = ("Weather here: " + ", ".join(bits) + ". Rain and snow dirty cars (more wash demand); "
               "comfortable days are prime washing days, while freezing days suppress it.") if bits else None
    pro = (f"~{int(rainy)} rainy days a year keep cars dirty and create wash demand."
           if rainy else "Mild, dry weather keeps cars washable year-round.")
    con = (f"~{int(freeze)} freezing days a year can shut washing down."
           if freeze else "Little weather-driven dirt to force frequent washes.")
    favorable = (pleasant or 0) >= 150 and (freeze or 0) <= 60
    return {"insight": insight, "pro": pro, "con": con,
            "conclusion": "Weather is " + ("favorable" if favorable else "mixed") + " for steady wash demand."}


def rule_competition(cdata: dict) -> dict:
    """Rule-based competition read — backs the Competition tab's insight card (ported verbatim)."""
    cdata = cdata or {}
    comps = cdata.get("competitors") or []
    count = cdata.get("count", len(comps))
    nearest = comps[0] if comps else {}
    nd = nearest.get("distance_miles")
    nr = nearest.get("rating")
    insight = f"{count} competing car wash{'es' if count != 1 else ''} within {COMPETITOR_RADIUS_MILES:.0f} miles"
    if nd is not None:
        insight += f"; nearest is {nd:.1f} mi away"
        if nr is not None:
            insight += f" ({nr:.1f}★)"
    insight += "."
    pro = ("Few nearby competitors — less rivalry for wash demand." if count <= 2
           else "A proven car-wash market — demand clearly exists here.")
    con = ("A close, well-rated competitor can pull demand away."
           if (nd is not None and nd <= 1.0) else "Several competitors share the local demand.")
    conclusion = ("Low competition — favorable." if count <= 2
                  else "Crowded — differentiation matters." if count >= 6 else "Moderate competition.")
    return {"insight": insight, "pro": pro, "con": con, "conclusion": conclusion}


def rule_retail(retail: dict) -> dict:
    """Rule-based retail read — backs the Retail anchors tab's insight card (ported verbatim)."""
    retail = retail or {}
    anchors = retail.get("anchors") or []
    grocery = retail.get("grocery_count_1mile") or 0
    food = retail.get("food_count_0_5miles") or 0
    parts = []
    for label, key in [("warehouse club", "costco_dist"), ("Walmart", "walmart_dist"), ("Target", "target_dist")]:
        d = retail.get(key)
        if d is not None:
            parts.append(f"{label} {d:.1f} mi")
    has_any = bool(anchors or grocery or food)
    insight = f"{len(anchors)} retail anchors within {RETAIL_RADIUS_FAR_MILES:.0f} miles"
    if parts:
        insight += f" ({', '.join(parts)})"
    insight += f"; {grocery} grocery within 1 mi, {food} food spots within ½ mi. "
    insight += ("Busy retail feeds errand trips and impulse washes." if has_any
                else "Sparse retail means weaker passing/errand traffic.")
    pro = ("Strong retail draw nearby pulls steady passing traffic." if has_any
           else "Less retail competition for the parcel.")
    con = ("Retail traffic alone doesn't guarantee wash conversion." if has_any
           else "Limited nearby shopping traffic to feed impulse washes.")
    conclusion = ("Good retail co-tenancy." if (len(anchors) >= 3 or grocery >= 2) else "Modest retail support.")
    return {"insight": insight, "pro": pro, "con": con, "conclusion": conclusion}


def rule_gas(gas: list) -> dict:
    """Rule-based gas read — backs the Gas stations tab's insight card (ported verbatim)."""
    gas = gas or []
    within = sorted([s for s in gas if s.get("distance_miles") is not None
                     and s["distance_miles"] <= GAS_RADIUS_FAR_MILES], key=lambda s: s["distance_miles"])
    nearest = within[0] if within else {}
    nd = nearest.get("distance_miles")
    nm = nearest.get("name")
    ht = is_high_traffic_gas_brand(nm)
    insight = f"{len(within)} gas stations within {GAS_RADIUS_FAR_MILES:.0f} miles"
    if nm and nd is not None:
        insight += f"; nearest {nm} at {nd:.1f} mi"
    if ht:
        insight += " (high-traffic brand)"
    insight += ". Busy fuel stops drive impulse car washes."
    pro = ("A close, high-traffic fuel brand means lots of passing drivers." if ht
           else "Nearby fuel stops add passing traffic.")
    con = ("Few/distant gas stations mean weaker fuel-stop traffic." if len(within) <= 1
           else "Fuel traffic helps but isn't decisive on its own.")
    conclusion = ("Strong fuel-driven traffic." if (ht or len(within) >= 4) else "Some fuel-driven traffic.")
    return {"insight": insight, "pro": pro, "con": con, "conclusion": conclusion}


# ───────────────────────── orchestrator ─────────────────────────
def analyze_site_context(lat: float, lon: float, address: Optional[str] = None,
                         include_ai: bool = False, demo: bool = False) -> Dict[str, Any]:
    """One synchronous call that backs the whole "🔎 Site analysis" page for a (lat, lon) pin.

    Fetches the four external dimensions in parallel (fetch_features), then assembles headline
    metrics, map markers and per-dimension data + rule-based insights — exactly what render() shows.
    When include_ai is True and the internal LLM is reachable (2 s socket pre-check), per-dimension
    AI write-ups are attached under each dimension's "ai_insight"; if unreachable the key is omitted
    and the rule-based insight stands (the page's "showing rule-based insights instead" fallback).
    demo=True hides the origin address on the map markers (passes address=None to build_markers).

    Returns a fully JSON-serializable dict (plain float/int/None — never NaN, no numpy scalars).
    """
    fetched = fetch_features(lat, lon)

    climate = fetched.get("climate") or {}
    cdata = fetched.get("competitors_data") or {}
    retail = fetched.get("retail_anchors") or {}
    gas = fetched.get("gas_stations") or []
    comps = cdata.get("competitors") or []
    anchors = retail.get("anchors") or []
    gas_in = [s for s in gas if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_FAR_MILES]

    # ── headline metrics (render() lines ~368-376) ──
    nd = comps[0].get("distance_miles") if comps else None
    pleasant = climate.get("days_pleasant_temp")
    metrics = {
        "competitors_4mi": int(len(comps)),
        "nearest_competitor_mi": _f(nd),
        "retail_anchors_3mi": int(len(anchors)),
        "gas_stations_3mi": int(len(gas_in)),
        "comfortable_days": _i(pleasant),
    }

    # ── map markers (demo hides the origin address) ──
    markers = build_markers(lat, lon, fetched, None if demo else address)

    # ── optional AI write-ups (rule insight always stays; ai_insight attached only when present) ──
    ai: Dict[str, Any] = {}
    if include_ai:
        ai = ai_summaries(fetched)

    def _attach_ai(dim: str, base: Dict[str, Any]) -> Dict[str, Any]:
        d = (ai.get(dim) or {}) if include_ai else {}
        if any(d.get(k) for k in ("insight", "pro", "con", "conclusion")):
            base["ai_insight"] = d
        return base

    # ── competition dimension (Competition tab) ──
    competition = {
        "count": int(len(comps)),
        "radius_miles": float(COMPETITOR_RADIUS_MILES),
        "nearest": ({
            "name": comps[0].get("name"),
            "distance_miles": _f(comps[0].get("distance_miles")),
            "rating": _f(comps[0].get("rating")),
            "user_rating_count": _i(comps[0].get("user_rating_count")),
            "primary_type": comps[0].get("primary_type"),
            "address": comps[0].get("address"),
        } if comps else None),
        "competitors": [{
            "name": c.get("name"),
            "distance_miles": _f(c.get("distance_miles")),
            "rating": _f(c.get("rating")),
            "user_rating_count": _i(c.get("user_rating_count")),
            "primary_type": c.get("primary_type"),
            "address": c.get("address"),
        } for c in comps],
        "insight": rule_competition(cdata),
    }
    _attach_ai("competition", competition)

    # ── retail dimension (Retail anchors tab) ──
    retail_dim = {
        "anchors": [{
            "name": a.get("name"),
            "type": a.get("type"),
            "distance_miles": _f(a.get("distance_miles")),
            "address": a.get("address"),
        } for a in anchors],
        "key_distances": {
            "costco": _f(retail.get("costco_dist")),
            "walmart": _f(retail.get("walmart_dist")),
            "target": _f(retail.get("target_dist")),
        },
        "grocery_1mi": int(retail.get("grocery_count_1mile") or 0),
        "food_0_5mi": int(retail.get("food_count_0_5miles") or 0),
        "insight": rule_retail(retail),
    }
    _attach_ai("retail", retail_dim)

    # ── gas dimension (Gas stations tab — sorted within-radius, matching the table) ──
    gas_sorted = sorted(gas_in, key=lambda s: s["distance_miles"])
    gas_dim = {
        "count": int(len(gas_in)),
        "radius_miles": float(GAS_RADIUS_FAR_MILES),
        "stations": [{
            "name": s.get("name"),
            "distance_miles": _f(s.get("distance_miles")),
            "rating": _f(s.get("rating")),
            "user_rating_count": _i(s.get("rating_count")),
            "high_traffic": bool(is_high_traffic_gas_brand(s.get("name"))),
            "address": s.get("address"),
        } for s in gas_sorted],
        "insight": rule_gas(gas),
    }
    _attach_ai("gas", gas_dim)

    # ── weather dimension (Weather tab — per-metric values via the config mapping) ──
    weather_metrics: List[Dict[str, Any]] = []
    if climate:
        for mk in WEATHER_METRIC_CONFIG:
            val, unit = get_weather_metric_value_from_climate(climate, mk)
            disp, sub = WEATHER_METRIC_DISPLAY.get(mk, (mk, ""))
            weather_metrics.append({
                "key": mk,
                "display": disp,
                "subtitle": sub,
                "value": _f(val),
                "unit": unit,
            })
    weather_dim = {
        "metrics": weather_metrics,
        "insight": rule_weather(climate),
    }
    _attach_ai("weather", weather_dim)

    return {
        "lat": _f(lat),
        "lon": _f(lon),
        "address": address,
        "has_api_key": bool(calib.GOOGLE_MAPS_API_KEY),
        "metrics": metrics,
        "markers": markers,
        "dimensions": {
            "competition": competition,
            "retail": retail_dim,
            "gas": gas_dim,
            "weather": weather_dim,
        },
    }
