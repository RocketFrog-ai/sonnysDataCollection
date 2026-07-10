"""
Site analysis (visual) · beta — an immersive Google-Earth-style site explorer.

Given the shared map pin (the same (lat, lon) used by Explore markets & Forecast) — or a typed
address / lat-lon — this renders a Google Maps Platform dashboard:

  LEFT  (70%)  Photorealistic 3D tiles centred + tilted on the site, with a "📍 YOU" site marker, rotate /
               zoom / pan / tilt, a Street-View toggle, Nearby-Places overlays (wash sites, gas, restaurants,
               retail, mass-merchants) filterable at 1 / 3 / 5-mile rings, and an "Anchor routes" overlay that
               draws the road each nearby Walmart/Costco-type anchor pulls traffic along and flags every car
               wash sitting on it (the washes that intercept anchor-bound customers).
  RIGHT (30%)  "Site intelligence" — trade-area feature values pulled from
               proforma/data/ref/merged_all_sites.csv (matched to the nearest site by lat/lon):
               population & growth, income, vehicles, hourly traffic, and commercial activity.
               The ONE thing read live from Google (not the CSV) is the *nearest wash-sites*
               list, surfaced in the map's Places layer.

Two API keys (complementary projects), kept separate on purpose:
  • GOOGLE_MAPS_JS_API_KEY  — loads the browser map + Photorealistic 3D tiles + Street View (Map Tiles API).
  • GOOGLE_MAPS_API_KEY     — server-side Places (New) nearby search + Geocoding (Places API New + Geocoding API).
Falls back to GOOGLE_MAPS_API_KEY for the map loader if the JS key isn't set.

No AI "corner-plot / true-false" labels — the map itself reveals frontage, intersections, parking and
surrounding activity.

Run:  streamlit run proforma/v1_5/ui/app.py   (from the repo root)
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Make the repo-root `app.*` package importable when run from streamlits/ (see site_analysis_page.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]   # proforma/v1_5/ui/x.py -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_APP_DIR = _REPO_ROOT / "app"
if _APP_DIR.is_dir() and (
    "app" not in sys.modules or not hasattr(sys.modules.get("app"), "__path__")
):
    _app_mod = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("app", None, is_package=True)
    )
    _app_mod.__path__ = [str(_APP_DIR)]
    sys.modules["app"] = _app_mod

from app.core import common as calib  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = _REPO_ROOT / "proforma" / "data" / "ref" / "merged_all_sites.csv"
EARTH_KM = 6371.0088
MILE_M = 1609.34
FETCH_RADIUS_M = 8050          # widest ring is 5 mi (~8047 m); the UI filters to 1 / 3 / 5 mi client-side
ENROUTE_BUFFER_M = 480.0       # a wash sits "on the road" to an anchor if within ~0.3 mi of the road corridor
CORRIDOR_APPROACH_M = 3200.0   # length of approach road drawn BEFORE the site (~2 mi) so the corridor leads INTO the anchor
ROUTE_COLORS = ["#e53935", "#8e24aa", "#1e88e5"]   # per-anchor road colours (the ≤3 nearest anchors)
# ── en-route detour scoring: a wash is a better site if stopping at it barely lengthens the drive to the anchor ──
CANDIDATE_BUFFER_M = 800.0     # cheap prefilter: only score washes within this of the driving corridor
MAX_CANDIDATES = 5             # per anchor, cap how many washes we spend a Routes (detour) call on — bounds cost
DETOUR_GOOD_MIN = 2.0          # a wash adding ≤ this many minutes to the anchor drive is a PRIME intercept (green)
DETOUR_OK_MIN = 5.0            # ≤ this = still "en route"; beyond it it's a real detour, not on the way

# Mass-merchant / big-box anchor brands (Walmart, Costco, …) — used to filter the anchor search results.
MASS_RE = re.compile(
    r"\b(wal[\s-]?mart|costco|sam'?s\s*club|target|bj'?s|meijer|kohl'?s|"
    r"home\s*depot|lowe'?s|ikea|best\s*buy)\b", re.I)

# Place categories → Places(New) primary types + marker colour. Shared by Python (fetch) & JS (render).
# Trimmed to the few POIs that actually co-locate with / drive car-wash demand (competitors, fuel, food & retail
# traffic, and big-box anchors) so the per-pin Places bill stays low — schools / hospitals / gyms / hotels /
# offices were dropped. "mass" uses POPULARITY ranking + a brand filter so the real anchors surface (see fetch).
CATS = [
    {"id": "car_wash",   "label": "🚿 Wash sites",    "types": ["car_wash"], "color": "#2979ff", "on": True},
    {"id": "gas",        "label": "⛽ Gas",            "types": ["gas_station"], "color": "#00897b", "on": False},
    {"id": "restaurant", "label": "🍔 Restaurants",    "types": ["restaurant"], "color": "#fb8c00", "on": False},
    {"id": "mall",       "label": "🛍️ Retail",         "types": ["shopping_mall", "department_store"], "color": "#8e24aa", "on": False},
    {"id": "mass",       "label": "🏬 Mass merchants", "types": ["department_store", "discount_store", "warehouse_store", "supermarket"],
     "color": "#c62828", "on": False, "rank": "POPULARITY", "brand": True},
]


def js_map_key() -> str:
    """Key for the browser map loader / 3D tiles (Map Tiles project). Falls back to the server key."""
    return os.getenv("GOOGLE_MAPS_JS_API_KEY", "") or (calib.GOOGLE_MAPS_API_KEY or "")


def server_key() -> str:
    """Key for server-side Places (New) + Geocoding."""
    return calib.GOOGLE_MAPS_API_KEY or ""


def routes_key() -> str:
    """Key for the Routes API (driving paths to anchors). The Places server-key project doesn't have Routes API
    enabled, so this defaults to GOOGLE_ROUTES_API_KEY / the JS-project key (which does), else the server key."""
    return os.getenv("GOOGLE_ROUTES_API_KEY", "") or js_map_key() or server_key()


# ───────────────────────────── trade-area data (from the CSV) ─────────────────────────────
@st.cache_data(show_spinner="Loading trade-area features…")
def load_features() -> pd.DataFrame:
    df = pd.read_csv(CSV, low_memory=False)
    df = df[df[["lat", "lon"]].notna().all(axis=1)].reset_index(drop=True)
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_site(df: pd.DataFrame, lat: float, lon: float):
    """Closest CSV site to the pin → (row, distance_km). The CSV's trade-area features describe this site."""
    d = haversine_km(lat, lon, df.lat.values, df.lon.values)
    i = int(np.argmin(d))
    return df.iloc[i], float(d[i])


def _num(row, col):
    if col not in row or pd.isna(row[col]):
        return None
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return None


def _money(v):
    return f"${v:,.0f}" if v is not None else "—"


def _int(v):
    return f"{v:,.0f}" if v is not None else "—"


def _pct(v, *, frac=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}%" if frac else f"{v:.1f}%"


def _one(v):
    return f"{v:.1f}" if v is not None else "—"


def build_intelligence(row, dist_km: float) -> list[dict]:
    """Format the CSV trade-area values into the right-panel sections (label / value / sub)."""
    g25 = _num(row, "Growth 2025-2020")
    g30 = _num(row, "Growth 2030-2025")
    traffic_keys = [
        ("Overnight", "Nearest StreetLight US Hourly-ttl_overnight"),
        ("Breakfast", "Nearest StreetLight US Hourly-ttl_breakfast"),
        ("Lunch", "Nearest StreetLight US Hourly-ttl_lunch"),
        ("Afternoon", "Nearest StreetLight US Hourly-ttl_afternoon"),
        ("Dinner", "Nearest StreetLight US Hourly-ttl_dinner"),
        ("Night", "Nearest StreetLight US Hourly-ttl_night"),
        ("Highway", "Nearest StreetLight US Hourly-Highway"),
    ]
    return [
        {
            "title": "Population & growth", "icon": "👥",
            "rows": [
                {"label": "2025 population (market)", "value": _int(_num(row, "2025 Estimate"))},
                {"label": "Growth ’20→’25", "value": _pct(g25, frac=True)},
                {"label": "Growth ’25→’30 (proj.)", "value": _pct(g30, frac=True)},
                {"label": "Average age", "value": _one(_num(row, "2025 Average Age"))},
                {"label": "Labor force", "value": _int(_num(row, "Labor Force"))},
                {"label": "Renter-occupied units", "value": _int(_num(row, "Renter-Occupied"))},
            ],
        },
        {
            "title": "Household income", "icon": "💵",
            "rows": [
                {"label": "Median HH income", "value": _money(_num(row, "Median Household Income"))},
                {"label": "Average HH income", "value": _money(_num(row, "Average Household Income"))},
                {"label": "HH with income $50k+", "value": _pct(_num(row, "2025 % HH with Income $50K+"))},
            ],
        },
        {
            "title": "Demand drivers — vehicles", "icon": "🚗",
            "rows": [
                {"label": "Total vehicles in market", "value": _int(_num(row, "Total Vehicles Available in the Market"))},
                {"label": "Avg vehicles / household", "value": _one(_num(row, "Average Number of Vehicles Available"))},
            ],
        },
        {
            "title": "Hourly traffic (StreetLight)", "icon": "🛣️",
            "rows": [{"label": k, "value": _int(_num(row, c))} for k, c in traffic_keys],
        },
        {
            "title": "Commercial activity", "icon": "🏬",
            "rows": [
                {"label": "Mass-merchant stores", "value": _int(_num(row, "Count of ChainXY VT - Mass Merchant"))},
                {"label": "Grocery stores", "value": _int(_num(row, "Count of ChainXY VT - Grocery"))},
                {"label": "Department stores", "value": _int(_num(row, "Count of ChainXY VT - Department Store"))},
            ],
        },
    ]


# ───────────────────────────── live Places (New) — server-side, old key ─────────────────────────────
def _search_one(types, lat, lon, key, rank="DISTANCE"):
    """One Places(New) searchNearby POST → list of {name, lat, lng, m, rating, n, type}.
    rank="POPULARITY" surfaces big well-rated anchors (Walmart/Costco) over nearby dollar/thrift stores.
    Field mask is name + location + primaryType only — that keeps the call on the Places Nearby Search
    *Pro* SKU ($32/1k). Adding rating / userRatingCount would bump every call to the Enterprise SKU
    ($35/1k), so they're intentionally omitted (rating / n come back null and the UI just drops the ★)."""
    import requests
    body = {
        "includedPrimaryTypes": types, "maxResultCount": 20, "rankPreference": rank,
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": float(FETCH_RADIUS_M)}},
    }
    headers = {
        "Content-Type": "application/json", "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "places.displayName,places.location,places.primaryType",
    }
    r = requests.post("https://places.googleapis.com/v1/places:searchNearby",
                      json=body, headers=headers, timeout=10)
    out = []
    for p in (r.json().get("places") or []):
        loc = p.get("location") or {}
        plat, plon = loc.get("latitude"), loc.get("longitude")
        if plat is None or plon is None:
            continue
        m = float(haversine_km(lat, lon, np.array([plat]), np.array([plon]))[0] * 1000.0)
        out.append({
            "name": (p.get("displayName") or {}).get("text") or "—",
            "lat": plat, "lng": plon, "m": round(m),
            "rating": p.get("rating"), "n": p.get("userRatingCount"),
            "type": p.get("primaryType"),
        })
    out.sort(key=lambda x: x["m"])
    return out


@st.cache_data(show_spinner="Fetching nearby places…", ttl=60 * 60)
def fetch_places(lat: float, lon: float, key: str) -> dict:
    """Server-fetch ALL categories (one Places(New) searchNearby per category, run in parallel) so every chip
    shows a real count immediately. We used to fetch only the on-by-default wash layer and lazy-load the rest in
    the browser, but the browser Place.searchNearby silently returns nothing here (the JS-map key path) while the
    SERVER key works for every category — so the other chips were stuck at 0. Fetching server-side is the reliable
    path; results flow to the JS as CFG.places and the lazy browser fetch becomes a no-op (data already present).
    Cached 1 h per (lat, lon). {} on no key."""
    if not key:
        return {}
    results: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(CATS))) as ex:
        futs = {ex.submit(_search_one, c["types"], lat, lon, key, c.get("rank", "DISTANCE")): c["id"] for c in CATS}
        for fut, cid in [(f, futs[f]) for f in futs]:
            try:
                results[cid] = fut.result()
            except Exception:
                results[cid] = []
    # brand-filter the anchor categories so dollar / thrift / corner stores don't masquerade as mass merchants
    for c in CATS:
        if c.get("brand"):
            results[c["id"]] = _filter_brand(results.get(c["id"], []))
    return results


def _filter_brand(items: list) -> list:
    """Keep only recognised big-box brands (Walmart, Costco, …), nearest first. Falls back to the genuine
    big-box place-types if no brand matched, so the anchor layer isn't silently empty in an unbranded area."""
    branded = [x for x in items if MASS_RE.search(x.get("name") or "")]
    if branded:
        return sorted(branded, key=lambda x: x["m"])
    keep = {"department_store", "discount_store", "warehouse_store"}
    return sorted([x for x in items if x.get("type") in keep], key=lambda x: x["m"])


@st.cache_data(show_spinner="Geocoding…")
def geocode(address: str, key: str):
    """Address → (lat, lon) via Google Geocoding API (server key). Cached. None on failure."""
    if not key:
        return None
    try:
        import requests
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json",
                         params={"address": address, "key": key}, timeout=8)
        js = r.json()
        if js.get("status") == "OK" and js.get("results"):
            loc = js["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
    except Exception:
        return None
    return None


# ───────────────────────────── routes to mass-merchant anchors (Routes API) ─────────────────────────────
def _decode_polyline(enc: str) -> list[tuple[float, float]]:
    """Decode a Google encoded-polyline string → [(lat, lon), …]."""
    coords, index, lat, lng, n = [], 0, 0, 0, len(enc)
    while index < n:
        for is_lat in (True, False):
            shift = result = 0
            while True:
                b = ord(enc[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lat:
                lat += d
            else:
                lng += d
        coords.append((lat * 1e-5, lng * 1e-5))
    return coords


def _pt_seg_m(plat, plon, alat, alon, blat, blon) -> float:
    """Metres from point P to segment A–B via a local equirectangular projection (fine at these scales)."""
    latref = math.radians((alat + blat) / 2.0)
    kx, ky = 111320.0 * math.cos(latref), 110540.0
    px, py = (plon - alon) * kx, (plat - alat) * ky
    bx, by = (blon - alon) * kx, (blat - alat) * ky
    seg2 = bx * bx + by * by
    t = 0.0 if seg2 == 0 else max(0.0, min(1.0, (px * bx + py * by) / seg2))
    return math.hypot(px - t * bx, py - t * by)


def _min_dist_to_path(wlat, wlon, path) -> float:
    best = 1e18
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        d = _pt_seg_m(wlat, wlon, a[0], a[1], b[0], b[1])
        if d < best:
            best = d
    return best


def _approach_origin(slat, slon, mlat, mlon):
    """A commuter start point ~CORRIDOR_APPROACH_M beyond the site, directly away from the anchor. Routing from
    here to the anchor traces the road corridor that carries anchor-bound traffic PAST the site — so the drawn
    path is 'the road to the mass merchant', not a trip that begins at our wash."""
    latref = math.radians(slat)
    kx, ky = 111320.0 * math.cos(latref), 110540.0
    dx, dy = (slon - mlon) * kx, (slat - mlat) * ky      # anchor → site direction (metres)
    n = math.hypot(dx, dy) or 1.0
    olon = slon + (dx / n) * CORRIDOR_APPROACH_M / kx
    olat = slat + (dy / n) * CORRIDOR_APPROACH_M / ky
    return olat, olon


def _route_one(olat, olon, dlat, dlon, key, via=None):
    """Driving route O→D via Routes API → (path[(lat,lon)…], distance_m, duration_s); (None, None, None) on failure.
    When `via=(lat,lon)` is given it becomes an intermediate STOP, so the route is O→via→D (used to price the
    detour of stopping at a car wash on the way to an anchor)."""
    import requests
    body = {
        "origin": {"location": {"latLng": {"latitude": olat, "longitude": olon}}},
        "destination": {"location": {"latLng": {"latitude": dlat, "longitude": dlon}}},
        "travelMode": "DRIVE", "routingPreference": "TRAFFIC_UNAWARE",
    }
    if via is not None:
        body["intermediates"] = [{"location": {"latLng": {"latitude": via[0], "longitude": via[1]}}}]
    headers = {
        "Content-Type": "application/json", "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline",
    }
    try:
        r = requests.post("https://routes.googleapis.com/directions/v2:computeRoutes",
                          json=body, headers=headers, timeout=12)
        rt = (r.json().get("routes") or [None])[0]
        if rt and (rt.get("polyline") or {}).get("encodedPolyline"):
            dur = rt.get("duration", "") or ""
            dur_s = int(dur[:-1]) if dur.endswith("s") and dur[:-1].isdigit() else None
            return _decode_polyline(rt["polyline"]["encodedPolyline"]), rt.get("distanceMeters"), dur_s
    except Exception:
        pass
    return None, None, None


@st.cache_data(show_spinner="Mapping drivable routes & car-wash stops to nearby anchors…", ttl=60 * 60)
def compute_routes(lat: float, lon: float, key: str, payload: str) -> list[dict]:
    """For the ≤3 nearest mass-merchant anchors, model the drive a commuter takes INTO the anchor (from a start
    ~2 mi out on the site's side, so it's 'the road to the mass merchant', not a trip that starts at our wash),
    then score how well each nearby car wash INTERCEPTS that trip by the real DETOUR of stopping there:

        detour = drive(commuter → wash → anchor)  −  drive(commuter → anchor)

    A low detour means the wash is naturally on the way to the anchor (a stronger intercept site); a big detour
    means it's off-route. We also score the pin itself ('your site') so you can see whether THIS location is
    en route. Each candidate carries its own drive-through polyline so the UI can draw the O→wash→anchor route.

    `payload` = JSON {"mass":[...], "wash":[...], "site":{lat,lng}} (a string so the result caches per pin).
    Cost note: 1 baseline Routes call + up to MAX_CANDIDATES detour calls per anchor (candidates prefiltered to
    those near the corridor). Cached 1 h. Falls back to a straight-line corridor (proximity flag, no detour) if
    the Routes API is unavailable."""
    if not key:
        return []
    data = json.loads(payload)
    dests = data.get("mass", [])[:3]
    wash = data.get("wash", [])                       # every nearby wash is a candidate — don't single out the site
    site = data.get("site")                           # the pin (our candidate build) — scored as its own "wash"
    out = []
    routes_api_ok = True   # if the first computeRoutes fails (e.g. Routes API disabled), skip the rest → straight-line
    for i, d in enumerate(dests):
        alat, alon = d["lat"], d["lng"]
        olat, olon = _approach_origin(lat, lon, alat, alon)          # commuter start ~2 mi out, anchor-opposite
        base_path = base_dist = base_dur = None
        if routes_api_ok:
            base_path, base_dist, base_dur = _route_one(olat, olon, alat, alon, key)
            if base_path is None:
                routes_api_ok = False
        straight = base_path is None
        if straight:                                                 # Routes API off/failed → straight-line corridor
            base_path = [(olat, olon), (alat, alon)]

        # candidate washes = the site itself + nearby washes within the corridor prefilter (nearest-to-path first, capped)
        candidates = []
        if site:
            candidates.append({"name": "📍 Your site", "lat": site["lat"], "lng": site["lng"], "is_site": True})
        near = sorted(
            ((_min_dist_to_path(w["lat"], w["lng"], base_path), w) for w in wash),
            key=lambda t: t[0],
        )
        seen = set()
        for dist_path, w in near:
            if dist_path > CANDIDATE_BUFFER_M:
                break
            nm = w.get("name") or "—"
            if nm in seen:
                continue
            seen.add(nm)
            candidates.append({"name": nm, "lat": w["lat"], "lng": w["lng"], "is_site": False})
            if len(candidates) >= MAX_CANDIDATES + (1 if site else 0):
                break

        washes = []
        for c in candidates:
            det_min = det_mi = via_poly = None
            en_route = False
            if not straight:                                          # price the real O→wash→anchor detour
                vp, vdist, vdur = _route_one(olat, olon, alat, alon, key, via=(c["lat"], c["lng"]))
                if vp is not None and base_dur is not None and vdur is not None:
                    det_min = round((vdur - base_dur) / 60.0, 1)
                    det_mi = round(((vdist or 0) - (base_dist or 0)) / MILE_M, 2)
                    via_poly = [[round(p[0], 6), round(p[1], 6)] for p in vp]
                    en_route = det_min <= DETOUR_OK_MIN
            else:                                                     # no routing → fall back to straight-line proximity
                en_route = _min_dist_to_path(c["lat"], c["lng"], base_path) <= ENROUTE_BUFFER_M
            washes.append({
                "name": c["name"], "lat": c["lat"], "lng": c["lng"], "is_site": c["is_site"],
                "detour_min": det_min, "detour_mi": det_mi, "via": via_poly, "en_route": en_route,
            })
        # en-route first, then by smallest detour (unknown detour last)
        washes.sort(key=lambda x: (not x["en_route"], x["detour_min"] if x["detour_min"] is not None else 1e9))
        out.append({
            "label": d.get("name") or "anchor", "lat": alat, "lng": alon,
            "color": ROUTE_COLORS[i % len(ROUTE_COLORS)],
            "dist_mi": round((d.get("m") or 0.0) / MILE_M, 1),       # straight-line site→anchor (from the Places fetch)
            "straight": straight,
            "polyline": [[round(p[0], 6), round(p[1], 6)] for p in base_path],
            "washes": washes,
            "washes_n": sum(1 for w in washes if w["en_route"]),
        })
    return out


# ───────────────────────────── the immersive dashboard (Maps JS) ─────────────────────────────
def dashboard_html(map_key: str, lat: float, lon: float, label: str,
                   match_name: str, match_dist_km: float, sections: list[dict], places: dict,
                   routes: list[dict]) -> str:
    cfg = json.dumps({
        "key": map_key, "lat": lat, "lon": lon, "label": label,
        "matchName": match_name, "matchDist": round(match_dist_km, 2),
        "sections": sections, "places": places, "routes": routes,
        "cats": [{"id": c["id"], "label": c["label"], "color": c["color"], "on": c["on"]} for c in CATS],
        "detourGood": DETOUR_GOOD_MIN, "detourOk": DETOUR_OK_MIN,
    })
    return r"""
<div id="root" style="font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
  <style>
    #root *{box-sizing:border-box}
    .dash{display:flex;gap:0;height:760px;border:1px solid #e3e6ea;border-radius:12px;overflow:hidden;background:#0b1726}
    .left{flex:7;position:relative;background:#0b1726;min-width:0}
    .right{flex:3;background:#f7f8fa;overflow-y:auto;padding:14px 14px 30px;border-left:1px solid #e3e6ea}
    #map3d,#map2d{position:absolute;inset:0;width:100%;height:100%}
    #map2d,#sv{display:none}
    #sv{position:absolute;inset:0}
    .ctrlbar{position:absolute;top:12px;left:12px;z-index:5;display:flex;gap:6px;flex-wrap:wrap}
    .btn{background:rgba(17,28,45,.86);color:#fff;border:1px solid rgba(255,255,255,.18);
         border-radius:8px;padding:7px 11px;font-size:12.5px;font-weight:600;cursor:pointer;backdrop-filter:blur(6px)}
    .btn:hover{background:rgba(34,52,82,.95)}
    .btn.on{background:#2f6df6;border-color:#2f6df6}
    .chips{position:absolute;bottom:12px;left:12px;right:12px;z-index:5;display:flex;gap:6px;flex-wrap:wrap}
    .chip{background:rgba(17,28,45,.86);color:#dfe7f5;border:1px solid rgba(255,255,255,.16);
          border-radius:20px;padding:5px 11px;font-size:12px;cursor:pointer;backdrop-filter:blur(6px);user-select:none}
    .chip.on{color:#fff;font-weight:600}
    .banner{position:absolute;top:12px;right:12px;z-index:6;background:rgba(180,60,30,.92);color:#fff;
            font-size:11.5px;padding:6px 10px;border-radius:8px;max-width:240px;display:none}
    .rbanner{position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:6;
             background:rgba(17,28,45,.92);color:#fff;font-size:12px;padding:7px 12px;border-radius:8px;
             max-width:72%;display:none;backdrop-filter:blur(6px);box-shadow:0 2px 10px rgba(0,0,0,.4)}
    .secthead{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#8a93a3;margin:16px 0 6px;font-weight:700}
    .h1{font-size:16px;font-weight:800;color:#10243f;margin:0 0 2px}
    .sub{font-size:12px;color:#6b7686;margin:0 0 4px}
    .card{background:#fff;border:1px solid #e7eaef;border-radius:10px;padding:10px 12px;margin-bottom:10px}
    .card .ct{font-size:13px;font-weight:700;color:#1f2d44;margin-bottom:7px}
    .kv{display:flex;justify-content:space-between;align-items:baseline;padding:3px 0;font-size:13px;border-top:1px dashed #eef1f5}
    .kv:first-of-type{border-top:0}
    .kv .k{color:#56606e}.kv .v{font-weight:700;color:#10243f}
    .kv .v.live{color:#2f6df6}
    .poi{font-size:12.5px;padding:6px 0;border-top:1px solid #eef1f5;display:flex;justify-content:space-between;gap:8px}
    .poi:first-child{border-top:0}
    .poi .nm{font-weight:600;color:#1f2d44}.poi .meta{color:#7b8494;white-space:nowrap}
    .star{color:#f5a623}
    .legend{font-size:11px;color:#8a93a3;margin-top:6px}
  </style>

  <div class="dash">
    <div class="left">
      <gmp-map-3d id="map3d" mode="hybrid"></gmp-map-3d>
      <div id="map2d"></div>
      <div id="sv"></div>
      <div class="ctrlbar">
        <div class="btn on" id="b3d">🌍 3D</div>
        <div class="btn" id="b2d">🗺️ Satellite</div>
        <div class="btn" id="bsv">🚶 Street View</div>
        <div class="btn" id="brot">↻ Orbit</div>
        <div class="btn" id="broutes">🛣️ Anchor routes</div>
      </div>
      <div class="banner" id="banner"></div>
      <div class="rbanner" id="rbanner"></div>
      <div id="fatal" style="display:none;position:absolute;inset:0;z-index:9;background:#0b1726;color:#e7eef9;
           padding:28px;font-size:13px;line-height:1.55;overflow:auto">
        <div style="font-size:15px;font-weight:800;margin-bottom:8px">🛑 Map could not load</div>
        <div id="fatalmsg"></div>
      </div>
      <div class="chips" id="chips"></div>
    </div>
    <div class="right" id="panel"></div>
  </div>
</div>

<script>
const CFG = __CFG__;
const SITE = {lat: CFG.lat, lng: CFG.lon};
const PLACES = CFG.places || {};
const ROUTES = CFG.routes || [];
const CATS = CFG.cats;
const MI = 1609.34;
const RADII = [1609,4828,8047];   // 1 / 3 / 5 miles
let radius = 1609;                 // default: 1-mile ring

function distLabel(m){ return m < MI*0.25 ? Math.round(m*3.28084)+' ft' : (m/MI).toFixed(1)+' mi'; }
function miLabel(m){ return (m/MI).toFixed(0)+' mi'; }

// ── right-hand intelligence panel ────────────────────────────────────────────
function renderPanel(){
  const p=document.getElementById('panel');
  let h=`<div class="h1">Site intelligence</div>
         <div class="sub">${CFG.label}</div>
         <div class="sub">Trade area: <b>${CFG.matchName}</b> · ${CFG.matchDist} km from pin</div>`;
  h+=`<div class="secthead">Nearby (live · Google Places)</div>
      <div class="card"><div class="ct">Nearest wash sites</div><div id="nearComp"></div>
      <div class="legend">Live from Google Places — the one layer not read from the CSV.</div></div>`;
  h+=`<div class="secthead">Car washes en route to mass merchants</div>
      <div class="card"><div class="ct">Detour to stop on the way to Walmart / Costco-type anchors</div><div id="routes"></div>
      <div class="legend">Each anchor's <b>traffic road</b> (coloured, arrows = direction of flow into the store) is
      drawn on the map; clicking a car wash overlays the <b style="color:#ff6d00">route with that stop</b> in orange
      and shows the <b>extra minutes</b>. A low <b>+min detour</b> = a wash that naturally intercepts anchor-bound
      traffic (a stronger site). Toggle <b>🛣️ Anchor routes</b> to start.</div></div>`;
  h+=`<div class="secthead">Trade-area features (merged_all_sites.csv)</div>`;
  for(const s of CFG.sections){
    h+=`<div class="card"><div class="ct">${s.icon} ${s.title}</div>`;
    for(const r of s.rows){
      h+=`<div class="kv"><span class="k">${r.label}</span><span class="v ${r.live?'live':''}">${r.value}</span></div>`;
    }
    h+=`</div>`;
  }
  p.innerHTML=h;
  renderRoutes();
}

function hav(la1,lo1,la2,lo2){ const R=6371008.8,r=Math.PI/180,
  s=Math.sin((la2-la1)*r/2)**2+Math.cos(la1*r)*Math.cos(la2*r)*Math.sin((lo2-lo1)*r/2)**2;
  return 2*R*Math.asin(Math.sqrt(Math.min(1,s))); }

let placesLib=null;
async function ensureCat(cat){          // lazy: fetch a category's nearby places in the browser, once, on first toggle
  if(PLACES[cat.id]) return;            // wash sites arrive pre-loaded from the server; other layers fetched here on demand
  try{
    if(!placesLib) placesLib=await google.maps.importLibrary("places");
    const {Place,SearchNearbyRankPreference}=placesLib;
    const resp=await Place.searchNearby({
      fields:["displayName","location","primaryType"],   // name-only → Pro SKU (no rating/userRatingCount → not Enterprise)
      locationRestriction:{center:{lat:SITE.lat,lng:SITE.lng},radius:5000},
      includedPrimaryTypes:cat.types,maxResultCount:20,
      rankPreference:SearchNearbyRankPreference.DISTANCE});
    PLACES[cat.id]=(resp.places||[]).map(p=>{
      const la=p.location.lat(),lo=p.location.lng();
      return {name:(p.displayName||'—'),lat:la,lng:lo,m:Math.round(hav(SITE.lat,SITE.lng,la,lo)),
              rating:p.rating,n:p.userRatingCount,type:p.primaryType};
    }).sort((a,b)=>a.m-b.m);
    delete built[cat.id];                // force updateMarkers to (re)build this layer from the new data
  }catch(e){ PLACES[cat.id]=[]; console.warn('lazy Places fetch failed for',cat.id,e); }
}

function renderChips(){
  const c=document.getElementById('chips'); let h='';
  for(const cat of CATS){
    const n=(PLACES[cat.id]||[]).filter(x=>x.m<=radius).length;
    h+=`<div class="chip ${cat.on?'on':''}" data-cat="${cat.id}" style="${cat.on?'background:'+cat.color:''}">${cat.label} ${n}</div>`;
  }
  h+=`<div class="chip" id="radchip" style="margin-left:auto">◎ ${miLabel(radius)}</div>`;
  c.innerHTML=h;
  c.querySelectorAll('.chip[data-cat]').forEach(el=>el.onclick=async()=>{
    const cat=CATS.find(x=>x.id===el.dataset.cat); cat.on=!cat.on; renderChips();
    if(cat.on) await ensureCat(cat);     // pay for a category's Places call only when someone switches it on
    renderChips(); updateMarkers();
  });
  document.getElementById('radchip').onclick=()=>{
    radius=RADII[(RADII.indexOf(radius)+1)%RADII.length]; renderChips(); updateMarkers();
  };
}

function renderNearComp(){
  const el=document.getElementById('nearComp'); if(!el) return;
  const list=(PLACES['car_wash']||[]).filter(x=>x.m<=radius).slice(0,6);
  if(!list.length){el.innerHTML='<span style="color:#7b8494">none within '+distLabel(radius)+'</span>';return;}
  el.innerHTML=list.map(c=>{
    const rt=c.rating?`<span class="star">★</span>${(+c.rating).toFixed(1)}`:'';
    return `<div class="poi"><span class="nm">${c.name}</span><span class="meta">${rt} · ${distLabel(c.m)}</span></div>`;
  }).join('');
}

function detourText(w){                   // "+2.3 min · +0.4 mi", or "near route" when we couldn't route
  if(w.detour_min==null) return 'near route';
  return '+'+w.detour_min+' min'+(w.detour_mi!=null?' · +'+w.detour_mi+' mi':'');
}
function renderRoutes(){                  // right-panel: per anchor, each nearby wash ranked by detour to stop en route
  const el=document.getElementById('routes'); if(!el) return;
  if(!ROUTES.length){ el.innerHTML='<span style="color:#7b8494">No Walmart / Costco-type anchors within 5 mi.</span>'; return; }
  let anyStraight=false;
  let h=ROUTES.map(r=>{
    const sw=`<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${r.color};margin-right:6px;vertical-align:middle"></span>`;
    const rows=(r.washes||[]).map(w=>{
      if(w.detour_min==null) anyStraight=true;
      const good = w.detour_min!=null && w.detour_min<=CFG.detourGood;
      const col = good?'#00a152':(w.en_route?'#1f2d44':'#9aa4b2');
      const tag = w.is_site?'📍 ':'';
      const star = good?' ⭐':'';
      const check = w.en_route?'':' <span style="color:#c0c6cf">(off-route)</span>';
      return `<div class="poi"><span class="nm" style="color:${col}">${tag}${w.name}${star}</span>`
           + `<span class="meta" style="color:${col}">${detourText(w)}${check}</span></div>`;
    }).join('') || '<div class="poi"><span class="meta" style="color:#7b8494">no washes near this drive</span></div>';
    return `<div style="margin-bottom:10px">`
         + `<div class="poi"><span class="nm">${sw}${r.label}</span>`
         + `<span class="meta">${r.dist_mi} mi · <b style="color:${r.color}">${r.washes_n} en route</b></span></div>`
         + rows + `</div>`;
  }).join('');
  if(anyStraight)
    h+=`<div class="legend" style="margin-top:6px">Straight-line only (couldn't route) — enable the <b>Routes API</b> on the Maps key for real drivable detours.</div>`;
  el.innerHTML=h;
}

// ── Maps bootstrap ───────────────────────────────────────────────────────────
let map2d, pano, advLib, the3d, orbiting=false, orbitTimer=null;
let routeObjs=[];   // drawn polylines + anchor / en-route-wash markers for the Anchor-routes overlay
const built={};   // catId -> [{data, m, am(2D), m3(3D)}]

(g=>{var h,a,k,p="The Google Maps JavaScript API",c="google",l="importLibrary",q="__ib__",m=document,b=window;
b=b[c]||(b[c]={});var d=b.maps||(b.maps={}),r=new Set,e=new URLSearchParams,
u=()=>h||(h=new Promise(async(f,n)=>{await (a=m.createElement("script"));e.set("libraries",[...r]+"");
for(k in g)e.set(k.replace(/[A-Z]/g,t=>"_"+t[0].toLowerCase()),g[k]);e.set("callback",c+".maps."+q);
a.src=`https://maps.${c}apis.com/maps/api/js?`+e;d[q]=f;a.onerror=()=>h=n(Error(p+" could not load."));
a.nonce=m.querySelector("script[nonce]")?.nonce||"";m.head.append(a)}));d[l]?console.warn(p+" only loads once. Ignoring:",g):d[l]=(f,...n)=>r.add(f)&&u().then(()=>d[l](f,...n))})
({key:CFG.key, v:"alpha"});

function showFatal(html){
  const f=document.getElementById('fatal'); document.getElementById('fatalmsg').innerHTML=html;
  f.style.display='block';
}
window.gm_authFailure=function(){
  showFatal('Google rejected the Maps key (<code>gm_authFailure</code>).<br>Check that <b>Maps JavaScript API</b> '
    +'is enabled and billing is on for the JS-key project, and that any key restriction allows this domain.');
};
function hasWebGL2(){ try{return !!document.createElement('canvas').getContext('webgl2');}catch(e){return false;} }

async function init(){
  if(!hasWebGL2()){
    showFatal('Your browser isn’t providing <b>WebGL2</b>, which Google’s Photorealistic 3D Maps require.<br><br>'
      +'➊ Open <code>chrome://settings/system</code> → turn on <b>“Use graphics acceleration when available”</b>.<br>'
      +'➋ Fully <b>quit &amp; relaunch</b> Chrome (not just reload).<br>'
      +'➌ Check <code>chrome://gpu</code> — the <b>WebGL2</b> line should say <i>Hardware accelerated</i>.<br><br>'
      +'On a remote desktop / VM / locked-down corporate browser the GPU may be unavailable and 3D can’t run there.');
    return;
  }
  try{ await google.maps.importLibrary("maps3d"); }catch(e){}
  advLib = await google.maps.importLibrary("marker");

  the3d=document.getElementById('map3d');
  if(the3d && customElements.get('gmp-map-3d')){
    try{
      the3d.center={lat:SITE.lat,lng:SITE.lng,altitude:0};
      the3d.range=600; the3d.tilt=67; the3d.heading=0;
      const {Marker3DElement}=await google.maps.importLibrary("maps3d");
      const youM=new Marker3DElement({position:{lat:SITE.lat,lng:SITE.lng,altitude:6},
        altitudeMode:"RELATIVE_TO_GROUND",extruded:true,label:"📍 YOU"});
      try{ if(advLib.PinElement) youM.append(new advLib.PinElement(
        {background:'#ffd400',borderColor:'#1a1a1a',glyphColor:'#1a1a1a',scale:1.9})); }catch(e){}
      the3d.append(youM);
    }catch(e){ show3dBanner(); }
  } else { show3dBanner(); }

  const {Map}=await google.maps.importLibrary("maps");
  map2d=new Map(document.getElementById('map2d'),{
    center:SITE, zoom:18, tilt:47, heading:0, mapTypeId:'hybrid',
    mapId:'DEMO_MAP_ID', streetViewControl:false, mapTypeControl:false, fullscreenControl:false});
  new advLib.AdvancedMarkerElement({map:map2d,position:SITE,content:pin('#ffd400','📍 YOU'),zIndex:9999});
  // Street View is loaded lazily on first click (see bsv) so we don't pay for a pano nobody opens.

  wireButtons(); updateMarkers();
}

function pin(color,txt){           // the "📍 YOU" site marker — big, dark text on a bright badge with a coloured halo
  const d=document.createElement('div');
  d.style.cssText=`background:${color};color:#1a1a1a;font:900 15px sans-serif;padding:8px 14px;border-radius:18px;
    border:3px solid #fff;box-shadow:0 0 0 3px ${color},0 3px 12px rgba(0,0,0,.7);white-space:nowrap`;
  d.textContent=txt; return d;
}
function dot(color){
  const d=document.createElement('div');
  d.style.cssText=`width:13px;height:13px;border-radius:50%;background:${color};border:2px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,.5)`;
  return d;
}
function show3dBanner(){
  const b=document.getElementById('banner'); b.style.display='block';
  b.textContent='Photorealistic 3D unavailable on this key — showing tilted satellite. Enable “Map Tiles API”.';
  document.getElementById('map3d').style.display='none';
  document.getElementById('map2d').style.display='block';
  document.getElementById('b2d').classList.add('on'); document.getElementById('b3d').classList.remove('on');
}

function wireButtons(){
  const show=(id)=>['map3d','map2d','sv'].forEach(x=>document.getElementById(x).style.display=(x===id?'block':'none'));
  const b3=document.getElementById('b3d'), b2=document.getElementById('b2d'), bs=document.getElementById('bsv');
  const setOn=(el)=>{[b3,b2,bs].forEach(x=>x.classList.remove('on'));el.classList.add('on');};
  b3.onclick=()=>{ if(document.getElementById('banner').style.display==='block'){show('map2d');setOn(b2);return;} show('map3d');setOn(b3); };
  b2.onclick=()=>{show('map2d');setOn(b2);};
  bs.onclick=async()=>{                  // lazy-load Street View only when first opened (saves the $14/1k pano load otherwise)
    if(!pano){
      const {StreetViewPanorama}=await google.maps.importLibrary("streetView");
      pano=new StreetViewPanorama(document.getElementById('sv'),{position:SITE,pov:{heading:30,pitch:5},zoom:1});
    }
    show('sv');setOn(bs);
  };
  document.getElementById('brot').onclick=()=>{
    orbiting=!orbiting; document.getElementById('brot').classList.toggle('on',orbiting);
    if(orbiting){orbitTimer=setInterval(()=>{
      if(the3d&&the3d.style.display!=='none'){the3d.heading=(the3d.heading+0.6)%360;}
      else if(map2d){map2d.setHeading((map2d.getHeading()+1)%360);}
    },60);} else clearInterval(orbitTimer);
  };
  const br=document.getElementById('broutes');      // 🛣️ draw driving routes to nearby mass-merchant anchors + flag en-route washes
  br.onclick=()=>{
    const on=!br.classList.contains('on'); br.classList.toggle('on',on);
    if(on){ show('map2d'); setOn(b2); drawRoutes(); }   // routes render on the 2D satellite view
    else clearRoutes();
  };
}

let viaObjs=[];                    // the drive-through (O→wash→anchor) overlay for the currently-selected stop
const DETOUR_COLOR='#ff6d00';      // the route WITH the car-wash stop is drawn in this contrasting colour (vs the traffic road)
function arrowIcons(color){        // repeated chevrons so a polyline reads as a DIRECTIONAL traffic road (points along the path)
  return [{icon:{path:google.maps.SymbolPath.FORWARD_CLOSED_ARROW,scale:3,strokeColor:'#ffffff',
    strokeWeight:1,fillColor:color,fillOpacity:1},offset:'6%',repeat:'90px'}];
}
function storePin(color,label){    // anchor (Walmart/Costco) marker — a coloured badge matching its route line
  const d=document.createElement('div');
  d.style.cssText=`background:${color};color:#fff;font:800 12px sans-serif;padding:5px 9px;border-radius:8px;
    border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.55);white-space:nowrap`;
  d.textContent='🏬 '+label; return d;
}
function washMarker(w,color){      // a car-wash STOP on the way to an anchor — styled by its detour (green=prime intercept)
  const d=document.createElement('div');
  if(w.is_site){                   // the pin itself — is OUR location on the way to this anchor?
    d.style.cssText=`background:#ffd400;color:#1a1a1a;font:900 11px sans-serif;padding:4px 8px;border-radius:14px;
      border:2px solid #fff;box-shadow:0 0 0 3px rgba(255,23,68,.55),0 2px 6px rgba(0,0,0,.6);white-space:nowrap;cursor:pointer`;
    d.textContent='📍 YOU'; return d;
  }
  const good = w.detour_min!=null && w.detour_min<=CFG.detourGood;
  const bg = good?'#00c853':(w.en_route?color:'#9aa4b2');       // green prime · anchor-colour on-route · grey off-route
  const ring = w.en_route?'#ff1744':'rgba(255,255,255,.85)';    // red "intercept" ring for en-route washes
  d.style.cssText=`width:15px;height:15px;border-radius:50%;background:${bg};border:3px solid ${ring};
    box-shadow:0 0 0 2px rgba(255,255,255,.6),0 1px 3px rgba(0,0,0,.6);cursor:pointer`;
  return d;
}
function clearVia(){ for(const o of viaObjs){ if(o.setMap) o.setMap(null); else o.map=null; } viaObjs=[]; }
function drawVia(r,w){             // overlay the route WITH the stop (origin→wash→anchor) in the DETOUR colour, on top of the traffic road
  clearVia();
  if(w.via && w.via.length){
    viaObjs.push(new google.maps.Polyline({path:w.via.map(p=>({lat:p[0],lng:p[1]})),map:map2d,
      strokeColor:DETOUR_COLOR,strokeOpacity:.95,strokeWeight:6,geodesic:true,zIndex:70,icons:arrowIcons(DETOUR_COLOR)}));
  }
  const rb=document.getElementById('rbanner');
  const det=(w.detour_min==null)?'straight-line only':('<b>'+detourText(w)+'</b> extra to stop');
  const legend=`<span style="color:${r.color}">▬▸ traffic to ${r.label}</span> &nbsp; `
             + `<span style="color:${DETOUR_COLOR}">▬▸ route with stop</span>`;
  rb.innerHTML=`${legend} &nbsp;·&nbsp; 🚗 <b>${w.name}</b> — ${det}`; rb.style.display='block';
}
function drawRoutes(){             // baseline commuter drive (thin) + anchor badge + each wash stop; click a stop → its drive-through
  clearRoutes();
  let best=null, bestR=null;
  for(const r of ROUTES){
    routeObjs.push(new google.maps.Polyline({path:r.polyline.map(p=>({lat:p[0],lng:p[1]})),map:map2d,
      strokeColor:r.color,strokeOpacity:.75,strokeWeight:5,geodesic:true,zIndex:45,
      icons:arrowIcons(r.color)}));   // the TRAFFIC road into the anchor — arrows show the direction of flow toward the store
    routeObjs.push(new advLib.AdvancedMarkerElement({map:map2d,position:{lat:r.lat,lng:r.lng},
      content:storePin(r.color,r.label),zIndex:60}));
    for(const w of (r.washes||[])){
      const el=washMarker(w,r.color); el.onclick=()=>drawVia(r,w);   // click a wash → draw its real O→wash→anchor route
      routeObjs.push(new advLib.AdvancedMarkerElement({map:map2d,position:{lat:w.lat,lng:w.lng},
        content:el,title:`${w.name} · ${detourText(w)} en route to ${r.label}`,zIndex:55}));
      if(w.en_route && w.detour_min!=null && (best==null || w.detour_min<best.detour_min)){ best=w; bestR=r; }
    }
  }
  if(map2d) map2d.setCenter(SITE);
  if(best) drawVia(bestR,best);    // show the single best drivable en-route example up front
}
function clearRoutes(){
  clearVia();
  const rb=document.getElementById('rbanner'); if(rb) rb.style.display='none';
  for(const o of routeObjs){ if(o.setMap) o.setMap(null); else o.map=null; }
  routeObjs=[];
}

// ── overlay markers from server-fetched places; toggle by category + radius ──
async function updateMarkers(){
  let Marker3DElement=null;
  if(the3d&&customElements.get('gmp-map-3d')&&document.getElementById('banner').style.display!=='block'){
    try{({Marker3DElement}=await google.maps.importLibrary("maps3d"));}catch(e){}
  }
  for(const cat of CATS){
    const data=PLACES[cat.id]||[];
    if(!built[cat.id]) built[cat.id]=data.map(d=>({d, m:d.m, am:null, m3:null}));
    for(const it of built[cat.id]){
      const vis = cat.on && it.m<=radius;
      const loc={lat:it.d.lat,lng:it.d.lng};
      // 2D
      if(vis && !it.am){
        it.am=new advLib.AdvancedMarkerElement({map:map2d,position:loc,content:dot(cat.color),
          title:`${it.d.name} · ${distLabel(it.m)}`});
      } else if(it.am){ it.am.map = vis ? map2d : null; }
      // 3D — colour the pin by category (default 3D markers are red; this gives each category its colour)
      if(vis && !it.m3 && Marker3DElement){
        it.m3=new Marker3DElement({position:{lat:loc.lat,lng:loc.lng,altitude:2},
          altitudeMode:"RELATIVE_TO_GROUND",label:it.d.name});
        try{ if(advLib.PinElement) it.m3.append(new advLib.PinElement(
          {background:cat.color,borderColor:'#ffffff',glyphColor:'#ffffff'})); }catch(e){}
        the3d.append(it.m3);
      } else if(!vis && it.m3){ it.m3.remove(); it.m3=null; }
    }
  }
  renderNearComp();
}

renderPanel(); renderChips(); init();
</script>
""".replace("__CFG__", cfg)


# ───────────────────────────── page entry point ─────────────────────────────
def render(demo: bool = False):
    st.title("🛰️ Site analysis (visual) · beta")
    st.caption("Google-Earth-style 3D explorer + live Places overlays (competitors, gas, food, retail & "
               "mass-merchants at 1 / 3 / 5-mile rings), with trade-area intelligence from `merged_all_sites.csv`. "
               "Toggle **🛣️ Anchor routes** to see the drive to each nearby Walmart/Costco-type anchor and the "
               "car washes on the way — ranked by the extra minutes to stop (low detour = a wash that naturally "
               "intercepts anchor-bound traffic). Click a wash to draw its drive-through route.")

    df = load_features()
    if "pin" not in st.session_state:
        r0 = df.iloc[0]
        st.session_state.pin = (float(r0.lat), float(r0.lon))

    # ── location input: address (geocoded) or lat/lon → moves the shared pin ──
    with st.expander("📍 Location — address or coordinates", expanded=False):
        c1, c2 = st.columns([3, 2])
        with c1:
            addr = st.text_input("Address", placeholder="e.g. 1600 Amphitheatre Pkwy, Mountain View, CA")
            if st.button("Geocode address", use_container_width=True) and addr.strip():
                ll = geocode(addr.strip(), server_key())
                if ll:
                    st.session_state.pin = ll
                    st.rerun()
                else:
                    st.warning("Couldn’t geocode that address (check the address / Geocoding API).")
        with c2:
            plat, plon = st.session_state.pin
            ilat = st.number_input("Latitude", value=float(plat), format="%.5f", key="vlat")
            ilon = st.number_input("Longitude", value=float(plon), format="%.5f", key="vlon")
            if st.button("Drop pin here", use_container_width=True):
                st.session_state.pin = (float(ilat), float(ilon))
                st.rerun()

    lat, lon = st.session_state.pin
    lat, lon = round(float(lat), 5), round(float(lon), 5)
    label = f"📍 {lat:.5f}, {lon:.5f}"

    if not js_map_key():
        st.error("No Google Maps key set — the map won't load. Set `GOOGLE_MAPS_JS_API_KEY` "
                 "(Map Tiles + Maps JavaScript API) and/or `GOOGLE_MAPS_API_KEY`.")
        return
    if not server_key():
        st.warning("`GOOGLE_MAPS_API_KEY` is not set — the Nearby-Places overlays & nearest-competitor "
                   "list (Places API New) will be empty. The 3D map still works.")

    row, dist_km = nearest_site(df, lat, lon)
    name = str(row.get("client_name") or "nearest site")
    sections = build_intelligence(row, dist_km)
    if dist_km > 15:
        st.info(f"Closest trade-area record is **{name}**, ~{dist_km:.0f} km away — its feature values may not "
                "describe this exact spot. The map & live Places still reflect the pin.")

    places = fetch_places(lat, lon, server_key()) if server_key() else {}
    routes = []
    if routes_key() and places.get("mass"):
        payload = json.dumps({"mass": places.get("mass", []), "wash": places.get("car_wash", []),
                              "site": {"lat": lat, "lng": lon}})   # score the pin itself as an en-route candidate
        routes = compute_routes(lat, lon, routes_key(), payload)

    components.html(
        dashboard_html(js_map_key(), lat, lon, label, name, dist_km, sections, places, routes),
        height=790, scrolling=False,
    )
