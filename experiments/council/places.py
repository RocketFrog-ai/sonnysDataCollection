"""
Council-local Google nearby-places — a self-contained copy of
`app/core/places/{search_nearby,nearby_competitors}.py`, so the council imports nothing from `app/`.

`nearby_competitors(lat, lon, radius_miles)` returns real car washes within the radius (driving distance)
via Google Places (New) + Distance Matrix. Reads `GOOGLE_MAPS_API_KEY` from env. **Degrades to an empty
result (never raises)** when the key is missing or the API errors, so the Competition expert can fall back
to a no-count LLM saturation read.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        load_dotenv()
    except Exception:
        pass


_load_env()

logger = logging.getLogger(__name__)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
_DIST_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
METERS_PER_MILE = 1609.34


def google_available() -> bool:
    return bool(GOOGLE_MAPS_API_KEY)


def _search_nearby(lat: float, lon: float, radius_miles: float, *, included_types=("car_wash",),
                   max_results: int = 20, api_key: Optional[str] = None) -> Optional[dict]:
    api_key = api_key or GOOGLE_MAPS_API_KEY
    radius_m = radius_miles * METERS_PER_MILE
    if not api_key or not (0.0 < radius_m <= 50000.0):
        return None
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": api_key,
               "X-Goog-FieldMask": ("places.displayName,places.location,places.rating,places.userRatingCount,"
                                    "places.formattedAddress,places.types,places.primaryTypeDisplayName,places.websiteUri")}
    payload: Dict[str, Any] = {
        "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": radius_m}},
        "maxResultCount": min(max(1, max_results), 20), "rankPreference": "DISTANCE"}
    if included_types:
        payload["includedTypes"] = list(included_types)
    try:
        r = requests.post(_SEARCH_URL, headers=headers, data=json.dumps(payload), timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("council.places search failed: %s", exc)
        return None


def _route_distances(lat: float, lon: float, dests: List[Tuple[float, float]], *,
                     api_key: Optional[str] = None) -> List[dict]:
    api_key = api_key or GOOGLE_MAPS_API_KEY
    if not dests or not api_key:
        return [{"distance_miles": None} for _ in dests]
    params = {"origins": f"{lat},{lon}", "destinations": "|".join(f"{a},{b}" for a, b in dests),
              "mode": "driving", "units": "imperial", "key": api_key}
    try:
        data = requests.get(_DIST_URL, params=params, timeout=20).json()
        els = (data.get("rows") or [{}])[0].get("elements") or []
        out = []
        for el in els:
            d = (el.get("distance") or {}).get("value") if el.get("status") == "OK" else None
            out.append({"distance_miles": round(d / METERS_PER_MILE, 2) if d is not None else None})
        return out or [{"distance_miles": None} for _ in dests]
    except Exception:
        return [{"distance_miles": None} for _ in dests]


def nearby_competitors(lat: float, lon: float, radius_miles: float = 3.0, *,
                       max_results: int = 20, api_key: Optional[str] = None) -> dict:
    """Real car washes within `radius_miles` (driving distance) via Google Places (New). Returns
    `{"competitors": [{name, distance_miles, rating, user_rating_count, address, primary_type, website,
    lat, lon}], "count": n}`. Empty (never raises) if no key / API error."""
    api_key = api_key or GOOGLE_MAPS_API_KEY
    res = _search_nearby(lat, lon, radius_miles, max_results=max_results, api_key=api_key)
    places = (res or {}).get("places") or []
    if not places:
        return {"competitors": [], "count": 0}
    dests = [(float(p["location"]["latitude"]), float(p["location"]["longitude"]))
             for p in places if (p.get("location") or {}).get("latitude") is not None]
    routes = _route_distances(lat, lon, dests, api_key=api_key)
    comps: List[dict] = []
    ri = 0
    for p in places:
        loc = p.get("location") or {}
        if loc.get("latitude") is None:
            continue
        dm = routes[ri].get("distance_miles") if ri < len(routes) else None
        ri += 1
        if dm is not None and dm > radius_miles:
            continue
        if "gas_station" in (p.get("types") or []):
            continue
        dn, ptd = p.get("displayName"), p.get("primaryTypeDisplayName")
        comps.append({
            "name": dn.get("text") if isinstance(dn, dict) else None,
            "distance_miles": dm,
            "rating": float(p["rating"]) if p.get("rating") is not None else None,
            "user_rating_count": int(p["userRatingCount"]) if p.get("userRatingCount") is not None else None,
            "address": p.get("formattedAddress"),
            "primary_type": (ptd.get("text") if isinstance(ptd, dict) else ptd) or "Car wash",
            "website": p.get("websiteUri"),
            "lat": float(loc["latitude"]), "lon": float(loc["longitude"]),
        })
    comps.sort(key=lambda c: (c["distance_miles"] is None, c["distance_miles"] or float("inf")))
    return {"competitors": comps, "count": len(comps)}
