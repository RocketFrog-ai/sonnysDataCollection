"""
Unified anchor retailer fetch: single Google Places searchNearby call for all retail anchor types
(Warehouse Club, Big Box, Grocery, Food & Beverage, Home Improvement) within a configurable radius.
Returns only name, type, distance_miles per anchor plus v3 feature extraction values.

Replaces the separate nearbyStores (Costco/Walmart/Target) + nearbyRetailers (grocery/food) dual-fetch.
"""

from __future__ import annotations

import logging
import math
import requests
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
METERS_PER_MILE = 1609.34
MAX_DESTINATIONS_PER_REQUEST = 25

DEFAULT_RADIUS_MILES = 3.0

# Google Places API types → anchor category
# Two batches: anchor stores + food/beverage (Places API limits includedTypes to 50)
ANCHOR_STORE_TYPES = [
    "warehouse_store",   # Costco, Sam's Club, BJ's
    "department_store",  # Target, Walmart, Meijer, Kohl's
    "grocery_store",     # Kroger, Publix, Safeway, H-E-B, Whole Foods, Aldi, Trader Joe's
    "supermarket",       # alternative grocery label
]
FOOD_TYPES = [
    "fast_food_restaurant",  # McDonald's, Chick-fil-A, Chipotle
    "coffee_shop",           # Starbucks, Dunkin', Panera
]

# Only preferred anchor brands — anything not in this list is dropped.
# Food & Beverage: keyword match is REQUIRED (no type-only fallback).
KEYWORD_MAP: List[Tuple[str, str, str]] = [
    # Warehouse Club
    ("costco",          "Warehouse Club",     "costco"),
    ("sam's club",      "Warehouse Club",     "costco"),
    ("bj's",            "Warehouse Club",     "costco"),
    # Big Box / Discount
    ("walmart",         "Supercenter",        "walmart"),
    ("target",          "Big Box / Discount", "target"),
    ("meijer",          "Big Box",            "walmart"),
    ("kohl's",          "Big Box",            None),
    ("kohls",           "Big Box",            None),
    # Grocery Anchors
    ("kroger",          "Grocery Anchor",     "grocery"),
    ("publix",          "Grocery Anchor",     "grocery"),
    ("h-e-b",           "Grocery Anchor",     "grocery"),
    ("heb",             "Grocery Anchor",     "grocery"),
    ("safeway",         "Grocery Anchor",     "grocery"),
    ("whole foods",     "Grocery Anchor",     "grocery"),
    ("aldi",            "Grocery Anchor",     "grocery"),
    ("trader joe's",    "Grocery Anchor",     "grocery"),
    ("trader joes",     "Grocery Anchor",     "grocery"),
    # Food & Beverage — ONLY preferred brands
    ("mcdonald",        "Food & Beverage",    "food"),
    ("chick-fil-a",     "Food & Beverage",    "food"),
    ("chick fil a",     "Food & Beverage",    "food"),
    ("starbucks",       "Food & Beverage",    "food"),
    ("dunkin",          "Food & Beverage",    "food"),
    ("chipotle",        "Food & Beverage",    "food"),
    ("panera",          "Food & Beverage",    "food"),
]

# Only grocery_store / supermarket fall back by type alone.
# warehouse_store, department_store, fast_food_restaurant, coffee_shop all
# REQUIRE a keyword match — prevents generic restaurants/stores slipping in.
PLACES_TYPE_FALLBACK: Dict[str, Tuple[str, str]] = {
    "grocery_store": ("Grocery Anchor", "grocery"),
    "supermarket":   ("Grocery Anchor", "grocery"),
}

# All non-grocery types must match a keyword to avoid false positives.
REQUIRE_KEYWORD_TYPES = {
    "warehouse_store",
    "department_store",
    "fast_food_restaurant",
    "coffee_shop",
}

# Priority for deduplication: keep the highest-priority match per place_id
ANCHOR_TYPE_PRIORITY = {
    "Warehouse Club":    10,
    "Supercenter":        9,
    "Big Box / Discount": 8,
    "Big Box":            7,
    "Grocery Anchor":     6,
    "Food & Beverage":    4,
    "General Retail":     1,
}


def _classify(name: Optional[str], place_types: Optional[List[str]]) -> Tuple[str, Optional[str]]:
    """Return (retail_type, brand_key). Returns ('General Retail', None) for unknown places."""
    if name:
        nl = name.lower()
        for keyword, rtype, bkey in KEYWORD_MAP:
            if keyword in nl:
                return rtype, bkey

    types_set = {t.lower() for t in (place_types or [])}

    # warehouse_store and department_store require a keyword match to avoid false positives
    if types_set & REQUIRE_KEYWORD_TYPES:
        return "General Retail", None

    for api_type, (rtype, bkey) in PLACES_TYPE_FALLBACK.items():
        if api_type in types_set:
            return rtype, bkey

    return "General Retail", None


def _straight_line_miles(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> float:
    """Haversine distance in miles."""
    R = 3958.8
    lat1, lon1 = math.radians(origin_lat), math.radians(origin_lon)
    lat2, lon2 = math.radians(dest_lat), math.radians(dest_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(min(1.0, a)))
    return round(R * c, 2)


def _batch_distances(
    api_key: str,
    origin_lat: float,
    origin_lon: float,
    dests: List[Tuple[float, float]],
) -> List[Optional[float]]:
    """Google Distance Matrix: up to 25 dests per request; batches if more. Returns distance_miles per dest (None on error)."""
    if not dests:
        return []
    out: List[Optional[float]] = []
    for i in range(0, len(dests), MAX_DESTINATIONS_PER_REQUEST):
        chunk = dests[i : i + MAX_DESTINATIONS_PER_REQUEST]
        destinations = "|".join(f"{lat},{lon}" for lat, lon in chunk)
        params = {
            "origins": f"{origin_lat},{origin_lon}",
            "destinations": destinations,
            "mode": "driving",
            "units": "imperial",
            "key": api_key,
        }
        try:
            resp = requests.get(DISTANCE_MATRIX_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Distance Matrix failed: %s", exc)
            out.extend([None] * len(chunk))
            continue
        if data.get("status") != "OK":
            logger.warning("Distance Matrix status=%s: %s", data.get("status"), data.get("error_message"))
            out.extend([None] * len(chunk))
            continue
        elements = ((data.get("rows") or [{}])[0]).get("elements") or []
        for el in elements:
            if el.get("status") == "OK":
                dist_m = (el.get("distance") or {}).get("value")
                out.append(round(dist_m / METERS_PER_MILE, 2) if dist_m is not None else None)
            else:
                out.append(None)
        while len(out) < i + len(chunk):
            out.append(None)
    return out


def _search_places(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float,
    included_types: List[str],
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """One searchNearby call; returns raw place dicts from Google."""
    radius_m = min(radius_miles * METERS_PER_MILE * 1.3, 50000)  # 30% buffer for driving vs straight-line
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.types,places.formattedAddress",
    }
    payload = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_m,
            }
        },
        "includedTypes": included_types,
        "maxResultCount": min(max(1, max_results), 20),
        "rankPreference": "DISTANCE",
    }
    try:
        resp = requests.post(PLACES_SEARCH_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("places") or []
    except Exception as exc:
        logger.warning("Places searchNearby failed for types=%s: %s", included_types, exc)
        return []


def get_nearby_retail_anchors(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float = DEFAULT_RADIUS_MILES,
    use_driving_distance: bool = False,
) -> Dict[str, Any]:
    """
    Fetch retail anchors within radius_miles. By default uses straight-line distance only
    (2 Places searchNearby calls, no Distance Matrix) to keep cost low. Set use_driving_distance=True
    to call Distance Matrix for driving distances.
    """
    if not api_key:
        return _empty_result()

    raw_anchor = _search_places(api_key, latitude, longitude, radius_miles, ANCHOR_STORE_TYPES, max_results=20)
    raw_food = _search_places(api_key, latitude, longitude, radius_miles, FOOD_TYPES, max_results=20)
    all_raw = raw_anchor + raw_food

    if not all_raw:
        logger.info("Retail anchors: Places searchNearby returned 0 places (anchor=%s food=%s)", len(raw_anchor), len(raw_food))
        return _empty_result()

    def _place_name(place: Dict[str, Any]) -> Optional[str]:
        dn = place.get("displayName")
        if isinstance(dn, dict):
            return dn.get("text")
        if isinstance(dn, str):
            return dn
        return None

    seen: Dict[str, Dict[str, Any]] = {}
    for place in all_raw:
        pid = place.get("id") or str(place.get("name", ""))
        name = _place_name(place)
        types = place.get("types") or []
        rtype, bkey = _classify(name, types)
        priority = ANCHOR_TYPE_PRIORITY.get(rtype, 0)
        if pid not in seen or priority > ANCHOR_TYPE_PRIORITY.get(seen[pid]["type"], 0):
            loc = place.get("location") or {}
            lat_val = loc.get("latitude")
            lon_val = loc.get("longitude")
            seen[pid] = {
                "name": name or (rtype if rtype != "General Retail" else "Store"),
                "type": rtype,
                "brand_key": bkey,
                "lat": lat_val,
                "lon": lon_val,
            }

    candidates = [v for v in seen.values() if v.get("lat") is not None and v.get("lon") is not None]
    if not candidates:
        logger.info("Retail anchors: no candidates with valid location (seen=%d)", len(seen))
        return _empty_result()

    if use_driving_distance:
        dests = [(c["lat"], c["lon"]) for c in candidates]
        distances = _batch_distances(api_key, latitude, longitude, dests)
    else:
        distances = [None] * len(candidates)

    use_straight_line = all(d is None for d in distances)
    if use_straight_line and use_driving_distance:
        logger.info("Retail anchors: Distance Matrix returned no distances; using straight-line for %d candidates", len(candidates))

    anchors: List[Dict[str, Any]] = []
    for c, dist in zip(candidates, distances):
        if dist is None:
            dist = _straight_line_miles(latitude, longitude, c["lat"], c["lon"])
        if dist > radius_miles:
            continue
        anchors.append({
            "name": c["name"],
            "type": c["type"],
            "distance_miles": dist,
            "_brand_key": c["brand_key"],
        })

    # Drop General Retail (unrecognised brands) — they add noise with no analytical value
    anchors = [a for a in anchors if a.get("type") != "General Retail"]

    if not anchors and candidates:
        logger.warning("Retail anchors: %d candidates but none within radius %.1f mi (all distances > radius or all General Retail)", len(candidates), radius_miles)

    anchors.sort(key=lambda a: a["distance_miles"])

    # Extract v3 features
    costco_dist: Optional[float] = None
    walmart_dist: Optional[float] = None
    target_dist: Optional[float] = None
    grocery_count_1mile = 0
    food_count_0_5miles = 0

    for a in anchors:
        bkey = a.get("_brand_key")
        d = a["distance_miles"]
        if bkey == "costco" and costco_dist is None:
            costco_dist = d
        if bkey == "walmart" and walmart_dist is None:
            walmart_dist = d
        if bkey == "target" and target_dist is None:
            target_dist = d
        if bkey == "grocery" and d <= 1.0:
            grocery_count_1mile += 1
        if bkey == "food" and d <= 0.5:
            food_count_0_5miles += 1

    # Strip internal _brand_key before returning
    clean_anchors = [{"name": a["name"], "type": a["type"], "distance_miles": a["distance_miles"]} for a in anchors]

    return {
        "anchors": clean_anchors,
        "costco_dist": costco_dist,
        "walmart_dist": walmart_dist,
        "target_dist": target_dist,
        "grocery_count_1mile": grocery_count_1mile,
        "food_count_0_5miles": food_count_0_5miles,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "anchors": [],
        "costco_dist": None,
        "walmart_dist": None,
        "target_dist": None,
        "grocery_count_1mile": 0,
        "food_count_0_5miles": 0,
    }
