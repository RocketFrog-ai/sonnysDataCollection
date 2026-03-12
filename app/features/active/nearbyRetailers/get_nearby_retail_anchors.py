"""
Unified anchor retailer fetch: single Google Places searchNearby call for all retail anchor types
(Warehouse Club, Big Box, Grocery, Food & Beverage, Home Improvement) within a configurable radius.
Returns only name, type, distance_miles per anchor plus v3 feature extraction values.

Replaces the separate nearbyStores (Costco/Walmart/Target) + nearbyRetailers (grocery/food) dual-fetch.
"""

from __future__ import annotations

import logging
import requests
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
METERS_PER_MILE = 1609.34

DEFAULT_RADIUS_MILES = 3.0

# Google Places API types → anchor category
# Two batches: anchor stores + food/beverage (Places API limits includedTypes to 50)
ANCHOR_STORE_TYPES = [
    "warehouse_store",       # Costco, Sam's Club, BJ's
    "department_store",      # Target, Walmart, Meijer, Kohl's
    "grocery_store",         # Kroger, Publix, Safeway, H-E-B
    "supermarket",           # alternative Grocery label
    "home_improvement_store",# Home Depot, Lowe's
    "wholesaler",            # backup for warehouse retailers
]
FOOD_TYPES = [
    "fast_food_restaurant",  # McDonald's, Chick-fil-A, Chipotle, Taco Bell
    "coffee_shop",           # Starbucks, Dunkin', Panera
]

# Name keyword → (retail_type, brand_key used for v3 extraction)
#   brand_key: "costco" | "walmart" | "target" | "grocery" | "food" | None
KEYWORD_MAP: List[Tuple[str, str, str]] = [
    # Warehouse Club
    ("costco",        "Warehouse Club",    "costco"),
    ("sam's club",    "Warehouse Club",    "costco"),   # treat like Costco-class
    ("bj's",          "Warehouse Club",    "costco"),
    # Big Box / Discount
    ("walmart",       "Supercenter",       "walmart"),
    ("target",        "Big Box / Discount","target"),
    ("meijer",        "Big Box",           "walmart"),  # closest class
    ("kohl's",        "Big Box",           None),
    # Home Improvement
    ("home depot",    "Home Improvement",  None),
    ("lowe's",        "Home Improvement",  None),
    ("lowes",         "Home Improvement",  None),
    # Grocery Anchors
    ("kroger",        "Grocery Anchor",    "grocery"),
    ("publix",        "Grocery Anchor",    "grocery"),
    ("safeway",       "Grocery Anchor",    "grocery"),
    ("whole foods",   "Grocery Anchor",    "grocery"),
    ("aldi",          "Grocery Anchor",    "grocery"),
    ("trader joe's",  "Grocery Anchor",    "grocery"),
    ("trader joes",   "Grocery Anchor",    "grocery"),
    ("h-e-b",         "Grocery Anchor",    "grocery"),
    ("heb",           "Grocery Anchor",    "grocery"),
    ("wegmans",       "Grocery Anchor",    "grocery"),
    ("sprouts",       "Grocery Anchor",    "grocery"),
    ("market",        "Grocery Anchor",    "grocery"),
    # Food & Beverage
    ("mcdonald",      "Food & Beverage",   "food"),
    ("chick-fil-a",   "Food & Beverage",   "food"),
    ("chick fil a",   "Food & Beverage",   "food"),
    ("starbucks",     "Food & Beverage",   "food"),
    ("dunkin",        "Food & Beverage",   "food"),
    ("chipotle",      "Food & Beverage",   "food"),
    ("panera",        "Food & Beverage",   "food"),
    ("burger king",   "Food & Beverage",   "food"),
    ("wendy's",       "Food & Beverage",   "food"),
    ("wendys",        "Food & Beverage",   "food"),
    ("taco bell",     "Food & Beverage",   "food"),
    ("subway",        "Food & Beverage",   "food"),
]

# Google Places type → fallback retail type when no name keyword matches
PLACES_TYPE_FALLBACK: Dict[str, Tuple[str, str]] = {
    "warehouse_store":        ("Warehouse Club",   "costco"),
    "wholesaler":             ("Warehouse Club",   "costco"),
    "department_store":       ("Big Box",          None),
    "grocery_store":          ("Grocery Anchor",   "grocery"),
    "supermarket":            ("Grocery Anchor",   "grocery"),
    "home_improvement_store": ("Home Improvement", None),
    "fast_food_restaurant":   ("Food & Beverage",  "food"),
    "coffee_shop":            ("Food & Beverage",  "food"),
}

# Priority for deduplication: keep the highest-priority match per place_id
ANCHOR_TYPE_PRIORITY = {
    "Warehouse Club":   10,
    "Supercenter":       9,
    "Big Box / Discount":8,
    "Big Box":           7,
    "Grocery Anchor":    6,
    "Home Improvement":  5,
    "Food & Beverage":   4,
    "General Retail":    1,
}


def _classify(name: Optional[str], place_types: Optional[List[str]]) -> Tuple[str, Optional[str]]:
    """Return (retail_type, brand_key). brand_key is 'costco'|'walmart'|'target'|'grocery'|'food'|None."""
    if name:
        nl = name.lower()
        for keyword, rtype, bkey in KEYWORD_MAP:
            if keyword in nl:
                return rtype, bkey
    for pt in (place_types or []):
        if pt in PLACES_TYPE_FALLBACK:
            rtype, bkey = PLACES_TYPE_FALLBACK[pt]
            return rtype, bkey
    return "General Retail", None


def _batch_distances(
    api_key: str,
    origin_lat: float,
    origin_lon: float,
    dests: List[Tuple[float, float]],
) -> List[Optional[float]]:
    """Google Distance Matrix batch call. Returns distance_miles per dest (None on error)."""
    if not dests:
        return []
    destinations = "|".join(f"{lat},{lon}" for lat, lon in dests)
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
        return [None] * len(dests)
    if data.get("status") != "OK":
        return [None] * len(dests)
    elements = ((data.get("rows") or [{}])[0]).get("elements") or []
    out: List[Optional[float]] = []
    for el in elements:
        if el.get("status") == "OK":
            dist_m = (el.get("distance") or {}).get("value")
            out.append(round(dist_m / METERS_PER_MILE, 2) if dist_m is not None else None)
        else:
            out.append(None)
    # Pad if fewer elements returned than expected
    while len(out) < len(dests):
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
) -> Dict[str, Any]:
    """
    Fetch all major retail anchors (Warehouse Club, Big Box, Grocery, Food & Beverage,
    Home Improvement) within radius_miles using:
      1. Google Places searchNearby — two calls (anchor stores + food types).
      2. Google Distance Matrix — one batch call for driving distances.

    Returns:
        {
            "anchors": [{"name": str, "type": str, "distance_miles": float}, ...],  # sorted by distance
            "costco_dist": float | None,     # nearest Warehouse Club distance (for costco_enc)
            "walmart_dist": float | None,    # nearest Supercenter/Big Box distance
            "target_dist": float | None,     # nearest Big Box / Discount distance
            "grocery_count_1mile": int,      # grocery anchors within 1 mile
            "food_count_0_5miles": int,      # food & beverage within 0.5 miles
        }
    """
    if not api_key:
        return _empty_result()

    # Two searchNearby calls (anchor stores + food), then merge
    raw_anchor = _search_places(api_key, latitude, longitude, radius_miles, ANCHOR_STORE_TYPES, max_results=20)
    raw_food   = _search_places(api_key, latitude, longitude, radius_miles, FOOD_TYPES, max_results=20)
    all_raw = raw_anchor + raw_food

    if not all_raw:
        return _empty_result()

    # Deduplicate by place_id, keep highest-priority anchor type
    seen: Dict[str, Dict[str, Any]] = {}
    for place in all_raw:
        pid = place.get("id") or str(place.get("name", ""))
        dn = place.get("displayName") or {}
        name = dn.get("text") if isinstance(dn, dict) else None
        types = place.get("types") or []
        rtype, bkey = _classify(name, types)
        priority = ANCHOR_TYPE_PRIORITY.get(rtype, 0)
        if pid not in seen or priority > ANCHOR_TYPE_PRIORITY.get(seen[pid]["type"], 0):
            seen[pid] = {
                "name": name,
                "type": rtype,
                "brand_key": bkey,
                "lat": (place.get("location") or {}).get("latitude"),
                "lon": (place.get("location") or {}).get("longitude"),
            }

    candidates = [v for v in seen.values() if v.get("lat") is not None and v.get("lon") is not None]
    if not candidates:
        return _empty_result()

    # Batch distance matrix
    dests = [(c["lat"], c["lon"]) for c in candidates]
    distances = _batch_distances(api_key, latitude, longitude, dests)

    anchors: List[Dict[str, Any]] = []
    for c, dist in zip(candidates, distances):
        if dist is None or dist > radius_miles:
            continue
        anchors.append({
            "name": c["name"],
            "type": c["type"],
            "distance_miles": dist,
            "_brand_key": c["brand_key"],
        })

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
