"""
Nearby complementary businesses (food/grocery) within a radius.
For anchor retail analysis (Costco, Walmart, Target, Grocery chains) use get_nearby_retail_anchors.
"""
import logging
import requests
from typing import Optional, Any

logger = logging.getLogger(__name__)

from app.utils import common as calib
from app.site_analysis.features.inactive.experimental_features.operationalHours.searchNearby import find_nearby_places

DEFAULT_RADIUS_MILES = 2
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/"
DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
METERS_PER_MILE = 1609.34

# Only types used for /v1/retailers: other grocery + food joint (Costco/Walmart/Target come from nearby_stores).
RETAIL_TYPES = [
    "grocery_store",
    "supermarket",
    "restaurant",
    "fast_food_restaurant",
    "coffee_shop",
]

TYPE_TO_CATEGORY = {
    "grocery_store": "Grocery",
    "supermarket": "Grocery",
    "restaurant": "Food Joint",
    "fast_food_restaurant": "Food Joint",
    "coffee_shop": "Food Joint",
}

# Preferred anchor food/grocery brands — always shown regardless of count limit.
# Non-preferred brands are capped to the closest MAX_NON_PREFERRED results.
PREFERRED_FOOD_BRANDS: set = {
    # Wholesale clubs
    "costco", "sam's club", "bj's wholesale", "bj's",
    # Big box
    "walmart", "target", "meijer", "kohl's", "kohls",
    # Grocery chains
    "kroger", "publix", "h-e-b", "heb", "safeway",
    "whole foods", "aldi", "trader joe's", "trader joes",
    # Fast food / café anchors
    "mcdonald's", "mcdonalds", "chick-fil-a", "chick fil a",
    "starbucks", "dunkin'", "dunkin", "chipotle", "panera bread", "panera",
}
MAX_NON_PREFERRED = 15


def _is_preferred_brand(name: Optional[str]) -> bool:
    if not name:
        return False
    nl = name.lower()
    return any(brand in nl for brand in PREFERRED_FOOD_BRANDS)


def _route_distances(
    api_key: str,
    origin_lat: float,
    origin_lon: float,
    dests: list[tuple[float, float]],
) -> list[dict]:
    """Route (driving) distance via Google Distance Matrix API — not straight-line.
    Returns list of {distance_miles, duration_text, ...}."""
    if not dests or not api_key:
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
    except Exception:
        return [{"distance_miles": None, "duration_text": None} for _ in dests]
    if data.get("status") != "OK":
        return [{"distance_miles": None, "duration_text": None} for _ in dests]
    rows = data.get("rows") or []
    if not rows:
        return [{"distance_miles": None, "duration_text": None} for _ in dests]
    elements = rows[0].get("elements") or []
    out = []
    for el in elements:
        if el.get("status") != "OK":
            out.append({"distance_miles": None, "duration_text": None})
            continue
        dist = el.get("distance") or {}
        dur = el.get("duration") or {}
        dist_val = dist.get("value")
        distance_miles = round(dist_val / METERS_PER_MILE, 2) if dist_val is not None else None
        out.append({
            "distance_miles": distance_miles,
            "duration_text": dur.get("text"),
        })
    return out


def _fetch_place_details(api_key: str, place_id: str, fields: str) -> Optional[dict]:
    if not place_id or not api_key:
        return None
    url = f"{PLACE_DETAILS_URL}{place_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": fields,
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _category_from_types(types: Optional[list]) -> str:
    if not types:
        return "Retail"
    for t in types:
        if t in TYPE_TO_CATEGORY:
            return TYPE_TO_CATEGORY[t]
    return types[0].replace("_", " ").title() if types else "Retail"


def _format_hours(regular_opening_hours: Optional[dict]) -> Optional[str]:
    """Format regularOpeningHours to a simple string like '7AM - 10PM' or '24 Hours'."""
    if not regular_opening_hours:
        return None
    # New Places API: regularOpeningHours has weekdayDescriptions or periods
    weekday_descriptions = regular_opening_hours.get("weekdayDescriptions")
    if weekday_descriptions and len(weekday_descriptions) > 0:
        # e.g. ["Monday: 7:00 AM – 10:00 PM", ...]; use first day as sample or check for "Open 24 hours"
        first = weekday_descriptions[0]
        if "24 hours" in first.lower() or "open 24" in first.lower():
            return "24 Hours"
        # Try to extract time range (e.g. "7:00 AM – 10:00 PM")
        for desc in weekday_descriptions:
            if ":" in desc:
                parts = desc.split(":", 1)
                if len(parts) == 2:
                    time_part = parts[1].strip()
                    if time_part and "–" in time_part:
                        return time_part.replace("–", "-").strip()
                    if time_part:
                        return time_part.strip()
        return weekday_descriptions[0] if weekday_descriptions else None
    return None


def get_nearby_retailers(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float = DEFAULT_RADIUS_MILES,
    fetch_place_details: bool = True,
) -> dict:
    """
    Returns all nearby retailers (complementary businesses) within radius_miles by driving distance.
    No cap on count: returns every retailer the API finds within the radius.
    All data from Google Places API (searchNearby, optional Place Details) and Distance Matrix.
    """
    if not api_key:
        return {
            "retailers": [],
            "count": 0,
            "avg_distance_miles": None,
        }

    # Search with slightly larger radius to get candidates (driving distance can exceed straight-line)
    search_radius_miles = max(radius_miles * 2.0, 1.0)
    search_radius_meters = search_radius_miles * 1609.34
    if search_radius_meters > 50000:
        search_radius_meters = 50000.0

    types_split = [
        ["grocery_store", "supermarket"],
        ["fast_food_restaurant", "coffee_shop"],
        ["restaurant"]
    ]
    
    places = []
    for t_list in types_split:
        res = find_nearby_places(
            api_key,
            latitude,
            longitude,
            radius_miles=search_radius_miles,
            included_types=t_list,
            max_results=20, # Google limits to 20 per request
            rank_preference="DISTANCE",
        )
        if res and "places" in res and res["places"]:
            places.extend(res["places"])

    if not places:
        logger.info("get_nearby_retailers: find_nearby_places returned no places")
        return {
            "retailers": [],
            "count": 0,
            "avg_distance_miles": None,
        }

    # Deduplicate places by ID
    unique_places = {}
    for p in places:
        pid = p.get("id") or str(p.get("name", ""))
        if pid not in unique_places:
            unique_places[pid] = p

    places = list(unique_places.values())
    logger.info("get_nearby_retailers: find_nearby_places returned %d unique places", len(places))
    dests = []
    for place in places:
        loc = place.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is not None and lon is not None:
            dests.append((float(lat), float(lon)))
        else:
            dests.append((None, None))

    valid_dests = [(a, b) for a, b in dests if a is not None and b is not None]
    route_results = _route_distances(api_key, latitude, longitude, valid_dests)
    distances_ok = sum(1 for r in route_results if r.get("distance_miles") is not None)
    logger.info("get_nearby_retailers: Distance Matrix got %d/%d distances", distances_ok, len(route_results))

    dest_idx = 0
    within = []
    for i, place in enumerate(places):
        loc = place.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            continue
        route = route_results[dest_idx] if dest_idx < len(route_results) else {}
        dest_idx += 1
        distance_miles = route.get("distance_miles")
        if distance_miles is None or distance_miles > radius_miles:
            continue

        display_name = place.get("displayName")
        name = display_name.get("text") if isinstance(display_name, dict) else None
        place_id = place.get("id") or (str(place.get("name", "")).replace("places/", "") if place.get("name") else None)
        rating = place.get("rating")
        if rating is not None:
            rating = float(rating)
        types = place.get("types") or []
        category = _category_from_types(types)
        regular_opening_hours = place.get("regularOpeningHours")
        hours_str = _format_hours(regular_opening_hours)

        website_uri = None
        google_maps_uri = None
        if place_id and fetch_place_details:
            details = _fetch_place_details(api_key, place_id, "websiteUri,googleMapsUri")
            if details:
                website_uri = details.get("websiteUri")
                google_maps_uri = details.get("googleMapsUri")

        retailer: dict[str, Any] = {
            "name": name,
            "category": category,
            "distance_miles": distance_miles,
            "rating": rating,
            "hours": hours_str,
            "address": place.get("formattedAddress") or place.get("shortFormattedAddress"),
            "place_id": place_id,
        }
        if website_uri:
            retailer["website"] = website_uri
        if google_maps_uri:
            retailer["google_maps_uri"] = google_maps_uri

        within.append(retailer)

    within.sort(key=lambda r: (r.get("distance_miles") is None, r.get("distance_miles") or float("inf")))

    # Preferred brands always shown; non-preferred capped to MAX_NON_PREFERRED closest.
    preferred = [r for r in within if _is_preferred_brand(r.get("name"))]
    non_preferred = [r for r in within if not _is_preferred_brand(r.get("name"))]
    filtered = preferred + non_preferred[:MAX_NON_PREFERRED]
    filtered.sort(key=lambda r: (r.get("distance_miles") is None, r.get("distance_miles") or float("inf")))

    within_distances = [r["distance_miles"] for r in filtered]
    avg_distance_miles = round(sum(within_distances) / len(within_distances), 2) if within_distances else None

    logger.info(
        "get_nearby_retailers: within radius %.2f mi -> %d retailers (%d preferred, %d other capped at %d); names=%s",
        radius_miles,
        len(filtered),
        len(preferred),
        min(len(non_preferred), MAX_NON_PREFERRED),
        MAX_NON_PREFERRED,
        [r.get("name") for r in filtered[:10]],
    )
    return {
        "retailers": filtered,
        "count": len(filtered),
        "avg_distance_miles": avg_distance_miles,
    }


if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv
    from app.site_analysis.features.active.nearbyRetailers.get_nearby_retail_anchors import get_nearby_retail_anchors

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    address = "1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
    if not api_key:
        print("Set GOOGLE_MAPS_API_KEY in .env")
    else:
        geo = calib.get_lat_long(address)
        if not geo:
            print(f"Could not geocode: {address}")
        else:
            lat, lon = geo["lat"], geo["lon"]
            print(f"\nAddress: {address}  |  lat={lat}  lon={lon}\n")

            print("=" * 60)
            print("ANCHOR RETAILERS (Costco / Walmart / Target / Grocery / F&B)")
            print("=" * 60)
            anchors_data = get_nearby_retail_anchors(api_key, lat, lon, radius_miles=3.0)
            anchors = anchors_data.get("anchors") or []
            within_1 = [a for a in anchors if a["distance_miles"] <= 1.0]
            between_1_and_3 = [a for a in anchors if 1.0 < a["distance_miles"] <= 3.0]
            nearest = anchors[0] if anchors else None

            if nearest:
                print(f"\nNearest Anchor: {nearest['name']}  |  {nearest['type']}  |  {nearest['distance_miles']} mi\n")
            print(f"Within 1 mile  ({len(within_1)}):")
            for a in within_1:
                print(f"  {a['name']:<35} {a['type']:<22} {a['distance_miles']} mi")
            print(f"\n1–3 miles ({len(between_1_and_3)}):")
            for a in between_1_and_3:
                print(f"  {a['name']:<35} {a['type']:<22} {a['distance_miles']} mi")
            print(f"\nKey distances extracted:")
            print(f"  costco_dist  : {anchors_data.get('costco_dist')}")
            print(f"  walmart_dist : {anchors_data.get('walmart_dist')}")
            print(f"  target_dist  : {anchors_data.get('target_dist')}")
            print(f"  grocery <=1mi: {anchors_data.get('grocery_count_1mile')}")
            print(f"  food <=0.5mi : {anchors_data.get('food_count_0_5miles')}")

            print("\n" + "=" * 60)
            print("NEARBY FOOD & GROCERY (0.5 mi, complementary)")
            print("=" * 60)
            food_data = get_nearby_retailers(api_key, lat, lon, radius_miles=0.5, fetch_place_details=False)
            print(json.dumps(food_data, indent=2, default=str))
