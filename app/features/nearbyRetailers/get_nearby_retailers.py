"""
Nearby retailers (complementary businesses) within a radius using Google Places API and Distance Matrix.
Returns only real API data: name, category, distance, rating, hours; optional website/link via Place Details.
"""
import requests
from typing import Optional, Any

from app.utils import common as calib
from app.features.experimental_features.operationalHours.searchNearby import find_nearby_places

DEFAULT_RADIUS_MILES = 2
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/"
DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
METERS_PER_MILE = 1609.34

# Only types used for /v1/retailers: other grocery + food joint (Costco/Walmart/Target come from nearby_stores).
RETAIL_TYPES = [
    "grocery_store",
    "supermarket",
    "restaurant",
]

TYPE_TO_CATEGORY = {
    "grocery_store": "Grocery",
    "supermarket": "Grocery",
    "restaurant": "Food Joint",
}


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

    # Single request: grocery + food only (no pharmacy, bank, gym, etc.)
    results = find_nearby_places(
        api_key,
        latitude,
        longitude,
        radius_miles=search_radius_miles,
        included_types=RETAIL_TYPES,
        max_results=20,
        rank_preference="DISTANCE",
    )

    if not results or "places" not in results or not results["places"]:
        return {
            "retailers": [],
            "count": 0,
            "avg_distance_miles": None,
        }

    places = results["places"]
    dests = []
    for place in places:
        loc = place.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is not None and lon is not None:
            dests.append((float(lat), float(lon)))
        else:
            dests.append((None, None))

    route_results = _route_distances(
        api_key, latitude, longitude,
        [(a, b) for a, b in dests if a is not None and b is not None],
    )

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
    within_distances = [r["distance_miles"] for r in within]
    avg_distance_miles = round(sum(within_distances) / len(within_distances), 2) if within_distances else None

    return {
        "retailers": within,
        "count": len(within),
        "avg_distance_miles": avg_distance_miles,
    }


if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    address = "1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
    if not api_key:
        print("Set GOOGLE_MAPS_API_KEY in .env")
    else:
        geo = calib.get_lat_long(address)
        if not geo:
            print(f"Could not geocode address: {address}")
        else:
            lat, lon = geo["lat"], geo["lon"]
            print(f"Address: {address}\nLat: {lat}, Lon: {lon}\n")
            data = get_nearby_retailers(
                api_key, lat, lon,
                radius_miles=0.5,
                fetch_place_details=True,
            )
            retailers = data["retailers"]
            print(f"Retailers within 0.5 mi: {data['count']}")
            print(f"Avg distance: {data['avg_distance_miles']} mi")
            print("=" * 60)
            for i, r in enumerate(retailers, 1):
                print(f"\n--- Retailer {i}: {r.get('name')} ---")
                print(json.dumps(r, indent=2, default=str))
            print("\n" + "=" * 60)
            if retailers:
                print(f"Keys per retailer: {list(retailers[0].keys())}")
