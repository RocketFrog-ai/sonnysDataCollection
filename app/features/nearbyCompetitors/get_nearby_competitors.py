"""
Nearby car wash competitors within a radius using Google Places API and Distance Matrix.
Route (driving) distance only. Real data: name, distance, rating, user_rating_count, website when from API.
No market share or threat level — API does not provide those.
"""
import requests
from typing import Optional, Any

from app.features.experimental_features.operationalHours.searchNearby import find_nearby_places

DEFAULT_RADIUS_MILES = 4.0
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/"
DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
METERS_PER_MILE = 1609.34


def _route_distances(
    api_key: str,
    origin_lat: float,
    origin_lon: float,
    dests: list[tuple[float, float]],
) -> list[dict]:
    """Route (driving) distance via Google Distance Matrix API. Returns list of {distance_miles, duration_text}."""
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


def get_nearby_competitors(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float = DEFAULT_RADIUS_MILES,
    fetch_place_details: bool = True,
    max_results: int = 20,
) -> dict:
    """
    Nearby car washes (competitors) within radius_miles by driving distance.
    Returns only real API data: name, distance_miles, rating, user_rating_count, address;
    optional website, google_maps_uri, primary_type_display_name from Place Details.
    No market share or threat level.
    """
    if not api_key:
        return {"competitors": [], "count": 0}

    results = find_nearby_places(
        api_key,
        latitude,
        longitude,
        radius_miles=radius_miles,
        included_types=["car_wash"],
        max_results=min(max_results, 20),
        rank_preference="DISTANCE",
    )

    if not results or "places" not in results or not results["places"]:
        return {"competitors": [], "count": 0}

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
    competitors = []
    for place in places:
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

        # Keep car washes only; exclude gas stations (Places can return gas stations when they have a car wash)
        place_types = place.get("types") or []
        if "gas_station" in place_types:
            continue

        display_name = place.get("displayName")
        name = display_name.get("text") if isinstance(display_name, dict) else None
        place_id = place.get("id") or (str(place.get("name", "")).replace("places/", "") if place.get("name") else None)
        rating = place.get("rating")
        if rating is not None:
            rating = float(rating)
        user_rating_count = place.get("userRatingCount")
        if user_rating_count is not None:
            user_rating_count = int(user_rating_count)
        address = place.get("formattedAddress") or place.get("shortFormattedAddress")

        website_uri = None
        google_maps_uri = None
        primary_type_display_name = None
        if place_id and fetch_place_details:
            details = _fetch_place_details(
                api_key, place_id,
                "websiteUri,googleMapsUri,primaryTypeDisplayName",
            )
            if details:
                website_uri = details.get("websiteUri")
                google_maps_uri = details.get("googleMapsUri")
                ptd = details.get("primaryTypeDisplayName")
                if isinstance(ptd, dict) and "text" in ptd:
                    primary_type_display_name = ptd.get("text")
                elif isinstance(ptd, str):
                    primary_type_display_name = ptd

        comp: dict[str, Any] = {
            "name": name,
            "distance_miles": distance_miles,
            "rating": rating,
            "user_rating_count": user_rating_count,
            "address": address,
            "place_id": place_id,
        }
        if primary_type_display_name:
            comp["primary_type_display_name"] = primary_type_display_name
        if website_uri:
            comp["website"] = website_uri
        if google_maps_uri:
            comp["google_maps_uri"] = google_maps_uri

        competitors.append(comp)

    competitors.sort(key=lambda c: (c.get("distance_miles") is None, c.get("distance_miles") or float("inf")))

    return {
        "competitors": competitors,
        "count": len(competitors),
    }


if __name__ == "__main__":
    import json
    import os
    from dotenv import load_dotenv
    from app.utils import common as calib

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
            data = get_nearby_competitors(
                api_key, lat, lon,
                radius_miles=4.0,
                fetch_place_details=True,
            )
            comps = data["competitors"]
            print(f"Competitors (car washes) within 4 mi: {data['count']}")
            print("=" * 60)
            for i, c in enumerate(comps, 1):
                print(f"\n--- {i}: {c.get('name')} ---")
                print(json.dumps(c, indent=2, default=str))
            print("\n" + "=" * 60)
            if comps:
                print(f"Keys per competitor: {list(comps[0].keys())}")
