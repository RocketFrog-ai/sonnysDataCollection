import os
import sys
import requests
from typing import Optional, Any

from dotenv import load_dotenv

# Allow running as script: project root must be on path for "app" package
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from app.utils import common as calib
from app.features.experimental_features.operationalHours.searchNearby import find_nearby_places

DEFAULT_MAX_GAS_STATIONS = 10
DEFAULT_RADIUS_MILES = 2.0
# Max radius for Places API searchNearby (50000 m ≈ 31 miles)
MAX_SEARCH_RADIUS_MILES = 31.0
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/"
DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
METERS_PER_MILE = 1609.34


def _route_distances(
    api_key: str,
    origin_lat: float,
    origin_lon: float,
    dests: list[tuple[float, float]],
) -> list[dict]:
    """
    Call Google Distance Matrix API (driving). Returns one dict per destination:
    { "distance_miles": float, "duration_seconds": int, "distance_text": str, "duration_text": str }
    or {"distance_miles": None, ...} if that destination failed. dests is list of (lat, lon).
    """
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
        return [{"distance_miles": None, "duration_seconds": None, "distance_text": None, "duration_text": None} for _ in dests]
    if data.get("status") != "OK":
        return [{"distance_miles": None, "duration_seconds": None, "distance_text": None, "duration_text": None} for _ in dests]
    rows = data.get("rows") or []
    if not rows:
        return [{"distance_miles": None, "duration_seconds": None, "distance_text": None, "duration_text": None} for _ in dests]
    elements = rows[0].get("elements") or []
    out = []
    for i, el in enumerate(elements):
        if el.get("status") != "OK":
            out.append({"distance_miles": None, "duration_seconds": None, "distance_text": None, "duration_text": None})
            continue
        dist = el.get("distance") or {}
        dur = el.get("duration") or {}
        # value is in meters for distance, seconds for duration
        dist_val = dist.get("value")
        distance_miles = round(dist_val / METERS_PER_MILE, 2) if dist_val is not None else None
        duration_seconds = dur.get("value")
        out.append({
            "distance_miles": distance_miles,
            "duration_seconds": duration_seconds,
            "distance_text": dist.get("text"),
            "duration_text": dur.get("text"),
        })
    return out


def _fetch_place_details(api_key: str, place_id: str, fields: str) -> Optional[dict]:
    """Fetch Place Details. Returns None on failure or if SKU not enabled."""
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


def get_nearby_gas_stations(
    api_key: str,
    latitude: float,
    longitude: float,
    radius_miles: float = DEFAULT_RADIUS_MILES,
    max_results: int = DEFAULT_MAX_GAS_STATIONS,
    fetch_place_details: bool = True,
):
    """
    Returns nearby gas stations within radius_miles by driving (route) distance.
    All fields are from Google APIs; no inference or hardcoded mapping.

    - From Nearby Search: name, rating, userRatingCount, location, formattedAddress,
      shortFormattedAddress, regularOpeningHours, id/name (place_id).
    - distance_miles, duration_seconds, duration_text: from Distance Matrix API (driving).
    - Only stations with route distance <= radius_miles are returned; sorted by distance_miles.
    - If fetch_place_details: Place Details is called per place for fuelOptions and types (raw).
    """
    if not api_key:
        return []

    results = find_nearby_places(
        api_key,
        latitude,
        longitude,
        radius_miles=radius_miles,
        included_types=["gas_station"],
        max_results=min(max(max_results, 1), 20),
        rank_preference="DISTANCE",
    )

    if not results or "places" not in results or not results["places"]:
        return []

    places = results["places"]
    dests = []
    for place in places:
        loc = place.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is not None and lon is not None:
            dests.append((float(lat), float(lon)))
        else:
            dests.append((None, None))

    route_results = _route_distances(api_key, latitude, longitude, [(a, b) for a, b in dests if a is not None and b is not None])
    # Map back: route_results[i] corresponds to the i-th place that had valid coords; we need to align by index
    dest_idx = 0
    out = []
    for i, place in enumerate(places):
        loc = place.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is not None and lon is not None:
            route = route_results[dest_idx] if dest_idx < len(route_results) else {}
            dest_idx += 1
            distance_miles = route.get("distance_miles")
            duration_seconds = route.get("duration_seconds")
            duration_text = route.get("duration_text")
        else:
            distance_miles = None
            duration_seconds = None
            duration_text = None

        # Only include stations with route distance <= radius_miles (exclude if no route data)
        if distance_miles is None or distance_miles > radius_miles:
            continue

        display_name = place.get("displayName")
        name = display_name.get("text") if isinstance(display_name, dict) else None
        place_id = place.get("id")
        if not place_id and place.get("name"):
            place_id = str(place.get("name", "")).replace("places/", "") or None
        rating = place.get("rating")
        if rating is not None:
            rating = float(rating)
        rating_count = place.get("userRatingCount")
        formatted_address = place.get("formattedAddress")
        short_address = place.get("shortFormattedAddress")
        regular_opening_hours = place.get("regularOpeningHours")

        fuel_options = None
        types = None
        if place_id and fetch_place_details:
            details = _fetch_place_details(api_key, place_id, "fuelOptions,types")
            if details:
                fuel_options = details.get("fuelOptions")
                types = details.get("types")

        station: dict[str, Any] = {
            "name": name,
            "rating": rating,
            "rating_count": rating_count,
            "distance_miles": distance_miles,
            "address": formatted_address or short_address,
            "regular_opening_hours": regular_opening_hours,
            "place_id": place_id,
        }
        if duration_seconds is not None:
            station["duration_seconds"] = duration_seconds
        if duration_text is not None:
            station["duration_text"] = duration_text
        if fuel_options is not None:
            station["fuel_options"] = fuel_options
        if types is not None:
            station["types"] = types

        out.append(station)

    out.sort(key=lambda s: (s.get("distance_miles") is None, s.get("distance_miles") or float("inf")))
    return out[: max_results]


def get_nearest_gas_station_only(
    api_key: str,
    latitude: float,
    longitude: float,
    fetch_place_details: bool = True,
) -> Optional[dict]:
    """
    Returns the single nearest gas station by driving distance (no radius/limit).
    Uses same data sources as get_nearby_gas_stations: Nearby Search with
    included_types=["gas_station"], Distance Matrix for driving distance, and
    optionally Place Details for fuelOptions and types. Only returns a place
    that is confirmed as type gas_station (from Place Details when fetched).
    """
    if not api_key:
        return None
    stations = get_nearby_gas_stations(
        api_key,
        latitude,
        longitude,
        radius_miles=MAX_SEARCH_RADIUS_MILES,
        max_results=20,
        fetch_place_details=fetch_place_details,
    )
    if not stations:
        return None
    nearest = stations[0]
    types_list = nearest.get("types")
    if types_list is not None and "gas_station" not in types_list:
        return None
    return nearest


def get_gas_station_info(latitude: float, longitude: float):
    """Legacy summary for analysis pipeline: distance to nearest and count within radius."""
    api_key = calib.GOOGLE_MAPS_API_KEY
    if not api_key:
        return None
    stations = get_nearby_gas_stations(
        api_key, latitude, longitude,
        radius_miles=2.5, max_results=5,
        fetch_place_details=False,
    )
    count = len(stations)
    distances = [s["distance_miles"] for s in stations if s.get("distance_miles") is not None]
    distance_from_nearest = round(min(distances), 2) if distances else None
    return {
        "distance_from_nearest_gas_station": distance_from_nearest,
        "count_of_gas_stations_5miles": count,
    }


if __name__ == "__main__":
    import json
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
            stations = get_nearby_gas_stations(
                api_key, lat, lon,
                radius_miles=2,
                max_results=5,
                fetch_place_details=True,
            )
            print(f"Stations returned: {len(stations)}\n")
            print("=" * 60)
            for i, s in enumerate(stations, 1):
                print(f"\n--- Station {i} ---")
                print(json.dumps(s, indent=2, default=str))
            print("\n" + "=" * 60)
            print(f"\nTop-level keys per station: {list(stations[0].keys()) if stations else 'none'}")
