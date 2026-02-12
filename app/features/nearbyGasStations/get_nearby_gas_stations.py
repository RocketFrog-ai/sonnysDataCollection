import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.utils import common as calib
from app.features.experimental_features.operationalHours.searchNearby import find_nearby_places

DEFAULT_MAX_GAS_STATIONS = 5
DEFAULT_RADIUS_MILES = 2.5


def _parse_operational_hours(place):
    days_hours = {
        'monday_hours': 'N/A', 'tuesday_hours': 'N/A', 'wednesday_hours': 'N/A',
        'thursday_hours': 'N/A', 'friday_hours': 'N/A', 'saturday_hours': 'N/A',
        'sunday_hours': 'N/A'
    }
    opening_hours = place.get("regularOpeningHours") or {}
    weekday_descriptions = opening_hours.get("weekdayDescriptions") or []
    if not weekday_descriptions:
        return days_hours
    for desc in weekday_descriptions:
        cleaned = (desc or '').replace('\u202f', ' ').replace('\u2009', ' ').strip()
        parts = cleaned.split(':', 1)
        if len(parts) == 2:
            day_name = parts[0].strip().lower()
            hours = parts[1].strip()
            key = f"{day_name}_hours"
            if key in days_hours:
                days_hours[key] = hours
    return days_hours


def get_nearby_gas_stations(
    api_key,
    latitude,
    longitude,
    radius_miles=DEFAULT_RADIUS_MILES,
    max_results=DEFAULT_MAX_GAS_STATIONS
):
    if not api_key:
        print("Error: API key is required for get_nearby_gas_stations.")
        return []

    results = find_nearby_places(
        api_key,
        latitude,
        longitude,
        radius_miles=radius_miles,
        included_types=['gas_station'],
        max_results=min(max(max_results, 1), 20),
        rank_preference="DISTANCE"
    )

    if not results or "places" not in results or not results["places"]:
        return []

    out = []
    for place in results["places"]:
        name = (place.get("displayName") or {}).get("text") or "N/A"
        loc = place.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        distance_miles = None
        if lat is not None and lon is not None:
            distance_miles = calib.calculate_distance(latitude, longitude, lat, lon)
        rating = place.get("rating")
        if rating is None:
            rating = "N/A"
        hours = _parse_operational_hours(place)
        out.append({
            "name": name,
            "distance_miles": distance_miles,
            "rating": rating,
            **hours
        })
    return out


def get_gas_station_info(latitude: float, longitude: float):
    api_key = calib.GOOGLE_MAPS_API_KEY
    if not api_key:
        return None
    stations = get_nearby_gas_stations(api_key, latitude, longitude, radius_miles=2.5, max_results=5)
    count = len(stations)
    distances = [s["distance_miles"] for s in stations if s.get("distance_miles") is not None]
    distance_from_nearest = min(distances) if distances else None
    return {
        "distance_from_nearest_gas_station": distance_from_nearest,
        "count_of_gas_stations_5miles": count,
    }


if __name__ == "__main__":
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
            stations = get_nearby_gas_stations(api_key, lat, lon, radius_miles=2.5, max_results=5)
            for s in stations:
                print(s)
