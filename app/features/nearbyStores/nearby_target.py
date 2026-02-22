"""
Nearest Target store by straight-line distance. Uses Places API searchText "Target"
and filters to places whose displayName contains "target" (so we get Target stores only).
"""
from app.utils import common as calib

API_KEY = calib.GOOGLE_MAPS_API_KEY


def _find_nearby_places_text(api_key, latitude, longitude, radius_miles=5, keyword="Target", max_results=20):
    """Search places by text query (Places API searchText). Same pattern as nearby_costcos."""
    import json
    import requests

    if not api_key:
        return None
    base_url = "https://places.googleapis.com/v1/places:searchText"
    radius_meters = radius_miles * 1609.34
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "*",
    }
    payload = {
        "locationBias": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius_meters,
            }
        },
        "textQuery": keyword,
        "maxResultCount": min(max(1, max_results), 20),
        "rankPreference": "RELEVANCE",
    }
    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=15)
        response.raise_for_status()
        data = response.json()
        if "places" not in data:
            return None
        filtered = []
        for place in data["places"]:
            place_lat = place.get("location", {}).get("latitude")
            place_lon = place.get("location", {}).get("longitude")
            if place_lat and place_lon:
                dist = calib.calculate_distance(latitude, longitude, place_lat, place_lon)
                if dist <= radius_miles:
                    filtered.append(place)
        data["places"] = filtered
        return data
    except Exception:
        return None


def get_target_info(latitude: float, longitude: float):
    """
    Returns distance to nearest Target (displayName must contain "target") within 5 miles.
    """
    if not API_KEY:
        return None
    target_data = _find_nearby_places_text(
        API_KEY, latitude, longitude,
        radius_miles=5, keyword="Target", max_results=20,
    )
    distance_from_nearest_target = None
    count_of_target_5miles = 0
    if target_data and "places" in target_data:
        target_places = [
            p for p in target_data["places"]
            if "target" in (p.get("displayName") or {}).get("text", "").lower()
        ]
        if target_places:
            count_of_target_5miles = len(target_places)
            distance_from_nearest_target = float("inf")
            for place in target_places:
                loc = place.get("location", {})
                plat, plon = loc.get("latitude"), loc.get("longitude")
                if plat is not None and plon is not None:
                    d = calib.calculate_distance(latitude, longitude, plat, plon)
                    if d < distance_from_nearest_target:
                        distance_from_nearest_target = d
            if distance_from_nearest_target == float("inf"):
                distance_from_nearest_target = None
    return {
        "distance_from_nearest_target": distance_from_nearest_target,
        "count_of_target_5miles": count_of_target_5miles,
    }
