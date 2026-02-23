import os
import json
import requests
from app.utils import common as calib

API_KEY = calib.GOOGLE_MAPS_API_KEY

def find_nearby_places(api_key, latitude, longitude, radius_miles=1, keyword=None, included_types=None, max_results=10):
    if keyword:
        base_url = "https://places.googleapis.com/v1/places:searchText"
    else:
        base_url = "https://places.googleapis.com/v1/places:searchNearby"

    radius_meters = radius_miles * 1609.34

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "*"
    }

    if keyword:
        payload = {
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "radius": radius_meters
                }
            },
            "maxResultCount": min(max(1, max_results), 20)
        }
    else:
        payload = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "radius": radius_meters
                }
            },
            "maxResultCount": min(max(1, max_results), 20)
        }

    if included_types:
        payload["includedTypes"] = included_types
    if keyword:
        payload["textQuery"] = keyword

    if not keyword:
        payload["rankPreference"] = "DISTANCE"
    else:
        payload["rankPreference"] = "RELEVANCE"

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        if "places" in data:
            filtered_places = []
            for place in data["places"]:
                place_lat = place.get("location", {}).get("latitude")
                place_lon = place.get("location", {}).get("longitude")
                if place_lat and place_lon:
                    distance = calib.calculate_distance(latitude, longitude, place_lat, place_lon)
                    if distance <= radius_miles:
                        filtered_places.append(place)
            data["places"] = filtered_places
        return data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        print(f"Response content: {response.text}")
    return None

def get_walmart_info(latitude: float, longitude: float):
    # API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    if not API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable not set.")
        return None

    car_wash_radius_miles = 500 / 1609.34
    carwash_data = find_nearby_places(
        API_KEY,
        latitude,
        longitude,
        radius_miles=car_wash_radius_miles,
        included_types=['car_wash'],
        max_results=1
    )

    center_latitude = latitude
    center_longitude = longitude
    if carwash_data and "places" in carwash_data and carwash_data["places"]:
        target_car_wash = carwash_data['places'][0]
        car_wash_location = target_car_wash.get('location', {})
        cw_lat = car_wash_location.get('latitude')
        cw_lon = car_wash_location.get('longitude')
        if cw_lat is not None and cw_lon is not None:
            center_latitude = cw_lat
            center_longitude = cw_lon

    walmart_data = find_nearby_places(
        API_KEY,
        center_latitude,
        center_longitude,
        radius_miles=5,
        keyword="Walmart",
        max_results=20
    )

    distance_from_nearest_walmart = float('inf')
    count_of_walmart_5miles = 0
    nearest_place = None

    if walmart_data and "places" in walmart_data:
        walmart_places = [
            p for p in walmart_data["places"]
            if "walmart" in p.get('displayName', {}).get('text', '').lower()
        ]
        if walmart_places:
            for place in walmart_places:
                place_loc = place.get('location', {})
                plat = place_loc.get('latitude')
                plon = place_loc.get('longitude')
                if plat and plon:
                    distance = calib.calculate_distance(
                        center_latitude, center_longitude,
                        plat, plon
                    )
                    if distance < distance_from_nearest_walmart:
                        distance_from_nearest_walmart = distance
                        nearest_place = place
            count_of_walmart_5miles = len(walmart_places)

    if distance_from_nearest_walmart == float('inf'):
        distance_from_nearest_walmart = None

    nearest_details = None
    if nearest_place is not None:
        dn = nearest_place.get('displayName') or {}
        nearest_details = {
            'name': dn.get('text') if isinstance(dn, dict) else None,
            'distance_miles': distance_from_nearest_walmart,
            'rating': nearest_place.get('rating'),
            'rating_count': nearest_place.get('userRatingCount'),
            'address': nearest_place.get('formattedAddress') or nearest_place.get('shortFormattedAddress'),
            'website': nearest_place.get('websiteUri'),
            'google_maps_uri': nearest_place.get('googleMapsUri'),
        }

    return {
        'distance_from_nearest_walmart': distance_from_nearest_walmart,
        'count_of_walmart_5miles': count_of_walmart_5miles,
        'nearest_walmart': nearest_details,
    }
