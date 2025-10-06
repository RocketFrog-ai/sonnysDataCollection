import requests
import json
import os
from dotenv import load_dotenv
from nearbyStores.geo_utils import calculate_distance

load_dotenv()

def find_nearby_places(api_key, latitude, longitude, radius_miles=1, keyword=None, included_types=None, max_results=10):
    """
    Finds nearby places using the Google Places API.
    """
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
        
        # Filter places by distance
        if "places" in data:
            filtered_places = []
            for place in data["places"]:
                place_lat = place.get("location", {}).get("latitude")
                place_lon = place.get("location", {}).get("longitude")
                
                if place_lat and place_lon:
                    distance = calculate_distance(latitude, longitude, place_lat, place_lon)
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

def get_costco_info(latitude: float, longitude: float):
    """
    Finds a nearby car wash and then finds the nearest Costco and counts
    Costcos within a 2-mile radius.
    """
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    if not API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable not set.")
        return None

    # Step 1: Find the nearest car wash
    car_wash_radius_miles = 500 / 1609.34
    carwash_data = find_nearby_places(
        API_KEY,
        latitude,
        longitude,
        radius_miles=car_wash_radius_miles,
        included_types=['car_wash'],
        max_results=1
    )

    if not carwash_data or "places" not in carwash_data or not carwash_data["places"]:
        print(f"No car wash found within 500m of the original location.")
        return None

    target_car_wash = carwash_data['places'][0]
    car_wash_name = target_car_wash.get('displayName', {}).get('text', 'N/A')
    car_wash_location = target_car_wash.get('location', {})
    car_wash_latitude = car_wash_location.get('latitude')
    car_wash_longitude = car_wash_location.get('longitude')

    if not car_wash_latitude or not car_wash_longitude:
        print("Could not determine the location of the car wash.")
        return None

    # Step 2: Find Costcos within 2 miles of the car wash
    costco_data = find_nearby_places(
        API_KEY,
        car_wash_latitude,
        car_wash_longitude,
        radius_miles=5,
        keyword="kohl's",
        max_results=20
    )

    distance_from_nearest_costco = float('inf')
    count_of_costco_5miles = 0

    if costco_data and "places" in costco_data:
        costco_places = [
            p for p in costco_data["places"] 
            if ''.join(p.get('displayName', {}).get('text', '').lower().split()).strip() == "kohl's"
        ]
        
        if costco_places:
            print(f"Found {len(costco_places)} Costco locations:")
            for place in costco_places:
                place_name = place.get('displayName', {}).get('text', 'N/A')
                costco_location = place.get('location', {})
                costco_latitude = costco_location.get('latitude')
                costco_longitude = costco_location.get('longitude')

                print(f"  - Name: {place_name}, Latitude: {costco_latitude}, Longitude: {costco_longitude}")

                if costco_latitude and costco_longitude:
                    distance = calculate_distance(
                        car_wash_latitude, car_wash_longitude,
                        costco_latitude, costco_longitude
                    )
                    if distance < distance_from_nearest_costco:
                        distance_from_nearest_costco = distance
            
            count_of_costco_5miles = len(costco_places)

    if distance_from_nearest_costco == float('inf'):
        distance_from_nearest_costco = None

    return {
        'car_wash_name': car_wash_name,
        'distance_from_nearest_costco': distance_from_nearest_costco,
        'count_of_costco_5miles': count_of_costco_5miles
    }
