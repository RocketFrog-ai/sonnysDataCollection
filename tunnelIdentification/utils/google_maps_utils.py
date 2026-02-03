import os
import requests
import json
import time
from requests.exceptions import RequestException

def get_satellite_image_name(place_id, satellite_image_base_dir):
    """Gets the name of the satellite image if it exists, using place_id as filename."""
    if place_id is None:
        return None
    satellite_filename = f"{place_id}.jpg"
    satellite_filepath = os.path.join(satellite_image_base_dir, satellite_filename)
    
    if os.path.exists(satellite_filepath):
        return satellite_filename
    return None

def download_satellite_image(api_key, latitude, longitude, place_id, satellite_image_base_dir):
    """Downloads a satellite image from Google Static Maps API, saving with place_id as filename."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"

    params = {
        "center": f"{latitude},{longitude}",
        "zoom": 20,
        "size": "640x640",
        "maptype": "hybrid",
        "key": api_key
    }

    satellite_filename = f"{place_id}.jpg"
    output_filepath = os.path.join(satellite_image_base_dir, satellite_filename)

    for attempt in range(3):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            output_dir = os.path.dirname(output_filepath)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_filepath, "wb") as f:
                f.write(response.content)
            print(f"Satellite image saved as '{output_filepath}'")
            return True
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed to download satellite image: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("All attempts failed.")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Final response content: {e.response.text}")
    return False

def find_nearby_places(api_key, latitude, longitude, radius_miles=1, included_types=None, max_results=10, rank_preference="POPULARITY"):
    """
    Finds nearby places using the Google Places API (Nearby Search New).
    """
    base_url = "https://places.googleapis.com/v1/places:searchNearby"

    radius_meters = radius_miles * 1609.34
    if not (0.0 < radius_meters <= 50000.0):
        print("Error: Radius must be between 0.0 (exclusive) and 50000.0 meters (inclusive).")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "*"
    }

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

    if included_types and len(included_types) > 0:
        payload["includedTypes"] = included_types
    
    if rank_preference:
        payload["rankPreference"] = rank_preference

    for attempt in range(3):
        try:
            response = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            return response.json()  
        except RequestException as e:
            print(f"Attempt {attempt + 1} for find_nearby_places failed: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("All attempts failed.")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        print(f"Final response content: {e.response.json()}")
                    except json.JSONDecodeError:
                        print(f"Final response content (not JSON): {e.response.text}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            if 'response' in locals() and response.text:
                print(f"Response content: {response.text}")
            return None # Do not retry on JSON decode error

    return None
