import pandas as pd
import os
import sys
import requests
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from competitors.utils.google_maps_utils import find_nearby_places

# Replace with your Google Static Maps API key
API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

def get_static_map_image(latitude, longitude, zoom, output_filepath, retries=3, backoff_factor=0.5):
    """
    Fetches a static map image from Google Static Maps API and saves it, with retry logic.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": "640x640",
        "maptype": "hybrid",
        "key": API_KEY
    }
    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code == 200:
                with open(output_filepath, "wb") as f:
                    f.write(response.content)
                return True
            else:
                print(f"Attempt {attempt + 1} failed: Status {response.status_code} for {output_filepath}")
                print(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed with exception: {e}")
        
        if attempt < retries - 1:
            sleep_time = backoff_factor * (2 ** attempt)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    print(f"Failed to retrieve image for {output_filepath} after {retries} attempts.")
    return False

def download_satellite_images(latitude, longitude, site_name):
    base_output_dir = 'typeOfSite/satellite_images'

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # print(f"Processing record: {site_name}")
    print(f"  - Original coordinates: {latitude}, {longitude}")

    # Find nearby car washes
    try:
        nearby_places = find_nearby_places(API_KEY, latitude, longitude, radius_miles=0.31, included_types=["car_wash"], rank_preference="DISTANCE") # 500 meters is roughly 0.31 miles
    except requests.exceptions.RequestException as e:
        print(f"  - ERROR: Could not connect to Google Maps API for nearby search: {e}")
        nearby_places = None

    new_latitude = latitude
    new_longitude = longitude

    place_name = None
    if nearby_places and nearby_places.get('places'):
        nearest_place = nearby_places['places'][0]
        if 'location' in nearest_place:
            new_latitude = nearest_place['location']['latitude']
            new_longitude = nearest_place['location']['longitude']
            place_name = nearest_place.get('displayName', {}).get('text')
            print(f"  - Found nearby car wash: {place_name}. New coordinates: {new_latitude}, {new_longitude}")
    else:
        print("  - No nearby car wash found. Using original coordinates.")


    safe_site_name = "".join([c for c in str(site_name) if c.isalnum() or c in (' ', '.', '_')]).rstrip()
    location_dir = os.path.join(base_output_dir, safe_site_name)

    if not os.path.exists(location_dir):
        os.makedirs(location_dir)

    for zoom_level in [18, 19, 20]:
        image_filename = os.path.join(location_dir, f"zoom_{zoom_level}.png")
        print(f"  - Downloading image with zoom level {zoom_level}...")
        if get_static_map_image(new_latitude, new_longitude, zoom_level, image_filename):
            print(f"    - Saved to {image_filename}")
        else:
            print(f"    - Failed to save image for zoom level {zoom_level}")
    
    return new_latitude, new_longitude, location_dir, place_name
