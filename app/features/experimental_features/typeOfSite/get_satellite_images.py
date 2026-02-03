import pandas as pd
import os
import sys
import requests
import time
from dotenv import load_dotenv
import cv2
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from competitors.utils.google_maps_utils import find_nearby_places

# Replace with your Google Static Maps API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

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
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Decode the image
            image_data = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            if image is None:
                print(f"Attempt {attempt + 1} failed: OpenCV could not decode the image.")
                continue

            # Get image dimensions
            height, width, _ = image.shape
            center_x, center_y = width // 2, height // 2

            # Draw a filled red circle at the center
            cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), -1)  # Red circle, filled

            # Encode the image back to PNG format
            success, buffer = cv2.imencode('.png', image)
            if not success:
                print(f"Attempt {attempt + 1} failed: OpenCV could not encode the image.")
                continue
            
            with open(output_filepath, "wb") as f:
                f.write(buffer)

            # Final check: ensure file is not empty
            if os.path.getsize(output_filepath) > 0:
                return True
            else:
                print(f"Attempt {attempt + 1} failed: Saved image file is empty.")
                continue

        except Exception as e:
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

    for zoom_level in [17, 18, 19]:
        image_filename = os.path.join(location_dir, f"zoom_{zoom_level}.png")
        print(f"  - Downloading image with zoom level {zoom_level}...")
        if get_static_map_image(new_latitude, new_longitude, zoom_level, image_filename):
            print(f"    - Saved to {image_filename}")
        else:
            print(f"    - Failed to save image for zoom level {zoom_level}")
    
    return new_latitude, new_longitude, location_dir, place_name
