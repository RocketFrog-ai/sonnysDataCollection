import pandas as pd
import os
import sys
import requests

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from competitors.utils.google_maps_utils import find_nearby_places

# Replace with your Google Static Maps API key
API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

def get_static_map_image(latitude, longitude, zoom, output_filepath):
    """
    Fetches a static map image from Google Static Maps API and saves it.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": "640x640",
        "maptype": "hybrid",
        "key": API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open(output_filepath, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to retrieve image for {output_filepath}. Status code: {response.status_code}")
        print(response.text)
        return False

def process_excel_data():
    excel_file_path = 'competitors/datasets/1mile_raw_data.xlsx'
    base_output_dir = 'siteAccessibility/satellite_images'

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    try:
        df = pd.read_excel(excel_file_path, nrows=20)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    for index, row in df.iterrows():
        site_name = row.iloc[0]
        latitude = row.iloc[1]
        longitude = row.iloc[2]

        print(f"Processing record {index + 1}: {site_name}")
        print(f"  - Original coordinates: {latitude}, {longitude}")

        # Find nearby car washes
        nearby_places = find_nearby_places(API_KEY, latitude, longitude, radius_miles=0.31, included_types=["car_wash"], rank_preference="DISTANCE") # 500 meters is roughly 0.31 miles

        if nearby_places and nearby_places.get('places'):
            nearest_place = nearby_places['places'][0]
            if 'location' in nearest_place:
                new_latitude = nearest_place['location']['latitude']
                new_longitude = nearest_place['location']['longitude']
                print(f"  - Found nearby car wash. New coordinates: {new_latitude}, {new_longitude}")
                latitude = new_latitude
                longitude = new_longitude
            else:
                print("  - Nearby place found, but location data is missing. Using original coordinates.")
        else:
            print("  - No nearby car wash found. Using original coordinates.")


        safe_site_name = "".join([c for c in str(site_name) if c.isalnum() or c in (' ', '.', '_')]).rstrip()
        location_dir = os.path.join(base_output_dir, safe_site_name)

        if not os.path.exists(location_dir):
            os.makedirs(location_dir)

        for zoom_level in [18, 19, 20]:
            image_filename = os.path.join(location_dir, f"zoom_{zoom_level}.png")
            print(f"  - Downloading image with zoom level {zoom_level}...")
            if get_static_map_image(latitude, longitude, zoom_level, image_filename):
                print(f"    - Saved to {image_filename}")
            else:
                print(f"    - Failed to save image for zoom level {zoom_level}")

if __name__ == "__main__":
    process_excel_data()
