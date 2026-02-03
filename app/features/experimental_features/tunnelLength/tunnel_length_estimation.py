import pandas as pd
import os
from dotenv import load_dotenv
import time
import traceback
import math
from utils.geo_utils import calculate_distance
from utils.google_maps_utils import get_satellite_image_name, download_satellite_image, find_nearby_places

# --- Constants ---
GOOGLE_MAPS_CONSTANT = 156543.03392
METERS_TO_FEET = 3.28084

def get_scale_for_location(latitude: float, zoom: int = 20) -> float:
    """
    Calculates the scale (feet per pixel) for a specific geographic location.
    """
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    meters_per_pixel = GOOGLE_MAPS_CONSTANT * math.cos(math.radians(latitude)) / (2**zoom)
    feet_per_pixel = meters_per_pixel * METERS_TO_FEET
    return feet_per_pixel

# Directory for satellite images
SATELLITE_IMAGE_BASE_DIR = "tunnelLength/satellite_images"
if not os.path.exists(SATELLITE_IMAGE_BASE_DIR):
    os.makedirs(SATELLITE_IMAGE_BASE_DIR)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python tunnel_length_estimation.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start_index_to_process = int(sys.argv[1])
        end_index_to_process = int(sys.argv[2])
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)

    load_dotenv()
    # --- Configuration ---
    API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    EXCEL_PATH = 'tunnelIdentification/unscaled_clean_dataset.xlsx'
    OUTPUT_DIR = 'tunnelLength/output_csv'
    OUTPUT_FILENAME = 'tunnel_length.csv'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    place_types_to_search = ['car_wash']
    max_num_results = 20
    ranking_method = "DISTANCE"

    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual value in the script.")
        sys.exit(1)

    try:
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

        if start_index_to_process < 0 or end_index_to_process > len(df) or start_index_to_process >= end_index_to_process:
            print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) -1}.")
            sys.exit(1)

        csv_headers = [
            "original_name_address", "original_latitude", "original_longitude",
            "found_car_wash_name", "satellite_image_name", "feets_per_pixel"
        ]

        if not os.path.exists(output_filepath):
            pd.DataFrame(columns=csv_headers).to_csv(output_filepath, index=False, mode='w')
            print(f"Created new CSV file with headers: {output_filepath}")
        else:
            print(f"Appending to existing CSV file: {output_filepath}")

        
        for index, row in df.iloc[start_index_to_process:end_index_to_process].iterrows():
            site_address = row.iloc[0]
            original_latitude = row.iloc[1]
            original_longitude = row.iloc[2]

            if pd.isna(site_address) or str(site_address).strip() == "":
                print(f"Skipping record {index} due to missing or empty site address.")
                continue

            print(f"\n--- Processing Record {index}: {site_address} ---")

            if pd.isna(original_latitude) or pd.isna(original_longitude):
                print(f"Skipping record {index} due to missing latitude or longitude.")
                continue

            results = find_nearby_places(
                API_KEY,
                latitude=original_latitude,
                longitude=original_longitude,
                radius_miles=0.31,  # 500 meters
                included_types=place_types_to_search,
                max_results=max_num_results,
                rank_preference=ranking_method
            )

            if results and "places" in results and len(results["places"]) > 0:
                place = results["places"][0]
                display_name = place.get("displayName", {}).get("text", "N/A")
                place_latitude = place.get("location", {}).get("latitude")
                place_longitude = place.get("location", {}).get("longitude")
                place_id = place.get("id")

                distance = calculate_distance(original_latitude, original_longitude, place_latitude, place_longitude)

                if distance <= 0.5: # 500m
                    satellite_image_name = get_satellite_image_name(place_id, SATELLITE_IMAGE_BASE_DIR)
                    if satellite_image_name is None and place_id and place_latitude and place_longitude:
                        if download_satellite_image(API_KEY, place_latitude, place_longitude, place_id, SATELLITE_IMAGE_BASE_DIR):
                            satellite_image_name = f"{place_id}.jpg"
                    
                    feets_per_pixel = get_scale_for_location(place_latitude) if place_latitude else None

                    record_data = {
                        "original_name_address": site_address,
                        "original_latitude": original_latitude,
                        "original_longitude": original_longitude,
                        "found_car_wash_name": display_name,
                        "satellite_image_name": satellite_image_name,
                        "feets_per_pixel": feets_per_pixel
                    }
                    pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)
                else:
                    print(f"Car wash '{display_name}' found but is {distance:.2f} km away, which is more than 500m.")

            else:
                print(f"No nearby car washes found for {site_address} or an error occurred.")
                record_data = {
                    "original_name_address": site_address,
                    "original_latitude": original_latitude,
                    "original_longitude": original_longitude,
                    "found_car_wash_name": "N/A",
                    "satellite_image_name": "N/A",
                    "feets_per_pixel": "N/A"
                }
                pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)


        print(f"\nProcessing complete. Results appended to {output_filepath}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print(traceback.format_exc())
