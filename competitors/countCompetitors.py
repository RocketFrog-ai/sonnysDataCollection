import requests
import json
import re
import pandas as pd
import os
import glob
from dotenv import load_dotenv
import time # Added for sleep
import traceback # Added for detailed error logging
from utils.competitor_matcher import match_competitors
from utils.placePhotos import get_photo_references_and_name, download_photo
from utils.keyword_classification import keywordclassifier # Import the classifier function
from utils.images_classification import visionModelResponse # Import the image classification function
from math import radians, sin, cos, sqrt, atan2

# Directory to save downloaded images (relative to competitors/)
IMAGE_DIR = "place_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Directory for satellite images (relative to competitors/)
SATELLITE_IMAGE_BASE_DIR = "satellite_images"
if not os.path.exists(SATELLITE_IMAGE_BASE_DIR):
    os.makedirs(SATELLITE_IMAGE_BASE_DIR)

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in miles."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None
    R = 3958.8  # Radius of Earth in miles

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename or directory name by replacing non-alphanumeric
    characters (including spaces) with underscores. This ensures consistency for path creation.
    Handles None input by returning an empty string.
    """
    if name is None:
        return ""
    # Ensure name is a string before applying regex
    name_str = str(name)
    # Replace all non-alphanumeric characters (including spaces) with a single underscore
    # and strip leading/trailing underscores.
    return re.sub(r'[^a-zA-Z0-9]+', '_', name_str).strip('_')

def get_place_image_count(original_name, found_car_wash_name):
    """Counts the number of place images for a given record."""
    safe_original_name = sanitize_filename(original_name)
    safe_found_car_wash_name = sanitize_filename(found_car_wash_name)
    
    place_images_path = os.path.join(IMAGE_DIR, safe_original_name, safe_found_car_wash_name)
    
    if os.path.exists(place_images_path):
        # Count only .jpg files
        count = len(glob.glob(os.path.join(place_images_path, "*.jpg")))
        return count
    return 0

def get_satellite_image_name(place_id):
    """Gets the name of the satellite image if it exists, using place_id as filename."""
    if place_id is None:
        return None
    satellite_filename = f"{place_id}.jpg"
    satellite_filepath = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_filename)
    
    if os.path.exists(satellite_filepath):
        return satellite_filename
    return None

def download_satellite_image(api_key, latitude, longitude, place_id):
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
    output_filepath = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_filename)

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors

        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_filepath, "wb") as f:
            f.write(response.content)
        print(f"Satellite image saved as '{output_filepath}'")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve or save satellite image for {output_filepath}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
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

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        print(f"Response content: {response.text}")
    return None

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python countCompetitors.py <start_index> <end_index>")
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
    EXCEL_PATH = 'datasets/1mile_raw_data.xlsx'
    OUTPUT_DIR = 'output_csv'
    OUTPUT_FILENAME = 'competitor_analysis.csv'
    SUMMARY_OUTPUT_FILENAME = 'competitor_summary.csv'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    summary_output_filepath = os.path.join(OUTPUT_DIR, SUMMARY_OUTPUT_FILENAME)

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
            "Original_Name_Address", "Original_Latitude", "Original_Longitude",
            "Found_Car_Wash_Name", "FoundInCompetitorList", "keywordClassification",
            "keywordClassificationExplanation", "number of place images",
            "satellite image", "imageClassification", "imageClassificationJustification",
            "is_competitor"
        ]

        if not os.path.exists(output_filepath):
            pd.DataFrame(columns=csv_headers).to_csv(output_filepath, index=False, mode='w')
            print(f"Created new CSV file with headers: {output_filepath}")
        else:
            print(f"Appending to existing CSV file: {output_filepath}")

        summary_csv_headers = [
            "original_address", "competitors_count", "distance_from_nearest_competitor",
            "rating_of_nearest_competitor", "count_for_ratings_of_nearest_competitor"
        ]
        if not os.path.exists(summary_output_filepath):
            pd.DataFrame(columns=summary_csv_headers).to_csv(summary_output_filepath, index=False, mode='w')
            print(f"Created new summary CSV file with headers: {summary_output_filepath}")
        else:
            print(f"Appending to existing summary CSV file: {summary_output_filepath}")
        
        for index, row in df.iloc[start_index_to_process:end_index_to_process].iterrows():
            site_address = row.iloc[0]
            original_latitude = row.iloc[1]
            original_longitude = row.iloc[2]

            if pd.isna(site_address) or str(site_address).strip() == "":
                print(f"Skipping record {index} due to missing or empty site address.")
                continue

            print(f"\n--- Processing Record {index}: {site_address} ---")

            competitors_count = 0
            nearest_competitor_distance = None
            nearest_competitor_rating = None
            nearest_competitor_rating_count = None
            first_competitor_found = False

            if pd.isna(original_latitude) or pd.isna(original_longitude):
                print(f"Skipping record {index} due to missing latitude or longitude.")
                continue

            results = find_nearby_places(
                API_KEY,
                latitude=original_latitude,
                longitude=original_longitude,
                radius_miles=1,
                included_types=place_types_to_search,
                max_results=max_num_results,
                rank_preference=ranking_method
            )

            if results and "places" in results:
                for i, place in enumerate(results["places"]):
                    if i == 0:
                        continue
                        
                    display_name = place.get("displayName", {}).get("text", "N/A")
                    place_latitude = place.get("location", {}).get("latitude")
                    place_longitude = place.get("location", {}).get("longitude")
                    place_id = place.get("id")
                    is_competitor = False
                    
                    # Initialize filter results
                    keyword_classification = None
                    keyword_explanation = None
                    num_place_images = 0
                    satellite_image_name = None
                    image_classification = None
                    image_justification = None

                    # Filter 1: Name Matching
                    _, found_competitors, _ = match_competitors([display_name])
                    found_in_competitor_list = bool(found_competitors)
                    
                    if found_in_competitor_list:
                        is_competitor = True
                    else:
                        # Filter 2: Keyword Classification
                        classification_result = keywordclassifier(display_name)
                        keyword_classification = classification_result.get("classification")
                        keyword_explanation = classification_result.get("explanation")
                        time.sleep(1)

                        if keyword_classification == "Competitor":
                            is_competitor = True
                        elif keyword_classification == "Can't say":
                            # Filter 3: Vision Model
                            satellite_image_name = get_satellite_image_name(place_id)
                            if satellite_image_name is None and place_id and place_latitude and place_longitude:
                                if download_satellite_image(API_KEY, place_latitude, place_longitude, place_id):
                                    satellite_image_name = f"{place_id}.jpg"
                            
                            if place_id:
                                photo_references, _ = get_photo_references_and_name(place_id)
                                if photo_references:
                                    for photo_idx, ref in enumerate(photo_references):
                                        download_photo(ref, site_address, display_name, photo_idx, IMAGE_DIR)
                                    num_place_images = get_place_image_count(site_address, display_name)

                            current_place_images_folder_path = os.path.join(IMAGE_DIR, sanitize_filename(site_address), sanitize_filename(display_name))
                            current_satellite_image_full_path = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_image_name) if satellite_image_name else ""

                            if num_place_images > 0 or (current_satellite_image_full_path and os.path.exists(current_satellite_image_full_path)):
                                try:
                                    image_classification_result = visionModelResponse(
                                        place_images_folder_path=current_place_images_folder_path,
                                        satellite_image_path=current_satellite_image_full_path
                                    )
                                    image_classification = image_classification_result.get("classification")
                                    image_justification = image_classification_result.get("justification")
                                    if image_classification == "Competitor":
                                        is_competitor = True
                                    time.sleep(1)
                                except Exception as e:
                                    print(f"ERROR calling visionModelResponse for {display_name}: {e}")
                                    print(traceback.format_exc())
                                    image_classification = "Error"
                                    image_justification = str(e)
                    
                    if is_competitor:
                        competitors_count += 1
                        if not first_competitor_found:
                            first_competitor_found = True
                            nearest_competitor_distance = calculate_distance(original_latitude, original_longitude, place_latitude, place_longitude)
                            nearest_competitor_rating = place.get("rating")
                            nearest_competitor_rating_count = place.get("userRatingCount")
                    
                    record_data = {
                        "Original_Name_Address": site_address,
                        "Original_Latitude": original_latitude,
                        "Original_Longitude": original_longitude,
                        "Found_Car_Wash_Name": display_name,
                        "FoundInCompetitorList": found_in_competitor_list,
                        "keywordClassification": keyword_classification,
                        "keywordClassificationExplanation": keyword_explanation,
                        "number of place images": num_place_images,
                        "satellite image": satellite_image_name,
                        "imageClassification": image_classification,
                        "imageClassificationJustification": image_justification,
                        "is_competitor": is_competitor
                    }
                    pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)

            else:
                print(f"No nearby car washes found for {site_address} or an error occurred.")
            
            summary_data = {
                "original_address": site_address,
                "competitors_count": competitors_count,
                "distance_from_nearest_competitor": nearest_competitor_distance,
                "rating_of_nearest_competitor": nearest_competitor_rating,
                "count_for_ratings_of_nearest_competitor": nearest_competitor_rating_count
            }
            pd.DataFrame([summary_data], columns=summary_csv_headers).to_csv(summary_output_filepath, index=False, mode='a', header=False)

        print(f"\nProcessing complete. Results appended to {output_filepath}")
        print(f"Summary results appended to {summary_output_filepath}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print(traceback.format_exc())
