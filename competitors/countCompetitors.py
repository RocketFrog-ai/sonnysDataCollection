import requests
import json
import re
import pandas as pd
import os
import glob # Added for file counting
import time # Added for sleep
import traceback # Added for detailed error logging
from competitor_matcher import match_competitors
from apiExamples.placePhotos import get_photo_references_and_name, download_photo
from apiExamples.keyword_classification import keywordclassifier # Import the classifier function
from apiExamples.images_classification import visionModelResponse # Import the image classification function

# Directory to save downloaded images (relative to app/)
IMAGE_DIR = "place_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Directory for satellite images (relative to app/)
SATELLITE_IMAGE_BASE_DIR = "satellite_images"
if not os.path.exists(SATELLITE_IMAGE_BASE_DIR):
    os.makedirs(SATELLITE_IMAGE_BASE_DIR)

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
    
    print(f"Checking place images path: {place_images_path}") # Debug print
    
    if os.path.exists(place_images_path):
        # Count only .jpg files
        count = len(glob.glob(os.path.join(place_images_path, "*.jpg")))
        print(f"Found {count} images in {place_images_path}") # Debug print
        return count
    print(f"Path does not exist: {place_images_path}") # Debug print
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

    Args:
        api_key (str): Your Google Maps Platform API Key.
        latitude (float): The latitude of the center point for the search.
        longitude (float): The longitude of the center point for the search.
        radius_miles (float, optional): The radius for the search in miles. Defaults to 1.
                                       This will be converted to meters.
        included_types (list, optional): A list of place types to search for (e.g., ["restaurant", "cafe"]).
                                         If None or empty, the API attempts to return all types. Defaults to None.
        max_results (int, optional): The maximum number of results to return (1-20). Defaults to 10.
        rank_preference (str, optional): How to rank results. "POPULARITY" or "DISTANCE".
                                         Defaults to "POPULARITY". If "DISTANCE" is used,
                                         included_types should not be specified as per some API guidelines
                                         (though the new API might be more flexible, typically distance ranking
                                         is for a general search).

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    base_url = "https://places.googleapis.com/v1/places:searchNearby"

    # Convert radius from miles to meters (1 mile = 1609.34 meters)
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

    # --- Configuration ---
    API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"  # Standardized API Key
    EXCEL_PATH = '/home/arpit/dataCollection/app/datasets/1mile_raw_data.xlsx'
    OUTPUT_DIR = '/home/arpit/dataCollection/app/output_csv'
    OUTPUT_FILENAME = 'competitor_analysis.csv'

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    # Optional: Specify types of places you're interested in.
    place_types_to_search = ['car_wash']
    max_num_results = 20
    ranking_method = "DISTANCE"

    # --- Validate API Key ---
    if API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual value in the script.")
        sys.exit(1)

    try:
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

        # Ensure indices are within the DataFrame bounds
        if start_index_to_process < 0 or end_index_to_process > len(df) or start_index_to_process >= end_index_to_process:
            print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) -1}.")
            sys.exit(1)

        # Define CSV headers
        csv_headers = [
            "Original_Name_Address", "Original_Latitude", "Original_Longitude",
            "Found_Car_Wash_Name", "FoundInCompetitorList", "keywordClassification",
            "keywordClassificationExplanation", "number of place images",
            "satellite image", "imageClassification", "imageClassificationJustification"
        ]

        # Check if CSV file exists, if not, create it with headers
        if not os.path.exists(output_filepath):
            pd.DataFrame(columns=csv_headers).to_csv(output_filepath, index=False, mode='w')
            print(f"Created new CSV file with headers: {output_filepath}")
        else:
            print(f"Appending to existing CSV file: {output_filepath}")

        for index, row in df.iloc[start_index_to_process:end_index_to_process].iterrows():
            site_address = row.iloc[0]
            original_latitude = row.iloc[1] # Renamed for clarity
            original_longitude = row.iloc[2] # Renamed for clarity

            if pd.isna(site_address) or str(site_address).strip() == "":
                print(f"Skipping record {index} due to missing or empty site address.")
                record_data = {
                    "Original_Name_Address": site_address,
                    "Original_Latitude": original_latitude,
                    "Original_Longitude": original_longitude,
                    "Found_Car_Wash_Name": "N/A",
                    "FoundInCompetitorList": False,
                    "keywordClassification": "Skipped (Missing Address)",
                    "keywordClassificationExplanation": "Record skipped due to missing or empty site address.",
                    "number of place images": None,
                    "satellite image": None,
                    "imageClassification": None,
                    "imageClassificationJustification": None
                }
                pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)
                continue # Skip to the next record

            print(f"Processing record {index}: {site_address}, Latitude: {original_latitude}, Longitude: {original_longitude}")

            if pd.isna(original_latitude) or pd.isna(original_longitude):
                print(f"Skipping record {index} due to missing latitude or longitude.")
                record_data = {
                    "Original_Name_Address": site_address,
                    "Original_Latitude": original_latitude,
                    "Original_Longitude": original_longitude,
                    "Found_Car_Wash_Name": "N/A",
                    "FoundInCompetitorList": False,
                    "keywordClassification": "Skipped (Missing Lat/Lon)",
                    "keywordClassificationExplanation": "Record skipped due to missing latitude or longitude.",
                    "number of place images": None,
                    "satellite image": None,
                    "imageClassification": None,
                    "imageClassificationJustification": None
                }
                pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)
                continue # Skip to the next record

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
                for place in results["places"]:
                    display_name = place.get("displayName", {}).get("text", "N/A")
                    
                    # Extract latitude and longitude for the found place
                    place_latitude = place.get("location", {}).get("latitude")
                    place_longitude = place.get("location", {}).get("longitude")

                    # Determine if it's a competitor using competitor_matcher.py
                    _, found_competitors, _ = match_competitors([display_name])
                    is_competitor = bool(found_competitors) # True if found, False otherwise

                    place_id = place.get("id") # Get the place ID

                    num_place_images = None
                    satellite_image_name = None
                    keyword_classification = None
                    keyword_explanation = None
                    image_classification = None
                    image_justification = None

                    # Apply filteration: if it's a competitor, subsequent columns are N/A
                    if not is_competitor:
                        # Perform keyword classification if not a competitor
                        classification_result = keywordclassifier(display_name)
                        keyword_classification = classification_result.get("classification")
                        keyword_explanation = classification_result.get("explanation")
                        time.sleep(1) # Add sleep to avoid rate limiting

                        # Collect images and perform image classification only if keywordClassification is 'Can't say'
                        if keyword_classification == "Can't say":
                            satellite_image_name = get_satellite_image_name(place_id) # Pass place_id

                            # If satellite image doesn't exist, attempt to download it
                            if satellite_image_name is None:
                                if place_latitude is not None and place_longitude is not None:
                                    if download_satellite_image(API_KEY, place_latitude, place_longitude, place_id): # Pass place_latitude, place_longitude, place_id
                                        satellite_image_name = f"{place_id}.jpg"
                                    else:
                                        satellite_image_name = None # Ensure it's None if download fails
                                else:
                                    print(f"Skipping satellite image download for {display_name} due to missing latitude/longitude.")
                                    satellite_image_name = None
                            
                            # Download place photos if place_id is available and keywordClassification is 'Can't say'
                            if place_id: # No need for is_competitor here, as we are already in `if not is_competitor` block
                                photo_references, place_name_for_photo = get_photo_references_and_name(place_id)
                                if photo_references:
                                    print(f"Downloading photos for '{display_name}'...")
                                    for i, ref in enumerate(photo_references):
                                        download_photo(ref, site_address, display_name, i, IMAGE_DIR) # Pass IMAGE_DIR
                                    # After downloading, count the images
                                    num_place_images = get_place_image_count(site_address, display_name)
                                else:
                                    print(f"No photos found for '{display_name}' (ID: {place_id}).")
                                    num_place_images = 0 # Set to 0 if no photos found/downloaded
                            
                                # Determine paths for image classification
                                current_place_images_folder_path = os.path.join(IMAGE_DIR, sanitize_filename(site_address), sanitize_filename(display_name))
                                current_satellite_image_full_path = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_image_name) if satellite_image_name else ""

                                # Perform image classification if (place images or satellite image are available)
                                if (num_place_images is not None and num_place_images > 0) or satellite_image_name is not None:
                                    print(f"Performing image classification for {display_name}...")
                                    try:
                                        image_classification_result = visionModelResponse(
                                            place_images_folder_path=current_place_images_folder_path,
                                            satellite_image_path=current_satellite_image_full_path
                                        )
                                        image_classification = image_classification_result.get("classification")
                                        image_justification = image_classification_result.get("justification")
                                        time.sleep(1) # Add sleep to avoid rate limiting
                                    except TypeError as e: # Specifically catch TypeError
                                        print(f"ERROR: TypeError calling visionModelResponse for {display_name}: {e}")
                                        print(traceback.format_exc()) # Print full traceback
                                        image_classification = "Error during classification (TypeError)"
                                        image_justification = f"A TypeError occurred during image classification: {e}"
                                    except Exception as e: # Catch any other general exceptions
                                        print(f"ERROR: Unexpected Exception calling visionModelResponse for {display_name}: {e}")
                                        print(traceback.format_exc()) # Print full traceback
                                        image_classification = "Error during classification (Unexpected)"
                                        image_justification = f"An unexpected error occurred during image classification: {e}"
                                else:
                                    print(f"Skipping image classification for {display_name} due to missing images or satellite image.")
                                    image_classification = None
                                    image_justification = None
                        else:
                            # If not 'Can't say', then no images are collected, so set to None
                            num_place_images = None
                            satellite_image_name = None
                            image_classification = None
                            image_justification = None
                    else:
                        # If it is a competitor, set all subsequent columns to N/A or None
                        keyword_classification = None
                        keyword_explanation = None
                        num_place_images = None
                        satellite_image_name = None
                        image_classification = None
                        image_justification = None
                    
                    record_data = {
                        "Original_Name_Address": site_address,
                        "Original_Latitude": original_latitude,
                        "Original_Longitude": original_longitude,
                        "Found_Car_Wash_Name": display_name,
                        "FoundInCompetitorList": is_competitor,
                        "keywordClassification": keyword_classification,
                        "keywordClassificationExplanation": keyword_explanation,
                        "number of place images": num_place_images,
                        "satellite image": satellite_image_name,
                        "imageClassification": image_classification,
                        "imageClassificationJustification": image_justification
                    }
                    pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)

            else:
                print(f"No nearby car washes found for {site_address} or an error occurred.")
                # Also add default values for new columns, respecting the filteration logic
                record_data = {
                    "Original_Name_Address": site_address,
                    "Original_Latitude": original_latitude,
                    "Original_Longitude": original_longitude,
                    "Found_Car_Wash_Name": "N/A",
                    "FoundInCompetitorList": False,
                    "keywordClassification": None, # Default for no found car wash
                    "keywordClassificationExplanation": None, # Default for no found car wash
                    "number of place images": None, # Default for no found car wash
                    "satellite image": None, # Default for no found car wash
                    "imageClassification": None, # Default for no found car wash
                    "imageClassificationJustification": None # Default for no found car wash
                }
                pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)

        print(f"\nProcessing complete. Results appended to {output_filepath}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
