import pandas as pd
import os
from dotenv import load_dotenv
import time
import traceback
from utils.competitor_matcher import match_competitors
from utils.placePhotos import get_photo_references_and_name, download_photo
from utils.keyword_classification import keywordclassifier
# from utils.gemini_images_classification import visionModelResponse
from utils.gpt_images_classification import visionModelResponse
from utils.file_utils import sanitize_filename, get_place_image_count
from utils.geo_utils import calculate_distance
from utils.google_maps_utils import get_satellite_image_name, download_satellite_image, find_nearby_places

# Directory to save downloaded images (relative to competitors/)
IMAGE_DIR = "tunnelIdentification/place_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Directory for satellite images (relative to competitors/)
SATELLITE_IMAGE_BASE_DIR = "tunnelIdentification/satellite_images"
if not os.path.exists(SATELLITE_IMAGE_BASE_DIR):
    os.makedirs(SATELLITE_IMAGE_BASE_DIR)

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
    EXCEL_PATH = 'tunnelIdentification/unscaled_clean_dataset.xlsx'
    OUTPUT_DIR = 'tunnelIdentification/output_csv'
    OUTPUT_FILENAME = 'tunnel_identification.csv'

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
            "found_car_wash_name",
            "found_in_competitor_list", "keyword_classification",
            "keyword_classification_explanation", "place_images_count",
            "satellite_image_filename", "image_classification", "image_classification_justification",
            "is_competitor"
        ]

        if not os.path.exists(output_filepath):
            pd.DataFrame(columns=csv_headers).to_csv(output_filepath, index=False, mode='w')
            print(f"Created new CSV file with headers: {output_filepath}")
        else:
            print(f"Appending to existing CSV file: {output_filepath}")

        
        for index, row in df.iloc[start_index_to_process:end_index_to_process].iterrows():
            satellite_image_filename = None
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
                radius_miles=1,
                included_types=place_types_to_search,
                max_results=max_num_results,
                rank_preference=ranking_method
            )

            if results and "places" in results:
                # Only process the first car wash found
                if len(results["places"]) > 0:
                    place = results["places"][0]
                    display_name = place.get("displayName", {}).get("text", "N/A")
                    place_latitude = place.get("location", {}).get("latitude")
                    place_longitude = place.get("location", {}).get("longitude")
                    place_id = place.get("id")
                    is_competitor = False
                    distance = calculate_distance(original_latitude, original_longitude, place_latitude, place_longitude)
                    rating = place.get("rating")
                    user_rating_count = place.get("userRatingCount")

                    
                    # Filter 1: Name Matching
                    _, found_competitors, _ = match_competitors([display_name])
                    found_in_competitor_list = bool(found_competitors)

                    if found_in_competitor_list:
                        is_competitor = True
                        keyword_classification = None
                        keyword_explanation = None
                        place_images_count = None
                        image_classification = None
                        image_justification = None
                    else:
                        # Filter 2: Keyword Classification
                        classification_result = keywordclassifier(display_name)
                        # print(f"####{classification_result}")
                        keyword_classification = classification_result.get("classification")
                        keyword_explanation = classification_result.get("explanation")
                        time.sleep(1)

                        if keyword_classification == "Competitor":
                            is_competitor = True
                            place_images_count = None
                            satellite_image_filename = None
                            image_classification = None
                            image_justification = None
                        elif keyword_classification == "Can't say":
                            # Filter 3: Vision Model
                            satellite_image_name = get_satellite_image_name(place_id, SATELLITE_IMAGE_BASE_DIR)
                            if satellite_image_name is None and place_id and place_latitude and place_longitude:
                                if download_satellite_image(API_KEY, place_latitude, place_longitude, place_id, SATELLITE_IMAGE_BASE_DIR):
                                    satellite_image_name = f"{place_id}.jpg"
                            
                            satellite_image_filename = satellite_image_name
                            place_images_count = 0  # Initialize place_images_count
                            if place_id:
                                photo_references, _ = get_photo_references_and_name(place_id)
                                if photo_references:
                                    for photo_idx, ref in enumerate(photo_references):
                                        download_photo(ref, site_address, display_name, photo_idx, IMAGE_DIR)
                                    place_images_count = get_place_image_count(site_address, display_name, IMAGE_DIR)

                            current_place_images_folder_path = os.path.join(IMAGE_DIR, sanitize_filename(site_address), sanitize_filename(display_name))
                            current_satellite_image_full_path = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_image_name) if satellite_image_name else ""

                            if (place_images_count is not None and place_images_count > 0) or (current_satellite_image_full_path and os.path.exists(current_satellite_image_full_path)):
                                try:
                                    image_result = visionModelResponse(
                                        place_images_folder_path=current_place_images_folder_path,
                                        satellite_image_path=current_satellite_image_full_path
                                    )
                                    image_classification = image_result.get("classification")
                                    image_justification = image_result.get("justification")
                                    if image_classification == "Competitor":
                                        is_competitor = True
                                    time.sleep(1)
                                except Exception as e:
                                    print(f"ERROR calling visionModelResponse for {display_name}: {e}")
                                    print(traceback.format_exc())
                                    image_classification = "Error"
                                    image_justification = str(e)
                    
                    record_data = {
                        "original_name_address": site_address,
                        "original_latitude": original_latitude,
                        "original_longitude": original_longitude,
                        "found_car_wash_name": display_name,
                        "found_in_competitor_list": found_in_competitor_list,
                        "keyword_classification": keyword_classification,
                        "keyword_classification_explanation": keyword_explanation,
                        "place_images_count": place_images_count,
                        "satellite_image_filename": satellite_image_filename,
                        "image_classification": image_classification,
                        "image_classification_justification": image_justification,
                        "is_competitor": is_competitor
                    }
                    pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)

            else:
                print(f"No nearby car washes found for {site_address} or an error occurred.")
                display_name = None
                found_in_competitor_list = None
                keyword_classification = None
                keyword_explanation = None
                place_images_count = None
                satellite_image_filename = None
                image_classification = None
                image_justification = None
                is_competitor = None
                record_data = {
                    "original_name_address": site_address,
                    "original_latitude": original_latitude,
                    "original_longitude": original_longitude,
                    "found_car_wash_name": display_name,
                    "found_in_competitor_list": found_in_competitor_list,
                    "keyword_classification": keyword_classification,
                    "keyword_explanation": keyword_explanation,
                    "place_images_count": place_images_count,
                    "satellite_image_filename": satellite_image_filename,
                    "image_classification": image_classification,
                    "image_classification_justification": image_justification,
                    "is_competitor": is_competitor
                }
                pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)


        print(f"\nProcessing complete. Results appended to {output_filepath}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print(traceback.format_exc())
