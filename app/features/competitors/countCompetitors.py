import os
import sys
import time
import traceback
import pandas as pd
from app.utils import common as calib
from app.features.competitors.utils.competitor_matcher import match_competitors
from app.features.competitors.utils.keyword_classification import keywordclassifier
from app.features.competitors.utils.gpt_images_classification import visionModelResponse
from app.features.competitors.utils.file_utils import sanitize_filename, get_place_image_count
from app.features.competitors.utils.placePhotos import get_photo_references_and_name, download_photo
from app.features.competitors.utils.google_maps_utils import get_satellite_image_name, download_satellite_image, find_nearby_places


RFW_HOME = calib.RFW_HOME
API_KEY = calib.GOOGLE_MAPS_API_KEY


# Directory to save downloaded images (relative to competitors/)
IMAGE_DIR = os.path.join(RFW_HOME, "place_images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Directory for satellite images (relative to competitors/)
SATELLITE_IMAGE_BASE_DIR = os.path.join(RFW_HOME, "satellite_images")
if not os.path.exists(SATELLITE_IMAGE_BASE_DIR):
    os.makedirs(SATELLITE_IMAGE_BASE_DIR)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python countCompetitors.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start_index_to_process = int(sys.argv[1])
        end_index_to_process = int(sys.argv[2])
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)

    # API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
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
            "original_name_address", "original_latitude", "original_longitude",
            "found_car_wash_name", "distance_miles", "google_rating", "google_user_rating_count",
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

        summary_csv_headers = ["original_address", "competitors_count"]
        for i in range(1, 7):
            summary_csv_headers.extend([
                f"competitor_{i}_distance_miles",
                f"competitor_{i}_google_rating",
                f"competitor_{i}_google_user_rating_count"
            ])
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

            competitors_data = []

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
                    distance = calib.calculate_distance(original_latitude, original_longitude, place_latitude, place_longitude)
                    rating = place.get("rating")
                    user_rating_count = place.get("userRatingCount")
                    
                    # Initialize filter results
                    keyword_classification = None
                    keyword_explanation = None
                    num_place_images = None
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
                            satellite_image_name = get_satellite_image_name(place_id, SATELLITE_IMAGE_BASE_DIR)
                            if satellite_image_name is None and place_id and place_latitude and place_longitude:
                                if download_satellite_image(API_KEY, place_latitude, place_longitude, place_id, SATELLITE_IMAGE_BASE_DIR):
                                    satellite_image_name = f"{place_id}.jpg"
                            
                            if place_id:
                                photo_references, _ = get_photo_references_and_name(place_id)
                                if photo_references:
                                    for photo_idx, ref in enumerate(photo_references):
                                        download_photo(ref, site_address, display_name, photo_idx, IMAGE_DIR)
                                    num_place_images = get_place_image_count(site_address, display_name, IMAGE_DIR)

                            current_place_images_folder_path = os.path.join(IMAGE_DIR, sanitize_filename(site_address), sanitize_filename(display_name))
                            current_satellite_image_full_path = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_image_name) if satellite_image_name else ""

                            if (num_place_images is not None and num_place_images > 0) or (current_satellite_image_full_path and os.path.exists(current_satellite_image_full_path)):
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
                        competitors_data.append({
                            "distance": distance,
                            "rating": rating,
                            "userRatingCount": user_rating_count
                        })
                    
                    record_data = {
                        "original_name_address": site_address,
                        "original_latitude": original_latitude,
                        "original_longitude": original_longitude,
                        "found_car_wash_name": display_name,
                        "distance_miles": distance,
                        "google_rating": rating,
                        "google_user_rating_count": user_rating_count,
                        "found_in_competitor_list": found_in_competitor_list,
                        "keyword_classification": keyword_classification,
                        "keyword_classification_explanation": keyword_explanation,
                        "place_images_count": num_place_images,
                        "satellite_image_filename": satellite_image_name,
                        "image_classification": image_classification,
                        "image_classification_justification": image_justification,
                        "is_competitor": is_competitor
                    }
                    pd.DataFrame([record_data], columns=csv_headers).to_csv(output_filepath, index=False, mode='a', header=False)

            else:
                print(f"No nearby car washes found for {site_address} or an error occurred.")
            
            summary_data = {
                "original_address": site_address,
                "competitors_count": len(competitors_data)
            }
            
            # Sort competitors by distance
            competitors_data.sort(key=lambda x: x['distance'])

            for i in range(6):
                if i < len(competitors_data):
                    summary_data[f"competitor_{i+1}_distance_miles"] = competitors_data[i]["distance"]
                    summary_data[f"competitor_{i+1}_google_rating"] = competitors_data[i]["rating"]
                    summary_data[f"competitor_{i+1}_google_user_rating_count"] = competitors_data[i]["userRatingCount"]
                else:
                    summary_data[f"competitor_{i+1}_distance_miles"] = None
                    summary_data[f"competitor_{i+1}_google_rating"] = None
                    summary_data[f"competitor_{i+1}_google_user_rating_count"] = None
            pd.DataFrame([summary_data], columns=summary_csv_headers).to_csv(summary_output_filepath, index=False, mode='a', header=False)

        print(f"\nProcessing complete. Results appended to {output_filepath}")
        print(f"Summary results appended to {summary_output_filepath}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        print(traceback.format_exc())
