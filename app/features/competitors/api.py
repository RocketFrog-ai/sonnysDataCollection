import pandas as pd
import os
from dotenv import load_dotenv
import time
import traceback
from utils.competitor_matcher import match_competitors
from utils.placePhotos import get_photo_references_and_name, download_photo
from utils.keyword_classification import keywordclassifier
from utils.gpt_images_classification import visionModelResponse
from utils.file_utils import sanitize_filename, get_place_image_count
from utils.geo_utils import calculate_distance
from utils.google_maps_utils import get_satellite_image_name, download_satellite_image, find_nearby_places

IMAGE_DIR = "competitors/place_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

SATELLITE_IMAGE_BASE_DIR = "competitors/satellite_images"
if not os.path.exists(SATELLITE_IMAGE_BASE_DIR):
    os.makedirs(SATELLITE_IMAGE_BASE_DIR)

def count_competitors(original_latitude, original_longitude):
    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not API_KEY or API_KEY == "YOUR_API_KEY":
        return {"error": "Please replace 'YOUR_API_KEY' with your actual value in the script."}

    place_types_to_search = ['car_wash']
    max_num_results = 20
    ranking_method = "DISTANCE"

    results = find_nearby_places(
        API_KEY,
        latitude=original_latitude,
        longitude=original_longitude,
        radius_miles=1,
        included_types=place_types_to_search,
        max_results=max_num_results,
        rank_preference=ranking_method
    )

    competitors_data = []
    if results and "places" in results:
        for i, place in enumerate(results["places"]):
            if i == 0:
                continue
                
            display_name = place.get("displayName", {}).get("text", "N/A")
            place_latitude = place.get("location", {}).get("latitude")
            place_longitude = place.get("location", {}).get("longitude")
            place_id = place.get("id")
            is_competitor = False
            distance = calculate_distance(original_latitude, original_longitude, place_latitude, place_longitude)
            rating = place.get("rating")
            user_rating_count = place.get("userRatingCount")
            
            _, found_competitors, _ = match_competitors([display_name])
            found_in_competitor_list = bool(found_competitors)
            
            if found_in_competitor_list:
                is_competitor = True
            else:
                classification_result = keywordclassifier(display_name)
                keyword_classification = classification_result.get("classification")
                time.sleep(1)

                if keyword_classification == "Competitor":
                    is_competitor = True
                elif keyword_classification == "Can't say":
                    satellite_image_name = get_satellite_image_name(place_id, SATELLITE_IMAGE_BASE_DIR)
                    if satellite_image_name is None and place_id and place_latitude and place_longitude:
                        if download_satellite_image(API_KEY, place_latitude, place_longitude, place_id, SATELLITE_IMAGE_BASE_DIR):
                            satellite_image_name = f"{place_id}.jpg"
                    
                    num_place_images = 0
                    if place_id:
                        photo_references, _ = get_photo_references_and_name(place_id)
                        if photo_references:
                            for photo_idx, ref in enumerate(photo_references):
                                download_photo(ref, "api_call", display_name, photo_idx, IMAGE_DIR)
                            num_place_images = get_place_image_count("api_call", display_name, IMAGE_DIR)

                    current_place_images_folder_path = os.path.join(IMAGE_DIR, sanitize_filename("api_call"), sanitize_filename(display_name))
                    current_satellite_image_full_path = os.path.join(SATELLITE_IMAGE_BASE_DIR, satellite_image_name) if satellite_image_name else ""

                    if (num_place_images is not None and num_place_images > 0) or (current_satellite_image_full_path and os.path.exists(current_satellite_image_full_path)):
                        try:
                            image_classification_result = visionModelResponse(
                                place_images_folder_path=current_place_images_folder_path,
                                satellite_image_path=current_satellite_image_full_path
                            )
                            image_classification = image_classification_result.get("classification")
                            if image_classification == "Competitor":
                                is_competitor = True
                            time.sleep(1)
                        except Exception as e:
                            print(f"ERROR calling visionModelResponse for {display_name}: {e}")
            
            if is_competitor:
                competitors_data.append({
                    "distance": distance,
                    "rating": rating,
                    "userRatingCount": user_rating_count
                })

    summary_data = {
        "original_address": f"{original_latitude}, {original_longitude}",
        "competitors_count": len(competitors_data)
    }
    
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
            
    return summary_data
