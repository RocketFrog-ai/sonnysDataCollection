import time
from app.utils import common as calib
from app.features.competitors.utils.competitor_matcher import match_competitors
from app.features.competitors.utils.keyword_classification import keywordclassifier
from app.features.competitors.utils.google_maps_utils import find_nearby_places

API_KEY = calib.GOOGLE_MAPS_API_KEY

def count_competitors(original_latitude, original_longitude):
    API_KEY = calib.GOOGLE_MAPS_API_KEY
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

    # Known (in competitor list) first, then keyword-only; within each group sort by distance.
    known_competitors = []
    keyword_competitors = []
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
                # "Can't say": Google way only (no photos/vision). Treat as not a competitor.

            if is_competitor:
                entry = {"distance": distance, "rating": rating, "userRatingCount": user_rating_count}
                if found_in_competitor_list:
                    known_competitors.append(entry)
                else:
                    keyword_competitors.append(entry)

    known_competitors.sort(key=lambda x: x["distance"])
    keyword_competitors.sort(key=lambda x: x["distance"])
    competitors_data = known_competitors + keyword_competitors

    summary_data = {
        "original_address": f"{original_latitude}, {original_longitude}",
        "competitors_count": len(competitors_data)
    }

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
