import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def find_nearby_places(api_key, latitude, longitude, radius_miles=2, included_types=None, max_results=10, rank_preference="POPULARITY"):
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
                                         Defaults to "POPULARITY".

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    base_url = "https://places.googleapis.com/v1/places:searchNearby"

    # Convert radius from miles to meters (1 mile = 1609.34 meters)
    # CORRECTED: Use the radius_miles argument passed to the function
    radius_meters = radius_miles * 1609.34
    if not (0.0 < radius_meters <= 50000.0):
        print("Error: Radius must be between 0.0 (exclusive) and 50000.0 meters (inclusive).")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "*" # Requesting all fields
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
        # if rank_preference == "DISTANCE" and "includedTypes" in payload:
        #     print("Warning: When rankPreference is 'DISTANCE', 'includedTypes' might not be effective or could be ignored by the API for optimal distance ranking.")

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()
        return response_data
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
    # --- Configuration ---
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    LATITUDE =  32.6003451
    LONGITUDE = -93.841208
    
    place_types_to_search = ['car_wash']
    max_num_results = 1
    ranking_method = "DISTANCE"
    search_radius_miles = 2 # You can change this value

    # --- Validate API Key and Coordinates ---
    if not API_KEY:
        print("Error: GOOGLE_MAPS_API_KEY not found in .env file.")
    elif (LATITUDE == 0.0 and LONGITUDE == 0.0 and LATITUDE == LONGITUDE): # Basic check, ensure not default 0,0
        print("Please set valid LATITUDE and LONGITUDE.")
    else:
        # --- Perform the Search ---
        search_type_message = f"of type '{', '.join(place_types_to_search)}'" if place_types_to_search else "of all types"
        print(f"Searching for places {search_type_message} near Latitude: {LATITUDE}, Longitude: {LONGITUDE} within a {search_radius_miles}-mile radius.")
        
        results = find_nearby_places(
            API_KEY,
            LATITUDE,
            LONGITUDE,
            radius_miles=search_radius_miles, # Pass the configurable radius
            included_types=place_types_to_search,
            max_results=max_num_results,
            rank_preference=ranking_method
        )

        # --- Display Results ---
        if results and "places" in results and results["places"]:
            nearest_place = results["places"][0]  # Get the first result, which is the nearest
            
            display_name = nearest_place.get("displayName", {}).get("text", "N/A")
            location = nearest_place.get("location", {})
            latitude = location.get("latitude", "N/A")
            longitude = location.get("longitude", "N/A")
            rating = nearest_place.get("rating", "N/A")
            rating_count = nearest_place.get("userRatingCount", "N/A")
            opening_hours = nearest_place.get("regularOpeningHours", {})
            weekday_desc = opening_hours.get("weekdayDescriptions", "N/A")
            business_status = nearest_place.get("businessStatus", "N/A")

            print("\n--- Nearest Car Wash Details ---")
            print(f"  Display Name: {display_name}")
            print(f"  Latitude: {latitude}")
            print(f"  Longitude: {longitude}")
            print(f"  Rating: {rating}")
            print(f"  Rating Count: {rating_count}")
            print(f"  Weekday Description: {weekday_desc}")
            print(f"  Business Status: {business_status}")
            
        else:
            print("\nNo places found or search failed.")
