import json
import math
import requests
from app.utils import common as calib

API_KEY = calib.GOOGLE_MAPS_API_KEY

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
        if rank_preference == "DISTANCE" and "includedTypes" in payload:
            print("Warning: When rankPreference is 'DISTANCE', 'includedTypes' might not be effective or could be ignored by the API for optimal distance ranking.")

    # print(f"Request Payload (also saved to request_payload.txt):\n{json.dumps(payload, indent=2)}")
    
    # # Save the request payload to a file
    # try:
    #     with open("request_payload.txt", "w", encoding="utf-8") as f:
    #         json.dump(payload, f, indent=2, ensure_ascii=False)
    #     print("Request payload saved to request_payload.txt")
    # except IOError as e:
    #     print(f"Error saving request payload to file: {e}")

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("places", [])
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
    # IMPORTANT: Replace YOUR_API_KEY with your actual key. Do not share your key publicly.
    # API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA" 
    LATITUDE =  42.55114
    LONGITUDE = -71.7638088
    
    place_types_to_search = ['car_wash']
    max_num_results = 20
    ranking_method = "DISTANCE"
    search_radius_miles = 1 # You can change this value

    # --- Validate API Key and Coordinates ---
    if API_KEY == "YOUR_API_KEY" or (LATITUDE == 0.0 and LONGITUDE == 0.0 and LATITUDE == LONGITUDE): # Basic check, ensure not default 0,0
        print("Please replace 'YOUR_API_KEY' with your actual key, and set valid LATITUDE and LONGITUDE.")
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

        # Haversine formula to calculate distance between two lat/lon points
        def haversine_distance(lat1, lon1, lat2, lon2):
            R = 6371  # Radius of Earth in kilometers
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad

            a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            distance_km = R * c
            distance_miles = distance_km * 0.621371 # Convert km to miles
            return distance_miles

        # Save the API response to a file
        if results:
            try:
                with open("api_response.txt", "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print("API response saved to api_response.txt")
            except IOError as e:
                print(f"Error saving API response to file: {e}")
        else:
            print("No results to save.")

        # --- Process and Display Required Results ---
        if results and len(results) > 1: # Ensure there's at least a second result
            total_car_washes = len(results) - 1 # Exclude the first result
            first_nearest_car_wash_actual = results[1] # This is the actual second result, now considered the first nearest

            # Extract rating and user rating count
            rating = first_nearest_car_wash_actual.get("rating", "N/A")
            user_rating_count = first_nearest_car_wash_actual.get("userRatingCount", "N/A")

            # Extract coordinates for distance calculation
            first_nearest_lat = first_nearest_car_wash_actual.get("location", {}).get("latitude")
            first_nearest_lon = first_nearest_car_wash_actual.get("location", {}).get("longitude")

            distance_to_first_nearest = "N/A"
            if first_nearest_lat is not None and first_nearest_lon is not None:
                distance_to_first_nearest = haversine_distance(
                    LATITUDE, LONGITUDE, first_nearest_lat, first_nearest_lon
                )
                distance_to_first_nearest = f"{distance_to_first_nearest:.2f} miles"

            print(f"\n--- Car Wash Analysis ---")
            print(f"Total nearby car washes found (excluding query location): {total_car_washes}")
            print(f"Distance from the first nearest car wash: {distance_to_first_nearest}")
            print(f"Rating of the first nearest car wash: {rating}")
            print(f"User rating count of the first nearest car wash: {user_rating_count}")

        elif results and len(results) == 1:
            print(f"\nFound 1 car wash (which is likely the query location). No other car washes found.")
            print(f"Total nearby car washes found (excluding query location): 0")
        else:
            print("\nNo car washes found matching your criteria.")
