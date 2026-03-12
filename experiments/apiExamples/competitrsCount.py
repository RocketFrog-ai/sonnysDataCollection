import requests
import json
import re
from app.competitor_matcher import match_competitors

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
    radius_meters = 3 * 1609.34
    if not (0.0 < radius_meters <= 50000.0):
        print("Error: Radius must be between 0.0 (exclusive) and 50000.0 meters (inclusive).")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        # Specify the fields you want in the response. This is REQUIRED.
        # Common fields: places.id, places.displayName, places.formattedAddress, places.types,
        #                places.location, places.primaryType, places.websiteUri, places.rating
        # "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.types,places.location,places.primaryType,places.rating,places.userRatingCount"
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
        "maxResultCount": min(max(1, max_results), 20) # Ensure it's between 1 and 20
    }

    # Add includedTypes only if provided and not empty
    if included_types and len(included_types) > 0:
        payload["includedTypes"] = included_types
    
    # Add rankPreference. If 'DISTANCE', usually you wouldn't specify includedTypes.
    # The documentation example for DISTANCE ranking does not use includedTypes.
    if rank_preference:
        payload["rankPreference"] = rank_preference
        # if rank_preference == "DISTANCE" and "includedTypes" in payload:
        #     print("Warning: When rankPreference is 'DISTANCE', 'includedTypes' might not be effective or could be ignored by the API for optimal distance ranking.")


    print(f"Request Payload: {json.dumps(payload, indent=2)}") # For debugging

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
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
    # --- Configuration ---
    API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"  # <--- REPLACE WITH YOUR ACTUAL API KEY
    # LATITUDE = 28.1879225 # <--- REPLACE WITH YOUR LATITUDE (e.g., 37.7937)
    # LONGITUDE = -80.6727404 # <--- REPLACE WITH YOUR LONGITUDE (e.g., -122.3965)
    LATITUDE = 33.8621251
    LONGITUDE = -84.6719291
    
    # Optional: Specify types of places you're interested in.
    # If you want all types, you can set this to None or an empty list.
    # Example: place_types_to_search = ["restaurant", "cafe"]
    place_types_to_search = ['car_wash']# None  Search for all types by default when None

    # Optional: Maximum number of results
    max_num_results = 10

    # Optional: Rank preference ("POPULARITY" or "DISTANCE")
    # If using "DISTANCE", it's often best to not specify place_types_to_search
    # ranking_method = "POPULARITY"
    ranking_method = "DISTANCE" # Uncomment to rank by distance

    # --- Validate API Key and Coordinates ---
    if API_KEY == "YOUR_API_KEY" or LATITUDE == 0.0 or LONGITUDE == 0.0:
        print("Please replace 'YOUR_API_KEY', 'YOUR_LATITUDE', and 'YOUR_LONGITUDE' with actual values in the script.")
    else:
        # --- Perform the Search ---
        print(f"Searching for places near Latitude: {LATITUDE}, Longitude: {LONGITUDE}.")
        
        results = find_nearby_places(
            API_KEY,
            LATITUDE,
            LONGITUDE,
            radius_miles=1,
            included_types=place_types_to_search,
            max_results=max_num_results,
            rank_preference=ranking_method
        )

        # --- Display Results ---
        competitor_names = []
        if results and "places" in results:
            print(f"\nFound {len(results['places'])} places:")
            for i, place in enumerate(results["places"]):
                print(f"\n--- Result {i+1} ---")
                display_name = place.get("displayName", {}).get("text", "N/A")
                competitor_names.append(display_name)
                address = place.get("formattedAddress", "N/A")
                place_id = place.get("id", "N/A")
                types = place.get("types", [])
                primary_type = place.get("primaryType", "N/A")
                location = place.get("location", {})
                rating = place.get("rating", "N/A")
                user_rating_count = place.get("userRatingCount", "N/A")

                print(f"  Name: {display_name}")
                print(f"  Address: {address}")
                print(f"  Place ID: {place_id}")
                print(f"  Primary Type: {primary_type}")
                print(f"  All Types: {', '.join(types)}")
                print(f"  Location: Lat={location.get('latitude', 'N/A')}, Lng={location.get('longitude', 'N/A')}")
                print(f"  Rating: {rating} (from {user_rating_count} reviews)")

            # Use the matching function from competitor_matcher.py
            # csv_filepath = 'Car_Wash_Advisory_Companies.csv'
            
            found_count, found_competitors, not_found_competitors = match_competitors(competitor_names)

            # Display results
            print("\n--- Matching Results ---")
            print(f"Total competitors found in CSV: {found_count}")

            if found_competitors:
                print("\nCompetitors found in CSV:")
                for competitor in found_competitors:
                    print(competitor)

            if not_found_competitors:
                print("\nCompetitors not found in CSV:")
                for competitor in not_found_competitors:
                    print(competitor)
            else:
                print("\nAll competitors were found in the CSV.")

        elif results:
            print("\nNo places found or an unexpected response format.")
            print(f"API Response: {json.dumps(results, indent=2)}")
        else:
            print("\nSearch failed.")
