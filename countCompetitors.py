import requests
import json
import re
import pandas as pd
import os
from competitor_matcher import match_competitors
from apiExamples.placePhotos import get_photo_references_and_name, download_photo

# Directory to save downloaded images
IMAGE_DIR = "place_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

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
    API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"  # <--- REPLACE WITH YOUR ACTUAL API KEY
    EXCEL_PATH = '/home/arpit/dataCollection/datasets/1mile_raw_data.xlsx'
    OUTPUT_DIR = '/home/arpit/dataCollection/output_csv'
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

    # Prepare a list to store all results
    all_results = []

    try:
        df = pd.read_excel(EXCEL_PATH, engine='openpyxl')

        # Ensure indices are within the DataFrame bounds
        if start_index_to_process < 0 or end_index_to_process > len(df) or start_index_to_process >= end_index_to_process:
            print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) -1}.")
            sys.exit(1)

        for index, row in df.iloc[start_index_to_process:end_index_to_process].iterrows():
            site_address = row.iloc[0]
            latitude = row.iloc[1]
            longitude = row.iloc[2]

            print(f"Processing record {index}: {site_address}, Latitude: {latitude}, Longitude: {longitude}")

            results = find_nearby_places(
                API_KEY,
                latitude=latitude,
                longitude=longitude,
                radius_miles=1,
                included_types=place_types_to_search,
                max_results=max_num_results,
                rank_preference=ranking_method
            )

            if results and "places" in results:
                for place in results["places"]:
                    display_name = place.get("displayName", {}).get("text", "N/A")
                    
                    # Determine if it's a competitor using competitor_matcher.py
                    _, found_competitors, _ = match_competitors([display_name])
                    is_competitor = bool(found_competitors) # True if found, False otherwise

                    place_id = place.get("id") # Get the place ID
                    
                    all_results.append({
                        "Original_Name_Address": site_address,
                        "Original_Latitude": latitude,
                        "Original_Longitude": longitude,
                        "Found_Car_Wash_Name": display_name,
                        "Is_Competitor": is_competitor
                    })

                    # Download photos if place_id is available and it's a competitor
                    if place_id and is_competitor:
                        photo_references, place_name_for_photo = get_photo_references_and_name(place_id)
                        if photo_references:
                            print(f"Downloading photos for '{display_name}'...")
                            for i, ref in enumerate(photo_references):
                                download_photo(ref, site_address, display_name, i)
                        else:
                            print(f"No photos found for '{display_name}' (ID: {place_id}).")
            else:
                print(f"No nearby car washes found for {site_address} or an error occurred.")
                # Optionally, add a record for sites with no found car washes
                all_results.append({
                    "Original_Name_Address": site_address,
                    "Original_Latitude": latitude,
                    "Original_Longitude": longitude,
                    "Found_Car_Wash_Name": "N/A",
                    "Is_Competitor": False
                })

        # Write all collected results to a CSV
        if all_results:
            output_df = pd.DataFrame(all_results)
            output_df.to_csv(output_filepath, index=False)
            print(f"\nSuccessfully wrote competitor analysis to {output_filepath}")
        else:
            print("\nNo data to write to CSV.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
