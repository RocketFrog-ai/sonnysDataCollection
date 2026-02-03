import requests
import json

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

    print(f"Request Payload (also saved to request_payload.txt):\n{json.dumps(payload, indent=2)}")
    
    # Save the request payload to a file
    try:
        with open("request_payload.txt", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("Request payload saved to request_payload.txt")
    except IOError as e:
        print(f"Error saving request payload to file: {e}")

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
    # IMPORTANT: Replace YOUR_API_KEY with your actual key. Do not share your key publicly.
    API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA" 
    LATITUDE =  32.6003451
    LONGITUDE = -93.841208
    
    place_types_to_search = ['car_wash']
    max_num_results = 10
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

        # --- Display Results (Optional - can be commented out if files are primary focus) ---
        # The terminal display is now optional as data is saved to files.
        # You can keep it if you want a quick look, or comment it out.
        if results and "places" in results:
            if results['places']: 
                print(f"\nFound {len(results['places'])} places. (Full data in api_response.txt)")
                # If you still want to print a summary to terminal:
                for i, place_data in enumerate(results["places"]):
                    print(f"\n--- Result {i+1} (Summary) ---")
                    display_name = place_data.get("displayName", {}).get("text", "N/A")
                    address = place_data.get("formattedAddress", "N/A")
                    print(f"  Name: {display_name}")
                    print(f"  Address: {address}")
                    # Add more summary fields if needed
            else:
                print("\nNo places found matching your criteria. (Full response in api_response.txt if API call was made)")
        elif results:
            print("\nSearch returned data, but it does not contain a 'places' list or it's an unexpected format. (Full data in api_response.txt)")
        else:
            print("\nSearch failed or no results returned. (Check logs and api_response.txt if it exists)")