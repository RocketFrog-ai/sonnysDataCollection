# Python script to call Google Roads API - Speed Limits

import requests
import json # For pretty printing the raw response

# IMPORTANT: Replace with your actual Google Maps API Key
# API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"
# API_KEY = "AIzaSyDq7-QEyUIezRe5xMf7LxsQBmnNOmRnfho"
# API_KEY = 'AIzaSyCZphNk0uM3e1sbOJG2wBTFjPR_E-Dm-Vw'
API_KEY = 'AIzaSyBt8VC_pohJU1cDObr9S8uMaBmLR8ORa5c'
ROADS_API_SPEED_LIMITS_URL = "https://roads.googleapis.com/v1/speedLimits"

def get_speed_limits_for_path(lat_lon_pairs, units="KPH"):
    """
    Fetches speed limits for a given path of latitude/longitude pairs.

    Args:
        lat_lon_pairs: A list of tuples, where each tuple is (latitude, longitude).
                       Example: [(60.170880, 24.942795), (60.170879, 24.942796)]
                       The API accepts a maximum of 100 pairs.
        units: "KPH" for kilometers per hour or "MPH" for miles per hour.
               Defaults to "KPH" as per API documentation.

    Returns:
        A dictionary containing the API response or None if an error occurs.
    """

    if not API_KEY or API_KEY == "YOUR_API_KEY":
        print("Error: Please replace 'YOUR_API_KEY' in the script with your actual Google Maps API key.")
        return None

    if not lat_lon_pairs:
        print("Error: The list of latitude/longitude pairs cannot be empty.")
        return None

    if len(lat_lon_pairs) > 100:
        print(f"Error: The API accepts a maximum of 100 latitude/longitude pairs. You provided {len(lat_lon_pairs)}.")
        return None

    # Format the path parameter: lat1,lon1|lat2,lon2|...
    path_param = "|".join([f"{lat},{lon}" for lat, lon in lat_lon_pairs])

    params = {
        "path": path_param,
        "units": units.upper(), # Ensure units are uppercase (KPH or MPH)
        "key": API_KEY
    }

    print(f"Requesting URL: {ROADS_API_SPEED_LIMITS_URL}")
    print(f"With parameters (key redacted): path={path_param}, units={units.upper()}")

    try:
        response = requests.get(ROADS_API_SPEED_LIMITS_URL, params=params)
        # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}") # Print text for more details on API errors
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except ValueError as json_err: # Includes JSONDecodeError if response is not valid JSON
        print(f"JSON decoding error: {json_err}")
        print(f"Response content: {response.text}") # Print text if JSON decoding fails
    return None

def main():
    # print("Google Roads API - Speed Limit Demo")
    # print("------------------------------------")
    # print("Notice: The Speed Limit service is primarily available to customers with an")
    # print("Asset Tracking license from Google Maps Platform.")
    # print("The accuracy of speed limit data returned by the Roads API")
    # print("cannot be guaranteed. The data may be estimated, inaccurate, incomplete, or outdated.\n")

    # --- Define your input latitude and longitude pairs here ---
    # Example coordinates (from Vasco da Gama bridge example in API documentation)
    # path=38.75807927603043,-9.03741754643809|38.6896537,-9.1770515|41.1399289,-8.6094075
    input_coordinates = [
        (40.2614149, -80.1747502),
        (26.8819403, -80.1164588),
        (33.72409, -84.171848)
    ]

    # You can change units to "MPH" if needed
    # desired_units = "MPH"
    desired_units = "KPH"

    print(f"Requesting speed limits for {len(input_coordinates)} points in {desired_units}.")

    api_response = get_speed_limits_for_path(input_coordinates, units=desired_units)

    if api_response:
        print("\n--- Raw API Response ---")
        print(json.dumps(api_response, indent=2))
        print("\n--- Processed Speed Limit Data ---")

        # Create a dictionary for easy lookup of speed limits by placeId
        # This is useful if the speedLimits array isn't ordered or if multiple
        # snapped points share the same placeId.
        speed_limits_map = {sl['placeId']: sl for sl in api_response.get('speedLimits', [])}
        snapped_points = api_response.get('snappedPoints', [])

        if not snapped_points and not api_response.get('speedLimits'):
            if 'warningMessage' in api_response:
                 print(f"Warning from API: {api_response['warningMessage']}")
            # Check for Google API standard error format
            elif 'error' in api_response and isinstance(api_response['error'], dict):
                 error_details = api_response['error']
                 print(f"Error from API: {error_details.get('message', 'Unknown error')}")
                 print(f"Status: {error_details.get('status', 'N/A')}")
            else:
                print("No snapped points or speed limits returned. The path might not be on roads, data might be unavailable, or there might have been an issue with the request (e.g. API key).")
            return

        if not snapped_points:
            print("No points were snapped to roads by the API.")
            # If by any chance speedLimits are returned without snappedPoints (unlikely for path requests)
            if api_response.get('speedLimits'):
                print("\nSpeed Limits Found (but not associated with specific input points due to missing snappedPoints):")
                for i, sl_info in enumerate(api_response.get('speedLimits', [])):
                    print(f"  Segment {i+1}:")
                    print(f"    Place ID: {sl_info.get('placeId')}")
                    print(f"    Speed Limit: {sl_info.get('speedLimit')} {sl_info.get('units')}")
            return

        print(f"\nDetails for the {len(input_coordinates)} input points:")
        # Iterate based on the original number of input points
        for original_idx in range(len(input_coordinates)):
            original_lat, original_lon = input_coordinates[original_idx]
            print(f"\nInput Point {original_idx} (Original: Lat={original_lat}, Lon={original_lon}):")

            # Find the snapped point that corresponds to the current original_idx
            current_snapped_point = None
            for sp in snapped_points:
                if sp.get('originalIndex') == original_idx:
                    current_snapped_point = sp
                    break
            
            if current_snapped_point:
                snapped_loc = current_snapped_point.get('location', {})
                snapped_lat = snapped_loc.get('latitude', 'N/A')
                snapped_lon = snapped_loc.get('longitude', 'N/A')
                place_id = current_snapped_point.get('placeId')

                print(f"  Snapped To: Lat={snapped_lat}, Lon={snapped_lon}")
                print(f"  Road Segment Place ID: {place_id}")

                if place_id and place_id in speed_limits_map:
                    sl_info = speed_limits_map[place_id]
                    print(f"  Speed Limit: {sl_info.get('speedLimit')} {sl_info.get('units')}")
                elif place_id:
                    print(f"  Speed Limit: Data not found in 'speedLimits' array for Place ID {place_id}.")
                else:
                    print("  Speed Limit: No Place ID returned for this snapped point.")
            else:
                print("  This point was not found in the API's 'snappedPoints' array (it might not have snapped to a road, or an issue occurred).")
        
        if 'warningMessage' in api_response:
            print(f"\nWarning from API: {api_response['warningMessage']}")
    else:
        print("\nFailed to retrieve data from the API.")

if __name__ == "__main__":
    main()


# The Core Problem:

# Your Google Cloud Project, associated with the API key you are using, does not have the necessary "Asset Tracking license" (or equivalent entitlement) enabled to access the speed limit data.

# How to Potentially Resolve This:

#     Asset Tracking License:

#         The speed limit service within the Roads API is a premium feature. You generally need to be enrolled in Google's Asset Tracking solution or have a specific agreement with Google that grants access to this feature.

#         Action: You will likely need to contact Google Maps Platform Sales or your Google Cloud account representative to inquire about obtaining the necessary license or entitlement for your project.

#     Ensure Roads API is Enabled:

#         While the error points to a licensing issue for speed limits, double-check that the "Roads API" itself is enabled for your project in the Google Cloud Console.

#         Go to Google Cloud Console -> APIs & Services -> Library.

#         Search for "Roads API" and ensure it's enabled for the project tied to your API key.

#         If it's not, enable it. However, this alone won't solve the "Speed limits are not available for this project" error if the licensing is the root cause.

#     Billing Account:

#         Ensure your project is linked to an active and valid billing account. Most Google Maps Platform APIs require this.

#     API Key Restrictions:

#         If you have restricted your API key (e.g., to specific APIs or IP addresses), ensure that the Roads API is allowed and that your request isn't being blocked by other restrictions. However, the error PERMISSION_DENIED for the service suggests this is less likely the primary issue than the licensing.

# In summary, the most probable reason is the lack of an Asset Tracking license.

# What to do now:

#     Contact Google: This is your primary path. Explain that you want to use the speed limit feature of the Roads API and inquire about the necessary licensing or contract.

#     Consider Alternatives (if licensing is a blocker):

#         OpenStreetMap (OSM): OSM data often contains maxspeed tags. You would need to query OSM data (e.g., via the Overpass API or by downloading OSM extracts and using a tool like Osmosis or a GIS) and then process it yourself to find speed limits for your coordinates. This is more complex but free.

#         Other commercial providers: There are other companies that provide road data, including speed limits, but they will also have their own licensing and costs.

# The "key redacted" part is good practice, but the issue here isn't the key itself being invalid, but rather what that key (and its associated project) is authorized to do.