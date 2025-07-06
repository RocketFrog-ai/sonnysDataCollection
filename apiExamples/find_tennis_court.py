import requests
import json
import os

def find_nearest_place(api_key, latitude, longitude, place_type="athletic_field", rank_preference="DISTANCE"):
    """
    Finds the nearest place of a specific type using the Google Places API.

    Args:
        api_key (str): Your Google Maps Platform API Key.
        latitude (float): The latitude of the center point for the search.
        longitude (float): The longitude of the center point for the search.
        place_type (str): The type of place to search for.
        rank_preference (str): How to rank results. "DISTANCE" is recommended for finding the nearest place.

    Returns:
        dict: The first place result from the API, or None if no places are found or an error occurs.
    """
    base_url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.location,places.displayName"
    }
    payload = {
        "includedTypes": [place_type],
        "maxResultCount": 1,
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": 50000.0  # 50km radius, as distance ranking is used.
            }
        },
        "rankPreference": rank_preference
    }

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        results = response.json()
        if "places" in results and results["places"]:
            return results["places"][0]
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.text}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        print(f"Response content: {response.text}")
    return None

def download_satellite_image(api_key, latitude, longitude, zoom=20, size="640x640", scale=1, maptype="hybrid"):
    """
    Downloads a satellite image from the Google Static Maps API.

    Args:
        api_key (str): Your Google Maps Platform API Key.
        latitude (float): The latitude for the center of the map.
        longitude (float): The longitude for the center of the map.
        zoom (int): The zoom level of the map.
        size (str): The dimensions of the image.
        maptype (str): The type of map to return.

    Returns:
        str: The filename of the saved image, or None if the download fails.
    """
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": size,
        "scale": scale,
        "maptype": maptype,
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Create a directory for the images if it doesn't exist
        if not os.path.exists("tennis_court_images"):
            os.makedirs("tennis_court_images")
            
        filename = f"tennis_court_images/tennis_court_{latitude}_{longitude}.png"
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Satellite image saved as '{filename}'")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve image. Status code: {response.status_code if 'response' in locals() else 'N/A'}. Error: {e}")
    return None

if __name__ == "__main__":
    # IMPORTANT: Replace with your actual Google Maps API Key.
    API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA"
    
    # --- Configuration: Set the coordinates to search from ---
    SEARCH_LATITUDE = 47.90284
    SEARCH_LONGITUDE = -97.07409
# 41.93622, -87.68012
    print(f"Searching for the nearest tennis court to Latitude: {SEARCH_LATITUDE}, Longitude: {SEARCH_LONGITUDE}")
    
    nearest_court = find_nearest_place(API_KEY, SEARCH_LATITUDE, SEARCH_LONGITUDE)

    if nearest_court:
        court_name = nearest_court.get("displayName", {}).get("text", "Unknown")
        location = nearest_court.get("location", {})
        court_lat = location.get("latitude")
        court_lon = location.get("longitude")

        if court_lat is not None and court_lon is not None:
            print(f"Found nearest tennis court: '{court_name}' at Latitude: {court_lat}, Longitude: {court_lon}")
            print("Downloading satellite image...")
            download_satellite_image(API_KEY, court_lat, court_lon)
        else:
            print("Could not determine the location of the nearest tennis court.")
    else:
        print("No tennis courts found nearby.")
