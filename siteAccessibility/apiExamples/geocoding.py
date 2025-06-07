import requests
import os

# Using the API key found in mapsStatic.py
# Ensure this key is also enabled for the Google Geocoding API in your Google Cloud Console.
GOOGLE_MAPS_API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

def get_address_from_coords(latitude: float, longitude: float) -> str:
    """
    Performs reverse geocoding to get an address from latitude and longitude
    using the Google Maps Geocoding API.

    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.

    Returns:
        A formatted address string if successful, or an error message string.
    """
    if GOOGLE_MAPS_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
        return "Error: Google Maps API key not configured."

    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{latitude},{longitude}",
        "key": GOOGLE_MAPS_API_KEY
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            # The first result is usually the most accurate
            return data["results"][0].get("formatted_address", "Error: Formatted address not found in response.")
        elif data.get("status") == "ZERO_RESULTS":
            return "Error: No address found for the given coordinates."
        else:
            error_message = data.get("error_message", "Unknown error from Geocoding API.")
            status = data.get("status", "N/A")
            return f"Error: Geocoding API request failed. Status: {status}. Message: {error_message}"

    except requests.exceptions.RequestException as e:
        return f"Error: Request to Geocoding API failed: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred during geocoding: {e}"

if __name__ == "__main__":
    # Example usage (replace with actual coordinates for testing)
    test_lat = 34.052235
    test_lon = -118.243683
    
    # Make sure to set your GOOGLE_MAPS_API_KEY environment variable or replace the placeholder in the script
    # For example, in your terminal before running: export GOOGLE_MAPS_API_KEY="YOUR_ACTUAL_KEY"
    
    print(f"Attempting to get address for Lat: {test_lat}, Lon: {test_lon}")
    address = get_address_from_coords(test_lat, test_lon)
    print(f"Formatted Address: {address}")

    test_lat_no_result = 0.0
    test_lon_no_result = 0.0
    print(f"\nAttempting to get address for Lat: {test_lat_no_result}, Lon: {test_lon_no_result} (expecting no result)")
    address_no_result = get_address_from_coords(test_lat_no_result, test_lon_no_result)
    print(f"Formatted Address: {address_no_result}")
