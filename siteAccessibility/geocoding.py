import requests
import json # For pretty printing the full response (optional)

def get_address_from_coordinates(latitude, longitude, api_key):
    """
    Fetches the address for given latitude and longitude using Google Geocoding API.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{latitude},{longitude}",
        "key": api_key
        # You can add other optional parameters like 'language', 'result_type', etc.
        # "language": "en",
        # "result_type": "street_address"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        data = response.json()

        if data['status'] == 'OK':
            # The first result is usually the most specific one.
            if data['results']:
                return data['results'][0]['formatted_address']
            else:
                return "No address found for these coordinates."
        elif data['status'] == 'ZERO_RESULTS':
            return "No address found for these coordinates (ZERO_RESULTS)."
        else:
            error_message = data.get('error_message', 'An unknown error occurred.')
            return f"Geocoding API error: {data['status']} - {error_message}"

    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except ValueError: # Includes JSONDecodeError
        return "Failed to decode JSON response."

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your actual API key and coordinates
    YOUR_API_KEY = "AIzaSyDq7-QEyUIezRe5xMf7LxsQBmnNOmRnfho" # <<< IMPORTANT: REPLACE THIS!
    test_latitude = 33.6124699  # Example: Googleplex
    test_longitude = -111.9964852 # Example: Googleplex

    if YOUR_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY":
        print("Please replace 'YOUR_GOOGLE_MAPS_API_KEY' with your actual API key.")
    else:
        address = get_address_from_coordinates(test_latitude, test_longitude, YOUR_API_KEY)
        print(f"Coordinates: {test_latitude}, {test_longitude}")
        print(f"Address: {address}")

        # You might want to see the full JSON response to understand its structure
        # response = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?latlng={test_latitude},{test_longitude}&key={YOUR_API_KEY}")
        # if response.ok:
        #     full_data = response.json()
        #     print("\nFull JSON Response:")
        #     print(json.dumps(full_data, indent=2))
        # else:
        #     print(f"\nFailed to get full response: {response.status_code}")