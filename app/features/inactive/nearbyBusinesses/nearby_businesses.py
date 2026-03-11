import os
import json
import requests
from app.utils import common as calib

API_KEY = calib.GOOGLE_MAPS_API_KEY


def find_nearby_places(api_key, latitude, longitude, radius_miles=2, included_types=None, max_results=10):
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

    payload["rankPreference"] = "DISTANCE"
    
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

def get_nearby_business_count(latitude: float, longitude: float):
    """
    Finds a nearby car wash and then counts the number of other businesses
    in its immediate vicinity.

    Args:
        latitude: The starting latitude.
        longitude: The starting longitude.

    Returns:
        A dictionary with the results, or None if an error occurs.
    """
    # API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    # API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not API_KEY:
        print("ERROR: GOOGLE_MAPS_API_KEY environment variable not set.")
        return None

    # --- Step 1: Find the nearest car wash ---
    
    car_wash_radius_miles = 500/1609
    carwash_data = find_nearby_places(
        API_KEY,
        latitude,
        longitude,
        radius_miles=car_wash_radius_miles,
        included_types=['car_wash'],
        max_results=1
    )

    if not carwash_data or "places" not in carwash_data or not carwash_data["places"]:
        print(f"No car wash found within 500m")
        return None

    # Assume the first result is the one we want
    target_car_wash = carwash_data['places'][0]
    car_wash_name = target_car_wash.get('displayName', {}).get('text', 'N/A')
    car_wash_location = target_car_wash.get('location', {})
    car_wash_latitude = car_wash_location.get('latitude', 'N/A')
    car_wash_longitude = car_wash_location.get('longitude', 'N/A')
    
    # print(f"\nFound car wash: '{car_wash_name}'")

    # print(f"Now searching for all businesses within 50m of '{car_wash_name}'...")

    nearby_radius_miles = 500/1609
    nearby_data = find_nearby_places(
        API_KEY,
        car_wash_latitude,
        car_wash_longitude,
        radius_miles=nearby_radius_miles,
        max_results=10 # Increased max results
    )
        
    if not nearby_data or "places" not in nearby_data:
        print(f"Could not find nearby places for the car wash.")
        return None

    # --- Step 3: Find the 5 nearest businesses (excluding the car wash itself) ---
    nearest_businesses = []
    for place in nearby_data['places']:
        place_name = place.get('displayName', {}).get('text', 'N/A')
        if place_name != car_wash_name:
            nearest_businesses.append(place)
        if len(nearest_businesses) >= 5:
            break

    car_wash_address = target_car_wash.get('formattedAddress', 'N/A')

    # Prepare the final results
    final_result = {
        'target_car_wash': {
            'name': car_wash_name,
            'address': car_wash_address,
        },
        'nearest_businesses': []
    }

    for i, business in enumerate(nearest_businesses):
        business_name = business.get('displayName', {}).get('text', 'N/A')
        business_address = business.get('formattedAddress', 'N/A')
        final_result['nearest_businesses'].append({
            'name': business_name,
            'address': business_address,
        })

    if final_result:
        distance = calib.calculate_distance(latitude, longitude, float(car_wash_latitude), float(car_wash_longitude))
        final_result['distance'] = distance
        
        # Calculate distance between car wash and nearest businesses
        for i, business in enumerate(nearest_businesses):
            business_latitude = business.get('location', {}).get('latitude')
            business_longitude = business.get('location', {}).get('longitude')
            if business_latitude and business_longitude:
                distance_car_wash_nearest_business = calculate_distance(
                    float(car_wash_latitude), float(car_wash_longitude),
                    float(business_latitude), float(business_longitude)
                )
                final_result['distance_car_wash_nearest_business_' + str(i+1)] = distance_car_wash_nearest_business
            else:
                final_result['distance_car_wash_nearest_business_' + str(i+1)] = None

        # print(f"Distance from input location: {distance:.2f} miles")
        # print(f"Car Wash: '{final_result['target_car_wash']['name']}'")
        # print(f"Address: {final_result['target_car_wash']['address']}'")
        # for i, business in enumerate(nearest_businesses):
        #     print(f"Nearest Business {i+1}: '{final_result['nearest_businesses'][i]['name']}'")
        #     print(f"Address: {final_result['nearest_businesses'][i]['address']}'")
        # print("--------------------")
    return final_result


if __name__ == "__main__":
    # --- Example Usage ---
    # Coordinates for a Shell station that includes a car wash and convenience store in San Francisco, CA
    # This is a good example of a mixed-use lot.
    EXAMPLE_LATITUDE = 34.4211501
    EXAMPLE_LONGITUDE = -103.1969689

    get_nearby_business_count(EXAMPLE_LATITUDE, EXAMPLE_LONGITUDE)
