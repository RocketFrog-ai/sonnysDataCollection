import requests
import json

# --- Configuration ---
# IMPORTANT: Replace with your actual API Key
API_KEY = "6MXNP_glhi1-xrIXGakfcKCYixg0k5gTRnF4p22W7k1s3TdzW0e41BanoZX6a-8mxmv6EeJwjr8jtXKW9nfkXpgzUlHTtu2m8ii8AOMPzeVgPkJoUxZa8ECGNlcvaHYx" # <--- REPLACE THIS!
BASE_URL = "https://api.yelp.com/v3/"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"
}

def search_businesses_by_coordinates(latitude, longitude, term="car wash", radius=5000, limit=10):
    """
    Searches for businesses near given coordinates.
    Args:
        latitude (float): Latitude of the search center.
        longitude (float): Longitude of the search center.
        term (str): Search term (e.g., "car wash", "restaurants").
        radius (int): Search radius in meters (max 40000, approx 25 miles).
        limit (int): Number of businesses to return (max 50).
    Returns:
        dict: JSON response from Yelp API or None if error.
    """
    endpoint = f"{BASE_URL}businesses/search"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "term": term,
        "radius": radius, # in meters
        "limit": limit,
        "sort_by": "distance" # or "best_match", "rating", "review_count"
    }

    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.content}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except json.JSONDecodeError:
        print("Error decoding JSON response from Yelp.")
        print(f"Response content: {response.text}")
    return None

def get_business_details(business_id):
    """
    Fetches detailed information for a specific business.
    Args:
        business_id (str): The Yelp ID of the business.
    Returns:
        dict: JSON response with business details or None if error.
    """
    endpoint = f"{BASE_URL}businesses/{business_id}"
    try:
        response = requests.get(endpoint, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while fetching details: {http_err}")
        print(f"Response content: {response.content}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred while fetching details: {req_err}")
    except json.JSONDecodeError:
        print("Error decoding JSON response from Yelp for business details.")
        print(f"Response content: {response.text}")
    return None

# --- Main Execution ---
if __name__ == "__main__":
    if API_KEY == "YOUR_YELP_API_KEY":
        print("ERROR: Please replace 'YOUR_YELP_API_KEY' with your actual Yelp API Key.")
    else:
        # Example coordinates (e.g., somewhere in San Francisco)
        # You can get these from Google Maps or another geolocation service
        example_latitude = 37.7749
        example_longitude = -122.4194

        print(f"Searching for car washes near Lat: {example_latitude}, Lon: {example_longitude}...\n")
        search_results = search_businesses_by_coordinates(example_latitude, example_longitude, term="car wash")

        if search_results and "businesses" in search_results:
            businesses = search_results["businesses"]
            if not businesses:
                print("No car washes found near this location.")
            else:
                print(f"Found {len(businesses)} car wash(es):")
                for i, business in enumerate(businesses):
                    print(f"\n--- {i+1}. {business.get('name', 'N/A')} ---")
                    print(f"  ID: {business.get('id', 'N/A')}")
                    print(f"  Rating: {business.get('rating', 'N/A')} ({business.get('review_count', 'N/A')} reviews)")
                    print(f"  Address: {', '.join(business.get('location', {}).get('display_address', ['N/A']))}")
                    print(f"  Distance: {business.get('distance', 'N/A'):.2f} meters") # distance is in meters

                    # "Services" are primarily indicated by categories
                    categories = business.get('categories', [])
                    if categories:
                        category_titles = [cat['title'] for cat in categories]
                        print(f"  Categories (indicating services offered): {', '.join(category_titles)}")
                    else:
                        print("  Categories: Not specified")

                    # Optionally, get more details for the first business found
                    if i == 0: # Let's get full details for the first result
                        print(f"\n  Fetching more details for '{business.get('name', 'N/A')}'...")
                        details = get_business_details(business.get('id'))
                        if details:
                            print(f"    Phone: {details.get('display_phone', 'N/A')}")
                            print(f"    Is currently open: {not details.get('is_closed', True) if 'hours' in details else 'Hours not available'}") # is_closed refers to permanently closed
                            if 'hours' in details and details['hours']:
                                print(f"    Open now: {details['hours'][0].get('is_open_now', 'Unknown')}")
                            # The 'special_hours' field might contain more info if available
                            # The 'attributes' field might sometimes contain service-related tags but it's not standardized for specific services
                            # For instance, attributes might include "garage_parking", "street_parking", "bike_parking", "wheelchair_accessible"
                            # but unlikely "Basic Wash", "Premium Wax".
                        else:
                            print("    Could not fetch additional details.")
        else:
            print("Failed to retrieve search results from Yelp.")