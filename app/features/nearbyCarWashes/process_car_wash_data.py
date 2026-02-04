import pandas as pd
import json
import math
from searchNearbyAll import find_nearby_places
from app.utils import common as calib


API_KEY = calib.GOOGLE_MAPS_API_KEY
EXCEL_FILE_PATH = "nearbyCarWashes/1mile_raw_data.xlsx" # Assuming this is the correct file based on previous context
OUTPUT_CSV_PATH = "nearbyCarWashes/car_wash_analysis.csv"

def process_car_wash_data(api_key, excel_file_path, output_csv_path):
    """
    Reads site data from an Excel file, finds nearby car washes using Google Places API,
    and outputs a CSV with car wash analysis.
    """
    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    results_list = []

    for index, row in df.iterrows():
        site_name = row['full_site_address']
        latitude = row['Latitude']
        longitude = row['Longitude']

        print(f"Processing site: {site_name} (Lat: {latitude}, Lon: {longitude})")

        # Call the find_nearby_places function
        # We request more results to ensure we get at least the second nearest if the first is the query location
        car_washes = find_nearby_places(
            api_key,
            latitude,
            longitude,
            radius_miles=1, # Using default 1 mile radius as per searchNearbyAll.py
            included_types=['car_wash'],
            max_results=20, # Requesting more results to ensure we get at least the second nearest
            rank_preference="DISTANCE"
        )

        num_nearby_car_washes = 0
        distance_from_nearest = "N/A"
        rating_of_nearest = "N/A"
        user_rating_count = "N/A"

        if car_washes and len(car_washes) > 1:
            num_nearby_car_washes = len(car_washes) - 1 # Exclude the query location itself
            
            # The actual second nearest is now considered the first nearest
            first_nearest_car_wash_actual = car_washes[1] 

            rating_of_nearest = first_nearest_car_wash_actual.get("rating", "N/A")
            user_rating_count = first_nearest_car_wash_actual.get("userRatingCount", "N/A")

            # Calculate distance to the actual second nearest car wash
            first_nearest_lat = first_nearest_car_wash_actual.get("location", {}).get("latitude")
            first_nearest_lon = first_nearest_car_wash_actual.get("location", {}).get("longitude")

            if first_nearest_lat is not None and first_nearest_lon is not None:
                # Haversine formula (copied from searchNearbyAll.py for self-containment, or could import if preferred)
                R = 6371  # Radius of Earth in kilometers
                lat1_rad = math.radians(latitude)
                lon1_rad = math.radians(longitude)
                lat2_rad = math.radians(first_nearest_lat)
                lon2_rad = math.radians(first_nearest_lon)

                dlon = lon2_rad - lon1_rad
                dlat = lat2_rad - lat1_rad

                a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

                distance_km = R * c
                distance_from_nearest = f"{distance_km * 0.621371:.2f} miles" # Convert km to miles

        elif car_washes and len(car_washes) == 1:
            print(f"  Found 1 car wash (likely the query location itself). No other car washes found.")
            num_nearby_car_washes = 0 # No other car washes found

        else:
            print(f"  No car washes found for {site_name}.")

        results_list.append({
            "site name": site_name, # Keep this column name for output CSV
            "latitude": latitude,
            "longitude": longitude,
            "number of nearby carwashes": num_nearby_car_washes,
            "distance from nearest": distance_from_nearest,
            "rating of nearest": rating_of_nearest,
            "user rating count": user_rating_count
        })

    output_df = pd.DataFrame(results_list)
    try:
        output_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccessfully generated CSV: {output_csv_path}")
    except IOError as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    # Ensure the API_KEY is set before running
    if API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual Google Maps Platform API Key in the script.")
    else:
        process_car_wash_data(API_KEY, EXCEL_FILE_PATH, OUTPUT_CSV_PATH)
