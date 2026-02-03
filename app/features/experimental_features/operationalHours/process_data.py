import pandas as pd
import os
import csv
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operationalHours.searchNearby import find_nearby_places

load_dotenv()

def process_data(start_index, end_index):
    excel_file_path = 'trafficLights/1mile_raw_data.xlsx'
    output_csv_path = 'operationalHours/operational_hours_analysis.csv'
    API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

    if not API_KEY:
        print("Error: GOOGLE_MAPS_API_KEY not found in .env file.")
        return

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) - 1}.")
        return

    fieldnames = [
        'full_site_address', 'Latitude', 'Longitude', 'display_name',
        'actual_latitude', 'actual_longitude', 'rating', 'rating_count',
        'business_status', 'monday_operational_hours', 'tuesday_operational_hours',
        'wednesday_operational_hours', 'thursday_operational_hours',
        'friday_operational_hours', 'saturday_operational_hours', 'sunday_operational_hours'
    ]
    
    # Prepare CSV file
    file_exists = os.path.isfile(output_csv_path)
    if not file_exists:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for index, row in df.iloc[start_index:end_index].iterrows():
        full_site_address = row['full_site_address']
        latitude = row['Latitude']
        longitude = row['Longitude']

        print(f"--- Processing record {index}: {full_site_address} ---")

        try:
            # Assuming we are looking for car washes as in the example
            results = find_nearby_places(
                API_KEY,
                latitude,
                longitude,
                radius_miles=2,
                included_types=['car_wash'],
                max_results=1,
                rank_preference="DISTANCE"
            )
            
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
            }

            if results and "places" in results and results["places"]:
                nearest_place = results["places"][0]
                
                output_row['display_name'] = nearest_place.get("displayName", {}).get("text", "N/A")
                location = nearest_place.get("location", {})
                output_row['actual_latitude'] = location.get("latitude", "N/A")
                output_row['actual_longitude'] = location.get("longitude", "N/A")
                output_row['rating'] = nearest_place.get("rating", "N/A")
                output_row['rating_count'] = nearest_place.get("userRatingCount", "N/A")
                output_row['business_status'] = nearest_place.get("businessStatus", "N/A")
                
                opening_hours = nearest_place.get("regularOpeningHours", {})
                weekday_descriptions = opening_hours.get("weekdayDescriptions", [])

                days_hours = {day: "N/A" for day in fieldnames[9:]}

                if weekday_descriptions:
                    for desc in weekday_descriptions:
                        cleaned_desc = desc.replace('\u202f', ' ').replace('\u2009', ' ')
                        parts = cleaned_desc.split(':', 1)
                        if len(parts) == 2:
                            day_name = parts[0].strip().lower()
                            hours = parts[1].strip()
                            column_name = f"{day_name}_operational_hours"
                            if column_name in days_hours:
                                days_hours[column_name] = hours
                
                for day, hours in days_hours.items():
                    output_row[day] = hours

            else:
                for col in fieldnames[3:]:
                    output_row[col] = 'N/A'


        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
            }
            for col in fieldnames[3:]:
                output_row[col] = 'ERROR'

        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output_row)
            print(f"  - Successfully processed and saved to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python operationalHours/process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
