import pandas as pd
import os
import csv
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nearbyBusinesses.nearby_businesses import get_nearby_business_count

def process_data(start_index, end_index):
    excel_file_path = 'trafficLights/1mile_raw_data.xlsx'
    output_csv_path = 'nearbyBusinesses/nearby_businesses_analysis.csv'

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) - 1}.")
        return

    fieldnames = [
        'full_site_address', 'Latitude', 'Longitude', 'display_name', 'actual_latitude', 'actual_longitude', 'nearby_businesses_count', 'business_names'
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
            nearby_businesses_data = get_nearby_business_count(latitude, longitude)
            
            if nearby_businesses_data:
                output_row = {
                    'full_site_address': full_site_address,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'display_name': nearby_businesses_data['target_car_wash']['name'],
                    'actual_latitude': nearby_businesses_data['target_car_wash']['location']['latitude'],
                    'actual_longitude': nearby_businesses_data['target_car_wash']['location']['longitude'],
                    'nearby_businesses_count': nearby_businesses_data['nearby_business_count'],
                    'business_names': ', '.join([business['name'] for business in nearby_businesses_data['associated_businesses']])
                }
            else:
                output_row = {
                    'full_site_address': full_site_address,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'display_name': 'ERROR',
                    'actual_latitude': 'ERROR',
                    'actual_longitude': 'ERROR',
                    'nearby_businesses_count': 'ERROR',
                    'business_names': 'ERROR'
                }

        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
                'display_name': 'ERROR',
                'actual_latitude': 'ERROR',
                'actual_longitude': 'ERROR',
                'nearby_businesses_count': 'ERROR',
                'business_names': 'ERROR'
            }


        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output_row)
            print(f"  - Successfully processed and saved to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nearbyBusinesses/process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
