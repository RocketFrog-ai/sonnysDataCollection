import pandas as pd
import os
import csv
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trafficLights.nearby_traffic_lights import get_nearby_traffic_lights, filter_duplicate_locations

def process_data(start_index, end_index):
    excel_file_path = 'trafficLights/1mile_raw_data.xlsx'
    output_csv_path = 'trafficLights/traffic_light_analysis.csv'

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) - 1}.")
        return

    fieldnames = [
        'full_site_address', 'Latitude', 'Longitude', 'nearby_traffic_lights_count'
    ]
    for i in range(1, 11):
        fieldnames.append(f'distance_nearest_traffic_light_{i}')
    
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
            sorted_traffic_lights = get_nearby_traffic_lights(latitude, longitude)
            unique_traffic_lights = filter_duplicate_locations(sorted_traffic_lights)
            
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
                'nearby_traffic_lights_count': len(unique_traffic_lights)
            }

            for i in range(10):
                if i < len(unique_traffic_lights):
                    output_row[f'distance_nearest_traffic_light_{i+1}'] = unique_traffic_lights[i]['distance_miles']
                else:
                    output_row[f'distance_nearest_traffic_light_{i+1}'] = None

        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
                'nearby_traffic_lights_count': 'ERROR'
            }
            for i in range(10):
                output_row[f'distance_nearest_traffic_light_{i+1}'] = None


        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output_row)
            print(f"  - Successfully processed and saved to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trafficLights/process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
