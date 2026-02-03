import pandas as pd
import os
import csv
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speedLimits.speed_limits import get_nearest_roads_with_speed

def process_data(start_index, end_index):
    excel_file_path = 'trafficLights/1mile_raw_data.xlsx'
    output_csv_path = 'speedLimits/process_data.csv'

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) - 1}.")
        return

    fieldnames = [
        'full_site_address', 'Latitude', 'Longitude',
        'nearestroad_1_name', 'distance_nearestroad_1', 'nearestroad_1_maxspeed',
        'nearestroad_2_name', 'distance_nearestroad_2', 'nearestroad_2_maxspeed',
        'nearestroad_3_name', 'distance_nearestroad_3', 'nearestroad_3_maxspeed',
        'nearestroad_4_name', 'distance_nearestroad_4', 'nearestroad_4_maxspeed',
        'nearestroad_5_name', 'distance_nearestroad_5', 'nearestroad_5_maxspeed'
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
            nearest_roads = get_nearest_roads_with_speed(latitude, longitude, 3218.69)

            closest_roads = {}
            unique_roads = []
            for road in nearest_roads:
                road_name = road['name']
                if road_name not in closest_roads:
                    closest_roads[road_name] = {'id': road['id'], 'distance': road['distance'], 'maxspeed': road['maxspeed']}
                    unique_roads.append(road)
                else:
                    if road['distance'] < closest_roads[road_name]['distance']:
                        closest_roads[road_name] = {'id': road['id'], 'distance': road['distance'], 'maxspeed': road['maxspeed']}

            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
            }

            for i in range(min(5, len(unique_roads))):
                output_row[f'nearestroad_{i+1}_name'] = unique_roads[i]['name']
                output_row[f'distance_nearestroad_{i+1}'] = unique_roads[i]['distance']
                output_row[f'nearestroad_{i+1}_maxspeed'] = unique_roads[i].get('maxspeed', 'N/A')
            
            for i in range(len(unique_roads), 5):
                output_row[f'nearestroad_{i+1}_name'] = None
                output_row[f'distance_nearestroad_{i+1}'] = None
                output_row[f'nearestroad_{i+1}_maxspeed'] = None

        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
            }
            for i in range(1, 6):
                output_row[f'nearestroad_{i}_name'] = 'ERROR'
                output_row[f'distance_nearestroad_{i}'] = 'ERROR'
                output_row[f'nearestroad_{i}_maxspeed'] = 'ERROR'


        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output_row)
            print(f"  - Successfully processed and saved to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python speedLimits/process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
