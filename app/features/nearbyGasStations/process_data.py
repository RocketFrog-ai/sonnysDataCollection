import os
import csv
import sys
import pandas as pd
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nearbyGasStations.get_nearby_gas_stations import (
    get_nearby_gas_stations,
    DEFAULT_MAX_GAS_STATIONS,
    DEFAULT_RADIUS_MILES,
)

load_dotenv()


def process_data(start_index, end_index):
    excel_file_path = 'trafficLights/1mile_raw_data.xlsx'
    output_csv_path = 'nearbyGasStations/nearby_gas_stations.csv'
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    if not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not found in .env file.")
        return

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Use 0 to {len(df) - 1}.")
        return

    day_columns = [
        'monday_hours', 'tuesday_hours', 'wednesday_hours', 'thursday_hours',
        'friday_hours', 'saturday_hours', 'sunday_hours'
    ]
    fieldnames = [
        'full_site_address', 'Latitude', 'Longitude', 'nearby_gas_stations_count'
    ]
    for i in range(1, DEFAULT_MAX_GAS_STATIONS + 1):
        fieldnames.append(f'gas_station_name_{i}')
        fieldnames.append(f'distance_gas_station_{i}')
        fieldnames.append(f'rating_gas_station_{i}')
        for d in day_columns:
            fieldnames.append(f'{d}_{i}')

    file_exists = os.path.isfile(output_csv_path)
    if not file_exists:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for index, row in df.iloc[start_index:end_index].iterrows():
        full_site_address = row['full_site_address']
        latitude = row['Latitude']
        longitude = row['Longitude']
        print(f"--- Processing record {index}: {full_site_address} ---")

        try:
            stations = get_nearby_gas_stations(
                api_key, latitude, longitude,
                radius_miles=DEFAULT_RADIUS_MILES,
                max_results=DEFAULT_MAX_GAS_STATIONS
            )
        except Exception as e:
            print(f"  - Error: {e}")
            stations = []

        output_row = {
            'full_site_address': full_site_address,
            'Latitude': latitude,
            'Longitude': longitude,
            'nearby_gas_stations_count': len(stations)
        }
        for i in range(1, DEFAULT_MAX_GAS_STATIONS + 1):
            idx = i - 1
            if idx < len(stations):
                s = stations[idx]
                output_row[f'gas_station_name_{i}'] = s.get('name', 'N/A')
                output_row[f'distance_gas_station_{i}'] = s.get('distance_miles')
                output_row[f'rating_gas_station_{i}'] = s.get('rating', 'N/A')
                for d in day_columns:
                    output_row[f'{d}_{i}'] = s.get(d, 'N/A')
            else:
                output_row[f'gas_station_name_{i}'] = ''
                output_row[f'distance_gas_station_{i}'] = None
                output_row[f'rating_gas_station_{i}'] = ''
                for d in day_columns:
                    output_row[f'{d}_{i}'] = ''

        with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(output_row)
        print(f"  - Wrote {len(stations)} gas stations to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nearbyGasStations/process_data.py <start_index> <end_index>")
        sys.exit(1)
    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index. Use integers.")
        sys.exit(1)
