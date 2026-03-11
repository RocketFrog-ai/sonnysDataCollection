import pandas as pd
import os
import csv
import sys
from nearbyBusinesses.nearby_businesses import get_nearby_business_count


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
        'full_site_address', 'Latitude', 'Longitude', 'car_wash_name', 'distance_from_original_location', 'car_wash_address',
        'nearest_business_name_1', 'nearest_business_address_1', 'distance_car_wash_nearest_business_1',
        'nearest_business_name_2', 'nearest_business_address_2', 'distance_car_wash_nearest_business_2',
        'nearest_business_name_3', 'nearest_business_address_3', 'distance_car_wash_nearest_business_3',
        'nearest_business_name_4', 'nearest_business_address_4', 'distance_car_wash_nearest_business_4',
        'nearest_business_name_5', 'nearest_business_address_5', 'distance_car_wash_nearest_business_5',
        'multiple_business_count'
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
        print(f"  - Latitude: {latitude}, Longitude: {longitude}")

        try:
            nearby_businesses_data = get_nearby_business_count(latitude, longitude)
            
            # print(f"  - Nearby businesses data: {nearby_businesses_data}")

            if nearby_businesses_data:
                # print(f"  - Target car wash: {nearby_businesses_data['target_car_wash']}")
                distance = nearby_businesses_data.get('distance', '')
                
                # Extract nearest businesses data
                nearest_businesses = nearby_businesses_data.get('nearest_businesses', [])
                
                # Calculate multiple_business_count
                multiple_business_count = sum(1 for business in nearest_businesses if business['address'] == nearby_businesses_data['target_car_wash']['address'])
                
                output_row = {
                    'full_site_address': full_site_address,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'car_wash_name': nearby_businesses_data['target_car_wash']['name'],
                    'distance_from_original_location': distance,
                    'car_wash_address': nearby_businesses_data['target_car_wash']['address'],
                    'nearest_business_name_1': nearest_businesses[0]['name'] if len(nearest_businesses) > 0 else '',
                    'nearest_business_address_1': nearest_businesses[0]['address'] if len(nearest_businesses) > 0 else '',
                    'distance_car_wash_nearest_business_1': nearby_businesses_data.get('distance_car_wash_nearest_business_1', ''),
                    'nearest_business_name_2': nearest_businesses[1]['name'] if len(nearest_businesses) > 1 else '',
                    'nearest_business_address_2': nearest_businesses[1]['address'] if len(nearest_businesses) > 1 else '',
                    'distance_car_wash_nearest_business_2': nearby_businesses_data.get('distance_car_wash_nearest_business_2', ''),
                    'nearest_business_name_3': nearest_businesses[2]['name'] if len(nearest_businesses) > 2 else '',
                    'nearest_business_address_3': nearest_businesses[2]['address'] if len(nearest_businesses) > 2 else '',
                    'distance_car_wash_nearest_business_3': nearby_businesses_data.get('distance_car_wash_nearest_business_3', ''),
                    'nearest_business_name_4': nearest_businesses[3]['name'] if len(nearest_businesses) > 3 else '',
                    'nearest_business_address_4': nearest_businesses[3]['address'] if len(nearest_businesses) > 3 else '',
                    'distance_car_wash_nearest_business_4': nearby_businesses_data.get('distance_car_wash_nearest_business_4', ''),
                    'nearest_business_name_5': nearest_businesses[4]['name'] if len(nearest_businesses) > 4 else '',
                    'nearest_business_address_5': nearest_businesses[4]['address'] if len(nearest_businesses) > 4 else '',
                    'distance_car_wash_nearest_business_5': nearby_businesses_data.get('distance_car_wash_nearest_business_5', ''),
                    'multiple_business_count': multiple_business_count
                }
            else:
                output_row = {
                    'full_site_address': full_site_address,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'car_wash_name': '',
                    'distance_from_original_location': '',
                    'car_wash_address': '',
                    'nearest_business_name_1': '',
                    'nearest_business_address_1': '',
                    'distance_car_wash_nearest_business_1': '',
                    'nearest_business_name_2': '',
                    'nearest_business_address_2': '',
                    'distance_car_wash_nearest_business_2': '',
                    'nearest_business_name_3': '',
                    'nearest_business_address_3': '',
                    'distance_car_wash_nearest_business_3': '',
                    'nearest_business_name_4': '',
                    'nearest_business_address_4': '',
                    'distance_car_wash_nearest_business_4': '',
                    'nearest_business_name_5': '',
                    'nearest_business_address_5': '',
                    'distance_car_wash_nearest_business_5': '',
                    'multiple_business_count': 0
                }

        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
                'car_wash_name': '',
                'distance_from_original_location': '',
                'car_wash_address': '',
                'nearest_business_name_1': '',
                'nearest_business_address_1': '',
                'distance_car_wash_nearest_business_1': '',
                'nearest_business_name_2': '',
                'nearest_business_address_2': '',
                'distance_car_wash_nearest_business_2': '',
                'nearest_business_name_3': '',
                'nearest_business_address_3': '',
                'distance_car_wash_nearest_business_3': '',
                'nearest_business_name_4': '',
                'nearest_business_address_4': '',
                'distance_car_wash_nearest_business_4': '',
                'nearest_business_name_5': '',
                'nearest_business_address_5': '',
                'distance_car_wash_nearest_business_5': '',
                'multiple_business_count': 0
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
