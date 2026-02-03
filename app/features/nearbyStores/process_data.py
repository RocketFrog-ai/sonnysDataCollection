import pandas as pd
import os
import csv
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nearbyStores.nearby_costcos import get_costco_info

def process_data(start_index, end_index):
    excel_file_path = 'climate/unscaled_clean_dataset.xlsx'
    output_csv_path = 'nearbyStores/Kohls_5miles.csv'

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) - 1}.")
        return

    fieldnames = [
        'full_site_address', 'Latitude', 'Longitude', 'car_wash_name',
        'distance_from_nearest_costco', 'count_of_costco_5miles'
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
            costco_data = get_costco_info(latitude, longitude)
            
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
                'car_wash_name': costco_data.get('car_wash_name') if costco_data else 'N/A',
                'distance_from_nearest_costco': costco_data.get('distance_from_nearest_costco') if costco_data else None,
                'count_of_costco_5miles': costco_data.get('count_of_costco_5miles') if costco_data else 0
            }

        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
                'car_wash_name': 'Error',
                'distance_from_nearest_costco': None,
                'count_of_costco_5miles': 0
            }

        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output_row)
            print(f"  - Successfully processed and saved to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python costco/process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
