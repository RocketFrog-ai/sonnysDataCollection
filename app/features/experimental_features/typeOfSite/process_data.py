import pandas as pd
import os
import csv
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typeOfSite.get_satellite_images import download_satellite_images
from typeOfSite.o4mini_images_classification import visionModelResponse

def process_data(start_index, end_index):
    excel_file_path = 'climate/unscaled_clean_dataset.xlsx'
    output_csv_path = 'typeOfSite/type_of_lot.csv'

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
        'new_latitude', 'new_longitude', 'carwash_name_in_image',
        'classification', 'justification'
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

        # Step 1: Download satellite images
        try:
            new_latitude, new_longitude, location_dir, nearby_car_wash_name = download_satellite_images(latitude, longitude, full_site_address)
        except Exception as e:
            print(f"  - An unexpected error occurred during image download: {e}")
            location_dir = None
            nearby_car_wash_name = None
            new_latitude, new_longitude = latitude, longitude


        if not location_dir or not os.listdir(location_dir):
            print("  - Failed to download images or folder is empty, skipping vision model.")
            vision_response = {"error": "Image download failed"}
        else:
            # Step 2: Get vision model response with retry logic
            retries = 10
            for attempt in range(retries):
                vision_response = visionModelResponse(location_dir)
                if "error" not in vision_response:
                    break
                print(f"  - Attempt {attempt + 1} failed: {vision_response.get('error')}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  - All {retries} attempts failed. Skipping record.")

        # Step 3: Write to CSV
        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if vision_response and 'error' not in vision_response:
                writer.writerow({
                    'full_site_address': full_site_address,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'new_latitude': new_latitude,
                    'new_longitude': new_longitude,
                    'carwash_name_in_image': nearby_car_wash_name,
                    'classification': vision_response.get('classification'),
                    'justification': vision_response.get('justification')
                })
                print(f"  - Successfully processed and saved to CSV.")
            else:
                print(f"  - Error from vision model: {vision_response.get('error')}")
                # Optionally write error to CSV
                writer.writerow({
                    'full_site_address': full_site_address,
                    'Latitude': latitude,
                    'Longitude': longitude,
                    'new_latitude': new_latitude,
                    'new_longitude': new_longitude,
                    'carwash_name_in_image': nearby_car_wash_name,
                    'classification': 'ERROR',
                    'justification': vision_response.get('error')
                })


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
