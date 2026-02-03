import pandas as pd
import os
import csv
import sys
import time

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from climate.open_meteo import get_climate_data

def process_data(start_index, end_index):
    excel_file_path = 'climate/unscaled_clean_dataset.xlsx'
    output_csv_path = 'climate/climate_analysis.csv'

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
        'total_precipitation_mm', 'rainy_days', 'total_snowfall_cm', 'snowy_days',
        'days_below_freezing', 'total_sunshine_hours', 'days_pleasant_temp',
        'avg_daily_max_windspeed_ms'
    ]
    
    # Prepare CSV file
    file_exists = os.path.isfile(output_csv_path)
    if not file_exists:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


    for index, row in df.iloc[start_index:end_index].iterrows():
        time.sleep(30) # Add a 1-second delay to avoid rate limiting
        full_site_address = row['full_site_address']
        latitude = row['Latitude']
        longitude = row['Longitude']

        print(f"--- Processing record {index}: {full_site_address} ---")

        try:
            climate_data = get_climate_data(latitude, longitude)
            
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
            }

            if climate_data:
                output_row.update(climate_data)
            else:
                for key in fieldnames[3:]:
                    output_row[key] = 'ERROR'


        except Exception as e:
            print(f"  - An unexpected error occurred: {e}")
            output_row = {
                'full_site_address': full_site_address,
                'Latitude': latitude,
                'Longitude': longitude,
            }
            for key in fieldnames[3:]:
                output_row[key] = 'ERROR'


        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(output_row)
            print(f"  - Successfully processed and saved to CSV.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python climate/process_data.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        process_data(start, end)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
