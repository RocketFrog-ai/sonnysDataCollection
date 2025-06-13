import pandas as pd
from geocoding import get_address_from_coordinates

# Replace with your Google Geocoding API key
API_KEY = "AIzaSyDq7-QEyUIezRe5xMf7LxsQBmnNOmRnfho"  # Replace with your actual key if different

def process_excel_data_and_get_addresses():
    """
    Reads the first 50 records from an Excel file, retrieves the address for each
    set of coordinates, and prints the addresses to the terminal.
    """
    excel_file_path = 'competitors/datasets/1mile_raw_data.xlsx'

    try:
        df = pd.read_excel(excel_file_path, nrows=50)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    for index, row in df.iterrows():
        latitude = row.iloc[1]
        longitude = row.iloc[2]

        print(f"Processing record {index + 1}: Coordinates ({latitude}, {longitude})")
        address = get_address_from_coordinates(latitude, longitude, API_KEY)
        print(f"  - Address: {address}\n")

if __name__ == "__main__":
    if API_KEY == "YOUR_GOOGLE_MAPS_API_KEY":
        print("Please replace 'YOUR_GOOGLE_MAPS_API_KEY' with your actual API key in the script.")
    else:
        process_excel_data_and_get_addresses()
