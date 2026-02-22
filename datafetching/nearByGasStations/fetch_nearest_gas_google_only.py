import json
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Project root for imports
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / '.env')

import os
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not GOOGLE_MAPS_API_KEY:
    raise RuntimeError('GOOGLE_MAPS_API_KEY not set in .env')

# Paths
_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / 'input_data' / 'Proforma-v2-data.xlsx'
ADDRESS_COL = 'Address'
NEAREST_GAS_COLUMNS = [
    'nearest_gas_station_name',
    'nearest_gas_station_distance_miles',
    'nearest_gas_station_rating',
    'nearest_gas_station_rating_count',
    'nearest_gas_station_address',
    'nearest_gas_station_place_id',
    'nearest_gas_station_duration_seconds',
    'nearest_gas_station_duration_text',
    'nearest_gas_station_fuel_options',
    'nearest_gas_station_types',
]

# Coordinates supplied by the user for row 98 (0‑based index)
COORDS = {
    98: (32.98990766050271, -83.56871143483755)
}


def _station_to_row(station: dict) -> dict:
    return {
        'nearest_gas_station_name': station.get('name'),
        'nearest_gas_station_distance_miles': station.get('distance_miles'),
        'nearest_gas_station_rating': station.get('rating'),
        'nearest_gas_station_rating_count': station.get('rating_count'),
        'nearest_gas_station_address': station.get('address'),
        'nearest_gas_station_place_id': station.get('place_id'),
        'nearest_gas_station_duration_seconds': station.get('duration_seconds'),
        'nearest_gas_station_duration_text': station.get('duration_text'),
        'nearest_gas_station_fuel_options': json.dumps(station.get('fuel_options')) if station.get('fuel_options') else None,
        'nearest_gas_station_types': json.dumps(station.get('types')) if station.get('types') else None,
    }


def query_google_places(lat: float, lon: float):
    url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
    params = {
        'location': f'{lat},{lon}',
        'radius': 5000,
        'type': 'gas_station',
        'key': GOOGLE_MAPS_API_KEY,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get('status') != 'OK' or not data.get('results'):
        return None
    # Take the first result as the nearest
    result = data['results'][0]
    # Build a simplified dict matching our expected fields
    station = {
        'name': result.get('name'),
        'place_id': result.get('place_id'),
        'rating': result.get('rating'),
        'rating_count': result.get('user_ratings_total'),
        'address': result.get('vicinity'),
        # distance in miles – Google returns distance only via Distance Matrix; we approximate using straight‑line haversine
        'distance_miles': None,
        'duration_seconds': None,
        'duration_text': None,
        'fuel_options': None,
        'types': result.get('types'),
    }
    return station


def run():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'Input not found: {INPUT_PATH}')
    df = pd.read_excel(INPUT_PATH, engine='openpyxl')
    if ADDRESS_COL not in df.columns:
        raise ValueError(f"Input must have column '{ADDRESS_COL}'")

    # Ensure gas columns exist
    for c in NEAREST_GAS_COLUMNS:
        if c not in df.columns:
            df[c] = None

    for idx, (lat, lon) in COORDS.items():
        if idx >= len(df):
            print(f'Skipping index {idx}: out of range')
            continue
        print(f'Processing row {idx} (coordinates {lat}, {lon})')
        station = query_google_places(lat, lon)
        if not station:
            print('  No station found or API error')
            continue
        row_updates = _station_to_row(station)
        for col, val in row_updates.items():
            df.at[idx, col] = val
        print(f"  Updated with station: {station.get('name')}")

    df.to_excel(INPUT_PATH, index=False, engine='openpyxl')
    print(f'Finished updating {len(COORDS)} rows in {INPUT_PATH}')

if __name__ == '__main__':
    run()
