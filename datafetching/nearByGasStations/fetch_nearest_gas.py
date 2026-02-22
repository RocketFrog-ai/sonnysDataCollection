import sys
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Project root on path for app imports (nearByGasStations -> datafetching -> project root)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.utils import common as calib
from app.features.nearbyGasStations.get_nearby_gas_stations import get_nearest_gas_station_only

# Input/output: datafetching/input_data/Proforma-v2-data.xlsx
_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "input_data" / "Proforma-v2-data.xlsx"
ADDRESS_COL = "Address"

NEAREST_GAS_COLUMNS = [
    "nearest_gas_station_name",
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
    "nearest_gas_station_address",
    "nearest_gas_station_place_id",
    "nearest_gas_station_duration_seconds",
    "nearest_gas_station_duration_text",
    "nearest_gas_station_fuel_options",
    "nearest_gas_station_types",
]


def _station_to_row(station: dict) -> dict:
    row = {
        "nearest_gas_station_name": station.get("name"),
        "nearest_gas_station_distance_miles": station.get("distance_miles"),
        "nearest_gas_station_rating": station.get("rating"),
        "nearest_gas_station_rating_count": station.get("rating_count"),
        "nearest_gas_station_address": station.get("address"),
        "nearest_gas_station_place_id": station.get("place_id"),
        "nearest_gas_station_duration_seconds": station.get("duration_seconds"),
        "nearest_gas_station_duration_text": station.get("duration_text"),
        "nearest_gas_station_fuel_options": json.dumps(station.get("fuel_options")) if station.get("fuel_options") is not None else None,
        "nearest_gas_station_types": json.dumps(station.get("types")) if station.get("types") is not None else None,
    }
    return row


def run(start_index: int = 0, end_index: int | None = None, fetch_place_details: bool = True) -> None:
    api_key = calib.GOOGLE_MAPS_API_KEY
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set in .env")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    df = pd.read_excel(INPUT_PATH, engine="openpyxl")
    if ADDRESS_COL not in df.columns:
        raise ValueError(f"Input must have column '{ADDRESS_COL}'")

    n = len(df)
    end_index = end_index if end_index is not None else n
    start_index = max(0, start_index)
    end_index = min(end_index, n)
    if start_index >= end_index:
        print("No rows to process (start_index >= end_index).")
        return

    new_rows = []
    for i in tqdm(range(start_index, end_index), desc="Nearest gas", unit="row"):
        row = df.iloc[i]
        address = row[ADDRESS_COL]
        if pd.isna(address) or not str(address).strip():
            new_row = row.to_dict()
            for c in NEAREST_GAS_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            print(f"  [{i}] skip (no address)")
            continue

        geo = calib.get_lat_long(str(address))
        if not geo:
            new_row = row.to_dict()
            for c in NEAREST_GAS_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            print(f"  [{i}] skip (geocode failed): {str(address)[:50]}...")
            continue

        lat, lon = geo["lat"], geo["lon"]
        try:
            station = get_nearest_gas_station_only(
                api_key, float(lat), float(lon),
                fetch_place_details=fetch_place_details,
            )
        except Exception as e:
            print(f"  [{i}] error: {e}")
            new_row = row.to_dict()
            for c in NEAREST_GAS_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        new_row = row.to_dict()
        if station:
            for k, v in _station_to_row(station).items():
                new_row[k] = v
            print(f"  [{i}] {station.get('name', 'N/A')} @ {station.get('distance_miles')} mi")
        else:
            for c in NEAREST_GAS_COLUMNS:
                new_row[c] = None
            print(f"  [{i}] no gas station found")
        new_rows.append(new_row)

    slice_df = pd.DataFrame(new_rows)
    if len(slice_df) == 0:
        print("No rows to write.")
        return

    # Add new columns to full df if not present, then update only the processed slice
    for c in NEAREST_GAS_COLUMNS:
        if c not in df.columns:
            df[c] = None
    for c in slice_df.columns:
        if c in df.columns:
            df.loc[df.index[start_index:end_index], c] = slice_df[c].values
    df.to_excel(INPUT_PATH, index=False, engine="openpyxl")
    print(f"Updated rows {start_index}–{end_index - 1} in {INPUT_PATH}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(start_index=start, end_index=end)
