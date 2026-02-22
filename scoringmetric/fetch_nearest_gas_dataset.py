"""
Batch fetch nearest gas station for each address in dataSET (1).xlsx and append
results to main_dataset.xlsx. Same file will be used for other features later.

Usage (from project root):
  python scoringmetric/fetch_nearest_gas_dataset.py [start_index] [end_index]

Run in order (e.g. 0 100, then 100 200, ...) to build the full file.
Output: scoringmetric/main_dataset.xlsx (append-only).
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd

# Project root on path for app imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.utils import common as calib
from app.features.nearbyGasStations.get_nearby_gas_stations import get_nearest_gas_station_only

# Column names for the single nearest gas station (aligned with /gas_station response)
NEAREST_GAS_COLUMNS = [
    "nearest_gas_station_name",
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
    "nearest_gas_station_address",
    "nearest_gas_station_place_id",
    "nearest_gas_station_duration_seconds",
    "nearest_gas_station_duration_text",
    "nearest_gas_station_fuel_options",  # JSON string if present
    "nearest_gas_station_types",         # JSON string if present
]


def _find_dataset_path() -> Path:
    base = Path(__file__).resolve().parent
    for p in [base / "dataSET (1).xlsx", Path("dataSET (1).xlsx"), base.parent / "dataSET (1).xlsx"]:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find dataSET (1).xlsx in scoringmetric, cwd, or project root.")


def _station_to_row(station: dict) -> dict:
    """Map one gas station dict (from get_nearest_gas_station_only) to flat row dict."""
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

    data_path = _find_dataset_path()
    df = pd.read_excel(data_path, engine="openpyxl")

    address_col = "full_site_address"
    if address_col not in df.columns:
        raise ValueError(f"Dataset must have column '{address_col}'")

    n = len(df)
    end_index = end_index if end_index is not None else n
    start_index = max(0, start_index)
    end_index = min(end_index, n)
    if start_index >= end_index:
        print("No rows to process (start_index >= end_index).")
        return

    # Optional: use existing Latitude/Longitude if present
    has_latlon = "Latitude" in df.columns and "Longitude" in df.columns

    out_path = Path(__file__).resolve().parent / "main_dataset.xlsx"
    # Build new rows for the slice [start_index:end_index]
    new_rows = []
    for i in range(start_index, end_index):
        row = df.iloc[i]
        address = row[address_col]
        if pd.isna(address) or not str(address).strip():
            new_row = row.to_dict()
            for c in NEAREST_GAS_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            print(f"  [{i}] skip (no address)")
            continue

        lat, lon = None, None
        if has_latlon:
            lat, lon = row.get("Latitude"), row.get("Longitude")
        if lat is None or lon is None or (pd.isna(lat) or pd.isna(lon)):
            geo = calib.get_lat_long(str(address))
            if not geo:
                new_row = row.to_dict()
                for c in NEAREST_GAS_COLUMNS:
                    new_row[c] = None
                new_rows.append(new_row)
                print(f"  [{i}] skip (geocode failed): {address[:50]}...")
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

    if out_path.exists():
        existing = pd.read_excel(out_path, engine="openpyxl")
        # Align columns: add any new columns to existing, then concat
        for c in slice_df.columns:
            if c not in existing.columns:
                existing[c] = None
        for c in existing.columns:
            if c not in slice_df.columns:
                slice_df[c] = None
        slice_df = slice_df[existing.columns]
        out_df = pd.concat([existing, slice_df], ignore_index=True)
        print(f"Appended {len(slice_df)} rows (total now {len(out_df)}).")
    else:
        out_df = slice_df
        print(f"Created file with {len(out_df)} rows.")

    out_df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(start_index=start, end_index=end)
