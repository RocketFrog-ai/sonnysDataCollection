"""
Fetch annual weather (31 Dec 2024 – 31 Dec 2025) for each address and add columns
in-place to Proforma-v2-weather.xlsx. Uses same logic as v1/weather/data (Open-Meteo).

Input/Output: datafetching/input_data/Proforma-v2-weather.xlsx (column: Address)

Usage (from project root):
  python datafetching/weather/fetch_weather.py [start_index] [end_index]
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Project root on path for app imports (weather -> datafetching -> project root)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.utils import common as calib
from app.server.app import get_climate

# Input/output: datafetching/input_data/Proforma-v2-weather.xlsx
_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "input_data" / "Proforma-v2-weather.xlsx"
ADDRESS_COL = "Address"

# Annual period: 31 Dec 2024 to 31 Dec 2025 (same as requested)
WEATHER_START_DATE = "2024-12-31"
WEATHER_END_DATE = "2025-12-31"

# Columns added (matches get_climate / get_climate_data_for_range from v1/weather/data)
WEATHER_COLUMNS = [
    "weather_total_precipitation_mm",
    "weather_rainy_days",
    "weather_total_snowfall_cm",
    "weather_days_below_freezing",
    "weather_total_sunshine_hours",
    "weather_days_pleasant_temp",
    "weather_avg_daily_max_windspeed_ms",
]


def _climate_to_row(climate: dict) -> dict:
    """Map get_climate() result to flat row dict (our column names)."""
    if not climate or climate.get("error"):
        return {c: None for c in WEATHER_COLUMNS}
    return {
        "weather_total_precipitation_mm": climate.get("total_precipitation_mm"),
        "weather_rainy_days": climate.get("rainy_days"),
        "weather_total_snowfall_cm": climate.get("total_snowfall_cm"),
        "weather_days_below_freezing": climate.get("days_below_freezing"),
        "weather_total_sunshine_hours": climate.get("total_sunshine_hours"),
        "weather_days_pleasant_temp": climate.get("days_pleasant_temp"),
        "weather_avg_daily_max_windspeed_ms": climate.get("avg_daily_max_windspeed_ms"),
    }


def run(start_index: int = 0, end_index: int | None = None) -> None:
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
    for i in tqdm(range(start_index, end_index), desc="Weather", unit="row"):
        row = df.iloc[i]
        address = row[ADDRESS_COL]
        if pd.isna(address) or not str(address).strip():
            new_row = row.to_dict()
            for c in WEATHER_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        geo = calib.get_lat_long(str(address))
        if not geo:
            new_row = row.to_dict()
            for c in WEATHER_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        lat, lon = geo["lat"], geo["lon"]
        try:
            climate = get_climate(
                float(lat), float(lon),
                start_date=WEATHER_START_DATE,
                end_date=WEATHER_END_DATE,
            )
        except Exception as e:
            new_row = row.to_dict()
            for c in WEATHER_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        new_row = row.to_dict()
        for k, v in _climate_to_row(climate).items():
            new_row[k] = v
        new_rows.append(new_row)

    slice_df = pd.DataFrame(new_rows)
    if len(slice_df) == 0:
        print("No rows to write.")
        return

    for c in WEATHER_COLUMNS:
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
