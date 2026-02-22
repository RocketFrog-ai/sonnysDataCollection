"""
Fetch nearby car wash competitors (4 miles, route distance) for each address and add
columns in-place to Proforma-v2-data_comp.xlsx. Uses v1/competitors/dynamics logic (car_wash only).

Input/Output: datafetching/data_retail/Proforma-v2-data_comp.xlsx (column: Address)

Columns added:
  - competitors_count_4miles
  - competitor_1_google_rating (nearest)
  - competitor_1_distance_miles (nearest)
  - competitor_1_google_user_rating_count (nearest)

Usage (from project root):
  python datafetching/competition/fetch_competition.py [start_index] [end_index]
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Project root on path for app imports (competition -> datafetching -> project root)
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.utils import common as calib
from app.features.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors

# Input/output: datafetching/data_retail/Proforma-v2-data_comp.xlsx
_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "data_retail" / "Proforma-v2-data_comp.xlsx"
ADDRESS_COL = "Address"
RADIUS_MILES = 4.0

COMPETITION_COLUMNS = [
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_google_user_rating_count",
]


def _dynamics_to_row(data: dict) -> dict:
    """Map get_nearby_competitors() result to flat row (nearest + count)."""
    count = data.get("count") or 0
    competitors = data.get("competitors") or []
    nearest = competitors[0] if competitors else None
    return {
        "competitors_count_4miles": count,
        "competitor_1_google_rating": nearest.get("rating") if nearest else None,
        "competitor_1_distance_miles": nearest.get("distance_miles") if nearest else None,
        "competitor_1_google_user_rating_count": nearest.get("user_rating_count") if nearest else None,
    }


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
    for i in tqdm(range(start_index, end_index), desc="Competition", unit="row"):
        row = df.iloc[i]
        address = row[ADDRESS_COL]
        if pd.isna(address) or not str(address).strip():
            new_row = row.to_dict()
            for c in COMPETITION_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        geo = calib.get_lat_long(str(address))
        if not geo:
            new_row = row.to_dict()
            for c in COMPETITION_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        lat, lon = geo["lat"], geo["lon"]
        try:
            data = get_nearby_competitors(
                api_key, float(lat), float(lon),
                radius_miles=RADIUS_MILES,
                fetch_place_details=fetch_place_details,
            )
        except Exception as e:
            new_row = row.to_dict()
            for c in COMPETITION_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        new_row = row.to_dict()
        for k, v in _dynamics_to_row(data).items():
            new_row[k] = v
        new_rows.append(new_row)

    slice_df = pd.DataFrame(new_rows)
    if len(slice_df) == 0:
        print("No rows to write.")
        return

    for c in COMPETITION_COLUMNS:
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
