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

_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "input_data" / "comp2.xlsx"
MIN_DISTANCE_THRESHOLD = 0.08
ADDRESS_COL = "Address"
RADIUS_MILES = 4.0

COMPETITION_COLUMNS = [
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_google_user_rating_count",
]


def _dynamics_to_row(data: dict) -> dict:
    competitors = data.get("competitors") or []
    filtered = [c for c in competitors if (c.get("distance_miles") or 0) >= MIN_DISTANCE_THRESHOLD]
    count = len(filtered)
    nearest = filtered[0] if filtered else None
    return {
        "competitors_count_4miles": count,
        "competitor_1_google_rating": nearest.get("rating") if nearest else None,
        "competitor_1_distance_miles": nearest.get("distance_miles") if nearest else None,
        "competitor_1_google_user_rating_count": nearest.get("user_rating_count") if nearest else None,
    }


def run(start_index: int = 0, end_index: int | None = None, fetch_place_details: bool = False, api_key: str | None = None) -> None:
    api_key = api_key or calib.GOOGLE_MAPS_API_KEY
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

    for c in COMPETITION_COLUMNS:
        if c not in df.columns:
            df[c] = None

    SAVE_EVERY = 10

    for idx, i in enumerate(tqdm(range(start_index, end_index), desc="Competition", unit="row")):
        row = df.iloc[i]
        address = row[ADDRESS_COL]
        if pd.isna(address) or not str(address).strip():
            for c in COMPETITION_COLUMNS:
                df.at[df.index[i], c] = None
        else:
            geo = calib.get_lat_long(str(address))
            if not geo:
                for c in COMPETITION_COLUMNS:
                    df.at[df.index[i], c] = None
            else:
                lat, lon = geo["lat"], geo["lon"]
                try:
                    data = get_nearby_competitors(
                        api_key, float(lat), float(lon),
                        radius_miles=RADIUS_MILES,
                        fetch_place_details=fetch_place_details,
                        max_results=15,
                    )
                    vals = _dynamics_to_row(data)
                    for c in COMPETITION_COLUMNS:
                        df.at[df.index[i], c] = vals.get(c)
                except Exception:
                    for c in COMPETITION_COLUMNS:
                        df.at[df.index[i], c] = None

        if (idx + 1) % SAVE_EVERY == 0 or i == end_index - 1:
            df.to_excel(INPUT_PATH, index=False, engine="openpyxl")

    print(f"Updated rows {start_index}–{end_index - 1} in {INPUT_PATH}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(start_index=start, end_index=end, api_key="AIzaSyB02AEDngo98265qF9YXiWR-372He5RBRg")
