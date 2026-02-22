"""
Fetch nearby retailers for each address and add columns in-place to Proforma-v2-data_retailers.xlsx.

Uses:
  - Costco: app get_costco_info (searchText "Costco", name must contain "costco")
  - Walmart: app get_walmart_info (searchText "Walmart", name must contain "walmart")
  - Target: app get_target_info (searchText "Target", name must contain "target")
  - Other grocery & avg distance: v1/retailers logic (get_nearby_retailers, 1 mi), Grocery only, exclude Costco/Walmart/Target

Columns added:
  - distance_nearest_costco
  - distance_nearest_walmart
  - distance_nearest_target
  - other_grocery_count_1mile
  - avg_distance_grocery_miles

Input/Output: datafetching/data_retail/Proforma-v2-data_retailers.xlsx (column: Address)

Usage (from project root):
  python datafetching/retailers/fetch_retailers.py [start_index] [end_index]
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.utils import common as calib
from app.features.nearbyStores.nearby_costcos import get_costco_info
from app.features.nearbyStores.nearby_walmart import get_walmart_info
from app.features.nearbyStores.nearby_target import get_target_info
from app.features.nearbyRetailers.get_nearby_retailers import get_nearby_retailers

_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "data_retail" / "Proforma-v2-data_retailers.xlsx"
ADDRESS_COL = "Address"
RETAILER_RADIUS_1MI = 1.0

RETAILER_COLUMNS = [
    "distance_nearest_costco",
    "distance_nearest_walmart",
    "distance_nearest_target",
    "other_grocery_count_1mile",
    "avg_distance_grocery_miles",
]

CHAIN_NAMES = ("costco", "walmart", "target")


def _is_other_grocery(retailer: dict) -> bool:
    """True if category is Grocery and name is not a chain (Costco/Walmart/Target)."""
    cat = (retailer.get("category") or "").strip()
    if cat != "Grocery":
        return False
    name = (retailer.get("name") or "").lower()
    return not any(chain in name for chain in CHAIN_NAMES)


def _retailers_row(
    costco_info: dict | None,
    walmart_info: dict | None,
    target_info: dict | None,
    retailers_data: dict,
) -> dict:
    """Build one row of retailer columns."""
    row = {
        "distance_nearest_costco": None,
        "distance_nearest_walmart": None,
        "distance_nearest_target": None,
        "other_grocery_count_1mile": None,
        "avg_distance_grocery_miles": None,
    }
    if costco_info:
        row["distance_nearest_costco"] = costco_info.get("distance_from_nearest_costco")
    if walmart_info:
        row["distance_nearest_walmart"] = walmart_info.get("distance_from_nearest_walmart")
    if target_info:
        row["distance_nearest_target"] = target_info.get("distance_from_nearest_target")

    retailers = (retailers_data or {}).get("retailers") or []
    other_grocery = [r for r in retailers if _is_other_grocery(r)]
    if other_grocery:
        row["other_grocery_count_1mile"] = len(other_grocery)
        dists = [r["distance_miles"] for r in other_grocery if r.get("distance_miles") is not None]
        row["avg_distance_grocery_miles"] = round(sum(dists) / len(dists), 2) if dists else None
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
    for i in tqdm(range(start_index, end_index), desc="Retailers", unit="row"):
        row = df.iloc[i]
        address = row[ADDRESS_COL]
        if pd.isna(address) or not str(address).strip():
            new_row = row.to_dict()
            for c in RETAILER_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        geo = calib.get_lat_long(str(address))
        if not geo:
            new_row = row.to_dict()
            for c in RETAILER_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        lat, lon = float(geo["lat"]), float(geo["lon"])
        costco_info = None
        walmart_info = None
        target_info = None
        retailers_data = None
        try:
            costco_info = get_costco_info(lat, lon)
        except Exception:
            pass
        try:
            walmart_info = get_walmart_info(lat, lon)
        except Exception:
            pass
        try:
            target_info = get_target_info(lat, lon)
        except Exception:
            pass
        try:
            retailers_data = get_nearby_retailers(
                api_key, lat, lon,
                radius_miles=RETAILER_RADIUS_1MI,
                fetch_place_details=fetch_place_details,
            )
        except Exception:
            pass

        new_row = row.to_dict()
        for k, v in _retailers_row(costco_info, walmart_info, target_info, retailers_data).items():
            new_row[k] = v
        new_rows.append(new_row)

    slice_df = pd.DataFrame(new_rows)
    if len(slice_df) == 0:
        print("No rows to write.")
        return

    for c in RETAILER_COLUMNS:
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
