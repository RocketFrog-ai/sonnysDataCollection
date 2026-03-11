import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.utils import common as calib
from app.features.nearbyStores.nearby_costcos import find_nearby_places as _find_places
from app.features.nearbyStores.nearby_target import _find_nearby_places_text
from app.features.nearbyRetailers.get_nearby_retailers import get_nearby_retailers
from app.features.experimental_features.operationalHours.searchNearby import find_nearby_places as _search_nearby

_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "input_data" / "Proforma-v2-data.xlsx"
GOOGLE_MAPS_API_KEY = "AIzaSyB02AEDngo98265qF9YXiWR-372He5RBRg"
ADDRESS_COL = "Address"
RETAILER_RADIUS_1MI = 1.0
FOOD_JOINTS_RADIUS = 0.5

RETAILER_COLUMNS = [
    "distance_nearest_costco",
    "distance_nearest_walmart",
    "distance_nearest_target",
    "other_grocery_count_1mile",
    "count_food_joints_0_5miles",
]

CHAIN_NAMES = ("costco", "walmart", "target")


def _get_chain_distance(api_key: str, lat: float, lon: float, keyword: str, name_contains: str) -> float | None:
    data = _find_places(api_key, lat, lon, radius_miles=5, keyword=keyword, max_results=20)
    if not data or "places" not in data:
        return None
    places = [p for p in data["places"] if name_contains in (p.get("displayName") or {}).get("text", "").lower()]
    if not places:
        return None
    best = float("inf")
    for p in places:
        loc = p.get("location", {})
        plat, plon = loc.get("latitude"), loc.get("longitude")
        if plat is not None and plon is not None:
            d = calib.calculate_distance(lat, lon, plat, plon)
            if d < best:
                best = d
    return round(best, 2) if best != float("inf") else None


def _get_target_distance(api_key: str, lat: float, lon: float) -> float | None:
    data = _find_nearby_places_text(api_key, lat, lon, radius_miles=5, keyword="Target", max_results=20)
    if not data or "places" not in data:
        return None
    places = [p for p in data["places"] if "target" in (p.get("displayName") or {}).get("text", "").lower()]
    if not places:
        return None
    best = float("inf")
    for p in places:
        loc = p.get("location", {})
        plat, plon = loc.get("latitude"), loc.get("longitude")
        if plat is not None and plon is not None:
            d = calib.calculate_distance(lat, lon, plat, plon)
            if d < best:
                best = d
    return round(best, 2) if best != float("inf") else None


def _count_food_joints(api_key: str, lat: float, lon: float) -> int:
    data = _search_nearby(api_key, lat, lon, radius_miles=FOOD_JOINTS_RADIUS,
                          included_types=["fast_food_restaurant", "cafe"], max_results=20, rank_preference="DISTANCE")
    if not data or "places" not in data:
        return 0
    count = 0
    for p in data.get("places", []):
        loc = p.get("location", {})
        plat, plon = loc.get("latitude"), loc.get("longitude")
        if plat is not None and plon is not None:
            d = calib.calculate_distance(lat, lon, plat, plon)
            if d <= FOOD_JOINTS_RADIUS:
                count += 1
    return count


def _is_other_grocery(retailer: dict) -> bool:
    cat = (retailer.get("category") or "").strip()
    if cat != "Grocery":
        return False
    name = (retailer.get("name") or "").lower()
    return not any(chain in name for chain in CHAIN_NAMES)


def _retailers_row(
    dist_costco: float | None,
    dist_walmart: float | None,
    dist_target: float | None,
    retailers_data: dict,
    food_joints_count: int,
) -> dict:
    row = {
        "distance_nearest_costco": dist_costco,
        "distance_nearest_walmart": dist_walmart,
        "distance_nearest_target": dist_target,
        "other_grocery_count_1mile": None,
        "count_food_joints_0_5miles": food_joints_count,
    }
    retailers = (retailers_data or {}).get("retailers") or []
    other_grocery = [r for r in retailers if _is_other_grocery(r)]
    if other_grocery:
        row["other_grocery_count_1mile"] = len(other_grocery)
    return row


def run(start_index: int = 0, end_index: int | None = None, api_key: str | None = None) -> None:
    api_key = api_key or GOOGLE_MAPS_API_KEY

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

    for c in RETAILER_COLUMNS:
        if c not in df.columns:
            df[c] = None

    SAVE_EVERY = 10

    for idx, i in enumerate(tqdm(range(start_index, end_index), desc="Retailers", unit="row")):
        row = df.iloc[i]
        address = row[ADDRESS_COL]
        if pd.isna(address) or not str(address).strip():
            for c in RETAILER_COLUMNS:
                df.at[df.index[i], c] = None
        else:
            geo = calib.get_lat_long(str(address))
            if not geo:
                for c in RETAILER_COLUMNS:
                    df.at[df.index[i], c] = None
            else:
                lat, lon = float(geo["lat"]), float(geo["lon"])
                dist_costco = None
                dist_walmart = None
                dist_target = None
                retailers_data = None
                food_count = 0
                try:
                    dist_costco = _get_chain_distance(api_key, lat, lon, "Costco", "costco")
                except Exception:
                    pass
                try:
                    dist_walmart = _get_chain_distance(api_key, lat, lon, "Walmart", "walmart")
                except Exception:
                    pass
                try:
                    dist_target = _get_target_distance(api_key, lat, lon)
                except Exception:
                    pass
                try:
                    retailers_data = get_nearby_retailers(api_key, lat, lon, radius_miles=RETAILER_RADIUS_1MI, fetch_place_details=False)
                except Exception:
                    pass
                try:
                    food_count = _count_food_joints(api_key, lat, lon)
                except Exception:
                    pass

                vals = _retailers_row(dist_costco, dist_walmart, dist_target, retailers_data, food_count)
                for c in RETAILER_COLUMNS:
                    df.at[df.index[i], c] = vals.get(c)

        if (idx + 1) % SAVE_EVERY == 0 or i == end_index - 1:
            df.to_excel(INPUT_PATH, index=False, engine="openpyxl")

    print(f"Updated rows {start_index}–{end_index - 1} in {INPUT_PATH}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run(start_index=start, end_index=end)
