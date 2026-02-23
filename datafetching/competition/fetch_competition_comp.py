import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from app.utils import common as calib
from app.features.nearbyCompetitors.get_nearby_competitors import (
    get_nearby_competitors,
)

_DATA_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = _DATA_DIR / "input_data" / "comp2.xlsx"
ADDRESS_COL = "Address"
RADIUS_MILES = 4.0

COMPETITION_COLUMNS = [
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_google_user_rating_count",
]

SELF_MATCH_DISTANCE_THRESHOLD = 0.1
SELF_MATCH_NAME_SIMILARITY_THRESHOLD = 0.7


def _normalize(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _address_overlap(input_addr: str, comp_addr: str) -> bool:
    if not input_addr or not comp_addr:
        return False
    zi = re.search(r"\d{5}(?:-\d{4})?", str(input_addr))
    zc = re.search(r"\d{5}(?:-\d{4})?", str(comp_addr))
    if zi and zc and zi.group() == zc.group():
        return True
    ni, nc = _normalize(input_addr), _normalize(comp_addr)
    ti, tc = set(ni.split()), set(nc.split())
    overlap = len(ti & tc) / max(len(ti), 1)
    return overlap >= 0.5


def _is_self_match(input_address_str: str, competitor: dict) -> bool:
    dist = competitor.get("distance_miles")
    if dist is None or dist >= SELF_MATCH_DISTANCE_THRESHOLD:
        return False
    parts = str(input_address_str or "").split(",", 1)
    input_name = (parts[0] or "").strip()
    input_addr = (parts[1] or "").strip() if len(parts) > 1 else ""
    comp_name = str(competitor.get("name") or "")
    comp_addr = str(competitor.get("address") or "")
    name_sim = _name_similarity(input_name, comp_name)
    addr_ov = _address_overlap(input_addr, comp_addr)
    return name_sim >= SELF_MATCH_NAME_SIMILARITY_THRESHOLD and addr_ov


def _to_row(data: dict) -> dict:
    count = data.get("count") or 0
    competitors = data.get("competitors") or []
    nearest = competitors[0] if competitors else None
    return {
        "competitors_count_4miles": count,
        "competitor_1_google_rating": nearest.get("rating") if nearest else None,
        "competitor_1_distance_miles": nearest.get("distance_miles") if nearest else None,
        "competitor_1_google_user_rating_count": nearest.get("user_rating_count") if nearest else None,
    }


def run(start_index: int = 0, end_index: int | None = None) -> None:
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
                fetch_place_details=False,
            )
        except Exception:
            new_row = row.to_dict()
            for c in COMPETITION_COLUMNS:
                new_row[c] = None
            new_rows.append(new_row)
            continue

        competitors = [c for c in (data.get("competitors") or []) if not _is_self_match(str(address), c)]
        data = {"competitors": competitors, "count": len(competitors)}

        new_row = row.to_dict()
        for k, v in _to_row(data).items():
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


def print_cost_estimate():
    df = pd.read_excel(INPUT_PATH, engine="openpyxl")
    n = len(df[df[ADDRESS_COL].notna() & (df[ADDRESS_COL].astype(str).str.strip() != "")])
    nearby_req = n
    dm_elements_low = n * 5
    dm_elements_high = n * 20
    print("=" * 60)
    print(f"GOOGLE API COST ESTIMATE ({INPUT_PATH.name})")
    print("=" * 60)
    print(f"Addresses to process: {n}")
    print()
    print("APIs used:")
    print("  - Geocoding: TomTom (no Google cost)")
    print("  - Nearby Search (Places API New): 1 req/address")
    print("  - Distance Matrix (Legacy): 1 origin x N dests per address")
    print("  - Place Details: SKIPPED (fetch_place_details=False)")
    print()
    print("Pricing (USD per 1000, before $200/mo credit):")
    print("  - Nearby Search Enterprise (rating/userRatingCount): $35 (first 1k free)")
    print("  - Distance Matrix: $5 (first 10k elements free)")
    print()
    nearby_bill = max(0, nearby_req - 1000)
    dm_bill_low = max(0, dm_elements_low - 10000)
    dm_bill_high = max(0, dm_elements_high - 10000)
    cost_nearby = (nearby_bill / 1000) * 35
    cost_dm_low = (dm_bill_low / 1000) * 5
    cost_dm_high = 0 if dm_bill_high == 0 else (dm_bill_high / 1000) * 5
    total_low = cost_nearby + cost_dm_low
    total_high = cost_nearby + cost_dm_high
    print("Estimated cost (pre-credit):")
    print(f"  Nearby Search: {nearby_req} req -> ${cost_nearby:.2f}")
    print(f"  Distance Matrix: {dm_elements_low}-{dm_elements_high} elements -> ${cost_dm_low:.2f}-${cost_dm_high:.2f}")
    print(f"  Total: ${total_low:.2f} - ${total_high:.2f}")
    print()
    print("With $200/month Google credit: likely $0")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cost":
        print_cost_estimate()
        sys.exit(0)
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print_cost_estimate()
    run(start_index=start, end_index=end)
