"""
Nearest-site feature lookup over the precomputed `merged_all_sites.csv`.

Given a (lat, lon) pin we find the single closest site in the dataset (haversine) and return its
precomputed features, grouped by theme (demographics / income / vehicles / housing / mass-merchants /
retail / traffic). This is the lightweight, offline counterpart to the live Google-Places/weather
fetch pipeline — one CSV row, no external calls.

The car-wash-competitor columns are intentionally excluded from the response.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# <repo>/app/site_analysis/server/site_features.py -> parents[3] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CSV = _REPO_ROOT / "earnest-proforma-2.0" / "data" / "merged_all_sites.csv"
CSV_PATH = Path(os.environ.get("MERGED_SITES_CSV", str(_DEFAULT_CSV)))

# Site lat/lon live in the lowercase `lat`/`lon` columns (the capitalized `Latitude`/`Longitude`
# and the `__latitude`/`__longitude` scaffolding columns are duplicates).
_LAT_COL, _LON_COL = "lat", "lon"

# Identity columns echoed back so the caller knows which dataset row matched.
_IDENTITY = ["Name", "client_name", "client_id", "site_id"]

# Themed grouping of the feature columns. Competitor ("Car Wash Competitors") columns are omitted
# on purpose. Duplicate income columns (the `.1` variants) and lat/lon scaffolding are also dropped.
_GROUPS: Dict[str, list[str]] = {
    "demographics": [
        "2025 Estimate",
        "Growth 2025-2020",
        "Growth 2030-2025",
        "2025 Average Age",
        "Labor Force",
        "Renter-Occupied",
    ],
    "income": [
        "Average Household Income",
        "Median Household Income",
        "2025 % HH with Income $50K+",
        "$100,000 to $124,999",
        "$125,000 to $149,999",
        "$150,000 to $174,999",
        "$175,000 to $199,999",
        "$200,000 to $249,999",
    ],
    "vehicles": [
        "1 vehicle",
        "2 vehicles",
        "3 vehicles",
        "4 vehicles",
        "5 or more vehicles",
        "Total Vehicles Available in the Market",
        "Average Number of Vehicles Available",
    ],
    "housing": [
        "Current Year Estimated Owner-Occupied Housing Units by Value",
    ],
    "mass_merchants": [
        "Count of ChainXY VT - Mass Merchant",
        "Nearest ChainXY VT - Mass Merchant-Chain Name",
        "Nearest ChainXY VT - Mass Merchant-Distance",
        "2nd Nearest ChainXY VT - Mass Merchant-Chain Name",
        "2nd Nearest ChainXY VT - Mass Merchant-Distance",
        "3rd Nearest ChainXY VT - Mass Merchant-Chain Name",
        "3rd Nearest ChainXY VT - Mass Merchant-Distance",
    ],
    "retail": [
        "Count of ChainXY VT - Grocery",
        "Count of ChainXY VT - Department Store",
    ],
    "traffic": [
        "Nearest StreetLight US Hourly-ttl_overnight",
        "Nearest StreetLight US Hourly-ttl_breakfast",
        "Nearest StreetLight US Hourly-ttl_lunch",
        "Nearest StreetLight US Hourly-ttl_afternoon",
        "Nearest StreetLight US Hourly-ttl_dinner",
        "Nearest StreetLight US Hourly-ttl_night",
        "Nearest StreetLight US Hourly-Highway",
    ],
}

_EARTH_RADIUS_MILES = 3958.7613


def _haversine_miles(lat1: float, lon1: float, lat2, lon2):
    """Great-circle distance in miles. lat2/lon2 may be array-likes (vectorized over the dataset)."""
    rlat1, rlon1 = math.radians(lat1), math.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + math.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_MILES * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


_df_cache: Optional[pd.DataFrame] = None


def _load_df() -> pd.DataFrame:
    """Load (and memoize) the sites dataset, dropping rows without coordinates."""
    global _df_cache
    if _df_cache is None:
        df = pd.read_csv(CSV_PATH)
        df = df.dropna(subset=[_LAT_COL, _LON_COL]).reset_index(drop=True)
        _df_cache = df
    return _df_cache


def _clean(value: Any) -> Any:
    """NaN -> None; numpy scalars -> native python."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def nearest_site_features(lat: float, lon: float) -> Dict[str, Any]:
    """Find the single closest dataset site to (lat, lon) and return its grouped features.

    Returns: {query, matched_site:{...,distance_miles}, features:{<group>:{col:value}}}.
    """
    df = _load_df()
    if df.empty:
        raise ValueError("No sites with coordinates available in the dataset.")

    distances = _haversine_miles(lat, lon, df[_LAT_COL].to_numpy(), df[_LON_COL].to_numpy())
    idx = int(np.argmin(distances))
    row = df.iloc[idx]

    matched = {key: _clean(row.get(key)) for key in _IDENTITY}
    matched["lat"] = _clean(row.get(_LAT_COL))
    matched["lon"] = _clean(row.get(_LON_COL))
    matched["distance_miles"] = round(float(distances[idx]), 4)

    features: Dict[str, Dict[str, Any]] = {}
    for group, cols in _GROUPS.items():
        features[group] = {col: _clean(row.get(col)) for col in cols if col in df.columns}

    return {
        "query": {"latitude": lat, "longitude": lon},
        "matched_site": matched,
        "features": features,
    }
