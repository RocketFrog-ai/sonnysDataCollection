"""
Site-factors lookup for a dropped pin — the council's per-site enrichment data
(experiments/council/data/Council--site-wise-data.csv: demographics, income bands, vehicle counts,
car-wash competitors, mass-merchant anchors, StreetLight traffic by daypart) served for the nearest
covered site to the pin.

Escalating match: the NEAREST row within 3 miles of the pin; if none, 6 miles; if none, 9 miles;
past that the pin has no data coverage. The CSV is a fixed research extract (one row per site the
council enriched), so a pin only "has" factors when it lands near one of those sites — hence the
honest not-found answer instead of interpolating.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from proforma.pnl.data import haversine_km

REPO_ROOT = Path(__file__).resolve().parents[3]
FACTORS_CSV = REPO_ROOT / "experiments" / "council" / "data" / "Council--site-wise-data.csv"

RADII_MILES = (3.0, 6.0, 9.0)   # the escalation ladder, in order
_MILE_KM = 1.609344

_CACHE: Dict[str, pd.DataFrame] = {}


def _load() -> pd.DataFrame:
    """The council site-wise extract, loaded once per process; rows without usable coords dropped."""
    if "df" not in _CACHE:
        df = pd.read_csv(FACTORS_CSV, low_memory=False)
        df = df[pd.to_numeric(df.lat, errors="coerce").notna()
                & pd.to_numeric(df.lon, errors="coerce").notna()].reset_index(drop=True)
        _CACHE["df"] = df
    return _CACHE["df"]


def _clean(v: Any) -> Any:
    """JSON-safe scalar: numpy → python, NaN/inf → None."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else None
    return v


def _bucket_factors(row: pd.Series) -> Dict[str, Any]:
    """The CSV row's 56 flat columns bucketed into labeled groups for the dashboard. The
    nearest/2nd/3rd competitor and mass-merchant column triplets become ranked lists; the two
    exact-duplicate income columns (`… .1`) are dropped."""
    g = lambda c: _clean(row[c])
    ordinals = ["Nearest", "2nd Nearest", "3rd Nearest"]

    competitors = []
    for i, o in enumerate(ordinals, start=1):
        name = g(f"{o} Car Wash Competitors-Name")
        if name is None:
            continue
        competitors.append({
            "rank": i, "name": name,
            "distance_miles": g(f"{o} Car Wash Competitors-Distance"),
            "competitor_type": g(f"{o} Car Wash Competitors-Competitor Type"),
            "car_wash_type": g(f"{o} Car Wash Competitors-Car Wash Type"),
            "website": g(f"{o} Car Wash Competitors-Website"),
        })

    mass_merchants = []
    for i, o in enumerate(ordinals, start=1):
        chain = g(f"{o} ChainXY VT - Mass Merchant-Chain Name")
        if chain is None:
            continue
        mass_merchants.append({
            "rank": i, "chain": chain,
            "distance_miles": g(f"{o} ChainXY VT - Mass Merchant-Distance"),
        })

    return {
        "Demographics": {
            "2025 Population Estimate": g("2025 Estimate"),
            "Growth 2025-2020": g("Growth 2025-2020"),
            "Growth 2030-2025": g("Growth 2030-2025"),
            "2025 Average Age": g("2025 Average Age"),
            "Labor Force": g("Labor Force"),
        },
        "Income": {
            "Average Household Income": g("Average Household Income"),
            "Median Household Income": g("Median Household Income"),
            "2025 % HH with Income $50K+": g("2025 % HH with Income $50K+"),
            "Households by Income Band": {
                b: g(b) for b in ["$100,000 to $124,999", "$125,000 to $149,999",
                                  "$150,000 to $174,999", "$175,000 to $199,999",
                                  "$200,000 to $249,999"]
            },
        },
        "Housing": {
            "Renter-Occupied": g("Renter-Occupied"),
            "Owner-Occupied Housing Units by Value":
                g("Current Year Estimated Owner-Occupied Housing Units by Value"),
        },
        "Vehicles": {
            "Households by Vehicle Count": {
                b: g(b) for b in ["1 vehicle", "2 vehicles", "3 vehicles",
                                  "4 vehicles", "5 or more vehicles"]
            },
            "Total Vehicles Available in the Market": g("Total Vehicles Available in the Market"),
            "Average Number of Vehicles Available": g("Average Number of Vehicles Available"),
        },
        "Car Wash Competitors": {
            "Count": g("Count of Car Wash Competitors"),
            "Nearest": competitors,
        },
        "Retail Anchors": {
            "Mass Merchant Count": g("Count of ChainXY VT - Mass Merchant"),
            "Nearest Mass Merchants": mass_merchants,
            "Grocery Count": g("Count of ChainXY VT - Grocery"),
            "Department Store Count": g("Count of ChainXY VT - Department Store"),
        },
        "Traffic (StreetLight)": {
            "Daypart Trips": {
                "Overnight": g("Nearest StreetLight US Hourly-ttl_overnight"),
                "Breakfast": g("Nearest StreetLight US Hourly-ttl_breakfast"),
                "Lunch": g("Nearest StreetLight US Hourly-ttl_lunch"),
                "Afternoon": g("Nearest StreetLight US Hourly-ttl_afternoon"),
                "Dinner": g("Nearest StreetLight US Hourly-ttl_dinner"),
                "Night": g("Nearest StreetLight US Hourly-ttl_night"),
            },
            "Highway Class": g("Nearest StreetLight US Hourly-Highway"),
        },
    }


def site_factors(lat: float, lon: float) -> Dict[str, Any]:
    """The site-factors row for a pin, matched at 3 → 6 → 9 miles (nearest row inside the first
    radius that holds any). Returns {found, radius_used_miles, match, factors} on a hit — `factors`
    bucketed into labeled groups (see _bucket_factors) — or {found: False, message: "Don't have
    data coverage"} when nothing lies within 9 miles."""
    df = _load()
    d_km = haversine_km(lat, lon, df.lat.values.astype(float), df.lon.values.astype(float))
    d_miles = d_km / _MILE_KM

    hit_idx: Optional[int] = None
    radius_used: Optional[float] = None
    for r in RADII_MILES:
        inside = np.where(d_miles <= r)[0]
        if len(inside):
            hit_idx = int(inside[np.argmin(d_miles[inside])])   # nearest row within this radius
            radius_used = r
            break

    out: Dict[str, Any] = {"lat": float(lat), "lon": float(lon),
                           "searched_radii_miles": list(RADII_MILES)}
    if hit_idx is None:
        out.update({"found": False, "message": "Don't have data coverage"})
        return out

    row = df.iloc[hit_idx]
    out.update({
        "found": True,
        "radius_used_miles": float(radius_used),
        "match": {
            "name": _clean(row["Name"]), "client_name": _clean(row["client_name"]),
            "client_id": _clean(row["client_id"]), "site_id": _clean(row["site_id"]),
            "lat": _clean(row["lat"]), "lon": _clean(row["lon"]),
            "dist_miles": round(float(d_miles[hit_idx]), 3),
        },
        "factors": _bucket_factors(row),
    })
    return out
