"""
Pin demographics from the council site-wise extract — the SAME source the Site-factors tab uses
(experiments/council/data/Council--site-wise-data.csv, one row per council-enriched site).

Matching mirrors app/pnl_analysis/modelling/site_factors.py: the nearest covered site within
3 miles of the pin; if none, 6 miles; if none, 9 miles; past that the pin honestly has no
coverage (the CSV is a fixed research extract — we never interpolate).

Shared by BOTH the Streamlit 🌐 True-market view and the /market-forecast API, so the population,
projected growth and vehicle base behind the demographic overlay agree everywhere. Streamlit-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from proforma.pnl.data import haversine_km

_REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORS_CSV = _REPO_ROOT / "experiments" / "council" / "data" / "Council--site-wise-data.csv"

RADII_MILES = (3.0, 6.0, 9.0)   # the escalation ladder, in order (matches site_factors)
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


def _num(row: pd.Series, col: str):
    if col not in row or pd.isna(row[col]):
        return None
    try:
        v = float(row[col])
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def pin_demographics(lat: float, lon: float) -> Dict[str, Any]:
    """Demographics for a pin from the nearest council-enriched site, matched at 3 → 6 → 9 miles.

    Returns {found: False, message} when no covered site lies within 9 miles, else:
      found                True
      population           2025 population estimate for the matched site's trade area
      growth_2030_2025     projected '25→'30 population growth (fraction, e.g. 0.059 = +5.9%)
      vehicles             total vehicles available in the market
      radius_used_miles    the escalation ring that produced the match (3 / 6 / 9)
      match                {name, client_name, dist_miles} of the council site whose record is used
    """
    df = _load()
    d_miles = haversine_km(lat, lon, df.lat.values.astype(float), df.lon.values.astype(float)) / _MILE_KM

    hit_idx = None
    radius_used = None
    for r in RADII_MILES:
        inside = np.where(d_miles <= r)[0]
        if len(inside):
            hit_idx = int(inside[np.argmin(d_miles[inside])])   # nearest row within this radius
            radius_used = r
            break

    if hit_idx is None:
        return {"found": False, "message": "Don't have data coverage",
                "searched_radii_miles": list(RADII_MILES)}

    row = df.iloc[hit_idx]
    return {
        "found": True,
        "population": _num(row, "2025 Estimate"),
        "growth_2030_2025": _num(row, "Growth 2030-2025"),
        "vehicles": _num(row, "Total Vehicles Available in the Market"),
        "radius_used_miles": float(radius_used),
        "match": {"name": None if pd.isna(row.get("Name")) else str(row["Name"]),
                  "client_name": None if pd.isna(row.get("client_name")) else str(row["client_name"]),
                  "dist_miles": round(float(d_miles[hit_idx]), 3)},
    }


def annual_pop_growth(growth_2030_2025) -> float:
    """The '25→'30 projected growth (a 5-year fraction) as a compound annual rate; 0.0 when unknown."""
    if growth_2030_2025 is None or not np.isfinite(growth_2030_2025):
        return 0.0
    return float((1.0 + float(growth_2030_2025)) ** (1.0 / 5.0) - 1.0)
