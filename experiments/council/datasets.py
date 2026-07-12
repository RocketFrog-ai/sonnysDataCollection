"""
Loaders for the council's two enrichment CSVs (council folder only).

`sitewise_for_pin(lat, lon)` — snap the pin to the nearest known sites in `Council--site-wise-data.csv`
and borrow/average their **demographic / income / vehicle** columns (the competitor columns are skipped —
Competition uses live Google places). `capex_for_pin(lat, lon)` — the nearest historical builds in
`Council--old-proforma-data.csv` give a **CAPEX** estimate (median of the nearest builds) + their tunnel
lengths, keyed by lat/lon. Both cached per process. Self-contained: uses only `data_1_6.haversine_km`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.council.data_1_6 import DATA_DIR, haversine_km

SITEWISE_CSV = DATA_DIR / "Council--site-wise-data.csv"
PROFORMA_CSV = DATA_DIR / "Council--old-proforma-data.csv"

_CACHE: Dict[str, Any] = {}

# demographic / income / vehicle reference columns to borrow (NO competitor columns)
_SITEWISE_FIELDS = {
    "population_2025": "2025 Estimate",
    "growth_2020_2025": "Growth 2025-2020",
    "growth_2025_2030": "Growth 2030-2025",
    "avg_age": "2025 Average Age",
    "labor_force": "Labor Force",
    "avg_household_income": "Average Household Income",
    "median_household_income": "Median Household Income",
    "avg_vehicles": "Average Number of Vehicles Available",
    "total_vehicles": "Total Vehicles Available in the Market",
    "pct_hh_income_50k_plus": "2025 % HH with Income $50K+",
    "mass_merchant_count": "Count of ChainXY VT - Mass Merchant",
    "grocery_count": "Count of ChainXY VT - Grocery",
}

_CAPEX_COL = "project_cost_total_investment[car_wash_acquisition_budget]"


def _load_sitewise() -> pd.DataFrame:
    if "sitewise" not in _CACHE:
        df = pd.read_csv(SITEWISE_CSV, low_memory=False)
        for col in ("lat", "lon"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        _CACHE["sitewise"] = df[df.lat.notna() & df.lon.notna()].reset_index(drop=True)
    return _CACHE["sitewise"]


def _load_proforma() -> pd.DataFrame:
    if "proforma" not in _CACHE:
        df = pd.read_csv(PROFORMA_CSV, low_memory=False)
        for col in ("lat", "lon", _CAPEX_COL, "tunnel_length_actual", "tunnel_length_predicted"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        _CACHE["proforma"] = df[df.lat.notna() & df.lon.notna()].reset_index(drop=True)
    return _CACHE["proforma"]


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def sitewise_for_pin(lat: float, lon: float, *, k: int = 3, radius_km: float = 25.0) -> Dict[str, Any]:
    """Demographic/income/vehicle profile for the pin = mean of the nearest ≤k known sites within
    `radius_km` (falls back to the single nearest if none are inside the radius). Competitor columns
    are deliberately excluded. Returns {fields…, _meta:{n, nearest_name, nearest_km}}."""
    df = _load_sitewise()
    if df.empty:
        return {"_meta": {"n": 0, "nearest_name": None, "nearest_km": None}}
    d = haversine_km(lat, lon, df.lat.values, df.lon.values)
    df = df.assign(_dist_km=d).sort_values("_dist_km")
    near = df[df._dist_km <= radius_km].head(k)
    if near.empty:
        near = df.head(1)
    out: Dict[str, Any] = {}
    for key, col in _SITEWISE_FIELDS.items():
        if col in near.columns:
            vals = pd.to_numeric(near[col], errors="coerce").dropna()
            out[key] = float(vals.mean()) if len(vals) else None
        else:
            out[key] = None
    row0 = near.iloc[0]
    out["_meta"] = {"n": int(len(near)),
                    "nearest_name": str(row0.get("client_name") or row0.get("Name") or ""),
                    "nearest_km": round(float(row0._dist_km), 2)}
    return out


def capex_for_pin(lat: float, lon: float, *, k: int = 5) -> Dict[str, Any]:
    """CAPEX estimate = median `project_cost_total_investment` of the nearest ≤k historical builds with a
    positive cost, plus the nearest build's tunnel lengths and meta. Builds are sparse (~619 nationwide),
    so this is nearest-k by distance (no hard radius). Returns {capex, capex_low, capex_high, tunnel_actual,
    tunnel_predicted, _meta}."""
    df = _load_proforma()
    if df.empty or _CAPEX_COL not in df.columns:
        return {"capex": None, "_meta": {"n": 0}}
    d = haversine_km(lat, lon, df.lat.values, df.lon.values)
    df = df.assign(_dist_km=d).sort_values("_dist_km")
    valid = df[df[_CAPEX_COL] > 0].head(k)
    if valid.empty:
        return {"capex": None, "_meta": {"n": 0}}
    costs = valid[_CAPEX_COL].astype(float)
    nearest = valid.iloc[0]
    return {
        "capex": float(costs.median()),
        "capex_low": float(costs.min()),
        "capex_high": float(costs.max()),
        "tunnel_actual": _f(nearest.get("tunnel_length_actual")),
        "tunnel_predicted": _f(nearest.get("tunnel_length_predicted")),
        "_meta": {"n": int(len(valid)),
                  "nearest_name": str(nearest.get("company_name") or nearest.get("address1") or ""),
                  "nearest_km": round(float(nearest._dist_km), 2)},
    }
