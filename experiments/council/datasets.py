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


_FT_PER_M = 3.28084


def capex_for_pin(lat: float, lon: float, *, k: int = 8, tunnel_ft: Optional[float] = None) -> Dict[str, Any]:
    """CAPEX estimate = median `project_cost_total_investment` of comparable historical builds (positive cost).

    **Demand-driven when `tunnel_ft` is given** (from the Capacity seat, whose tunnel length scales with the
    projected peak volume): match the builds by TUNNEL LENGTH — a bigger tunnel = a bigger build = more CAPEX —
    since tunnel size is the dominant CAPEX driver and the old-proforma tunnel lengths are in metres. Without a
    tunnel length it falls back to the nearest builds by location. Returns {capex, capex_low, capex_high, basis, _meta}."""
    df = _load_proforma()
    if df.empty or _CAPEX_COL not in df.columns:
        return {"capex": None, "_meta": {"n": 0}}
    valid = df[df[_CAPEX_COL] > 0].copy()
    if valid.empty:
        return {"capex": None, "_meta": {"n": 0}}

    if tunnel_ft is not None and "tunnel_length_actual" in valid.columns:
        tunnel_m = float(tunnel_ft) / _FT_PER_M                      # old-proforma tunnel lengths are metres
        vt = valid[valid.tunnel_length_actual.notna() & (valid.tunnel_length_actual > 0)].copy()
        if len(vt):
            vt["_tdiff"] = (vt.tunnel_length_actual.astype(float) - tunnel_m).abs()
            near = vt.sort_values("_tdiff").head(k)
            costs = near[_CAPEX_COL].astype(float)
            return {"capex": float(costs.median()), "capex_low": float(costs.min()), "capex_high": float(costs.max()),
                    "basis": f"scaled to a ~{tunnel_ft:.0f} ft (~{tunnel_m:.0f} m) tunnel — {len(near)} similar-size builds",
                    "_meta": {"n": int(len(near)), "mode": "tunnel", "tunnel_m": round(tunnel_m, 1)}}

    d = haversine_km(lat, lon, valid.lat.values, valid.lon.values)
    valid = valid.assign(_dist_km=d).sort_values("_dist_km").head(k)
    costs = valid[_CAPEX_COL].astype(float)
    nearest = valid.iloc[0]
    return {
        "capex": float(costs.median()), "capex_low": float(costs.min()), "capex_high": float(costs.max()),
        "tunnel_actual": _f(nearest.get("tunnel_length_actual")), "tunnel_predicted": _f(nearest.get("tunnel_length_predicted")),
        "basis": f"nearest {len(valid)} builds by location",
        "_meta": {"n": int(len(valid)), "mode": "geo",
                  "nearest_name": str(nearest.get("company_name") or nearest.get("address1") or ""),
                  "nearest_km": round(float(nearest._dist_km), 2)},
    }
