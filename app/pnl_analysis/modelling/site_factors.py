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

# join/merge artifacts in the CSV that carry no information for an API consumer
_DROP_COLS = ["Latitude", "Longitude", "_Match", "__longitude", "__latitude", "__name", "client_id_1"]
# identity fields presented in the `match` block rather than under `factors`
_ID_COLS = ["Name", "client_name", "client_id", "site_id", "lat", "lon"]

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


def site_factors(lat: float, lon: float) -> Dict[str, Any]:
    """The site-factors row for a pin, matched at 3 → 6 → 9 miles (nearest row inside the first
    radius that holds any). Returns {found, radius_used_miles, match, factors} on a hit, or
    {found: False, message: "Don't have data coverage"} when nothing lies within 9 miles."""
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
    factors = {c: _clean(row[c]) for c in df.columns if c not in _DROP_COLS + _ID_COLS}
    out.update({
        "found": True,
        "radius_used_miles": float(radius_used),
        "match": {
            "name": _clean(row["Name"]), "client_name": _clean(row["client_name"]),
            "client_id": _clean(row["client_id"]), "site_id": _clean(row["site_id"]),
            "lat": _clean(row["lat"]), "lon": _clean(row["lon"]),
            "dist_miles": round(float(d_miles[hit_idx]), 3),
        },
        "factors": factors,
    })
    return out
