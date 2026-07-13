"""
Council-local forward forecast — pure panel math, self-contained (no coldstart / no `proforma` import).

`project_site(lat, lon)` builds a 5-year monthly washcount + revenue trajectory for a hypothetical new
build from its **12-mile neighbour cluster** in the historical panel:
  • mature anchor  = median mature level of qualified neighbours (≥30 mo history, ≥24 mo to trust, COVID-2020
                     cohort dropped, non-left-censored) — the level a mature site in this market holds;
  • ramp curve     = median of how those same donors actually grew over their own months 0→59
                     (each donor normalised by its own mature level); synthetic S-curve fallback if thin;
  • pricing        = cluster ASP (mem/ret) × the membership split → monthly revenue.
Returns the trajectory + headline scalars (anchor, peak-month washes, 5-yr wash & revenue, ramp-to-90%).

This replaces the trained coldstart forecast, which lives in `proforma/` and needs its 46 MB artifact —
out of bounds under "council folder only". Grounded entirely in `Council--historical-data.csv`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.council import config as C
from experiments.council import data_1_6 as D

HORIZON = 60


def _qualified_donors(neighbours: pd.DataFrame) -> pd.DataFrame:
    """Reliable comparables / ramp donors: not left-censored, opened outside the COVID-drop years, with
    ≥MIN_HISTORY_MONTHS records and a trustworthy mature level (≥MODEL_MIN_MONTHS, mat_n≥4 ⇒ mature_level notna)."""
    if neighbours.empty:
        return neighbours
    yr = neighbours.op_start.dt.year
    q = neighbours[
        (~neighbours.left_censored)
        & (~yr.isin(C.DROP_OPEN_YEARS))
        & (neighbours.n_obs >= C.MIN_HISTORY_MONTHS)
        & (neighbours.n_obs >= C.MODEL_MIN_MONTHS)
        & (neighbours.mature_level.notna())
    ].copy()
    return q


def _learned_ramp(df: pd.DataFrame, donor_keys: List[str], mature: pd.Series) -> Optional[np.ndarray]:
    """Median normalised ramp[0..HORIZON-1] across donors = tot_wash(rel) / donor's own mature level."""
    if not donor_keys:
        return None
    sub = df[df.site_key.isin(donor_keys) & (df.rel >= 0) & (df.rel < HORIZON)].copy()
    if sub.empty:
        return None
    sub["norm"] = sub.tot_wash_count / sub.site_key.map(mature)
    piv = sub.pivot_table(index="rel", columns="site_key", values="norm", aggfunc="mean")
    med = piv.median(axis=1)                                   # median across donors at each age
    arr = pd.Series(index=range(HORIZON), dtype=float)
    arr.loc[med.index] = med.values
    arr = arr.interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).mean()
    if not np.isfinite(arr.values).any():
        return None
    return np.clip(arr.to_numpy(), 0.0, 1.5)


def _synthetic_ramp() -> np.ndarray:
    """Fallback S-curve reaching ~90% of plateau by ~month 18 (used when the cluster has too few donors)."""
    m = np.arange(HORIZON)
    return np.clip(1.0 - np.exp(-(m + 1) / 8.0), 0.0, 1.0)


def _cluster_asp(df: pd.DataFrame, keys: List[str]) -> Dict[str, Optional[float]]:
    """Cluster membership/retail ASP = median of donors' own mature-window ASP (falls back to their overall)."""
    if not keys:
        return {"asp_mem": None, "asp_ret": None, "mem_share": None}
    sub = df[df.site_key.isin(keys)]
    mat = sub[(sub.rel >= D.MAT_LO) & (sub.rel <= D.MAT_HI)]
    base = mat if len(mat) else sub
    asp_mem = pd.to_numeric(base.ASP_mem, errors="coerce").replace(0, np.nan).median()
    asp_ret = pd.to_numeric(base.ASP_ret, errors="coerce").replace(0, np.nan).median()
    mem_share = pd.to_numeric(base.mem_share_wash, errors="coerce").median()
    return {"asp_mem": float(asp_mem) if np.isfinite(asp_mem) else None,
            "asp_ret": float(asp_ret) if np.isfinite(asp_ret) else None,
            "mem_share": float(mem_share) if np.isfinite(mem_share) else None}


def project_site(lat: float, lon: float, *, radius_km: float = C.HISTORICAL_CLUSTER_KM,
                 horizon: int = HORIZON) -> Dict[str, Any]:
    """A council-local 5-yr forecast for a hypothetical build at (lat, lon). Pure panel math; never raises."""
    df, site = D.load_panel_1_6()
    neigh = D.neighbours_within(site, lat, lon, radius_km)
    donors = _qualified_donors(neigh)

    # mature anchor: qualified donors → else any matured neighbour → else the global healthy floor
    if len(donors) and donors.mature_level.notna().any():
        anchor = float(donors.mature_level.median())
        anchor_src = f"{len(donors)} considered site(s) (≥{C.MIN_HISTORY_MONTHS}mo, matured, ex-2020)"
    elif neigh.mature_level.notna().any():
        anchor = float(neigh.mature_level.median())
        anchor_src = f"{int(neigh.mature_level.notna().sum())} matured neighbours (relaxed bar)"
    else:
        anchor = D.mature_floor(site)
        anchor_src = "global healthy floor (no site meets the consideration bar)"

    donor_keys = donors.site_key.tolist()
    mature_map = donors.set_index("site_key").mature_level if len(donors) else pd.Series(dtype=float)
    ramp = _learned_ramp(df, donor_keys, mature_map)
    ramp_src = "learned"
    if ramp is None:
        ramp, ramp_src = _synthetic_ramp(), "synthetic (thin cluster)"
    ramp = ramp[:horizon]

    asp = _cluster_asp(df, donor_keys or neigh.site_key.tolist())
    mem_share = asp["mem_share"] if asp["mem_share"] is not None else 0.5
    asp_mem = asp["asp_mem"] if asp["asp_mem"] else 12.0
    asp_ret = asp["asp_ret"] if asp["asp_ret"] else 8.0

    total_wash = anchor * ramp
    mem_wash = total_wash * mem_share
    ret_wash = total_wash * (1.0 - mem_share)
    revenue = mem_wash * asp_mem + ret_wash * asp_ret

    # ramp-to-90%: first month the trajectory reaches 90% of the anchor plateau
    reach = np.where(total_wash >= 0.9 * anchor)[0]
    ramp_to_90 = int(reach[0]) if len(reach) else horizon

    # observed local peak month (corroboration for the tunnel formula)
    local_recent = float(df[df.site_key.isin(neigh.site_key)].groupby("site_key").tot_wash_count
                         .apply(lambda s: s.tail(6).mean()).median()) if len(neigh) else None

    return {
        "months": list(range(horizon)),
        "total_wash": [float(x) for x in total_wash],
        "mem_wash": [float(x) for x in mem_wash],
        "ret_wash": [float(x) for x in ret_wash],
        "revenue": [float(x) for x in revenue],
        "ramp": [float(x) for x in ramp],
        # headline scalars
        "mature_anchor": anchor,
        "anchor_source": anchor_src,
        "peak_month_washes": float(np.nanmax(total_wash)),
        "wash_5yr": float(np.nansum(total_wash)),
        "revenue_5yr": float(np.nansum(revenue)),
        "mem_share": float(mem_share),
        "asp_mem": float(asp_mem),
        "asp_ret": float(asp_ret),
        "ramp_to_90pct_months": ramp_to_90,
        "ramp_source": ramp_src,
        "n_donors": int(len(donors)),
        "n_neighbours": int(len(neigh)),
        "local_recent_wash": local_recent,
        "region": str(donors.region.iloc[0]) if len(donors) and "region" in donors else (
            str(neigh.region.iloc[0]) if len(neigh) and "region" in neigh else None),
    }
