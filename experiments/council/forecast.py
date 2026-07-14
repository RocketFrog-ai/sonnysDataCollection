"""
Council forward forecast — the PRODUCTION cold-start model over HTTP, panel math as the fallback.

`project_site(lat, lon)` returns a 5-year monthly washcount + revenue trajectory for a hypothetical build.
Per Dhruv: the repo already HAS a trained cold-start forecaster (plateau × ramp LightGBM with P10–P90
bands) — the council should not re-derive washcount from cluster medians. So the projection is now
API-first: `POST {FORECAST_API_BASE}/v1/pnl_analysis/pinpoint-forecast` (code stays self-contained —
no `proforma` import; the server owns the 46 MB artifact). When the backend isn't running, the original
pure-panel projection (considered-sites median anchor × learned ramp) takes over, clearly labeled.

Either way the LOCAL grounding stays panel-computed and on the board: the consideration set, cluster ASP
(pricing), membership split fallback, observed local peak. The model supplies the LEVEL + BAND + SHAPE;
the cluster supplies the prices and the debate material.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from experiments.council import config as C
from experiments.council import data_1_6 as D

HORIZON = 60
API_BASE = os.getenv("COUNCIL_FORECAST_API", "http://localhost:8001")   # python -m app.main serves on :8001
API_TIMEOUT = float(os.getenv("COUNCIL_FORECAST_API_TIMEOUT", "60"))


def _coldstart_api(lat: float, lon: float, horizon: int) -> Optional[Dict[str, Any]]:
    """The production cold-start forecast via the FastAPI backend (pinpoint-forecast: trajectory + P10-P90).
    Returns the parsed JSON, or None on ANY failure (backend down, timeout, bad payload) — never raises."""
    try:
        import requests
        r = requests.post(f"{API_BASE}/v1/pnl_analysis/pinpoint-forecast",
                          json={"latitude": lat, "longitude": lon, "horizon_months": int(horizon)},
                          timeout=API_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        traj, summ = j.get("trajectory") or {}, j.get("summary") or {}
        if not traj.get("total_med") or summ.get("plateau_med") is None:
            return None
        return j
    except Exception:
        return None


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
    """A 5-yr forecast for a hypothetical build at (lat, lon): the production cold-start model (API) when
    reachable — mature level + P10-P90 band + trajectory shape — else pure panel math. Never raises."""
    df, site = D.load_panel_1_6()
    neigh = D.neighbours_within(site, lat, lon, radius_km)
    donors = _qualified_donors(neigh)
    donor_keys = donors.site_key.tolist()

    # local pricing + split, ALWAYS panel-computed (the model doesn't price; the cluster does)
    asp = _cluster_asp(df, donor_keys or neigh.site_key.tolist())
    asp_mem = asp["asp_mem"] if asp["asp_mem"] else 12.0
    asp_ret = asp["asp_ret"] if asp["asp_ret"] else 8.0

    api = _coldstart_api(lat, lon, horizon)
    anchor_lo = anchor_hi = None
    if api is not None:
        # ── the production model: level + band + shape ──
        summ, traj = api["summary"], api["trajectory"]
        anchor = float(summ["plateau_med"])
        anchor_lo, anchor_hi = float(summ["plateau_lo"]), float(summ["plateau_hi"])
        total_wash = np.asarray(traj["total_med"], dtype=float)[:horizon]
        mem_wash = np.asarray(traj.get("mem_med") or [], dtype=float)[:horizon]
        if len(mem_wash) != len(total_wash):
            mem_wash = total_wash * float(summ.get("mem_share", 0.6))
        ret_wash = total_wash - mem_wash
        mem_share = float(summ.get("mem_share", 0.6))
        forecast_source = "coldstart-api"
        anchor_src = (f"production cold-start model p50 (p10 {anchor_lo:,.0f} – p90 {anchor_hi:,.0f}); "
                      f"ramp: {summ.get('ramp_source')}")
        ramp = np.divide(total_wash, max(anchor, 1e-9))
        ramp_src = str(summ.get("ramp_source") or "model")
    else:
        # ── panel fallback: considered-sites median anchor × learned ramp ──
        if len(donors) and donors.mature_level.notna().any():
            anchor = float(donors.mature_level.median())
            anchor_src = f"{len(donors)} considered site(s) (≥{C.MIN_HISTORY_MONTHS}mo, matured, ex-2020)"
        elif neigh.mature_level.notna().any():
            anchor = float(neigh.mature_level.median())
            anchor_src = f"{int(neigh.mature_level.notna().sum())} matured neighbours (relaxed bar)"
        else:
            anchor = D.mature_floor(site)
            anchor_src = "global healthy floor (no site meets the consideration bar)"
        mature_map = donors.set_index("site_key").mature_level if len(donors) else pd.Series(dtype=float)
        ramp = _learned_ramp(df, donor_keys, mature_map)
        ramp_src = "learned"
        if ramp is None:
            ramp, ramp_src = _synthetic_ramp(), "synthetic (thin cluster)"
        ramp = ramp[:horizon]
        mem_share = asp["mem_share"] if asp["mem_share"] is not None else 0.5
        total_wash = anchor * ramp
        mem_wash = total_wash * mem_share
        ret_wash = total_wash * (1.0 - mem_share)
        forecast_source = "panel (API unavailable)"

    revenue = mem_wash * asp_mem + ret_wash * asp_ret

    # ramp-to-90%: first month the trajectory reaches 90% of the anchor plateau
    reach = np.where(total_wash >= 0.9 * anchor)[0]
    ramp_to_90 = int(reach[0]) if len(reach) else horizon

    # observed local peak month (corroboration for the tunnel formula)
    local_recent = float(df[df.site_key.isin(neigh.site_key)].groupby("site_key").tot_wash_count
                         .apply(lambda s: s.tail(6).mean()).median()) if len(neigh) else None

    return {
        "months": list(range(len(total_wash))),
        "total_wash": [float(x) for x in total_wash],
        "mem_wash": [float(x) for x in mem_wash],
        "ret_wash": [float(x) for x in ret_wash],
        "revenue": [float(x) for x in revenue],
        "ramp": [float(x) for x in ramp],
        # headline scalars
        "mature_anchor": anchor,
        "mature_anchor_lo": anchor_lo,          # P10 (model runs only)
        "mature_anchor_hi": anchor_hi,          # P90 (model runs only)
        "anchor_source": anchor_src,
        "forecast_source": forecast_source,
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
