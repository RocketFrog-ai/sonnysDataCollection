"""
Leakage-clean, as-of-T features for the SIGNAL decider.

The exploration found that a car wash's realized mature LEVEL is near-unpredictable pre-build (the trained
forecast's apparent skill was operator-identity leakage; an honest as-of-T refit → corr ~0). But the binary
good/bad OUTCOME is modestly predictable (AUC ~0.62) from LOCAL-MARKET STRUCTURE + operator scale — NOT from
demographics/traffic (near-zero predictors, and leaky 2025 snapshots).

Every feature here is STRICT: a neighbour/operator only contributes its mature level if its OWN 18–30mo
maturity window ended before T (op_start + 30mo <= T), so nothing post-T leaks in. Audited: 0 leakage
violations. Ported from the exploration agent's build_features.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from council import data_1_6 as D

RADIUS_KM = 20.0

# the decider's feature set (best leakage-clean subset; op count is the pre-T portfolio, not the leaky total)
FEATURES: List[str] = ["local_recent_wash_mean", "nbr_mat_min_strict", "nbr_mat_std_strict", "op_n_sites_preT"]


def site_features(focal_key: str, lat: float, lon: float, as_of, *, df: Optional[pd.DataFrame] = None,
                  site: Optional[pd.DataFrame] = None, radius_km: float = RADIUS_KM) -> Dict[str, Any]:
    """Compute the as-of-T market-structure + operator features for one (hypothetical) build at (lat, lon)
    opening at T=`as_of`. Pure lookup on the 1.6 CSV; leakage-clean by construction."""
    if df is None or site is None:
        df, site = D.load_panel_1_6()
    T = pd.Timestamp(as_of)
    client_id = str(focal_key).split("::")[0]
    cs = site[site.has_coords]
    d = D.haversine_km(lat, lon, cs.lat.values, cs.lon.values)
    in_r = (d <= radius_km) & (cs.site_key.values != focal_key) & (cs.op_start.values < np.datetime64(T))
    pre = cs[in_r].copy()
    pre["dist_km"] = d[in_r]

    # STRICT matured neighbours: their own 18–30mo window ended before T → their mature_level is knowable at T
    matured = pre[(pre.op_start + pd.DateOffset(months=D.MAT_HI) <= T) & pre.mature_level.notna()]
    lv = matured.mature_level.dropna()
    rec: Dict[str, Any] = {
        "n_pre_nbrs": int(len(pre)),
        "n_pre_nbrs_5km": int((pre.dist_km <= 5).sum()),
        "n_matured_pre_nbrs": int(len(lv)),
        "nbr_mat_min_strict": float(lv.min()) if len(lv) else np.nan,
        "nbr_mat_std_strict": float(lv.std()) if len(lv) > 1 else np.nan,
        "nbr_mat_median_strict": float(lv.median()) if len(lv) else np.nan,
        "nbr_mat_cv_strict": float(lv.std() / lv.mean()) if len(lv) > 1 and lv.mean() else np.nan,
    }
    # local recent demand: pre-T neighbours' mean monthly washes over the 6 months before T
    pnl = df[df.site_key.isin(pre.site_key) & (df.date < T)]
    recent = pnl[pnl.date >= (T - pd.DateOffset(months=6))]
    rec["local_recent_wash_mean"] = (float(recent.groupby("site_key")["tot_wash_count"].mean().mean())
                                     if len(recent) else np.nan)

    # operator scale + track record, both known at T (portfolio = sites opened before T; +1 for this build)
    op = site[(site.client_id == client_id) & (site.site_key != focal_key)]
    rec["op_n_sites_preT"] = int((op.op_start < T).sum()) + 1
    op_strict = op[(op.op_start + pd.DateOffset(months=D.MAT_HI) <= T) & op.mature_level.notna()]
    rec["op_n_prior_matured"] = int(len(op_strict))
    rec["op_track_mean_strict"] = float(op_strict.mature_level.mean()) if len(op_strict) else np.nan
    rec["op_track_goodfrac_strict"] = (float((op_strict.mature_level >= D.mature_floor(site)).mean())
                                       if len(op_strict) else np.nan)
    return rec


def build_matrix(*, radius_km: float = RADIUS_KM) -> pd.DataFrame:
    """Feature matrix + label for ALL focal candidates (~420) — the decider's training/eval data. Offline,
    no LLM. Columns: site_key, client_id, t_open, <features>, y (good build), realized_mature, group (operator)."""
    from council.scorer import realized_outcome
    df, site = D.load_panel_1_6()
    cand = D.focal_candidates(radius_km=radius_km)
    rows = []
    for _, r in cand.iterrows():
        f = site_features(r.site_key, r.lat, r.lon, r["t_open"], df=df, site=site, radius_km=radius_km)
        out = realized_outcome(r.site_key, r["t_open"], df=df, site=site)
        f.update({"site_key": r.site_key, "client_id": str(r.site_key).split("::")[0],
                  "t_open": r["t_open"], "y": int(bool(out["realized_good_build"])),
                  "realized_mature": out["realized_mature_washes"], "express_like": bool(r.express_like)})
        rows.append(f)
    m = pd.DataFrame(rows)
    m["group"] = m["client_id"]                                   # GroupKFold groups sites by operator
    return m
