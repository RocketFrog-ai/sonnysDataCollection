"""End-to-end cold-start forecast engine.

Given a candidate site (lat, lon, optional state, optional open date),
produce a 60-month forecast plus Year 1–5 totals and optional
confidence bands.

The engine is intentionally thin: all the modelling intelligence lives
in peer_groups, lifecycle, features, and the trained boosters. This
module just walks month 1..60, builds one feature row per month, and
calls predict_blend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import h3

from . import config as C
from .features import FEATURE_COLS, _region_code
from .peer_groups import resolve_peer_features
from .lifecycle import get_ramp


def _resolve_h3(h3_id: str | None, calendar_month: int, h3_ref: pd.DataFrame | None) -> dict:
    if h3_id is None or h3_ref is None or len(h3_ref) == 0:
        return {"h3_ref_level": np.nan, "h3_seasonal_factor": np.nan,
                "h3_peer_count": 0, "h3_used": 0}
    # Look up the candidate's own hex first; if not present try the 1-ring
    # neighbours so a brand-new hex with no peers can still inherit local
    # signal from the next-door hex.
    for cell in [h3_id] + list(h3.grid_disk(h3_id, 1)):
        sub = h3_ref[(h3_ref["h3_id"] == cell) & (h3_ref["calendar_month"] == calendar_month)]
        if len(sub) and sub.iloc[0]["h3_peer_count"] >= C.MIN_PEERS_PER_H3:
            r = sub.iloc[0]
            return {"h3_ref_level": float(r["h3_ref_level"]),
                    "h3_seasonal_factor": float(r["h3_seasonal_factor"]),
                    "h3_peer_count": int(r["h3_peer_count"]),
                    "h3_used": 1}
    return {"h3_ref_level": np.nan, "h3_seasonal_factor": np.nan,
            "h3_peer_count": 0, "h3_used": 0}


# Crude state-from-coords fallback. CBSA assignment handles US-mainland sites;
# this is only needed when we don't have a state passed in.
def _region_from_state(state: str | None) -> str:
    if not state:
        return "Other"
    NE = {"CT","ME","MA","NH","RI","VT","NJ","NY","PA"}
    MW = {"IL","IN","IA","KS","MI","MN","MO","NE","ND","OH","SD","WI"}
    S = {"AL","AR","DE","DC","FL","GA","KY","LA","MD","MS","NC","OK","SC","TN","TX","VA","WV"}
    W = {"AK","AZ","CA","CO","HI","ID","MT","NV","NM","OR","UT","WA","WY"}
    s = state.upper()
    if s in NE: return "Northeast"
    if s in MW: return "Midwest"
    if s in S: return "South"
    if s in W: return "West"
    return "Other"


def forecast_site(
    lat: float, lon: float, state: str | None,
    open_date: pd.Timestamp,
    cbsa_id: str | None, cbsa_name: str | None,
    models: dict,
    cbsa_ref: pd.DataFrame,
    rollups: dict,
    ramp: pd.DataFrame,
    horizon: int = C.FORECAST_HORIZON_MONTHS,
    site_history: dict | None = None,
    h3_ref: pd.DataFrame | None = None,
    h3_id: str | None = None,
) -> pd.DataFrame:
    region = _region_from_state(state)
    if h3_id is None:
        h3_id = h3.latlng_to_cell(float(lat), float(lon), C.H3_RES)
    rows = []
    for k in range(horizon):
        ym_ts = open_date + pd.DateOffset(months=k)
        calendar_month = ym_ts.month
        peer = resolve_peer_features(cbsa_id, state, region, calendar_month, cbsa_ref, rollups)
        h3_info = _resolve_h3(h3_id, calendar_month, h3_ref)
        ramp_series = get_ramp(ramp, region)
        # Cap age at the horizon's ramp index
        age_idx = min(k, int(ramp_series.index.max()))
        ramp_factor = float(ramp_series.loc[age_idx])
        # If the caller provided observed history (e.g. a re-forecast of a
        # known site), use those stats; otherwise impute from peer reference.
        if site_history:
            site_mean   = site_history.get("site_train_mean",   peer["peer_ref_level"])
            site_median = site_history.get("site_train_median", peer["peer_ref_level"])
            site_max    = site_history.get("site_train_max",    peer["peer_ref_level"] * 1.5)
            site_std    = site_history.get("site_train_std",    peer["peer_ref_level"] * 0.2)
            site_recent = site_history.get("site_train_recent6", peer["peer_recent"] or peer["peer_ref_level"])
            site_months = site_history.get("site_train_months", 0)
        else:
            site_mean   = peer["peer_ref_level"]
            site_median = peer["peer_ref_level"]
            site_max    = peer["peer_ref_level"] * 1.5
            site_std    = peer["peer_ref_level"] * 0.2
            site_recent = peer["peer_recent"] if peer["peer_recent"] else peer["peer_ref_level"]
            site_months = 0
        # Anchor level: site_mean > H3 local > CBSA peer
        if site_months > 0:
            anchor_level = site_mean
            seasonal_used = peer["peer_seasonal"]
        elif h3_info["h3_used"]:
            anchor_level = h3_info["h3_ref_level"]
            seasonal_used = h3_info["h3_seasonal_factor"]
        else:
            anchor_level = peer["peer_ref_level"]
            seasonal_used = peer["peer_seasonal"]
        anchor = anchor_level * seasonal_used * ramp_factor
        trend_ratio = peer["peer_recent"] / peer["peer_ref_level"] if (peer["peer_ref_level"] and peer["peer_recent"]) else 1.0
        if not np.isfinite(trend_ratio):
            trend_ratio = 1.0
        trend_ratio = float(np.clip(trend_ratio, 0.5, 2.0))
        site_vs_peer_ratio = (site_mean / peer["peer_ref_level"]) if peer["peer_ref_level"] else 1.0
        if not np.isfinite(site_vs_peer_ratio):
            site_vs_peer_ratio = 1.0
        site_vs_peer_ratio = float(np.clip(site_vs_peer_ratio, 0.2, 5.0))
        rows.append({
            "month_offset": k,
            "year_month": ym_ts.strftime("%Y-%m"),
            "site_age_months": k,
            "calendar_month": calendar_month,
            "calendar_month_sin": np.sin(2 * np.pi * calendar_month / 12),
            "calendar_month_cos": np.cos(2 * np.pi * calendar_month / 12),
            "latitude": lat, "longitude": lon,
            "region_code": _region_code(region),
            "cbsa_code": 0,
            "peer_ref_level": peer["peer_ref_level"],
            "peer_recent": peer["peer_recent"],
            "peer_seasonal": peer["peer_seasonal"],
            "peer_count": peer["peer_count"],
            "peer_level_resolved": peer["peer_level_resolved"],
            "peer_trend_ratio": trend_ratio,
            "h3_ref_level": h3_info["h3_ref_level"] if np.isfinite(h3_info["h3_ref_level"]) else peer["peer_ref_level"],
            "h3_seasonal_factor": h3_info["h3_seasonal_factor"] if np.isfinite(h3_info["h3_seasonal_factor"]) else peer["peer_seasonal"],
            "h3_peer_count": h3_info["h3_peer_count"],
            "h3_used": h3_info["h3_used"],
            "h3_vs_cbsa_ratio": float(np.clip(
                (h3_info["h3_ref_level"] / peer["peer_ref_level"])
                if (h3_info["h3_ref_level"] and peer["peer_ref_level"]) else 1.0,
                0.3, 3.0,
            )),
            "ramp_factor": ramp_factor,
            "anchor_wash": anchor,
            "log_anchor": float(np.log1p(max(anchor, 1))),
            "maturity_indicator": int(age_idx >= C.MATURITY_MONTHS),
            "site_train_mean": site_mean, "site_train_median": site_median,
            "site_train_max": site_max, "site_train_std": site_std,
            "site_train_recent6": site_recent, "site_train_months": site_months,
            "site_vs_peer_ratio": site_vs_peer_ratio,
        })
    frame = pd.DataFrame(rows)
    from .train import predict_blend
    frame["predicted_washes"] = np.clip(predict_blend(models, frame), 0, None)

    # Confidence band from training residual stddev per cohort (very simple).
    young_sd = models.get("young", {}).get("residual_sd_log", 0.25)
    mature_sd = models.get("mature", {}).get("residual_sd_log", 0.20)
    sd = np.where(frame["site_age_months"] < C.MATURITY_MONTHS, young_sd, mature_sd)
    frame["predicted_washes_low"] = (frame["predicted_washes"] * np.exp(-1.28 * sd)).clip(lower=0)
    frame["predicted_washes_high"] = frame["predicted_washes"] * np.exp(1.28 * sd)

    frame["cbsa_id"] = cbsa_id
    frame["cbsa_name"] = cbsa_name
    return frame


def year_totals(forecast: pd.DataFrame) -> pd.DataFrame:
    f = forecast.copy()
    f["year_idx"] = (f["month_offset"] // 12) + 1
    agg = f.groupby("year_idx").agg(
        predicted_year_total=("predicted_washes", "sum"),
        predicted_low_total=("predicted_washes_low", "sum"),
        predicted_high_total=("predicted_washes_high", "sum"),
    ).reset_index()
    return agg


def explain_forecast(forecast: pd.DataFrame) -> dict:
    """Produce a human-readable narrative for one forecast.

    Returns peer-resolution level, peer mature reference level, ramp summary,
    seasonality summary, and 5-year totals — the four ingredients that
    together determine the prediction.
    """
    f = forecast
    yt = year_totals(f)
    return {
        "cbsa_name": str(f["cbsa_name"].iloc[0]),
        "cbsa_id": str(f["cbsa_id"].iloc[0]),
        "peer_level_resolved": str(f["peer_level_resolved"].iloc[0]),
        "peer_ref_level_per_month": float(f["peer_ref_level"].iloc[0]),
        "peer_count": int(f["peer_count"].iloc[0]) if pd.notna(f["peer_count"].iloc[0]) else 0,
        "ramp_month0": float(f["ramp_factor"].iloc[0]),
        "ramp_month24": float(f.loc[f["site_age_months"] == 24, "ramp_factor"].iloc[0])
            if (f["site_age_months"] == 24).any() else float("nan"),
        "seasonal_peak_month": int(f.loc[f["peer_seasonal"].idxmax(), "calendar_month"]),
        "seasonal_trough_month": int(f.loc[f["peer_seasonal"].idxmin(), "calendar_month"]),
        "year_totals": yt.to_dict(orient="records"),
        "5_year_total": float(yt["predicted_year_total"].sum()),
    }
