"""Lifecycle ramp curves.

A ramp curve maps ``site_age_months`` to an expected fraction of the
site's eventual mature monthly volume. We learn one ramp curve per
region (and a national fallback) using sites that have an observable
mature phase, then use it to project month-0..N for a brand-new site.

Because new sites in 2024–2025 don't have many years of history, we
estimate ramp shape from MT sites (which DO have mature observations)
and validate against LT cohort trajectories. The output is a smooth
monotone-ish multiplier table that lifts a peer mature reference to a
month-of-life estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _smooth(series: pd.Series, win: int = 5) -> pd.Series:
    return series.rolling(win, min_periods=1, center=True).median()


def estimate_ramp_curves(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Return a long DataFrame: (level, level_value, age_months, ramp_factor).

    ``level`` is one of ``region`` or ``national``. ``ramp_factor`` is the
    median ratio of monthly washes to the site's own mature reference.
    Capped to [0.05, 1.4] so a stale outlier can't blow up the curve.
    """
    drop_dup = [c for c in ("state", "region") if c in panel.columns and c in static.columns]
    base = panel.drop(columns=drop_dup) if drop_dup else panel
    p = base.merge(static[["client_id_location_id", "region", "state"]],
                   on="client_id_location_id", how="left")
    # Per-site mature reference (median of last 12 months that are mature).
    refs = []
    for sid, g in p.groupby("client_id_location_id"):
        mat = g[g["site_age_months"] >= C.MATURITY_MONTHS]
        if len(mat) < 6:
            mat = g.sort_values("site_age_months").tail(6)
        ref = mat["wash_count_total"].median()
        if ref and ref > 0:
            refs.append((sid, ref))
    site_ref = pd.DataFrame(refs, columns=["client_id_location_id", "site_mature_ref"])
    p = p.merge(site_ref, on="client_id_location_id", how="inner")
    p["ramp_obs"] = (p["wash_count_total"] / p["site_mature_ref"]).clip(0.05, 1.4)

    out_pieces = []
    # National
    g = (p.groupby("site_age_months")["ramp_obs"].median().reset_index()
           .rename(columns={"site_age_months": "age_months", "ramp_obs": "ramp_factor"}))
    g["ramp_factor"] = _smooth(g["ramp_factor"], 5)
    g["level"] = "national"
    g["level_value"] = "ALL"
    out_pieces.append(g)

    # Regional
    for region, gr in p.groupby("region"):
        if pd.isna(region):
            continue
        gg = (gr.groupby("site_age_months")["ramp_obs"].median().reset_index()
              .rename(columns={"site_age_months": "age_months", "ramp_obs": "ramp_factor"}))
        if len(gg) < 6:
            continue
        gg["ramp_factor"] = _smooth(gg["ramp_factor"], 5)
        gg["level"] = "region"
        gg["level_value"] = str(region)
        out_pieces.append(gg)

    ramp = pd.concat(out_pieces, ignore_index=True)
    # Extend to FORECAST_HORIZON_MONTHS by carrying the median of months >= MATURITY_MONTHS.
    ramp = _extend_to_horizon(ramp, horizon=C.FORECAST_HORIZON_MONTHS)
    return ramp


def _extend_to_horizon(ramp: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for (lvl, val), g in ramp.groupby(["level", "level_value"]):
        g = g.set_index("age_months")["ramp_factor"]
        # Fill 0..horizon
        idx = pd.Index(range(0, horizon + 1), name="age_months")
        g = g.reindex(idx)
        # Interpolate interior, then ffill / bfill at edges.
        g = g.interpolate("linear").bfill().ffill()
        # After maturity, anchor at median of [MATURITY_MONTHS, MATURITY_MONTHS+12]
        mature_window = g.loc[C.MATURITY_MONTHS:C.MATURITY_MONTHS + 12].median()
        if not np.isnan(mature_window):
            g.loc[C.MATURITY_MONTHS:] = mature_window
        out = g.reset_index().rename(columns={0: "ramp_factor"})
        out["level"] = lvl
        out["level_value"] = val
        rows.append(out)
    out = pd.concat(rows, ignore_index=True)
    out = out[["level", "level_value", "age_months", "ramp_factor"]]
    return out


def get_ramp(ramp: pd.DataFrame, region: str | None) -> pd.Series:
    """Return a ramp_factor series indexed by age_months for the given region.
    Falls back to national.
    """
    sub = ramp[(ramp["level"] == "region") & (ramp["level_value"] == str(region))]
    if len(sub) == 0:
        sub = ramp[ramp["level"] == "national"]
    return sub.set_index("age_months")["ramp_factor"].sort_index()
