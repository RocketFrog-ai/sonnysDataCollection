"""Neighbour-based spatial prediction for a new car-wash site.

A brand-new site has no history, so the only thing we can weight neighbours by
is geographic distance. `predict_site` finds the sites inside the 20 km cluster
radius and blends their KPI trajectories with inverse-distance weighting (IDW)
or a simple mean. Per-month weights are renormalised over the neighbours that
actually have data that month, so sparse coverage produces an honest gap (NaN)
rather than a value pulled toward zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .neighbours import find_neighbours


@dataclass
class PredictionResult:
    predictions: dict                      # kpi -> pd.Series indexed by month_axis (real units)
    neighbours: pd.DataFrame               # site_uid, client, state, lat, lon, dist_km, mode
    neighbour_trajectories: dict           # kpi -> DataFrame (neighbour_uid x month_axis)
    meta: dict = field(default_factory=dict)

    def normalized(self) -> dict:
        """Min-max (0-1) version of each predicted series, for the shape view."""
        return {k: _minmax(s) for k, s in self.predictions.items()}

    def neighbour_trajectories_normalized(self) -> dict:
        """Min-max each neighbour row independently (per KPI) for the shape view."""
        out = {}
        for kpi, traj in self.neighbour_trajectories.items():
            out[kpi] = traj.apply(_minmax, axis=1)
        return out


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return s * np.nan if hi == lo else s
    return (s - lo) / (hi - lo)


def predict_site(
    lat: float,
    lon: float,
    *,
    panels: dict,
    sites: pd.DataFrame,
    month_axis: pd.PeriodIndex,
    kpis=C.TARGET_KPIS,
    buffer_km: float = C.BUFFER_KM,
    agg: str = "idw",
    min_neighbours: int = C.MIN_NEIGHBOURS,
    knn_k: int = C.KNN_FALLBACK_K,
    exclude_uid: str | None = None,
) -> PredictionResult:
    """Predict each KPI trajectory for a new site at (lat, lon)."""
    nbrs = find_neighbours(
        lat, lon, sites,
        buffer_km=buffer_km, min_neighbours=min_neighbours,
        knn_k=knn_k, exclude_uid=exclude_uid,
    )

    warnings = []
    if nbrs["mode"].iloc[0] == "knn":
        warnings.append(
            f"No sites within {buffer_km:.0f} km — using the {len(nbrs)} nearest "
            f"(up to {nbrs['dist_km'].max():.0f} km). Predictions are less reliable."
        )

    # Distance weights (computed once; a new site has no history to correlate on).
    if agg == "idw":
        w = 1.0 / (nbrs["dist_km"].to_numpy() + C.IDW_EPS_KM)
    elif agg == "mean":
        w = np.ones(len(nbrs))
    else:
        raise ValueError(f"agg must be 'idw' or 'mean', got {agg!r}")

    uids = nbrs["site_uid"].tolist()
    predictions, trajectories = {}, {}
    per_month_counts = {}

    for kpi in kpis:
        traj = panels[kpi].reindex(index=uids, columns=month_axis)   # N x M
        V = traj.to_numpy(float)
        mask = ~np.isnan(V)
        W = w[:, None] * mask                                        # zero weight where missing
        num = np.nansum(np.where(mask, W * V, 0.0), axis=0)
        den = W.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = np.where(den > 0, num / den, np.nan)

        predictions[kpi] = pd.Series(pred, index=month_axis)
        trajectories[kpi] = traj
        per_month_counts[kpi] = mask.sum(axis=0)

    meta = {
        "n_neighbours": int(len(nbrs)),
        "mean_dist_km": float(nbrs["dist_km"].mean()),
        "max_dist_km": float(nbrs["dist_km"].max()),
        "mode": nbrs["mode"].iloc[0],
        "agg": agg,
        "bbox": (
            float(nbrs["lat"].min()), float(nbrs["lat"].max()),
            float(nbrs["lon"].min()), float(nbrs["lon"].max()),
        ),
        "per_month_counts": per_month_counts,
        "warnings": warnings,
    }
    return PredictionResult(predictions, nbrs, trajectories, meta)


if __name__ == "__main__":
    from .data_loader import load_all

    ds = load_all()
    # Predict at a known-dense cluster (the largest, in FL).
    row = ds.sites.loc[ds.sites.state == "FL"].iloc[0]
    res = predict_site(
        row.lat, row.lon,
        panels=ds.panels, sites=ds.sites, month_axis=ds.month_axis,
        exclude_uid=row.site_uid,
    )
    print(f"location ~ {row.site_uid} ({row.state})")
    print(f"neighbours: {res.meta['n_neighbours']}  "
          f"mean {res.meta['mean_dist_km']:.1f} km  mode={res.meta['mode']}")
    for kpi in C.TARGET_KPIS:
        s = res.predictions[kpi].dropna()
        print(f"  {kpi:28s} last={s.iloc[-1]:10.1f}  mean={s.mean():10.1f}")
