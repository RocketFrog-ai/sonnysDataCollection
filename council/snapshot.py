"""
Point-in-time, leakage-controlled market snapshot for the backtest.

`build_snapshot(focal_key, lat, lon, as_of=T)` returns exactly the inputs a council member is
allowed to see when "sitting" at the focal site's opening month T — i.e. the LOCAL MARKET as it
looked strictly before T:

  • panel_asof       — monthly rows for the pre-T neighbours, `date < T` only (feeds compute_metrics).
  • sites_meta_asof  — one row per pre-T neighbour (site_key, name, op_start, dist_km, is_entrant,
                       left_censored) in the shape compute_metrics / market_insights require.
  • nearby_washes    — those neighbours as {name, distance_miles} for the competition seat (leakage-free;
                       never live Google Places, which is a present-day snapshot).

Leakage closed here: future months (`date < T`), later openings counted as incumbents (`op_start < T`),
and the competition seat's nearby set. The focal site itself has no pre-T rows (it opens at T), so it is
absent from the panel — compute_metrics degrades to a clean market-level read of the neighbours, which is
exactly the basis for a build decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from council import data_1_6 as D

MIN_NBR_MONTHS = 6   # a neighbour with fewer pre-T months adds noise, not signal → dropped


@dataclass
class Snapshot:
    focal_key: str
    lat: float
    lon: float
    as_of: pd.Timestamp
    panel_asof: pd.DataFrame
    sites_meta_asof: pd.DataFrame
    nearby_washes: List[Dict[str, Any]] = field(default_factory=list)
    n_neighbours: int = 0


def _neighbourhood_asof(site: pd.DataFrame, df: pd.DataFrame, focal_key: str, lat: float, lon: float,
                        as_of: pd.Timestamp, radius_km: float, min_nbr_months: int) -> pd.DataFrame:
    """Fork of `market._neighbourhood` with the as-of filters. Neighbours = coord-valid sites within
    `radius_km` that (a) opened strictly before T and (b) have ≥ `min_nbr_months` observations before T.
    `is_entrant` is re-derived AS OF T (opened after the earliest pre-T site, non-left-censored) so a
    later opening can never leak in as an incumbent."""
    nb = D.neighbours_within(site, lat, lon, radius_km, exclude_key=focal_key)
    nb = nb[nb.op_start < as_of].copy()
    if nb.empty:
        nb["is_entrant"] = pd.Series(dtype=bool)
        return nb
    # enough pre-T history per neighbour
    pre = df[df.site_key.isin(nb.site_key) & (df.date < as_of)]
    obs = pre.groupby("site_key").size()
    nb = nb[nb.site_key.map(obs).fillna(0) >= min_nbr_months].copy()
    if nb.empty:
        nb["is_entrant"] = pd.Series(dtype=bool)
        return nb
    nb = nb.sort_values("op_start")
    earliest = nb.op_start.min()
    nb["is_entrant"] = (~nb.left_censored) & (nb.op_start > earliest)
    return nb.reset_index(drop=True)


def build_snapshot(focal_key: str, lat: float, lon: float, as_of: pd.Timestamp, *,
                   radius_km: float = 20.0, min_nbr_months: int = MIN_NBR_MONTHS) -> Snapshot:
    """Assemble the pre-T snapshot for one focal build. `as_of` is the focal site's opening month T."""
    df, site = D.load_panel_1_6()
    as_of = pd.Timestamp(as_of)

    nb = _neighbourhood_asof(site, df, focal_key, lat, lon, as_of, radius_km, min_nbr_months)

    meta = nb[["site_key", "op_start", "dist_km", "is_entrant", "left_censored", "client_name"]].rename(
        columns={"client_name": "name"}).reset_index(drop=True) if len(nb) else pd.DataFrame(
        columns=["site_key", "op_start", "dist_km", "is_entrant", "left_censored", "name"])

    panel = df[df.site_key.isin(nb.site_key) & (df.date < as_of)].copy() if len(nb) else df.iloc[0:0].copy()

    nearby_washes = [{"name": str(cn), "distance_miles": round(float(dk) * 0.621, 1)}
                     for cn, dk in zip(nb.client_name, nb.dist_km)] if len(nb) else []

    return Snapshot(focal_key=focal_key, lat=float(lat), lon=float(lon), as_of=as_of,
                    panel_asof=panel, sites_meta_asof=meta, nearby_washes=nearby_washes,
                    n_neighbours=int(len(nb)))


def build_live_snapshot(lat: float, lon: float, *, radius_km: float = 20.0,
                        min_nbr_months: int = MIN_NBR_MONTHS) -> Snapshot:
    """Present-day snapshot for a LIVE council read at an arbitrary pin: no time cut (every existing site
    is a neighbour, full history), and a synthetic focal key not in the data (a hypothetical new build).
    Reuses build_snapshot with a far-future `as_of` so nothing is filtered out."""
    return build_snapshot("__live__::0", lat, lon, pd.Timestamp("2100-01-01"),
                          radius_km=radius_km, min_nbr_months=min_nbr_months)
