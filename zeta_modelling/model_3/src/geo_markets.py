"""CBSA / MSA market assignment from the Census shapefile.

Strategy
--------
1. Load CBSA polygons (Core-Based Statistical Areas; includes Metropolitan
   Statistical Areas and Micropolitan Statistical Areas).
2. For each site point (lon, lat) use a spatial join (`sjoin`, predicate
   "within") to find the containing CBSA in one vectorised pass.
3. For points that fall outside every CBSA polygon (rural / between metros)
   snap to the **nearest** CBSA centroid if it is within
   ``NEAREST_CBSA_KM_MAX`` km, otherwise label as ``NON_CBSA_<STATE>`` so
   they cluster by state rather than becoming a thousand singletons.

This gives a stable, interpretable, production-friendly market id that does
NOT change run-to-run (unlike DBSCAN which depends on density parameters
and the exact point set).
"""
from __future__ import annotations

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import config as C


def load_cbsa() -> gpd.GeoDataFrame:
    """Load the CBSA polygon layer and reproject to WGS84 lon/lat."""
    cbsa = gpd.read_file(C.CBSA_SHP)
    if cbsa.crs is None:
        cbsa = cbsa.set_crs(4269)  # NAD83 — the Census default
    cbsa = cbsa.to_crs(4326)
    keep = ["CBSAFP", "NAME", "LSAD", "geometry"]
    cbsa = cbsa[[c for c in keep if c in cbsa.columns]].copy()
    # Pre-compute centroid lat/lon for nearest-fallback lookups.
    # Use an equal-area projection so centroids are sensible.
    cen = cbsa.to_crs(5070).geometry.centroid.to_crs(4326)
    cbsa["cbsa_centroid_lon"] = cen.x
    cbsa["cbsa_centroid_lat"] = cen.y
    return cbsa


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def assign_markets(sites_df: pd.DataFrame, cbsa: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Add ``cbsa_id`` / ``cbsa_name`` columns to a sites frame.

    Parameters
    ----------
    sites_df : DataFrame with columns latitude, longitude, state (optional)
    cbsa : pre-loaded CBSA polygons (optional)
    """
    if cbsa is None:
        cbsa = load_cbsa()

    pts = sites_df[["latitude", "longitude"]].copy()
    pts["__idx"] = np.arange(len(pts))
    gdf = gpd.GeoDataFrame(
        pts,
        geometry=[Point(xy) for xy in zip(pts["longitude"], pts["latitude"])],
        crs=4326,
    )

    joined = gpd.sjoin(gdf, cbsa[["CBSAFP", "NAME", "geometry"]], how="left", predicate="within")
    # First polygon wins for any rare double-cover.
    joined = joined.drop_duplicates("__idx", keep="first").sort_values("__idx")

    out = sites_df.copy()
    out["cbsa_id"] = joined["CBSAFP"].values
    out["cbsa_name"] = joined["NAME"].values

    # Fallback 1: snap to nearest CBSA centroid if within threshold.
    mask = out["cbsa_id"].isna()
    if mask.any():
        miss = out[mask][["latitude", "longitude"]].reset_index(drop=True)
        c_lat = cbsa["cbsa_centroid_lat"].values
        c_lon = cbsa["cbsa_centroid_lon"].values
        for i, row in miss.iterrows():
            d = _haversine_km(row["latitude"], row["longitude"], c_lat, c_lon)
            j = int(np.argmin(d))
            if d[j] <= C.NEAREST_CBSA_KM_MAX:
                idx = out.index[mask][i]
                out.at[idx, "cbsa_id"] = cbsa.iloc[j]["CBSAFP"]
                out.at[idx, "cbsa_name"] = cbsa.iloc[j]["NAME"] + " (nearest)"

    # Fallback 2: tag remaining as NON_CBSA_<STATE>.
    mask = out["cbsa_id"].isna()
    if mask.any():
        st = out.loc[mask, "state"].fillna("UNK").astype(str)
        out.loc[mask, "cbsa_id"] = "NON_CBSA_" + st
        out.loc[mask, "cbsa_name"] = "Non-CBSA " + st
    return out


def assign_h3(lat: float, lon: float, res: int = C.H3_RES) -> str:
    """Return the H3 cell id for a (lat, lon) at the given resolution."""
    return h3.latlng_to_cell(float(lat), float(lon), res)


def add_h3(df: pd.DataFrame, res: int = C.H3_RES) -> pd.DataFrame:
    """Add an ``h3_id`` column to a dataframe of points."""
    out = df.copy()
    out["h3_id"] = [h3.latlng_to_cell(float(la), float(lo), res)
                    for la, lo in zip(out["latitude"], out["longitude"])]
    return out


def local_disk_peers(h3_id: str, rings: int = C.H3_DISK_RINGS) -> set:
    """Return the set of H3 cells in the k-ring disk around ``h3_id``.

    With r=8, rings=1 covers the candidate hex plus its 6 neighbours
    (~3 km²). Increase ``rings`` for a wider local catchment.
    """
    return set(h3.grid_disk(h3_id, rings))


def assign_single(lat: float, lon: float, state: str | None, cbsa: gpd.GeoDataFrame) -> dict:
    """Convenience: assign one (lat,lon) — used by the forecast engine."""
    df = pd.DataFrame({"latitude": [lat], "longitude": [lon], "state": [state]})
    out = assign_markets(df, cbsa)
    out["h3_id"] = h3.latlng_to_cell(float(lat), float(lon), C.H3_RES)
    return {
        "cbsa_id": out.iloc[0]["cbsa_id"],
        "cbsa_name": out.iloc[0]["cbsa_name"],
        "h3_id": out.iloc[0]["h3_id"],
    }
