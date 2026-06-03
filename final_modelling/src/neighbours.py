"""Geographic neighbour search + precomputed cluster bounding boxes.

`find_neighbours` is the geographic core of the model: given a candidate
location it returns the sites inside the BUFFER_KM radius (the proven co-moving
neighbourhood), with a k-nearest fallback when the radius is too sparse.

`precompute_cluster_bboxes` re-runs the notebook's 20 km complete-linkage
clustering so the Streamlit map can shade the regions where neighbours exist.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Haversine
# --------------------------------------------------------------------------- #
def haversine_km_vec(lat0: float, lon0: float, lats, lons) -> np.ndarray:
    """Great-circle distance (km) from one point to many, using EARTH_KM."""
    lat0r, lon0r = np.radians(lat0), np.radians(lon0)
    latr, lonr = np.radians(np.asarray(lats, float)), np.radians(np.asarray(lons, float))
    dlat = latr - lat0r
    dlon = lonr - lon0r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0r) * np.cos(latr) * np.sin(dlon / 2) ** 2
    return 2 * C.EARTH_KM * np.arcsin(np.sqrt(a))


def cluster_radius_km(lat, lon) -> float:
    """Max haversine distance of any member from the cluster centroid (km)."""
    lat, lon = np.radians(np.asarray(lat)), np.radians(np.asarray(lon))
    if len(lat) <= 1:
        return 0.0
    clat, clon = lat.mean(), lon.mean()
    a = np.sin((lat - clat) / 2) ** 2 + np.cos(lat) * np.cos(clat) * np.sin((lon - clon) / 2) ** 2
    return float(np.max(2 * C.EARTH_KM * np.arcsin(np.sqrt(a))))


# --------------------------------------------------------------------------- #
# Neighbour search
# --------------------------------------------------------------------------- #
def find_neighbours(
    lat: float,
    lon: float,
    sites: pd.DataFrame,
    buffer_km: float = C.BUFFER_KM,
    min_neighbours: int = C.MIN_NEIGHBOURS,
    knn_k: int = C.KNN_FALLBACK_K,
    exclude_uid: str | None = None,
) -> pd.DataFrame:
    """Return neighbour sites of (lat, lon), sorted by distance.

    Within `buffer_km` if at least `min_neighbours` qualify (mode='radius');
    otherwise the `knn_k` nearest sites regardless of distance (mode='knn').
    `exclude_uid` hides a site from its own search (used by LOOCV).
    """
    s = sites
    if exclude_uid is not None:
        s = s[s["site_uid"] != exclude_uid]

    dist = haversine_km_vec(lat, lon, s["lat"].values, s["lon"].values)
    out = s.assign(dist_km=dist)

    within = out[out["dist_km"] <= buffer_km]
    if len(within) >= min_neighbours:
        res = within.copy()
        res["mode"] = "radius"
    else:
        res = out.nsmallest(knn_k, "dist_km").copy()
        res["mode"] = "knn"

    cols = ["site_uid", "client", "state", "lat", "lon", "dist_km", "mode"]
    return res.sort_values("dist_km")[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cluster bounding boxes (precomputed once for the map)
# --------------------------------------------------------------------------- #
def precompute_cluster_bboxes(sites: pd.DataFrame) -> list[dict]:
    """Re-run the notebook's 20 km complete-linkage clustering and return one
    bounding-box dict per kept cluster (>= MIN_SITES sites)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances

    coords_rad = np.radians(sites[["lat", "lon"]].to_numpy())
    Dkm = pairwise_distances(coords_rad, metric="haversine") * C.EARTH_KM

    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=C.BUFFER_KM,
        metric="precomputed",
        linkage="complete",
    ).fit(Dkm)

    lab = pd.Series(agg.labels_)
    keep = lab.value_counts()
    keep = keep[keep >= C.MIN_SITES].index

    boxes = []
    for new_id, old in enumerate(keep):
        idx = np.where(lab.values == old)[0]
        g = sites.iloc[idx]
        boxes.append(
            {
                "cluster_id": int(new_id),
                "n_sites": int(len(idx)),
                "dominant_state": str(g["state"].mode().iat[0]) if len(g) else "",
                "lat_min": float(g["lat"].min()),
                "lat_max": float(g["lat"].max()),
                "lon_min": float(g["lon"].min()),
                "lon_max": float(g["lon"].max()),
                "centroid_lat": float(g["lat"].mean()),
                "centroid_lon": float(g["lon"].mean()),
                "radius_km": round(cluster_radius_km(g["lat"].values, g["lon"].values), 2),
            }
        )
    return boxes


def save_cluster_bboxes(sites: pd.DataFrame, path=C.BBOX_PATH) -> list[dict]:
    boxes = precompute_cluster_bboxes(sites)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(boxes, indent=1))
    return boxes


def load_cluster_bboxes(path=C.BBOX_PATH) -> list[dict]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    from .data_loader import load_all

    ds = load_all()
    boxes = save_cluster_bboxes(ds.sites)
    covered = sum(b["n_sites"] for b in boxes)
    print(f"Wrote {len(boxes)} cluster bounding boxes -> {C.BBOX_PATH}")
    print(f"Sites inside a >=3-site cluster: {covered:,} / {len(ds.sites):,}")
    biggest = sorted(boxes, key=lambda b: -b["n_sites"])[:5]
    for b in biggest:
        print(f"  cluster {b['cluster_id']:3d}: {b['n_sites']:3d} sites  "
              f"{b['dominant_state']:>3}  r={b['radius_km']:.1f} km")
