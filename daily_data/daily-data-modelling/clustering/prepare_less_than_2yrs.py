"""
Prepare <2-year dataset for clustering-ready CSV (input to **clustering_v2**, not train code here).

Production train/project lives in **../clustering_v2/**. Run this only to (re)build
`less_than-2yrs-clustering-ready.csv` beside the modelling CSVs.

Steps:
1) Load less_than-2yrs.csv
2) Build calendar_day and lag/time features (wash-based lags only; no weather/gas placeholders)
3) Add latitude/longitude (map from existing datasets, then geocode unresolved addresses)
4) Fit site-level DBSCAN clusters (12km, 18km) and merge onto all rows
5) Drop all-null columns and save clustering-ready CSV
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.common import get_lat_long


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "less_than-2yrs.csv"
MORE_THAN_PATH = ROOT / "more_than-2yrs.csv"
MASTER_PATH = ROOT / "master_daily_with_site_metadata.csv"
OUTPUT_PATH = ROOT / "less_than-2yrs-clustering-ready.csv"
GEOCODE_CACHE_PATH = CLUSTER_DIR / "results" / "less_than_2yrs_geocode_cache.json"

EARTH_RADIUS_KM = 6371.0088


def _load_geocode_cache() -> Dict[str, Dict[str, float]]:
    if GEOCODE_CACHE_PATH.exists():
        with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_geocode_cache(cache: Dict[str, Dict[str, float]]) -> None:
    GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _build_geo_maps() -> Tuple[Dict[int, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    site_geo: Dict[int, Tuple[float, float]] = {}
    addr_geo: Dict[str, Tuple[float, float]] = {}

    for path, site_col, addr_col in [
        (MORE_THAN_PATH, "site_client_id", "Address"),
        (MASTER_PATH, "site_client_id", "Address"),
    ]:
        if not path.exists():
            continue
        src = pd.read_csv(path, low_memory=False, usecols=[site_col, addr_col, "latitude", "longitude"])
        src = src.dropna(subset=["latitude", "longitude"])

        for _, r in src.iterrows():
            sid = r.get(site_col)
            if pd.notna(sid):
                site_geo[int(sid)] = (float(r["latitude"]), float(r["longitude"]))
            addr = str(r.get(addr_col) or "").strip()
            if addr:
                addr_geo[addr] = (float(r["latitude"]), float(r["longitude"]))
    return site_geo, addr_geo


def _resolve_geocode(address: str, cache: Dict[str, Dict[str, float]]) -> Tuple[float | None, float | None]:
    key = (address or "").strip()
    if not key:
        return None, None
    cached = cache.get(key)
    if cached:
        return cached.get("latitude"), cached.get("longitude")
    geo = get_lat_long(key)
    if not geo:
        cache[key] = {"latitude": None, "longitude": None}
        return None, None
    lat, lon = geo.get("lat"), geo.get("lon")
    if lat is None or lon is None:
        cache[key] = {"latitude": None, "longitude": None}
        return None, None
    cache[key] = {"latitude": float(lat), "longitude": float(lon)}
    return float(lat), float(lon)


def _fit_dbscan(df_sites: pd.DataFrame, radius_km: float) -> np.ndarray:
    coords = np.radians(df_sites[["latitude", "longitude"]].to_numpy(dtype=float))
    eps = radius_km / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps, min_samples=2, metric="haversine")
    return model.fit_predict(coords)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input missing: {INPUT_PATH}")

    print(f"Loading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df["month_number"] = pd.to_numeric(df["month_number"], errors="coerce")
    df["year_number"] = pd.to_numeric(df["year_number"], errors="coerce")
    df = df[df["month_number"].notna() & df["year_number"].notna()].copy()
    df["month_number"] = df["month_number"].astype(int)
    df["year_number"] = df["year_number"].astype(int)
    # Cumulative month label from the source panel: 1-12 = first operational year,
    # 13-24 = second year (not calendar Jan-Dec alone). Then:
    #   period_index = (year_number - 1) * 12 + month_number
    # yields 1-12 for year 1 and 25-36 for year 2 — values 13-24 never occur by construction.
    df["period_index"] = (df["year_number"] - 1) * 12 + df["month_number"]
    # Synthetic month-start dates for modeling (not necessarily literal transaction dates).
    base_period = pd.Period("2024-01", freq="M")
    df["calendar_day"] = (
        (df["period_index"] - 1)
        .astype(int)
        .apply(lambda x: (base_period + x).to_timestamp())
    )
    df["year_month"] = df["calendar_day"].dt.to_period("M")

    # Standardized columns expected by clustering/model scripts.
    df["Address"] = df["address"]
    # Monthly panel: create a monotonic period index instead of day-of-month semantics.
    ym_order = sorted(df["year_month"].dropna().unique())
    ym_rank = {p: i + 1 for i, p in enumerate(ym_order)}
    df["day_number"] = df["year_month"].map(ym_rank).astype(float)
    df["day_of_week"] = "monthly_panel"
    # Reuse model feature slot with month-of-year seasonality signal.
    df["day_of_week_feature"] = (df["period_index"] - 1) % 12

    # Encodings used by model inputs.
    if "region_enc" not in df.columns:
        region_vals = sorted([x for x in df["region"].dropna().astype(str).unique()])
        region_map = {v: i for i, v in enumerate(region_vals)}
        df["region_enc"] = df["region"].astype(str).map(region_map).astype(float)
    if "state_enc" not in df.columns:
        state_vals = sorted([x for x in df["state"].dropna().astype(str).unique()])
        state_map = {v: i for i, v in enumerate(state_vals)}
        df["state_enc"] = df["state"].astype(str).map(state_map).astype(float)

    site_geo, addr_geo = _build_geo_maps()
    cache = _load_geocode_cache()

    site_snapshot = (
        df.sort_values("calendar_day")
        .groupby("site_client_id", as_index=False)
        .last()[["site_client_id", "Address"]]
        .copy()
    )
    site_snapshot["latitude"] = np.nan
    site_snapshot["longitude"] = np.nan

    resolved_from_existing = 0
    resolved_from_geocode = 0

    for idx, row in site_snapshot.iterrows():
        sid = int(row["site_client_id"])
        addr = str(row["Address"] or "").strip()

        if sid in site_geo:
            lat, lon = site_geo[sid]
            site_snapshot.at[idx, "latitude"] = lat
            site_snapshot.at[idx, "longitude"] = lon
            resolved_from_existing += 1
            continue

        if addr in addr_geo:
            lat, lon = addr_geo[addr]
            site_snapshot.at[idx, "latitude"] = lat
            site_snapshot.at[idx, "longitude"] = lon
            resolved_from_existing += 1
            continue

        lat, lon = _resolve_geocode(addr, cache)
        if lat is not None and lon is not None:
            site_snapshot.at[idx, "latitude"] = lat
            site_snapshot.at[idx, "longitude"] = lon
            resolved_from_geocode += 1

    _save_geocode_cache(cache)

    unresolved = int(site_snapshot["latitude"].isna().sum())
    print(
        "Geo resolution:",
        {
            "sites_total": int(len(site_snapshot)),
            "from_existing": resolved_from_existing,
            "from_geocode": resolved_from_geocode,
            "unresolved": unresolved,
        },
    )

    df = df.merge(
        site_snapshot[["site_client_id", "latitude", "longitude"]],
        on="site_client_id",
        how="left",
    )

    # Monthly time-series lag features per site.
    df = df.sort_values(["site_client_id", "period_index"]).reset_index(drop=True)
    grp = df.groupby("site_client_id", sort=False)
    df["prev_wash_count"] = grp["wash_count_total"].shift(1)     # previous month
    df["last_week_same_day"] = grp["wash_count_total"].shift(12) # same month last year
    df["running_avg_7_days"] = (
        grp["wash_count_total"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # Site-level DBSCAN labels and merge to all rows.
    valid_sites = site_snapshot.dropna(subset=["latitude", "longitude"]).copy()
    valid_sites["dbscan_cluster_12km"] = _fit_dbscan(valid_sites, 12.0)
    valid_sites["dbscan_cluster_18km"] = _fit_dbscan(valid_sites, 18.0)

    df = df.merge(
        valid_sites[["site_client_id", "dbscan_cluster_12km", "dbscan_cluster_18km"]],
        on="site_client_id",
        how="left",
    )
    df["dbscan_cluster_12km"] = df["dbscan_cluster_12km"].fillna(-1).astype(int)
    df["dbscan_cluster_18km"] = df["dbscan_cluster_18km"].fillna(-1).astype(int)

    # Drop columns with no usable values (no NaN-only placeholders for weather/gas/etc.).
    keep = [c for c in df.columns if df[c].notna().any()]
    dropped = sorted(set(df.columns) - set(keep))
    if dropped:
        print(f"Dropping {len(dropped)} all-null / empty columns: {dropped[:20]}{'...' if len(dropped) > 20 else ''}")
    df = df[keep]

    print(f"Saving: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
