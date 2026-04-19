"""
Top-10 Cluster Details
-----------------------
For each DBSCAN radius (6km, 12km, 18km) outputs:
  - results/top10_details_{radius}.json   — rich JSON per cluster
  - results/top10_sites_{radius}.json     — flat site-level records
  - results/top10_summary_{radius}.json — one object per cluster with all stats
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("CLUSTER_DATA_PATH", str(BASE.parent / "master_daily_with_site_metadata.csv")))
RESULTS_DIR = Path(os.getenv("CLUSTER_RESULTS_DIR", str(BASE / "results")))
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["calendar_day"] = pd.to_datetime(df["calendar_day"])

CLUSTER_COLS = {
    "dbscan_cluster_12km": "12km",
    "dbscan_cluster_18km": "18km",
}

# site-level identity columns (one value per location)
SITE_ID_COLS = [
    "location_id", "site_client_id", "client_id",
    "street", "city", "zip", "state", "region",
    "Address", "latitude", "longitude",
    "primary_carwash_type", "tunnel_count",
    "age_on_30_sep_25", "official_website",
    "competitors_count_4miles", "current_count",
]
SITE_KEY_COL = "site_client_id"
TOP_N = 10
LARGE_MIN_SITES = 5


def safe(val):
    """Convert numpy scalars / NaN to JSON-safe Python types."""
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 6)
    return val


def latest_site_snapshot_within_cluster(data, cluster_col, cluster_id, keep_cols, site_key_col):
    """
    Return one latest row per location for the given cluster only.
    This preserves correct site->cluster mapping when cluster assignments
    vary across days/records.
    """
    cdf = data[data[cluster_col] == cluster_id].copy()
    if cdf.empty:
        return cdf

    cdf = cdf.sort_values("calendar_day", ascending=False)
    # Take the first row by date per location, keeping row integrity
    idx = cdf.groupby(site_key_col, sort=False)["calendar_day"].idxmax()
    snap = cdf.loc[idx, keep_cols].copy()
    return snap.reset_index(drop=True)


def build_cluster_stats(sub, cluster_col):
    stats = (
        sub.groupby(cluster_col)["wash_count_total"]
        .agg(
            site_days="count",
            total_washes="sum",
            mean_daily=lambda x: round(x.mean(), 2),
            median_daily="median",
            std_daily=lambda x: round(x.std(), 2),
            min_daily="min",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            max_daily="max",
        )
        .reset_index()
    )
    site_counts = (
        sub.groupby(cluster_col)[SITE_KEY_COL]
        .nunique()
        .rename("n_sites")
        .reset_index()
    )
    stats = stats.merge(site_counts, on=cluster_col, how="left")
    stats["range_daily"] = stats["max_daily"] - stats["min_daily"]
    stats["iqr_daily"] = stats["p75"] - stats["p25"]
    return stats


def export_ranked_outputs(sub, cluster_col, radius, cluster_stats, prefix, ranking_label, min_sites=None):
    avail_site_cols = [c for c in SITE_ID_COLS if c in sub.columns]
    site_cols_for_cluster = [c for c in dict.fromkeys(avail_site_cols + [cluster_col])]

    ranked = cluster_stats.copy()
    if min_sites is not None:
        ranked = ranked[ranked["n_sites"] >= min_sites].copy()
    ranked = ranked.sort_values(["median_daily", "total_washes"], ascending=[False, False]).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    top_ids = ranked.head(TOP_N)[cluster_col].tolist()
    output = {
        "radius": radius,
        "ranking": ranking_label,
        "min_sites_filter": min_sites,
        "clusters": [],
    }
    flat_rows = []

    print(f"  {ranking_label}: selected {len(top_ids)} clusters")
    for rank, cid in enumerate(top_ids, 1):
        cstats = ranked[ranked[cluster_col] == cid].iloc[0]
        sites_in_cluster = latest_site_snapshot_within_cluster(
            sub, cluster_col, cid, site_cols_for_cluster, SITE_KEY_COL
        )

        zips = (
            sorted([str(z).strip() for z in sites_in_cluster["zip"].dropna().unique() if str(z).strip()])
            if "zip" in sites_in_cluster
            else []
        )
        cities = sorted(sites_in_cluster["city"].dropna().unique().tolist()) if "city" in sites_in_cluster else []
        states = sorted(sites_in_cluster["state"].dropna().unique().tolist()) if "state" in sites_in_cluster else []

        site_list = []
        for _, s in sites_in_cluster.iterrows():
            site_rec = {
                "location_id": safe(s.get("location_id")),
                "site_client_id": safe(s.get("site_client_id")),
                "client_name": safe(s.get("client_id")),
                "address": safe(s.get("Address") or s.get("street")),
                "street": safe(s.get("street")),
                "city": safe(s.get("city")),
                "state": safe(s.get("state")),
                "zip": safe(s.get("zip")),
                "latitude": safe(s.get("latitude")),
                "longitude": safe(s.get("longitude")),
                "carwash_type": safe(s.get("primary_carwash_type")),
                "tunnel_count": safe(s.get("tunnel_count")),
                "age_years": safe(s.get("age_on_30_sep_25")),
                "competitors_4mi": safe(s.get("competitors_count_4miles")),
                "current_membership": safe(s.get("current_count")),
                "website": safe(s.get("official_website")),
                "region": safe(s.get("region")),
            }
            site_list.append(site_rec)
            flat_rows.append({
                "radius": radius,
                "ranking": ranking_label,
                "cluster_rank": rank,
                "cluster_id": int(cid),
                "cluster_median_daily": float(cstats["median_daily"]),
                "cluster_total_washes": float(cstats["total_washes"]),
                "cluster_n_sites": int(cstats["n_sites"]),
                **{k: v for k, v in site_rec.items()},
            })

        cluster_rec = {
            "rank": rank,
            "cluster_id": int(cid),
            "radius": radius,
            "ranking": ranking_label,
            "performance": {
                "median_daily_wash_count": float(cstats["median_daily"]),
                "mean_daily_wash_count": float(cstats["mean_daily"]),
                "total_washes_2yr": float(cstats["total_washes"]),
                "min_daily": float(cstats["min_daily"]),
                "max_daily": float(cstats["max_daily"]),
                "range_daily": float(cstats["range_daily"]),
                "p25_daily": float(cstats["p25"]),
                "p75_daily": float(cstats["p75"]),
                "iqr_daily": float(cstats["iqr_daily"]),
                "std_daily": float(cstats["std_daily"]),
                "site_day_records": int(cstats["site_days"]),
            },
            "geography": {
                "zip_codes": zips,
                "cities": cities,
                "states": states,
                "n_sites": int(cstats["n_sites"]),
                "centroid_lat": (
                    round(float(sites_in_cluster["latitude"].mean()), 6)
                    if ("latitude" in sites_in_cluster and not sites_in_cluster.empty)
                    else None
                ),
                "centroid_lon": (
                    round(float(sites_in_cluster["longitude"].mean()), 6)
                    if ("longitude" in sites_in_cluster and not sites_in_cluster.empty)
                    else None
                ),
            },
            "sites": site_list,
        }
        output["clusters"].append(cluster_rec)

        print(
            f"    #{rank:2d} Cluster {cid:>3d} | Median {cstats['median_daily']:>7.1f}"
            f" | Sites {int(cstats['n_sites']):>2d} | ZIPs {zips[:6]}"
        )

    json_path = RESULTS_DIR / f"{prefix}_details_{radius}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str, allow_nan=False)
    print(f"  → {json_path.name}")

    sites_json_path = RESULTS_DIR / f"{prefix}_sites_{radius}.json"
    with open(sites_json_path, "w") as f:
        json.dump(flat_rows, f, indent=2, default=str, allow_nan=False)
    print(f"  → {sites_json_path.name}")

    summary_rows = [
        {
            "rank": c["rank"],
            "cluster_id": c["cluster_id"],
            "radius": c["radius"],
            "ranking": c["ranking"],
            "n_sites": c["geography"]["n_sites"],
            "zip_codes": "|".join(str(z) for z in c["geography"]["zip_codes"]),
            "cities": "|".join(c["geography"]["cities"]),
            "states": "|".join(c["geography"]["states"]),
            "centroid_lat": c["geography"]["centroid_lat"],
            "centroid_lon": c["geography"]["centroid_lon"],
            **{f"perf_{k}": v for k, v in c["performance"].items()},
        }
        for c in output["clusters"]
    ]
    summary_json_path = RESULTS_DIR / f"{prefix}_summary_{radius}.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str, allow_nan=False)
    print(f"  → {summary_json_path.name}")


for ccol, radius in CLUSTER_COLS.items():
    print(f"\n{'='*60}")
    print(f"  Radius: {radius}")
    print(f"{'='*60}")

    sub = df[df[ccol] != -1].copy()
    if SITE_KEY_COL not in sub.columns:
        raise ValueError(f"Required site key column missing: {SITE_KEY_COL}")

    cluster_stats = build_cluster_stats(sub, ccol)

    export_ranked_outputs(
        sub=sub,
        cluster_col=ccol,
        radius=radius,
        cluster_stats=cluster_stats,
        prefix="top10",
        ranking_label="top10_elite_by_median_daily",
        min_sites=None,
    )
    export_ranked_outputs(
        sub=sub,
        cluster_col=ccol,
        radius=radius,
        cluster_stats=cluster_stats,
        prefix="top10_large",
        ranking_label=f"top10_large_by_median_daily_nsites_ge_{LARGE_MIN_SITES}",
        min_sites=LARGE_MIN_SITES,
    )

print("\nAll done!")
print(f"Results saved to: {RESULTS_DIR}")
