"""
Cluster Assignment Quality Backtest
-----------------------------------
Validates how well we can assign cluster IDs from lat/lon only
for unseen (test) data before putting behind an API.

Method:
1) Time split 80/20 (train/test)
2) Build train cluster centroids (lat/lon mean per cluster)
3) Assign each test row to nearest centroid by haversine distance
4) If nearest distance > radius_km, mark as unassigned (-1)
5) Evaluate assignment quality vs existing cluster label in test rows
6) Also evaluate wash-count range compatibility on assigned cluster

Outputs:
- results/cluster_assignment_quality_summary.json
- results/cluster_assignment_quality_12km.json
- results/cluster_assignment_quality_18km.json
- results/cluster_assignment_test_rows_12km.json
- results/cluster_assignment_test_rows_18km.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("CLUSTER_DATA_PATH", str(BASE.parent / "master_daily_with_site_metadata.csv")))
RESULTS_DIR = Path(os.getenv("CLUSTER_RESULTS_DIR", str(BASE / "results")))
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "wash_count_total"
SITE_KEY = "site_client_id"
CONFIGS = [
    ("dbscan_cluster_12km", "12km", 12.0),
    ("dbscan_cluster_18km", "18km", 18.0),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km.
    lat1/lon1: shape (N, 1), lat2/lon2: shape (1, M)
    """
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def assign_nearest_cluster(test_df, centroids_df, radius_km):
    tlat = test_df["latitude"].to_numpy().reshape(-1, 1)
    tlon = test_df["longitude"].to_numpy().reshape(-1, 1)
    clat = centroids_df["centroid_lat"].to_numpy().reshape(1, -1)
    clon = centroids_df["centroid_lon"].to_numpy().reshape(1, -1)

    dist = haversine_km(tlat, tlon, clat, clon)
    best_idx = np.argmin(dist, axis=1)
    best_dist = dist[np.arange(len(test_df)), best_idx]
    best_cluster = centroids_df["cluster_id"].to_numpy()[best_idx]

    assigned = np.where(best_dist <= radius_km, best_cluster, -1)
    return assigned.astype(int), best_dist


def build_train_range(train_df):
    return (
        train_df.groupby("cluster_id")[TARGET_COL]
        .agg(
            train_min="min",
            train_max="max",
            train_p10=lambda x: x.quantile(0.10),
            train_p25=lambda x: x.quantile(0.25),
            train_p75=lambda x: x.quantile(0.75),
            train_p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )


def main():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])

    # Minimal required columns
    req = {"calendar_day", "latitude", "longitude", TARGET_COL, SITE_KEY}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary_rows = []

    for cluster_col, radius_label, radius_km in CONFIGS:
        print(f"\n{'='*72}")
        print(f"Assignment quality for {radius_label} ({cluster_col})")
        print(f"{'='*72}")

        sub = df[
            (df[cluster_col] != -1)
            & df["latitude"].notna()
            & df["longitude"].notna()
            & df[TARGET_COL].notna()
        ].copy()
        sub = sub.sort_values("calendar_day").reset_index(drop=True)
        sub["cluster_id"] = sub[cluster_col].astype(int)

        split_idx = int(len(sub) * 0.80)
        train = sub.iloc[:split_idx].copy()
        test = sub.iloc[split_idx:].copy()
        split_date = test["calendar_day"].min().date()
        print(f"Rows: {len(sub):,} | Train: {len(train):,} | Test: {len(test):,} | Test starts: {split_date}")

        # Train centroids by cluster
        centroids = (
            train.groupby("cluster_id")
            .agg(
                centroid_lat=("latitude", "mean"),
                centroid_lon=("longitude", "mean"),
                train_cluster_rows=("cluster_id", "size"),
                train_cluster_sites=(SITE_KEY, "nunique"),
            )
            .reset_index()
        )

        # Assign test rows from lat/lon only
        pred_cluster, pred_dist_km = assign_nearest_cluster(test, centroids, radius_km)
        test_eval = test[
            ["calendar_day", SITE_KEY, "latitude", "longitude", "cluster_id", TARGET_COL]
        ].copy()
        test_eval["assigned_cluster_id"] = pred_cluster
        test_eval["assign_distance_km"] = pred_dist_km
        test_eval["is_assigned"] = (test_eval["assigned_cluster_id"] != -1).astype(int)
        test_eval["exact_cluster_match"] = (
            test_eval["assigned_cluster_id"] == test_eval["cluster_id"]
        ).astype(int)

        # Range compatibility using assigned cluster
        train_ranges = build_train_range(train)
        test_eval = test_eval.merge(
            train_ranges.add_prefix("assigned_"),
            left_on="assigned_cluster_id",
            right_on="assigned_cluster_id",
            how="left",
        )
        y = test_eval[TARGET_COL]
        test_eval["in_assigned_minmax"] = (
            (y >= test_eval["assigned_train_min"]) & (y <= test_eval["assigned_train_max"])
        ).astype(int)
        test_eval["in_assigned_p10_p90"] = (
            (y >= test_eval["assigned_train_p10"]) & (y <= test_eval["assigned_train_p90"])
        ).astype(int)
        test_eval["in_assigned_iqr"] = (
            (y >= test_eval["assigned_train_p25"]) & (y <= test_eval["assigned_train_p75"])
        ).astype(int)

        # Off-range size for p10-p90
        test_eval["off_below_p10"] = np.maximum(0, test_eval["assigned_train_p10"] - y)
        test_eval["off_above_p90"] = np.maximum(0, y - test_eval["assigned_train_p90"])
        test_eval["off_abs_p10_p90"] = test_eval["off_below_p10"] + test_eval["off_above_p90"]

        # Metrics
        coverage_rate = float(test_eval["is_assigned"].mean())
        exact_match_rate_all = float(test_eval["exact_cluster_match"].mean())
        assigned_only = test_eval[test_eval["is_assigned"] == 1].copy()
        exact_match_rate_assigned = (
            float(assigned_only["exact_cluster_match"].mean()) if len(assigned_only) else 0.0
        )

        in_range_assigned = (
            float(assigned_only["in_assigned_p10_p90"].mean()) if len(assigned_only) else 0.0
        )
        outside_assigned = assigned_only[assigned_only["in_assigned_p10_p90"] == 0]
        avg_off_when_outside = (
            float(outside_assigned["off_abs_p10_p90"].mean()) if len(outside_assigned) else 0.0
        )

        # Save detail rows
        out_cols = [
            "calendar_day",
            SITE_KEY,
            "latitude",
            "longitude",
            "cluster_id",
            "assigned_cluster_id",
            "assign_distance_km",
            "is_assigned",
            "exact_cluster_match",
            TARGET_COL,
            "assigned_train_p10",
            "assigned_train_p90",
            "in_assigned_p10_p90",
            "off_abs_p10_p90",
        ]
        test_eval[out_cols].to_json(
            RESULTS_DIR / f"cluster_assignment_test_rows_{radius_label}.json",
            orient="records",
            date_format="iso",
            indent=2,
        )

        payload = {
            "radius": radius_label,
            "cluster_col": cluster_col,
            "assign_method": "nearest_train_centroid_haversine_with_radius_gate",
            "radius_gate_km": radius_km,
            "split": {
                "type": "time_based_80_20",
                "test_start_date": str(split_date),
                "n_rows_total": int(len(sub)),
                "n_rows_train": int(len(train)),
                "n_rows_test": int(len(test)),
            },
            "assignment_quality": {
                "coverage_rate_assigned_not_noise": coverage_rate,
                "exact_match_rate_all_test_rows": exact_match_rate_all,
                "exact_match_rate_assigned_rows_only": exact_match_rate_assigned,
                "avg_assignment_distance_km_assigned": (
                    float(assigned_only["assign_distance_km"].mean()) if len(assigned_only) else None
                ),
            },
            "range_compatibility_on_assigned_cluster": {
                "in_p10_p90_rate": in_range_assigned,
                "avg_off_abs_when_outside_p10_p90": avg_off_when_outside,
            },
        }
        with open(RESULTS_DIR / f"cluster_assignment_quality_{radius_label}.json", "w") as f:
            json.dump(payload, f, indent=2)

        summary_rows.append(
            {
                "radius": radius_label,
                "cluster_col": cluster_col,
                "test_start_date": str(split_date),
                "coverage_rate_assigned_not_noise": coverage_rate,
                "exact_match_rate_all_test_rows": exact_match_rate_all,
                "exact_match_rate_assigned_rows_only": exact_match_rate_assigned,
                "avg_assignment_distance_km_assigned": (
                    float(assigned_only["assign_distance_km"].mean()) if len(assigned_only) else np.nan
                ),
                "in_p10_p90_rate_on_assigned_cluster": in_range_assigned,
                "avg_off_abs_when_outside_p10_p90_on_assigned_cluster": avg_off_when_outside,
            }
        )

        print(
            f"Coverage={coverage_rate:.3f}, Exact(all)={exact_match_rate_all:.3f}, "
            f"Exact(assigned)={exact_match_rate_assigned:.3f}, "
            f"InRange(p10-p90 on assigned)={in_range_assigned:.3f}"
        )

    pd.DataFrame(summary_rows).to_json(
        RESULTS_DIR / "cluster_assignment_quality_summary.json", orient="records", date_format="iso", indent=2
    )
    print("\nSaved: results/cluster_assignment_quality_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()

