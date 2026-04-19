"""
Cluster Range Membership Backtest
---------------------------------
Checks whether test-period wash counts belong to expected train-time cluster ranges.

Evaluates three granularities:
  - daily
  - monthly (cluster-month total wash count)
  - halfyear (cluster-halfyear total wash count)

Outputs:
  - results/range_backtest_summary.json
  - results/range_backtest_{radius}.json
  - results/range_membership_daily_{radius}.json
  - results/range_membership_monthly_{radius}.json
  - results/range_membership_halfyear_{radius}.json
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
CLUSTER_COLS = {
    "dbscan_cluster_12km": "12km",
    "dbscan_cluster_18km": "18km",
}


def add_range_columns(g, value_col):
    out = (
        g.groupby("cluster_id")[value_col]
        .agg(
            train_count="count",
            train_min="min",
            train_max="max",
            train_p10=lambda x: x.quantile(0.10),
            train_p25=lambda x: x.quantile(0.25),
            train_median="median",
            train_p75=lambda x: x.quantile(0.75),
            train_p90=lambda x: x.quantile(0.90),
        )
        .reset_index()
    )
    return out


def evaluate_membership(test_df, ranges_df, value_col):
    m = test_df.merge(ranges_df, on="cluster_id", how="left")
    y = m[value_col]

    m["in_minmax"] = ((y >= m["train_min"]) & (y <= m["train_max"])).astype(int)
    m["in_p10_p90"] = ((y >= m["train_p10"]) & (y <= m["train_p90"])).astype(int)
    m["in_iqr"] = ((y >= m["train_p25"]) & (y <= m["train_p75"])).astype(int)

    # Distance from expected operating range (p10-p90)
    m["off_below_p10"] = np.maximum(0, m["train_p10"] - y)
    m["off_above_p90"] = np.maximum(0, y - m["train_p90"])
    m["off_abs_p10_p90"] = m["off_below_p10"] + m["off_above_p90"]

    range_width = (m["train_p90"] - m["train_p10"]).replace(0, np.nan)
    m["off_pct_of_range_p10_p90"] = m["off_abs_p10_p90"] / range_width
    m["is_outside_p10_p90"] = (m["off_abs_p10_p90"] > 0).astype(int)
    return m


def period_summary(eval_df):
    outside = eval_df[eval_df["is_outside_p10_p90"] == 1]
    return {
        "rows_test": int(len(eval_df)),
        "in_minmax_rate": float(eval_df["in_minmax"].mean()),
        "in_p10_p90_rate": float(eval_df["in_p10_p90"].mean()),
        "in_iqr_rate": float(eval_df["in_iqr"].mean()),
        "outside_p10_p90_rate": float(eval_df["is_outside_p10_p90"].mean()),
        "avg_off_abs_p10_p90_all_rows": float(eval_df["off_abs_p10_p90"].mean()),
        "avg_off_abs_p10_p90_when_outside": float(outside["off_abs_p10_p90"].mean()) if len(outside) else 0.0,
        "median_off_abs_p10_p90_when_outside": float(outside["off_abs_p10_p90"].median()) if len(outside) else 0.0,
        "avg_off_pct_of_range_when_outside": float(outside["off_pct_of_range_p10_p90"].mean()) if len(outside) else 0.0,
    }


def main():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])
    df["month"] = df["calendar_day"].dt.to_period("M").astype(str)
    df["half_year"] = df["calendar_day"].dt.year.astype(str) + "-H" + (
        ((df["calendar_day"].dt.month - 1) // 6) + 1
    ).astype(str)

    df = df[df[TARGET_COL].notna()].copy()

    summary_rows = []

    for cluster_col, radius in CLUSTER_COLS.items():
        print(f"\n{'='*68}")
        print(f"Range backtest for {radius} ({cluster_col})")
        print(f"{'='*68}")

        sub = df[df[cluster_col] != -1].copy()
        sub = sub.sort_values("calendar_day").reset_index(drop=True)
        sub["cluster_id"] = sub[cluster_col].astype(int)

        split_idx = int(len(sub) * 0.80)
        train = sub.iloc[:split_idx].copy()
        test = sub.iloc[split_idx:].copy()
        split_date = test["calendar_day"].min().date()

        print(f"Rows: {len(sub):,} | Train: {len(train):,} | Test: {len(test):,} | Test starts: {split_date}")

        # Daily
        train_daily = train[["cluster_id", TARGET_COL]].copy()
        test_daily = test[["calendar_day", "cluster_id", TARGET_COL]].copy()
        daily_ranges = add_range_columns(train_daily, TARGET_COL)
        eval_daily = evaluate_membership(test_daily, daily_ranges, TARGET_COL)
        daily_summary = period_summary(eval_daily)

        # Monthly totals per cluster
        train_monthly = (
            train.groupby(["cluster_id", "month"])[TARGET_COL]
            .sum()
            .reset_index(name="wash_count_monthly")
        )
        test_monthly = (
            test.groupby(["cluster_id", "month"])[TARGET_COL]
            .sum()
            .reset_index(name="wash_count_monthly")
        )
        monthly_ranges = add_range_columns(train_monthly, "wash_count_monthly")
        eval_monthly = evaluate_membership(test_monthly, monthly_ranges, "wash_count_monthly")
        monthly_summary = period_summary(eval_monthly)

        # Half-year totals per cluster
        train_half = (
            train.groupby(["cluster_id", "half_year"])[TARGET_COL]
            .sum()
            .reset_index(name="wash_count_halfyear")
        )
        test_half = (
            test.groupby(["cluster_id", "half_year"])[TARGET_COL]
            .sum()
            .reset_index(name="wash_count_halfyear")
        )
        half_ranges = add_range_columns(train_half, "wash_count_halfyear")
        eval_half = evaluate_membership(test_half, half_ranges, "wash_count_halfyear")
        half_summary = period_summary(eval_half)

        eval_daily.to_json(
            RESULTS_DIR / f"range_membership_daily_{radius}.json", orient="records", date_format="iso", indent=2
        )
        eval_monthly.to_json(
            RESULTS_DIR / f"range_membership_monthly_{radius}.json", orient="records", date_format="iso", indent=2
        )
        eval_half.to_json(
            RESULTS_DIR / f"range_membership_halfyear_{radius}.json", orient="records", date_format="iso", indent=2
        )

        payload = {
            "radius": radius,
            "cluster_col": cluster_col,
            "split": {
                "type": "time_based_80_20",
                "test_start_date": str(split_date),
                "n_rows_total": int(len(sub)),
                "n_rows_train": int(len(train)),
                "n_rows_test": int(len(test)),
            },
            "daily": daily_summary,
            "monthly": monthly_summary,
            "halfyear": half_summary,
        }
        with open(RESULTS_DIR / f"range_backtest_{radius}.json", "w") as f:
            json.dump(payload, f, indent=2)

        print(
            f"Daily in p10-p90: {daily_summary['in_p10_p90_rate']:.3f}, "
            f"Monthly: {monthly_summary['in_p10_p90_rate']:.3f}, "
            f"Halfyear: {half_summary['in_p10_p90_rate']:.3f}"
        )

        summary_rows.append(
            {
                "radius": radius,
                "cluster_col": cluster_col,
                "test_start_date": str(split_date),
                "daily_in_p10_p90_rate": daily_summary["in_p10_p90_rate"],
                "daily_avg_off_when_outside": daily_summary["avg_off_abs_p10_p90_when_outside"],
                "monthly_in_p10_p90_rate": monthly_summary["in_p10_p90_rate"],
                "monthly_avg_off_when_outside": monthly_summary["avg_off_abs_p10_p90_when_outside"],
                "halfyear_in_p10_p90_rate": half_summary["in_p10_p90_rate"],
                "halfyear_avg_off_when_outside": half_summary["avg_off_abs_p10_p90_when_outside"],
            }
        )

    pd.DataFrame(summary_rows).to_json(
        RESULTS_DIR / "range_backtest_summary.json", orient="records", date_format="iso", indent=2
    )
    print("\nSaved: results/range_backtest_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
