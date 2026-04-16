#!/usr/bin/env python3
"""
Additional region-focused EDA plots written to plots_2.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
CSV_DEFAULT = REPO / "app" / "modelling" / "ds" / "datasets" / "master_daily_with_site_metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "plots_2"
WEEKEND = {"Saturday", "Sunday"}
REGION_ORDER = ["South", "Midwest", "West", "Northeast"]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    df["wash_count_total"] = pd.to_numeric(df["wash_count_total"], errors="coerce")
    df = df.dropna(subset=["calendar_day", "region", "wash_count_total"]).copy()
    # Keep realistic observations for stable trend/distribution visuals.
    df = df.loc[df["wash_count_total"] >= 0].copy()
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND)
    df["month_name"] = df["calendar_day"].dt.strftime("%b")
    return df


def fig_save(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def month_order() -> list[str]:
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_region_monthly_seasonality_heatmap(df: pd.DataFrame):
    d = (
        df.groupby(["region", "month_name"], as_index=False)["wash_count_total"]
        .mean()
        .pivot(index="region", columns="month_name", values="wash_count_total")
        .reindex(index=REGION_ORDER)
    )
    d = d.reindex(columns=month_order())
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(d, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.4, cbar_kws={"label": "Mean wash_count_total"}, ax=ax)
    ax.set_title("Region-wise monthly seasonality (mean washes per site-day)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Region")
    fig_save(fig, "01_region_monthly_seasonality_heatmap.png")


def plot_region_weekly_yoy(df: pd.DataFrame):
    d = df.copy()
    iso = d["calendar_day"].dt.isocalendar()
    d["iso_week"] = iso.week.astype(int)
    d["iso_year"] = iso.year.astype(int)

    wk = (
        d.groupby(["region", "iso_year", "iso_week"], as_index=False)["wash_count_total"]
        .mean()
        .sort_values(["region", "iso_year", "iso_week"])
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, region in enumerate(REGION_ORDER):
        ax = axes[i]
        sub = wk[wk["region"] == region]
        for yr, clr in [(2024, "#2b6cb0"), (2025, "#dd6b20")]:
            ys = sub[sub["iso_year"] == yr]
            if ys.empty:
                continue
            ax.plot(ys["iso_week"], ys["wash_count_total"], lw=1.8, color=clr, label=str(yr))
        ax.set_title(region)
        ax.grid(alpha=0.25)
        if i >= 2:
            ax.set_xlabel("ISO week")
        if i % 2 == 0:
            ax.set_ylabel("Mean wash_count_total")
        ax.set_xlim(1, 53)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Weekly trend by region: 2024 vs 2025 (YoY view)", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_save(fig, "02_region_weekly_yoy_2024_vs_2025.png")


def plot_region_distribution_boxplot(df: pd.DataFrame):
    cap = df["wash_count_total"].quantile(0.99)
    d = df.copy()
    d["wash_count_capped"] = np.minimum(d["wash_count_total"], cap)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.boxplot(
        data=d,
        x="region",
        y="wash_count_capped",
        order=REGION_ORDER,
        showfliers=False,
        color="#8ecae6",
        ax=ax,
    )
    ax.set_title("Wash-count distribution by region (capped at p99)")
    ax.set_xlabel("Region")
    ax.set_ylabel("wash_count_total (p99-capped)")
    ax.grid(alpha=0.2, axis="y")
    fig_save(fig, "03_region_distribution_boxplot_p99_capped.png")


def plot_weekend_uplift_heatmap(df: pd.DataFrame):
    d = (
        df.groupby(["region", "month_name", "is_weekend"], as_index=False)["wash_count_total"]
        .mean()
    )
    wd = d[d["is_weekend"] == False].rename(columns={"wash_count_total": "weekday_mean"})
    we = d[d["is_weekend"] == True].rename(columns={"wash_count_total": "weekend_mean"})
    m = wd.merge(we, on=["region", "month_name"], how="inner")
    m["uplift_pct"] = (m["weekend_mean"] - m["weekday_mean"]) / m["weekday_mean"] * 100
    pv = (
        m.pivot(index="region", columns="month_name", values="uplift_pct")
        .reindex(index=REGION_ORDER)
        .reindex(columns=month_order())
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        pv,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.4,
        cbar_kws={"label": "Weekend uplift % vs weekday"},
        ax=ax,
    )
    ax.set_title("Weekend uplift by region and month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Region")
    fig_save(fig, "04_weekend_uplift_pct_region_month_heatmap.png")


def main():
    csv_path = Path(os.environ.get("CARWASH_CSV", str(CSV_DEFAULT)))
    if not csv_path.exists():
        alt = Path(__file__).resolve().parent / "master_daily_with_site_metadata.csv"
        if alt.exists():
            csv_path = alt
        else:
            raise SystemExit(f"CSV not found: {csv_path}")

    sns.set_theme(style="whitegrid", context="notebook")
    df = load_data(csv_path)
    plot_region_monthly_seasonality_heatmap(df)
    plot_region_weekly_yoy(df)
    plot_region_distribution_boxplot(df)
    plot_weekend_uplift_heatmap(df)
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
