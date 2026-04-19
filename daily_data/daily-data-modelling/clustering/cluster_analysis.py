"""
Cluster Performance Analysis
-----------------------------
Trains historical wash data (2024-2025) onto DBSCAN clusters (6km, 12km, 18km).
Produces:
  - Daily / Monthly / 6-Monthly stats per cluster
  - Top-10 clusters by performance
  - Full feature profile per cluster
  - Rich visualisations saved to clustering/plots/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("CLUSTER_DATA_PATH", str(BASE.parent / "master_daily_with_site_metadata.csv")))
PLOTS_DIR = Path(os.getenv("CLUSTER_PLOTS_DIR", str(BASE / "plots")))
PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(os.getenv("CLUSTER_RESULTS_DIR", str(BASE / "results")))
RESULTS_DIR.mkdir(exist_ok=True)

PALETTE = "tab20"
TOP_N = 10

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["calendar_day"] = pd.to_datetime(df["calendar_day"])
df["month"] = df["calendar_day"].dt.to_period("M")
df["half_year"] = df["calendar_day"].dt.year.astype(str) + "-H" + (
    ((df["calendar_day"].dt.month - 1) // 6) + 1
).astype(str)

print(f"  Rows: {len(df):,}  |  Date range: {df['calendar_day'].min().date()} → {df['calendar_day'].max().date()}")

# ──────────────────────────────────────────────
# Feature groups
# ──────────────────────────────────────────────
WASH_FEATURES = [
    "wash_count_total", "wash_count_retail", "wash_count_membership",
    "wash_count_voucher", "last_week_same_day", "running_avg_7_days",
    "prev_wash_count",
]
WEATHER_FEATURES = [
    "weather_total_precipitation_mm", "weather_rainy_days",
    "weather_total_snowfall_cm", "weather_days_below_freezing",
    "weather_total_sunshine_hours", "weather_days_pleasant_temp",
    "weather_avg_daily_max_windspeed_ms",
]
SITE_FEATURES = [
    "nearest_gas_station_distance_miles", "nearest_gas_station_rating",
    "nearest_gas_station_rating_count", "competitors_count_4miles",
    "competitor_1_google_rating", "competitor_1_distance_miles",
    "competitor_1_rating_count", "other_grocery_count_1mile",
    "count_food_joints_0_5miles (0.5 mile)", "current_count",
    "age_on_30_sep_25", "tunnel_count",
]
ALL_NUMERIC_FEATURES = WASH_FEATURES + WEATHER_FEATURES + SITE_FEATURES

CLUSTER_COLS = ["dbscan_cluster_12km", "dbscan_cluster_18km"]

# ──────────────────────────────────────────────
# Helper: stats summary
# ──────────────────────────────────────────────
def cluster_stats(data, cluster_col, value_col="wash_count_total"):
    grp = data.groupby(cluster_col)[value_col]
    s = grp.agg(
        count="count",
        total="sum",
        mean="mean",
        median="median",
        std="std",
        min="min",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
        max="max",
    ).reset_index()
    s["range"] = s["max"] - s["min"]
    s["iqr"] = s["p75"] - s["p25"]
    return s.sort_values("median", ascending=False)


def top10(stats_df, cluster_col):
    return stats_df[stats_df[cluster_col] != -1].head(TOP_N)[cluster_col].tolist()


# ══════════════════════════════════════════════
# 1.  Per-cluster analysis for each radius
# ══════════════════════════════════════════════
all_top10 = {}

for ccol in CLUSTER_COLS:
    radius = ccol.split("_")[-1]
    print(f"\n{'='*60}")
    print(f"  Cluster radius: {radius}  ({df[df[ccol]!=-1][ccol].nunique()} clusters, "
          f"{(df[ccol]==-1).sum():,} noise rows excluded)")
    print(f"{'='*60}")

    sub = df[df[ccol] != -1].copy()

    # ── 1a. Daily stats
    daily_stats = cluster_stats(sub, ccol)
    daily_stats.to_json(RESULTS_DIR / f"daily_stats_{radius}.json", orient="records", date_format="iso", indent=2)
    print(f"  [Daily]  Top-5 clusters by median wash count:")
    print(daily_stats[[ccol, "median", "mean", "total", "min", "max", "range"]].head(5).to_string(index=False))

    # ── 1b. Monthly stats
    monthly = (
        sub.groupby([ccol, "month"])["wash_count_total"].sum().reset_index()
    )
    monthly_stats = cluster_stats(monthly, ccol, "wash_count_total")
    monthly_stats.to_json(RESULTS_DIR / f"monthly_stats_{radius}.json", orient="records", date_format="iso", indent=2)

    # ── 1c. Half-year stats
    half = (
        sub.groupby([ccol, "half_year"])["wash_count_total"].sum().reset_index()
    )
    half_stats = cluster_stats(half, ccol, "wash_count_total")
    half_stats.to_json(RESULTS_DIR / f"halfyear_stats_{radius}.json", orient="records", date_format="iso", indent=2)

    # ── Top-10
    t10 = top10(daily_stats, ccol)
    all_top10[ccol] = t10
    print(f"  Top-10 cluster IDs: {t10}")

    sub10 = sub[sub[ccol].isin(t10)].copy()
    sub10[ccol] = sub10[ccol].astype(str)

    # ── Feature profile (mean per cluster, all features)
    avail_feats = [f for f in ALL_NUMERIC_FEATURES if f in sub.columns]
    feat_profile = sub.groupby(ccol)[avail_feats].mean().reset_index()
    feat_profile.to_json(RESULTS_DIR / f"feature_profile_{radius}.json", orient="records", date_format="iso", indent=2)

    # ══════════════════════════════════════════
    #  VISUALISATIONS
    # ══════════════════════════════════════════

    # ── V1. Box plot: daily wash count – top 10 clusters
    fig, ax = plt.subplots(figsize=(14, 6))
    order = [str(c) for c in t10]
    sns.boxplot(
        data=sub10, x=ccol, y="wash_count_total",
        order=order, palette=PALETTE, ax=ax, fliersize=2,
    )
    ax.set_title(f"Daily Wash Count Distribution – Top 10 Clusters ({radius})", fontsize=14, pad=12)
    ax.set_xlabel("Cluster ID", fontsize=11)
    ax.set_ylabel("Daily Wash Count (total)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"boxplot_daily_{radius}.png", dpi=150)
    plt.close()

    # ── V2. Bar: median daily wash count – top 10
    fig, ax = plt.subplots(figsize=(12, 5))
    t10_daily = daily_stats[daily_stats[ccol].isin(t10)].set_index(ccol).loc[t10]
    colors = sns.color_palette(PALETTE, len(t10))
    bars = ax.bar(
        [str(c) for c in t10],
        t10_daily["median"].values,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.errorbar(
        range(len(t10)),
        t10_daily["median"].values,
        yerr=[
            t10_daily["median"].values - t10_daily["min"].values,
            t10_daily["max"].values - t10_daily["median"].values,
        ],
        fmt="none", color="grey", capsize=4, linewidth=1,
    )
    for bar, val in zip(bars, t10_daily["median"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Median Daily Wash Count – Top 10 Clusters ({radius})", fontsize=14, pad=12)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Median Wash Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"bar_median_daily_{radius}.png", dpi=150)
    plt.close()

    # ── V3. Monthly heatmap – top 10 clusters
    monthly_pivot = (
        monthly[monthly[ccol].isin(t10)]
        .assign(month_str=lambda x: x["month"].astype(str))
        .pivot_table(index=ccol, columns="month_str", values="wash_count_total", aggfunc="sum")
    )
    monthly_pivot.index = monthly_pivot.index.astype(str)
    fig, ax = plt.subplots(figsize=(max(16, len(monthly_pivot.columns) * 0.8), 6))
    sns.heatmap(
        monthly_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.3, ax=ax, cbar_kws={"label": "Monthly Wash Count"},
    )
    ax.set_title(f"Monthly Wash Count Heatmap – Top 10 Clusters ({radius})", fontsize=14, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cluster ID")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"heatmap_monthly_{radius}.png", dpi=150)
    plt.close()

    # ── V4. Half-year grouped bar – top 10
    half_t10 = half[half[ccol].isin(t10)].copy()
    half_t10[ccol] = half_t10[ccol].astype(str)
    half_pivot = half_t10.pivot_table(index="half_year", columns=ccol, values="wash_count_total", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(10, 5))
    half_pivot.plot(kind="bar", ax=ax, colormap=PALETTE, edgecolor="white", linewidth=0.4)
    ax.set_title(f"6-Monthly Wash Count – Top 10 Clusters ({radius})", fontsize=14, pad=12)
    ax.set_xlabel("Half-Year Period")
    ax.set_ylabel("Total Wash Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"bar_halfyear_{radius}.png", dpi=150)
    plt.close()

    # ── V5. Time series: rolling 30-day total per top-10 cluster
    ts = (
        sub10.groupby([ccol, "calendar_day"])["wash_count_total"]
        .sum()
        .reset_index()
        .sort_values("calendar_day")
    )
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = sns.color_palette(PALETTE, len(t10))
    for cid, color in zip(t10, colors):
        grp = ts[ts[ccol] == str(cid)].set_index("calendar_day")["wash_count_total"]
        rolled = grp.resample("D").sum().rolling(30, min_periods=7).mean()
        ax.plot(rolled.index, rolled.values, label=f"C{cid}", color=color, linewidth=1.4)
    ax.set_title(f"30-Day Rolling Avg Daily Wash Count – Top 10 Clusters ({radius})", fontsize=14, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Daily Wash Count (30d roll)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"timeseries_rolling_{radius}.png", dpi=150)
    plt.close()

    # ── V6. Wash-type breakdown (retail / membership / voucher) per top-10
    wtype_cols = ["wash_count_retail", "wash_count_membership", "wash_count_voucher"]
    avail_wtype = [c for c in wtype_cols if c in sub.columns]
    if avail_wtype:
        wtype = (
            sub10.groupby(ccol)[avail_wtype].mean().reset_index()
            .set_index(ccol)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        wtype.loc[[str(c) for c in t10 if str(c) in wtype.index]].plot(
            kind="bar", ax=ax, colormap="Set2", edgecolor="white"
        )
        ax.set_title(f"Avg Daily Wash Mix (Retail / Membership / Voucher) – Top 10 Clusters ({radius})", fontsize=13)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Avg Daily Count")
        ax.legend(["Retail", "Membership", "Voucher"], title="Type")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / f"bar_washmix_{radius}.png", dpi=150)
        plt.close()

    # ── V7. Feature heatmap (normalised) – top 10 clusters × all features
    fp_top10 = feat_profile[feat_profile[ccol].isin(t10)].set_index(ccol)
    fp_top10.index = fp_top10.index.astype(str)
    fp_norm = (fp_top10 - fp_top10.min()) / (fp_top10.max() - fp_top10.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(max(18, len(fp_norm.columns)), 5))
    sns.heatmap(
        fp_norm.T, annot=False, cmap="coolwarm", linewidths=0.2, ax=ax,
        cbar_kws={"label": "Normalised Value (0-1)"},
    )
    ax.set_title(f"Feature Profile – Top 10 Clusters ({radius})", fontsize=14, pad=12)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Feature")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"feature_heatmap_{radius}.png", dpi=150)
    plt.close()

    # ── V8. Geographic scatter – cluster locations coloured by performance
    geo = (
        sub.groupby(ccol)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"), median_wc=("wash_count_total", "median"))
        .reset_index()
    )
    geo["is_top10"] = geo[ccol].isin(t10)
    fig, ax = plt.subplots(figsize=(12, 7))
    sc_other = ax.scatter(
        geo.loc[~geo["is_top10"], "lon"],
        geo.loc[~geo["is_top10"], "lat"],
        c=geo.loc[~geo["is_top10"], "median_wc"],
        cmap="Blues", s=40, alpha=0.5, edgecolors="none", label="Other clusters",
    )
    sc_top = ax.scatter(
        geo.loc[geo["is_top10"], "lon"],
        geo.loc[geo["is_top10"], "lat"],
        c=geo.loc[geo["is_top10"], "median_wc"],
        cmap="Reds", s=200, alpha=0.9, edgecolors="black", linewidths=0.6,
        label="Top-10 clusters", zorder=5,
    )
    for _, row in geo[geo["is_top10"]].iterrows():
        ax.annotate(
            str(int(row[ccol])),
            (row["lon"], row["lat"]),
            textcoords="offset points", xytext=(5, 5), fontsize=7,
        )
    plt.colorbar(sc_top, ax=ax, label="Median Daily Wash Count")
    ax.set_title(f"Geographic Distribution of Clusters – {radius} (Top-10 highlighted)", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"geo_scatter_{radius}.png", dpi=150)
    plt.close()

    # ── V9. Stats summary table as image
    summary_df = daily_stats[daily_stats[ccol] != -1].head(TOP_N).copy()
    summary_df[ccol] = summary_df[ccol].astype(str)
    cols_show = [ccol, "count", "total", "median", "mean", "std", "min", "p25", "p75", "max", "range", "iqr"]
    summary_df = summary_df[cols_show].round(1)
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary_df.values,
        colLabels=[c.replace("_", " ").title() for c in cols_show],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
    ax.set_title(f"Top-10 Cluster Stats (Daily Wash Count) – {radius}", fontsize=13, pad=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"table_top10_{radius}.png", dpi=150)
    plt.close()

    print(f"  Plots saved for {radius}")

# ══════════════════════════════════════════════
# 2.  Cross-radius comparison (focus on wash count)
# ══════════════════════════════════════════════
print("\n Generating cross-radius comparison …")

comparison_rows = []
for ccol in CLUSTER_COLS:
    radius = ccol.split("_")[-1]
    sub = df[df[ccol] != -1].copy()
    s = cluster_stats(sub, ccol).head(1)
    comparison_rows.append({
        "radius": radius,
        "n_clusters": sub[ccol].nunique(),
        "best_cluster_id": s[ccol].values[0],
        "best_median_daily": s["median"].values[0],
        "best_total": s["total"].values[0],
        "avg_median_across_clusters": cluster_stats(sub, ccol)["median"].mean(),
    })

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_json(RESULTS_DIR / "cross_radius_comparison.json", orient="records", date_format="iso", indent=2)
print(comp_df.to_string(index=False))

# ══════════════════════════════════════════════
# 3.  Combined Top-10 summary (12km – primary)
# ══════════════════════════════════════════════
print("\n Generating combined summary (primary: 12km) …")
ccol = "dbscan_cluster_12km"
radius = "12km"
sub = df[df[ccol] != -1].copy()
sub[ccol] = sub[ccol].astype(int)

t10_ids = all_top10[ccol]
sub10 = sub[sub[ccol].isin(t10_ids)].copy()

# Monthly time series stacked area
monthly_ts = (
    sub10.groupby(["month", ccol])["wash_count_total"].sum()
    .reset_index()
    .assign(month_str=lambda x: x["month"].astype(str))
)
pivot = monthly_ts.pivot_table(index="month_str", columns=ccol, values="wash_count_total", aggfunc="sum").fillna(0)
pivot = pivot[[c for c in t10_ids if c in pivot.columns]]

fig, ax = plt.subplots(figsize=(16, 6))
colors = sns.color_palette(PALETTE, len(pivot.columns))
ax.stackplot(
    range(len(pivot)),
    [pivot[c].values for c in pivot.columns],
    labels=[f"C{c}" for c in pivot.columns],
    colors=colors, alpha=0.82,
)
ax.set_xticks(range(len(pivot)))
ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
ax.set_title("Monthly Wash Count – Top 10 Clusters (12km) – Stacked Area", fontsize=14, pad=12)
ax.set_ylabel("Total Wash Count")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "stacked_monthly_top10_12km.png", dpi=150)
plt.close()

# Radar / spider chart – top 5 clusters, normalised feature scores
radar_features = [
    "wash_count_total", "wash_count_membership", "wash_count_retail",
    "competitors_count_4miles", "weather_total_sunshine_hours",
    "current_count", "tunnel_count",
]
avail_radar = [f for f in radar_features if f in sub.columns]
t5 = t10_ids[:5]
radar_data = sub[sub[ccol].isin(t5)].groupby(ccol)[avail_radar].mean()
radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)

angles = np.linspace(0, 2 * np.pi, len(avail_radar), endpoint=False).tolist()
angles += angles[:1]
labels = [f.replace("_", " ").replace("wash count", "wc").replace("weather ", "").title() for f in avail_radar]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_r = sns.color_palette("Set1", len(t5))
for (cid, row), color in zip(radar_norm.iterrows(), colors_r):
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, color=color, linewidth=1.8, label=f"C{cid}")
    ax.fill(angles, vals, color=color, alpha=0.12)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=9)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7)
ax.set_title("Feature Radar – Top 5 Clusters (12km)", fontsize=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "radar_top5_12km.png", dpi=150)
plt.close()

print(f"\n  All plots saved to: {PLOTS_DIR}")
print(f"  All JSON results saved to: {RESULTS_DIR}")

# ══════════════════════════════════════════════
# 4.  Print final top-10 leaderboard (12km)
# ══════════════════════════════════════════════
print("\n" + "═"*60)
print("  TOP-10 PERFORMING CLUSTERS (12km radius)")
print("═"*60)
sub12 = df[df["dbscan_cluster_12km"] != -1].copy()
daily12 = cluster_stats(sub12, "dbscan_cluster_12km")
top10_df = daily12[daily12["dbscan_cluster_12km"] != -1].head(10)
for rank, (_, row) in enumerate(top10_df.iterrows(), 1):
    zips = df[df["dbscan_cluster_12km"] == row["dbscan_cluster_12km"]]["zip"].dropna().unique()[:5]
    cities = df[df["dbscan_cluster_12km"] == row["dbscan_cluster_12km"]]["city"].dropna().unique()[:3]
    print(f"  #{rank:2d}  Cluster {int(row['dbscan_cluster_12km']):>3d}"
          f"  │  Median: {row['median']:>7.1f}  │  Avg: {row['mean']:>7.1f}"
          f"  │  Max: {row['max']:>7.0f}  │  Total: {row['total']:>10,.0f}"
          f"  │  ZIPs: {list(zips)}  │  Cities: {list(cities)}")
print("═"*60)
print("\nDone!  Open the plots/ folder to explore all visualisations.")
