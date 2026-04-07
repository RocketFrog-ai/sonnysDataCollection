"""
Extended EDA for modelling2/finale.csv — additional plots + summary stats.

Run:  python eda_extended.py

Outputs: eda_output/extended/*.png and eda_output/extended/eda_summary.txt
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = ROOT / "finale.csv"
OUT_DIR = ROOT / "eda_output" / "extended"
TIER_SPLITS = [19, 31, 31, 19]


def wash_quantile_labels(s: pd.Series, splits: list[int]) -> pd.Series:
    cum = np.cumsum([0] + list(splits))
    bounds = np.percentile(s.dropna(), cum)
    bounds[0] = float(s.min())
    bounds[-1] = float(s.max())
    return pd.cut(
        s,
        bins=bounds,
        labels=[f"Q{i}" for i in range(1, len(splits) + 1)],
        include_lowest=True,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    cur = pd.to_numeric(df["current_count"], errors="coerce")
    prev = pd.to_numeric(df["previous_count"], errors="coerce")
    ratio = cur / prev.replace(0, np.nan)
    delta = cur - prev
    pct_change = (cur - prev) / prev.replace(0, np.nan) * 100.0

    # --- YoY dynamics ---
    fig, ax = plt.subplots(figsize=(8, 5))
    r_clip = ratio.clip(0.3, 3.0)
    sns.histplot(r_clip.dropna(), kde=True, ax=ax, color="#2c5282")
    ax.set_title("YoY volume ratio (current ÷ previous), clipped to [0.3, 3] for display")
    ax.set_xlabel("ratio")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "10_hist_yoy_ratio_clipped.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(pct_change.clip(-50, 80).dropna(), kde=True, ax=ax, color="#276749")
    ax.set_title("YoY % change vs previous, clipped to [-50%, 80%] for display")
    ax.set_xlabel("% change")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "11_hist_yoy_pct_change_clipped.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(delta.dropna(), kde=True, ax=ax, color="#744210")
    ax.set_title("Absolute change (current − previous)")
    ax.set_xlabel("wash count delta")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "12_hist_yoy_delta.png", dpi=150)
    plt.close(fig)

    # --- By region (string) ---
    if "region" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        order = df.groupby("region")["current_count"].median().sort_values(ascending=False).index
        sns.boxplot(
            data=df,
            x="region",
            y="current_count",
            order=order,
            ax=ax,
            hue="region",
            palette="Set2",
            legend=False,
        )
        ax.tick_params(axis="x", rotation=25)
        ax.set_title("current_count by region (median sort)")
        ax.set_ylabel("current_count")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "13_box_current_by_region.png", dpi=150)
        plt.close(fig)

    # --- By car wash type ---
    if "primary_carwash_type" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ct = df.copy()
        ct["primary_carwash_type"] = ct["primary_carwash_type"].fillna("Unknown")
        order = ct.groupby("primary_carwash_type")["current_count"].median().sort_values(ascending=False).index
        sns.boxplot(
            data=ct,
            x="primary_carwash_type",
            y="current_count",
            order=order,
            ax=ax,
            hue="primary_carwash_type",
            palette="Pastel1",
            legend=False,
        )
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("current_count by primary_carwash_type")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "14_box_current_by_carwash_type.png", dpi=150)
        plt.close(fig)

    # --- Volume tier (same splits as benchmark) ---
    df["_wash_q"] = wash_quantile_labels(cur, TIER_SPLITS)
    fig, ax = plt.subplots(figsize=(8, 5))
    wq = df.dropna(subset=["_wash_q"])
    sns.boxplot(
        data=wq, x="_wash_q", y="age_on_30_sep_25", ax=ax, hue="_wash_q", palette="magma", legend=False
    )
    ax.set_title("Site age by wash-volume tier (current_count quartiles)")
    ax.set_xlabel("tier")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "15_box_age_by_volume_tier.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df.dropna(subset=["_wash_q"]), x="_wash_q", ax=ax, hue="_wash_q", palette="viridis", legend=False
    )
    ax.set_title("Site count per volume tier")
    ax.set_xlabel("tier")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "16_count_sites_per_tier.png", dpi=150)
    plt.close(fig)

    # --- Competitors vs volume ---
    if "competitors_count_4miles" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        cc = pd.to_numeric(df["competitors_count_4miles"], errors="coerce")
        sns.scatterplot(x=cc, y=cur, alpha=0.45, hue=df["region"] if "region" in df.columns else None, ax=ax)
        ax.set_title("current_count vs competitors within 4 miles")
        ax.set_xlabel("competitors_count_4miles")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "17_scatter_competitors_vs_current.png", dpi=150)
        plt.close(fig)

    # --- Top correlations with current_count ---
    num = df.select_dtypes(include=[np.number]).copy()
    if "tunnel_count" in num.columns:
        num = num.drop(columns=["tunnel_count"])
    cor = num.corr(numeric_only=True)["current_count"].drop("current_count", errors="ignore").dropna()
    cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
    top = cor.head(18)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#c53030" if v < 0 else "#2b6cb0" for v in top.values]
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Strongest linear correlations with current_count (numeric features)")
    ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "18_bar_corr_with_current_count.png", dpi=150)
    plt.close(fig)

    # --- ECDF current ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.sort(cur.dropna())
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color="#553c9a", lw=2)
    ax.set_xlabel("current_count")
    ax.set_ylabel("F(x)")
    ax.set_title("Empirical CDF of current_count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "19_ecdf_current_count.png", dpi=150)
    plt.close(fig)

    # --- Naive residual (current - previous) vs previous ---
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=prev, y=delta, alpha=0.4, ax=ax, color="#dd6b20")
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("previous_count")
    ax.set_ylabel("current − previous")
    ax.set_title("Change vs previous volume")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "20_scatter_delta_vs_previous.png", dpi=150)
    plt.close(fig)

    # --- Pair plot (compact set) ---
    pair_cols = [
        c
        for c in [
            "current_count",
            "previous_count",
            "age_on_30_sep_25",
            "competitors_count_4miles",
            "nearest_gas_station_distance_miles",
        ]
        if c in df.columns
    ]
    if len(pair_cols) >= 3:
        pp = df[pair_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pp) > 400:
            pp = pp.sample(400, random_state=42)
        g = sns.pairplot(pp, corner=True, plot_kws={"alpha": 0.35, "s": 12}, diag_kind="kde")
        g.fig.suptitle("Pair plot (sample ≤400 rows)", y=1.02)
        g.savefig(OUT_DIR / "21_pairplot_key_numeric.png", dpi=120)
        plt.close("all")

    # --- Summary text ---
    lines = [
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "current_count:",
        cur.describe().to_string(),
        "",
        "previous_count:",
        prev.describe().to_string(),
        "",
        "YoY ratio (current/previous):",
        ratio.describe().to_string(),
        "",
        "Categorical cardinalities:",
    ]
    for col in ["region", "state", "primary_carwash_type", "_match_type"]:
        if col in df.columns:
            lines.append(f"  {col}: {df[col].nunique()} unique")
    lines.append("")
    lines.append("Volume tier counts:")
    lines.append(df["_wash_q"].value_counts().sort_index().to_string())
    (OUT_DIR / "eda_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
