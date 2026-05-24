"""Regenerate ALL hypothesis-testing charts with chem-source sites removed.

Drops every row where source == 'chem' in less_than-2yrs.csv, then re-runs:
  - the two/three/four-body interaction plots + aggregate plots
  - the backtesting localisation plots
  - the 5 insight plots (age cliff, membership indicator, ramp, donut, rotation)

Outputs go to interaction_outputs_no_chem/ — the original interaction_outputs/ is untouched.
Re-run: python run_no_chem_analysis.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "interaction_outputs_no_chem"
NO_CHEM_DATA = BASE / "_no_chem_data"


def build_chem_free_data_dir() -> Path:
    """Write a data dir whose less_than-2yrs.csv has all source=='chem' rows removed."""
    (NO_CHEM_DATA / "data").mkdir(parents=True, exist_ok=True)

    lt2 = pd.read_csv(BASE / "less_than-2yrs.csv", low_memory=False)
    before = lt2["client_id_location_id"].nunique()
    lt2 = lt2[lt2["source"] != "chem"].copy()
    after = lt2["client_id_location_id"].nunique()
    lt2.to_csv(NO_CHEM_DATA / "less_than-2yrs.csv", index=False)

    shutil.copy(BASE / "more_than-2yrs_monthly.csv", NO_CHEM_DATA / "more_than-2yrs_monthly.csv")
    if (BASE / "backtesting.xlsx").exists():
        shutil.copy(BASE / "backtesting.xlsx", NO_CHEM_DATA / "backtesting.xlsx")
    loc = BASE / "data" / "backtesting_localisation.tsv"
    if loc.exists():
        shutil.copy(loc, NO_CHEM_DATA / "data" / "backtesting_localisation.tsv")

    print(f"[chem filter] less_than-2yrs.csv sites: {before} -> {after} (removed {before - after} chem sites)")
    return NO_CHEM_DATA


def run_legacy_plots(data_dir: Path) -> None:
    """Re-run interaction + backtesting orchestration against the chem-free data dir."""
    import backtesting_analysis as B
    import run_site_interaction_plots as R

    B.DATA_DIR = data_dir
    B.OUT_DIR = OUT / "backtesting"
    B.PLOTS_DIR = B.OUT_DIR / "plots"
    B.DATA_OUT_DIR = B.OUT_DIR / "data"

    R.DATA_DIR = data_dir
    R.OUT_DIR = OUT

    R.main()
    print(f"[legacy] interaction + backtesting plots -> {OUT}")


# --------------------------------------------------------------------------------------
# Insight plots (chem-free). Generated AFTER curate_outputs so they are not deleted.
# --------------------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _load_insight_frames(data_dir: Path):
    new = pd.read_csv(data_dir / "less_than-2yrs.csv", low_memory=False)
    old = pd.read_csv(data_dir / "more_than-2yrs_monthly.csv", low_memory=False)
    for df in (new, old):
        df["year_month_dt"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")
    new = new[(new["year_month_dt"] >= "2024-01-01") & (new["year_month_dt"] <= "2025-12-01")].copy()
    old = old[(old["year_month_dt"] >= "2024-01-01") & (old["year_month_dt"] <= "2025-12-01")].copy()
    return new, old


def generate_insight_plots(data_dir: Path, out_dir: Path) -> None:
    new, old = _load_insight_frames(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Plot 1: Age lifecycle cliff (OLD only — unaffected by chem filter) ----
    ann = old.groupby(["client_id_location_id", "age_on_30_sep_25", old["year_month_dt"].dt.year])[
        "wash_count_total"
    ].mean().unstack(level=2)
    ann.columns = ["a2024", "a2025"]
    ann = ann.reset_index()
    ann["yoy"] = (ann["a2025"] / ann["a2024"] - 1) * 100
    ann = ann.merge(
        old[["client_id_location_id", "primary_carwash_type"]].drop_duplicates(),
        on="client_id_location_id",
    )
    counts = ann.groupby("age_on_30_sep_25").size()
    age_pts = ann.groupby("age_on_30_sep_25")["yoy"].median().reset_index()
    age_pts = age_pts[age_pts["age_on_30_sep_25"].map(counts) >= 5]
    exp = ann[ann["primary_carwash_type"] == "Express Tunnel"]
    exp_counts = exp.groupby("age_on_30_sep_25").size()
    exp_pts = exp.groupby("age_on_30_sep_25")["yoy"].median().reset_index()
    exp_pts = exp_pts[exp_pts["age_on_30_sep_25"].map(exp_counts) >= 5]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(age_pts["age_on_30_sep_25"], age_pts["yoy"], "o-", label="All formats", color="#1f77b4", lw=2, ms=7)
    ax.plot(exp_pts["age_on_30_sep_25"], exp_pts["yoy"], "s-", label="Express Tunnel only", color="#d62728", lw=2, ms=6, alpha=0.85)
    ax.axvspan(3.5, 5.5, color="orange", alpha=0.10, label="Tipping zone (age 4-5)")
    ax.set_xlabel("Site age (years, as of 30 Sep 2025)")
    ax.set_ylabel("Median YoY wash-count growth (2024 -> 2025), %")
    ax.set_title("Wash-volume lifecycle: a sharp growth->decay flip near year 4-5 [chem-free]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "age_lifecycle_cliff.png", dpi=140)
    plt.close(fig)

    # ---- Plot 2: Membership-share Q1-2024 leading indicator (OLD only) ----
    old = old.copy()
    old["m_share"] = old["wash_count_membership"] / old["wash_count_total"].replace(0, np.nan)
    early = old[old["year_month_dt"] <= "2024-03-01"].groupby("client_id_location_id")["m_share"].mean().rename("m_q1")
    ann2 = old.groupby(["client_id_location_id", old["year_month_dt"].dt.year])["wash_count_total"].mean().unstack()
    ann2.columns = ["a2024", "a2025"]
    ann2["yoy"] = (ann2["a2025"] / ann2["a2024"] - 1) * 100
    df = ann2.join(early).dropna()
    df["decile"] = pd.qcut(df["m_q1"], 10, labels=False, duplicates="drop") + 1
    gb = df.groupby("decile").agg(vol=("a2024", "median"), yoy=("yoy", "median")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(gb["decile"], gb["vol"], color="#2ca02c", alpha=0.8)
    axes[0].set_xlabel("Q1-2024 membership-share decile (1=lowest, 10=highest)")
    axes[0].set_ylabel("Median monthly washes (2024)")
    axes[0].set_title("Early membership share predicts VOLUME")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(gb["decile"], gb["yoy"], color="#9467bd", alpha=0.85)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("Q1-2024 membership-share decile")
    axes[1].set_ylabel("Median YoY growth 2024->2025 (%)")
    axes[1].set_title("... and predicts GROWTH a year out")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Membership share in early life is a powerful leading indicator [chem-free]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "membership_leading_indicator.png", dpi=140)
    plt.close(fig)

    # ---- Plot 3: New-site ramp, multi vs single (NEW sites with op_start) ----
    sub = new[new["operational_start_date"].notna()].copy()
    sub["op"] = pd.to_datetime(sub["operational_start_date"])
    sub["mso"] = (
        (sub["year_month_dt"].dt.year - sub["op"].dt.year) * 12
        + (sub["year_month_dt"].dt.month - sub["op"].dt.month)
    )
    sub = sub[(sub["mso"] >= 0) & (sub["mso"] <= 11)].copy()
    sub["m_share"] = sub["wash_count_membership"] / sub["wash_count_total"].replace(0, np.nan)
    ramp = sub.groupby(["client_type", "mso"]).agg(
        total=("wash_count_total", "median"),
        m_share=("m_share", "median"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ct, color in [("multi_site", "#d62728"), ("single_site", "#1f77b4")]:
        s = ramp[ramp["client_type"] == ct]
        axes[0].plot(s["mso"], s["total"], "o-", label=ct.replace("_", " "), color=color, lw=2)
        axes[1].plot(s["mso"], s["m_share"] * 100, "o-", label=ct.replace("_", " "), color=color, lw=2)
    axes[0].set_xlabel("Months since opening")
    axes[0].set_ylabel("Median monthly washes")
    axes[0].set_title("Total washes per month")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Months since opening")
    axes[1].set_ylabel("Median membership share (%)")
    axes[1].set_title("Membership share")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.suptitle("New-site ramp: multi-site vs single-site [chem-free]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_vs_single_ramp.png", dpi=140)
    plt.close(fig)

    # ---- Plot 4: Cannibalization donut (OLD YoY vs distance to nearest NEW entrant) ----
    old_sites = old[["client_id_location_id", "latitude", "longitude"]].dropna().drop_duplicates()
    new_sites = new[["client_id_location_id", "latitude", "longitude"]].dropna().drop_duplicates()
    oa = old_sites[["latitude", "longitude"]].values
    na = new_sites[["latitude", "longitude"]].values
    min_dist = [
        _haversine_km(oa[i, 0], oa[i, 1], na[:, 0], na[:, 1]).min() for i in range(len(oa))
    ]
    old_sites = old_sites.assign(min_dist_to_new_km=min_dist)

    od = old.copy()
    od["yr"] = od["year_month_dt"].dt.year
    a = od.groupby(["client_id_location_id", "yr"])["wash_count_total"].mean().unstack()
    a.columns = ["avg_2024", "avg_2025"]
    a = a.dropna()
    a["yoy"] = (a["avg_2025"] / a["avg_2024"] - 1) * 100
    a = a.merge(old_sites[["client_id_location_id", "min_dist_to_new_km"]], on="client_id_location_id")

    def db(d):
        if d < 1:
            return "<1 km"
        if d < 2:
            return "1-2 km"
        if d < 3:
            return "2-3 km"
        if d < 5:
            return "3-5 km"
        if d < 10:
            return "5-10 km"
        if d < 25:
            return "10-25 km"
        return "25+ km"

    order = ["<1 km", "1-2 km", "2-3 km", "3-5 km", "5-10 km", "10-25 km", "25+ km"]
    a["bin"] = a["min_dist_to_new_km"].apply(db)
    g = a.groupby("bin").agg(
        median_yoy=("yoy", "median"),
        pct_declining=("yoy", lambda x: (x < 0).mean() * 100),
    ).reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = ["#7fbf7b" if v >= 0 else "#d73027" for v in g["median_yoy"].fillna(0)]
    axes[0].bar(order, g["median_yoy"], color=colors, alpha=0.85)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Median YoY wash growth, 2024->2025 (%)")
    axes[0].set_title("OLD-site growth vs distance to nearest NEW entrant")
    axes[0].axvspan(0.5, 3.5, color="red", alpha=0.10)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(order, g["pct_declining"], color="#fc8d59", alpha=0.85)
    axes[1].axhline(50, color="black", lw=0.8, ls="--", label="50% line")
    axes[1].set_ylabel("% of OLD sites with negative YoY growth")
    axes[1].set_title("Share of OLD sites that declined")
    axes[1].axvspan(0.5, 3.5, color="red", alpha=0.10)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Cannibalization donut: peak damage at 2-5 km [chem-free]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "cannibalization_donut.png", dpi=140)
    plt.close(fig)

    # ---- Plot 5: Great Rotation (retail vs membership growth) ----
    common = [
        "client_id_location_id", "client_id", "year_month_dt", "region",
        "wash_count_retail", "wash_count_membership", "wash_count_total",
    ]
    panel = pd.concat([new[common], old[common]], ignore_index=True)
    chain = panel[["client_id", "client_id_location_id"]].drop_duplicates().groupby("client_id").size().rename("chain_size")
    panel = panel.merge(chain, on="client_id", how="left")
    panel["is_multi"] = panel["chain_size"] > 1
    panel["yr"] = panel["year_month_dt"].dt.year
    annn = panel.groupby(["client_id_location_id", "is_multi", "region", "yr"])[
        ["wash_count_retail", "wash_count_membership"]
    ].sum().reset_index()
    pr = annn.pivot_table(index=["client_id_location_id", "is_multi", "region"], columns="yr", values="wash_count_retail").rename(columns=lambda c: f"r_{c}").reset_index()
    pm = annn.pivot_table(index=["client_id_location_id", "is_multi", "region"], columns="yr", values="wash_count_membership").rename(columns=lambda c: f"m_{c}").reset_index()
    rot = pr.merge(pm, on=["client_id_location_id", "is_multi", "region"]).dropna()

    def agg_yoy(frame, a, b):
        return (frame[b].sum() / frame[a].sum() - 1) * 100

    reg = rot.groupby("region").apply(
        lambda f: pd.Series({"retail": agg_yoy(f, "r_2024", "r_2025"), "membership": agg_yoy(f, "m_2024", "m_2025")})
    ).reindex(["Northeast", "West", "South", "Midwest"])
    ms = rot.groupby("is_multi").apply(
        lambda f: pd.Series({"retail": agg_yoy(f, "r_2024", "r_2025"), "membership": agg_yoy(f, "m_2024", "m_2025")})
    )
    ms.index = ["Single-site (chain=1)" if not i else "Multi-site (chain>=2)" for i in ms.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(reg))
    w = 0.35
    axes[0].bar(x - w / 2, reg["retail"], w, label="Retail washes YoY", color="#a6bddb")
    axes[0].bar(x + w / 2, reg["membership"], w, label="Membership washes YoY", color="#1c9099")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg.index)
    axes[0].set_ylabel("Aggregate YoY % (2024->2025)")
    axes[0].set_title("By region")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    x = np.arange(len(ms))
    axes[1].bar(x - w / 2, ms["retail"], w, label="Retail washes YoY", color="#a6bddb")
    axes[1].bar(x + w / 2, ms["membership"], w, label="Membership washes YoY", color="#1c9099")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ms.index)
    axes[1].set_ylabel("Aggregate YoY % (2024->2025)")
    axes[1].set_title("Multi vs Single")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("The Great Rotation: membership vs retail growth [chem-free]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "great_rotation.png", dpi=140)
    plt.close(fig)

    print(f"[insights] 5 insight plots -> {out_dir}")


def main() -> None:
    data_dir = build_chem_free_data_dir()
    run_legacy_plots(data_dir)
    generate_insight_plots(data_dir, OUT / "plots" / "insights_2")
    print(f"\nDone — all chem-free outputs in {OUT}")


if __name__ == "__main__":
    main()
