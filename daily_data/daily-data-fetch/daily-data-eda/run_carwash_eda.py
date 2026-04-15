#!/usr/bin/env python3
"""
Car wash EDA: loads master daily + site metadata, writes plots + markdown summary.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# Paths
REPO = Path(__file__).resolve().parents[3]
CSV_DEFAULT = REPO / "app" / "modelling" / "ds" / "datasets" / "master_daily_with_site_metadata.csv"
OUT_DIR = Path(__file__).resolve().parent / "plots"
REPORT_PATH = Path(__file__).resolve().parent / "EDA_CARWASH_REPORT.md"

WEEKEND = {"Saturday", "Sunday"}
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

WEATHER_COLS = [
    "weather_total_precipitation_mm",
    "weather_rainy_days",
    "weather_total_snowfall_cm",
    "weather_days_below_freezing",
    "weather_total_sunshine_hours",
    "weather_days_pleasant_temp",
    "weather_avg_daily_max_windspeed_ms",
]

WEATHER_SHORT_LABELS = [
    "Precip (mm)",
    "Rainy days",
    "Snow (cm)",
    "Days freeze",
    "Sunshine (h)",
    "Pleasant days",
    "Max wind (m/s)",
]


def site_id_to_region(df: pd.DataFrame) -> dict[int, str]:
    """One region per site_client_id from non-null row labels (constant per site in this panel)."""
    out: dict[int, str] = {}
    for sid, g in df.groupby("site_client_id"):
        r = g["region"].dropna()
        if r.empty:
            continue
        out[int(sid)] = r.mode().iloc[0]
    return out


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    df["is_weekend"] = df["day_of_week"].isin(WEEKEND)
    for c in ["wash_count_retail", "wash_count_membership", "wash_count_voucher", "wash_count_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def fig_save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p.relative_to(OUT_DIR.parent)


def plot_daily_total_timeseries(df: pd.DataFrame):
    daily = df.groupby("calendar_day", as_index=False)["wash_count_total"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily["calendar_day"], daily["wash_count_total"], color="#2c5282", lw=0.8)
    ax.set_title("Total car washes per calendar day\n(sum over site-days; one row per site_client_id × date)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wash count (sum across sites)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig_save(fig, "01_daywise_total_wash_count.png")


def plot_weekday_weekend_mean_bar(df: pd.DataFrame, seed: int = 42):
    """Mean wash_count_total: weekday vs weekend, with note on row-count ratio + equal-n check."""
    g = df.groupby("is_weekend", as_index=False).agg(
        n=("wash_count_total", "size"),
        mean_washes=("wash_count_total", "mean"),
    )
    g = g.set_index("is_weekend").reindex([False, True]).reset_index()
    g["label"] = g["is_weekend"].map({True: "Weekend", False: "Weekday"})
    n_wd, n_we = int(g.iloc[0]["n"]), int(g.iloc[1]["n"])
    ratio = n_wd / n_we if n_we else float("nan")
    expected = 5 / 2  # Mon–Fri vs Sat–Sun in a week

    wd_mask = ~df["is_weekend"]
    we_mask = df["is_weekend"]
    n_min = min(n_wd, n_we)
    mu_wd_bal = (
        df.loc[wd_mask, "wash_count_total"].sample(n=n_min, random_state=seed).mean()
        if n_min
        else float("nan")
    )
    mu_we_bal = (
        df.loc[we_mask, "wash_count_total"].sample(n=n_min, random_state=seed).mean()
        if n_min
        else float("nan")
    )
    lift_bal = (mu_we_bal - mu_wd_bal) / mu_wd_bal * 100 if mu_wd_bal else 0

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    colors = ["#2b6cb0", "#dd6b20"]
    bars = ax.bar(g["label"], g["mean_washes"], color=colors, width=0.55, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Mean wash_count_total per site-day")
    ax.set_title("Average car washes: weekday vs weekend")
    ymax = float(g["mean_washes"].max())
    ax.set_ylim(0, ymax * 1.2)
    wd, we = g.iloc[0], g.iloc[1]
    lift = (we["mean_washes"] - wd["mean_washes"]) / wd["mean_washes"] * 100 if wd["mean_washes"] else 0
    for bar, r in zip(bars, g.itertuples()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * 0.02,
            f"{h:.1f}\nn={int(r.n):,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.text(
        0.5,
        0.97,
        f"Weekend mean is +{lift:.1f}% vs weekday (using all rows)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="#4a5568",
    )

    note = (
        f"Why n differs: each week has 5 weekday days vs 2 weekend days, so you expect "
        f"roughly {expected:.1f}× more weekday rows than weekend rows.\n"
        f"Here: {n_wd:,} / {n_we:,} ≈ {ratio:.2f}×. Each row is one site-day (`site_client_id` × date). "
        f"Bar heights are means (not totals).\n"
        f"Equal-n check (random {n_min:,} rows from each, seed={seed}): "
        f"weekday mean {mu_wd_bal:.1f} vs weekend {mu_we_bal:.1f} → +{lift_bal:.1f}% weekend lift."
    )
    fig.text(0.5, 0.055, note, ha="center", va="bottom", fontsize=8.5, color="#2d3748")

    plt.tight_layout(rect=[0, 0.2, 1, 1])
    return fig_save(fig, "02_avg_washes_weekday_vs_weekend.png")


def plot_mean_washes_by_day_of_week(df: pd.DataFrame):
    """Mean wash_count_total by calendar DOW; each row = one site-day (site_client_id × date)."""
    agg = df.groupby("day_of_week")["wash_count_total"].agg(mean="mean", count="count").reindex(DOW_ORDER)
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = ["#3182ce"] * 5 + ["#dd6b20", "#ed8936"]
    bars = ax.bar(DOW_ORDER, agg["mean"], color=colors, edgecolor="white", linewidth=1)
    ax.set_ylabel("Mean wash_count_total (per site-day)")
    ax.set_title("Average car wash count (total) by day of week")
    ax.tick_params(axis="x", rotation=22)
    ymax = float(agg["mean"].max())
    ax.set_ylim(0, ymax * 1.18)
    for bar, (_, row) in zip(bars, agg.iterrows()):
        h = bar.get_height()
        if pd.isna(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * 0.01,
            f"{h:.0f}\nn={int(row['count']):,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.text(
        0.99,
        0.97,
        "Blue = Mon–Fri, orange = Sat–Sun",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#718096",
    )
    plt.tight_layout()
    return fig_save(fig, "02b_mean_washes_by_day_of_week.png")


def region_averages_balanced(df: pd.DataFrame, seed: int = 42):
    """Equal sites per region (random `site_client_id`) vs mean over all site-days in each region."""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=["region"])
    counts = d.groupby("region").size()
    vol_weighted = d.groupby("region")["wash_count_total"].mean()

    site_reg = site_id_to_region(df)
    sites_by_region: dict[str, list[int]] = {}
    for sid, reg in site_reg.items():
        sites_by_region.setdefault(reg, []).append(sid)
    n_min_sites = min(len(v) for v in sites_by_region.values())

    balanced_vals: dict[str, float] = {}
    for reg, site_list in sites_by_region.items():
        idx = rng.choice(len(site_list), size=n_min_sites, replace=False)
        chosen = [site_list[i] for i in idx]
        sub = df.loc[df["site_client_id"].isin(chosen), "wash_count_total"]
        balanced_vals[reg] = float(sub.mean())

    balanced = pd.Series(balanced_vals).sort_values(ascending=False)
    vol_weighted = vol_weighted.reindex(balanced.index)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(balanced))
    w = 0.35
    ax.bar(
        x - w / 2,
        balanced.values,
        w,
        label=f"Balanced by site (n={n_min_sites} site_client_id / region, all their site-days)",
        color="#2b6cb0",
    )
    ax.bar(
        x + w / 2,
        vol_weighted.values,
        w,
        label="All site-days in region (more days where more sites)",
        color="#9f7aea",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(balanced.index))
    ax.set_ylabel("Mean wash_count_total (per site-day)")
    ax.set_title("Regional mean washes: equal # of sites vs full regional panel")
    ax.legend(loc="upper right", fontsize=7.5)
    plt.tight_layout()
    meta = {
        "n_sites_balanced": n_min_sites,
        "sites_per_region": {r: len(v) for r, v in sites_by_region.items()},
        "region_row_counts": counts.to_dict(),
    }
    return fig_save(fig, "03_region_avg_daily_balanced.png"), balanced, vol_weighted, meta


def plot_zip_distribution(df: pd.DataFrame):
    zip_tot = df.groupby("zip", as_index=False)["wash_count_total"].sum().sort_values("wash_count_total", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(zip_tot["wash_count_total"], bins=40, color="#319795", edgecolor="white")
    axes[0].set_xlabel("Total washes in zip (all days, all sites)")
    axes[0].set_ylabel("Number of zips")
    axes[0].set_title("Zip-level total wash volume distribution")
    top = zip_tot.head(25)
    axes[1].barh(top["zip"].astype(str)[::-1], top["wash_count_total"][::-1], color="#319795")
    axes[1].set_xlabel("Total washes")
    axes[1].set_title("Top 25 zips by total wash volume")
    fig.suptitle(
        "Zip = row label in data (sums all site-days in that zip code over the panel)",
        fontsize=9,
        color="#4a5568",
        y=1.03,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig_save(fig, "04_zip_distribution.png"), zip_tot


def plot_region_weather_and_wash_correlation(df: pd.DataFrame):
    """
    (1) Mean weather by region (z-scored across regions for comparable heatmap colors).
    (2) Pearson r: daily regional total washes vs mean weather that day (region × calendar_day).
    """
    d = df.dropna(subset=["region"]).copy()
    for c in WEATHER_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    reg_mean = d.groupby("region")[WEATHER_COLS].mean()
    z = (reg_mean - reg_mean.mean(axis=0)) / reg_mean.std(axis=0).replace(0, np.nan)

    daily = d.groupby(["region", "calendar_day"], as_index=False).agg(
        wash_day_total=("wash_count_total", "sum"),
        **{col: (col, "mean") for col in WEATHER_COLS},
    )

    corr_rows = []
    for reg in sorted(daily["region"].unique()):
        sub = daily.loc[daily["region"] == reg]
        row = {"region": reg}
        for col in WEATHER_COLS:
            pair = sub[["wash_day_total", col]].dropna()
            if len(pair) < 40:
                row[col] = np.nan
            else:
                row[col] = pair["wash_day_total"].corr(pair[col])
        corr_rows.append(row)
    corr_df = pd.DataFrame(corr_rows).set_index("region")[WEATHER_COLS]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    im0 = axes[0].imshow(z.T.values, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2)
    axes[0].set_xticks(range(len(z.index)))
    axes[0].set_xticklabels(list(z.index), rotation=15, ha="right")
    axes[0].set_yticks(range(len(WEATHER_SHORT_LABELS)))
    axes[0].set_yticklabels(WEATHER_SHORT_LABELS, fontsize=9)
    axes[0].set_title("Weather by region (z-score across regions)\n0 = typical; red = high vs other regions")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Z-score")

    im1 = axes[1].imshow(corr_df.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(WEATHER_COLS)))
    axes[1].set_xticklabels(WEATHER_SHORT_LABELS, rotation=35, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(corr_df.index)))
    axes[1].set_yticklabels(list(corr_df.index))
    axes[1].set_title(
        "Correlation: daily total washes vs daily mean weather\n"
        "(per region × calendar day; Pearson r)"
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Pearson r")

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            v = corr_df.iloc[i, j]
            if pd.notna(v):
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    out = fig_save(fig, "07_region_weather_and_wash_correlation.png")

    # Small companion: raw regional mean weather table as barh strip — skip to avoid plot sprawl

    return out, reg_mean, corr_df


def sankey_region_state_city_zip(
    df: pd.DataFrame,
    top_cities_per_state: int = 8,
    top_zips_per_city_node: int = 6,
):
    """Region → state → city → zip. Buckets tail cities/zips to keep the diagram usable."""
    base = df.copy()
    base["city"] = base["city"].fillna("Unknown").astype(str).str.strip()
    base["state"] = base["state"].fillna("?").astype(str)
    flow = base.groupby(["region", "state", "city", "zip"], as_index=False)["wash_count_total"].sum()

    city_vol = flow.groupby(["state", "city"], as_index=False)["wash_count_total"].sum()
    keep_city: set[tuple[str, str]] = set()
    for st, chunk in city_vol.groupby("state"):
        chunk = chunk.sort_values("wash_count_total", ascending=False)
        top = chunk.head(top_cities_per_state)
        for _, r in top.iterrows():
            keep_city.add((st, r["city"]))

    def city_node(row):
        st, ct = row["state"], row["city"]
        if (st, ct) in keep_city:
            return f"{ct}|{st}"
        return f"Other cities ({st})"

    flow["city_node"] = flow.apply(city_node, axis=1)

    zip_parts = []
    for cnode, chunk in flow.groupby("city_node"):
        zorder = chunk.groupby("zip")["wash_count_total"].sum().sort_values(ascending=False)
        keep_z = set(zorder.head(top_zips_per_city_node).index)
        ch = chunk.copy()
        ch["zip_node"] = ch["zip"].map(lambda z, kz=keep_z, cn=cnode: str(z) if z in kz else f"Other zips ({cn})")
        zip_parts.append(ch)
    flow = pd.concat(zip_parts, ignore_index=True)

    agg = flow.groupby(["region", "state", "city_node", "zip_node"], as_index=False)["wash_count_total"].sum()

    r_s = agg.groupby(["region", "state"], as_index=False)["wash_count_total"].sum()
    s_c = agg.groupby(["state", "city_node"], as_index=False)["wash_count_total"].sum()
    c_z = agg.groupby(["city_node", "zip_node"], as_index=False)["wash_count_total"].sum()

    nodes = []
    node_index = {}

    def add_node(name):
        if name not in node_index:
            node_index[name] = len(nodes)
            nodes.append(name)

    for _, r in r_s.iterrows():
        add_node(f"R:{r['region']}")
        add_node(f"S:{r['state']}")
    for _, r in s_c.iterrows():
        add_node(f"S:{r['state']}")
        add_node(f"C:{r['city_node']}")
    for _, r in c_z.iterrows():
        add_node(f"C:{r['city_node']}")
        add_node(f"Z:{r['zip_node']}")

    sources, targets, values = [], [], []
    for _, r in r_s.iterrows():
        sources.append(node_index[f"R:{r['region']}"])
        targets.append(node_index[f"S:{r['state']}"])
        values.append(float(r["wash_count_total"]))
    for _, r in s_c.iterrows():
        sources.append(node_index[f"S:{r['state']}"])
        targets.append(node_index[f"C:{r['city_node']}"])
        values.append(float(r["wash_count_total"]))
    for _, r in c_z.iterrows():
        sources.append(node_index[f"C:{r['city_node']}"])
        targets.append(node_index[f"Z:{r['zip_node']}"])
        values.append(float(r["wash_count_total"]))

    def pretty_label(n: str, max_len: int = 28) -> str:
        s = (
            n.replace("R:", "Region: ")
            .replace("S:", "State: ")
            .replace("C:", "City: ")
            .replace("Z:", "Zip: ")
        )
        if len(s) > max_len:
            return s[: max_len - 1] + "…"
        return s

    n_links = len(sources)
    link_color = ["rgba(100,120,160,0.14)"] * n_links

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="perpendicular",
                node=dict(
                    label=[pretty_label(n) for n in nodes],
                    pad=48,
                    thickness=10,
                    line=dict(color="rgba(0,0,0,0.35)", width=0.5),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_color,
                ),
            )
        ]
    )
    fig.update_layout(
        title_text=(
            f"Car wash flow: region → state → city → zip "
            f"(top {top_cities_per_state} cities/state, top {top_zips_per_city_node} zips/city bucket + Other)"
        ),
        font=dict(size=9, family="Arial, sans-serif"),
        height=1500,
        width=1680,
        margin=dict(l=48, r=280, t=100, b=48),
        paper_bgcolor="#fafafa",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = OUT_DIR / "05_sankey_region_state_city_zip.html"
    png_path = OUT_DIR / "05_sankey_region_state_city_zip.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=1680, height=1500, scale=2)
    except Exception:
        png_path = None
    return html_path.relative_to(OUT_DIR.parent), png_path.relative_to(OUT_DIR.parent) if png_path and png_path.exists() else None


def plot_region_state_flow_static(df: pd.DataFrame):
    """Matplotlib fallback: stacked bars region → state shares (no Chrome required)."""
    flow = df.groupby(["region", "state"], as_index=False)["wash_count_total"].sum()
    pivot = flow.pivot(index="state", columns="region", values="wash_count_total").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:25]]
    fig, ax = plt.subplots(figsize=(10, 7))
    pivot.plot(kind="barh", stacked=True, ax=ax, cmap="tab10")
    ax.set_xlabel("Total wash count")
    ax.set_title("Geographic flow (static): top 25 states by volume, stacked by region (row labels)")
    ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig_save(fig, "05b_region_state_stacked_top25.png")


def retail_membership_voucher_charts(df: pd.DataFrame):
    paths = []
    by_region = df.groupby("region", as_index=False)[
        ["wash_count_retail", "wash_count_membership", "wash_count_voucher"]
    ].sum()
    by_region_pct = by_region.set_index("region")
    tot = by_region_pct.sum(axis=1)
    for c in by_region_pct.columns:
        by_region_pct[c] = by_region_pct[c] / tot * 100
    by_region_pct = by_region_pct.reset_index()

    colors = {"wash_count_retail": "#c53030", "wash_count_membership": "#2b6cb0", "wash_count_voucher": "#d69e2e"}
    labels = {"wash_count_retail": "Retail", "wash_count_membership": "Membership", "wash_count_voucher": "Voucher"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    bottom = np.zeros(len(by_region))
    for col in ["wash_count_retail", "wash_count_membership", "wash_count_voucher"]:
        vals = by_region[col].values
        axes[0].bar(by_region["region"], vals, bottom=bottom, label=labels[col], color=colors[col])
        bottom += vals
    axes[0].set_ylabel("Total wash count")
    axes[0].set_title("By region: absolute counts (stacked)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].tick_params(axis="x", rotation=12)

    x = np.arange(len(by_region_pct))
    w = 0.25
    axes[1].bar(x - w, by_region_pct["wash_count_retail"], w, label="Retail %", color=colors["wash_count_retail"])
    axes[1].bar(x, by_region_pct["wash_count_membership"], w, label="Membership %", color=colors["wash_count_membership"])
    axes[1].bar(x + w, by_region_pct["wash_count_voucher"], w, label="Voucher %", color=colors["wash_count_voucher"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(by_region_pct["region"])
    axes[1].set_ylabel("Share of washes (%)")
    axes[1].set_title("By region: channel mix (%)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].tick_params(axis="x", rotation=12)

    fig.suptitle("Retail / membership / voucher by region", fontsize=11, y=1.02)
    plt.tight_layout()
    paths.append(fig_save(fig, "06_channel_mix_by_region.png"))

    by_state = df.groupby("state", as_index=False)[
        ["wash_count_retail", "wash_count_membership", "wash_count_voucher"]
    ].sum()
    by_state["total"] = by_state[["wash_count_retail", "wash_count_membership", "wash_count_voucher"]].sum(axis=1)
    by_state = by_state.sort_values("total", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    states_rev = by_state["state"].astype(str).values[::-1]
    r = by_state["wash_count_retail"].values[::-1]
    m = by_state["wash_count_membership"].values[::-1]
    v = by_state["wash_count_voucher"].values[::-1]
    ax.barh(states_rev, r, label=labels["wash_count_retail"], color=colors["wash_count_retail"])
    ax.barh(states_rev, m, left=r, label=labels["wash_count_membership"], color=colors["wash_count_membership"])
    ax.barh(states_rev, v, left=r + m, label=labels["wash_count_voucher"], color=colors["wash_count_voucher"])
    ax.set_xlabel("Total wash count")
    ax.set_title("Top 20 states: retail / membership / voucher (absolute)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    paths.append(fig_save(fig, "06c_channel_mix_by_state_top20.png"))

    totals = df[["wash_count_retail", "wash_count_membership", "wash_count_voucher"]].sum()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        totals.values,
        labels=[labels[k] for k in totals.index],
        autopct="%1.1f%%",
        colors=[colors[k] for k in totals.index],
        startangle=90,
    )
    ax.set_title("Overall channel mix (total counts)")
    plt.tight_layout()
    paths.append(fig_save(fig, "06d_channel_mix_overall_pie.png"))

    return paths, by_region, by_state, totals


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

    plot_daily_total_timeseries(df)
    plot_weekday_weekend_mean_bar(df)
    plot_mean_washes_by_day_of_week(df)
    _, balanced, vol_w, reg_meta = region_averages_balanced(df)
    plot_zip_distribution(df)
    html_rel, png_rel = sankey_region_state_city_zip(df)
    plot_region_state_flow_static(df)
    _, by_region, by_state, totals = retail_membership_voucher_charts(df)
    plot_region_weather_and_wash_correlation(df)

    # Summary stats for markdown
    wk = df.groupby("is_weekend")["wash_count_total"]
    mu_wd, mu_we = wk.get_group(False).mean(), wk.get_group(True).mean()
    weekend_pct_lift = (mu_we - mu_wd) / mu_wd * 100 if mu_wd else 0
    ch_tot = float(totals.sum())
    pct = {k: 100 * float(totals[k]) / ch_tot for k in totals.index}

    n_sites = int(df["site_client_id"].nunique())
    n_sites_region = len(site_id_to_region(df))
    summary_lines = [
        f"- Site-days (rows): **{len(df):,}** — one row per **`site_client_id` × `calendar_day`**",
        f"- Distinct sites (`site_client_id`): **{n_sites}**; with non-null `region` on at least one row: **{n_sites_region}**",
        f"- Distinct `location_id` (chain-style id, not unique per site): **{df['location_id'].nunique()}**",
        f"- Date range: **{df['calendar_day'].min().date()}** → **{df['calendar_day'].max().date()}**",
        f"- Weekday rows: mean `wash_count_total` **{mu_wd:.2f}** (n={wk.get_group(False).shape[0]:,})",
        f"- Weekend rows: mean **{mu_we:.2f}** (n={wk.get_group(True).shape[0]:,}) → **+{weekend_pct_lift:.1f}%** vs weekday mean",
        f"- Weekday/weekend row ratio **{wk.get_group(False).shape[0] / max(wk.get_group(True).shape[0], 1):.2f}×** (≈ **5/2** expected from the calendar; see plot 02 footnote + equal-*n* check).",
        "",
        "**Mean `wash_count_total` by day of week** (each row = one site-day; see `plots/02b_mean_washes_by_day_of_week.png`):",
    ]
    dow_agg = df.groupby("day_of_week")["wash_count_total"].agg(mean="mean", count="count").reindex(DOW_ORDER)
    for d in DOW_ORDER:
        r = dow_agg.loc[d]
        summary_lines.append(f"- **{d}:** {r['mean']:.2f} (n={int(r['count']):,})")
    summary_lines.extend(
        [
            "",
            f"**Site-balanced regional means** (random **{reg_meta['n_sites_balanced']}** `site_client_id` per region, all their site-days; seed=42):",
        ]
    )
    for reg, v in balanced.items():
        summary_lines.append(f"- {reg}: **{v:.2f}**")
    summary_lines.append("")
    summary_lines.append("**Sites per region** (from row `region` labels, one region per site):")
    for reg, c in sorted(reg_meta["sites_per_region"].items(), key=lambda x: -x[1]):
        summary_lines.append(f"- {reg}: **{c}** sites")
    summary_lines.append("")
    summary_lines.append("**Mean over all site-days in region** (more site-days where there are more sites):")
    for reg, v in vol_w.items():
        summary_lines.append(f"- {reg}: **{v:.2f}**")
    summary_lines.append("")
    summary_lines.append("**Row counts by region (full data):**")
    for reg, c in sorted(reg_meta["region_row_counts"].items(), key=lambda x: -x[1]):
        summary_lines.append(f"- {reg}: **{c:,}**")
    summary_lines.append("")
    summary_lines.append("**Overall channel totals:**")
    summary_lines.append(f"- Retail: **{int(totals['wash_count_retail']):,}** ({pct['wash_count_retail']:.1f}%)")
    summary_lines.append(f"- Membership: **{int(totals['wash_count_membership']):,}** ({pct['wash_count_membership']:.1f}%)")
    summary_lines.append(f"- Voucher: **{int(totals['wash_count_voucher']):,}** ({pct['wash_count_voucher']:.1f}%)")

    plot_md = "\n".join(f"![{p.stem}](plots/{p.name})" for p in sorted((OUT_DIR).glob("*.png")))
    sankey_md = f"- Interactive Sankey: [open `plots/{html_rel.name}`](plots/{html_rel.name}) in a browser.\n"
    if png_rel:
        sankey_md += f"\n![Sankey](plots/{png_rel.name})\n"

    try:
        csv_disp = f"`{csv_path.relative_to(REPO)}`"
    except ValueError:
        csv_disp = f"`{csv_path}`"

    report = f"""# Car wash exploratory data analysis

**Source data:** {csv_disp}  
**Generated:** auto from `run_carwash_eda.py`

## What we measured

- **Day-wise:** sum of `wash_count_total` across all site-days per calendar day.
- **Weekday vs weekend:** mean `wash_count_total` per **site-day** (`site_client_id` × date) (`02_*.png`).
- **By day of week:** same grain for Mon–Sun (`02b_*.png`).
- **Region averages:** **Blue** = random **`site_client_id`** sample with **equal site count per region** (min = Northeast’s site count), mean over all site-days from those sites. **Purple** = mean over **every** site-day row in that region (South has more sites → more rows).
- **Zip distribution:** histogram of total washes by zip and top 25 zips.
- **Sankey:** flow **region → state → city → zip** (`05_sankey_region_state_city_zip.html`). Top cities per state and top zips per city bucket; tails → “Other cities (ST)” / “Other zips (…)”.
- **Channel mix:** `wash_count_retail`, `wash_count_membership`, `wash_count_voucher` by region, top states, and overall.
- **Weather vs washes:** mean weather by region (z-scored heatmap) and **Pearson correlation** of **daily regional total** `wash_count_total` with **mean same-day weather** in that region (`07_*.png`). Not causal (season, mix of sites, shared weather fields).

## Key numbers

{chr(10).join(summary_lines)}

## Findings (brief)

1. **Panel shape:** Each row is **one site-day**: **`site_client_id` × `calendar_day`** (482 sites in this extract). `location_id` is **not** unique per site (many sites share a chain `location_id`). Use **`site_client_id`** (or `client_id` + `location_id`) as the site key.
2. **Day-of-week pattern:** Plot `02b_mean_washes_by_day_of_week.png` (and Key numbers) show **Saturday** and **Friday** as the strongest days on average; **Sunday** is closer to mid-week. The pooled weekend vs weekday bar (`02_avg_washes_weekday_vs_weekend.png`) still shows ~**+15%** weekend lift because **Saturday** pulls the weekend average up. **Unequal weekday/weekend row counts** match the 5:2 calendar; the equal-*n* note on plot 02 checks that lift is not a sample-size artifact.
3. **Regional chart (plot 03):** **Blue** bars equalize **how many sites** enter each region (**{reg_meta["n_sites_balanced"]}** `site_client_id` per region, seed=42). **Purple** = all site-days in the region. Gaps between blue and purple show when volume/site mix in the full panel differs from an equal-site sample.
4. **Zip concentration:** Total wash volume by zip is long-tailed: a minority of zips drive most counts (see histogram + top-25 bar chart).
5. **Sankey:** Interactive HTML shows **region → state → city → zip**; low-volume cities and zips roll into **Other** buckets so the graph stays readable. Static PNG needs Chrome for Kaleido; use HTML or `05b_*.png` for a simpler regional view.
6. **Channels:** Retail dominates overall (**{pct["wash_count_retail"]:.1f}%** of washes), membership is **{pct["wash_count_membership"]:.1f}%**, vouchers **{pct["wash_count_voucher"]:.1f}%**. `06_channel_mix_by_region.png` pairs **absolute** stacked counts (left) with **percent mix** by region (right).
7. **Weather (plot 07):** Left panel compares **average weather** across regions (z-scores so different units share one color scale). Right panel correlates **each region’s daily total washes** (sum of all site-days that day in that region) with **that day’s average weather** in that region. Strong |r| suggests co-movement (often seasonality or market mix), not that weather alone drives volume.

## Zip plots (brief)

- **Left histogram:** How total wash volume is spread across zip codes — most zips are modest; a long tail of **high-volume zips** (dense markets / many site-days).
- **Right bar chart:** The **top 25 zips** by cumulative washes; labels are the zip codes in the data.

## Plots

{plot_md}

### Sankey (region → state → zip)

{sankey_md}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Plots in {OUT_DIR}")


if __name__ == "__main__":
    main()
