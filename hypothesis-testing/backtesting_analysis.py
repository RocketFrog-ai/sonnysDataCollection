"""Join backtesting.xlsx to panels; within-cluster factor comparison by localisation (1–4)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from site_interaction_analysis_lib import build_panel, configure_plotting, haversine_miles

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "interaction_outputs" / "backtesting"
PLOTS_DIR = OUT_DIR / "plots"
DATA_OUT_DIR = OUT_DIR / "data"

LOCALISATION_LABELS: dict[int, str] = {
    1: "Localisation 1 — DFW (Frisco / Lewisville / Prosper, TX)",
    2: "Localisation 2 — Tampa Bay FL + Beaver Dam WI",
    3: "Localisation 3 — Colorado metros",
    4: "Localisation 4 — Tennessee",
}

RETAIL_COLORS: dict[str, str] = {
    "Walmart": "#FF3366",
    "Costco": "#00B4FF",
    "Small Local": "#39FF14",
    "Local Mall": "#FFD700",
    "Local Wholesale": "#FF6600",
    "No major retail": "#9aa0a6",
}


def norm_addr(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


MANUAL_SITE_IDS = {
    norm_addr("TruShine, 2227 Fort Henry Dr, Kingsport, TN, 3766"): "trushine_5",
}


def normalize_retail(value: object) -> str:
    """Sheet uses blank or literal 'None' when no anchor — not a Python missing value."""
    if pd.isna(value):
        return "No major retail"
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "n/a"}:
        return "No major retail"
    text = text.replace("Coscto", "Costco")
    return re.sub(r"\s+", " ", text)


def parse_count_label(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace(",", "")
    mult = 1.0
    if text.endswith("K"):
        mult = 1_000.0
        text = text[:-1]
    elif text.endswith("M"):
        mult = 1_000_000.0
        text = text[:-1]
    try:
        return float(text) * mult
    except ValueError:
        return np.nan


def parse_competition(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.startswith(">"):
        return float(re.sub(r"[^\d.]", "", text) or 5) + 0.5
    if text.startswith("<"):
        return float(re.sub(r"[^\d.]", "", text) or 5) - 0.5
    nums = re.findall(r"\d+", text)
    return float(nums[0]) if nums else np.nan


def load_localisation_map() -> pd.DataFrame:
    for path in (
        DATA_DIR / "backtesting_localisation.tsv",
        DATA_DIR / "data" / "backtesting_localisation.tsv",
        OUT_DIR / "plots" / "del.tsv",
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        df.columns = [str(c).strip() for c in df.columns]
        if "localisation" not in df.columns:
            continue
        out = df[["Address", "localisation"]].copy()
        out["addr_norm"] = out["Address"].map(norm_addr)
        out["localisation"] = pd.to_numeric(out["localisation"], errors="coerce").astype(int)
        return out[["addr_norm", "localisation"]]
    raise FileNotFoundError("backtesting_localisation.tsv / del.tsv not found")


def load_backtesting() -> pd.DataFrame:
    bt = pd.read_excel(DATA_DIR / "backtesting.xlsx", sheet_name="Back testing")
    bt = bt.rename(columns=lambda c: str(c).strip())
    bt["addr_norm"] = bt["Address"].map(norm_addr)
    loc = load_localisation_map()
    bt = bt.merge(loc, on="addr_norm", how="left")
    bt["cohort"] = bt[">2 years/<2 years"].astype(str).str.strip()
    bt["population_n"] = bt["Key Demographics- Population"].map(parse_count_label)
    bt["vehicles_n"] = bt["Key Demograhics-Vehicles Available"].map(parse_count_label)
    bt["pop_growth_pct"] = pd.to_numeric(
        bt["Key Demographics - Propulation Growth % 2020-2025"], errors="coerce"
    )
    bt["traffic_n"] = pd.to_numeric(bt["Traffic Count"], errors="coerce")
    bt["competition_n"] = bt["Competition (<3 mil radius)"].map(parse_competition)
    bt["retail_anchor"] = bt["Nearby Major Retail"].map(normalize_retail)
    bt["area_profile"] = bt["Area Profile"].fillna("Unknown").astype(str).str.strip()
    bt["is_walmart_anchor"] = bt["retail_anchor"].str.fullmatch("Walmart", case=False)
    bt["wash_2024_bt"] = pd.to_numeric(bt["2024 car wash count"], errors="coerce")
    bt["wash_2025_bt"] = pd.to_numeric(bt["2025 car wash count"], errors="coerce")
    return bt


def load_panel_address_directory() -> pd.DataFrame:
    """Exact Address match lookup from lt2 + gt2 monthly CSVs."""
    keep = [
        "client_id_location_id",
        "client_id",
        "Address",
        "street",
        "city",
        "state",
        "zip",
        "latitude",
        "longitude",
        "operational_start_date",
    ]
    lt2 = pd.read_csv(DATA_DIR / "less_than-2yrs.csv", usecols=keep + ["client_type"], low_memory=False)
    gt2 = pd.read_csv(DATA_DIR / "more_than-2yrs_monthly.csv", usecols=keep, low_memory=False)
    gt2["client_type"] = np.nan
    raw = pd.concat([lt2, gt2], ignore_index=True)
    raw["addr_norm"] = raw["Address"].map(norm_addr)
    raw["operational_start_date"] = pd.to_datetime(raw["operational_start_date"], errors="coerce")
    raw["source_cohort"] = np.where(raw["client_type"].notna(), "lt2", "gt2")
    raw = raw.sort_values(["addr_norm", "source_cohort"])
    return raw.drop_duplicates("addr_norm", keep="first")


def site_directory(panel: pd.DataFrame) -> pd.DataFrame:
    sites = load_panel_address_directory()
    return sites.merge(
        panel[["client_id_location_id", "cohort"]].drop_duplicates(),
        on="client_id_location_id",
        how="left",
        suffixes=("", "_panel"),
    )


def format_start_label(value: object) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "n/a"
    return ts.strftime("%Y-%m-%d")


def build_address_match_table(bt: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Report exact address match + operational_start_date from panel CSVs."""
    rows: list[dict[str, object]] = []
    for _, row in bt.iterrows():
        addr_norm = row["addr_norm"]
        hit = sites[sites["addr_norm"] == addr_norm]
        if hit.empty and addr_norm in MANUAL_SITE_IDS:
            hit = sites[sites["client_id_location_id"] == MANUAL_SITE_IDS[addr_norm]]
        matched = not hit.empty
        panel_addr = hit.iloc[0]["Address"] if matched else None
        exact = matched and panel_addr == row["Address"]
        rows.append(
            {
                "backtesting_address": row["Address"],
                "addr_norm": addr_norm,
                "localisation": row.get("localisation"),
                "address_match": matched,
                "address_exact_string": exact if matched else False,
                "panel_address": panel_addr,
                "client_id_location_id": hit.iloc[0]["client_id_location_id"] if matched else None,
                "operational_start_date": hit.iloc[0]["operational_start_date"] if matched else pd.NaT,
                "panel_cohort": hit.iloc[0].get("source_cohort") if matched else None,
            }
        )
    out = pd.DataFrame(rows)
    out["operational_start_date"] = pd.to_datetime(out["operational_start_date"], errors="coerce")
    out["start_label"] = out["operational_start_date"].map(format_start_label)
    return out


def annual_washes_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    monthly = panel.groupby(["client_id_location_id", "calendar_month"], as_index=False)["wash_count_total"].sum()
    monthly["year"] = monthly["calendar_month"].dt.year
    annual = monthly.groupby(["client_id_location_id", "year"], as_index=False)["wash_count_total"].sum()
    wide = annual.pivot(index="client_id_location_id", columns="year", values="wash_count_total").reset_index()
    return wide.rename(columns=lambda c: f"wash_{c}_panel" if isinstance(c, int) else c)


def neighbor_counts(
    sites: pd.DataFrame, target_ids: list[str], radii_mi: tuple[float, ...] = (3.0, 10.0)
) -> pd.DataFrame:
    idx_map = {sid: i for i, sid in enumerate(sites["client_id_location_id"])}
    lat = sites["latitude"].to_numpy(float)
    lon = sites["longitude"].to_numpy(float)
    rows: list[dict[str, object]] = []
    for sid in target_ids:
        if sid not in idx_map:
            continue
        i = idx_map[sid]
        dists = haversine_miles(lat[i], lon[i], lat, lon)
        row: dict[str, object] = {"client_id_location_id": sid}
        for r in radii_mi:
            row[f"neighbors_{int(r)}mi"] = int(((dists > 0) & (dists <= r)).sum())
        others = dists[dists > 0]
        row["nearest_neighbor_mi"] = float(others.min()) if len(others) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def match_backtesting(bt: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    matched = bt.merge(sites, on="addr_norm", how="left", suffixes=("_bt", ""))
    for addr_key, site_id in MANUAL_SITE_IDS.items():
        mask = matched["addr_norm"].eq(addr_key) & matched["client_id_location_id"].isna()
        if mask.any():
            manual = sites[sites["client_id_location_id"] == site_id].iloc[0]
            for col in sites.columns:
                if col != "addr_norm":
                    matched.loc[mask, col] = manual[col]
    return matched


def cluster_span_mi(group: pd.DataFrame) -> float:
    g = group.dropna(subset=["latitude", "longitude"])
    if len(g) < 2:
        return float("nan")
    coords = g[["latitude", "longitude"]].to_numpy()
    max_d = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            max_d = max(
                max_d,
                haversine_miles(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]),
            )
    return round(max_d, 1)


def build_matched_table() -> pd.DataFrame:
    panel, _ = build_panel(DATA_DIR)
    sites = site_directory(panel)
    bt = load_backtesting()
    address_match = build_address_match_table(bt, sites)
    address_match.to_csv(DATA_OUT_DIR / "backtesting_address_match.csv", index=False)

    matched = match_backtesting(bt, sites)
    annual = annual_washes_from_panel(panel)
    matched = matched.merge(annual, on="client_id_location_id", how="left")

    ids = matched["client_id_location_id"].dropna().tolist()
    matched = matched.merge(neighbor_counts(sites, ids), on="client_id_location_id", how="left")

    # Washes from backtesting.xlsx only (not monthly panel sums).
    matched["wash_2024"] = pd.to_numeric(matched["wash_2024_bt"], errors="coerce")
    matched["wash_2025"] = pd.to_numeric(matched["wash_2025_bt"], errors="coerce")
    matched["wash_plot"] = matched["wash_2024"]
    matched["wash_best"] = matched["wash_2024"]
    matched["operational_start_date"] = pd.to_datetime(matched["operational_start_date"], errors="coerce")
    matched["start_label"] = matched["operational_start_date"].map(format_start_label)
    matched["short_label"] = matched.apply(
        lambda r: f"{r.get('client_id', str(r['Address']).split(',')[0])} ({r.get('city', '')})",
        axis=1,
    )
    matched["plot_label"] = matched.apply(
        lambda r: f"{r['short_label']}\nstart {r['start_label']}",
        axis=1,
    )
    matched["cluster_label"] = matched["localisation"].map(LOCALISATION_LABELS)
    matched["site_tag"] = matched.apply(
        lambda r: (
            f"{r['retail_anchor']} | start {r['start_label']} | {r.get('Accessibility', '?')} | "
            f"traffic {r['traffic_n']:,.0f}"
            if pd.notna(r.get("traffic_n"))
            else f"{r['retail_anchor']} | start {r['start_label']} | {r.get('Accessibility', '?')}"
        ),
        axis=1,
    )
    return matched, address_match


def _retail_color(anchor: str) -> str:
    return RETAIL_COLORS.get(str(anchor), "#c9d1d9")


def _site_short_name(label: str) -> str:
    return str(label).split("(")[0].strip()[:14]


def _scatter_labeled(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_col: str,
    xlabel: str,
    title: str,
    *,
    x_pct: bool = False,
    y_col: str = "wash_plot",
) -> None:
    for _, row in sub.iterrows():
        x = row[x_col]
        if pd.isna(x) or pd.isna(row[y_col]):
            continue
        ax.scatter(
            x * 100 if x_pct else x,
            row[y_col],
            s=120,
            c=_retail_color(row["retail_anchor"]),
            zorder=3,
            edgecolors="black",
            linewidths=0.4,
        )
        start = row.get("start_label", "n/a")
        label = (
            f"{_site_short_name(row['short_label'])}\nstart {start}"
            if start != "n/a"
            else _site_short_name(row["short_label"])
        )
        ax.annotate(
            label,
            (x * 100 if x_pct else x, row[y_col]),
            fontsize=6 if start == "n/a" else 5.5,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("2024 car wash count (sheet)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def _categorical_washes(
    ax: plt.Axes,
    sub: pd.DataFrame,
    col: str,
    title: str,
    *,
    color: str = "#4C78A8",
    annotate_start: bool = False,
) -> None:
    order = sub.groupby(col)["wash_plot"].median().sort_values(ascending=False).index.tolist()
    for i, cat in enumerate(order):
        pts = sub.loc[sub[col] == cat]
        ax.scatter([i] * len(pts), pts["wash_plot"], s=100, c=color, zorder=3, edgecolors="black", linewidth=0.4)
        for _, row in pts.iterrows():
            if annotate_start:
                text = f"{_site_short_name(row['short_label'])}\nstart {row['start_label']}"
                fs = 5.5
            else:
                text = _site_short_name(row["short_label"])
                fs = 6
            ax.annotate(
                text,
                (i, row["wash_plot"]),
                fontsize=fs,
                ha="center",
                xytext=(0, 5),
                textcoords="offset points",
            )
    ax.set_xticks(range(len(order)), order, rotation=15, ha="right")
    ax.set_ylabel("Annual washes")
    ax.set_title(title)


def plot_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    cols = {
        "wash_plot": "Washes (2024)",
        "traffic_n": "Traffic",
        "population_n": "Population",
        "vehicles_n": "Vehicles",
        "pop_growth_pct": "Pop growth %",
        "competition_n": "Competition (3mi)",
        "neighbors_3mi": "Sites within 3mi",
        "neighbors_10mi": "Sites within 10mi",
        "nearest_neighbor_mi": "Nearest site (mi)",
    }
    sub = df[list(cols.keys())].rename(columns=cols).apply(pd.to_numeric, errors="coerce")
    if sub.dropna(how="all").shape[0] < 4:
        return
    corr = sub.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), list(cols.values()), rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), list(cols.values()))
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Spearman correlations (all 18 backtest sites — exploratory)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_global_traffic_scatter(df: pd.DataFrame, out: Path) -> None:
    sub = df[["traffic_n", "wash_plot", "short_label", "localisation"]].dropna()
    if len(sub) < 3:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for loc, group in sub.groupby("localisation"):
        label = LOCALISATION_LABELS.get(int(loc), f"Loc {loc}")
        ax.scatter(group["traffic_n"], group["wash_plot"], s=80, alpha=0.9, label=label)
        for _, row in group.iterrows():
            ax.annotate(
                str(row["short_label"])[:20],
                (row["traffic_n"], row["wash_plot"]),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    rho = sub["traffic_n"].corr(sub["wash_plot"], method="spearman")
    ax.set_xlabel("Traffic count (sheet)")
    ax.set_ylabel("2024 car wash count (sheet)")
    ax.set_title(f"Traffic vs washes (ρ = {rho:.2f}) — colored by localisation 1–4")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_localisation_wash_years(cluster_df: pd.DataFrame, loc_id: int, out: Path) -> None:
    """Side-by-side 2024 vs 2025 car wash count from backtesting.xlsx."""
    sub = cluster_df.dropna(subset=["wash_2024"]).copy()
    if sub.empty:
        return

    title = LOCALISATION_LABELS.get(loc_id, f"Localisation {loc_id}")
    sub = sub.sort_values("wash_2024", ascending=True)
    n = len(sub)
    y = np.arange(n)
    height = 0.35

    fig, ax = plt.subplots(figsize=(12, max(4.5, n * 0.9)))
    ax.barh(y - height / 2, sub["wash_2024"], height=height, color="#6baed6", label="2024 car wash count", edgecolor="white")
    has_2025 = sub["wash_2025"].notna()
    ax.barh(
        y[has_2025] + height / 2,
        sub.loc[has_2025, "wash_2025"],
        height=height,
        color="#2171b5",
        label="2025 car wash count",
        edgecolor="white",
    )
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["wash_2024"] * 1.01, i - height / 2, f"{row['wash_2024']:,.0f}", va="center", fontsize=7, color="#333")
        if pd.notna(row["wash_2025"]):
            ax.text(row["wash_2025"] * 1.01, i + height / 2, f"{row['wash_2025']:,.0f}", va="center", fontsize=7, color="#333")
        else:
            ax.text(ax.get_xlim()[1] * 0.02, i + height / 2, "no 2025 in sheet", va="center", fontsize=7, color="#888")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["plot_label"].tolist(), fontsize=8)
    ax.set_xlabel("Car wash count (backtesting.xlsx)")
    ax.set_title(f"{title}\n2024 vs 2025 car wash count columns from backtesting.xlsx")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_localisation_dashboard(cluster_df: pd.DataFrame, loc_id: int, out: Path) -> None:
    """One image per localisation: 8 panels — washes + 7 factor comparisons."""
    sub = cluster_df.dropna(subset=["wash_plot"]).copy()
    if sub.empty:
        return

    title = LOCALISATION_LABELS.get(loc_id, f"Localisation {loc_id}")
    span = cluster_span_mi(sub)
    n = len(sub)

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.suptitle(
        f"{title}\n{n} sites | span ≈ {span:.0f} mi | Y = 2024 car wash count (backtesting.xlsx)",
        fontsize=11,
        y=1.01,
    )

    # Row 0: overview
    ax = axes[0, 0]
    sub_sorted = sub.sort_values("wash_plot", ascending=True)
    colors = [_retail_color(a) for a in sub_sorted["retail_anchor"]]
    ax.barh(sub_sorted["plot_label"], sub_sorted["wash_plot"], color=colors, edgecolor="white", lw=0.5)
    xmax = float(sub_sorted["wash_plot"].max()) * 1.28
    for i, (_, row) in enumerate(sub_sorted.iterrows()):
        ax.text(
            row["wash_plot"] + xmax * 0.02,
            i,
            f"{row['retail_anchor']} · start {row['start_label']}",
            va="center",
            fontsize=6.5,
        )
    ax.set_xlim(0, xmax)
    ax.set_xlabel("2024 car wash count (sheet)")
    ax.set_title("Wash volume by site (2024)")

    _scatter_labeled(axes[0, 1], sub, "traffic_n", "Traffic count", "Traffic vs washes")

    # Row 1: retail & accessibility
    ax = axes[1, 0]
    order = sub.groupby("retail_anchor")["wash_plot"].median().sort_values(ascending=False).index.tolist()
    for i, anchor in enumerate(order):
        pts = sub.loc[sub["retail_anchor"] == anchor]
        ax.scatter([i] * len(pts), pts["wash_plot"], s=100, c=_retail_color(anchor), zorder=3, edgecolors="black", linewidths=0.4)
        for _, row in pts.iterrows():
            start = row.get("start_label", "n/a")
            text = (
                f"{_site_short_name(row['short_label'])}\nstart {start}"
                if start != "n/a"
                else _site_short_name(row["short_label"])
            )
            ax.annotate(text, (i, row["wash_plot"]), fontsize=5.5 if start != "n/a" else 6, ha="center", xytext=(0, 5), textcoords="offset points")
    ax.set_xticks(range(len(order)), order, rotation=15, ha="right")
    ax.set_ylabel("Annual washes")
    ax.set_title("Washes by nearby major retail")

    _categorical_washes(axes[1, 1], sub, "Accessibility", "Washes by accessibility", annotate_start=True)

    # Row 2: demographics
    _scatter_labeled(
        axes[2, 0],
        sub,
        "pop_growth_pct",
        "Population growth % (2020–25)",
        "Population growth vs washes",
        x_pct=True,
    )
    _scatter_labeled(axes[2, 1], sub, "vehicles_n", "Vehicles available", "Vehicles vs washes")

    # Row 3: area & competition
    _categorical_washes(axes[3, 0], sub, "area_profile", "Washes by area profile", color="#E45756")
    _scatter_labeled(
        axes[3, 1],
        sub,
        "competition_n",
        "Competition score (<3 mi radius)",
        "Competition vs washes",
    )

    handles = [Patch(color=_retail_color(a), label=a) for a in sub["retail_anchor"].unique()]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), fontsize=9, title="Retail anchor", bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_readme(matched: pd.DataFrame, cluster_meta: pd.DataFrame) -> None:
    n_match = int(matched["client_id_location_id"].notna().sum())
    cluster_lines = "\n".join(
        f"- **{LOCALISATION_LABELS.get(int(r.localisation), r.localisation)}** — {int(r.n_sites)} sites, span {r.span_mi} mi"
        for _, r in cluster_meta.iterrows()
    )
    text = f"""# Backtesting — localisation clusters (1–4)

Matched **{n_match}/18** sites. Clusters follow `backtesting_localisation.tsv` / `del.tsv` (not auto geography).

Retail blank / "None" in Excel → **No major retail** (Colorado sites have no listed anchor).

## Clusters
{cluster_lines}

## plots/
| File | Content |
|------|---------|
| `cluster_localisation_1.png` … `_4.png` | 8-panel factor dashboard — **2024 car wash count** (sheet) |
| `cluster_localisation_*_washes_2024_2025.png` | Optional: 2024 vs 2025 side-by-side (not used on factor charts) |
| `factor_correlation_heatmap.png` | All-site exploratory (2024 washes) |
| `wash_vs_traffic.png` | Traffic vs **2024** washes, colored by localisation |

### Wash counts
All factor charts use **`2024 car wash count`** from `backtesting.xlsx`. Year comparison uses **`2024`** and **`2025 car wash count`** columns.

### Address match
`data/backtesting_address_match.csv` — exact normalized `Address` match to `less_than-2yrs.csv` / `more_than-2yrs_monthly.csv` + `operational_start_date` when present in those files.

Re-run: `python backtesting_analysis.py`
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> None:
    configure_plotting()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    matched, address_match = build_matched_table()
    meta_rows = []
    for loc, group in matched.groupby("localisation"):
        meta_rows.append(
            {
                "localisation": int(loc),
                "cluster_label": LOCALISATION_LABELS.get(int(loc), str(loc)),
                "n_sites": len(group),
                "span_mi": cluster_span_mi(group),
            }
        )
    cluster_meta = pd.DataFrame(meta_rows).sort_values("localisation")
    matched = matched.merge(cluster_meta, on="localisation", how="left")

    matched.to_csv(DATA_OUT_DIR / "backtesting_matched.csv", index=False)
    print(f"Address match: {address_match['address_match'].sum()}/{len(address_match)} exact norm matches")
    print(f"  operational_start_date known: {address_match['start_label'].ne('n/a').sum()}/{len(address_match)}")
    cluster_meta.to_csv(DATA_OUT_DIR / "backtesting_clusters.csv", index=False)

    plot_correlation_heatmap(matched, PLOTS_DIR / "factor_correlation_heatmap.png")
    plot_global_traffic_scatter(matched, PLOTS_DIR / "wash_vs_traffic.png")

    for loc_id in sorted(matched["localisation"].dropna().unique()):
        group = matched[matched["localisation"] == loc_id]
        lid = int(loc_id)
        plot_localisation_dashboard(group, lid, PLOTS_DIR / f"cluster_localisation_{lid}.png")
        plot_localisation_wash_years(group, lid, PLOTS_DIR / f"cluster_localisation_{lid}_washes_2024_2025.png")

    write_readme(matched, cluster_meta)
    print(f"Wrote {DATA_OUT_DIR / 'backtesting_matched.csv'}")
    print(cluster_meta.to_string(index=False))
    print(f"Plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
