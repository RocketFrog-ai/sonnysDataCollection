"""Regenerate the full hypothesis-testing plot suite from the new chem-free CSVs.

Inputs (new schema, `calendar_year` + `calendar_month`, no `primary_carwash_type`,
no `client_type` on GT2Y):
  - _no_chem_data/less_than-2yrs-nochem.csv
  - _no_chem_data/more_than-2yrs_monthly-nochem.csv

What this writes (under interaction_outputs_nochem_v2/):
  * Full interaction suite via site_interaction_analysis_lib:
      - plots/two_body/{examples_all_sites, trends_by_site_type_combo, avg_existing_single_new_multi_trend}.png
      - plots/three_body/{examples_all_sites, avg_all_triples_trend,
                          avg_all_triples_trend_overall, avg_new_multi_intro_trend}.png
      - plots/four_body/{examples_all_sites, avg_all_quads_trend, avg_all_quads_trend_overall}.png
      - plots/aggregate/{any_new_operator_effect, market_saturation_threshold}.png
      - data/{two_body_pair_deltas, three_body_triple_deltas, four_body_quad_deltas}.csv
  * Insight plots (chem-free, new-schema):
      - plots/insights_2/{age_lifecycle_cliff, membership_leading_indicator,
                          multi_vs_single_ramp, cannibalization_donut, great_rotation}.png
  * Extra plots: ramp / cannibal donut / membership-share predictor split by site type AND region:
      - plots/insights_2/multi_vs_single_ramp__by_type_region.png
      - plots/insights_2/cannibalization_donut__by_type_region.png
      - plots/insights_2/membership_leading_indicator__by_type_region.png

Re-run: `python hypothesis-testing/run_nochem_v2_analysis.py`
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SRC = BASE / "_no_chem_data"
OUT = BASE / "interaction_outputs_nochem_v2"
LT_FILE = "less_than-2yrs-nochem.csv"
GT_FILE = "more_than-2yrs_monthly-nochem.csv"
FULL_SITE_FILE = "full-site-wise_with_yearly_agg.csv"

WASH_FLOOR = 500          # drop operational anomalies / soft opens
DROP_PARTIAL_LAUNCH_DAY = 15  # rows with msl == 0 and op_day > this are dropped


# --------------------------------------------------------------------------------------
# 1.  Harmonize the new -nochem CSVs into the legacy schema so the existing lib runs.
# --------------------------------------------------------------------------------------

def _apply_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the modeling-notebook hygiene filters: drop partial-launch months and soft opens."""
    df = df.copy()
    df["op_date"] = pd.to_datetime(df["operational_start_date"], errors="coerce")
    df["op_day"] = df["op_date"].dt.day
    df["msl"] = (
        (df["year_month_dt"].dt.year - df["op_date"].dt.year) * 12
        + (df["year_month_dt"].dt.month - df["op_date"].dt.month)
    )
    mask_partial = (df["msl"] == 0) & (df["op_day"] > DROP_PARTIAL_LAUNCH_DAY)
    df = df[~mask_partial]
    df = df[df["wash_count_total"] >= WASH_FLOOR]
    return df.drop(columns=["op_date", "op_day", "msl"])


def _site_lookup() -> pd.DataFrame:
    """Canonical site metadata keyed by client_id_location_id."""
    lookup_cols = [
        "client_id_location_id",
        "client_type",
        "operational_start_date",
        "age_on_30_sep_25",
    ]
    full = pd.read_csv(SRC / FULL_SITE_FILE, usecols=lookup_cols, low_memory=False)
    full = full.dropna(subset=["client_id_location_id"]).copy()
    full["client_id_location_id"] = full["client_id_location_id"].astype(str)
    full = full.sort_values("client_id_location_id").drop_duplicates("client_id_location_id", keep="last")
    return full


def _enrich_site_meta(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Fill client/site metadata from full-site-wise_with_yearly_agg."""
    out = df.copy()
    out["client_id_location_id"] = out["client_id_location_id"].astype(str)
    out = out.merge(
        lookup.rename(
            columns={
                "client_type": "client_type_lookup",
                "operational_start_date": "operational_start_date_lookup",
                "age_on_30_sep_25": "age_on_30_sep_25_lookup",
            }
        ),
        on="client_id_location_id",
        how="left",
    )
    if "client_type" not in out.columns:
        out["client_type"] = np.nan
    out["client_type"] = out["client_type"].fillna(out["client_type_lookup"])
    out["site_type"] = out["client_type"]
    out["operational_start_date"] = out["operational_start_date"].fillna(out["operational_start_date_lookup"])
    out["age_on_30_sep_25"] = out["age_on_30_sep_25"].fillna(out["age_on_30_sep_25_lookup"])
    return out.drop(
        columns=[
            "client_type_lookup",
            "operational_start_date_lookup",
            "age_on_30_sep_25_lookup",
        ],
        errors="ignore",
    )


def harmonize_to_legacy(staging_dir: Path) -> Path:
    """Write legacy-schema CSVs (year_month, month_number, dbscan_cluster_12km) into staging_dir.

    Returns the staging dir so it can be passed as `data_dir` to build_panel().
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    lt = pd.read_csv(SRC / LT_FILE, low_memory=False)
    gt = pd.read_csv(SRC / GT_FILE, low_memory=False)
    lookup = _site_lookup()
    lt = _enrich_site_meta(lt, lookup)
    gt = _enrich_site_meta(gt, lookup)

    for df in (lt, gt):
        df["year_month_dt"] = pd.to_datetime(
            df["calendar_year"].astype(str)
            + "-"
            + df["calendar_month"].astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )

    lt = lt[(lt["year_month_dt"] >= "2024-01-01") & (lt["year_month_dt"] <= "2025-12-01")].copy()
    gt = gt[(gt["year_month_dt"] >= "2024-01-01") & (gt["year_month_dt"] <= "2025-12-01")].copy()
    lt = _apply_hygiene(lt)
    gt = _apply_hygiene(gt)

    # Legacy lib expects `year_month` as a string. Use first-of-month "YYYY-MM" with day appended.
    for df in (lt, gt):
        df["year_month"] = df["year_month_dt"].dt.strftime("%Y-%m")
        df["dbscan_cluster_12km"] = np.nan  # column required by lib but never used downstream

    # lt expects `month_number` = months since launch + 1.
    lt["op_date"] = pd.to_datetime(lt["operational_start_date"], errors="coerce")
    lt["month_number"] = (
        (lt["year_month_dt"].dt.year - lt["op_date"].dt.year) * 12
        + (lt["year_month_dt"].dt.month - lt["op_date"].dt.month)
        + 1
    )
    lt = lt.drop(columns=["op_date", "year_month_dt"])
    gt = gt.drop(columns=["year_month_dt"])

    lt.to_csv(staging_dir / "less_than-2yrs.csv", index=False)
    gt.to_csv(staging_dir / "more_than-2yrs_monthly.csv", index=False)
    gt_filled = int(gt["client_type"].notna().sum())
    print(f"[harmonize] gt client_type populated rows={gt_filled:,}/{len(gt):,}")
    print(
        f"[harmonize] lt rows={len(lt):,} sites={lt['client_id_location_id'].nunique()}   "
        f"gt rows={len(gt):,} sites={gt['client_id_location_id'].nunique()}"
    )
    return staging_dir


# --------------------------------------------------------------------------------------
# 2.  Standard interaction-plot suite via the existing lib.
# --------------------------------------------------------------------------------------

def run_interaction_suite(data_dir: Path) -> None:
    """Use the existing lib, unchanged, to produce 2/3/4-body + aggregate plots."""
    from site_interaction_analysis_lib import (
        build_panel,
        build_pair_deltas,
        build_quad_deltas,
        build_sites,
        build_triple_deltas,
        configure_plotting,
        find_pairs,
        find_quads,
        find_triples,
        plot_any_new_operator_effect,
        plot_existing_single_new_multi_trend,
        plot_four_body_all_quads_trend,
        plot_four_body_all_quads_trend_overall,
        plot_market_saturation,
        plot_new_multi_three_body_trend,
        plot_pair_examples_all,
        plot_three_body_all_triples_trend,
        plot_three_body_all_triples_trend_overall,
        plot_two_body_trends_by_type_combo,
        prepare_interaction_dirs,
    )

    configure_plotting()
    out = prepare_interaction_dirs(OUT)

    panel, validation = build_panel(data_dir)
    print(
        f"[panel] {validation['rows']:,} rows  | sites: lt2={validation['lt2_sites']}, "
        f"gt2={validation['gt2_sites']}, total={validation['sites']}"
    )

    sites, _, dist = build_sites(panel)
    pairs_df = find_pairs(sites, dist, 10.0, 6)
    triples_df = find_triples(sites, dist, 10.0, 6)
    quads_df = find_quads(sites, dist, 10.0, 6)
    pair_deltas = build_pair_deltas(panel, pairs_df, 6, 3)
    triple_deltas = build_triple_deltas(panel, triples_df, 6, 3)
    quad_deltas = build_quad_deltas(panel, quads_df, 6, 3)
    print(
        f"[deltas] pairs={len(pair_deltas)}  triples={len(triple_deltas)}  quads={len(quad_deltas)}"
    )

    pair_deltas.to_csv(out["data"] / "two_body_pair_deltas.csv", index=False)
    triple_deltas.to_csv(out["data"] / "three_body_triple_deltas.csv", index=False)
    quad_deltas.to_csv(out["data"] / "four_body_quad_deltas.csv", index=False)

    plot_pair_examples_all(pair_deltas, panel, out["two_body"] / "examples_all_sites.png")
    plot_existing_single_new_multi_trend(
        pair_deltas, panel, out["two_body"] / "avg_existing_single_new_multi_trend.png", 6
    )
    plot_two_body_trends_by_type_combo(
        pair_deltas, panel, out["two_body"] / "trends_by_site_type_combo.png", 6
    )

    # 3-body / 4-body example grids: use our local versions that match the 2-body theme
    # (same ncols, panel size, marker/line styles, title compactness). The lib's
    # versions diverge stylistically; we keep the lib for the trend/aggregate plots.
    plot_triple_examples_all_pair_theme(
        triple_deltas, panel, out["three_body"] / "examples_all_sites.png"
    )
    plot_three_body_all_triples_trend(
        triple_deltas, panel, out["three_body"] / "avg_all_triples_trend.png", 6
    )
    plot_three_body_all_triples_trend_overall(
        triple_deltas, panel, out["three_body"] / "avg_all_triples_trend_overall.png", 6
    )
    plot_new_multi_three_body_trend(
        triple_deltas, panel, out["three_body"] / "avg_new_multi_intro_trend.png", 6
    )
    # 3-body 2x2 type-combo plot (A_type × C_type) — analogue of the 2-body trends_by_site_type_combo
    plot_three_body_trends_by_type_combo(
        triple_deltas, panel, out["three_body"] / "trends_by_site_type_combo.png", 6
    )
    # 3-body FULL type breakdown: 2x4 grid covering every (A x B x C) single/multi combo
    plot_three_body_trends_by_full_type_combo(
        triple_deltas, panel, out["three_body"] / "trends_by_full_type_combo.png", 6
    )

    plot_quad_examples_all_pair_theme(
        quad_deltas, panel, out["four_body"] / "examples_all_sites.png"
    )
    plot_four_body_all_quads_trend(
        quad_deltas, panel, out["four_body"] / "avg_all_quads_trend.png", 6
    )
    plot_four_body_all_quads_trend_overall(
        quad_deltas, panel, out["four_body"] / "avg_all_quads_trend_overall.png", 6
    )
    # 4-body 2x2 type-combo plot (A_type × D_type), with B/C breakdown printed in panel titles
    plot_four_body_trends_by_type_combo(
        quad_deltas, panel, out["four_body"] / "trends_by_site_type_combo.png", 6
    )
    # 4-body FULL type breakdown: 4x4 grid covering every (A x B x C x D) single/multi combo (16 panels)
    plot_four_body_trends_by_full_type_combo(
        quad_deltas, panel, out["four_body"] / "trends_by_full_type_combo.png", 6
    )

    plot_any_new_operator_effect(
        pair_deltas, panel, out["aggregate"] / "any_new_operator_effect.png", 6
    )
    plot_market_saturation(pair_deltas, out["aggregate"] / "market_saturation_threshold.png")
    print(f"[interaction-suite] outputs -> {OUT}")


# --------------------------------------------------------------------------------------
# 2b. 3-body / 4-body example grids re-themed to match 2-body exactly.
#     - same ncols (9), per-panel size (3.6 x 3.0), marker (2.5) and line width (1.2)
#     - same compact title style as 2-body
#     - extra vertical dashed lines mark each entrant's launch month (so the
#       temporal cascade A -> B -> C [-> D] is unmistakable in each panel)
# --------------------------------------------------------------------------------------

def _build_helpers():
    """Lazy import of lib helpers used by the re-themed example grids."""
    from site_interaction_analysis_lib import (
        CALENDAR_X_END,
        CALENDAR_X_START,
        PAIR_EVENT_METRIC,
        _apply_calendar_xaxis,
        _site_calendar_series,
        _site_state_label,
        _site_type_label,
        month_floor,
    )

    return {
        "CALENDAR_X_START": CALENDAR_X_START,
        "CALENDAR_X_END": CALENDAR_X_END,
        "PAIR_EVENT_METRIC": PAIR_EVENT_METRIC,
        "_apply_calendar_xaxis": _apply_calendar_xaxis,
        "_site_calendar_series": _site_calendar_series,
        "_site_state_label": _site_state_label,
        "_site_type_label": _site_type_label,
        "month_floor": month_floor,
    }


def _site_meta(panel: pd.DataFrame, site_id: str) -> tuple[str, str, str]:
    sub = panel[panel["client_id_location_id"] == site_id]
    cohort = "gt2"
    if "month_number" in sub.columns and sub["month_number"].notna().any():
        cohort = "lt2"
    start = "-"
    if "operational_start_date" in sub.columns:
        s = pd.to_datetime(sub["operational_start_date"], errors="coerce").dropna()
        if not s.empty:
            start = s.min().strftime("%Y-%m-%d")
    age = "-"
    if "age_on_30_sep_25" in sub.columns:
        a = pd.to_numeric(sub["age_on_30_sep_25"], errors="coerce").dropna()
        if not a.empty:
            age = f"{float(a.median()):.1f}"
    return cohort, start, age


def plot_triple_examples_all_pair_theme(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    ncols: int = 9,
) -> None:
    H = _build_helpers()
    x_start, x_end = H["CALENDAR_X_START"], H["CALENDAR_X_END"]
    metric = H["PAIR_EVENT_METRIC"]

    triples = triple_deltas.sort_values(["event_month", "A_to_C_miles", "B_to_C_miles"]).reset_index(drop=True)
    n = len(triples)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig_w = max(30.0, ncols * 3.6)
    fig_h = max(30.0, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    series_defs = [
        ("A (oldest)", "A_site", "A_launch_month", "#1f77b4", False),
        ("B (middle)", "B_site", "B_launch_month", "#2ca02c", True),
        ("C (newest)", "C_site", "C_launch_month", "#d62728", True),
    ]

    for ax, (_, tri) in zip(axes_flat, triples.iterrows()):
        b_launch = H["month_floor"](tri["B_launch_month"])
        c_launch = H["month_floor"](tri["C_launch_month"])
        a_launch = H["month_floor"](tri["A_launch_month"])
        state = tri.get("state") or H["_site_state_label"](panel, tri["C_site"])
        for label, site_col, launch_col, color, from_launch in series_defs:
            min_month = H["month_floor"](tri[launch_col]) if from_launch else None
            s = H["_site_calendar_series"](
                panel, tri[site_col], x_start, x_end,
                from_launch=from_launch, min_calendar_month=min_month,
            )
            ax.plot(s["calendar_month"], s[metric],
                    marker="o", ms=2.5, lw=1.2, color=color, label=label)
        ax.axvline(b_launch, color="#2ca02c", ls="--", lw=0.9, alpha=0.75)
        ax.axvline(c_launch, color="#d62728", ls="--", lw=0.9, alpha=0.75)
        a_type = H["_site_type_label"](panel, tri["A_site"])
        b_type = H["_site_type_label"](panel, tri["B_site"])
        c_type = H["_site_type_label"](panel, tri["C_site"])
        a_cohort, a_start, a_age = _site_meta(panel, tri["A_site"])
        b_cohort, b_start, b_age = _site_meta(panel, tri["B_site"])
        c_cohort, c_start, c_age = _site_meta(panel, tri["C_site"])
        a_to_c = tri["A_to_C_miles"]; b_to_c = tri["B_to_C_miles"]
        ax.set_title(
            f"{state} | {tri['market_zip']} | A-C {a_to_c:.1f} mi  B-C {b_to_c:.1f} mi\n"
            f"A {a_launch.strftime('%Y-%m')} ({a_type}, {a_cohort}, start {a_start}, age {a_age})  "
            f"B {b_launch.strftime('%Y-%m')} ({b_type}, {b_cohort}, start {b_start}, age {b_age})\n"
            f"C {c_launch.strftime('%Y-%m')} ({c_type}, {c_cohort}, start {c_start}, age {c_age})",
            fontsize=6,
        )
        ax.tick_params(axis="y", labelsize=6)
        H["_apply_calendar_xaxis"](ax, x_start, x_end, labelsize=5, month_interval=1)

    for ax in axes_flat[:n]:
        ax.set_xlabel("Month", fontsize=6)
    for ax in axes_flat[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("Car washes", fontsize=7)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"All three-body triples ({n}) - same area, temporal cascade A -> B -> C\n"
        "Blue = A (older incumbent) | Green = B (middle entrant) | Red = C (newest) | "
        "panel title includes type + cohort (lt2/gt2) + start date + age | "
        "dashed = each entrant's launch month",
        fontsize=14, y=1.02,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.12, hspace=0.78, wspace=0.35)
    fig.savefig(out_path, dpi=120, pad_inches=0.15)
    plt.close(fig)


def _draw_split_medians(
    ax,
    traces: pd.DataFrame,
    series_defs: list,
    *,
    window: int,
    event_id_col: str,
    title: str,
    ylabel: str,
    ymax_cap: float = 320.0,
    show_faint: bool = True,
) -> None:
    """Like the lib's `_draw_event_medians` but each series carries an optional filter.

    series_defs: list of (col, filter_dict_or_None, color, linestyle, label).
    For e.g. splitting B into B-single (solid) and B-multi (dashed), pass two
    rows with col='B_index' and filter={'B_type':'single'} / {'B_type':'multi'}.
    """
    if traces.empty:
        ax.set_title(title, fontsize=10)
        return

    plotted_cols: list[str] = []
    for col, filt, color, ls, label in series_defs:
        sub = traces
        if filt:
            for k, v in filt.items():
                sub = sub[sub[k] == v]
        if sub.empty:
            continue
        plotted_cols.append(col)
        if show_faint:
            for _, group in sub.groupby(event_id_col):
                s = group.sort_values("relative_month")
                ax.plot(s["relative_month"], s[col], color=color, alpha=0.08,
                        lw=0.7, zorder=1, linestyle=ls)
        summary = (
            sub.groupby("relative_month")[col]
            .agg(median="median",
                 q25=lambda s: s.quantile(0.25),
                 q75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("relative_month")
        )
        ax.fill_between(summary["relative_month"], summary["q25"], summary["q75"],
                        color=color, alpha=0.13, zorder=2)
        n_eff = int(sub[event_id_col].nunique())
        ax.plot(summary["relative_month"], summary["median"],
                color=color, lw=2.2, marker="o", ms=4, linestyle=ls,
                label=f"{label} (n={n_eff})", zorder=3)

    if plotted_cols:
        ymax = float(traces[plotted_cols].quantile(0.98).max())
        ax.set_ylim(0, max(150.0, min(ymax * 1.1, ymax_cap)))
    ax.axvline(0, color="black", ls="--", lw=1.0)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel("Months relative to newest-site launch (0)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)


def _panel_meta_summary(deltas_df: pd.DataFrame, ids: set, body: str) -> str:
    """For a set of triple_ids or quad_ids, return 'regions: X(n)/Y(m) | avg dist ...'.

    Used to enrich panel titles with the region breakdown and the average
    site-to-newest distance(s) for the events in that panel.
    """
    if not ids:
        return ""
    if body == "triple":
        id_col_builder = lambda r: f"{r['A_site']}|{r['B_site']}|{r['C_site']}"
        dist_cols = [("A-C", "A_to_C_miles"), ("B-C", "B_to_C_miles")]
    else:
        id_col_builder = lambda r: f"{r['A_site']}|{r['B_site']}|{r['C_site']}|{r['D_site']}"
        dist_cols = [("A-D", "A_to_D_miles"), ("B-D", "B_to_D_miles"), ("C-D", "C_to_D_miles")]

    sub = deltas_df.copy()
    sub["_id"] = sub.apply(id_col_builder, axis=1)
    sub = sub[sub["_id"].isin(ids)]
    if sub.empty:
        return ""
    region_counts = sub["region"].value_counts()
    region_str = "/".join(f"{r}({int(n)})" for r, n in region_counts.items())
    dist_str = ", ".join(f"{label} {sub[col].mean():.1f}mi" for label, col in dist_cols if col in sub.columns)
    return f"regions: {region_str}  |  avg {dist_str}"


def _triple_breakdown(triples_for_panel: pd.DataFrame, role: str) -> str:
    """e.g. 'B: 12 single / 6 multi' — shows the pooled-role mix inside a 2x2 panel."""
    if triples_for_panel.empty:
        return f"{role}: -"
    col = f"{role}_type"
    if col not in triples_for_panel.columns:
        return f"{role}: -"
    vc = triples_for_panel.drop_duplicates(subset=[f"{role}_index"], keep="first").assign(_=1)  # noop, keep all
    vc = triples_for_panel.groupby(col).size()
    parts = [f"{int(n)} {t}" for t, n in vc.items()]
    return f"{role}: " + " / ".join(parts) if parts else f"{role}: -"


def plot_three_body_trends_by_type_combo(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """2x2 grid: median A/B/C/market trend for each (A_type x C_type) combo.

    B is pooled within each panel; the per-panel title prints B's single/multi
    mix. For the fully un-pooled view (every A×B×C combo on its own panel) see
    `trends_by_full_type_combo.png`.
    """
    from site_interaction_analysis_lib import (
        THREE_BODY_TREND_SERIES,
        _draw_event_medians,
        build_triple_event_traces,
    )

    all_traces = build_triple_event_traces(triple_deltas, panel, window=window)
    combos = [
        ("single", "single"),
        ("single", "multi"),
        ("multi",  "single"),
        ("multi",  "multi"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    counts: list[dict[str, object]] = []
    for ax, (a_type, c_type) in zip(axes.flat, combos):
        sub = all_traces[(all_traces["A_type"] == a_type) & (all_traces["C_type"] == c_type)] if not all_traces.empty else all_traces
        n_triples = int(sub["triple_id"].nunique()) if not sub.empty else 0
        per_triple = sub.drop_duplicates("triple_id") if not sub.empty else sub
        b_breakdown = _triple_breakdown(per_triple, "B")
        meta = _panel_meta_summary(triple_deltas, set(per_triple["triple_id"]) if not per_triple.empty else set(), "triple")
        title = f"A={a_type}  →  C={c_type}    ({b_breakdown})\n{meta}"
        counts.append({"A_type": a_type, "C_type": c_type, "n_triples": n_triples})
        if sub.empty:
            ax.text(0.5, 0.5, "No triples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue
        _draw_event_medians(
            ax, sub,
            window=window,
            title=title,
            series_specs=THREE_BODY_TREND_SERIES,
            event_id_col="triple_id",
            ylabel="Index ((A+B) pre-launch = 100)",
            show_faint=True,
        )
    fig.suptitle(
        "Three-body trends by (A × C) — B's single/multi mix + region/distance per panel\n"
        "Blue=A | Green=B | Red=C | Orange=A+B+C market | 100 = (A+B) pre-launch | month 0 = C launch",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pd.DataFrame(counts)


def plot_three_body_trends_by_full_type_combo(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """2x4 grid covering EVERY (A_type, B_type, C_type) combination (8 panels).

    Rows = A_type; columns = (B_type, C_type) combos. Each panel shows the same
    A/B/C/market traces but filtered to one exact 3-type slice so every site's
    type is explicit (no pooling).
    """
    from site_interaction_analysis_lib import (
        THREE_BODY_TREND_SERIES,
        _draw_event_medians,
        build_triple_event_traces,
    )

    all_traces = build_triple_event_traces(triple_deltas, panel, window=window)
    col_combos = [
        ("single", "single"),
        ("single", "multi"),
        ("multi",  "single"),
        ("multi",  "multi"),
    ]
    row_types = ["single", "multi"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    counts: list[dict[str, object]] = []
    for r, a_type in enumerate(row_types):
        for c, (b_type, c_type) in enumerate(col_combos):
            ax = axes[r, c]
            sub = all_traces[
                (all_traces["A_type"] == a_type)
                & (all_traces["B_type"] == b_type)
                & (all_traces["C_type"] == c_type)
            ] if not all_traces.empty else all_traces
            n_triples = int(sub["triple_id"].nunique()) if not sub.empty else 0
            ids = set(sub["triple_id"]) if not sub.empty else set()
            meta = _panel_meta_summary(triple_deltas, ids, "triple")
            title = f"A={a_type}  B={b_type}  C={c_type}\n{meta}"
            counts.append({"A_type": a_type, "B_type": b_type, "C_type": c_type, "n_triples": n_triples})
            if sub.empty:
                ax.text(0.5, 0.5, f"No triples\n(A={a_type}, B={b_type}, C={c_type})",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_title(title, fontsize=9)
                continue
            _draw_event_medians(
                ax, sub,
                window=window,
                title=title,
                series_specs=THREE_BODY_TREND_SERIES,
                event_id_col="triple_id",
                ylabel="Index ((A+B) pre-launch = 100)" if c == 0 else "",
                show_faint=True,
            )
    fig.suptitle(
        "Three-body FULL breakdown — every (A, B, C) single/multi combination (8 panels)\n"
        "Blue=A | Green=B | Red=C | Orange=A+B+C market | 100 = (A+B) pre-launch | month 0 = C launch",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pd.DataFrame(counts)


def plot_four_body_trends_by_type_combo(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """2x2 grid: median A/B/C/D/market trend for each (A_type x D_type) combo.

    Within each panel B and C are POOLED across both single and multi; the
    panel subtitle prints the B and C single/multi mix so the inner-site
    composition is visible. Full 2x2x2x2 = 16-panel breakdown is too sparse
    (only ~24 quads total), so we stick to the 2x2 with explicit mix labels.
    """
    from site_interaction_analysis_lib import (
        FOUR_BODY_TREND_SERIES,
        _draw_event_medians,
        build_quad_event_traces,
    )

    all_traces = build_quad_event_traces(quad_deltas, panel, window=window)
    combos = [
        ("single", "single"),
        ("single", "multi"),
        ("multi",  "single"),
        ("multi",  "multi"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    counts: list[dict[str, object]] = []
    for ax, (a_type, d_type) in zip(axes.flat, combos):
        sub = all_traces[(all_traces["A_type"] == a_type) & (all_traces["D_type"] == d_type)] if not all_traces.empty else all_traces
        n_quads = int(sub["quad_id"].nunique()) if not sub.empty else 0
        per_quad = sub.drop_duplicates("quad_id") if not sub.empty else sub
        b_breakdown = _triple_breakdown(per_quad, "B")
        c_breakdown = _triple_breakdown(per_quad, "C")
        ids = set(per_quad["quad_id"]) if not per_quad.empty else set()
        meta = _panel_meta_summary(quad_deltas, ids, "quad")
        title = f"A={a_type}  →  D={d_type}    ({b_breakdown};  {c_breakdown})\n{meta}"
        counts.append({"A_type": a_type, "D_type": d_type, "n_quads": n_quads})
        if sub.empty:
            ax.text(0.5, 0.5, "No quads", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=8)
            continue
        _draw_event_medians(
            ax, sub,
            window=window,
            title=title,
            series_specs=FOUR_BODY_TREND_SERIES,
            event_id_col="quad_id",
            ylabel="Index ((A+B+C) pre-launch = 100)",
            show_faint=True,
        )
    fig.suptitle(
        "Four-body trends by (A × D) — B/C mix + region/distance per panel\n"
        "Blue=A | Green=B | Orange=C | Red=D | Purple=A+B+C+D market | 100 = (A+B+C) pre-launch | month 0 = D launch",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pd.DataFrame(counts)


def plot_four_body_trends_by_full_type_combo(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """Compact grid covering only the populated (A, B, C, D) single/multi combinations.

    With only ~24 quads, most 4-type cells are empty. We enumerate all 16
    combinations, keep only the ones with >= 1 quad, and lay them out in the
    most square grid possible (ncols = ceil(sqrt(n_populated))). Each panel
    shows A/B/C/D/market traces filtered to that exact 4-type slice.
    """
    import math
    from site_interaction_analysis_lib import (
        FOUR_BODY_TREND_SERIES,
        _draw_event_medians,
        build_quad_event_traces,
    )

    all_traces = build_quad_event_traces(quad_deltas, panel, window=window)

    types = ["single", "multi"]
    populated: list[tuple[str, str, str, str, pd.DataFrame, int]] = []
    counts: list[dict[str, object]] = []
    for a_type in types:
        for b_type in types:
            for c_type in types:
                for d_type in types:
                    sub = all_traces[
                        (all_traces["A_type"] == a_type)
                        & (all_traces["B_type"] == b_type)
                        & (all_traces["C_type"] == c_type)
                        & (all_traces["D_type"] == d_type)
                    ] if not all_traces.empty else all_traces
                    n_q = int(sub["quad_id"].nunique()) if not sub.empty else 0
                    counts.append({"A_type": a_type, "B_type": b_type,
                                   "C_type": c_type, "D_type": d_type, "n_quads": n_q})
                    if n_q > 0:
                        populated.append((a_type, b_type, c_type, d_type, sub, n_q))

    n = len(populated)
    if n == 0:
        return pd.DataFrame(counts)

    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig_w = max(14.0, ncols * 5.0)
    fig_h = max(8.0,  nrows * 4.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    for idx, (a_type, b_type, c_type, d_type, sub, n_q) in enumerate(populated):
        ax = axes_flat[idx]
        ids = set(sub["quad_id"]) if not sub.empty else set()
        meta = _panel_meta_summary(quad_deltas, ids, "quad")
        title = f"A={a_type}  B={b_type}  C={c_type}  D={d_type}\n{meta}"
        _draw_event_medians(
            ax, sub,
            window=window,
            title=title,
            series_specs=FOUR_BODY_TREND_SERIES,
            event_id_col="quad_id",
            ylabel="Index ((A+B+C) pre-launch = 100)" if (idx % ncols == 0) else "",
            show_faint=True,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Four-body FULL breakdown - {n} populated (A, B, C, D) single/multi combinations (empties hidden)\n"
        "Blue=A | Green=B | Orange=C | Red=D | Purple=A+B+C+D market | 100 = (A+B+C) pre-launch | month 0 = D launch",
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pd.DataFrame(counts)


def plot_quad_examples_all_pair_theme(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    ncols: int = 9,
) -> None:
    H = _build_helpers()
    x_start, x_end = H["CALENDAR_X_START"], H["CALENDAR_X_END"]
    metric = H["PAIR_EVENT_METRIC"]

    quads = quad_deltas.sort_values(["event_month", "A_to_D_miles", "B_to_D_miles", "C_to_D_miles"]).reset_index(drop=True)
    n = len(quads)
    if n == 0:
        return
    nrows = int(np.ceil(n / ncols))
    fig_w = max(30.0, ncols * 3.6)
    fig_h = max(20.0, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    series_defs = [
        ("A (oldest)", "A_site", "A_launch_month", "#1f77b4", False),
        ("B", "B_site", "B_launch_month", "#2ca02c", True),
        ("C", "C_site", "C_launch_month", "#ff7f0e", True),
        ("D (newest)", "D_site", "D_launch_month", "#d62728", True),
    ]

    for ax, (_, q) in zip(axes_flat, quads.iterrows()):
        b_launch = H["month_floor"](q["B_launch_month"])
        c_launch = H["month_floor"](q["C_launch_month"])
        d_launch = H["month_floor"](q["D_launch_month"])
        a_launch = H["month_floor"](q["A_launch_month"])
        state = q.get("state") or H["_site_state_label"](panel, q["D_site"])
        for label, site_col, launch_col, color, from_launch in series_defs:
            min_month = H["month_floor"](q[launch_col]) if from_launch else None
            s = H["_site_calendar_series"](
                panel, q[site_col], x_start, x_end,
                from_launch=from_launch, min_calendar_month=min_month,
            )
            ax.plot(s["calendar_month"], s[metric],
                    marker="o", ms=2.5, lw=1.2, color=color, label=label)
        ax.axvline(b_launch, color="#2ca02c", ls="--", lw=0.9, alpha=0.75)
        ax.axvline(c_launch, color="#ff7f0e", ls="--", lw=0.9, alpha=0.75)
        ax.axvline(d_launch, color="#d62728", ls="--", lw=0.9, alpha=0.75)
        a_type = H["_site_type_label"](panel, q["A_site"])
        b_type = H["_site_type_label"](panel, q["B_site"])
        c_type = H["_site_type_label"](panel, q["C_site"])
        d_type = H["_site_type_label"](panel, q["D_site"])
        a_cohort, a_start, a_age = _site_meta(panel, q["A_site"])
        b_cohort, b_start, b_age = _site_meta(panel, q["B_site"])
        c_cohort, c_start, c_age = _site_meta(panel, q["C_site"])
        d_cohort, d_start, d_age = _site_meta(panel, q["D_site"])
        ax.set_title(
            f"{state} | {q['market_zip']} | A-D {q['A_to_D_miles']:.1f}mi  "
            f"B-D {q['B_to_D_miles']:.1f}mi  C-D {q['C_to_D_miles']:.1f}mi\n"
            f"A {a_launch.strftime('%Y-%m')} ({a_type}, {a_cohort}, start {a_start}, age {a_age})  "
            f"B {b_launch.strftime('%Y-%m')} ({b_type}, {b_cohort}, start {b_start}, age {b_age})\n"
            f"C {c_launch.strftime('%Y-%m')} ({c_type}, {c_cohort}, start {c_start}, age {c_age})  "
            f"D {d_launch.strftime('%Y-%m')} ({d_type}, {d_cohort}, start {d_start}, age {d_age})",
            fontsize=6,
        )
        ax.tick_params(axis="y", labelsize=6)
        H["_apply_calendar_xaxis"](ax, x_start, x_end, labelsize=5, month_interval=1)

    for ax in axes_flat[:n]:
        ax.set_xlabel("Month", fontsize=6)
    for ax in axes_flat[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("Car washes", fontsize=7)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"All four-body quads ({n}) - same area, temporal cascade A -> B -> C -> D\n"
        "Blue = A (oldest) | Green = B | Orange = C | Red = D (newest) | "
        "panel title includes type + cohort (lt2/gt2) + start date + age | "
        "dashed = each entrant's launch month",
        fontsize=14, y=1.02,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.12, hspace=0.78, wspace=0.35)
    fig.savefig(out_path, dpi=120, pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 3.  Insight plots — directly from the -nochem CSVs (own loader, native schema).
# --------------------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _load_native_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    new = pd.read_csv(SRC / LT_FILE, low_memory=False)
    old = pd.read_csv(SRC / GT_FILE, low_memory=False)
    lookup = _site_lookup()
    new = _enrich_site_meta(new, lookup)
    old = _enrich_site_meta(old, lookup)
    for df in (new, old):
        df["year_month_dt"] = pd.to_datetime(
            df["calendar_year"].astype(str)
            + "-"
            + df["calendar_month"].astype(str).str.zfill(2)
            + "-01",
            errors="coerce",
        )
    new = new[(new["year_month_dt"] >= "2024-01-01") & (new["year_month_dt"] <= "2025-12-01")].copy()
    old = old[(old["year_month_dt"] >= "2024-01-01") & (old["year_month_dt"] <= "2025-12-01")].copy()
    new = _apply_hygiene(new)
    old = _apply_hygiene(old)
    # Fallback for any residual missing client_type after location-key enrichment.
    chain = (
        pd.concat([new[["client_id", "client_id_location_id"]],
                   old[["client_id", "client_id_location_id"]]])
        .drop_duplicates().groupby("client_id").size()
    )
    inferred = chain.map(lambda n: "single_site" if n == 1 else "multi_site")
    old["client_type"] = old["client_type"].fillna(old["client_id"].map(inferred))
    old["site_type"] = old["client_type"]
    new["site_type"] = new["client_type"]
    return new, old


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _client_type_label(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower().replace(" ", "_")
    if text in {"single_site", "single"}:
        return "single"
    if text in {"multi_site", "multi"}:
        return "multi"
    return text


def generate_insight_plots(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    new, old = _load_native_frames()
    _ensure_dir(out_dir)

    # ---- Plot 1: Age lifecycle cliff (OLD only) ----
    ann = (
        old.groupby(["client_id_location_id", "age_on_30_sep_25", old["year_month_dt"].dt.year])[
            "wash_count_total"
        ]
        .mean()
        .unstack(level=2)
    )
    ann.columns = ["a2024", "a2025"]
    ann = ann.reset_index()
    ann["yoy"] = (ann["a2025"] / ann["a2024"] - 1) * 100
    counts = ann.groupby("age_on_30_sep_25").size()
    age_pts = ann.groupby("age_on_30_sep_25")["yoy"].median().reset_index()
    age_pts = age_pts[age_pts["age_on_30_sep_25"].map(counts) >= 5]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.plot(age_pts["age_on_30_sep_25"], age_pts["yoy"], "o-",
            color="#1f77b4", lw=2, ms=7, label="All sites")
    ax.axvspan(3.5, 5.5, color="orange", alpha=0.10, label="Tipping zone (age 4-5)")
    ax.set_xlabel("Site age (years, as of 30 Sep 2025)")
    ax.set_ylabel("Median YoY wash-count growth (2024 → 2025), %")
    ax.set_title("Wash-volume lifecycle: a sharp growth→decay flip near year 4-5 [chem-free]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "age_lifecycle_cliff.png", dpi=140)
    plt.close(fig)

    # ---- Plot 2: Membership-share Q1-2024 leading indicator (OLD only) ----
    od = old.copy()
    od["m_share"] = od["wash_count_membership"] / od["wash_count_total"].replace(0, np.nan)
    early = (
        od[od["year_month_dt"] <= "2024-03-01"]
        .groupby("client_id_location_id")["m_share"].mean().rename("m_q1")
    )
    ann2 = od.groupby(["client_id_location_id", od["year_month_dt"].dt.year])[
        "wash_count_total"
    ].mean().unstack()
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
    axes[1].set_ylabel("Median YoY growth 2024→2025 (%)")
    axes[1].set_title("... and predicts GROWTH a year out")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Membership share is a leading indicator [chem-free, new-schema]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "membership_leading_indicator.png", dpi=140)
    plt.close(fig)

    # ---- Plot 3: New-site ramp (multi vs single) ----
    sub = new[new["operational_start_date"].notna()].copy()
    sub["op"] = pd.to_datetime(sub["operational_start_date"])
    sub["mso"] = (
        (sub["year_month_dt"].dt.year - sub["op"].dt.year) * 12
        + (sub["year_month_dt"].dt.month - sub["op"].dt.month)
    )
    sub = sub[(sub["mso"] >= 0) & (sub["mso"] <= 11)].copy()
    sub["m_share"] = sub["wash_count_membership"] / sub["wash_count_total"].replace(0, np.nan)
    ramp = (
        sub.groupby(["client_type", "mso"])
        .agg(total=("wash_count_total", "median"),
             m_share=("m_share", "median"))
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ct, color in [("multi_site", "#d62728"), ("single_site", "#1f77b4")]:
        s = ramp[ramp["client_type"] == ct]
        axes[0].plot(s["mso"], s["total"], "o-", label=ct.replace("_", " "), color=color, lw=2)
        axes[1].plot(s["mso"], s["m_share"] * 100, "o-", label=ct.replace("_", " "), color=color, lw=2)
    axes[0].set_xlabel("Months since opening"); axes[0].set_ylabel("Median monthly washes")
    axes[0].set_title("Total washes per month"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("Months since opening"); axes[1].set_ylabel("Median membership share (%)")
    axes[1].set_title("Membership share"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("New-site ramp: multi-site vs single-site [chem-free, new-schema]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "multi_vs_single_ramp.png", dpi=140)
    plt.close(fig)

    # ---- Plot 4: Cannibalization donut ----
    old_sites = old[["client_id_location_id", "latitude", "longitude"]].dropna().drop_duplicates()
    new_sites = new[["client_id_location_id", "latitude", "longitude"]].dropna().drop_duplicates()
    oa = old_sites[["latitude", "longitude"]].values
    na = new_sites[["latitude", "longitude"]].values
    if len(na) > 0:
        min_dist = [_haversine_km(oa[i, 0], oa[i, 1], na[:, 0], na[:, 1]).min() for i in range(len(oa))]
    else:
        min_dist = [np.nan] * len(oa)
    old_sites = old_sites.assign(min_dist_to_new_km=min_dist)

    odd = old.copy()
    odd["yr"] = odd["year_month_dt"].dt.year
    a = odd.groupby(["client_id_location_id", "yr"])["wash_count_total"].mean().unstack()
    a.columns = ["avg_2024", "avg_2025"]
    a = a.dropna()
    a["yoy"] = (a["avg_2025"] / a["avg_2024"] - 1) * 100
    a = a.merge(
        old_sites[["client_id_location_id", "min_dist_to_new_km"]], on="client_id_location_id"
    )

    def db(d):
        if d < 1: return "<1 km"
        if d < 2: return "1-2 km"
        if d < 3: return "2-3 km"
        if d < 5: return "3-5 km"
        if d < 10: return "5-10 km"
        if d < 25: return "10-25 km"
        return "25+ km"

    order = ["<1 km", "1-2 km", "2-3 km", "3-5 km", "5-10 km", "10-25 km", "25+ km"]
    a["bin"] = a["min_dist_to_new_km"].apply(db)
    g = (
        a.groupby("bin")
        .agg(median_yoy=("yoy", "median"),
             pct_declining=("yoy", lambda x: (x < 0).mean() * 100))
        .reindex(order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = ["#7fbf7b" if v >= 0 else "#d73027" for v in g["median_yoy"].fillna(0)]
    axes[0].bar(order, g["median_yoy"], color=colors, alpha=0.85)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Median YoY wash growth, 2024→2025 (%)")
    axes[0].set_title("OLD-site growth vs distance to nearest NEW entrant")
    axes[0].axvspan(0.5, 3.5, color="red", alpha=0.10)
    axes[0].grid(alpha=0.3, axis="y"); axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(order, g["pct_declining"], color="#fc8d59", alpha=0.85)
    axes[1].axhline(50, color="black", lw=0.8, ls="--", label="50% line")
    axes[1].set_ylabel("% of OLD sites with negative YoY growth")
    axes[1].set_title("Share of OLD sites that declined")
    axes[1].axvspan(0.5, 3.5, color="red", alpha=0.10)
    axes[1].grid(alpha=0.3, axis="y"); axes[1].legend(); axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Cannibalization donut: peak damage at 2-5 km [chem-free, new-schema]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "cannibalization_donut.png", dpi=140)
    plt.close(fig)

    # ---- Plot 5: Great Rotation ----
    common = [
        "client_id_location_id", "client_id", "year_month_dt", "region",
        "wash_count_retail", "wash_count_membership", "wash_count_total",
    ]
    panel = pd.concat([new[common], old[common]], ignore_index=True)
    chain = (
        panel[["client_id", "client_id_location_id"]].drop_duplicates()
        .groupby("client_id").size().rename("chain_size")
    )
    panel = panel.merge(chain, on="client_id", how="left")
    panel["is_multi"] = panel["chain_size"] > 1
    panel["yr"] = panel["year_month_dt"].dt.year
    annn = panel.groupby(["client_id_location_id", "is_multi", "region", "yr"])[
        ["wash_count_retail", "wash_count_membership"]
    ].sum().reset_index()
    pr = annn.pivot_table(
        index=["client_id_location_id", "is_multi", "region"],
        columns="yr", values="wash_count_retail",
    ).rename(columns=lambda c: f"r_{c}").reset_index()
    pm = annn.pivot_table(
        index=["client_id_location_id", "is_multi", "region"],
        columns="yr", values="wash_count_membership",
    ).rename(columns=lambda c: f"m_{c}").reset_index()
    rot = pr.merge(pm, on=["client_id_location_id", "is_multi", "region"]).dropna()

    def agg_yoy(frame, a_col, b_col):
        return (frame[b_col].sum() / frame[a_col].sum() - 1) * 100

    reg = (
        rot.groupby("region")
        .apply(lambda f: pd.Series({"retail": agg_yoy(f, "r_2024", "r_2025"),
                                     "membership": agg_yoy(f, "m_2024", "m_2025")}))
        .reindex(["Northeast", "West", "South", "Midwest"])
    )
    ms = rot.groupby("is_multi").apply(
        lambda f: pd.Series({"retail": agg_yoy(f, "r_2024", "r_2025"),
                              "membership": agg_yoy(f, "m_2024", "m_2025")})
    )
    ms.index = ["Single-site (chain=1)" if not i else "Multi-site (chain>=2)" for i in ms.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(len(reg)); w = 0.35
    axes[0].bar(x - w / 2, reg["retail"], w, label="Retail washes YoY", color="#a6bddb")
    axes[0].bar(x + w / 2, reg["membership"], w, label="Membership washes YoY", color="#1c9099")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(reg.index)
    axes[0].set_ylabel("Aggregate YoY % (2024→2025)"); axes[0].set_title("By region")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    x = np.arange(len(ms))
    axes[1].bar(x - w / 2, ms["retail"], w, label="Retail washes YoY", color="#a6bddb")
    axes[1].bar(x + w / 2, ms["membership"], w, label="Membership washes YoY", color="#1c9099")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ms.index)
    axes[1].set_ylabel("Aggregate YoY % (2024→2025)"); axes[1].set_title("Multi vs Single")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("The Great Rotation: membership vs retail growth [chem-free, new-schema]", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "great_rotation.png", dpi=140)
    plt.close(fig)

    print(f"[insights-base] 5 standard insight plots -> {out_dir}")
    return new, old


# --------------------------------------------------------------------------------------
# 4.  Extra splits: ramp / donut / membership Q1 — by site type AND region.
# --------------------------------------------------------------------------------------

REGION_ORDER = ["Northeast", "South", "Midwest", "West"]
TYPE_ORDER = ["single", "multi"]


def _grid_by_type_region(figsize=(15, 7), sharey=True):
    fig, axes = plt.subplots(
        len(TYPE_ORDER), len(REGION_ORDER), figsize=figsize,
        sharex=True, sharey=sharey, squeeze=False,
    )
    return fig, axes


def extras_by_type_region(new: pd.DataFrame, old: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)

    # -------- Ramp (NEW sites, first 12 months) split by type × region --------
    sub = new[new["operational_start_date"].notna()].copy()
    sub["op"] = pd.to_datetime(sub["operational_start_date"])
    sub["mso"] = (
        (sub["year_month_dt"].dt.year - sub["op"].dt.year) * 12
        + (sub["year_month_dt"].dt.month - sub["op"].dt.month)
    )
    sub = sub[(sub["mso"] >= 0) & (sub["mso"] <= 11)].copy()
    sub["m_share"] = sub["wash_count_membership"] / sub["wash_count_total"].replace(0, np.nan)
    sub["type_lbl"] = sub["client_type"].map(_client_type_label)

    fig, axes = _grid_by_type_region(figsize=(16, 7), sharey=False)
    for r, t in enumerate(TYPE_ORDER):
        for c, reg in enumerate(REGION_ORDER):
            ax = axes[r, c]
            cell = sub[(sub["type_lbl"] == t) & (sub["region"] == reg)]
            n_sites = cell["client_id_location_id"].nunique()
            if cell.empty:
                ax.text(0.5, 0.5, "no sites", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                agg = cell.groupby("mso").agg(
                    total=("wash_count_total", "median"),
                    m_share=("m_share", "median"),
                ).reset_index()
                ax2 = ax.twinx()
                ax.plot(agg["mso"], agg["total"], "o-", color="#1f77b4", lw=2,
                        label="Median monthly washes")
                ax2.plot(agg["mso"], agg["m_share"] * 100, "s--", color="#d62728", lw=1.6, ms=4,
                         label="Median membership share %")
                ax2.tick_params(axis="y", colors="#d62728", labelsize=7)
                ax2.set_ylim(0, 100)
                if c == len(REGION_ORDER) - 1:
                    ax2.set_ylabel("Membership %", color="#d62728", fontsize=8)
            ax.set_title(f"{t}  ×  {reg}  (n={n_sites})", fontsize=9)
            ax.set_xticks(range(0, 12, 2))
            ax.grid(alpha=0.3)
            if r == len(TYPE_ORDER) - 1:
                ax.set_xlabel("Months since opening")
            if c == 0:
                ax.set_ylabel("Median monthly washes (blue)", color="#1f77b4", fontsize=8)
                ax.tick_params(axis="y", colors="#1f77b4")
    fig.suptitle(
        "New-site ramp by site type × region (NEW sites, first 12 months)  "
        "— blue: median total washes; red dashed: membership share %",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "multi_vs_single_ramp__by_type_region.png", dpi=140)
    plt.close(fig)

    # -------- Cannibalization donut split by old-site type × region --------
    new_sites = new[["client_id_location_id", "latitude", "longitude"]].dropna().drop_duplicates()
    old_sites = (
        old[["client_id_location_id", "client_type", "region", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"]).drop_duplicates(["client_id_location_id"])
    )
    old_sites["type_lbl"] = old_sites["client_type"].map(_client_type_label)
    if len(new_sites) > 0:
        oa = old_sites[["latitude", "longitude"]].values
        na = new_sites[["latitude", "longitude"]].values
        old_sites["min_dist_km"] = [
            _haversine_km(oa[i, 0], oa[i, 1], na[:, 0], na[:, 1]).min() for i in range(len(oa))
        ]
    else:
        old_sites["min_dist_km"] = np.nan

    odd = old.copy()
    odd["yr"] = odd["year_month_dt"].dt.year
    a = odd.groupby(["client_id_location_id", "yr"])["wash_count_total"].mean().unstack()
    a.columns = ["avg_2024", "avg_2025"]
    a = a.dropna()
    a["yoy"] = (a["avg_2025"] / a["avg_2024"] - 1) * 100
    a = a.reset_index().merge(
        old_sites[["client_id_location_id", "min_dist_km", "type_lbl", "region"]],
        on="client_id_location_id",
    )

    def db(d):
        if d < 1: return "<1"
        if d < 2: return "1-2"
        if d < 3: return "2-3"
        if d < 5: return "3-5"
        if d < 10: return "5-10"
        return "10+"

    order = ["<1", "1-2", "2-3", "3-5", "5-10", "10+"]
    a["bin"] = a["min_dist_km"].apply(db)

    fig, axes = _grid_by_type_region(figsize=(15, 6), sharey=True)
    for r, t in enumerate(TYPE_ORDER):
        for c, reg in enumerate(REGION_ORDER):
            ax = axes[r, c]
            cell = a[(a["type_lbl"] == t) & (a["region"] == reg)]
            if cell.empty:
                ax.text(0.5, 0.5, "no sites", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                g = cell.groupby("bin")["yoy"].median().reindex(order)
                counts = cell.groupby("bin").size().reindex(order).fillna(0).astype(int)
                colors = ["#7fbf7b" if v >= 0 else "#d73027" for v in g.fillna(0)]
                ax.bar(order, g.values, color=colors, alpha=0.85)
                for i, (val, n) in enumerate(zip(g.values, counts.values)):
                    if pd.notna(val):
                        ax.text(i, val, f"{int(n)}", ha="center",
                                va="bottom" if val >= 0 else "top", fontsize=7)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{t}  ×  {reg}  (n={int(cell.shape[0])})", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            ax.tick_params(axis="x", rotation=15)
            if c == 0:
                ax.set_ylabel("Median YoY % (2024→2025)", fontsize=8)
            if r == len(TYPE_ORDER) - 1:
                ax.set_xlabel("Distance to nearest NEW entrant (km)")
    fig.suptitle(
        "Cannibalization by old-site type × region  —  median YoY 2024→2025 vs distance to nearest new entrant",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cannibalization_donut__by_type_region.png", dpi=140)
    plt.close(fig)

    # -------- Membership-share Q1-2024 leading indicator split by type × region --------
    od = old.copy()
    od["m_share"] = od["wash_count_membership"] / od["wash_count_total"].replace(0, np.nan)
    early = (
        od[od["year_month_dt"] <= "2024-03-01"]
        .groupby("client_id_location_id")["m_share"].mean().rename("m_q1")
    )
    ann2 = od.groupby(["client_id_location_id", od["year_month_dt"].dt.year])[
        "wash_count_total"
    ].mean().unstack()
    ann2.columns = ["a2024", "a2025"]
    ann2["yoy"] = (ann2["a2025"] / ann2["a2024"] - 1) * 100
    site_meta = (
        old[["client_id_location_id", "client_type", "region"]]
        .drop_duplicates("client_id_location_id")
    )
    site_meta["type_lbl"] = site_meta["client_type"].map(_client_type_label)
    df = ann2.join(early).dropna().reset_index().merge(site_meta, on="client_id_location_id")

    fig, axes = _grid_by_type_region(figsize=(15, 6.5), sharey=False)
    for r, t in enumerate(TYPE_ORDER):
        for c, reg in enumerate(REGION_ORDER):
            ax = axes[r, c]
            cell = df[(df["type_lbl"] == t) & (df["region"] == reg)]
            if len(cell) < 5:
                ax.text(0.5, 0.5, f"n={len(cell)} (skip)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                n_bins = max(2, min(5, len(cell) // 3))
                cell = cell.copy()
                cell["bin"] = pd.qcut(cell["m_q1"], n_bins, labels=False, duplicates="drop") + 1
                gb = cell.groupby("bin").agg(
                    vol=("a2024", "median"),
                    yoy=("yoy", "median"),
                ).reset_index()
                ax2 = ax.twinx()
                ax.bar(gb["bin"], gb["vol"], color="#2ca02c", alpha=0.8, label="Median monthly washes (2024)")
                ax2.plot(gb["bin"], gb["yoy"], "o--", color="#9467bd", lw=2, ms=6,
                         label="Median YoY %")
                ax2.axhline(0, color="gray", lw=0.6, ls=":")
                ax2.tick_params(axis="y", colors="#9467bd", labelsize=7)
                if c == len(REGION_ORDER) - 1:
                    ax2.set_ylabel("YoY %", color="#9467bd", fontsize=8)
            ax.set_title(f"{t}  ×  {reg}  (n={len(cell)})", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
            if r == len(TYPE_ORDER) - 1:
                ax.set_xlabel("Q1-2024 membership-share bin (low→high)")
            if c == 0:
                ax.set_ylabel("Median monthly washes 2024", color="#2ca02c", fontsize=8)
                ax.tick_params(axis="y", colors="#2ca02c")
    fig.suptitle(
        "Membership-share Q1-2024 as leading indicator, by site type × region  "
        "(green bars = 2024 volume; purple line = YoY growth)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "membership_leading_indicator__by_type_region.png", dpi=140)
    plt.close(fig)

    print(f"[extras] type×region splits -> {out_dir}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nochem_v2_") as tmp:
        staging = Path(tmp)
        harmonize_to_legacy(staging)
        run_interaction_suite(staging)

    new, old = generate_insight_plots(OUT / "plots" / "insights_2")
    extras_by_type_region(new, old, OUT / "plots" / "insights_2")
    print(f"\nDone — all outputs in {OUT}")


if __name__ == "__main__":
    main()
