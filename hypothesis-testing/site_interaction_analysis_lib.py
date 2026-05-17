from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["wash_count_retail", "wash_count_membership", "wash_count_total"]
METRIC_LABELS = {
    "wash_count_retail": "Retail",
    "wash_count_membership": "Membership",
    "wash_count_total": "Total",
}
PAIR_EVENT_METRIC = "wash_count_total"


def configure_plotting() -> None:
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def month_floor(values: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()
    if pd.isna(values):
        return pd.NaT
    return pd.Timestamp(values).to_period("M").to_timestamp()


def month_diff(later: pd.Series | pd.Timestamp, earlier: pd.Series | pd.Timestamp) -> pd.Series | float:
    later_ts = pd.to_datetime(later, errors="coerce")
    earlier_ts = pd.to_datetime(earlier, errors="coerce")
    if isinstance(later_ts, pd.Series):
        if isinstance(earlier_ts, pd.Series):
            earlier_year = earlier_ts.dt.year
            earlier_month = earlier_ts.dt.month
        else:
            earlier_year = earlier_ts.year if pd.notna(earlier_ts) else np.nan
            earlier_month = earlier_ts.month if pd.notna(earlier_ts) else np.nan
        diff = (later_ts.dt.year - earlier_year) * 12 + (later_ts.dt.month - earlier_month)
        return diff.astype("float")
    if isinstance(earlier_ts, pd.Series):
        later_year = later_ts.year if pd.notna(later_ts) else np.nan
        later_month = later_ts.month if pd.notna(later_ts) else np.nan
        diff = (later_year - earlier_ts.dt.year) * 12 + (later_month - earlier_ts.dt.month)
        return diff.astype("float")
    if pd.isna(later_ts) or pd.isna(earlier_ts):
        return np.nan
    return float((later_ts.year - earlier_ts.year) * 12 + (later_ts.month - earlier_ts.month))


def fmt_pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.1f}%"


def fmt_num(value: float, digits: int = 1) -> str:
    return "n/a" if pd.isna(value) else f"{value:.{digits}f}"


def build_panel(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    lt2 = pd.read_csv(data_dir / "less_than-2yrs.csv", low_memory=False)
    gt2 = pd.read_csv(data_dir / "more_than-2yrs_monthly.csv", low_memory=False)

    keep = [
        "client_id_location_id",
        "client_id",
        "operational_start_date",
        "year_month",
        "wash_count_retail",
        "wash_count_membership",
        "wash_count_total",
        "latitude",
        "longitude",
        "zip",
        "state",
        "region",
        "dbscan_cluster_12km",
    ]

    lt2_panel = lt2[keep + ["client_type"]].copy()
    lt2_panel["raw_month_number"] = pd.to_numeric(lt2["month_number"], errors="coerce")
    lt2_panel["cohort"] = "lt2"

    gt2_panel = gt2[keep].copy()
    gt2_panel["client_type"] = np.nan
    gt2_panel["raw_month_number"] = np.nan
    gt2_panel["cohort"] = "gt2"

    panel = pd.concat([lt2_panel, gt2_panel], ignore_index=True)
    panel["year_month"] = month_floor(panel["year_month"])
    panel["operational_start_date"] = pd.to_datetime(panel["operational_start_date"], errors="coerce")
    panel["launch_month"] = month_floor(panel["operational_start_date"])

    for col in [
        "wash_count_retail",
        "wash_count_membership",
        "wash_count_total",
        "latitude",
        "longitude",
        "raw_month_number",
    ]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    # Prefer lt2 row when the same site-month exists in both files (rare); otherwise keep both cohorts.
    panel = panel.sort_values(["client_id_location_id", "year_month", "cohort"])
    panel = panel.drop_duplicates(["client_id_location_id", "year_month"], keep="first").reset_index(drop=True)

    panel = add_site_month_numbers(panel)
    panel["site_month_zero_based"] = panel["site_month_number"] - 1
    panel["is_prelaunch_row"] = panel["site_month_number"].le(0)
    panel = add_calendar_month(panel)
    panel = add_client_type(panel)

    lt2_check = panel.loc[panel["cohort"] == "lt2"].copy()
    impacted_mask = (
        lt2_check["raw_month_number"].notna()
        & lt2_check["site_month_number"].notna()
        & (lt2_check["raw_month_number"] != lt2_check["site_month_number"])
    )
    impacted_rows = lt2_check.loc[
        impacted_mask,
        [
            "client_id_location_id",
            "operational_start_date",
            "launch_month",
            "year_month",
            "raw_month_number",
            "site_month_number",
        ],
    ].sort_values(["client_id_location_id", "year_month"])

    gt2_panel_final = panel.loc[panel["cohort"] == "gt2"]
    lt2_panel_final = panel.loc[panel["cohort"] == "lt2"]

    validation = {
        "sites": int(panel["client_id_location_id"].nunique()),
        "rows": int(len(panel)),
        "month_min": panel["calendar_month"].min(),
        "month_max": panel["calendar_month"].max(),
        "lt2_sites": int(lt2_panel_final["client_id_location_id"].nunique()),
        "gt2_sites": int(gt2_panel_final["client_id_location_id"].nunique()),
        "lt2_rows": int(len(lt2_panel_final)),
        "gt2_rows": int(len(gt2_panel_final)),
        "lt2_month_min": lt2_panel_final["calendar_month"].min(),
        "lt2_month_max": lt2_panel_final["calendar_month"].max(),
        "gt2_month_min": gt2_panel_final["calendar_month"].min(),
        "gt2_month_max": gt2_panel_final["calendar_month"].max(),
        "lt2_rows_with_changed_month_number": int(impacted_mask.sum()),
        "lt2_sites_with_changed_month_number": int(impacted_rows["client_id_location_id"].nunique()),
        "lt2_prelaunch_rows": int(
            ((panel["cohort"] == "lt2") & panel["is_prelaunch_row"] & panel["site_month_number"].notna()).sum()
        ),
        "examples": impacted_rows.head(12),
    }
    return panel, validation


def add_site_month_numbers(panel: pd.DataFrame) -> pd.DataFrame:
    """Site month 1 = launch month (month containing operational_start_date).

    lt2: month_number in the source file is months since launch (1 = inception month).
    gt2: year_month is already a true calendar month.
    """
    panel = panel.copy()
    lt2 = panel["cohort"] == "lt2"
    gt2 = panel["cohort"] == "gt2"

    panel["site_month_number"] = np.nan
    if gt2.any():
        panel.loc[gt2, "site_month_number"] = month_diff(panel.loc[gt2, "year_month"], panel.loc[gt2, "launch_month"]) + 1

    if lt2.any():
        panel.loc[lt2, "site_month_number"] = panel.loc[lt2, "raw_month_number"]

    return panel


def format_client_type(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower().replace(" ", "_")
    if text in {"single_site", "single"}:
        return "single"
    if text in {"multi_site", "multi"}:
        return "multi"
    return text


def add_client_type(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach single_site / multi_site label per location (from lt2 client_type)."""
    panel = panel.copy()
    if "client_type" not in panel.columns:
        panel["client_type"] = np.nan

    known = panel.loc[panel["client_type"].notna(), ["client_id_location_id", "client_type"]].drop_duplicates(
        "client_id_location_id"
    )
    loc_type = known.set_index("client_id_location_id")["client_type"]
    panel["client_type"] = panel["client_id_location_id"].map(loc_type).fillna(panel["client_type"])

    client_type = (
        panel.loc[panel["client_type"].notna(), ["client_id", "client_type"]]
        .drop_duplicates("client_id")
        .set_index("client_id")["client_type"]
    )
    panel["client_type"] = panel["client_type"].fillna(panel["client_id"].map(client_type))

    locs_per_client = panel.groupby("client_id")["client_id_location_id"].nunique()
    inferred = locs_per_client.map(lambda n: "single_site" if n == 1 else "multi_site")
    panel["client_type"] = panel["client_type"].fillna(panel["client_id"].map(inferred))
    return panel


def add_calendar_month(panel: pd.DataFrame) -> pd.DataFrame:
    """Map each row to a true calendar month for plotting.

    gt2 rows already use real calendar months in year_month (2024–2025).
    lt2 year_month labels are unreliable; calendar_month = launch_month + (month_number - 1).
    """
    panel = panel.copy()
    panel["calendar_month"] = panel["year_month"]

    lt2 = panel["cohort"] == "lt2"
    operational = lt2 & panel["site_month_number"].ge(1) & panel["launch_month"].notna()
    if operational.any():
        offsets = (panel.loc[operational, "site_month_number"] - 1).astype(int)
        panel.loc[operational, "calendar_month"] = [
            launch_month + pd.DateOffset(months=int(offset))
            for launch_month, offset in zip(panel.loc[operational, "launch_month"], offsets)
        ]
    return panel


def build_sites(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    sites = (
        panel.dropna(subset=["latitude", "longitude"])
        .groupby("client_id_location_id")
        .agg(
            client_id=("client_id", "first"),
            start_date=("operational_start_date", "min"),
            launch_month=("launch_month", "min"),
            lat=("latitude", "first"),
            lon=("longitude", "first"),
            zip=("zip", "first"),
            state=("state", "first"),
            region=("region", "first"),
            cohort=("cohort", "first"),
            client_type=("client_type", "first"),
            n_months=("year_month", "nunique"),
            first_obs=("year_month", "min"),
            last_obs=("year_month", "max"),
            prelaunch_rows=("is_prelaunch_row", "sum"),
        )
        .reset_index()
        .sort_values("client_id_location_id")
        .reset_index(drop=True)
    )
    sites["has_launch_month"] = sites["launch_month"].notna()
    sites["zip_peer_count"] = sites.groupby("zip")["client_id_location_id"].transform("size") - 1
    site_lookup = sites.set_index("client_id_location_id")
    site_distances = build_distance_matrix(sites)
    return sites, site_lookup, site_distances


def haversine_miles(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius_miles = 3958.7613
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_miles * np.arcsin(np.sqrt(a))


def build_distance_matrix(sites: pd.DataFrame) -> np.ndarray:
    coords = sites[["lat", "lon"]].to_numpy(float)
    lat1 = coords[:, 0][:, None]
    lon1 = coords[:, 1][:, None]
    lat2 = coords[:, 0][None, :]
    lon2 = coords[:, 1][None, :]
    return haversine_miles(lat1, lon1, lat2, lon2)


def find_pairs(
    sites: pd.DataFrame,
    site_distances: np.ndarray,
    max_neighbor_miles: float = 10.0,
    pre_buffer_months: int = 6,
) -> pd.DataFrame:
    pairs: list[dict[str, object]] = []
    for i, new_row in sites.iterrows():
        if not new_row["has_launch_month"]:
            continue

        cutoff = new_row["launch_month"] - pd.DateOffset(months=pre_buffer_months)
        cand = sites[(sites.index != i) & sites["has_launch_month"] & (sites["launch_month"] <= cutoff)].copy()
        if cand.empty:
            continue

        cand["distance_miles"] = site_distances[i, cand.index]
        cand = cand[cand["distance_miles"] <= max_neighbor_miles].sort_values(["distance_miles", "launch_month"])
        if cand.empty:
            continue

        same_zip = cand[cand["zip"] == new_row["zip"]]
        if not same_zip.empty:
            cand = same_zip
            match_rule = "same_zip_preferred"
        else:
            match_rule = "nearest_within_radius"

        chosen = cand.iloc[0]
        pairs.append(
            {
                "market_zip": new_row["zip"],
                "match_rule": match_rule,
                "zip_match": bool(chosen["zip"] == new_row["zip"]),
                "new_site": new_row["client_id_location_id"],
                "new_client": new_row["client_id"],
                "new_start": new_row["start_date"],
                "new_launch_month": new_row["launch_month"],
                "new_zip": new_row["zip"],
                "existing_site": chosen["client_id_location_id"],
                "existing_client": chosen["client_id"],
                "existing_start": chosen["start_date"],
                "existing_launch_month": chosen["launch_month"],
                "existing_zip": chosen["zip"],
                "distance_miles": float(chosen["distance_miles"]),
                "state": new_row["state"],
                "region": new_row["region"],
                "new_cohort": new_row.get("cohort"),
                "existing_cohort": chosen.get("cohort"),
            }
        )

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pairs_df

    pairs_df = pairs_df.sort_values("distance_miles").reset_index(drop=True)
    pairs_df["distance_band"] = pd.cut(
        pairs_df["distance_miles"],
        bins=[0, 3, 5, 8, 10],
        labels=["0-3 mi", "3-5 mi", "5-8 mi", "8-10 mi"],
        include_lowest=True,
    )
    return pairs_df


def window_mean(
    panel_df: pd.DataFrame,
    site_id: str,
    event_month: pd.Timestamp,
    side: str,
    months: int,
    metrics: list[str] | None = None,
) -> tuple[dict[str, float], int]:
    metrics = metrics or METRICS
    event_month = month_floor(event_month)
    sub = panel_df[panel_df["client_id_location_id"] == site_id].copy()
    month_col = sub["calendar_month"] if "calendar_month" in sub.columns else sub["year_month"]
    if side == "pre":
        mask = (month_col < event_month) & (month_col >= event_month - pd.DateOffset(months=months))
    else:
        mask = (month_col >= event_month) & (month_col < event_month + pd.DateOffset(months=months))
    window = sub.loc[mask]
    if window.empty:
        return {metric: np.nan for metric in metrics}, 0
    return {metric: window[metric].mean() for metric in metrics}, int(len(window))


def build_pair_deltas(
    panel: pd.DataFrame,
    pairs_df: pd.DataFrame,
    pre_post_window: int = 6,
    min_months: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, pair in pairs_df.iterrows():
        event_month = pair["new_launch_month"]
        ex_pre, n_pre = window_mean(panel, pair["existing_site"], event_month, "pre", pre_post_window)
        ex_post, n_post = window_mean(panel, pair["existing_site"], event_month, "post", pre_post_window)
        nw_post, n_new_post = window_mean(panel, pair["new_site"], event_month, "post", pre_post_window)
        if n_pre < min_months or n_post < min_months:
            continue

        row = {
            **pair.to_dict(),
            "event_month": event_month,
            "n_pre_months": n_pre,
            "n_post_months": n_post,
            "n_new_post_months": n_new_post,
        }
        for metric in METRICS:
            row[f"existing_pre_{metric}"] = ex_pre[metric]
            row[f"existing_post_{metric}"] = ex_post[metric]
            row[f"existing_delta_{metric}"] = ex_post[metric] - ex_pre[metric]
            row[f"existing_pct_{metric}"] = pct_change(ex_pre[metric], ex_post[metric])
            row[f"new_post_{metric}"] = nw_post[metric]
            row[f"combined_post_{metric}"] = safe_add(ex_post[metric], nw_post[metric])
            row[f"combined_pct_{metric}"] = pct_change(ex_pre[metric], row[f"combined_post_{metric}"])
        rows.append(row)

    pair_deltas = pd.DataFrame(rows)
    if pair_deltas.empty:
        return pair_deltas

    pair_deltas["regime"] = pair_deltas.apply(pair_regime, axis=1)
    return pair_deltas.sort_values(["distance_miles", "event_month"]).reset_index(drop=True)


def pct_change(before: float, after: float) -> float:
    if pd.isna(before) or before <= 0 or pd.isna(after):
        return np.nan
    return (after - before) / before * 100


def safe_add(*values: float) -> float:
    total = 0.0
    saw_value = False
    for value in values:
        if pd.notna(value):
            total += float(value)
            saw_value = True
    return total if saw_value else np.nan


def pair_regime(row: pd.Series) -> str:
    existing = row["existing_pct_wash_count_total"]
    combined = row["combined_pct_wash_count_total"]
    if pd.isna(existing) or pd.isna(combined):
        return "unknown"
    if existing <= -5 and combined <= 5:
        return "mostly cannibalization"
    if existing <= -5 and combined > 5:
        return "cannibalization plus expansion"
    if existing > -5 and combined > 5:
        return "market expansion"
    return "flat to mixed"


def summarize_pairs(pair_deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(pair_deltas[f"existing_pct_{metric}"].notna().sum()),
                "existing_median_pct": pair_deltas[f"existing_pct_{metric}"].median(),
                "existing_mean_pct": pair_deltas[f"existing_pct_{metric}"].mean(),
                "combined_median_pct": pair_deltas[f"combined_pct_{metric}"].median(),
                "combined_mean_pct": pair_deltas[f"combined_pct_{metric}"].mean(),
                "new_site_post_mean": pair_deltas[f"new_post_{metric}"].mean(),
                "same_zip_pairs": int(pair_deltas["zip_match"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_pair_event_profile(pair_deltas: pd.DataFrame, panel: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, pair in pair_deltas.iterrows():
        event_month = pair["event_month"]
        pre_base, _ = window_mean(panel, pair["existing_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
        base = pre_base[PAIR_EVENT_METRIC]
        if pd.isna(base) or base <= 0:
            continue

        existing = (
            panel.loc[panel["client_id_location_id"] == pair["existing_site"], ["calendar_month", PAIR_EVENT_METRIC]]
            .assign(relative_month=lambda df: month_diff(df["calendar_month"], event_month).astype(int))
        )
        new_site = (
            panel.loc[
                (panel["client_id_location_id"] == pair["new_site"]) & panel["site_month_number"].ge(1),
                ["calendar_month", PAIR_EVENT_METRIC],
            ]
            .assign(relative_month=lambda df: month_diff(df["calendar_month"], event_month).astype(int))
        )
        merged = existing.merge(new_site, on="relative_month", how="outer", suffixes=("_existing", "_new"))
        merged = merged[merged["relative_month"].between(-window, window)]
        for _, row in merged.iterrows():
            relative_month = int(row["relative_month"])
            new_value = row.get(f"{PAIR_EVENT_METRIC}_new", np.nan)
            if relative_month < 0:
                new_value = np.nan
            existing_value = row.get(f"{PAIR_EVENT_METRIC}_existing", np.nan)
            combined_value = safe_add(existing_value, new_value if pd.notna(new_value) else np.nan)
            rows.append(
                {
                    "pair_id": f"{pair['existing_site']}->{pair['new_site']}",
                    "relative_month": relative_month,
                    "existing_index": existing_value / base * 100 if pd.notna(existing_value) else np.nan,
                    "new_index": new_value / base * 100 if pd.notna(new_value) else np.nan,
                    "combined_index": combined_value / base * 100 if pd.notna(combined_value) else np.nan,
                }
            )

    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    return (
        profile.groupby("relative_month")
        .agg(
            existing_index=("existing_index", "median"),
            new_index=("new_index", "median"),
            combined_index=("combined_index", "median"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
        .sort_values("relative_month")
    )


def choose_pair_examples(pair_deltas: pd.DataFrame, n_examples: int = 4) -> pd.DataFrame:
    if pair_deltas.empty:
        return pair_deltas
    ranked = pair_deltas[(pair_deltas["n_pre_months"] >= 4) & (pair_deltas["n_post_months"] >= 4)].copy()
    ranked["priority"] = (~ranked["zip_match"]).astype(int)
    return ranked.sort_values(["priority", "distance_miles", "event_month"]).head(n_examples).reset_index(drop=True)


def find_triples(
    sites: pd.DataFrame,
    site_distances: np.ndarray,
    max_neighbor_miles: float = 10.0,
    pre_buffer_months: int = 6,
) -> pd.DataFrame:
    triples: list[dict[str, object]] = []
    for i, c_row in sites.iterrows():
        if not c_row["has_launch_month"]:
            continue

        older = sites[
            (sites.index != i)
            & sites["has_launch_month"]
            & (sites["launch_month"] <= c_row["launch_month"] - pd.DateOffset(months=pre_buffer_months))
        ].copy()
        if len(older) < 2:
            continue

        older["distance_to_c_miles"] = site_distances[i, older.index]
        older = older[older["distance_to_c_miles"] <= max_neighbor_miles]
        if len(older) < 2:
            continue

        same_zip = older[older["zip"] == c_row["zip"]]
        pool = same_zip if len(same_zip) >= 2 else older
        selection_rule = "same_zip_preferred" if len(same_zip) >= 2 else "nearest_within_radius"
        pool = pool.sort_values(["distance_to_c_miles", "launch_month"]).reset_index().rename(columns={"index": "site_index"})

        best = None
        for left in range(len(pool)):
            for right in range(left + 1, len(pool)):
                s1 = pool.iloc[left]
                s2 = pool.iloc[right]
                gap = abs(month_diff(s2["launch_month"], s1["launch_month"]))
                if pd.isna(gap) or gap < pre_buffer_months:
                    continue
                a_row, b_row = sorted([s1, s2], key=lambda item: item["launch_month"])
                score = (
                    a_row["distance_to_c_miles"] + b_row["distance_to_c_miles"],
                    max(a_row["distance_to_c_miles"], b_row["distance_to_c_miles"]),
                )
                if best is None or score < best[0]:
                    best = (score, a_row, b_row)

        if best is None:
            continue

        _, a_row, b_row = best
        triples.append(
            {
                "market_zip": c_row["zip"],
                "selection_rule": selection_rule,
                "same_zip_triplet": bool(a_row["zip"] == b_row["zip"] == c_row["zip"]),
                "A_site": a_row["client_id_location_id"],
                "A_start": a_row["start_date"],
                "A_launch_month": a_row["launch_month"],
                "A_lat": a_row["lat"],
                "A_lon": a_row["lon"],
                "A_zip": a_row["zip"],
                "B_site": b_row["client_id_location_id"],
                "B_start": b_row["start_date"],
                "B_launch_month": b_row["launch_month"],
                "B_lat": b_row["lat"],
                "B_lon": b_row["lon"],
                "B_zip": b_row["zip"],
                "C_site": c_row["client_id_location_id"],
                "C_start": c_row["start_date"],
                "C_launch_month": c_row["launch_month"],
                "C_lat": c_row["lat"],
                "C_lon": c_row["lon"],
                "C_zip": c_row["zip"],
                "A_to_C_miles": float(a_row["distance_to_c_miles"]),
                "B_to_C_miles": float(b_row["distance_to_c_miles"]),
                "AB_miles": float(site_distances[int(a_row["site_index"]), int(b_row["site_index"])]),
                "state": c_row["state"],
                "region": c_row["region"],
                "A_cohort": a_row.get("cohort"),
                "B_cohort": b_row.get("cohort"),
                "C_cohort": c_row.get("cohort"),
            }
        )

    triples_df = pd.DataFrame(triples)
    if triples_df.empty:
        return triples_df
    return triples_df.sort_values(["A_to_C_miles", "B_to_C_miles"]).reset_index(drop=True)


def build_triple_deltas(
    panel: pd.DataFrame,
    triples_df: pd.DataFrame,
    pre_post_window: int = 6,
    min_months: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, triple in triples_df.iterrows():
        event_month = triple["C_launch_month"]
        out = {**triple.to_dict(), "event_month": event_month}
        valid = True
        for label, site_col in [("A", "A_site"), ("B", "B_site")]:
            pre, n_pre = window_mean(panel, triple[site_col], event_month, "pre", pre_post_window)
            post, n_post = window_mean(panel, triple[site_col], event_month, "post", pre_post_window)
            out[f"{label}_n_pre"] = n_pre
            out[f"{label}_n_post"] = n_post
            if n_pre < min_months or n_post < min_months:
                valid = False
            for metric in METRICS:
                out[f"{label}_pre_{metric}"] = pre[metric]
                out[f"{label}_post_{metric}"] = post[metric]
                out[f"{label}_pct_{metric}"] = pct_change(pre[metric], post[metric])

        if not valid:
            continue

        c_post, c_n_post = window_mean(panel, triple["C_site"], event_month, "post", pre_post_window)
        out["C_n_post"] = c_n_post
        for metric in METRICS:
            out[f"C_post_{metric}"] = c_post[metric]
            out[f"region_pre_{metric}"] = safe_add(out[f"A_pre_{metric}"], out[f"B_pre_{metric}"])
            out[f"region_post_{metric}"] = safe_add(
                out[f"A_post_{metric}"],
                out[f"B_post_{metric}"],
                out[f"C_post_{metric}"],
            )
            out[f"region_pct_{metric}"] = pct_change(out[f"region_pre_{metric}"], out[f"region_post_{metric}"])
        rows.append(out)

    triple_deltas = pd.DataFrame(rows)
    if triple_deltas.empty:
        return triple_deltas
    return triple_deltas.sort_values(["A_to_C_miles", "B_to_C_miles", "event_month"]).reset_index(drop=True)


def find_quads(
    sites: pd.DataFrame,
    site_distances: np.ndarray,
    max_neighbor_miles: float = 10.0,
    pre_buffer_months: int = 6,
) -> pd.DataFrame:
    quads: list[dict[str, object]] = []
    for i, d_row in sites.iterrows():
        if not d_row["has_launch_month"]:
            continue

        older = sites[
            (sites.index != i)
            & sites["has_launch_month"]
            & (sites["launch_month"] <= d_row["launch_month"] - pd.DateOffset(months=pre_buffer_months))
        ].copy()
        if len(older) < 3:
            continue

        older["distance_to_d_miles"] = site_distances[i, older.index]
        older = older[older["distance_to_d_miles"] <= max_neighbor_miles]
        if len(older) < 3:
            continue

        same_zip = older[older["zip"] == d_row["zip"]]
        pool = same_zip if len(same_zip) >= 3 else older
        selection_rule = "same_zip_preferred" if len(same_zip) >= 3 else "nearest_within_radius"
        pool = pool.sort_values(["distance_to_d_miles", "launch_month"]).reset_index().rename(columns={"index": "site_index"})

        best = None
        n_pool = len(pool)
        for left in range(n_pool):
            for mid in range(left + 1, n_pool):
                for right in range(mid + 1, n_pool):
                    s1, s2, s3 = pool.iloc[left], pool.iloc[mid], pool.iloc[right]
                    gap12 = abs(month_diff(s2["launch_month"], s1["launch_month"]))
                    gap23 = abs(month_diff(s3["launch_month"], s2["launch_month"]))
                    if pd.isna(gap12) or pd.isna(gap23) or gap12 < pre_buffer_months or gap23 < pre_buffer_months:
                        continue
                    a_row, b_row, c_row = sorted([s1, s2, s3], key=lambda item: item["launch_month"])
                    score = (
                        a_row["distance_to_d_miles"] + b_row["distance_to_d_miles"] + c_row["distance_to_d_miles"],
                        max(a_row["distance_to_d_miles"], b_row["distance_to_d_miles"], c_row["distance_to_d_miles"]),
                    )
                    if best is None or score < best[0]:
                        best = (score, a_row, b_row, c_row)

        if best is None:
            continue

        _, a_row, b_row, c_row = best
        quads.append(
            {
                "market_zip": d_row["zip"],
                "selection_rule": selection_rule,
                "same_zip_quad": bool(a_row["zip"] == b_row["zip"] == c_row["zip"] == d_row["zip"]),
                "A_site": a_row["client_id_location_id"],
                "A_start": a_row["start_date"],
                "A_launch_month": a_row["launch_month"],
                "A_lat": a_row["lat"],
                "A_lon": a_row["lon"],
                "A_zip": a_row["zip"],
                "B_site": b_row["client_id_location_id"],
                "B_start": b_row["start_date"],
                "B_launch_month": b_row["launch_month"],
                "B_lat": b_row["lat"],
                "B_lon": b_row["lon"],
                "B_zip": b_row["zip"],
                "C_site": c_row["client_id_location_id"],
                "C_start": c_row["start_date"],
                "C_launch_month": c_row["launch_month"],
                "C_lat": c_row["lat"],
                "C_lon": c_row["lon"],
                "C_zip": c_row["zip"],
                "D_site": d_row["client_id_location_id"],
                "D_start": d_row["start_date"],
                "D_launch_month": d_row["launch_month"],
                "D_lat": d_row["lat"],
                "D_lon": d_row["lon"],
                "D_zip": d_row["zip"],
                "A_to_D_miles": float(a_row["distance_to_d_miles"]),
                "B_to_D_miles": float(b_row["distance_to_d_miles"]),
                "C_to_D_miles": float(c_row["distance_to_d_miles"]),
                "AB_miles": float(site_distances[int(a_row["site_index"]), int(b_row["site_index"])]),
                "AC_miles": float(site_distances[int(a_row["site_index"]), int(c_row["site_index"])]),
                "BC_miles": float(site_distances[int(b_row["site_index"]), int(c_row["site_index"])]),
                "state": d_row["state"],
                "region": d_row["region"],
                "A_cohort": a_row.get("cohort"),
                "B_cohort": b_row.get("cohort"),
                "C_cohort": c_row.get("cohort"),
                "D_cohort": d_row.get("cohort"),
            }
        )

    quads_df = pd.DataFrame(quads)
    if quads_df.empty:
        return quads_df
    return quads_df.sort_values(["A_to_D_miles", "B_to_D_miles", "C_to_D_miles"]).reset_index(drop=True)


def build_quad_deltas(
    panel: pd.DataFrame,
    quads_df: pd.DataFrame,
    pre_post_window: int = 6,
    min_months: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, quad in quads_df.iterrows():
        event_month = quad["D_launch_month"]
        out = {**quad.to_dict(), "event_month": event_month}
        valid = True
        for label, site_col in [("A", "A_site"), ("B", "B_site"), ("C", "C_site")]:
            pre, n_pre = window_mean(panel, quad[site_col], event_month, "pre", pre_post_window)
            post, n_post = window_mean(panel, quad[site_col], event_month, "post", pre_post_window)
            out[f"{label}_n_pre"] = n_pre
            out[f"{label}_n_post"] = n_post
            if n_pre < min_months or n_post < min_months:
                valid = False
            for metric in METRICS:
                out[f"{label}_pre_{metric}"] = pre[metric]
                out[f"{label}_post_{metric}"] = post[metric]
                out[f"{label}_pct_{metric}"] = pct_change(pre[metric], post[metric])

        if not valid:
            continue

        d_post, d_n_post = window_mean(panel, quad["D_site"], event_month, "post", pre_post_window)
        out["D_n_post"] = d_n_post
        for metric in METRICS:
            out[f"D_post_{metric}"] = d_post[metric]
            out[f"region_pre_{metric}"] = safe_add(
                out[f"A_pre_{metric}"],
                out[f"B_pre_{metric}"],
                out[f"C_pre_{metric}"],
            )
            out[f"region_post_{metric}"] = safe_add(
                out[f"A_post_{metric}"],
                out[f"B_post_{metric}"],
                out[f"C_post_{metric}"],
                out[f"D_post_{metric}"],
            )
            out[f"region_pct_{metric}"] = pct_change(out[f"region_pre_{metric}"], out[f"region_post_{metric}"])
        rows.append(out)

    quad_deltas = pd.DataFrame(rows)
    if quad_deltas.empty:
        return quad_deltas
    return quad_deltas.sort_values(["A_to_D_miles", "B_to_D_miles", "C_to_D_miles", "event_month"]).reset_index(drop=True)


def summarize_quads(quad_deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "n_quads": int(len(quad_deltas)),
                "A_median_pct": quad_deltas[f"A_pct_{metric}"].median(),
                "B_median_pct": quad_deltas[f"B_pct_{metric}"].median(),
                "C_median_pct": quad_deltas[f"C_pct_{metric}"].median(),
                "region_median_pct": quad_deltas[f"region_pct_{metric}"].median(),
                "A_mean_pct": quad_deltas[f"A_pct_{metric}"].mean(),
                "B_mean_pct": quad_deltas[f"B_pct_{metric}"].mean(),
                "C_mean_pct": quad_deltas[f"C_pct_{metric}"].mean(),
                "region_mean_pct": quad_deltas[f"region_pct_{metric}"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_triples(triple_deltas: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "n_triples": int(len(triple_deltas)),
                "A_median_pct": triple_deltas[f"A_pct_{metric}"].median(),
                "B_median_pct": triple_deltas[f"B_pct_{metric}"].median(),
                "region_median_pct": triple_deltas[f"region_pct_{metric}"].median(),
                "A_mean_pct": triple_deltas[f"A_pct_{metric}"].mean(),
                "B_mean_pct": triple_deltas[f"B_pct_{metric}"].mean(),
                "region_mean_pct": triple_deltas[f"region_pct_{metric}"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_triple_event_profile(triple_deltas: pd.DataFrame, panel: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, triple in triple_deltas.iterrows():
        event_month = triple["event_month"]
        base_a, _ = window_mean(panel, triple["A_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
        base_b, _ = window_mean(panel, triple["B_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
        base = safe_add(base_a[PAIR_EVENT_METRIC], base_b[PAIR_EVENT_METRIC])
        if pd.isna(base) or base <= 0:
            continue

        traces = {
            "A": _site_event_wash_series(panel, triple["A_site"], event_month),
            "B": _site_event_wash_series(panel, triple["B_site"], event_month),
            "C": _site_event_wash_series(panel, triple["C_site"], event_month, newest_site=True),
        }

        for relative_month in range(-window, window + 1):
            a_value = traces["A"].get(relative_month, np.nan)
            b_value = traces["B"].get(relative_month, np.nan)
            c_value = traces["C"].get(relative_month, np.nan)
            rows.append(
                {
                    "triple_id": f"{triple['A_site']}|{triple['B_site']}|{triple['C_site']}",
                    "relative_month": relative_month,
                    "A_index": a_value / base * 100 if pd.notna(a_value) else np.nan,
                    "B_index": b_value / base * 100 if pd.notna(b_value) else np.nan,
                    "C_index": c_value / base * 100 if pd.notna(c_value) else np.nan,
                    "region_index": safe_add(a_value, b_value, c_value) / base * 100
                    if pd.notna(safe_add(a_value, b_value, c_value))
                    else np.nan,
                }
            )

    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    return (
        profile.groupby("relative_month")
        .agg(
            A_index=("A_index", "median"),
            B_index=("B_index", "median"),
            C_index=("C_index", "median"),
            region_index=("region_index", "median"),
            n_triples=("triple_id", "nunique"),
        )
        .reset_index()
        .sort_values("relative_month")
    )


def choose_triple_examples(triple_deltas: pd.DataFrame, n_examples: int = 4) -> pd.DataFrame:
    if triple_deltas.empty:
        return triple_deltas
    show = triple_deltas.assign(max_to_c=triple_deltas[["A_to_C_miles", "B_to_C_miles"]].max(axis=1))
    return show.sort_values(["max_to_c", "event_month"]).head(n_examples).reset_index(drop=True)


def plot_pair_summary(pair_deltas: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, METRICS):
        series_map = {
            "Existing site": pair_deltas[f"existing_pct_{metric}"],
            "Combined market": pair_deltas[f"combined_pct_{metric}"],
        }
        for y_pos, (label, series), color in zip([1, 0], series_map.items(), ["#1f77b4", "#ff7f0e"]):
            clean = series.replace([np.inf, -np.inf], np.nan).dropna()
            q1, med, q3 = clean.quantile([0.25, 0.5, 0.75])
            ax.hlines(y_pos, q1, q3, color=color, lw=7, alpha=0.35)
            ax.scatter(med, y_pos, s=80, color=color, zorder=3)
            ax.text(q3 + 3, y_pos, f"median {med:+.1f}%", va="center", fontsize=8)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_yticks([0, 1], ["Combined market", "Existing site"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("% change: 6 months before vs 6 months after")
    fig.suptitle("Two-body impact summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pair_event_profile(profile: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(profile["relative_month"], profile["existing_index"], marker="o", lw=2, color="#1f77b4", label="Existing site")
    ax.plot(profile["relative_month"], profile["new_index"], marker="o", lw=2, color="#d62728", label="New site")
    ax.plot(profile["relative_month"], profile["combined_index"], marker="o", lw=2, color="#ff7f0e", label="Combined market")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xlabel("Months relative to the new site's launch month")
    ax.set_ylabel("Index (existing site's pre-launch mean = 100)")
    ax.set_title("Two-body event-time profile for total washes")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pair_distance_bands(pair_deltas: pd.DataFrame, out_path: Path) -> None:
    ordered = ["0-3 mi", "3-5 mi", "5-8 mi", "8-10 mi"]
    data = [
        pair_deltas.loc[pair_deltas["distance_band"] == band, "existing_pct_wash_count_total"].dropna()
        for band in ordered
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    box = ax.boxplot(data, labels=ordered, patch_artist=True, widths=0.55)
    for patch in box["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.65)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Distance band")
    ax.set_ylabel("Existing-site % change in total washes")
    ax.set_title("Two-body effect by distance band")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


CALENDAR_X_START = pd.Timestamp("2024-01-01")
CALENDAR_X_END = pd.Timestamp("2025-12-01")


def _site_type_label(panel: pd.DataFrame, site_id: str) -> str:
    values = panel.loc[panel["client_id_location_id"] == site_id, "client_type"].dropna()
    if values.empty:
        return "unknown"
    return format_client_type(values.iloc[0])


def _site_state_label(panel: pd.DataFrame, site_id: str) -> str:
    values = panel.loc[panel["client_id_location_id"] == site_id, "state"].dropna()
    if values.empty:
        return "?"
    return str(values.iloc[0])


def _apply_calendar_xaxis(
    ax: plt.Axes,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    *,
    labelsize: int = 6,
    month_interval: int = 2,
) -> None:
    """Force visible month labels on every subplot (do not rely on sharex)."""
    ax.set_xlim(x_start, x_end)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", which="both", labelbottom=True, labelsize=labelsize, rotation=90, pad=1)
    for label in ax.get_xticklabels():
        label.set_visible(True)


def _site_calendar_series(
    panel: pd.DataFrame,
    site_id: str,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    *,
    from_launch: bool = False,
    min_calendar_month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    sub = panel[panel["client_id_location_id"] == site_id].copy()
    if from_launch:
        sub = sub[sub["site_month_number"].ge(1)]
    if min_calendar_month is not None:
        sub = sub[sub["calendar_month"].ge(month_floor(min_calendar_month))]
    sub = sub[sub["calendar_month"].between(x_start, x_end)]
    return sub.sort_values("calendar_month")


def filter_pairs_by_site_types(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    existing_type: str | None = None,
    new_type: str | None = None,
) -> pd.DataFrame:
    """Filter pairs by formatted client_type labels (single / multi / unknown)."""
    if pair_deltas.empty:
        return pair_deltas
    existing_type = format_client_type(existing_type) if existing_type else None
    new_type = format_client_type(new_type) if new_type else None

    keep: list[bool] = []
    for _, pair in pair_deltas.iterrows():
        ex_ok = existing_type is None or _site_type_label(panel, pair["existing_site"]) == existing_type
        nw_ok = new_type is None or _site_type_label(panel, pair["new_site"]) == new_type
        keep.append(ex_ok and nw_ok)
    return pair_deltas.loc[keep].reset_index(drop=True)


def plot_pair_examples_all(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    x_start: pd.Timestamp = CALENDAR_X_START,
    x_end: pd.Timestamp = CALENDAR_X_END,
    ncols: int = 9,
    *,
    suptitle: str | None = None,
) -> None:
    """One subplot per pair: existing vs new monthly washes on a shared calendar axis.

    Uses calendar_month (Jan 2024 – Dec 2025). For lt2 sites, month_number is months
    since launch (1 = inception month); calendar = launch_month + (month_number - 1).
    """
    pairs = pair_deltas.sort_values(["event_month", "distance_miles"]).reset_index(drop=True)
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    nrows = int(np.ceil(n_pairs / ncols))
    fig_w = max(30.0, ncols * 3.6)
    fig_h = max(30.0, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    for ax, (_, pair) in zip(axes_flat, pairs.iterrows()):
        launch_month = month_floor(pair["new_launch_month"])
        state = pair.get("state") or _site_state_label(panel, pair["new_site"])
        ex = _site_calendar_series(panel, pair["existing_site"], x_start, x_end)
        nw = _site_calendar_series(panel, pair["new_site"], x_start, x_end, from_launch=True)

        ax.plot(
            ex["calendar_month"],
            ex[PAIR_EVENT_METRIC],
            marker="o",
            ms=2.5,
            lw=1.2,
            color="#1f77b4",
            label="Existing",
        )
        ax.plot(
            nw["calendar_month"],
            nw[PAIR_EVENT_METRIC],
            marker="o",
            ms=2.5,
            lw=1.2,
            color="#d62728",
            label="New",
        )
        ax.axvline(launch_month, color="black", ls="--", lw=0.9, alpha=0.75)
        ex_type = _site_type_label(panel, pair["existing_site"])
        nw_type = _site_type_label(panel, pair["new_site"])
        ex_launch = month_floor(pair["existing_launch_month"])
        ex_cohort = pair.get("existing_cohort", "")
        nw_cohort = pair.get("new_cohort", "")
        ax.set_title(
            f"{state} | {pair['market_zip']} | {pair['distance_miles']:.1f} mi\n"
            f"existing start {ex_launch.strftime('%Y-%m')} ({ex_cohort}, {ex_type}) | "
            f"new start {launch_month.strftime('%Y-%m')} ({nw_cohort}, {nw_type})",
            fontsize=6,
        )
        ax.tick_params(axis="y", labelsize=6)
        _apply_calendar_xaxis(ax, x_start, x_end, labelsize=5, month_interval=1)

    for ax in axes_flat[:n_pairs]:
        ax.set_xlabel("Month", fontsize=6)

    for ax in axes_flat[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("Car washes", fontsize=7)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    if suptitle is None:
        suptitle = (
            f"All two-body pairs ({n_pairs}): monthly washes, Jan 2024 – Dec 2025\n"
            "Blue = existing (gt2/lt2) | Red = new (lt2) | dashed = new launch | title = operational start dates"
        )
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.12, hspace=0.78, wspace=0.35)
    fig.savefig(out_path, dpi=120, pad_inches=0.15)
    plt.close(fig)


def build_pair_event_traces(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    window: int = 6,
) -> pd.DataFrame:
    """Pair-level event-time indices (existing pre-launch mean = 100)."""
    rows: list[dict[str, float]] = []
    for _, pair in pair_deltas.iterrows():
        event_month = pair["event_month"]
        pre_base, _ = window_mean(panel, pair["existing_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
        base = pre_base[PAIR_EVENT_METRIC]
        if pd.isna(base) or base <= 0:
            continue

        existing = (
            panel.loc[panel["client_id_location_id"] == pair["existing_site"], ["calendar_month", PAIR_EVENT_METRIC]]
            .assign(relative_month=lambda df: month_diff(df["calendar_month"], event_month).astype(int))
        )
        new_site = (
            panel.loc[
                (panel["client_id_location_id"] == pair["new_site"]) & panel["site_month_number"].ge(1),
                ["calendar_month", PAIR_EVENT_METRIC],
            ]
            .assign(relative_month=lambda df: month_diff(df["calendar_month"], event_month).astype(int))
        )
        merged = existing.merge(new_site, on="relative_month", how="outer", suffixes=("_existing", "_new"))
        merged = merged[merged["relative_month"].between(-window, window)]
        pair_id = f"{pair['existing_site']}->{pair['new_site']}"
        for _, row in merged.iterrows():
            relative_month = int(row["relative_month"])
            new_value = row.get(f"{PAIR_EVENT_METRIC}_new", np.nan)
            if relative_month < 0:
                new_value = np.nan
            existing_value = row.get(f"{PAIR_EVENT_METRIC}_existing", np.nan)
            rows.append(
                {
                    "pair_id": pair_id,
                    "relative_month": relative_month,
                    "existing_index": existing_value / base * 100 if pd.notna(existing_value) else np.nan,
                    "new_index": new_value / base * 100 if pd.notna(new_value) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_pair_pattern_existing_single_new_multi(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """One summary plot: all existing-single / new-multi pairs aligned to new-site launch."""
    filtered = filter_pairs_by_site_types(
        pair_deltas, panel, existing_type="single", new_type="multi"
    )
    traces = build_pair_event_traces(filtered, panel, window=window)
    if traces.empty:
        return filtered

    fig, ax = plt.subplots(figsize=(11, 6.5))
    series_styles = [
        ("existing_index", "#1f77b4", "Existing (single)"),
        ("new_index", "#d62728", "New (multi)"),
    ]

    for col, color, label in series_styles:
        for pair_id, group in traces.groupby("pair_id"):
            sub = group.sort_values("relative_month")
            ax.plot(
                sub["relative_month"],
                sub[col],
                color=color,
                alpha=0.12,
                lw=0.8,
                zorder=1,
            )

        summary = (
            traces.groupby("relative_month")[col]
            .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("relative_month")
        )
        ax.fill_between(
            summary["relative_month"],
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.18,
            zorder=2,
        )
        ax.plot(
            summary["relative_month"],
            summary["median"],
            color=color,
            lw=2.8,
            marker="o",
            ms=5,
            label=f"{label} (median)",
            zorder=3,
        )

    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel("Months relative to new-site launch (month 0)")
    ax.set_ylabel("Wash index (existing site's pre-launch mean = 100)")
    n_pairs = int(traces["pair_id"].nunique())
    ax.set_title(
        f"Existing single vs new multi ({n_pairs} pairs): event-time pattern\n"
        "Faint lines = individual pairs | band = 25th–75th percentile | bold = median",
        fontsize=12,
    )
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return filtered


def plot_pair_examples_existing_single_new_multi(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    **kwargs: object,
) -> pd.DataFrame:
    """Grid plot for pairs where the existing site is single and the new site is multi."""
    filtered = filter_pairs_by_site_types(
        pair_deltas, panel, existing_type="single", new_type="multi"
    )
    n_pairs = len(filtered)
    plot_pair_examples_all(
        filtered,
        panel,
        out_path,
        suptitle=(
            f"Two-body pairs: existing = single, new = multi ({n_pairs}), Jan 2024 – Dec 2025\n"
            "Blue = existing (single) | Red = new (multi, from launch) | dashed = new-site launch"
        ),
        **kwargs,
    )
    return filtered


def filter_triples_newest_type(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    newest_type: str,
) -> pd.DataFrame:
    if triple_deltas.empty:
        return triple_deltas
    newest_type = newest_type.lower()
    keep = [_site_type_label(panel, row["C_site"]) == newest_type for _, row in triple_deltas.iterrows()]
    return triple_deltas.loc[keep].reset_index(drop=True)


def _site_event_wash_series(
    panel: pd.DataFrame,
    site_id: str,
    event_month: pd.Timestamp,
    *,
    newest_site: bool = False,
) -> pd.Series:
    """Washes by month relative to event; NaN before operational launch (site_month ≥ 1)."""
    sub = panel.loc[
        panel["client_id_location_id"] == site_id,
        ["calendar_month", "site_month_number", PAIR_EVENT_METRIC],
    ].copy()
    sub = sub[sub["site_month_number"].ge(1)]
    sub["relative_month"] = month_diff(sub["calendar_month"], event_month).astype(int)
    series = sub.set_index("relative_month")[PAIR_EVENT_METRIC]
    if newest_site:
        series = series.copy()
        series.loc[series.index < 0] = np.nan
    return series


# One unique hue per role×type series (solid = single, dashed = multi).
BODY_TYPE_DISTINCT_COLORS: list[str] = [
    "#00F0FF",  # A single — cyan
    "#FF3366",  # A multi — rose
    "#39FF14",  # B single — neon green
    "#FFD700",  # B multi — gold
    "#FF6600",  # C single — orange
    "#FF00FF",  # C multi — magenta
    "#00FFAA",  # D single — mint
    "#9B59FF",  # D multi — violet
]
BODY_TYPE_SERIES_ORDER: list[tuple[str, str]] = [
    ("A", "single"),
    ("A", "multi"),
    ("B", "single"),
    ("B", "multi"),
    ("C", "single"),
    ("C", "multi"),
    ("D", "single"),
    ("D", "multi"),
]
BODY_ROLE_MARKERS: dict[str, str] = {"A": "o", "B": "s", "C": "^", "D": "D"}
BODY_TYPE_LINE_STYLES: dict[str, str] = {"single": "-", "multi": (0, (6, 3))}

# Overall pool (all single/multi mixes combined per role).
THREE_BODY_POOLED_DARK: list[tuple[str, str, str, str]] = [
    ("A_index", "#00F0FF", "o", "A (oldest)"),
    ("B_index", "#39FF14", "s", "B (middle)"),
    ("C_index", "#FF6600", "^", "C (newest)"),
    ("region_index", "#F5F5F5", "P", "A+B+C market"),
]
FOUR_BODY_POOLED_DARK: list[tuple[str, str, str, str]] = [
    ("A_index", "#00F0FF", "o", "A (oldest)"),
    ("B_index", "#39FF14", "s", "B"),
    ("C_index", "#FFD700", "^", "C"),
    ("D_index", "#FF00FF", "D", "D (newest)"),
    ("region_index", "#F5F5F5", "P", "A+B+C+D market"),
]
BODY_TREND_REGION_STYLE: dict[str, object] = {
    "color": "#F5F5F5",
    "ls": (0, (5, 3)),
    "marker": "P",
}
BODY_TREND_DARK_BG = "#0d1117"
BODY_TREND_DARK_AXES = "#161b22"


def _style_for_role_type(role: str, site_type: str) -> dict[str, object]:
    key = (role, site_type)
    try:
        idx = BODY_TYPE_SERIES_ORDER.index(key)
    except ValueError:
        idx = 0
    return {
        "color": BODY_TYPE_DISTINCT_COLORS[idx % len(BODY_TYPE_DISTINCT_COLORS)],
        "ls": BODY_TYPE_LINE_STYLES.get(site_type, "-"),
        "marker": BODY_ROLE_MARKERS.get(role, "o"),
    }


def _apply_body_trend_dark_theme(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(BODY_TREND_DARK_BG)
    ax.set_facecolor(BODY_TREND_DARK_AXES)
    for spine in ax.spines.values():
        spine.set_color("#484f58")
    ax.tick_params(colors="#e6edf3", labelsize=9)
    ax.xaxis.label.set_color("#e6edf3")
    ax.yaxis.label.set_color("#e6edf3")
    ax.title.set_color("#e6edf3")
    ax.grid(True, color="#30363d", alpha=0.85, lw=0.8)


def build_triple_event_traces(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    window: int = 6,
    *,
    align: str = "newest",
) -> pd.DataFrame:
    """Event-time traces; align='newest' → month 0 = C, align='middle' → month 0 = B."""
    rows: list[dict[str, float]] = []
    align = align.lower()
    for _, triple in triple_deltas.iterrows():
        if align == "middle":
            event_month = month_floor(triple["B_launch_month"])
            pre_a, _ = window_mean(panel, triple["A_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            base = pre_a[PAIR_EVENT_METRIC]
            anchor_role = "B"
        else:
            event_month = month_floor(triple["C_launch_month"])
            pre_a, _ = window_mean(panel, triple["A_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            pre_b, _ = window_mean(panel, triple["B_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            base = safe_add(pre_a[PAIR_EVENT_METRIC], pre_b[PAIR_EVENT_METRIC])
            anchor_role = "C"
        if pd.isna(base) or base <= 0:
            continue

        a_type = _site_type_label(panel, triple["A_site"])
        b_type = _site_type_label(panel, triple["B_site"])
        c_type = _site_type_label(panel, triple["C_site"])
        traces = {
            "A": _site_event_wash_series(panel, triple["A_site"], event_month),
            "B": _site_event_wash_series(
                panel, triple["B_site"], event_month, newest_site=(anchor_role == "B")
            ),
            "C": _site_event_wash_series(
                panel, triple["C_site"], event_month, newest_site=(anchor_role == "C")
            ),
        }

        triple_id = f"{triple['A_site']}|{triple['B_site']}|{triple['C_site']}"
        for relative_month in range(-window, window + 1):
            a_val = traces["A"].get(relative_month, np.nan)
            b_val = traces["B"].get(relative_month, np.nan)
            c_val = traces["C"].get(relative_month, np.nan)
            region_val = safe_add(a_val, b_val, c_val)
            rows.append(
                {
                    "triple_id": triple_id,
                    "align": align,
                    "relative_month": relative_month,
                    "A_type": a_type,
                    "B_type": b_type,
                    "C_type": c_type,
                    "A_index": a_val / base * 100 if pd.notna(a_val) else np.nan,
                    "B_index": b_val / base * 100 if pd.notna(b_val) else np.nan,
                    "C_index": c_val / base * 100 if pd.notna(c_val) else np.nan,
                    "region_index": region_val / base * 100 if pd.notna(region_val) else np.nan,
                    "region_wash": region_val,
                }
            )

    return pd.DataFrame(rows)


def build_quad_event_traces(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    window: int = 6,
    *,
    align: str = "newest",
) -> pd.DataFrame:
    """Event-time traces; align='newest' → month 0 = D, align='middle' → month 0 = C."""
    rows: list[dict[str, float]] = []
    align = align.lower()
    for _, quad in quad_deltas.iterrows():
        if align == "middle":
            event_month = month_floor(quad["C_launch_month"])
            pre_a, _ = window_mean(panel, quad["A_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            pre_b, _ = window_mean(panel, quad["B_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            base = safe_add(pre_a[PAIR_EVENT_METRIC], pre_b[PAIR_EVENT_METRIC])
            anchor_role = "C"
        else:
            event_month = month_floor(quad["D_launch_month"])
            pre_a, _ = window_mean(panel, quad["A_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            pre_b, _ = window_mean(panel, quad["B_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            pre_c, _ = window_mean(panel, quad["C_site"], event_month, "pre", window, [PAIR_EVENT_METRIC])
            base = safe_add(pre_a[PAIR_EVENT_METRIC], pre_b[PAIR_EVENT_METRIC], pre_c[PAIR_EVENT_METRIC])
            anchor_role = "D"
        if pd.isna(base) or base <= 0:
            continue

        a_type = _site_type_label(panel, quad["A_site"])
        b_type = _site_type_label(panel, quad["B_site"])
        c_type = _site_type_label(panel, quad["C_site"])
        d_type = _site_type_label(panel, quad["D_site"])
        traces = {
            "A": _site_event_wash_series(panel, quad["A_site"], event_month),
            "B": _site_event_wash_series(panel, quad["B_site"], event_month),
            "C": _site_event_wash_series(
                panel, quad["C_site"], event_month, newest_site=(anchor_role == "C")
            ),
            "D": _site_event_wash_series(
                panel, quad["D_site"], event_month, newest_site=(anchor_role == "D")
            ),
        }

        quad_id = f"{quad['A_site']}|{quad['B_site']}|{quad['C_site']}|{quad['D_site']}"
        for relative_month in range(-window, window + 1):
            a_val = traces["A"].get(relative_month, np.nan)
            b_val = traces["B"].get(relative_month, np.nan)
            c_val = traces["C"].get(relative_month, np.nan)
            d_val = traces["D"].get(relative_month, np.nan)
            region_val = safe_add(a_val, b_val, c_val, d_val)
            rows.append(
                {
                    "quad_id": quad_id,
                    "align": align,
                    "relative_month": relative_month,
                    "A_type": a_type,
                    "B_type": b_type,
                    "C_type": c_type,
                    "D_type": d_type,
                    "A_index": a_val / base * 100 if pd.notna(a_val) else np.nan,
                    "B_index": b_val / base * 100 if pd.notna(b_val) else np.nan,
                    "C_index": c_val / base * 100 if pd.notna(c_val) else np.nan,
                    "D_index": d_val / base * 100 if pd.notna(d_val) else np.nan,
                    "region_index": region_val / base * 100 if pd.notna(region_val) else np.nan,
                }
            )
    return pd.DataFrame(rows)


THREE_BODY_TREND_SERIES: list[tuple[str, str, str]] = [
    ("A_index", "#1f77b4", "A (oldest)"),
    ("B_index", "#2ca02c", "B (middle)"),
    ("C_index", "#d62728", "C (newest)"),
    ("region_index", "#ff7f0e", "A+B+C market"),
]

FOUR_BODY_TREND_SERIES: list[tuple[str, str, str]] = [
    ("A_index", "#1f77b4", "A (oldest)"),
    ("B_index", "#2ca02c", "B"),
    ("C_index", "#ff7f0e", "C"),
    ("D_index", "#d62728", "D (newest)"),
    ("region_index", "#9467bd", "A+B+C+D market"),
]


def _add_pair_cumulative(
    traces: pd.DataFrame,
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    if traces.empty:
        return traces
    pre_means: dict[str, float] = {}
    for _, pair in pair_deltas.iterrows():
        pre_base, _ = window_mean(panel, pair["existing_site"], pair["event_month"], "pre", window, [PAIR_EVENT_METRIC])
        pre_means[f"{pair['existing_site']}->{pair['new_site']}"] = float(pre_base[PAIR_EVENT_METRIC])

    cum_rows: list[dict[str, float]] = []
    for pair_id, group in traces.groupby("pair_id"):
        base = pre_means.get(pair_id, np.nan)
        if pd.isna(base) or base <= 0:
            continue
        group = group.sort_values("relative_month")
        running = 0.0
        for _, row in group.iterrows():
            existing = row["existing_index"] / 100 * base if pd.notna(row["existing_index"]) else 0.0
            new = row["new_index"] / 100 * base if pd.notna(row["new_index"]) else 0.0
            combined = safe_add(existing, new)
            if row["relative_month"] >= 0:
                running += float(combined) if pd.notna(combined) else 0.0
            cum_rows.append(
                {
                    "pair_id": pair_id,
                    "relative_month": int(row["relative_month"]),
                    "combined_cum_index": running / base * 100,
                }
            )
    return traces.merge(pd.DataFrame(cum_rows), on=["pair_id", "relative_month"], how="left")


def _draw_event_medians(
    ax: plt.Axes,
    traces: pd.DataFrame,
    *,
    window: int,
    title: str,
    series_specs: list[tuple[str, str, str]],
    event_id_col: str,
    ylabel: str = "Index (pre-launch baseline = 100)",
    show_faint: bool = True,
    ymax_cap: float = 320.0,
) -> None:
    """Median + IQR event-time lines for arbitrary site roles."""
    if traces.empty:
        ax.set_title(title)
        return

    cols = [c for c, _, _ in series_specs]
    for col, color, label in series_specs:
        if show_faint:
            for _, group in traces.groupby(event_id_col):
                sub = group.sort_values("relative_month")
                ax.plot(sub["relative_month"], sub[col], color=color, alpha=0.1, lw=0.8, zorder=1)

        summary = (
            traces.groupby("relative_month")[col]
            .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("relative_month")
        )
        ax.fill_between(summary["relative_month"], summary["q25"], summary["q75"], color=color, alpha=0.18, zorder=2)
        ax.plot(
            summary["relative_month"],
            summary["median"],
            color=color,
            lw=2.4,
            marker="o",
            ms=4,
            label=f"{label} (median)",
            zorder=3,
        )

    ymax = float(traces[cols].quantile(0.98).max())
    ax.set_ylim(0, max(150.0, min(ymax * 1.1, ymax_cap)))
    ax.axvline(0, color="black", ls="--", lw=1.1)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel("Months relative to newest-site launch (0)")
    ax.set_ylabel(ylabel)
    n_events = int(traces[event_id_col].nunique())
    ax.set_title(f"{title}\n(n = {n_events})", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)


def _save_event_trend_figure(
    traces: pd.DataFrame,
    out_path: Path,
    *,
    series_specs: list[tuple[str, str, str]],
    event_id_col: str,
    title: str,
    suptitle: str,
    window: int = 6,
    ylabel: str = "Index (pre-launch baseline = 100)",
    ymax_cap: float = 320.0,
) -> None:
    if traces.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    _draw_event_medians(
        ax,
        traces,
        window=window,
        title=title,
        series_specs=series_specs,
        event_id_col=event_id_col,
        ylabel=ylabel,
        ymax_cap=ymax_cap,
    )
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def _draw_multi_body_trend_by_site_type(
    ax: plt.Axes,
    traces: pd.DataFrame,
    *,
    roles: list[str],
    event_id_col: str,
    window: int,
    title: str,
    ylabel: str,
    ymax_cap: float,
    show_faint: bool = False,
    region_col: str = "region_index",
    region_label: str = "Market (sum, post-launch only)",
    dark_theme: bool = True,
) -> None:
    """Median event-time lines split by single vs multi for each role."""
    if traces.empty:
        ax.set_title(title)
        return

    index_cols = [f"{role}_index" for role in roles]
    min_cohort = 2
    ref_color = "#8b949e" if dark_theme else "gray"
    launch_color = "#f0f6fc" if dark_theme else "black"

    for role in roles:
        col = f"{role}_index"
        type_col = f"{role}_type"
        for site_type in ("single", "multi"):
            style = _style_for_role_type(role, site_type)
            color = str(style["color"])
            ls = style["ls"]
            marker = str(style["marker"])
            subset = traces[traces[type_col] == site_type]
            n_events = int(subset[event_id_col].nunique()) if not subset.empty else 0
            if n_events < min_cohort:
                continue

            if show_faint:
                faint_alpha = 0.06 if dark_theme else 0.08
                for _, group in subset.groupby(event_id_col):
                    sub = group.sort_values("relative_month")
                    ax.plot(
                        sub["relative_month"],
                        sub[col],
                        color=color,
                        alpha=faint_alpha,
                        lw=0.6,
                        ls=ls,
                        zorder=1,
                    )

            summary = (
                subset.groupby("relative_month")[col]
                .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
                .reset_index()
                .sort_values("relative_month")
            )
            band_alpha = 0.22 if dark_theme else 0.15
            ax.fill_between(
                summary["relative_month"],
                summary["q25"],
                summary["q75"],
                color=color,
                alpha=band_alpha,
                zorder=2,
            )
            ax.plot(
                summary["relative_month"],
                summary["median"],
                color=color,
                lw=3.0 if dark_theme else 2.2,
                ls=ls,
                marker=marker,
                ms=5 if dark_theme else 3.5,
                mew=0.8,
                mec=BODY_TREND_DARK_BG if dark_theme else "white",
                label=f"{role} · {site_type} (n={n_events})",
                zorder=4,
            )

    if region_col in traces.columns:
        region_style = BODY_TREND_REGION_STYLE
        region_color = str(region_style["color"])
        summary = (
            traces.groupby("relative_month")[region_col]
            .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("relative_month")
        )
        ax.fill_between(
            summary["relative_month"],
            summary["q25"],
            summary["q75"],
            color=region_color,
            alpha=0.14,
            zorder=2,
        )
        ax.plot(
            summary["relative_month"],
            summary["median"],
            color=region_color,
            lw=2.4,
            ls=region_style["ls"],
            marker=str(region_style["marker"]),
            ms=4,
            mew=0.6,
            mec=BODY_TREND_DARK_BG if dark_theme else "white",
            label=region_label,
            zorder=3,
        )

    ymax = float(traces[index_cols + [region_col]].quantile(0.98).max())
    ax.set_ylim(0, max(150.0, min(ymax * 1.1, ymax_cap)))
    ax.axvline(0, color=launch_color, ls="--", lw=1.2, alpha=0.9)
    ax.axhline(100, color=ref_color, ls=":", lw=1.1)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel("Months relative to newest-site launch (0)")
    ax.set_ylabel(ylabel)
    n_events = int(traces[event_id_col].nunique())
    ax.set_title(f"{title}\n(n = {n_events} events)", fontsize=10)
    legend = ax.legend(
        loc="upper left",
        fontsize=7.5,
        frameon=True,
        ncol=2,
        facecolor=BODY_TREND_DARK_AXES if dark_theme else "white",
        edgecolor="#484f58" if dark_theme else "0.8",
        labelcolor="#e6edf3" if dark_theme else "black",
    )
    if dark_theme:
        legend.get_frame().set_alpha(0.95)
    if not dark_theme:
        ax.grid(True, alpha=0.3)


def _save_multi_body_trend_by_site_type(
    traces: pd.DataFrame,
    out_path: Path,
    *,
    roles: list[str],
    event_id_col: str,
    title: str,
    suptitle: str,
    window: int = 6,
    ylabel: str = "Index (pre-launch baseline = 100)",
    ymax_cap: float = 320.0,
    region_label: str = "Market (sum, post-launch only)",
) -> None:
    if traces.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 7))
    _apply_body_trend_dark_theme(fig, ax)
    _draw_multi_body_trend_by_site_type(
        ax,
        traces,
        roles=roles,
        event_id_col=event_id_col,
        window=window,
        title=title,
        ylabel=ylabel,
        ymax_cap=ymax_cap,
        region_label=region_label,
        dark_theme=True,
    )
    fig.suptitle(suptitle, fontsize=10.5, y=1.02, color="#e6edf3")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_multi_body_trend_pooled(
    ax: plt.Axes,
    traces: pd.DataFrame,
    *,
    series_specs: list[tuple[str, str, str, str]],
    event_id_col: str,
    window: int,
    title: str,
    ylabel: str,
    ymax_cap: float,
    xlabel: str = "Months relative to anchor launch (0)",
) -> None:
    """One median line per role (all site types pooled) on dark theme."""
    if traces.empty:
        ax.set_title(title)
        return

    cols = [col for col, _, _, _ in series_specs]
    for col, color, marker, label in series_specs:
        if col not in traces.columns:
            continue
        summary = (
            traces.groupby("relative_month")[col]
            .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("relative_month")
        )
        ax.fill_between(
            summary["relative_month"],
            summary["q25"],
            summary["q75"],
            color=color,
            alpha=0.2,
            zorder=2,
        )
        ax.plot(
            summary["relative_month"],
            summary["median"],
            color=color,
            lw=3.2,
            marker=marker,
            ms=5.5,
            mew=0.8,
            mec=BODY_TREND_DARK_BG,
            label=label,
            zorder=4,
        )

    ymax = float(traces[cols].quantile(0.98).max())
    ax.set_ylim(0, max(150.0, min(ymax * 1.1, ymax_cap)))
    ax.axvline(0, color="#f0f6fc", ls="--", lw=1.2, alpha=0.9)
    ax.axhline(100, color="#8b949e", ls=":", lw=1.1)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    n_events = int(traces[event_id_col].nunique())
    ax.set_title(f"{title}\n(n = {n_events} events)", fontsize=10)
    legend = ax.legend(
        loc="upper left",
        fontsize=7.5,
        frameon=True,
        facecolor=BODY_TREND_DARK_AXES,
        edgecolor="#484f58",
        labelcolor="#e6edf3",
    )
    legend.get_frame().set_alpha(0.95)


def _save_multi_body_trend_pooled(
    traces: pd.DataFrame,
    out_path: Path,
    *,
    series_specs: list[tuple[str, str, str, str]],
    event_id_col: str,
    title: str,
    suptitle: str,
    window: int = 6,
    ylabel: str = "Index (pre-launch baseline = 100)",
    ymax_cap: float = 320.0,
    xlabel: str = "Months relative to anchor launch (0)",
) -> None:
    if traces.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    _apply_body_trend_dark_theme(fig, ax)
    _draw_multi_body_trend_pooled(
        ax,
        traces,
        series_specs=series_specs,
        event_id_col=event_id_col,
        window=window,
        title=title,
        ylabel=ylabel,
        ymax_cap=ymax_cap,
        xlabel=xlabel,
    )
    fig.suptitle(suptitle, fontsize=10.5, y=1.02, color="#e6edf3")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_multi_body_dual_alignment_overall(
    triple_or_quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    *,
    body: str,
    series_specs: list[tuple[str, str, str, str]],
    event_id_col: str,
    build_traces: object,
    window: int = 6,
    ylabel: str,
    ymax_cap: float,
    suptitle: str,
    middle_panel: tuple[str, str, str],
    newest_panel: tuple[str, str, str],
) -> None:
    """Side-by-side overall pool: middle-site anchor (left) vs newest-site anchor (right)."""
    middle_title, middle_xlabel, middle_ylabel_note = middle_panel
    newest_title, newest_xlabel, newest_ylabel_note = newest_panel

    traces_middle = build_traces(triple_or_quad_deltas, panel, window=window, align="middle")
    traces_newest = build_traces(triple_or_quad_deltas, panel, window=window, align="newest")
    if traces_middle.empty and traces_newest.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=True)
    _apply_body_trend_dark_theme(fig, axes[0])
    _apply_body_trend_dark_theme(fig, axes[1])

    if not traces_middle.empty:
        _draw_multi_body_trend_pooled(
            axes[0],
            traces_middle,
            series_specs=series_specs,
            event_id_col=event_id_col,
            window=window,
            title=middle_title,
            ylabel=middle_ylabel_note or ylabel,
            ymax_cap=ymax_cap,
            xlabel=middle_xlabel,
        )
    else:
        axes[0].text(0.5, 0.5, "No events", ha="center", va="center", color="#e6edf3", transform=axes[0].transAxes)

    if not traces_newest.empty:
        _draw_multi_body_trend_pooled(
            axes[1],
            traces_newest,
            series_specs=series_specs,
            event_id_col=event_id_col,
            window=window,
            title=newest_title,
            ylabel=ylabel,
            ymax_cap=ymax_cap,
            xlabel=newest_xlabel,
        )
    else:
        axes[1].text(0.5, 0.5, "No events", ha="center", va="center", color="#e6edf3", transform=axes[1].transAxes)

    fig.suptitle(suptitle, fontsize=11, y=1.02, color="#e6edf3")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_pair_event_medians(
    ax: plt.Axes,
    traces: pd.DataFrame,
    *,
    window: int,
    title: str,
    show_faint: bool = True,
    show_combined: bool = True,
    event_id_col: str = "pair_id",
) -> None:
    work = traces.copy()
    series_specs: list[tuple[str, str, str]] = [
        ("existing_index", "#1f77b4", "Existing"),
        ("new_index", "#d62728", "New"),
    ]
    if show_combined:
        work["combined_index"] = work["existing_index"] + work["new_index"].fillna(0)
        series_specs.append(("combined_index", "#ff7f0e", "Combined (monthly)"))
    _draw_event_medians(
        ax,
        work,
        window=window,
        title=title,
        series_specs=series_specs,
        event_id_col=event_id_col,
        ylabel="Index (incumbent pre-launch = 100)",
        show_faint=show_faint,
    )


def plot_aggregate_event_trend(
    traces: pd.DataFrame,
    out_path: Path,
    *,
    series_specs: list[tuple[str, str, str]],
    event_id_col: str,
    title: str,
    ylabel: str = "Index (pre-launch baseline = 100)",
    window: int = 6,
    show_cumulative: str | None = None,
) -> None:
    if traces.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for col, color, label in series_specs:
        for _, group in traces.groupby(event_id_col):
            sub = group.sort_values("relative_month")
            ax.plot(sub["relative_month"], sub[col], color=color, alpha=0.1, lw=0.8, zorder=1)

        summary = (
            traces.groupby("relative_month")[col]
            .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            .reset_index()
            .sort_values("relative_month")
        )
        ax.fill_between(summary["relative_month"], summary["q25"], summary["q75"], color=color, alpha=0.18, zorder=2)
        ax.plot(
            summary["relative_month"],
            summary["median"],
            color=color,
            lw=2.8,
            marker="o",
            ms=5,
            label=f"{label} (median)",
            zorder=3,
        )

    ymax = float(traces[[c for c, _, _ in series_specs]].quantile(0.98).max())
    ax.set_ylim(0, max(150.0, min(ymax * 1.1, 280.0)))
    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xticks(range(-window, window + 1))
    ax.set_xlabel("Months relative to newest-site launch (month 0)")
    ax.set_ylabel(ylabel)
    n_events = int(traces[event_id_col].nunique())
    ax.set_title(f"{title}\n(n = {n_events} events)", fontsize=12)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_any_new_operator_effect(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> None:
    traces = build_pair_event_traces(pair_deltas, panel, window=window)
    traces["combined_index"] = traces["existing_index"] + traces["new_index"].fillna(0)
    plot_aggregate_event_trend(
        traces,
        out_path,
        series_specs=[
            ("existing_index", "#1f77b4", "Incumbent site"),
            ("new_index", "#d62728", "New site"),
            ("combined_index", "#ff7f0e", "Combined market (monthly)"),
        ],
        event_id_col="pair_id",
        title="Effect of any new nearby operator on running sites (all two-body pairs)",
        window=window,
    )


def plot_existing_single_new_multi_trend(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    filtered = filter_pairs_by_site_types(pair_deltas, panel, existing_type="single", new_type="multi")
    traces = build_pair_event_traces(filtered, panel, window=window)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _draw_pair_event_medians(
        ax,
        traces,
        window=window,
        title="Existing single → new multi",
        show_faint=True,
        show_combined=True,
    )
    fig.suptitle(
        "Monthly index around launch (100 = incumbent's 6-month pre-launch mean)\n"
        "Faint = each pair | band = 25th–75th | bold = median",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return filtered


def plot_two_body_trends_by_type_combo(
    pair_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """2×2 grid: median event-time trend for each existing×new single/multi combo."""
    combos = [
        ("single", "single", "Existing single → new single"),
        ("single", "multi", "Existing single → new multi"),
        ("multi", "single", "Existing multi → new single"),
        ("multi", "multi", "Existing multi → new multi"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    counts: list[dict[str, object]] = []

    for ax, (ex_type, nw_type, title) in zip(axes.flat, combos):
        filtered = filter_pairs_by_site_types(pair_deltas, panel, existing_type=ex_type, new_type=nw_type)
        counts.append({"existing": ex_type, "new": nw_type, "n": len(filtered)})
        if filtered.empty:
            ax.text(0.5, 0.5, "No pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            continue
        traces = build_pair_event_traces(filtered, panel, window=window)
        _draw_pair_event_medians(ax, traces, window=window, title=title, show_faint=True, show_combined=True)

    fig.suptitle(
        "Two-body trends by site type (single / multi)\n"
        "Blue = incumbent | Red = new (from month 0) | Orange = combined monthly | 100 = incumbent pre-launch",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pd.DataFrame(counts)


def plot_three_body_all_triples_trend(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """Median trend by role × site type (each series has a unique color)."""
    traces = build_triple_event_traces(triple_deltas, panel, window=window)
    _save_multi_body_trend_by_site_type(
        traces,
        out_path,
        roles=["A", "B", "C"],
        event_id_col="triple_id",
        title="All three-body triples (by site type)",
        suptitle=(
            "Three-body: each line = role + single/multi — unique color per series\n"
            "solid = single | dashed = multi | 100 = (A+B) pre-launch | month 0 = C launch"
        ),
        window=window,
        ylabel="Index (A+B pre-launch = 100)",
        region_label="A+B+C market",
    )
    return triple_deltas


def plot_three_body_all_triples_trend_overall(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """Overall pool: dual panel — month 0 = B (middle) | month 0 = C (newest)."""
    _save_multi_body_dual_alignment_overall(
        triple_deltas,
        panel,
        out_path,
        body="triple",
        series_specs=THREE_BODY_POOLED_DARK,
        event_id_col="triple_id",
        build_traces=build_triple_event_traces,
        window=window,
        ylabel="Index (pre-launch baseline = 100)",
        ymax_cap=320.0,
        suptitle=(
            "Three-body overall pool — left: months relative to B (middle) | right: months relative to C (newest)\n"
            "Left baseline = A pre-launch | right baseline = (A+B) pre-launch | lines start at each site's launch"
        ),
        middle_panel=(
            "Month 0 = B (middle site)",
            "Months relative to B launch (0)",
            "Index (A pre-launch = 100)",
        ),
        newest_panel=(
            "Month 0 = C (newest site)",
            "Months relative to C launch (0)",
            "",
        ),
    )
    return triple_deltas


def plot_new_multi_three_body_trend(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    filtered = filter_triples_newest_type(triple_deltas, panel, "multi")
    traces = build_triple_event_traces(filtered, panel, window=window)
    _save_multi_body_trend_by_site_type(
        traces,
        out_path,
        roles=["A", "B", "C"],
        event_id_col="triple_id",
        title="Newest site (C) is multi — A/B any type",
        suptitle=(
            "Three-body subset: C = multi only\n"
            "Colors by single/multi per role | lines start at each site's launch | month 0 = C"
        ),
        window=window,
        ylabel="Index (A+B pre-launch = 100)",
        region_label="A+B+C market (post-launch washes only)",
    )
    return filtered


def plot_four_body_all_quads_trend(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """Median trend by role × site type (each series has a unique color)."""
    traces = build_quad_event_traces(quad_deltas, panel, window=window)
    _save_multi_body_trend_by_site_type(
        traces,
        out_path,
        roles=["A", "B", "C", "D"],
        event_id_col="quad_id",
        title="All four-body quads (by site type)",
        suptitle=(
            "Four-body: each line = role + single/multi — unique color per series\n"
            "solid = single | dashed = multi | 100 = (A+B+C) pre-launch | month 0 = D launch"
        ),
        window=window,
        ylabel="Index (A+B+C pre-launch = 100)",
        ymax_cap=400.0,
        region_label="A+B+C+D market",
    )
    return quad_deltas


def plot_four_body_all_quads_trend_overall(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    window: int = 6,
) -> pd.DataFrame:
    """Overall pool: dual panel — month 0 = C (middle) | month 0 = D (newest)."""
    _save_multi_body_dual_alignment_overall(
        quad_deltas,
        panel,
        out_path,
        body="quad",
        series_specs=FOUR_BODY_POOLED_DARK,
        event_id_col="quad_id",
        build_traces=build_quad_event_traces,
        window=window,
        ylabel="Index (pre-launch baseline = 100)",
        ymax_cap=400.0,
        suptitle=(
            "Four-body overall pool — left: months relative to C (middle) | right: months relative to D (newest)\n"
            "Left baseline = (A+B) pre-launch | right baseline = (A+B+C) pre-launch | lines start at each site's launch"
        ),
        middle_panel=(
            "Month 0 = C (middle site)",
            "Months relative to C launch (0)",
            "Index (A+B pre-launch = 100)",
        ),
        newest_panel=(
            "Month 0 = D (newest site)",
            "Months relative to D launch (0)",
            "",
        ),
    )
    return quad_deltas


def plot_market_saturation(pair_deltas: pd.DataFrame, out_path: Path) -> None:
    df = pair_deltas.copy()
    df["regime"] = df.apply(pair_regime, axis=1)
    df["loss_making"] = (df["existing_pct_wash_count_total"] <= -5) & (df["combined_pct_wash_count_total"] <= 5)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    colors = {
        "mostly cannibalization": "#d62728",
        "cannibalization plus expansion": "#ff7f0e",
        "market expansion": "#2ca02c",
        "flat to mixed": "#7f7f7f",
    }

    ax = axes[0]
    for regime, group in df.groupby("regime"):
        ax.scatter(
            group["distance_miles"],
            group["existing_pct_wash_count_total"],
            c=colors.get(regime, "#333333"),
            alpha=0.65,
            s=28,
            label=regime,
        )
    ax.axhline(-5, color="black", ls=":", lw=1)
    ax.axhline(-10, color="gray", ls=":", lw=1)
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Incumbent % change (6mo post vs pre)")
    ax.set_title("Incumbent impact vs distance")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    ax.scatter(
        df["existing_pct_wash_count_total"],
        df["combined_pct_wash_count_total"],
        c=df["loss_making"].map({True: "#d62728", False: "#1f77b4"}),
        alpha=0.65,
        s=32,
    )
    ax.axvline(-5, color="black", ls=":", lw=1)
    ax.axhline(5, color="black", ls=":", lw=1)
    xlim = (
        float(df["existing_pct_wash_count_total"].quantile(0.02) - 5),
        float(df["existing_pct_wash_count_total"].quantile(0.98) + 5),
    )
    ylim = (
        float(min(-20, df["combined_pct_wash_count_total"].min() - 10)),
        float(df["combined_pct_wash_count_total"].quantile(0.98) + 10),
    )
    ax.fill_between([xlim[0], -5], ylim[0], 5, alpha=0.12, color="#d62728", label="Loss-making zone\n(existing ≤ −5%, combined ≤ +5%)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Incumbent % change")
    ax.set_ylabel("Combined market % change")
    n_loss = int(df["loss_making"].sum())
    ax.set_title(f"Saturation / loss-making map ({n_loss}/{len(df)} pairs)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    band_stats = []
    for band in ["0-3 mi", "3-5 mi", "5-8 mi", "8-10 mi"]:
        sub = df[df["distance_band"] == band]
        if sub.empty:
            continue
        band_stats.append(
            {
                "band": band,
                "median_existing": sub["existing_pct_wash_count_total"].median(),
                "median_combined": sub["combined_pct_wash_count_total"].median(),
                "pct_loss_making": 100 * sub["loss_making"].mean(),
                "n": len(sub),
            }
        )
    stats = pd.DataFrame(band_stats)
    x = np.arange(len(stats))
    w = 0.35
    ax.bar(x - w / 2, stats["median_existing"], width=w, label="Median incumbent %", color="#1f77b4")
    ax.bar(x + w / 2, stats["median_combined"], width=w, label="Median combined %", color="#ff7f0e")
    ax.axhline(5, color="black", ls=":", lw=1)
    ax.axhline(-5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['band']}\n(n={int(r['n'])})" for _, r in stats.iterrows()])
    ax.set_ylabel("% change (6mo post vs pre)")
    ax.set_title("Medians by distance band")
    ax.legend(fontsize=8)
    for i, row in stats.iterrows():
        ax.text(i, ax.get_ylim()[1] * 0.92, f"{row['pct_loss_making']:.0f}% loss-making", ha="center", fontsize=7)

    fig.suptitle(
        "Market saturation: when is a new site loss-making for the incumbent?\n"
        "Red zone ≈ incumbent down >5% and local market barely grows (≤+5%)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_triple_examples_all(
    triple_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    x_start: pd.Timestamp = CALENDAR_X_START,
    x_end: pd.Timestamp = CALENDAR_X_END,
    ncols: int = 6,
) -> None:
    """One subplot per triple: A/B/C monthly washes on a shared calendar axis."""
    triples = triple_deltas.sort_values(["event_month", "A_to_C_miles", "B_to_C_miles"]).reset_index(drop=True)
    n_triples = len(triples)
    if n_triples == 0:
        return

    nrows = int(np.ceil(n_triples / ncols))
    fig_w = max(24.0, ncols * 4.0)
    fig_h = max(24.0, nrows * 3.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()

    site_styles = [
        ("A (oldest)", "A_site", "A_launch_month", "#1f77b4", False, None),
        ("B (middle)", "B_site", "B_launch_month", "#2ca02c", True, "B_launch_month"),
        ("C (newest)", "C_site", "C_launch_month", "#d62728", True, "C_launch_month"),
    ]

    for ax in axes_flat[n_triples:]:
        ax.set_visible(False)

    for ax, (_, triple) in zip(axes_flat, triples.iterrows()):
        b_launch = month_floor(triple["B_launch_month"])
        c_launch = month_floor(triple["C_launch_month"])
        state = triple.get("state") or _site_state_label(panel, triple["C_site"])
        for label, site_col, launch_col, color, from_launch, _ in site_styles:
            min_month = month_floor(triple[launch_col]) if from_launch else None
            series = _site_calendar_series(
                panel,
                triple[site_col],
                x_start,
                x_end,
                from_launch=from_launch,
                min_calendar_month=min_month,
            )
            ax.plot(
                series["calendar_month"],
                series[PAIR_EVENT_METRIC],
                marker="o",
                ms=2.0,
                lw=1.1,
                color=color,
                label=label,
            )
        ax.axvline(b_launch, color="#2ca02c", ls="--", lw=1.0, alpha=0.85)
        ax.axvline(c_launch, color="#d62728", ls="--", lw=1.0, alpha=0.85)
        a_type = _site_type_label(panel, triple["A_site"])
        b_type = _site_type_label(panel, triple["B_site"])
        c_type = _site_type_label(panel, triple["C_site"])
        a_launch = month_floor(triple["A_launch_month"])
        a_cohort = triple.get("A_cohort", "")
        b_cohort = triple.get("B_cohort", "")
        c_cohort = triple.get("C_cohort", "")
        ax.set_title(
            f"{state} | {triple['market_zip']}\n"
            f"A start {a_launch.strftime('%Y-%m')} ({a_cohort}, {a_type}) | "
            f"B start {b_launch.strftime('%Y-%m')} ({b_cohort}, {b_type}) | "
            f"C start {c_launch.strftime('%Y-%m')} ({c_cohort}, {c_type})\n"
            f"A–C {triple['A_to_C_miles']:.1f} mi | B–C {triple['B_to_C_miles']:.1f} mi",
            fontsize=5.5,
        )
        ax.tick_params(axis="y", labelsize=6)
        _apply_calendar_xaxis(ax, x_start, x_end, labelsize=5, month_interval=1)

    for ax in axes_flat[:n_triples]:
        ax.set_xlabel("Month", fontsize=6)

    for ax in axes_flat[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("Car washes", fontsize=7)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"All three-body triples ({n_triples}): monthly washes, Jan 2024 – Dec 2025\n"
        "Blue = A | Green = B | Red = C | dashed = B/C launch | title = start dates (gt2/lt2) + single/multi",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.12, hspace=0.88, wspace=0.38)
    fig.savefig(out_path, dpi=120, pad_inches=0.15)
    plt.close(fig)


def plot_quad_examples_all(
    quad_deltas: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    x_start: pd.Timestamp = CALENDAR_X_START,
    x_end: pd.Timestamp = CALENDAR_X_END,
    ncols: int = 6,
) -> None:
    """One subplot per quad: A/B/C/D monthly washes on a shared calendar axis."""
    quads = quad_deltas.sort_values(["event_month", "A_to_D_miles", "B_to_D_miles", "C_to_D_miles"]).reset_index(drop=True)
    n_quads = len(quads)
    if n_quads == 0:
        return

    nrows = int(np.ceil(n_quads / ncols))
    fig_w = max(24.0, ncols * 4.0)
    fig_h = max(20.0, nrows * 3.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes_flat = np.atleast_1d(axes).ravel()

    site_styles = [
        ("A (oldest)", "A_site", "A_launch_month", "#1f77b4", False, None),
        ("B", "B_site", "B_launch_month", "#2ca02c", True, "B_launch_month"),
        ("C", "C_site", "C_launch_month", "#ff7f0e", True, "C_launch_month"),
        ("D (newest)", "D_site", "D_launch_month", "#d62728", True, "D_launch_month"),
    ]

    for ax in axes_flat[n_quads:]:
        ax.set_visible(False)

    for ax, (_, quad) in zip(axes_flat, quads.iterrows()):
        b_launch = month_floor(quad["B_launch_month"])
        c_launch = month_floor(quad["C_launch_month"])
        d_launch = month_floor(quad["D_launch_month"])
        state = quad.get("state") or _site_state_label(panel, quad["D_site"])
        for label, site_col, launch_col, color, from_launch, _ in site_styles:
            min_month = month_floor(quad[launch_col]) if from_launch else None
            series = _site_calendar_series(
                panel,
                quad[site_col],
                x_start,
                x_end,
                from_launch=from_launch,
                min_calendar_month=min_month,
            )
            ax.plot(
                series["calendar_month"],
                series[PAIR_EVENT_METRIC],
                marker="o",
                ms=2.0,
                lw=1.1,
                color=color,
                label=label,
            )
        ax.axvline(b_launch, color="#2ca02c", ls="--", lw=1.0, alpha=0.85)
        ax.axvline(c_launch, color="#ff7f0e", ls="--", lw=1.0, alpha=0.85)
        ax.axvline(d_launch, color="#d62728", ls="--", lw=1.0, alpha=0.85)
        a_type = _site_type_label(panel, quad["A_site"])
        b_type = _site_type_label(panel, quad["B_site"])
        c_type = _site_type_label(panel, quad["C_site"])
        d_type = _site_type_label(panel, quad["D_site"])
        a_launch = month_floor(quad["A_launch_month"])
        ax.set_title(
            f"{state} | {quad['market_zip']}\n"
            f"A {a_launch.strftime('%Y-%m')} ({quad.get('A_cohort', '')}, {a_type}) | "
            f"B {b_launch.strftime('%Y-%m')} ({quad.get('B_cohort', '')}, {b_type}) | "
            f"C {c_launch.strftime('%Y-%m')} ({quad.get('C_cohort', '')}, {c_type}) | "
            f"D {d_launch.strftime('%Y-%m')} ({quad.get('D_cohort', '')}, {d_type})\n"
            f"A–D {quad['A_to_D_miles']:.1f} mi | B–D {quad['B_to_D_miles']:.1f} mi | C–D {quad['C_to_D_miles']:.1f} mi",
            fontsize=5.0,
        )
        ax.tick_params(axis="y", labelsize=6)
        _apply_calendar_xaxis(ax, x_start, x_end, labelsize=5, month_interval=1)

    for ax in axes_flat[:n_quads]:
        ax.set_xlabel("Month", fontsize=6)

    for ax in axes_flat[::ncols]:
        if ax.get_visible():
            ax.set_ylabel("Car washes", fontsize=7)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        f"All four-body quads ({n_quads}): monthly washes, Jan 2024 – Dec 2025\n"
        "Blue = A | Green = B | Orange = C | Red = D | dashed = B/C/D launch | title = start dates + single/multi",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.12, hspace=0.95, wspace=0.38)
    fig.savefig(out_path, dpi=120, pad_inches=0.15)
    plt.close(fig)


def plot_pair_examples(examples: pd.DataFrame, panel: pd.DataFrame, out_path: Path, window: int = 6) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (_, pair) in zip(axes.flat, examples.iterrows()):
        event_month = pair["event_month"]
        ex = panel[panel["client_id_location_id"] == pair["existing_site"]].copy()
        nw = panel[panel["client_id_location_id"] == pair["new_site"]].copy()
        ex["relative_month"] = month_diff(ex["year_month"], event_month).astype(int)
        nw["relative_month"] = month_diff(nw["year_month"], event_month).astype(int)
        ex = ex[ex["relative_month"].between(-window, window)]
        nw = nw[nw["relative_month"].between(-window, window) & nw["site_month_number"].ge(1)]
        combined = ex[["relative_month", PAIR_EVENT_METRIC]].merge(
            nw[["relative_month", PAIR_EVENT_METRIC]],
            on="relative_month",
            how="outer",
            suffixes=("_existing", "_new"),
        ).sort_values("relative_month")
        combined["combined_total"] = combined.apply(
            lambda row: safe_add(row.get(f"{PAIR_EVENT_METRIC}_existing", np.nan), row.get(f"{PAIR_EVENT_METRIC}_new", np.nan)),
            axis=1,
        )
        ax.plot(ex["relative_month"], ex[PAIR_EVENT_METRIC], marker="o", lw=1.8, color="#1f77b4", label="Existing")
        ax.plot(nw["relative_month"], nw[PAIR_EVENT_METRIC], marker="o", lw=1.8, color="#d62728", label="New")
        ax.plot(combined["relative_month"], combined["combined_total"], marker="o", lw=1.8, color="#ff7f0e", label="Combined")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_title(f"{pair['market_zip']} | {pair['distance_miles']:.1f} mi")
        ax.text(
            0.03,
            0.97,
            f"existing launch {pair['existing_launch_month'].date()}\nnew launch {pair['new_launch_month'].date()}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "lightgray"},
        )
    for ax in axes[-1]:
        ax.set_xlabel("Months relative to new-site launch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Monthly total washes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Representative two-body examples", fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_triple_summary(triple_deltas: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, METRICS):
        series_map = {
            "A (oldest)": triple_deltas[f"A_pct_{metric}"],
            "B (middle)": triple_deltas[f"B_pct_{metric}"],
            "A+B+C market": triple_deltas[f"region_pct_{metric}"],
        }
        for y_pos, (label, series), color in zip(
            [2, 1, 0],
            series_map.items(),
            ["#1f77b4", "#2ca02c", "#ff7f0e"],
        ):
            clean = series.replace([np.inf, -np.inf], np.nan).dropna()
            q1, med, q3 = clean.quantile([0.25, 0.5, 0.75])
            ax.hlines(y_pos, q1, q3, color=color, lw=7, alpha=0.35)
            ax.scatter(med, y_pos, s=80, color=color, zorder=3)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_yticks([0, 1, 2], ["A+B+C market", "B (middle)", "A (oldest)"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("% change: 6 months before vs 6 months after C")
    fig.suptitle("Three-body impact summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_triple_event_profile(profile: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(profile["relative_month"], profile["A_index"], marker="o", lw=2, color="#1f77b4", label="A (oldest)")
    ax.plot(profile["relative_month"], profile["B_index"], marker="o", lw=2, color="#2ca02c", label="B (middle)")
    ax.plot(profile["relative_month"], profile["C_index"], marker="o", lw=2, color="#d62728", label="C (newest)")
    ax.plot(profile["relative_month"], profile["region_index"], marker="o", lw=2, color="#ff7f0e", label="A+B+C market")
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_xlabel("Months relative to C's launch month")
    ax.set_ylabel("Index (A+B pre-launch mean = 100)")
    ax.set_title("Three-body event-time profile for total washes")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_triple_networks(examples: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (_, triple) in zip(axes.flat, examples.iterrows()):
        points = {
            "A": (triple["A_lon"], triple["A_lat"], "#1f77b4", triple["A_launch_month"]),
            "B": (triple["B_lon"], triple["B_lat"], "#2ca02c", triple["B_launch_month"]),
            "C": (triple["C_lon"], triple["C_lat"], "#d62728", triple["C_launch_month"]),
        }
        for left, right, dist in [
            ("A", "B", triple["AB_miles"]),
            ("A", "C", triple["A_to_C_miles"]),
            ("B", "C", triple["B_to_C_miles"]),
        ]:
            ax.plot(
                [points[left][0], points[right][0]],
                [points[left][1], points[right][1]],
                color="gray",
                lw=1.2,
                alpha=0.7,
            )
            ax.text(
                (points[left][0] + points[right][0]) / 2,
                (points[left][1] + points[right][1]) / 2,
                f"{dist:.1f} mi",
                fontsize=7,
                color="gray",
            )
        for label, (lon, lat, color, launch_month) in points.items():
            ax.scatter(lon, lat, s=180, c=color, edgecolors="black", linewidths=0.9, zorder=3)
            ax.annotate(f"{label}\n{launch_month.date()}", (lon, lat), textcoords="offset points", xytext=(8, 6), fontsize=8)
        ax.set_title(f"{triple['market_zip']} | max dist {max(triple['A_to_C_miles'], triple['B_to_C_miles']):.1f} mi")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Representative three-body layouts", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    panel: pd.DataFrame,
    validation: dict[str, object],
    pair_deltas: pd.DataFrame,
    triple_deltas: pd.DataFrame,
    pair_profile: pd.DataFrame,
    triple_profile: pd.DataFrame,
    max_neighbor_miles: float,
    pre_post_window: int,
) -> Path:
    report = f"""# Site Interaction Report

## Method
- Data used (unified monthly panel):
  - `less_than-2yrs.csv`: {validation['lt2_sites']} sites, {validation['lt2_rows']:,} site-month rows ({validation['lt2_month_min'].date()} to {validation['lt2_month_max'].date()}).
  - `more_than-2yrs_monthly.csv`: {validation['gt2_sites']} sites, {validation['gt2_rows']:,} site-month rows ({validation['gt2_month_min'].date()} to {validation['gt2_month_max'].date()}).
- Combined panel: {validation['sites']} unique sites, {validation['rows']:,} site-month rows, calendar span {validation['month_min'].date()} to {validation['month_max'].date()}.
- Typical roles: newer launches from lt2; older neighbors from gt2 monthly (true calendar months in `year_month`).
- Month indexing fix: the launch month is now the month that contains `operational_start_date`, so launch month = 1, the month before launch = 0, and earlier months are negative.
- Rows impacted by that correction in `less_than-2yrs.csv`: {validation['lt2_rows_with_changed_month_number']:,} rows across {validation['lt2_sites_with_changed_month_number']} sites.
- Prelaunch rows still present in the raw less-than-2-years panel: {validation['lt2_prelaunch_rows']:,}. These rows stay in the data for transparency but are excluded from new-site launch visuals before month 1.
- Local-market rule: prefer same-ZIP older sites when available, but still require the match to be within {max_neighbor_miles:.0f} miles.
- Pre/post window: {pre_post_window} months before vs {pre_post_window} months after the launch month.

## Two-body findings
- Usable nearby pairs: {len(pair_deltas)}.
- Existing-site source: {(pair_deltas['existing_cohort'] == 'gt2').sum() if 'existing_cohort' in pair_deltas.columns else 'n/a'} from gt2 monthly, {(pair_deltas['existing_cohort'] == 'lt2').sum() if 'existing_cohort' in pair_deltas.columns else 'n/a'} from lt2.
- New-site source: {(pair_deltas['new_cohort'] == 'lt2').sum() if 'new_cohort' in pair_deltas.columns else 'n/a'} from lt2, {(pair_deltas['new_cohort'] == 'gt2').sum() if 'new_cohort' in pair_deltas.columns else 'n/a'} from gt2 monthly.
- Median existing-site total change: {fmt_pct(pair_deltas['existing_pct_wash_count_total'].median())}.
- Median combined-market total change: {fmt_pct(pair_deltas['combined_pct_wash_count_total'].median())}.
- Same-ZIP share among usable pairs: {(pair_deltas['zip_match'].mean() * 100):.0f}%.
- Most common regime: {pair_deltas['regime'].value_counts().index[0]}.
- In the event-time chart, the median combined market reaches {fmt_num(pair_profile['combined_index'].max(), 1)} on the pre-launch index scale while the existing site bottoms at {fmt_num(pair_profile['existing_index'].min(), 1)}.

## Three-body findings
- Usable triples: {len(triple_deltas)}.
- Median A-site total change after C launches: {fmt_pct(triple_deltas['A_pct_wash_count_total'].median())}.
- Median B-site total change after C launches: {fmt_pct(triple_deltas['B_pct_wash_count_total'].median())}.
- Median full A+B+C market total change: {fmt_pct(triple_deltas['region_pct_wash_count_total'].median())}.
- In the event-time chart, the full market peaks at {fmt_num(triple_profile['region_index'].max(), 1)} on the A+B pre-launch index scale.

## Plain-English takeaway
- The fatal month-indexing bug is fixed: all launch comparisons now align to the launch month, not the panel's January 2024 calendar.
- Nearby launches usually pressure the closest older site, but the local market often still expands once the new site is added.
- The cleaner event-time plots are the easiest way to read the result: blue and green older-site lines soften after launch, while the orange combined-market line usually stays above the pre-launch baseline.
"""
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "site_interaction_report.md"
    report_path.write_text(report)
    return report_path


INTERACTION_README = """# Site interaction outputs

## plots/two_body/
- `examples_all_sites.png` — every usable pair (calendar grid)
- `avg_existing_single_new_multi_trend.png` — pooled median when existing=single, new=multi
- `trends_by_site_type_combo.png` — 2×2 grid for all four single/multi combinations

## plots/three_body/
- `examples_all_sites.png` — every usable triple
- `avg_all_triples_trend.png` — by role + single/multi (unique color per series)
- `avg_all_triples_trend_overall.png` — overall pool dual panel: month 0 = B (middle) | month 0 = C (newest)
- `avg_new_multi_intro_trend.png` — subset when newest site (C) is multi

## plots/four_body/
- `examples_all_sites.png` — every usable quad
- `avg_all_quads_trend.png` — by role + single/multi (unique color per series)
- `avg_all_quads_trend_overall.png` — overall pool dual panel: month 0 = C (middle) | month 0 = D (newest)

## plots/aggregate/
- `any_new_operator_effect.png` — all pairs: incumbent + combined market
- `market_saturation_threshold.png` — loss-making zone (incumbent ≤−5%, combined ≤+5%)

## data/
- `two_body_pair_deltas.csv`, `three_body_triple_deltas.csv`, `four_body_quad_deltas.csv`

## report/
- `site_interaction_report.md`

## backtesting/
Factor hypothesis from `backtesting.xlsx` matched to panel sites (n=18). See `backtesting/README.md`.

Re-run `site_interaction_analysis.ipynb`, `python run_site_interaction_plots.py`, or `python backtesting_analysis.py`.
"""


def prepare_interaction_dirs(out_dir: Path) -> dict[str, Path]:
    plots = out_dir / "plots"
    backtesting = out_dir / "backtesting"
    dirs = {
        "plots": plots,
        "two_body": plots / "two_body",
        "three_body": plots / "three_body",
        "four_body": plots / "four_body",
        "aggregate": plots / "aggregate",
        "data": out_dir / "data",
        "report": out_dir / "report",
        "backtesting": backtesting,
        "backtesting_plots": backtesting / "plots",
        "backtesting_data": backtesting / "data",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_interaction_readme(out_dir: Path) -> Path:
    readme_path = out_dir / "README.md"
    readme_path.write_text(INTERACTION_README)
    return readme_path


def interaction_keep_set() -> set[str]:
    return {
        "README.md",
        "plots/two_body/examples_all_sites.png",
        "plots/two_body/avg_existing_single_new_multi_trend.png",
        "plots/two_body/trends_by_site_type_combo.png",
        "plots/three_body/examples_all_sites.png",
        "plots/three_body/avg_all_triples_trend.png",
        "plots/three_body/avg_all_triples_trend_overall.png",
        "plots/three_body/avg_new_multi_intro_trend.png",
        "plots/four_body/examples_all_sites.png",
        "plots/four_body/avg_all_quads_trend.png",
        "plots/four_body/avg_all_quads_trend_overall.png",
        "plots/aggregate/any_new_operator_effect.png",
        "plots/aggregate/market_saturation_threshold.png",
        "data/two_body_pair_deltas.csv",
        "data/three_body_triple_deltas.csv",
        "data/four_body_quad_deltas.csv",
        "report/site_interaction_report.md",
        "backtesting/README.md",
        "backtesting/data/backtesting_matched.csv",
        "backtesting/data/backtesting_clusters.csv",
        "backtesting/plots/factor_correlation_heatmap.png",
        "backtesting/plots/wash_vs_traffic.png",
        "backtesting/plots/cluster_localisation_1.png",
        "backtesting/plots/cluster_localisation_1_washes_2024_2025.png",
        "backtesting/plots/cluster_localisation_2.png",
        "backtesting/plots/cluster_localisation_2_washes_2024_2025.png",
        "backtesting/plots/cluster_localisation_3.png",
        "backtesting/plots/cluster_localisation_3_washes_2024_2025.png",
        "backtesting/plots/cluster_localisation_4.png",
        "backtesting/plots/cluster_localisation_4_washes_2024_2025.png",
    }


def curate_outputs(out_dir: Path, keep: set[str] | None = None) -> None:
    keep_relative = interaction_keep_set() if keep is None else {p.replace("\\", "/") for p in keep}
    if not out_dir.exists():
        return
    for path in sorted(out_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if not path.is_file():
            continue
        rel = path.relative_to(out_dir).as_posix()
        if rel not in keep_relative:
            path.unlink()
    for path in sorted(out_dir.rglob("*"), reverse=True):
        if path.is_dir() and path != out_dir and not any(path.iterdir()):
            path.rmdir()
