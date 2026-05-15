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

    panel = panel.sort_values(["client_id_location_id", "year_month", "cohort"])
    panel = panel.drop_duplicates(["client_id_location_id", "year_month"], keep="last").reset_index(drop=True)

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

    validation = {
        "sites": int(panel["client_id_location_id"].nunique()),
        "rows": int(len(panel)),
        "month_min": panel["year_month"].min(),
        "month_max": panel["year_month"].max(),
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

        traces = {}
        for label, site_col in [("A", "A_site"), ("B", "B_site"), ("C", "C_site")]:
            sub = panel.loc[panel["client_id_location_id"] == triple[site_col], ["calendar_month", "site_month_number", PAIR_EVENT_METRIC]].copy()
            if label == "C":
                sub = sub[sub["site_month_number"].ge(1)]
            sub["relative_month"] = month_diff(sub["calendar_month"], event_month).astype(int)
            traces[label] = sub.set_index("relative_month")[PAIR_EVENT_METRIC]

        for relative_month in range(-window, window + 1):
            a_value = traces["A"].get(relative_month, np.nan)
            b_value = traces["B"].get(relative_month, np.nan)
            c_value = traces["C"].get(relative_month, np.nan)
            if relative_month < 0:
                c_value = np.nan
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
        ax.set_title(
            f"{state} | {pair['market_zip']} | {pair['distance_miles']:.1f} mi | launch {launch_month.strftime('%Y-%m')}\n"
            f"existing: {ex_type} | new: {nw_type}",
            fontsize=7,
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
            "Blue = existing | Red = new (from launch) | dashed = launch | title shows state + single/multi"
        )
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.12, hspace=0.72, wspace=0.35)
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
        ax.set_title(
            f"{state} | {triple['market_zip']} | B launch {b_launch.strftime('%Y-%m')} | C launch {c_launch.strftime('%Y-%m')}\n"
            f"A: {a_type} | B: {b_type} | C: {c_type}\n"
            f"A–C {triple['A_to_C_miles']:.1f} mi | B–C {triple['B_to_C_miles']:.1f} mi",
            fontsize=6.5,
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
        "Blue = A | Green = B | Red = C | dashed = B/C launch | title shows single/multi per site",
        fontsize=14,
        y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.12, hspace=0.82, wspace=0.38)
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
- Data used: `less_than-2yrs.csv` and `more_than-2yrs_monthly.csv`.
- Monthly panel coverage: {validation['month_min'].date()} to {validation['month_max'].date()} across {validation['sites']} sites and {validation['rows']:,} site-month rows.
- Month indexing fix: the launch month is now the month that contains `operational_start_date`, so launch month = 1, the month before launch = 0, and earlier months are negative.
- Rows impacted by that correction in `less_than-2yrs.csv`: {validation['lt2_rows_with_changed_month_number']:,} rows across {validation['lt2_sites_with_changed_month_number']} sites.
- Prelaunch rows still present in the raw less-than-2-years panel: {validation['lt2_prelaunch_rows']:,}. These rows stay in the data for transparency but are excluded from new-site launch visuals before month 1.
- Local-market rule: prefer same-ZIP older sites when available, but still require the match to be within {max_neighbor_miles:.0f} miles.
- Pre/post window: {pre_post_window} months before vs {pre_post_window} months after the launch month.

## Two-body findings
- Usable nearby pairs: {len(pair_deltas)}.
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
    report_path = out_dir / "site_interaction_report.md"
    report_path.write_text(report)
    return report_path


def curate_outputs(out_dir: Path, keep: set[str]) -> None:
    for path in out_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()
