"""Data loading, cleaning, deduplication, and panel construction.

Key decisions:
* LT (less_than-2yrs) raw file has year_number=1→2024, year_number=2→2026 in
  the year_month column. The 2026 label is wrong — those are year-2 months
  (2025) mislabelled. We fix by replacing '2026' with '2025' on year_number=2
  rows using the month_number field to reconstruct the correct YYYY-MM.
* No 2026 data is used at all (OBSERVED_CUTOFF_YM = 2025-12).
* operational_start_date imputation:
    - MT (>2yr): if missing, use age_on_30_sep_25 (integer years) to back-
      calculate: op_start = Sep 30 2025 minus age_years * 365 days.
    - LT (<2yr): if missing, use the site's first observed year_month (the LT
      dataset is constructed so month_number=1 = the open month).
* Drop rows after OBSERVED_CUTOFF_YM; drop Australian sites; dedupe.
* Floor extreme-low wash counts (bottom 0.5%) at 1; drop sites < 3 months.
* Short interior gap fill (≤2 consecutive months).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


US_BBOX = dict(lat_min=24.0, lat_max=50.0, lon_min=-125.0, lon_max=-66.0)
ALASKA_HAWAII = dict(lat_min=18.0, lat_max=71.5, lon_min=-178.0, lon_max=-66.0)

# Sep 30 2025 — reference date for age_on_30_sep_25 in MT file.
_AGE_REF_DATE = pd.Timestamp("2025-09-30")


def _in_us(lat: pd.Series, lon: pd.Series) -> pd.Series:
    in_contig = lat.between(US_BBOX["lat_min"], US_BBOX["lat_max"]) & lon.between(
        US_BBOX["lon_min"], US_BBOX["lon_max"]
    )
    in_akhi = lat.between(ALASKA_HAWAII["lat_min"], ALASKA_HAWAII["lat_max"]) & lon.between(
        ALASKA_HAWAII["lon_min"], ALASKA_HAWAII["lon_max"]
    )
    return in_contig | in_akhi


def _fix_lt_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """Remap LT year_number=2 rows from mislabelled 2026 -> correct 2025.

    The LT raw file labels year_number=1 as 2024 (correct) and year_number=2
    as 2026 (wrong — those are the second-year months, i.e. 2025). The
    month_number column (1..24) gives the calendar month unambiguously:
      year_number=1 -> month_number 1..12  -> YYYY = 2024, MM = month_number
      year_number=2 -> month_number 13..24 -> YYYY = 2025, MM = month_number-12
    """
    if "year_number" not in df.columns or "month_number" not in df.columns:
        return df
    mask2 = df["year_number"] == 2
    cal_month = (df.loc[mask2, "month_number"] - 12).clip(1, 12)
    df.loc[mask2, "year_month"] = "2025-" + cal_month.astype(str).str.zfill(2)
    return df


def _impute_lt_op_start(df: pd.DataFrame) -> pd.DataFrame:
    """For LT sites missing operational_start_date, use the first observed month.

    NOTE: this is a weak fallback. The LT file is a *calendar* snapshot —
    month_number=1 is Jan-2024 for every site, NOT the site's open month — so
    the earliest observed year_month is only a lower-bound proxy for the open
    date. With EXCLUDE_CHEM=True every surviving LT (control) site already has
    a real operational_start_date, so this path should not fire.
    """
    missing = df["operational_start_date"].isna()
    if not missing.any():
        return df
    first_obs = (
        df.groupby("client_id_location_id")["ym_ts"]
        .min()
        .rename("first_ym_ts")
    )
    df = df.join(first_obs, on="client_id_location_id")
    df.loc[missing, "operational_start_date"] = df.loc[missing, "first_ym_ts"]
    df = df.drop(columns=["first_ym_ts"])
    return df


def _impute_mt_op_start(df: pd.DataFrame) -> pd.DataFrame:
    """For MT sites missing operational_start_date, use age_on_30_sep_25.

    age_on_30_sep_25 is the site's age in full years at Sep 30 2025.
    op_start ≈ _AGE_REF_DATE - age_years * 365 days.
    """
    missing = df["operational_start_date"].isna()
    if not missing.any() or "age_on_30_sep_25" not in df.columns:
        return df
    age_yrs = pd.to_numeric(df.loc[missing, "age_on_30_sep_25"], errors="coerce").fillna(2)
    implied = _AGE_REF_DATE - pd.to_timedelta(age_yrs * 365, unit="D")
    df.loc[missing, "operational_start_date"] = implied.values
    return df


def _load_lt() -> pd.DataFrame:
    df = pd.read_csv(C.LT_CSV, low_memory=False)
    keep = [
        "client_id_location_id", "operational_start_date", "year_month",
        "wash_count_total", "latitude", "longitude", "state", "region",
        "city", "zip", "source", "year_number", "month_number",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    # Drop the chem-source cohort: those rows have no operational_start_date,
    # are duplicated, and their wash volumes are confounded by the chemical
    # program. Only the control cohort is a clean baseline for forecasting.
    if C.EXCLUDE_CHEM and "source" in df.columns:
        before = df["client_id_location_id"].nunique()
        df = df[df["source"] != "chem"].copy()
        after = df["client_id_location_id"].nunique()
        print(f"[data_prep] EXCLUDE_CHEM: LT sites {before} -> {after} "
              f"(dropped {before - after} chem-source sites)")
    # Fix mislabelled year before any further processing.
    df = _fix_lt_year_month(df)
    df["cohort"] = "young"
    return df


def _load_mt() -> pd.DataFrame:
    df = pd.read_csv(C.MT_CSV, low_memory=False)
    keep = [
        "client_id_location_id", "operational_start_date", "year_month",
        "wash_count_total", "latitude", "longitude", "state", "region",
        "city", "zip", "primary_carwash_type", "age_on_30_sep_25",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["cohort"] = "mature"
    return df


def load_and_clean() -> pd.DataFrame:
    """Return one long-format panel with both cohorts cleaned and aligned."""
    lt = _load_lt()
    mt = _load_mt()
    df = pd.concat([lt, mt], ignore_index=True, sort=False)

    # 1) types
    df["year_month"] = df["year_month"].astype(str).str[:7]
    df["wash_count_total"] = pd.to_numeric(df["wash_count_total"], errors="coerce")
    for c in ("latitude", "longitude"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["operational_start_date"] = pd.to_datetime(df["operational_start_date"], errors="coerce")
    df["ym_ts"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")

    # 2) drop rows with missing essentials
    before = len(df)
    df = df.dropna(subset=["client_id_location_id", "year_month", "wash_count_total",
                            "latitude", "longitude", "ym_ts"])
    df = df[df["wash_count_total"] > 0]

    # 3) US-only filter
    df = df[_in_us(df["latitude"], df["longitude"])].copy()

    # 4) drop future / projected months (no 2026 data)
    df = df[df["year_month"] <= C.OBSERVED_CUTOFF_YM].copy()

    # 5) dedupe (site, ym) keep max
    df = df.sort_values(["client_id_location_id", "year_month", "wash_count_total"])
    df = df.drop_duplicates(["client_id_location_id", "year_month"], keep="last")

    # 6) impute operational_start_date
    #    LT: use first observed month; MT: use age_on_30_sep_25
    lt_mask = df["cohort"] == "young"
    mt_mask = df["cohort"] == "mature"
    df_lt = _impute_lt_op_start(df[lt_mask].copy())
    df_mt = _impute_mt_op_start(df[mt_mask].copy())
    df = pd.concat([df_lt, df_mt], ignore_index=True, sort=False)
    # Any remaining nulls (shouldn't be any) fall back to first observed month.
    still_missing = df["operational_start_date"].isna()
    if still_missing.any():
        first_obs = df.groupby("client_id_location_id")["ym_ts"].min().rename("_fo")
        df = df.join(first_obs, on="client_id_location_id")
        df.loc[still_missing, "operational_start_date"] = df.loc[still_missing, "_fo"]
        df = df.drop(columns=["_fo"])

    # 7) site_age_months — months since opening (0 = open month)
    df["site_age_months"] = (
        (df["ym_ts"].dt.year - df["operational_start_date"].dt.year) * 12
        + (df["ym_ts"].dt.month - df["operational_start_date"].dt.month)
    )

    # 8) drop pre-open rows (negative age) — data artefacts
    df = df[df["site_age_months"] >= 0].copy()

    # 9) drop the partial first month if op_start is mid-month (>= day 15)
    partial = (df["operational_start_date"].dt.day >= 15) & (df["site_age_months"] == 0)
    df = df[~partial].copy()

    # 10) floor extreme low values (likely partial reads) — keep upper tail
    floor = df["wash_count_total"].quantile(0.005)
    df["wash_count_total"] = df["wash_count_total"].clip(lower=max(floor, 1))

    # 11) drop sites with < 3 months of history
    n_by_site = df.groupby("client_id_location_id")["year_month"].nunique()
    keep_sites = n_by_site[n_by_site >= 3].index
    df = df[df["client_id_location_id"].isin(keep_sites)].copy()

    after = len(df)
    print(f"[data_prep] rows {before} -> {after}; sites kept: {df['client_id_location_id'].nunique()}")

    # 12) short-gap repair within contiguous segments (≤2 months only).
    df = _fill_short_gaps(df, max_gap_months=2)

    df = df.reset_index(drop=True)
    df["calendar_month"] = df["ym_ts"].dt.month
    df["calendar_year"] = df["ym_ts"].dt.year
    return df


def _fill_short_gaps(df: pd.DataFrame, max_gap_months: int = 2) -> pd.DataFrame:
    """Patch only short interior gaps (≤ max_gap_months) within a site's history."""
    static_cols = ["operational_start_date", "latitude", "longitude", "state",
                   "region", "city", "zip", "cohort", "source",
                   "primary_carwash_type", "age_on_30_sep_25"]
    static_cols = [c for c in static_cols if c in df.columns]
    out_pieces = []
    for sid, g in df.groupby("client_id_location_id", sort=False):
        g = g.sort_values("ym_ts").reset_index(drop=True)
        diffs = ((g["ym_ts"].diff().dt.days / 30).round()).fillna(1).astype(int)
        patches = []
        for i in range(1, len(g)):
            step = diffs.iloc[i]
            if 1 < step <= max_gap_months + 1:
                a, b = g.iloc[i - 1]["wash_count_total"], g.iloc[i]["wash_count_total"]
                for k in range(1, step):
                    new_ts = g.iloc[i - 1]["ym_ts"] + pd.DateOffset(months=k)
                    val = a + (b - a) * (k / step)
                    row = {c: g.iloc[i - 1][c] for c in static_cols}
                    row["client_id_location_id"] = sid
                    row["ym_ts"] = new_ts
                    row["year_month"] = new_ts.strftime("%Y-%m")
                    row["wash_count_total"] = val
                    patches.append(row)
        out_pieces.append(g)
        if patches:
            out_pieces.append(pd.DataFrame(patches))
    out = pd.concat(out_pieces, ignore_index=True, sort=False)
    out["site_age_months"] = (
        (out["ym_ts"].dt.year - out["operational_start_date"].dt.year) * 12
        + (out["ym_ts"].dt.month - out["operational_start_date"].dt.month)
    )
    return out


def site_static_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per site with static attributes — used for market assignment."""
    cols = ["client_id_location_id", "operational_start_date", "latitude",
            "longitude", "state", "region", "city", "zip", "cohort"]
    cols = [c for c in cols if c in panel.columns]
    out = panel[cols].drop_duplicates("client_id_location_id").reset_index(drop=True)
    return out
