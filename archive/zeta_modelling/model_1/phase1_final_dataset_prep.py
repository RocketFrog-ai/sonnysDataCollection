from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FINAL_COLS = [
    "site_id",
    "date",
    "age_in_months",
    "real_age_months",
    "monthly_volume",
    "latitude",
    "longitude",
    "cluster_id",
    "maturity_bucket",
]


def make_site_id(df: pd.DataFrame) -> pd.Series:
    def _normalize_site_client_id(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        numeric = pd.to_numeric(s, errors="coerce")
        is_int_like = numeric.notna() & (numeric % 1 == 0)
        s.loc[is_int_like] = numeric.loc[is_int_like].astype("Int64").astype(str)
        is_float_like = numeric.notna() & ~is_int_like
        s.loc[is_float_like] = numeric.loc[is_float_like].map(
            lambda x: f"{x:.6f}".rstrip("0").rstrip(".")
        )
        return s

    if "site_client_id" in df.columns and df["site_client_id"].notna().any():
        return _normalize_site_client_id(df["site_client_id"])

    if {"latitude", "longitude"}.issubset(df.columns):
        return (
            df["latitude"].round(6).astype(str)
            + "_"
            + df["longitude"].round(6).astype(str)
        )

    raise ValueError("Could not create site_id. Need site_client_id or latitude/longitude.")


def build_date_from_year_month(df: pd.DataFrame) -> pd.Series:
    month_number = pd.to_numeric(df["month_number"], errors="coerce")

    year = pd.Series(pd.NA, index=df.index, dtype="Int64")
    month = pd.Series(pd.NA, index=df.index, dtype="Int64")

    in_2024 = month_number.between(1, 12)
    in_2025 = month_number.between(13, 24)

    year.loc[in_2024] = 2024
    month.loc[in_2024] = month_number.loc[in_2024].astype("Int64")

    year.loc[in_2025] = 2025
    month.loc[in_2025] = (month_number.loc[in_2025] - 12).astype("Int64")

    date = pd.to_datetime(
        {
            "year": year,
            "month": month,
            "day": 1,
        },
        errors="coerce",
    )
    return date


def maturity_bucket(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x < 36:
        return "young"
    if x < 84:
        return "mid"
    return "mature"


def standardize_less_than_2yrs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site_id"] = make_site_id(out)
    out["date"] = build_date_from_year_month(out)
    out = out.dropna(subset=["site_id", "date"])
    out = out.sort_values(["site_id", "date"])

    out["age_in_months"] = out.groupby("site_id").cumcount() + 1
    out["real_age_months"] = out["age_in_months"]
    out["monthly_volume"] = pd.to_numeric(out["wash_count_total"], errors="coerce").fillna(0)
    out["cluster_id"] = out["dbscan_cluster_12km"] if "dbscan_cluster_12km" in out.columns else pd.NA
    out["maturity_bucket"] = out["real_age_months"].apply(maturity_bucket)

    return out[FINAL_COLS]


def standardize_more_than_2yrs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site_id"] = make_site_id(out)
    out["date"] = build_date_from_year_month(out)
    out = out.dropna(subset=["site_id", "date"])

    agg_map: dict[str, str] = {
        "wash_count_total": "sum",
        "latitude": "first",
        "longitude": "first",
    }
    if "dbscan_cluster_12km" in out.columns:
        agg_map["dbscan_cluster_12km"] = "first"
    if "age_on_30_sep_25" in out.columns:
        agg_map["age_on_30_sep_25"] = "first"

    monthly = (
        out.groupby(["site_id", "date"], as_index=False)
        .agg(agg_map)
        .sort_values(["site_id", "date"])
    )

    monthly["age_in_months"] = monthly.groupby("site_id").cumcount() + 1 + 24
    if "age_on_30_sep_25" in monthly.columns:
        monthly["real_age_months"] = pd.to_numeric(monthly["age_on_30_sep_25"], errors="coerce") * 12
    else:
        monthly["real_age_months"] = monthly["age_in_months"]

    monthly["monthly_volume"] = pd.to_numeric(monthly["wash_count_total"], errors="coerce").fillna(0)
    monthly["cluster_id"] = (
        monthly["dbscan_cluster_12km"] if "dbscan_cluster_12km" in monthly.columns else pd.NA
    )
    monthly["maturity_bucket"] = monthly["real_age_months"].apply(maturity_bucket)

    return monthly[FINAL_COLS]


def validate(final_df: pd.DataFrame) -> None:
    dmin = final_df["date"].min()
    dmax = final_df["date"].max()
    leakage = final_df[final_df["date"] > pd.Timestamp("2025-12-31")]
    dup_count = int(final_df.duplicated(subset=["site_id", "date"]).sum())
    dup_count_numeric_site = int(
        final_df.assign(site_id_numeric=pd.to_numeric(final_df["site_id"], errors="coerce"))
        .duplicated(subset=["site_id_numeric", "date"])
        .sum()
    )
    bad_age_count = int((final_df["real_age_months"] < final_df["age_in_months"]).sum())
    age_stats = final_df[["age_in_months", "real_age_months"]].describe(include="all")
    missing = final_df.isnull().sum()

    print("Check 1: Date range")
    print(dmin, dmax)
    print("\nCheck 2: No leakage rows > 2025-12-31")
    print(len(leakage))
    print("\nCheck 3: Duplicate rows on (site_id, date)")
    print(dup_count)
    print("\nCheck 4: Duplicate rows after numeric site_id coercion")
    print(dup_count_numeric_site)
    print("\nCheck 5: real_age_months >= age_in_months violations")
    print(bad_age_count)
    print("\nCheck 6: Age sanity describe")
    print(age_stats)
    print("\nCheck 7: Missing values")
    print(missing)

    if pd.isna(dmin) or pd.isna(dmax):
        raise ValueError("Date creation failed; found NaT range.")
    if dmin < pd.Timestamp("2024-01-01") or dmax > pd.Timestamp("2025-12-01"):
        raise ValueError(f"Unexpected date range: {dmin} to {dmax}")
    if len(leakage) > 0:
        raise ValueError("Leakage detected: rows after 2025-12-31 exist.")
    if dup_count > 0:
        raise ValueError("Duplicate (site_id, date) rows found.")
    if dup_count_numeric_site > 0:
        raise ValueError("Duplicate (numeric(site_id), date) rows found.")
    if bad_age_count > 0:
        raise ValueError("Found rows where real_age_months < age_in_months.")


def build_final_dataset(less_path: Path, more_path: Path) -> pd.DataFrame:
    less_df = pd.read_csv(less_path, low_memory=False)
    more_df = pd.read_csv(more_path, low_memory=False)

    less_final = standardize_less_than_2yrs(less_df)
    more_final = standardize_more_than_2yrs(more_df)

    final_df = pd.concat([less_final, more_final], ignore_index=True)

    # Critical cleanup 1: enforce one row per (site_id, date).
    final_df = final_df.sort_values(["site_id", "date", "age_in_months"])
    final_df = final_df.drop_duplicates(subset=["site_id", "date"], keep="first")

    # Critical cleanup 2: enforce real age >= lifecycle age.
    final_df["real_age_months"] = final_df[["real_age_months", "age_in_months"]].max(axis=1)

    # Minor cleanup: remove rows without geolocation.
    final_df = final_df.dropna(subset=["latitude", "longitude"])

    # Recompute bucket after real age correction.
    final_df["maturity_bucket"] = final_df["real_age_months"].apply(maturity_bucket)
    final_df = final_df.sort_values(["site_id", "date"]).reset_index(drop=True)
    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 final monthly 2024-2025 dataset prep.")
    parser.add_argument(
        "--less-input",
        type=Path,
        default=Path("zeta_modelling/data/less_than-2yrs.csv"),
    )
    parser.add_argument(
        "--more-input",
        type=Path,
        default=Path("zeta_modelling/data/more_than-2yrs.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_df = build_final_dataset(args.less_input, args.more_input)
    validate(final_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(final_df):,} rows to {args.output}")
    print("Final columns:", list(final_df.columns))


if __name__ == "__main__":
    main()
