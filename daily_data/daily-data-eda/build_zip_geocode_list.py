#!/usr/bin/env python3
"""
Build a zip-code geocode table (lat/lon) for the carwash panel.

Uses pgeocode (US postal database) for deterministic, no-API-key lookup.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pgeocode


EDA_DIR = Path(__file__).resolve().parent
CSV_PATH = EDA_DIR / "master_daily_with_site_metadata.csv"
OUT_CSV = EDA_DIR / "zip_lat_lon_lookup.csv"


def normalize_zip(v: object) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    # Keep 5-digit zip; some values may be floats/strings.
    if "." in s:
        s = s.split(".", 1)[0]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(5)[:5]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing source CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, usecols=["zip", "state", "city"])
    df["zip5"] = df["zip"].map(normalize_zip)
    base = df.dropna(subset=["zip5"]).copy()

    nomi = pgeocode.Nominatim("us")
    zips = sorted(base["zip5"].unique())
    g = nomi.query_postal_code(zips)

    geo = pd.DataFrame(
        {
            "zip5": g["postal_code"].astype(str),
            "latitude": pd.to_numeric(g["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(g["longitude"], errors="coerce"),
            "state_geocode": g["state_code"].astype(str),
            "county_name": g.get("county_name", pd.Series([None] * len(g))),
        }
    )

    ref = (
        base.groupby("zip5", as_index=False)
        .agg(
            city_sample=("city", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
            state_sample=("state", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
            n_site_days=("zip5", "size"),
        )
        .sort_values("zip5")
    )

    out = ref.merge(geo, on="zip5", how="left")
    out["has_lat_lon"] = out["latitude"].notna() & out["longitude"].notna()
    out.to_csv(OUT_CSV, index=False)

    print(f"Wrote: {OUT_CSV}")
    print(f"Unique zips: {len(out):,}")
    print(f"With lat/lon: {int(out['has_lat_lon'].sum()):,}")
    print(f"Missing lat/lon: {int((~out['has_lat_lon']).sum()):,}")


if __name__ == "__main__":
    main()
