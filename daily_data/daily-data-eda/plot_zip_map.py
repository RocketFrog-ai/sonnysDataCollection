#!/usr/bin/env python3
"""
Create an interactive US map of zip codes with bubble size by total washes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


EDA_DIR = Path(__file__).resolve().parent
CSV_PATH = EDA_DIR / "master_daily_with_site_metadata.csv"
LOOKUP_PATH = EDA_DIR / "zip_lat_lon_lookup.csv"
OUT_HTML = EDA_DIR / "plots" / "08_zip_map_wash_volume.html"


def normalize_zip(v: object) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if "." in s:
        s = s.split(".", 1)[0]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(5)[:5]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing source CSV: {CSV_PATH}")
    if not LOOKUP_PATH.exists():
        raise SystemExit(
            f"Missing geocode file: {LOOKUP_PATH}\nRun build_zip_geocode_list.py first."
        )

    df = pd.read_csv(CSV_PATH, usecols=["zip", "wash_count_total", "region", "state", "city"])
    df["wash_count_total"] = pd.to_numeric(df["wash_count_total"], errors="coerce").fillna(0)
    df["zip5"] = df["zip"].map(normalize_zip)
    agg = (
        df.dropna(subset=["zip5"])
        .groupby("zip5", as_index=False)
        .agg(
            total_washes=("wash_count_total", "sum"),
            avg_washes=("wash_count_total", "mean"),
            n_site_days=("wash_count_total", "size"),
            region=("region", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
            state=("state", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
            city=("city", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else None),
        )
    )

    geo = pd.read_csv(
        LOOKUP_PATH,
        usecols=["zip5", "latitude", "longitude", "has_lat_lon"],
        dtype={"zip5": str},
    )
    geo["zip5"] = geo["zip5"].map(normalize_zip)
    m = agg.merge(geo, on="zip5", how="left")
    m = m[m["has_lat_lon"] == True].copy()  # noqa: E712

    if m.empty:
        raise SystemExit("No zip points with lat/lon found to plot.")

    m["hover_zip"] = m["zip5"].astype(str)
    fig = px.scatter_map(
        m,
        lat="latitude",
        lon="longitude",
        size="total_washes",
        color="region",
        hover_name="hover_zip",
        hover_data={
            "city": True,
            "state": True,
            "total_washes": ":,.0f",
            "avg_washes": ":.1f",
            "n_site_days": ":,",
            "latitude": False,
            "longitude": False,
            "region": False,
            "hover_zip": False,
        },
        zoom=3.1,
        center={"lat": 37.8, "lon": -96.0},
        map_style="open-street-map",
        size_max=24,
        opacity=0.55,
        title="Zip-level car wash volume map (bubble size = total wash_count_total)",
    )
    fig.update_layout(margin=dict(l=12, r=12, t=46, b=8), legend_title_text="Region")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML))
    print(f"Wrote: {OUT_HTML}")
    print(f"Plotted zips: {len(m):,}")


if __name__ == "__main__":
    main()
