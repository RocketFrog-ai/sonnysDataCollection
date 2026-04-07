"""
Build a long-format CSV: one row per site per day for calendar year 2025.

Reads site rows from modelling2/finale.csv, builds an address from
client_id + street + city + state + zip, geocodes to lat/lon, then fetches
daily fields from Open-Meteo Archive (same stack as app/features/active/weather/open_meteo.py).

Weather is requested once per distinct (lat, lon) after rounding, then merged back to every site
so duplicate coordinates do not multiply API calls.

Usage (from project root):
  python modelling2/fetch_daily_weather_2025.py
  python modelling2/fetch_daily_weather_2025.py --start 0 --end 50

Env: optional TomTom geocoding (TOMTOM_* in .env). If TomTom is unavailable or returns no
result, falls back to Open-Meteo Geocoding API (no key).

Rate limiting: minimum delay between geocode and between weather requests, plus built-in
429 handling in fetch_open_meteo_weather_data.

Checkpointing: appends to the output CSV every N sites (default 50). A full run from
--start 0 removes an existing output file first; use --start > 0 to append to an existing file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from app.features.active.weather.open_meteo import fetch_open_meteo_weather_data
from app.utils import common as calib

FINALE_CSV = Path(__file__).resolve().parent / "finale.csv"
OUT_CSV = Path(__file__).resolve().parent / "finale_daily_weather_2025.csv"
GEOCODE_CACHE_PATH = Path(__file__).resolve().parent / ".finale_geocode_cache.json"

WEATHER_START = "2025-01-01"
WEATHER_END = "2025-12-31"

# Be conservative with Open-Meteo / geocoding (non-commercial fair use)
DEFAULT_GEOCODE_DELAY_SEC = float(os.getenv("WEATHER_2025_GEOCODE_DELAY_SEC", "1.0"))
DEFAULT_WEATHER_DELAY_SEC = float(os.getenv("WEATHER_2025_ARCHIVE_DELAY_SEC", "1.2"))

OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
COORD_ROUND_DECIMALS = 4

DEFAULT_SAVE_EVERY_SITES = int(os.getenv("WEATHER_2025_SAVE_EVERY_SITES", "50"))

OUTPUT_COLUMNS = [
    "site_client_id",
    "location_id",
    "client_id",
    "street",
    "city",
    "state",
    "zip",
    "address_used",
    "latitude",
    "longitude",
    "date",
    "precipitation_sum",
    "snowfall_sum",
    "temperature_2m_min",
    "temperature_2m_max",
    "sunshine_duration",
    "windspeed_10m_max",
    "weather_error",
]


def _build_address(row: pd.Series) -> str:
    parts = [
        str(row.get("client_id", "") or "").strip(),
        str(row.get("street", "") or "").strip(),
        str(row.get("city", "") or "").strip(),
        str(row.get("state", "") or "").strip(),
        str(row.get("zip", "") or "").strip(),
    ]
    return ", ".join(p for p in parts if p)


def _load_geocode_cache() -> dict:
    if not GEOCODE_CACHE_PATH.exists():
        return {}
    try:
        with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_geocode_cache(cache: dict) -> None:
    with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=0, sort_keys=True)


def _geocode_open_meteo(address: str, timeout: int = 30) -> tuple[float, float] | None:
    """Free geocoding fallback; US-focused query."""
    params = {
        "name": address,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    try:
        r = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(5 + random.random() * 2)
            r = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if not results:
            return None
        lat = results[0].get("latitude")
        lon = results[0].get("longitude")
        if lat is None or lon is None:
            return None
        return float(lat), float(lon)
    except requests.RequestException:
        return None


def _geocode(address: str, cache: dict, delay_sec: float) -> tuple[float, float] | None:
    if not address.strip():
        return None
    key = address.strip()
    if key in cache:
        entry = cache[key]
        if entry.get("lat") is not None and entry.get("lon") is not None:
            return float(entry["lat"]), float(entry["lon"])
        return None

    time.sleep(delay_sec)
    geo = None
    if calib.TOMTOM_API_KEY and calib.TOMTOM_GEOCODE_API_URL:
        geo = calib.get_lat_long(address)

    if geo:
        lat, lon = float(geo["lat"]), float(geo["lon"])
    else:
        pair = _geocode_open_meteo(address)
        if not pair:
            cache[key] = {"lat": None, "lon": None, "source": "none"}
            _save_geocode_cache(cache)
            return None
        lat, lon = pair

    cache[key] = {"lat": lat, "lon": lon, "source": "tomtom" if geo else "open_meteo_geocode"}
    _save_geocode_cache(cache)
    return lat, lon


def _coord_key(lat: float, lon: float) -> tuple[float, float]:
    return (round(lat, COORD_ROUND_DECIMALS), round(lon, COORD_ROUND_DECIMALS))


def _fetch_daily_for_coords(
    lat: float, lon: float, delay_sec: float
) -> pd.DataFrame:
    time.sleep(delay_sec)
    df = fetch_open_meteo_weather_data(lat, lon, WEATHER_START, WEATHER_END)
    return df


def _daily_rows_for_site(
    row: pd.Series, coord_to_df: dict[tuple[float, float], pd.DataFrame]
) -> list[dict]:
    """Expand one finale row to 365 daily dicts (or placeholders on failure)."""
    lat, lon = row.get("_lat"), row.get("_lon")
    base = {
        "site_client_id": row.get("site_client_id"),
        "location_id": row.get("location_id"),
        "client_id": row.get("client_id"),
        "street": row.get("street"),
        "city": row.get("city"),
        "state": row.get("state"),
        "zip": row.get("zip"),
        "address_used": row.get("_address"),
        "latitude": lat,
        "longitude": lon,
    }
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return [
            {
                **base,
                "date": None,
                "precipitation_sum": None,
                "snowfall_sum": None,
                "temperature_2m_min": None,
                "temperature_2m_max": None,
                "sunshine_duration": None,
                "windspeed_10m_max": None,
                "weather_error": "geocode_failed",
            }
            for _ in range(365)
        ]

    wdf = coord_to_df.get(_coord_key(float(lat), float(lon)))
    if wdf is None or wdf.empty:
        return [
            {
                **base,
                "date": None,
                "precipitation_sum": None,
                "snowfall_sum": None,
                "temperature_2m_min": None,
                "temperature_2m_max": None,
                "sunshine_duration": None,
                "windspeed_10m_max": None,
                "weather_error": "open_meteo_empty",
            }
            for _ in range(365)
        ]

    wdf = wdf.copy().reset_index()
    if "time" not in wdf.columns:
        return [
            {
                **base,
                "date": None,
                "precipitation_sum": None,
                "snowfall_sum": None,
                "temperature_2m_min": None,
                "temperature_2m_max": None,
                "sunshine_duration": None,
                "windspeed_10m_max": None,
                "weather_error": "no_time_column",
            }
            for _ in range(365)
        ]

    wind_col = "windspeed_10m_max"
    if "wind_speed_10m_max" in wdf.columns:
        wind_col = "wind_speed_10m_max"

    out: list[dict] = []
    for _, drow in wdf.iterrows():
        t = drow["time"]
        date_str = pd.Timestamp(t).date().isoformat() if pd.notna(t) else None
        out.append(
            {
                **base,
                "date": date_str,
                "precipitation_sum": drow.get("precipitation_sum"),
                "snowfall_sum": drow.get("snowfall_sum"),
                "temperature_2m_min": drow.get("temperature_2m_min"),
                "temperature_2m_max": drow.get("temperature_2m_max"),
                "sunshine_duration": drow.get("sunshine_duration"),
                "windspeed_10m_max": drow.get(wind_col),
                "weather_error": "",
            }
        )
    return out


def _append_batch_csv(path: Path, rows: list[dict], *, write_header: bool) -> None:
    batch_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    batch_df.to_csv(path, mode="a", header=write_header, index=False)


def run(
    start_row: int = 0,
    end_row: int | None = None,
    geocode_delay_sec: float = DEFAULT_GEOCODE_DELAY_SEC,
    weather_delay_sec: float = DEFAULT_WEATHER_DELAY_SEC,
    output_path: Path | None = None,
    save_every_sites: int = DEFAULT_SAVE_EVERY_SITES,
) -> Path:
    out = output_path or OUT_CSV
    if not FINALE_CSV.exists():
        raise FileNotFoundError(f"Missing input: {FINALE_CSV}")

    df = pd.read_csv(FINALE_CSV)
    n = len(df)
    end_row = n if end_row is None else min(end_row, n)
    start_row = max(0, start_row)
    if start_row >= end_row:
        raise ValueError("start_row must be < end_row")

    slice_df = df.iloc[start_row:end_row].copy()
    slice_df["_address"] = slice_df.apply(_build_address, axis=1)

    save_every_sites = max(1, int(save_every_sites))
    if start_row == 0 and out.exists():
        out.unlink()

    cache = _load_geocode_cache()
    coord_to_df: dict[tuple[float, float], pd.DataFrame] = {}
    total_sites = len(slice_df)

    with tqdm(
        total=total_sites,
        desc=f"Sites (save every {save_every_sites})",
        unit="site",
    ) as pbar:
        for chunk_start in range(0, total_sites, save_every_sites):
            chunk_end = min(chunk_start + save_every_sites, total_sites)
            sub = slice_df.iloc[chunk_start:chunk_end]

            lat_list: list[float | None] = []
            lon_list: list[float | None] = []
            for _, row in sub.iterrows():
                addr = row["_address"]
                if not addr or not str(addr).strip():
                    lat_list.append(None)
                    lon_list.append(None)
                    continue
                pair = _geocode(str(addr), cache, geocode_delay_sec)
                if pair:
                    lat_list.append(pair[0])
                    lon_list.append(pair[1])
                else:
                    lat_list.append(None)
                    lon_list.append(None)

            sub = sub.copy()
            sub["_lat"] = lat_list
            sub["_lon"] = lon_list

            for _, row in sub.iterrows():
                lat, lon = row["_lat"], row["_lon"]
                if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
                    continue
                k = _coord_key(float(lat), float(lon))
                if k not in coord_to_df:
                    coord_to_df[k] = _fetch_daily_for_coords(
                        float(lat), float(lon), weather_delay_sec
                    )

            rows_batch: list[dict] = []
            for _, row in sub.iterrows():
                rows_batch.extend(_daily_rows_for_site(row, coord_to_df))

            write_header = not out.exists()
            _append_batch_csv(out, rows_batch, write_header=write_header)
            pbar.update(len(sub))

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Daily 2025 weather for finale.csv sites")
    p.add_argument("--start", type=int, default=0, help="Start row index (inclusive)")
    p.add_argument("--end", type=int, default=None, help="End row index (exclusive); default=all")
    p.add_argument(
        "--output",
        type=Path,
        default=OUT_CSV,
        help=f"Output CSV path (default: {OUT_CSV})",
    )
    p.add_argument(
        "--geocode-delay",
        type=float,
        default=DEFAULT_GEOCODE_DELAY_SEC,
        help=f"Seconds between geocode requests (default {DEFAULT_GEOCODE_DELAY_SEC})",
    )
    p.add_argument(
        "--weather-delay",
        type=float,
        default=DEFAULT_WEATHER_DELAY_SEC,
        help=f"Seconds between Open-Meteo archive requests (default {DEFAULT_WEATHER_DELAY_SEC})",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY_SITES,
        help=f"Append output CSV after this many sites (default {DEFAULT_SAVE_EVERY_SITES}; env WEATHER_2025_SAVE_EVERY_SITES)",
    )
    args = p.parse_args()
    path = run(
        start_row=args.start,
        end_row=args.end,
        geocode_delay_sec=args.geocode_delay,
        weather_delay_sec=args.weather_delay,
        output_path=args.output,
        save_every_sites=args.save_every,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
