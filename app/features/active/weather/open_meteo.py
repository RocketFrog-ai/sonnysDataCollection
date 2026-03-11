import json
import os
import requests
import pandas as pd
from datetime import datetime, date
import time

from app.features.active.weather.usa_states import (
    get_usa_state_coordinates,
    get_state_abbr_to_name,
    get_state_bbox,
    get_state_for_point,
)

# --- Configuration ---
START_YEAR = 2015
END_YEAR = 2024
START_DATE_STR = f"{START_YEAR}-01-01"
END_DATE_STR = f"{END_YEAR}-12-31"

# Thresholds
RAIN_DAY_THRESHOLD_MM = 1.0
SNOW_DAY_THRESHOLD_CM = 0.1
FREEZING_THRESHOLD_C = 0.0
PLEASANT_TEMP_MIN_C = 15.0
PLEASANT_TEMP_MAX_C = 25.0

BASE_URL_HISTORICAL_WEATHER = "https://archive-api.open-meteo.com/v1/archive"

# Precomputed USA reference (state capitals + national + state bbox). Build with build_reference.py.
REFERENCE_JSON_DIR = os.path.join(os.path.dirname(__file__), "data")
REFERENCE_JSON_PATH = os.path.join(REFERENCE_JSON_DIR, "weather_reference_usa.json")


def _wind_speed_column(df):
    """Return wind speed column; API may use wind_speed_10m_max or windspeed_10m_max."""
    if "wind_speed_10m_max" in df.columns:
        return df["wind_speed_10m_max"]
    return df["windspeed_10m_max"]


def fetch_open_meteo_weather_data(latitude, longitude, start_date, end_date, retries=5, backoff_factor=20):
    start_date = str(start_date).strip()
    end_date = str(end_date).strip()
    # Daily variables: see https://open-meteo.com/en/docs/historical-weather-api
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join([
            # Current
            "precipitation_sum",
            "snowfall_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "sunshine_duration",
            "windspeed_10m_max",
            # Additional daily variables from Open-Meteo archive API
            "weather_code",
            "rain_sum",
            "precipitation_hours",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "sunrise",
            "sunset",
            "daylight_duration",
            "wind_gusts_10m_max",
            "wind_direction_10m_dominant",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ]),
        "timezone": "UTC"
    }
    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL_HISTORICAL_WEATHER, params=weather_params)
            response.raise_for_status()
            data = response.json()

            if 'daily' not in data or not data['daily'].get('time'):
                print(f"Warning: 'daily' data not found or empty in weather response for {latitude},{longitude}.")
                return pd.DataFrame()

            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                wait = backoff_factor * (2 ** attempt)
                print(f"Rate limited. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Error fetching weather data for {latitude},{longitude}: {e}")
                return pd.DataFrame()
        except (KeyError, ValueError) as e:
            print(f"Error processing weather data for {latitude},{longitude}: {e}")
            return pd.DataFrame()
    print(f"Failed to fetch weather data for {latitude},{longitude} after {retries} retries.")
    return pd.DataFrame()

def get_default_weather_range():
    today = date.today()
    end_year = today.year if today.month == 12 else today.year - 1
    end_date = date(end_year, 12, 31)
    start_date = date(end_year - 1, 12, 31)
    return start_date.isoformat(), end_date.isoformat()


# In-memory cache for USA national/state climate (key: (start_date, end_date))
_usa_reference_cache = {}
_USA_REFERENCE_CACHE_TTL_SEC = 3600 * 24  # 24 hours
_usa_reference_cache_time = None


def load_usa_reference_from_file(path=None, start_date_str=None, end_date_str=None):
    """
    Load precomputed national + state_averages + state_bbox_averages from JSON.
    Build the file once with: python -m app.features.active.weather.build_reference

    If start_date_str and end_date_str are given, only return data if the file's
    date range matches (so API uses correct reference for requested range).
    Returns None if file missing or date range mismatch.
    """
    p = path or REFERENCE_JSON_PATH
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if start_date_str is not None and end_date_str is not None:
        if data.get("start_date") != str(start_date_str).strip() or data.get("end_date") != str(end_date_str).strip():
            return None
    return {
        "national_average": data.get("national_average"),
        "state_averages": data.get("state_averages") or {},
        "state_names": data.get("state_names") or {},
        "state_bbox_averages": data.get("state_bbox_averages") or {},
    }


def example_how_we_find_weather_for_one_state(state_abbr="IN", start_date="2024-01-01", end_date="2024-12-31"):
    """
    Example: how we find weather for a single state.

    Step 1: Look up the state in usa_states → we use the STATE CAPITAL (lat, lon)
           as the single representative point for that state.
    Step 2: Call Open-Meteo with that (lat, lon) and the date range.
    Step 3: Aggregate daily data over the range → one set of metrics per state.

    Run: python -m app.features.active.weather.open_meteo --state-example-one IN
    """
    from app.features.active.weather.usa_states import get_usa_state_coordinates

    coords = get_usa_state_coordinates()
    match = [c for c in coords if c[0] == state_abbr.upper()]
    if not match:
        print(f"State '{state_abbr}' not found. Use a two-letter code (e.g. IN, CA, TX).")
        return None
    abbr, name, lat, lon = match[0]
    print("How we find weather for a state")
    print("=" * 50)
    print(f"State: {name} ({abbr})")
    print(f"Representative point: state CAPITAL coordinates from usa_states.py")
    print(f"  → lat={lat}, lon={lon}")
    print(f"Date range: {start_date} to {end_date}")
    print()
    print("Call: get_climate_data_for_range(lat, lon, start_date, end_date)")
    print("  → Open-Meteo Archive API returns daily data for that point")
    print("  → We sum/aggregate over the range (e.g. total_precipitation_mm, rainy_days)")
    print()
    climate = get_climate_data_for_range(lat, lon, start_date, end_date)
    if climate:
        print(f"Result for {abbr} ({name}):")
        for k, v in list(climate.items())[:10]:
            print(f"  {k}: {v}")
        if len(climate) > 10:
            print(f"  ... and {len(climate) - 10} more metrics")
    return climate


def get_usa_national_and_state_climate(start_date_str, end_date_str, use_cache=True, delay_between_states_sec=0.3, prefer_file=True):
    """
    USA reference: national_average + state_averages (+ state_bbox_averages if from file).

    If prefer_file=True (default), load from precomputed JSON first (no API calls).
    Build the file once with: python -m app.features.active.weather.build_reference

    Otherwise fetch climate for each state capital and compute national average (51 API calls).
    Returns national_average, state_averages, state_names, and state_bbox_averages (from file or {}).
    """
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()
    cache_key = (start_date_str, end_date_str)

    if prefer_file:
        from_file = load_usa_reference_from_file(path=None, start_date_str=start_date_str, end_date_str=end_date_str)
        if from_file is not None:
            if use_cache:
                _usa_reference_cache[cache_key] = from_file
            return from_file

    if use_cache and cache_key in _usa_reference_cache:
        return _usa_reference_cache[cache_key]

    state_averages = {}
    for abbr, name, lat, lon in get_usa_state_coordinates():
        climate = get_climate_data_for_range(lat, lon, start_date_str, end_date_str)
        if climate:
            state_averages[abbr] = {"state_name": name, **climate}
        if delay_between_states_sec > 0:
            time.sleep(delay_between_states_sec)

    if not state_averages:
        return {"national_average": None, "state_averages": {}, "state_names": {}, "state_bbox_averages": {}}

    # Build national average: mean across states for each numeric metric
    state_names = get_state_abbr_to_name()
    first = next(iter(state_averages.values()))
    national_average = {}
    for key, val in first.items():
        if key == "state_name":
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            values = [s.get(key) for s in state_averages.values() if s.get(key) is not None]
            national_average[key] = float(sum(values) / len(values)) if values else None
        else:
            national_average[key] = None

    result = {
        "national_average": national_average,
        "state_averages": state_averages,
        "state_names": state_names,
        "state_bbox_averages": {},  # only populated from precomputed JSON
    }
    if use_cache:
        _usa_reference_cache[cache_key] = result
    return result


def get_climate_data_for_range(latitude, longitude, start_date_str, end_date_str):
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()
    weather_df = fetch_open_meteo_weather_data(latitude, longitude, start_date_str, end_date_str)
    if weather_df.empty:
        return None
    total_precipitation = weather_df["precipitation_sum"].sum(skipna=True)
    rainy_days = (weather_df["precipitation_sum"] > RAIN_DAY_THRESHOLD_MM).sum()
    total_snowfall = weather_df["snowfall_sum"].sum(skipna=True)
    days_below_freezing = (weather_df["temperature_2m_min"] < FREEZING_THRESHOLD_C).sum()
    total_sunshine_hours = weather_df["sunshine_duration"].sum(skipna=True) / 3600.0
    weather_df_copy = weather_df.copy()
    weather_df_copy.loc[:, "temp_avg_approx"] = (
        weather_df_copy["temperature_2m_min"] + weather_df_copy["temperature_2m_max"]
    ) / 2
    days_pleasant_temp = (
        (weather_df_copy["temp_avg_approx"] >= PLEASANT_TEMP_MIN_C)
        & (weather_df_copy["temp_avg_approx"] <= PLEASANT_TEMP_MAX_C)
    ).sum()
    wind_col = _wind_speed_column(weather_df)
    avg_daily_max_windspeed_ms = wind_col.mean(skipna=True)

    out = {
        "total_precipitation_mm": float(total_precipitation),
        "rainy_days": int(rainy_days),
        "total_snowfall_cm": float(total_snowfall),
        "days_below_freezing": int(days_below_freezing),
        "total_sunshine_hours": float(total_sunshine_hours),
        "days_pleasant_temp": int(days_pleasant_temp),
        "avg_daily_max_windspeed_ms": float(avg_daily_max_windspeed_ms),
    }

    if "rain_sum" in weather_df.columns:
        out["total_rain_mm"] = float(weather_df["rain_sum"].sum(skipna=True))
    if "precipitation_hours" in weather_df.columns:
        out["total_precipitation_hours"] = float(weather_df["precipitation_hours"].sum(skipna=True))
    if "daylight_duration" in weather_df.columns:
        out["total_daylight_hours"] = float(weather_df["daylight_duration"].sum(skipna=True) / 3600.0)
    if "wind_gusts_10m_max" in weather_df.columns:
        out["avg_daily_max_wind_gust_ms"] = float(weather_df["wind_gusts_10m_max"].mean(skipna=True))
    if "shortwave_radiation_sum" in weather_df.columns:
        out["total_shortwave_radiation_mj"] = float(weather_df["shortwave_radiation_sum"].sum(skipna=True))
    if "et0_fao_evapotranspiration" in weather_df.columns:
        out["total_et0_evapotranspiration_mm"] = float(weather_df["et0_fao_evapotranspiration"].sum(skipna=True))
    return out


def bbox_to_grid_points(min_lat, max_lat, min_lon, max_lon, n_per_axis=5):
    """
    Generate a grid of (lat, lon) points inside a bounding box.
    Useful to approximate "weather over an area" by sampling then aggregating.

    Args:
        min_lat, max_lat: Latitude bounds (e.g. 32.0, 34.0).
        min_lon, max_lon: Longitude bounds (e.g. -87.0, -85.0).
        n_per_axis: Number of points along each axis (default 5 → 5×5 = 25 points).

    Returns:
        List of (latitude, longitude) tuples inside the box.
    """
    if n_per_axis < 1:
        n_per_axis = 1
    lats = [min_lat + (max_lat - min_lat) * i / (n_per_axis - 1) for i in range(n_per_axis)] if n_per_axis > 1 else [min_lat]
    lons = [min_lon + (max_lon - min_lon) * i / (n_per_axis - 1) for i in range(n_per_axis)] if n_per_axis > 1 else [min_lon]
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def get_climate_data_for_polygon(points, start_date_str, end_date_str, delay_between_points_sec=0.2):
    """
    Get area-aggregated weather for a polygon or any set of (lat, lon) points.

    Open-Meteo is point-based; there is no native polygon API. So we:
    1. Fetch climate for each point (same date range) via get_climate_data_for_range.
    2. Aggregate: for each metric take the mean across all points (and int metrics
       like rainy_days are rounded after averaging).

    Use this for:
    - Polygon vertices: pass list of (lat, lon) around the boundary.
    - Grid over a bbox: use bbox_to_grid_points(min_lat, max_lat, min_lon, max_lon, n).

    Args:
        points: List of (latitude, longitude) tuples, e.g. [(32.3, -86.9), (34.1, -85.1), ...].
        start_date_str, end_date_str: Same as get_climate_data_for_range (ISO date strings).
        delay_between_points_sec: Pause between API calls to reduce rate limiting (default 0.2).

    Returns:
        Dict like get_climate_data_for_range (same metrics) plus "num_points_used".
        Returns None if no point returned valid data.
    """
    if not points:
        return None
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()
    results = []
    for i, (lat, lon) in enumerate(points):
        climate = get_climate_data_for_range(lat, lon, start_date_str, end_date_str)
        if climate:
            results.append(climate)
        if delay_between_points_sec > 0 and i < len(points) - 1:
            time.sleep(delay_between_points_sec)
    if not results:
        return None
    # Aggregate: mean for numeric keys; round for counts (rainy_days, days_below_freezing, etc.)
    first = results[0]
    out = {"num_points_used": len(results)}
    for key, val in first.items():
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        vals = [r.get(key) for r in results if r.get(key) is not None]
        if not vals:
            out[key] = None
            continue
        mean_val = sum(vals) / len(vals)
        if key in ("rainy_days", "days_below_freezing", "days_pleasant_temp"):
            out[key] = int(round(mean_val))
        else:
            out[key] = float(mean_val)
    return out


def get_climate_data_for_state_bbox(state_abbr, start_date_str, end_date_str, n_per_axis=5):
    """
    Area-averaged weather over a state's bounding box (grid of points, then mean).

    Uses STATE_BOUNDING_BOXES in usa_states.py; only states with a defined bbox
    are supported (e.g. CA). Returns same metrics as get_climate_data_for_range
    plus num_points_used and bbox used.
    """
    bbox = get_state_bbox(state_abbr)
    if not bbox:
        return None
    min_lat, max_lat, min_lon, max_lon = bbox
    points = bbox_to_grid_points(min_lat, max_lat, min_lon, max_lon, n_per_axis=n_per_axis)
    climate = get_climate_data_for_polygon(points, start_date_str, end_date_str, delay_between_points_sec=0.2)
    if not climate:
        return None
    climate["bbox"] = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}
    climate["state_abbr"] = state_abbr.upper()
    return climate


def get_weather_for_point_and_california_region(lat, lon, start_date_str, end_date_str, ca_bbox_n=5):
    """
    Weather at a point (e.g. Burbank) plus California bbox average and USA reference.

    Only the point is fetched via API. National, state_averages, and state_bbox_averages
    come from precomputed JSON (build_reference.py) when available.
    """
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()
    point = get_climate_data_for_range(lat, lon, start_date_str, end_date_str)
    usa_ref = get_usa_national_and_state_climate(start_date_str, end_date_str, use_cache=True, prefer_file=True)
    california_state = (usa_ref.get("state_averages") or {}).get("CA")
    national_average = usa_ref.get("national_average")
    california_bbox = (usa_ref.get("state_bbox_averages") or {}).get("CA")
    if california_bbox is None:
        california_bbox = get_climate_data_for_state_bbox("CA", start_date_str, end_date_str, n_per_axis=ca_bbox_n)
    return {
        "point": point,
        "california_bbox": california_bbox,
        "california_state": california_state,
        "national_average": national_average,
    }


def get_weather_point_vs_state_reference(lat, lon, start_date_str=None, end_date_str=None):
    """
    Get weather for a single (lat, lon) and compare with the two state-level references:
    state capital (one point per state) and state bounding-box average (grid over state).

    Resolves state from (lat, lon) via bounding box containment. Uses precomputed reference
    from build_reference (weather_reference_usa.json) when available.

    Returns dict with:
      - point: climate at (lat, lon) from get_climate_data_for_range
      - state_capital: climate at that state's capital (state_averages[state_abbr])
      - state_bbox: area-averaged climate over that state's bbox (state_bbox_averages[state_abbr])
      - state_abbr, state_name: resolved state
      - national_average: USA-wide average (mean of state capitals)
    Any of point / state_capital / state_bbox may be None if unavailable.
    """
    if start_date_str is None or end_date_str is None:
        start_date_str, end_date_str = START_DATE_STR, END_DATE_STR
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()

    point = get_climate_data_for_range(lat, lon, start_date_str, end_date_str)
    state_abbr = get_state_for_point(lat, lon)
    state_name = None
    state_capital = None
    state_bbox = None
    national_average = None

    usa_ref = get_usa_national_and_state_climate(
        start_date_str, end_date_str, use_cache=True, prefer_file=True
    )
    national_average = usa_ref.get("national_average")
    state_averages = usa_ref.get("state_averages") or {}
    state_bbox_averages = usa_ref.get("state_bbox_averages") or {}
    state_names = usa_ref.get("state_names") or {}

    if state_abbr:
        state_name = state_names.get(state_abbr) or state_abbr
        cap_data = state_averages.get(state_abbr)
        if cap_data and isinstance(cap_data, dict):
            state_capital = {k: v for k, v in cap_data.items() if k != "state_name"}
        state_bbox = state_bbox_averages.get(state_abbr)

    return {
        "point": point,
        "state_capital": state_capital,
        "state_bbox": state_bbox,
        "state_abbr": state_abbr,
        "state_name": state_name,
        "national_average": national_average,
        "start_date": start_date_str,
        "end_date": end_date_str,
    }


# Metrics to show in point vs state reference comparison (same keys in point, state_capital, state_bbox)
_POINT_VS_REFERENCE_METRICS = [
    "total_precipitation_mm",
    "rainy_days",
    "total_snowfall_cm",
    "days_below_freezing",
    "total_sunshine_hours",
    "days_pleasant_temp",
    "avg_daily_max_windspeed_ms",
]


def print_weather_point_vs_state_reference(lat, lon, start_date_str=None, end_date_str=None):
    """
    Print a side-by-side comparison: point (lat, lon) vs state capital vs state bbox.
    Shows how the same metrics differ between the three approaches for one location.
    """
    result = get_weather_point_vs_state_reference(lat, lon, start_date_str, end_date_str)
    state_label = f"{result.get('state_name') or 'Unknown'} ({result.get('state_abbr') or '?'})"
    print("=" * 72)
    print("WEATHER: POINT vs STATE CAPITAL vs STATE BBOX (same lat/lon reference)")
    print("=" * 72)
    print(f"  Point: ({lat}, {lon})  |  State: {state_label}")
    print(f"  Date range: {result.get('start_date')} to {result.get('end_date')}")
    print()
    print(f"  {'Metric':<36} | {'Point (lat,lon)':>16} | {'State capital':>16} | {'State bbox':>16}")
    print("-" * 72)
    for key in _POINT_VS_REFERENCE_METRICS:
        p = result.get("point") or {}
        sc = result.get("state_capital") or {}
        sb = result.get("state_bbox") or {}
        pv = p.get(key)
        scv = sc.get(key)
        sbv = sb.get(key)
        def _fmt(v):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.2f}"
            return str(v)
        print(f"  {key:<36} | {_fmt(pv):>16} | {_fmt(scv):>16} | {_fmt(sbv):>16}")
    print("=" * 72)
    print("  Point = weather at this (lat, lon). State capital = weather at state capital only.")
    print("  State bbox = average over a grid of points covering the state's bounding box.")
    return result


def get_climate_data(latitude, longitude, start_year, end_year):
    START_DATE_STR = f"{start_year}-01-01"
    END_DATE_STR = f"{end_year}-12-31"
    weather_df = fetch_open_meteo_weather_data(latitude, longitude, START_DATE_STR, END_DATE_STR)

    if weather_df.empty:
        return None

    annual_metrics_list = []
    for year_val in range(int(start_year), int(end_year) + 1):
        year_df = weather_df[weather_df.index.year == year_val]

        if year_df.empty:
            annual_metrics_list.append({"year": year_val})
            continue

        total_precipitation = year_df['precipitation_sum'].sum(skipna=True)
        rainy_days = (year_df['precipitation_sum'] > RAIN_DAY_THRESHOLD_MM).sum()
        total_snowfall = year_df['snowfall_sum'].sum(skipna=True)
        snowy_days = (year_df['snowfall_sum'] > SNOW_DAY_THRESHOLD_CM).sum()
        days_below_freezing = (year_df['temperature_2m_min'] < FREEZING_THRESHOLD_C).sum()
        total_sunshine_hours = year_df['sunshine_duration'].sum(skipna=True) / 3600.0

        year_df_copy = year_df.copy()
        year_df_copy.loc[:, 'temp_avg_approx'] = (year_df_copy['temperature_2m_min'] + year_df_copy['temperature_2m_max']) / 2
        days_pleasant_temp = (
            (year_df_copy['temp_avg_approx'] >= PLEASANT_TEMP_MIN_C) &
            (year_df_copy['temp_avg_approx'] <= PLEASANT_TEMP_MAX_C)
        ).sum()

        wind_col = _wind_speed_column(year_df)
        avg_daily_max_windspeed_ms = wind_col.mean(skipna=True)

        row = {
            "year": year_val,
            "total_precipitation_mm": total_precipitation,
            "rainy_days": rainy_days,
            "total_snowfall_cm": total_snowfall,
            "snowy_days": snowy_days,
            "days_below_freezing": days_below_freezing,
            "total_sunshine_hours": total_sunshine_hours,
            "days_pleasant_temp": days_pleasant_temp,
            "avg_daily_max_windspeed_ms": avg_daily_max_windspeed_ms,
        }
        if "rain_sum" in year_df.columns:
            row["total_rain_mm"] = year_df["rain_sum"].sum(skipna=True)
        if "precipitation_hours" in year_df.columns:
            row["total_precipitation_hours"] = year_df["precipitation_hours"].sum(skipna=True)
        if "daylight_duration" in year_df.columns:
            row["total_daylight_hours"] = year_df["daylight_duration"].sum(skipna=True) / 3600.0
        if "wind_gusts_10m_max" in year_df.columns:
            row["avg_daily_max_wind_gust_ms"] = year_df["wind_gusts_10m_max"].mean(skipna=True)
        if "shortwave_radiation_sum" in year_df.columns:
            row["total_shortwave_radiation_mj"] = year_df["shortwave_radiation_sum"].sum(skipna=True)
        if "et0_fao_evapotranspiration" in year_df.columns:
            row["total_et0_evapotranspiration_mm"] = year_df["et0_fao_evapotranspiration"].sum(skipna=True)
        annual_metrics_list.append(row)

    if not annual_metrics_list:
        return None

    annual_metrics_df = pd.DataFrame(annual_metrics_list)
    if annual_metrics_df.empty or annual_metrics_df.drop(columns=['year']).isnull().all().all():
        return None

    climatological_averages = annual_metrics_df.drop(columns=['year']).mean(skipna=True)
    return climatological_averages.to_dict()


def run_state_weather_methodology_example(start_date="2024-01-01", end_date="2024-12-31", states_subset=None):
    """
    Methodological example: how state-wise and national reference works.

    - Fetches climate for each state (or a subset) over the same date range.
    - Builds state_averages (one value per state) and national_average (mean across states).
    - Prints a small numeric example so you can see the flow.

    Usage:
        python -c "from app.features.active.weather.open_meteo import run_state_weather_methodology_example; run_state_weather_methodology_example()"
    Or with a subset (faster):
        run_state_weather_methodology_example(states_subset=["AL", "CA", "TX"])
    """
    coords = get_usa_state_coordinates()
    if states_subset:
        coords = [(a, n, la, lo) for a, n, la, lo in coords if a in states_subset]
        if not coords:
            print("states_subset not found in USA_STATES")
            return

    print("=" * 60)
    print("METHODOLOGY EXAMPLE: State-wise and national weather reference")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"States: {[c[0] for c in coords]}")
    print()

    # Step 1: per-state values (same as get_climate_data_for_range per state)
    state_values = {}
    for abbr, name, lat, lon in coords:
        print(f"  Fetching {abbr} ({name}) at ({lat:.2f}, {lon:.2f})...")
        climate = get_climate_data_for_range(lat, lon, start_date, end_date)
        if climate:
            state_values[abbr] = {"state_name": name, **climate}
        time.sleep(0.3)

    if not state_values:
        print("No data received.")
        return

    # Step 2: show one metric per state (e.g. total_precipitation_mm)
    metric = "total_precipitation_mm"
    print()
    print(f"--- Example metric: {metric} (mm) ---")
    vals = []
    for abbr, data in state_values.items():
        v = data.get(metric)
        if v is not None:
            vals.append(v)
            print(f"  {abbr} ({data.get('state_name', '')}): {v:.1f}")
    if vals:
        national_val = sum(vals) / len(vals)
        print(f"  → national_average[{metric}] = mean({vals}) = {national_val:.1f}")
    print()

    # Step 3: full national average (all numeric metrics)
    national_average = {}
    first = next(iter(state_values.values()))
    for key, val in first.items():
        if key == "state_name":
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            values = [s.get(key) for s in state_values.values() if s.get(key) is not None]
            national_average[key] = float(sum(values) / len(values)) if values else None

    print("--- national_average (mean across these states) ---")
    for k, v in list(national_average.items())[:8]:
        print(f"  {k}: {v:.2f}" if v is not None else f"  {k}: N/A")
    print("  ...")
    print()
    print("State-wise and national reference use state capitals and bbox grids.")


if __name__ == "__main__":
    import sys
    if "--state-example" in sys.argv:
        # Run methodology example (subset of 3 states for speed)
        run_state_weather_methodology_example(
            start_date="2024-01-01",
            end_date="2024-12-31",
            states_subset=["AL", "CA", "TX"],
        )
    elif "--state-example-one" in sys.argv:
        # How we find weather for one state (e.g. IN, CA)
        idx = sys.argv.index("--state-example-one")
        state = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "IN"
        example_how_we_find_weather_for_one_state(state_abbr=state)
    elif "--compare-point-state" in sys.argv:
        # Show point vs state capital vs state bbox for a given (lat, lon)
        idx = sys.argv.index("--compare-point-state")
        if idx + 2 < len(sys.argv):
            try:
                lat = float(sys.argv[idx + 1])
                lon = float(sys.argv[idx + 2])
            except ValueError:
                lat, lon = 34.1808, -118.3090  # Burbank default
        else:
            lat, lon = 34.1808, -118.3090  # Burbank default
        start_date, end_date = get_default_weather_range()
        print_weather_point_vs_state_reference(lat, lon, start_date, end_date)
    elif "--address" in sys.argv:
        # Example: point + California bbox + CA state + national (e.g. Burbank address)
        idx = sys.argv.index("--address")
        address = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
        from app.utils.common import get_lat_long
        geo = get_lat_long(address)
        if not geo or geo.get("lat") is None:
            print("Could not geocode address:", address)
        else:
            lat, lon = geo["lat"], geo["lon"]
            start_date, end_date = get_default_weather_range()
            print("Input address:", address)
            print("Geocoded:", lat, lon)
            print("Date range:", start_date, "to", end_date)
            print("\nFetching: point weather + California bbox average + CA state + national...")
            result = get_weather_for_point_and_california_region(lat, lon, start_date, end_date, ca_bbox_n=4)
            print("\n--- Point (your address) ---")
            if result["point"]:
                for k, v in list(result["point"].items())[:8]:
                    print(f"  {k}: {v}")
                print("  ...")
            print("\n--- California bounding box (area average) ---")
            if result["california_bbox"]:
                b = result["california_bbox"].get("bbox", {})
                print(f"  bbox: lat [{b.get('min_lat')}, {b.get('max_lat')}], lon [{b.get('min_lon')}, {b.get('max_lon')}]")
                print(f"  num_points_used:", result["california_bbox"].get("num_points_used"))
                for k, v in list(result["california_bbox"].items())[:6]:
                    if k not in ("bbox", "state_abbr", "num_points_used"):
                        print(f"  {k}: {v}")
            else:
                print("  (not available)")
            print("\n--- California state (Sacramento) ---")
            if result["california_state"]:
                for k, v in list(result["california_state"].items())[:6]:
                    if k != "state_name":
                        print(f"  {k}: {v}")
            print("\n--- National average ---")
            if result["national_average"]:
                for k, v in list(result["national_average"].items())[:6]:
                    print(f"  {k}: {v}")
    else:
        LATITUDE = 38.6808632
        LONGITUDE = -87.5201897

        print(f"--- Fetching and Processing Open-Meteo Weather Data ---")
        print(f"Location: Lat={LATITUDE}, Lon={LONGITUDE}")
        print(f"Period: {START_DATE_STR} to {END_DATE_STR}\n")

        climate_data = get_climate_data(LATITUDE, LONGITUDE, "2024", "2025")

        if climate_data:
            print("\n--- Open-Meteo Climatological Weather Averages ---")
            for metric, value in climate_data.items():
                if pd.isna(value):
                    print(f"{metric}: Not Available (NaN)")
                else:
                    print(f"{metric}: {value:.2f}")
        else:
            print("Could not fetch or process weather data.")
        print("\nRun with --state-example to see state-wise/national methodology example.")
