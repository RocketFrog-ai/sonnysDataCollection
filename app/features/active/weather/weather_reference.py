"""
USA / state-capital / polygon weather reference (state averages, bbox, point-vs-state).

Build precomputed JSON with: python -m app.features.active.weather.build_reference
"""

import json
import os
import time

from app.features.active.weather.open_meteo import (
    get_climate_data_for_range,
    get_default_weather_range,
)
from app.features.active.weather.usa_states import (
    get_state_abbr_to_name,
    get_state_bbox,
    get_state_for_point,
    get_usa_state_coordinates,
)

REFERENCE_JSON_DIR = os.path.join(os.path.dirname(__file__), "data")
REFERENCE_JSON_PATH = os.path.join(REFERENCE_JSON_DIR, "weather_reference_usa.json")

_usa_reference_cache = {}
_USA_REFERENCE_CACHE_TTL_SEC = 3600 * 24


def load_usa_reference_from_file(path=None, start_date_str=None, end_date_str=None):
    """Load precomputed national + state_averages + state_bbox_averages from JSON."""
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


def get_usa_national_and_state_climate(start_date_str, end_date_str, use_cache=True, delay_between_states_sec=0.3, prefer_file=True):
    """USA reference: national_average + state_averages (+ state_bbox_averages if from file)."""
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
        "state_bbox_averages": {},
    }
    if use_cache:
        _usa_reference_cache[cache_key] = result
    return result


def bbox_to_grid_points(min_lat, max_lat, min_lon, max_lon, n_per_axis=5):
    """Grid of (lat, lon) points inside a bounding box."""
    if n_per_axis < 1:
        n_per_axis = 1
    lats = [min_lat + (max_lat - min_lat) * i / (n_per_axis - 1) for i in range(n_per_axis)] if n_per_axis > 1 else [min_lat]
    lons = [min_lon + (max_lon - min_lon) * i / (n_per_axis - 1) for i in range(n_per_axis)] if n_per_axis > 1 else [min_lon]
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def get_climate_data_for_polygon(points, start_date_str, end_date_str, delay_between_points_sec=0.2):
    """Area-aggregated weather for a polygon (list of lat,lon). Returns same metrics as get_climate_data_for_range + num_points_used."""
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
        if key in ("rainy_days", "snowy_days", "days_below_freezing", "days_pleasant_temp"):
            out[key] = int(round(mean_val))
        else:
            out[key] = float(mean_val)
    return out


def get_climate_data_for_state_bbox(state_abbr, start_date_str, end_date_str, n_per_axis=5):
    """Area-averaged weather over a state bounding box."""
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
    """Weather at point plus California bbox average and USA reference."""
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
    """Weather at (lat, lon) plus state capital and state bbox reference."""
    if start_date_str is None or end_date_str is None:
        start_date_str, end_date_str = get_default_weather_range()
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()

    point = get_climate_data_for_range(lat, lon, start_date_str, end_date_str)
    state_abbr = get_state_for_point(lat, lon)
    state_name = None
    state_capital = None
    state_bbox = None
    national_average = None

    usa_ref = get_usa_national_and_state_climate(start_date_str, end_date_str, use_cache=True, prefer_file=True)
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


_POINT_VS_REFERENCE_METRICS = [
    "total_precipitation_mm", "rainy_days", "snowy_days", "total_snowfall_cm",
    "days_below_freezing", "total_sunshine_hours", "days_pleasant_temp", "avg_daily_max_windspeed_ms",
]


def print_weather_point_vs_state_reference(lat, lon, start_date_str=None, end_date_str=None):
    """Print point vs state capital vs state bbox comparison."""
    result = get_weather_point_vs_state_reference(lat, lon, start_date_str, end_date_str)
    state_label = f"{result.get('state_name') or 'Unknown'} ({result.get('state_abbr') or '?'})"
    print("=" * 72)
    print("WEATHER: POINT vs STATE CAPITAL vs STATE BBOX")
    print("=" * 72)
    print(f"  Point: ({lat}, {lon})  |  State: {state_label}")
    print(f"  Date range: {result.get('start_date')} to {result.get('end_date')}")
    print()
    print(f"  {'Metric':<36} | {'Point (lat,lon)':>16} | {'State capital':>16} | {'State bbox':>16}")
    print("-" * 72)
    for key in _POINT_VS_REFERENCE_METRICS:
        p, sc, sb = result.get("point") or {}, result.get("state_capital") or {}, result.get("state_bbox") or {}
        pv, scv, sbv = p.get(key), sc.get(key), sb.get(key)
        def _fmt(v):
            if v is None: return "—"
            return f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"  {key:<36} | {_fmt(pv):>16} | {_fmt(scv):>16} | {_fmt(sbv):>16}")
    print("=" * 72)
    return result


def example_how_we_find_weather_for_one_state(state_abbr="IN", start_date="2024-01-01", end_date="2024-12-31"):
    """Example: weather for one state (capital point)."""
    coords = get_usa_state_coordinates()
    match = [c for c in coords if c[0] == state_abbr.upper()]
    if not match:
        print(f"State '{state_abbr}' not found.")
        return None
    abbr, name, lat, lon = match[0]
    climate = get_climate_data_for_range(lat, lon, start_date, end_date)
    if climate:
        print(f"Result for {abbr} ({name}):")
        for k, v in list(climate.items())[:10]:
            print(f"  {k}: {v}")
    return climate


def run_state_weather_methodology_example(start_date="2024-01-01", end_date="2024-12-31", states_subset=None):
    """Methodology example: state-wise and national reference."""
    coords = get_usa_state_coordinates()
    if states_subset:
        coords = [(a, n, la, lo) for a, n, la, lo in coords if a in states_subset]
    if not coords:
        print("No states found.")
        return
    print("=" * 60)
    print("METHODOLOGY EXAMPLE: State-wise and national weather reference")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    state_values = {}
    for abbr, name, lat, lon in coords:
        climate = get_climate_data_for_range(lat, lon, start_date, end_date)
        if climate:
            state_values[abbr] = {"state_name": name, **climate}
        time.sleep(0.3)
    if not state_values:
        print("No data received.")
        return
    first = next(iter(state_values.values()))
    national_average = {}
    for key, val in first.items():
        if key == "state_name":
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            values = [s.get(key) for s in state_values.values() if s.get(key) is not None]
            national_average[key] = float(sum(values) / len(values)) if values else None
    print("national_average sample:", {k: national_average.get(k) for k in list(national_average)[:5]})


if __name__ == "__main__":
    import sys
    if "--state-example" in sys.argv:
        run_state_weather_methodology_example(start_date="2024-01-01", end_date="2024-12-31", states_subset=["AL", "CA", "TX"])
    elif "--state-example-one" in sys.argv:
        idx = sys.argv.index("--state-example-one")
        state = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "IN"
        example_how_we_find_weather_for_one_state(state_abbr=state)
    elif "--compare-point-state" in sys.argv:
        idx = sys.argv.index("--compare-point-state")
        try:
            lat = float(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 34.1808
            lon = float(sys.argv[idx + 2]) if idx + 2 < len(sys.argv) else -118.3090
        except ValueError:
            lat, lon = 34.1808, -118.3090
        start_date, end_date = get_default_weather_range()
        print_weather_point_vs_state_reference(lat, lon, start_date, end_date)
    elif "--address" in sys.argv:
        idx = sys.argv.index("--address")
        address = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "1208-1398 N Griffith Park Dr, Burbank, CA 91506, USA"
        from app.utils.common import get_lat_long
        geo = get_lat_long(address)
        if not geo or geo.get("lat") is None:
            print("Could not geocode:", address)
        else:
            lat, lon = geo["lat"], geo["lon"]
            start_date, end_date = get_default_weather_range()
            result = get_weather_for_point_and_california_region(lat, lon, start_date, end_date, ca_bbox_n=4)
            print("Point:", result.get("point"))
            print("CA bbox:", result.get("california_bbox"))
    else:
        print("Usage: --state-example | --state-example-one [ABBR] | --compare-point-state [lat lon] | --address [addr]")
