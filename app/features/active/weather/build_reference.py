import json
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from app.features.active.weather.open_meteo import get_default_weather_range
from app.features.active.weather.weather_reference import (
    get_climate_data_for_state_bbox,
    get_usa_national_and_state_climate,
)
from app.features.active.weather.usa_states import STATE_BOUNDING_BOXES

BBOX_GRID_N = 3


def build_reference(start_date=None, end_date=None, out_path=None):
    if start_date is None or end_date is None:
        start_date, end_date = get_default_weather_range()
    start_date = str(start_date).strip()
    end_date = str(end_date).strip()

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "data", "weather_reference_usa.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Building USA weather reference (state capitals + national + state bbox)...")
    print(f"  Date range: {start_date} to {end_date}")
    print("  State capitals + national average...")
    usa_ref = get_usa_national_and_state_climate(
        start_date, end_date, use_cache=False, delay_between_states_sec=0.25
    )
    state_bbox_averages = {}
    for abbr in sorted(STATE_BOUNDING_BOXES.keys()):
        print(f"  State bbox: {abbr}...")
        bbox_data = get_climate_data_for_state_bbox(
            abbr, start_date, end_date, n_per_axis=BBOX_GRID_N
        )
        if bbox_data:
            state_bbox_averages[abbr] = bbox_data

    payload = {
        "start_date": start_date,
        "end_date": end_date,
        "national_average": usa_ref.get("national_average"),
        "state_averages": usa_ref.get("state_averages"),
        "state_names": usa_ref.get("state_names"),
        "state_bbox_averages": state_bbox_averages,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else None
    end = sys.argv[2] if len(sys.argv) > 2 else None
    build_reference(start, end)
