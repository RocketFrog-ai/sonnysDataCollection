"""Feature metadata and `ML_FEATURE_ORDER` for ExtraTrees."""
from __future__ import annotations

from typing import Dict, List, Optional


QUANTILE_LABELS: Dict[int, str] = {
    1: "Q1 (Bottom 25%)",
    2: "Q2 (25–50%)",
    3: "Q3 (50–75%)",
    4: "Q4 (Top 25%)",
}
QUANTILE_TIER_NAMES: Dict[int, str] = {
    1: "Low Performer",
    2: "Below Median",
    3: "Above Median",
    4: "High Performer",
}

# Short labels for API / LLM narratives (same Q1–Q4 tiers as the model).
QUANTILE_UI_CATEGORIES: Dict[int, str] = {
    1: "Poor",
    2: "Fair",
    3: "Good",
    4: "Strong",
}


def get_category_for_quantile(q: Optional[int]) -> Optional[str]:
    if q is None:
        return None
    return QUANTILE_UI_CATEGORIES.get(int(q))


# ── Tier Strategy Presets ────────────────────────────────────────────────────
# Percentile splits (sum must be 100)
TIER_PRESETS: Dict[str, List[int]] = {
    "3-class-standard":      [33, 33, 34],
    "3-class-bottom-heavy":  [40, 40, 20],
    "4-class-standard":      [25, 25, 25, 25],
    "4-class-wide-middle":   [20, 30, 30, 20],
    "4-class-top-heavy":     [10, 20, 30, 40],
    "4-class-ultra-wide-middle": [15, 35, 35, 15],
    "4-class-90pct-custom":   [19, 31, 31, 19],
    "2-class-standard":      [50, 50],
}

# ── Feature directions (derived from actual Spearman correlation sign) ────────
FEATURE_DIRECTIONS: Dict[str, str] = {
    "weather_total_precipitation_mm":       "neutral",   # r=+0.007
    "weather_rainy_days":                   "higher",    # r=+0.085
    "weather_total_snowfall_cm":            "lower",     # r=-0.087
    "weather_days_below_freezing":          "lower",     # r=-0.061
    "weather_total_sunshine_hours":         "neutral",   # r=+0.048
    "weather_days_pleasant_temp":           "higher",    # r=+0.139
    "weather_avg_daily_max_windspeed_ms":   "lower",     # r=-0.059
    "nearest_gas_station_distance_miles":   "neutral",   # r=+0.028
    "nearest_gas_station_rating":           "higher",    # r=+0.112
    "nearest_gas_station_rating_count":     "neutral",   # r=+0.032
    "competitors_count_4miles":             "neutral",   # r=+0.035
    "competitor_1_google_rating":           "neutral",   # r=+0.028
    "competitor_1_distance_miles":          "lower",     # r=-0.083
    "competitor_1_rating_count":            "higher",    # r=+0.120
    "costco_enc":                           "lower",     # lower=closer=better, r=+0.073
    "distance_nearest_walmart(5 mile)":     "lower",     # r=-0.088
    "distance_nearest_target (5 mile)":     "lower",     # r=-0.089
    "other_grocery_count_1mile":            "neutral",   # r=+0.022
    "count_food_joints_0_5miles (0.5 mile)":"neutral",   # r=+0.063
    "age_on_30_sep_25":                     "lower",     # r=-0.335
    "region_enc":                           "neutral",
    "state_enc":                            "neutral",
    "competition_quality":                  "higher",
    "gas_station_draw":                     "higher",
    "retail_proximity":                     "higher",
    "weather_drive_score":                  "higher",
    "tunnel_count":                         "higher",    # r=+0.891
    "effective_capacity":                   "higher",    # r=+0.74
    # ── DISPLAY-ONLY features (computed in analyze output but NOT in ML model) ──
    # carwash_type_encoded is already captured by effective_capacity.
    # Including it directly reduces accuracy by 0.9% due to multicollinearity.
    "carwash_type_encoded":                 "lower",    
}

FEATURE_SIGNAL: Dict[str, float] = {
    "weather_total_precipitation_mm":       0.007,
    "weather_rainy_days":                   0.085,
    "weather_total_snowfall_cm":            0.087,
    "weather_days_below_freezing":          0.061,
    "weather_total_sunshine_hours":         0.048,
    "weather_days_pleasant_temp":           0.139,
    "weather_avg_daily_max_windspeed_ms":   0.059,
    "nearest_gas_station_distance_miles":   0.028,
    "nearest_gas_station_rating":           0.112,
    "nearest_gas_station_rating_count":     0.032,
    "competitors_count_4miles":             0.035,
    "competitor_1_google_rating":           0.028,
    "competitor_1_distance_miles":          0.083,
    "competitor_1_rating_count":            0.120,
    "costco_enc":                           0.073,
    "distance_nearest_walmart(5 mile)":     0.088,
    "distance_nearest_target (5 mile)":     0.089,
    "other_grocery_count_1mile":            0.022,
    "count_food_joints_0_5miles (0.5 mile)":0.063,
    "age_on_30_sep_25":                     0.335,
    "region_enc":                           0.050,
    "state_enc":                            0.060,
    "competition_quality":                  0.130,
    "gas_station_draw":                     0.120,
    "retail_proximity":                     0.095,
    "weather_drive_score":                  0.115,
    "tunnel_count":                         0.891,
    "effective_capacity":                   0.738,
    "carwash_type_encoded":                 0.354,  # DISPLAY-ONLY; not in ML model
}
SIGNAL_THRESHOLD = 0.07

FEATURE_LABELS: Dict[str, str] = {
    "weather_total_precipitation_mm":       "Annual Precipitation (mm)",
    "weather_rainy_days":                   "Rainy Days / Year",
    "weather_total_snowfall_cm":            "Annual Snowfall (cm)",
    "weather_days_below_freezing":          "Days Below Freezing",
    "weather_total_sunshine_hours":         "Annual Sunshine Hours",
    "weather_days_pleasant_temp":           "Pleasant Temp Days",
    "weather_avg_daily_max_windspeed_ms":   "Avg Max Wind Speed (m/s)",
    "nearest_gas_station_distance_miles":   "Nearest Gas Station (miles)",
    "nearest_gas_station_rating":           "Gas Station Rating",
    "nearest_gas_station_rating_count":     "Gas Station Review Count",
    "competitors_count_4miles":             "Competitors within 4 Miles",
    "competitor_1_google_rating":           "Nearest Competitor Rating",
    "competitor_1_distance_miles":          "Nearest Competitor (miles)",
    "competitor_1_rating_count":            "Competitor Review Count",
    "costco_enc":                           "Costco Distance (mi, 99=none)",
    "distance_nearest_walmart(5 mile)":     "Distance to Walmart (mi)",
    "distance_nearest_target (5 mile)":     "Distance to Target (mi)",
    "other_grocery_count_1mile":            "Grocery Stores within 1 Mile",
    "count_food_joints_0_5miles (0.5 mile)":"Food Joints within 0.5 Mile",
    "age_on_30_sep_25":                     "Site Age (years)",
    "region_enc":                           "Region",
    "state_enc":                            "State",
    "competition_quality":                  "Competition Quality Score",
    "gas_station_draw":                     "Gas Station Draw Score",
    "retail_proximity":                     "Retail Proximity Score",
    "weather_drive_score":                  "Weather Drive Score",
    "tunnel_count":                         "Tunnel Count (proxy)",
    "effective_capacity":                   "Effective Capacity (tunnels × is-Express)",
    "carwash_type_encoded":                 "Car Wash Type (1=Express, 2=Flex/Mobile, 3=Hand Wash)",
}

# Features shown in analyze() output but intentionally excluded from the ML model.
# They are already captured by other features (effective_capacity encodes carwash_type).
DISPLAY_ONLY_FEATURES = {"carwash_type_encoded"}

# Canonical feature order for the ML model — must match the v3 benchmark order so that
# ExtraTrees random_state=42 feature-sampling produces the same 63.1% CV accuracy.
# Column order matters for tree-based models with fixed random seeds: the same splits
# are drawn differently when the feature index assignments change.
ML_FEATURE_ORDER: List[str] = [
    "weather_total_precipitation_mm", "weather_rainy_days", "weather_total_snowfall_cm",
    "weather_days_below_freezing", "weather_total_sunshine_hours", "weather_days_pleasant_temp",
    "weather_avg_daily_max_windspeed_ms", "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating", "nearest_gas_station_rating_count",
    "competitors_count_4miles", "competitor_1_google_rating",
    "competitor_1_distance_miles", "competitor_1_rating_count",
    "costco_enc",
    "distance_nearest_walmart(5 mile)", "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile", "count_food_joints_0_5miles (0.5 mile)",
    "age_on_30_sep_25", "region_enc", "state_enc",
    "competition_quality", "gas_station_draw", "retail_proximity", "weather_drive_score",
    "tunnel_count", "effective_capacity",
]
