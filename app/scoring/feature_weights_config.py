"""
Feature Weights & Direction Config — tunable per-feature direction and weights.
Source: P-2.0 Data Points Mapping (Sheet4). Edit here to change which way is
better for a feature or to adjust dimension/overall weights.
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Direction per feature: "higher_is_better" | "lower_is_better" | "moderate_is_best"
# Change a value here to flip or set direction without touching the profiler.
# ---------------------------------------------------------------------------
FEATURE_DIRECTION: Dict[str, str] = {
    # Weather (from CSV)
    "weather_total_precipitation_mm": "moderate_is_best",
    "weather_rainy_days": "higher_is_better",
    "weather_total_snowfall_cm": "higher_is_better",
    "weather_days_below_freezing": "moderate_is_best",
    "weather_total_sunshine_hours": "higher_is_better",
    "weather_days_pleasant_temp": "higher_is_better",
    "weather_avg_daily_max_windspeed_ms": "lower_is_better",
    # Gas
    "nearest_gas_station_distance_miles": "lower_is_better",
    "nearest_gas_station_rating": "higher_is_better",
    "nearest_gas_station_rating_count": "higher_is_better",
    # Competition (CSV: Competitor)
    "competitors_count_4miles": "moderate_is_best",
    "competitor_1_google_rating": "lower_is_better",
    "competitor_1_distance_miles": "higher_is_better",
    "competitor_1_rating_count": "lower_is_better",
    # Retailers
    "distance_nearest_costco(5 mile)": "lower_is_better",
    "distance_nearest_walmart(5 mile)": "lower_is_better",
    "distance_nearest_target (5 mile)": "lower_is_better",
    "other_grocery_count_1mile": "higher_is_better",
    "count_food_joints_0_5miles (0.5 mile)": "higher_is_better",
}

# ---------------------------------------------------------------------------
# Per-feature weights (profiler column names). From CSV "Overall Feature Weightage".
# Used for dimension score = weighted avg within dimension; overall = weighted avg of all.
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS: Dict[str, float] = {
    # Weather (category weight 0.25)
    "weather_total_precipitation_mm": 0.0375,
    "weather_rainy_days": 0.05,
    "weather_total_snowfall_cm": 0.05,
    "weather_days_below_freezing": 0.025,
    "weather_total_sunshine_hours": 0.025,
    "weather_days_pleasant_temp": 0.0375,
    "weather_avg_daily_max_windspeed_ms": 0.025,
    # Nearby Gas Station (0.1)
    "nearest_gas_station_distance_miles": 0.05,
    "nearest_gas_station_rating": 0.02,
    "nearest_gas_station_rating_count": 0.03,
    # Competitor (0.35)
    "competitors_count_4miles": 0.1225,
    "competitor_1_google_rating": 0.0525,
    "competitor_1_distance_miles": 0.105,
    "competitor_1_rating_count": 0.07,
    # Retailers (0.3)
    "distance_nearest_costco(5 mile)": 0.09,
    "distance_nearest_walmart(5 mile)": 0.06,
    "distance_nearest_target (5 mile)": 0.045,
    "other_grocery_count_1mile": 0.06,
    "count_food_joints_0_5miles (0.5 mile)": 0.045,
}

# ---------------------------------------------------------------------------
# Which features belong to each dimension (for dimension_score).
# CSV: Weather, Nearby Gas Station, Competitor, Retailers.
# ---------------------------------------------------------------------------
DIMENSION_FEATURES: Dict[str, List[str]] = {
    "Weather": [
        "weather_total_precipitation_mm",
        "weather_rainy_days",
        "weather_total_snowfall_cm",
        "weather_days_below_freezing",
        "weather_total_sunshine_hours",
        "weather_days_pleasant_temp",
        "weather_avg_daily_max_windspeed_ms",
    ],
    "Gas": [
        "nearest_gas_station_distance_miles",
        "nearest_gas_station_rating",
        "nearest_gas_station_rating_count",
    ],
    "Retail Proximity": [
        "distance_nearest_costco(5 mile)",
        "distance_nearest_walmart(5 mile)",
        "distance_nearest_target (5 mile)",
        "other_grocery_count_1mile",
        "count_food_joints_0_5miles (0.5 mile)",
    ],
    "Competition": [
        "competitors_count_4miles",
        "competitor_1_distance_miles",
        "competitor_1_google_rating",
        "competitor_1_rating_count",
    ],
}

# ---------------------------------------------------------------------------
# Dimension weights for overall score (from CSV "Category Weightage").
# Overall score = weighted average of dimension scores using these weights.
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS_FOR_OVERALL: Optional[Dict[str, float]] = {
    "Weather": 0.25,
    "Gas": 0.1,
    "Competition": 0.35,
    "Retail Proximity": 0.3,
}
