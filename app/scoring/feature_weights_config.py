"""
Feature Weights Config — tunable weights for dimension scores and overall site score.
Edit this file to change how much each feature (and dimension) contributes.

Weights are relative; they are normalized per dimension and for overall score.
"""

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Per-feature weights (profiler/Proforma column names).
# Used for: (1) dimension score = weighted avg of feature scores in that dimension
#           (2) overall score = weighted avg of all scored features
# Tune these to emphasize or de-emphasize specific features.
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS: Dict[str, float] = {
    # Weather
    "weather_total_precipitation_mm": 1.0,
    "weather_rainy_days": 1.0,
    "weather_total_snowfall_cm": 1.0,
    "weather_days_below_freezing": 1.0,
    "weather_total_sunshine_hours": 1.2,
    "weather_days_pleasant_temp": 1.2,
    "weather_avg_daily_max_windspeed_ms": 1.0,
    # Gas (standalone dimension for gas endpoint)
    "nearest_gas_station_distance_miles": 1.0,
    "nearest_gas_station_rating": 1.2,
    "nearest_gas_station_rating_count": 0.8,
    # Retail Proximity (retailers + gas)
    "distance_nearest_costco(5 mile)": 1.0,
    "distance_nearest_walmart(5 mile)": 1.0,
    "distance_nearest_target (5 mile)": 1.0,
    "other_grocery_count_1mile": 1.0,
    "count_food_joints_0_5miles (0.5 mile)": 1.0,
    # Competition
    "competitors_count_4miles": 1.2,
    "competitor_1_distance_miles": 1.0,
    "competitor_1_google_rating": 0.8,
    "competitor_1_rating_count": 0.8,
}

# ---------------------------------------------------------------------------
# Which profiler features belong to each dimension (for dimension_score).
# Dimension score = weighted average of final_scores of these features only.
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
        "nearest_gas_station_distance_miles",
        "nearest_gas_station_rating",
        "nearest_gas_station_rating_count",
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
# Optional: dimension weights for overall score (if you want overall = weighted avg of dimension scores).
# Set to None to use feature-level weights only (overall = weighted avg of all feature scores).
# ---------------------------------------------------------------------------
DIMENSION_WEIGHTS_FOR_OVERALL: Optional[Dict[str, float]] = None  # e.g. {"Weather": 0.25, "Retail Proximity": 0.25, "Competition": 0.2, "Gas": 0.1}
# When None, overall_score = weighted average of all feature final_scores using FEATURE_WEIGHTS.
