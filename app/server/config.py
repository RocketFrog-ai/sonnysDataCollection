"""
Server-level config: Redis/Celery, route constants, and API configs (e.g. weather metrics).
"""

from typing import Literal, Optional, Tuple

from app.utils import common as calib


# -----------------------------------------------------------------------------
# Redis & Celery
# -----------------------------------------------------------------------------

REDIS_HOST = calib.REDIS_HOST
REDIS_PORT = calib.REDIS_PORT
REDIS_DB = calib.REDIS_DB
REDIS_PASSWORD = calib.REDIS_PASSWORD
CELERY_BROKER_URL = calib.CELERY_BROKER_URL
CELERY_RESULT_BACKEND = calib.CELERY_RESULT_BACKEND


# -----------------------------------------------------------------------------
# Profiling dimensions
# -----------------------------------------------------------------------------

DIMENSIONS = ("Weather", "Gas", "Retail Proximity", "Competition")


# -----------------------------------------------------------------------------
# Weather data API — metric keys and mapping to climate fields
# -----------------------------------------------------------------------------
# Used by POST /weather/data/{metric_key}.
# Each entry: (climate_key, unit, description).
# -----------------------------------------------------------------------------

WeatherMetricKey = Literal[
    "dirt-trigger-days",
    "dirt-deposit-severity",
    "comfortable-washing-days",
    "shutdown-risk-days",
]

WEATHER_METRIC_CONFIG: dict = {
    "dirt-trigger-days": ("rainy_days", "days/year", "Rainy Days"),
    "dirt-deposit-severity": ("total_snowfall_cm", "cm snowfall/year", "Total Annual Snowfall"),
    "comfortable-washing-days": ("days_pleasant_temp", "days/year", "Comfortable Washing Days"),
    "shutdown-risk-days": ("days_below_freezing", "days/year", "Days Below Freezing (< 32°F)"),
}

# Display label for API responses: (display_name, subtitle)
WEATHER_METRIC_DISPLAY: dict = {
    "dirt-trigger-days": ("Dirt Trigger Days Window", "Rainy Days"),
    "dirt-deposit-severity": ("Dirt Deposit Severity", "Total Annual Snowfall"),
    "comfortable-washing-days": ("Comfortable Washing Days", "Days with 60–80°F temperatures"),
    "shutdown-risk-days": ("Shutdown Risk Days", "Days Below Freezing (< 32°F)"),
}

# Map weather metric_key (API) → v3 feature_analysis key (quantile_result)
WEATHER_METRIC_TO_V3_FEATURE: dict = {
    "dirt-trigger-days": "weather_rainy_days",
    "dirt-deposit-severity": "weather_total_snowfall_cm",
    "comfortable-washing-days": "weather_days_pleasant_temp",
    "shutdown-risk-days": "weather_days_below_freezing",
}


# Competition (nearby car washes within 4 miles): API metric keys and v3 mapping
COMPETITION_RADIUS_MILES = 4.0

COMPETITION_METRIC_TO_V3_FEATURE: dict = {
    "same-format-count": "competitors_count_4miles",
    "distance-to-nearest": "competitor_1_distance_miles",
    "nearest-google-rating": "competitor_1_google_rating",
    "nearest-review-count": "competitor_1_rating_count",
    "nearest-brand-strength": "competition_quality",
}
COMPETITION_METRIC_DISPLAY: dict = {
    "same-format-count": ("Nearby Same-format Car Washes", "Within 4 miles"),
    "distance-to-nearest": ("Distance to Nearest Same-format Car Wash", "miles"),
    "nearest-google-rating": ("Nearest Competitor Google Rating", "stars"),
    "nearest-review-count": ("Nearest Competitor Review Count", "reviews"),
    "nearest-brand-strength": ("Nearest Same-format Carwash Brand Strength", "Rating and reviews"),
}


def nearest_brand_strength_from_quantile(
    category: Optional[str] = None,
    wash_q: Optional[float] = None,
) -> str:
    """
    Derive High/Medium/Low from v3 predictor quartile/category for nearest competitor
    (competition_quality = rating × log(review_count+1)). Uses v3 output, not raw rating/review thresholds.
    """
    if category:
        m = {"Strong": "High", "Good": "Medium", "Fair": "Medium", "Poor": "Low"}
        out = m.get(category)
        if out:
            return out
    if wash_q is not None and 1 <= wash_q <= 4:
        return {4: "High", 3: "Medium", 2: "Medium", 1: "Low"}.get(int(wash_q), "Unknown")
    return "Unknown"


# -----------------------------------------------------------------------------
# Retail proximity: anchor retailers within 1 and 3 miles
# -----------------------------------------------------------------------------

RETAIL_RADIUS_NEAR_MILES = 1.0
RETAIL_RADIUS_FAR_MILES = 3.0

# Map retail metric_key (API) → v3 feature_analysis key
RETAIL_METRIC_TO_V3_FEATURE: dict = {
    "costco-distance": "costco_enc",
    "walmart-distance": "distance_nearest_walmart(5 mile)",
    "target-distance": "distance_nearest_target (5 mile)",
    "grocery-count": "other_grocery_count_1mile",
    "food-joint-count": "count_food_joints_0_5miles (0.5 mile)",
}

# For retail_score: percentiles from these v3 keys
RETAIL_SCORE_V3_KEYS = [
    "costco_enc",
    "distance_nearest_walmart(5 mile)",
    "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile",
]

# Anchor type by keyword in name (lower-case)
ANCHOR_TYPE_BY_KEYWORD: dict = {
    "costco": "Warehouse Club",
    "sam's club": "Warehouse Club",
    "bj's wholesale": "Warehouse Club",
    "walmart": "Supercenter",
    "target": "Big Box / Discount",
    "meijer": "Big Box",
    "kohl's": "Big Box",
    "kroger": "Grocery Anchor",
    "publix": "Grocery Anchor",
    "safeway": "Grocery Anchor",
    "whole foods": "Grocery Anchor",
    "aldi": "Grocery Anchor",
    "trader joe's": "Grocery Anchor",
    "h-e-b": "Grocery Anchor",
    "mcdonald's": "Food & Beverage",
    "chick-fil-a": "Food & Beverage",
    "starbucks": "Food & Beverage",
    "dunkin": "Food & Beverage",
    "chipotle": "Food & Beverage",
    "panera": "Food & Beverage",
}

# Category from retailer fetch category field
RETAILER_CATEGORY_TYPE: dict = {
    "Grocery": "Grocery Anchor",
    "Food Joint": "Food & Beverage",
    "Retail": "General Retail",
}


def anchor_type_from_name_or_category(name: Optional[str], category: Optional[str] = None) -> str:
    """Derive anchor retail type from place name (keyword match) or category."""
    if name:
        name_lower = name.lower()
        for keyword, anchor_type in ANCHOR_TYPE_BY_KEYWORD.items():
            if keyword in name_lower:
                return anchor_type
    if category:
        return RETAILER_CATEGORY_TYPE.get(category, category)
    return "Retail"


# -----------------------------------------------------------------------------
# Gas stations
# -----------------------------------------------------------------------------

GAS_RADIUS_NEAR_MILES = 1.0
GAS_RADIUS_FAR_MILES = 3.0

# Map gas metric_key → v3 feature_analysis key
GAS_METRIC_TO_V3_FEATURE: dict = {
    "gas-distance": "nearest_gas_station_distance_miles",
    "gas-rating": "nearest_gas_station_rating",
    "gas-review-count": "nearest_gas_station_rating_count",
}

# For gas_score: percentiles from these v3 keys
GAS_SCORE_V3_KEYS = [
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
]

# High-traffic fuel brands (lower-case keywords)
HIGH_TRAFFIC_GAS_BRANDS = frozenset({
    "shell", "chevron", "exxon", "bp", "sunoco", "arco",
    "costco", "quiktrip", "wawa", "circle k", "marathon",
    "valero", "speedway", "pilot", "loves", "kwik trip",
    "76", "texaco",
})


def is_high_traffic_gas_brand(name: Optional[str]) -> bool:
    """Return True if the gas station name contains a known high-traffic brand."""
    if not name:
        return False
    name_lower = name.lower()
    return any(brand in name_lower for brand in HIGH_TRAFFIC_GAS_BRANDS)


# -----------------------------------------------------------------------------
# Site score: business-logic feature weights
# Maps v3 feature_analysis key → absolute weight (all weights sum to 1.0).
# Formula: site_score = Σ (adjusted_percentile_i × weight_i) / 100  → result 0–100.
#
# Weight derivation:
#   Weather 0.25:  sunshine hrs 0.035 (0.015 avg + 0.02 total sunny days merged),
#                  precipitation 0.02, snowfall 0.09 (0.05 snowfall + 0.04 snowy days merged),
#                  days_below_freezing 0.02, pleasant_temp 0.03, wind_speed 0.005, rainy_days 0.05
#   Competition 0.35: count 0.12, rating 0.07, distance 0.10, reviews 0.06
#   Retail 0.30:  costco 0.08, walmart 0.06, target 0.04, grocery_count 0.07, food_count 0.05
#   Gas 0.10:     distance 0.04, draw_score 0.03, rating_count 0.025, rating 0.005
# -----------------------------------------------------------------------------

SITE_SCORE_WEIGHTS: dict = {
    # Weather
    "weather_total_sunshine_hours":           0.035,
    "weather_total_precipitation_mm":         0.020,
    "weather_total_snowfall_cm":              0.090,
    "weather_days_below_freezing":            0.020,
    "weather_days_pleasant_temp":             0.030,
    "weather_avg_daily_max_windspeed_ms":     0.005,
    "weather_rainy_days":                     0.050,
    # Competition
    "competitors_count_4miles":               0.120,
    "competitor_1_google_rating":             0.070,
    "competitor_1_distance_miles":            0.100,
    "competitor_1_rating_count":              0.060,
    # Retail
    "costco_enc":                             0.080,
    "distance_nearest_walmart(5 mile)":       0.060,
    "distance_nearest_target (5 mile)":       0.040,
    "other_grocery_count_1mile":              0.070,
    "count_food_joints_0_5miles (0.5 mile)":  0.050,
    # Gas
    "nearest_gas_station_distance_miles":     0.040,
    "gas_station_draw":                       0.030,
    "nearest_gas_station_rating_count":       0.025,
    "nearest_gas_station_rating":             0.005,
}

SITE_SCORE_CATEGORY_WEIGHTS: dict = {
    "Weather":     0.25,
    "Competition": 0.35,
    "Retail":      0.30,
    "Gas":         0.10,
}

SITE_SCORE_FEATURE_CATEGORY: dict = {
    "weather_total_sunshine_hours":           "Weather",
    "weather_total_precipitation_mm":         "Weather",
    "weather_total_snowfall_cm":              "Weather",
    "weather_days_below_freezing":            "Weather",
    "weather_days_pleasant_temp":             "Weather",
    "weather_avg_daily_max_windspeed_ms":     "Weather",
    "weather_rainy_days":                     "Weather",
    "competitors_count_4miles":               "Competition",
    "competitor_1_google_rating":             "Competition",
    "competitor_1_distance_miles":            "Competition",
    "competitor_1_rating_count":              "Competition",
    "costco_enc":                             "Retail",
    "distance_nearest_walmart(5 mile)":       "Retail",
    "distance_nearest_target (5 mile)":       "Retail",
    "other_grocery_count_1mile":              "Retail",
    "count_food_joints_0_5miles (0.5 mile)":  "Retail",
    "nearest_gas_station_distance_miles":     "Gas",
    "gas_station_draw":                       "Gas",
    "nearest_gas_station_rating_count":       "Gas",
    "nearest_gas_station_rating":             "Gas",
}


def get_weather_metric_value_from_climate(
    climate: dict,
    metric_key: WeatherMetricKey,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Derive value and unit for one weather metric from a climate dict.
    Returns (value, unit) or (None, unit) if value missing.
    """
    config = WEATHER_METRIC_CONFIG.get(metric_key)
    if not config:
        return None, None
    climate_key, unit, _ = config
    if metric_key == "dirt-trigger-days":
        rainy = climate.get("rainy_days")
        if rainy is None:
            return None, unit
        # When backend adds snowy_days: value = (rainy or 0) + (climate.get("snowy_days") or 0)
        return float(rainy), unit
    val = climate.get(climate_key)
    if val is None:
        return None, unit
    return float(val), unit
