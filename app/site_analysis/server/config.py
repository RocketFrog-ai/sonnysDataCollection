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
    "dirt-trigger-days": ("Dirt Creation Days", "Rainy Days"),
    "dirt-deposit-severity": ("Dirt Deposit Severity", "Total Annual Snowfall"),
    "comfortable-washing-days": ("Comfortable Washing Days", "Days with 60–80°F temperatures"),
    "shutdown-risk-days": ("Shutdown Risk Days", "Days Below Freezing (< 32°F)"),
}


# Competition (nearby car washes within 4 miles)
COMPETITION_RADIUS_MILES = 4.0

COMPETITION_METRIC_DISPLAY: dict = {
    "same-format-count": ("Nearby Same-format Car Washes", "Within 4 miles"),
    "distance-to-nearest": ("Distance to Nearest Same-format Car Wash", "miles"),
    "nearest-google-rating": ("Nearest Competitor Google Rating", "stars"),
    "nearest-review-count": ("Nearest Competitor Review Count", "reviews"),
    "nearest-brand-strength": ("Nearest Same-format Carwash Brand Strength", "Rating and reviews"),
}


# -----------------------------------------------------------------------------
# Retail proximity: anchor retailers within 1 and 3 miles
# -----------------------------------------------------------------------------

RETAIL_RADIUS_NEAR_MILES = 1.0
RETAIL_RADIUS_FAR_MILES = 3.0

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
