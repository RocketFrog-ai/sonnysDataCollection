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
