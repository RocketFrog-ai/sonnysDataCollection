

import logging
import random
import time
from typing import Optional

import pandas as pd
import requests
from datetime import date

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

START_YEAR = 2015
END_YEAR = 2024
START_DATE_STR = f"{START_YEAR}-01-01"
END_DATE_STR = f"{END_YEAR}-12-31"

RAIN_DAY_THRESHOLD_MM = 1.0
SNOW_DAY_THRESHOLD_CM = 0.1
FREEZING_THRESHOLD_C = 0.0
PLEASANT_TEMP_MIN_C = 15.0
PLEASANT_TEMP_MAX_C = 25.0

BASE_URL_HISTORICAL_WEATHER = "https://archive-api.open-meteo.com/v1/archive"

_DAILY_VARIABLES = [
    "precipitation_sum",
    "snowfall_sum",
    "temperature_2m_min",
    "temperature_2m_max",
    "sunshine_duration",
    "windspeed_10m_max",
]


def _wind_speed_column(df):
    if "wind_speed_10m_max" in df.columns:
        return df["wind_speed_10m_max"]
    return df["windspeed_10m_max"]


# -----------------------------------------------------------------------------
# Fetch
# -----------------------------------------------------------------------------

# Rate limit: exponential backoff (base seconds * 2^attempt), max wait cap, jitter
OPEN_METEO_BACKOFF_BASE_SEC = 5
OPEN_METEO_BACKOFF_MAX_SEC = 120
OPEN_METEO_RETRIES = 5
OPEN_METEO_TIMEOUT_SEC = 90


def _rate_limit_delay(attempt: int, retry_after_header: Optional[int]) -> float:
    """Compute delay for rate limit: use Retry-After if provided, else exponential backoff with jitter."""
    if retry_after_header is not None and retry_after_header > 0:
        wait = min(retry_after_header, OPEN_METEO_BACKOFF_MAX_SEC)
    else:
        wait = min(OPEN_METEO_BACKOFF_BASE_SEC * (2 ** attempt), OPEN_METEO_BACKOFF_MAX_SEC)
        # Add jitter (±25%) to avoid thundering herd when multiple workers retry
        jitter = wait * 0.25 * (2 * random.random() - 1)
        wait = max(1, wait + jitter)
    return wait


def fetch_open_meteo_weather_data(
    latitude,
    longitude,
    start_date,
    end_date,
    retries=OPEN_METEO_RETRIES,
    backoff_base_sec=OPEN_METEO_BACKOFF_BASE_SEC,
):
    """Fetch daily weather from Open-Meteo archive for (lat, lon) and date range.
    On 429 (rate limit), waits with exponential backoff (or Retry-After if present) and retries.
    """
    start_date = str(start_date).strip()
    end_date = str(end_date).strip()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(_DAILY_VARIABLES),
        "timezone": "UTC",
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(
                BASE_URL_HISTORICAL_WEATHER, params=params, timeout=OPEN_METEO_TIMEOUT_SEC
            )
            if response.status_code == 429:
                # Rate limited: back off then retry
                retry_after = None
                try:
                    ra = response.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = int(ra)
                except (TypeError, ValueError):
                    pass
                wait = _rate_limit_delay(attempt, retry_after)
                logger.warning(
                    "Open-Meteo rate limit (429), attempt %s/%s — waiting %.1fs before retry",
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            if "daily" not in data or not data["daily"].get("time"):
                return pd.DataFrame()
            df = pd.DataFrame(data["daily"])
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            last_error = e
            resp = getattr(e, "response", None)
            status = getattr(resp, "status_code", None) if resp else None
            if status == 429:
                retry_after = None
                if resp and resp.headers:
                    try:
                        ra = resp.headers.get("Retry-After")
                        if ra is not None:
                            retry_after = int(ra)
                    except (TypeError, ValueError):
                        pass
                wait = _rate_limit_delay(attempt, retry_after)
                logger.warning(
                    "Open-Meteo rate limit (429), attempt %s/%s — waiting %.1fs before retry",
                    attempt + 1,
                    retries,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.warning("Open-Meteo request failed (attempt %s/%s): %s", attempt + 1, retries, e)
                return pd.DataFrame()
        except (KeyError, ValueError) as e:
            logger.warning("Open-Meteo response parse error: %s", e)
            return pd.DataFrame()
    if last_error:
        logger.warning("Open-Meteo failed after %s attempts: %s", retries, last_error)
    return pd.DataFrame()


def get_default_weather_range():
    """Return (start_date, end_date) ISO strings for default climate range (prior full year)."""
    today = date.today()
    end_year = today.year if today.month == 12 else today.year - 1
    end_date = date(end_year, 12, 31)
    start_date = date(end_year - 1, 12, 31)
    return start_date.isoformat(), end_date.isoformat()


def get_climate_data_for_range(latitude, longitude, start_date_str, end_date_str):
    """
    Aggregate daily weather over a date range into metrics used by v3 and UI.

    Returns: rainy_days, snowy_days, total_snowfall_cm, days_pleasant_temp, days_below_freezing,
    total_precipitation_mm, total_sunshine_hours, avg_daily_max_windspeed_ms.
    """
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()
    weather_df = fetch_open_meteo_weather_data(latitude, longitude, start_date_str, end_date_str)
    if weather_df.empty:
        return None

    total_precipitation = weather_df["precipitation_sum"].sum(skipna=True)
    rainy_days = (weather_df["precipitation_sum"] > RAIN_DAY_THRESHOLD_MM).sum()
    total_snowfall = weather_df["snowfall_sum"].sum(skipna=True)
    snowy_days = (weather_df["snowfall_sum"] > SNOW_DAY_THRESHOLD_CM).sum()
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

    return {
        "total_precipitation_mm": float(total_precipitation),
        "rainy_days": int(rainy_days),
        "snowy_days": int(snowy_days),
        "total_snowfall_cm": float(total_snowfall),
        "days_below_freezing": int(days_below_freezing),
        "total_sunshine_hours": float(total_sunshine_hours),
        "days_pleasant_temp": int(days_pleasant_temp),
        "avg_daily_max_windspeed_ms": float(avg_daily_max_windspeed_ms),
    }


def get_climate_data(latitude, longitude, start_year="2024", end_year="2025"):
    """
    Multi-year climatological averages for (lat, lon).
    Default 2024–2025 for backward compatibility with process_data.
    """
    start_str = f"{start_year}-01-01"
    end_str = f"{end_year}-12-31"
    weather_df = fetch_open_meteo_weather_data(latitude, longitude, start_str, end_str)
    if weather_df.empty:
        return None

    annual_metrics_list = []
    for year_val in range(int(start_year), int(end_year) + 1):
        year_df = weather_df[weather_df.index.year == year_val]
        if year_df.empty:
            annual_metrics_list.append({"year": year_val})
            continue

        total_precipitation = year_df["precipitation_sum"].sum(skipna=True)
        rainy_days = (year_df["precipitation_sum"] > RAIN_DAY_THRESHOLD_MM).sum()
        total_snowfall = year_df["snowfall_sum"].sum(skipna=True)
        snowy_days = (year_df["snowfall_sum"] > SNOW_DAY_THRESHOLD_CM).sum()
        days_below_freezing = (year_df["temperature_2m_min"] < FREEZING_THRESHOLD_C).sum()
        total_sunshine_hours = year_df["sunshine_duration"].sum(skipna=True) / 3600.0
        year_df_copy = year_df.copy()
        year_df_copy.loc[:, "temp_avg_approx"] = (
            year_df_copy["temperature_2m_min"] + year_df_copy["temperature_2m_max"]
        ) / 2
        days_pleasant_temp = (
            (year_df_copy["temp_avg_approx"] >= PLEASANT_TEMP_MIN_C)
            & (year_df_copy["temp_avg_approx"] <= PLEASANT_TEMP_MAX_C)
        ).sum()
        wind_col = _wind_speed_column(year_df)
        avg_daily_max_windspeed_ms = wind_col.mean(skipna=True)

        annual_metrics_list.append({
            "year": year_val,
            "total_precipitation_mm": total_precipitation,
            "rainy_days": rainy_days,
            "total_snowfall_cm": total_snowfall,
            "snowy_days": snowy_days,
            "days_below_freezing": days_below_freezing,
            "total_sunshine_hours": total_sunshine_hours,
            "days_pleasant_temp": days_pleasant_temp,
            "avg_daily_max_windspeed_ms": avg_daily_max_windspeed_ms,
        })

    if not annual_metrics_list:
        return None
    annual_metrics_df = pd.DataFrame(annual_metrics_list)
    if annual_metrics_df.empty or "year" not in annual_metrics_df.columns:
        return None
    if annual_metrics_df.drop(columns=["year"], errors="ignore").isnull().all().all():
        return None
    climatological_averages = annual_metrics_df.drop(columns=["year"]).mean(skipna=True)
    return climatological_averages.to_dict()
