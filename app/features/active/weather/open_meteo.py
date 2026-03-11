"""
Open-Meteo historical weather for site analysis.

Core only: fetch daily data, aggregate to rainy_days, snowy_days, total_snowfall_cm,
days_pleasant_temp, days_below_freezing (+ v3 metrics). USA/state/polygon reference
lives in weather_reference.py.
"""

import pandas as pd
import requests
from datetime import date

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

def fetch_open_meteo_weather_data(latitude, longitude, start_date, end_date, retries=5, backoff_factor=20):
    """Fetch daily weather from Open-Meteo archive for (lat, lon) and date range."""
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
    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL_HISTORICAL_WEATHER, params=params)
            response.raise_for_status()
            data = response.json()
            if "daily" not in data or not data["daily"].get("time"):
                return pd.DataFrame()
            df = pd.DataFrame(data["daily"])
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "status_code", None) == 429:
                import time
                time.sleep(backoff_factor * (2**attempt))
            else:
                return pd.DataFrame()
        except (KeyError, ValueError):
            return pd.DataFrame()
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
