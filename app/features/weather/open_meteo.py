import requests
import pandas as pd
from datetime import datetime, date
import time

# --- Configuration ---
START_YEAR = 2015
END_YEAR = 2024
START_DATE_STR = f"{START_YEAR}-01-01"
END_DATE_STR = f"{END_YEAR}-12-31"

# Thresholds
RAIN_DAY_THRESHOLD_MM = 1.0
SNOW_DAY_THRESHOLD_CM = 0.1
FREEZING_THRESHOLD_C = 0.0
PLEASANT_TEMP_MIN_C = 15.0
PLEASANT_TEMP_MAX_C = 25.0

BASE_URL_HISTORICAL_WEATHER = "https://archive-api.open-meteo.com/v1/archive"

def fetch_open_meteo_weather_data(latitude, longitude, start_date, end_date, retries=5, backoff_factor=20):
    start_date = str(start_date).strip()
    end_date = str(end_date).strip()
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "precipitation_sum",
            "snowfall_sum",
            "temperature_2m_max",
            "temperature_2m_min",
            "sunshine_duration",
            "windspeed_10m_max",
        ],
        "timezone": "UTC"
    }
    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL_HISTORICAL_WEATHER, params=weather_params)
            response.raise_for_status()
            data = response.json()

            if 'daily' not in data or not data['daily'].get('time'):
                print(f"Warning: 'daily' data not found or empty in weather response for {latitude},{longitude}.")
                return pd.DataFrame()

            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                wait = backoff_factor * (2 ** attempt)
                print(f"Rate limited. Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print(f"Error fetching weather data for {latitude},{longitude}: {e}")
                return pd.DataFrame()
        except (KeyError, ValueError) as e:
            print(f"Error processing weather data for {latitude},{longitude}: {e}")
            return pd.DataFrame()
    print(f"Failed to fetch weather data for {latitude},{longitude} after {retries} retries.")
    return pd.DataFrame()

def get_default_weather_range():
    today = date.today()
    end_year = today.year if today.month == 12 else today.year - 1
    end_date = date(end_year, 12, 31)
    start_date = date(end_year - 1, 12, 31)
    return start_date.isoformat(), end_date.isoformat()


def get_climate_data_for_range(latitude, longitude, start_date_str, end_date_str):
    start_date_str = str(start_date_str).strip()
    end_date_str = str(end_date_str).strip()
    weather_df = fetch_open_meteo_weather_data(latitude, longitude, start_date_str, end_date_str)
    if weather_df.empty:
        return None
    total_precipitation = weather_df["precipitation_sum"].sum(skipna=True)
    rainy_days = (weather_df["precipitation_sum"] > RAIN_DAY_THRESHOLD_MM).sum()
    total_snowfall = weather_df["snowfall_sum"].sum(skipna=True)
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
    avg_daily_max_windspeed_ms = weather_df["windspeed_10m_max"].mean(skipna=True)
    return {
        "total_precipitation_mm": float(total_precipitation),
        "rainy_days": int(rainy_days),
        "total_snowfall_cm": float(total_snowfall),
        "days_below_freezing": int(days_below_freezing),
        "total_sunshine_hours": float(total_sunshine_hours),
        "days_pleasant_temp": int(days_pleasant_temp),
        "avg_daily_max_windspeed_ms": float(avg_daily_max_windspeed_ms),
    }


def get_climate_data(latitude, longitude, start_year, end_year):
    START_DATE_STR = f"{start_year}-01-01"
    END_DATE_STR = f"{end_year}-12-31"
    weather_df = fetch_open_meteo_weather_data(latitude, longitude, START_DATE_STR, END_DATE_STR)

    if weather_df.empty:
        return None

    annual_metrics_list = []
    for year_val in range(int(start_year), int(end_year) + 1):
        year_df = weather_df[weather_df.index.year == year_val]

        if year_df.empty:
            annual_metrics_list.append({"year": year_val})
            continue

        total_precipitation = year_df['precipitation_sum'].sum(skipna=True)
        rainy_days = (year_df['precipitation_sum'] > RAIN_DAY_THRESHOLD_MM).sum()
        total_snowfall = year_df['snowfall_sum'].sum(skipna=True)
        snowy_days = (year_df['snowfall_sum'] > SNOW_DAY_THRESHOLD_CM).sum()
        days_below_freezing = (year_df['temperature_2m_min'] < FREEZING_THRESHOLD_C).sum()
        total_sunshine_hours = year_df['sunshine_duration'].sum(skipna=True) / 3600.0

        year_df_copy = year_df.copy()
        year_df_copy.loc[:, 'temp_avg_approx'] = (year_df_copy['temperature_2m_min'] + year_df_copy['temperature_2m_max']) / 2
        days_pleasant_temp = (
            (year_df_copy['temp_avg_approx'] >= PLEASANT_TEMP_MIN_C) &
            (year_df_copy['temp_avg_approx'] <= PLEASANT_TEMP_MAX_C)
        ).sum()

        avg_daily_max_windspeed_ms = year_df['windspeed_10m_max'].mean(skipna=True)

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
    if annual_metrics_df.empty or annual_metrics_df.drop(columns=['year']).isnull().all().all():
        return None

    climatological_averages = annual_metrics_df.drop(columns=['year']).mean(skipna=True)
    return climatological_averages.to_dict()


if __name__ == "__main__":
    LATITUDE = 38.6808632
    LONGITUDE = -87.5201897
    
    print(f"--- Fetching and Processing Open-Meteo Weather Data ---")
    print(f"Location: Lat={LATITUDE}, Lon={LONGITUDE}")
    print(f"Period: {START_DATE_STR} to {END_DATE_STR}\n")

    climate_data = get_climate_data(LATITUDE, LONGITUDE, "2024", "2025")

    if climate_data:
        print("\n--- Open-Meteo Climatological Weather Averages ---")
        for metric, value in climate_data.items():
            if pd.isna(value):
                print(f"{metric}: Not Available (NaN)")
            else:
                print(f"{metric}: {value:.2f}")
    else:
        print("Could not fetch or process weather data.")
