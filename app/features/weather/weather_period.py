import sys
import requests
from datetime import datetime, timedelta

# --- Configuration ---
BASE_URL_HISTORICAL_WEATHER = "https://archive-api.open-meteo.com/v1/archive"

def fetch_weather_data(latitude, longitude, start_date, end_date):
    """Fetches historical weather data from Open-Meteo."""
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

    try:
        response = requests.get(BASE_URL_HISTORICAL_WEATHER, params=weather_params)
        response.raise_for_status()
        data = response.json()

        if 'daily' not in data or not data['daily'].get('time'):
            print(f"Warning: No daily data found in response.")
            return None

        return data['daily']
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"Error processing weather data: {e}")
        return None

def get_weather_averages(latitude, longitude, last_n_days):
    """
    Fetches weather data for the last n days and calculates verbose averages.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        last_n_days: Number of days to look back from today
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=last_n_days)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"WEATHER ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"Location: Latitude {latitude}°, Longitude {longitude}°")
    print(f"Period: {start_date_str} to {end_date_str} ({last_n_days} days)")
    print(f"{'='*70}\n")

    weather_data = fetch_weather_data(latitude, longitude, start_date_str, end_date_str)

    if weather_data is None:
        print("❌ Could not fetch weather data.")
        return None

    actual_days = len(weather_data['time'])
    print(f"📊 Data Points Analyzed: {actual_days} days\n")

    # Extract data arrays and filter out None values
    temp_max = [x for x in weather_data['temperature_2m_max'] if x is not None]
    temp_min = [x for x in weather_data['temperature_2m_min'] if x is not None]
    precip = [x for x in weather_data['precipitation_sum'] if x is not None]
    snow = [x for x in weather_data['snowfall_sum'] if x is not None]
    sunshine = [x for x in weather_data['sunshine_duration'] if x is not None]
    wind = [x for x in weather_data['windspeed_10m_max'] if x is not None]

    # Calculate averages and statistics
    print(f"{'─'*70}")
    print("🌡️  TEMPERATURE")
    print(f"{'─'*70}")
    avg_max_temp = sum(temp_max) / len(temp_max) if temp_max else 0
    avg_min_temp = sum(temp_min) / len(temp_min) if temp_min else 0
    avg_temp = (avg_max_temp + avg_min_temp) / 2
    max_temp_recorded = max(temp_max) if temp_max else 0
    min_temp_recorded = min(temp_min) if temp_min else 0

    print(f"  Average Daily Maximum:     {avg_max_temp:>8.2f}°C")
    print(f"  Average Daily Minimum:     {avg_min_temp:>8.2f}°C")
    print(f"  Average Temperature:       {avg_temp:>8.2f}°C")
    print(f"  Highest Recorded:          {max_temp_recorded:>8.2f}°C")
    print(f"  Lowest Recorded:           {min_temp_recorded:>8.2f}°C")
    print()

    print(f"{'─'*70}")
    print("💧 PRECIPITATION")
    print(f"{'─'*70}")
    total_precip = sum(precip) if precip else 0
    avg_daily_precip = total_precip / len(precip) if precip else 0
    rainy_days = sum(1 for p in precip if p > 1.0)
    max_daily_precip = max(precip) if precip else 0

    print(f"  Total Precipitation:       {total_precip:>8.2f} mm")
    print(f"  Average Daily:             {avg_daily_precip:>8.2f} mm/day")
    print(f"  Rainy Days (>1mm):         {rainy_days:>8} days ({rainy_days/actual_days*100:.1f}%)")
    print(f"  Maximum Daily:             {max_daily_precip:>8.2f} mm")
    print()

    print(f"{'─'*70}")
    print("❄️  SNOWFALL")
    print(f"{'─'*70}")
    total_snow = sum(snow) if snow else 0
    avg_daily_snow = total_snow / len(snow) if snow else 0
    snowy_days = sum(1 for s in snow if s > 0.1)
    max_daily_snow = max(snow) if snow else 0

    print(f"  Total Snowfall:            {total_snow:>8.2f} cm")
    print(f"  Average Daily:             {avg_daily_snow:>8.2f} cm/day")
    print(f"  Snowy Days (>0.1cm):       {snowy_days:>8} days ({snowy_days/actual_days*100:.1f}%)")
    print(f"  Maximum Daily:             {max_daily_snow:>8.2f} cm")
    print()

    print(f"{'─'*70}")
    print("☀️  SUNSHINE")
    print(f"{'─'*70}")
    total_sunshine_hours = sum(sunshine) / 3600.0 if sunshine else 0
    avg_daily_sunshine = (sum(sunshine) / len(sunshine)) / 3600.0 if sunshine else 0
    max_daily_sunshine = max(sunshine) / 3600.0 if sunshine else 0

    print(f"  Total Sunshine Duration:   {total_sunshine_hours:>8.2f} hours")
    print(f"  Average Daily:             {avg_daily_sunshine:>8.2f} hours/day")
    print(f"  Maximum Daily:             {max_daily_sunshine:>8.2f} hours")
    print()

    print(f"{'─'*70}")
    print("💨 WIND SPEED")
    print(f"{'─'*70}")
    avg_max_wind = sum(wind) / len(wind) if wind else 0
    max_wind_recorded = max(wind) if wind else 0
    min_wind = min(wind) if wind else 0

    print(f"  Average Daily Maximum:     {avg_max_wind:>8.2f} m/s ({avg_max_wind*3.6:.2f} km/h)")
    print(f"  Highest Recorded:          {max_wind_recorded:>8.2f} m/s ({max_wind_recorded*3.6:.2f} km/h)")
    print(f"  Lowest Maximum:            {min_wind:>8.2f} m/s ({min_wind*3.6:.2f} km/h)")
    print()

    print(f"{'─'*70}")
    print("🌡️  TEMPERATURE EXTREMES")
    print(f"{'─'*70}")
    days_below_freezing = sum(1 for t in temp_min if t < 0.0)
    days_above_30 = sum(1 for t in temp_max if t > 30.0)

    temp_avg = [(temp_min[i] + temp_max[i]) / 2 for i in range(len(temp_min))]
    days_pleasant = sum(1 for t in temp_avg if 15.0 <= t <= 25.0)

    print(f"  Days Below Freezing:       {days_below_freezing:>8} days ({days_below_freezing/actual_days*100:.1f}%)")
    print(f"  Days Above 30°C:           {days_above_30:>8} days ({days_above_30/actual_days*100:.1f}%)")
    print(f"  Pleasant Days (15-25°C):   {days_pleasant:>8} days ({days_pleasant/actual_days*100:.1f}%)")
    print()

    print(f"{'='*70}\n")

    return {
        'avg_max_temp': avg_max_temp,
        'avg_min_temp': avg_min_temp,
        'avg_temp': avg_temp,
        'total_precipitation': total_precip,
        'avg_daily_precipitation': avg_daily_precip,
        'rainy_days': rainy_days,
        'total_snowfall': total_snow,
        'snowy_days': snowy_days,
        'total_sunshine_hours': total_sunshine_hours,
        'avg_daily_sunshine': avg_daily_sunshine,
        'avg_max_windspeed': avg_max_wind,
        'days_below_freezing': days_below_freezing,
        'days_pleasant_temp': days_pleasant
    }

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python weather_period.py <latitude> <longitude> <last_n_days>")
        print("Example: python weather_period.py 38.6808632 -87.5201897 30")
        sys.exit(1)

    try:
        latitude = float(sys.argv[1])
        longitude = float(sys.argv[2])
        last_n_days = int(sys.argv[3])

        if last_n_days < 1:
            print("Error: last_n_days must be at least 1")
            sys.exit(1)

        result = get_weather_averages(latitude, longitude, last_n_days)

        if result is None:
            sys.exit(1)

    except ValueError:
        print("Error: Invalid input. Latitude and longitude must be numbers, last_n_days must be an integer.")
        sys.exit(1)
