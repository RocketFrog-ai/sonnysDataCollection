import sys
import requests
from collections import defaultdict
from datetime import datetime, timedelta

# --- Configuration ---
BASE_URL_HISTORICAL_WEATHER = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo returns wind speed in km/h (default); we display km/h and m/s.

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

def _compute_stats_for_days(weather_data, day_indices):
    """Compute the same stats (temp, precip, snow, sunshine, wind, extremes) for a subset of days."""
    if not day_indices:
        return None
    n = len(day_indices)
    temp_max = [weather_data['temperature_2m_max'][i] for i in day_indices if weather_data['temperature_2m_max'][i] is not None]
    temp_min = [weather_data['temperature_2m_min'][i] for i in day_indices if weather_data['temperature_2m_min'][i] is not None]
    precip = [weather_data['precipitation_sum'][i] for i in day_indices if weather_data['precipitation_sum'][i] is not None]
    snow = [weather_data['snowfall_sum'][i] for i in day_indices if weather_data['snowfall_sum'][i] is not None]
    sunshine = [weather_data['sunshine_duration'][i] for i in day_indices if weather_data['sunshine_duration'][i] is not None]
    wind = [weather_data['windspeed_10m_max'][i] for i in day_indices if weather_data['windspeed_10m_max'][i] is not None]

    avg_max_temp = sum(temp_max) / len(temp_max) if temp_max else 0
    avg_min_temp = sum(temp_min) / len(temp_min) if temp_min else 0
    avg_temp = (avg_max_temp + avg_min_temp) / 2 if (temp_max and temp_min) else 0
    total_precip = sum(precip) if precip else 0
    rainy_days = sum(1 for p in precip if p > 1.0)
    total_snow = sum(snow) if snow else 0
    snowy_days = sum(1 for s in snow if s > 0.1)
    total_sunshine_hours = sum(sunshine) / 3600.0 if sunshine else 0
    avg_daily_sunshine = (sum(sunshine) / len(sunshine)) / 3600.0 if sunshine else 0
    sunny_days = sum(1 for s in sunshine if s >= 6 * 3600)  # >= 6 h sunshine (WMO-style)
    avg_max_wind = sum(wind) / len(wind) if wind else 0
    days_below_freezing = sum(1 for t in temp_min if t < 0.0)
    days_above_30 = sum(1 for t in temp_max if t > 30.0)
    temp_avg = [(temp_min[i] + temp_max[i]) / 2 for i in range(min(len(temp_min), len(temp_max)))]
    days_pleasant = sum(1 for t in temp_avg if 15.0 <= t <= 25.0)

    return {
        'days': n,
        'sunny_days': sunny_days,
        'avg_max_temp': avg_max_temp,
        'avg_min_temp': avg_min_temp,
        'avg_temp': avg_temp,
        'max_temp': max(temp_max) if temp_max else 0,
        'min_temp': min(temp_min) if temp_min else 0,
        'total_precipitation': total_precip,
        'avg_daily_precipitation': total_precip / n if n else 0,
        'rainy_days': rainy_days,
        'total_snowfall': total_snow,
        'snowy_days': snowy_days,
        'total_sunshine_hours': total_sunshine_hours,
        'avg_daily_sunshine': avg_daily_sunshine,
        'avg_max_windspeed': avg_max_wind,
        'max_wind': max(wind) if wind else 0,
        'days_below_freezing': days_below_freezing,
        'days_above_30': days_above_30,
        'days_pleasant_temp': days_pleasant,
    }


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def _print_monthly_breakdown(weather_data, for_year=None):
    """
    Print the same metrics broken down by month.
    If for_year is set (e.g. 2025), always show all 12 months Jan–Dec for that year;
    months with no data show "No data".
    """
    times = weather_data['time']
    month_to_indices = defaultdict(list)
    for i, t in enumerate(times):
        month_key = t[:7]  # "YYYY-MM"
        month_to_indices[month_key].append(i)

    if for_year is not None:
        # Show every month in the year (Jan..Dec)
        month_keys = [f"{for_year}-{m:02d}" for m in range(1, 13)]
    else:
        month_keys = sorted(month_to_indices.keys())

    print(f"{'='*70}")
    print("📅 MONTHLY BREAKDOWN (same values as annual)")
    print(f"{'='*70}\n")

    for month_key in month_keys:
        indices = month_to_indices.get(month_key, [])
        s = _compute_stats_for_days(weather_data, indices) if indices else None
        _, _, m = month_key.partition("-")
        month_label = f"{month_key} ({MONTH_NAMES[int(m) - 1]})" if m else month_key
        print(f"{'─'*70}")
        if s is None:
            print(f"  📆 {month_label}  — No data")
            print()
            continue
        n = s['days']
        pct = lambda x: (x / n * 100) if n else 0
        print(f"  📆 {month_label}  ({n} days)")
        print(f"{'─'*70}")
        print(f"  🌡️  Temp:     avg {s['avg_temp']:.1f}°C  (max avg {s['avg_max_temp']:.1f}°C / min avg {s['avg_min_temp']:.1f}°C)  range [{s['min_temp']:.1f}, {s['max_temp']:.1f}]°C")
        print(f"  💧 Precip:   total {s['total_precipitation']:.1f} mm  |  rainy days {s['rainy_days']} ({pct(s['rainy_days']):.0f}%)")
        print(f"  ❄️  Snow:     total {s['total_snowfall']:.1f} cm  |  snowy days {s['snowy_days']} ({pct(s['snowy_days']):.0f}%)")
        print(f"  ☀️  Sunshine: total {s['total_sunshine_hours']:.1f} h  |  avg/day {s['avg_daily_sunshine']:.1f} h")
        print(f"  💨 Wind:     avg max {s['avg_max_windspeed']:.1f} km/h ({s['avg_max_windspeed']/3.6:.1f} m/s)  |  peak {s['max_wind']:.1f} km/h ({s['max_wind']/3.6:.1f} m/s)")
        print(f"  🌡️  Extremes: below 0°C {s['days_below_freezing']} ({pct(s['days_below_freezing']):.0f}%)  |  above 30°C {s['days_above_30']}  |  pleasant 15–25°C {s['days_pleasant_temp']} ({pct(s['days_pleasant_temp']):.0f}%)")
        print()
    print(f"{'='*70}\n")


def get_weather_averages(latitude, longitude, last_n_days):
    """
    Fetches weather data for the last n days and calculates verbose averages.
    Prints full-period (annual) summary and monthly breakdown of the same values.

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

    print(f"  Average Daily Maximum:     {avg_max_wind:>8.1f} km/h ({avg_max_wind/3.6:.1f} m/s)")
    print(f"  Highest Recorded:          {max_wind_recorded:>8.1f} km/h ({max_wind_recorded/3.6:.1f} m/s)")
    print(f"  Lowest Maximum:            {min_wind:>8.1f} km/h ({min_wind/3.6:.1f} m/s)")
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

    # Monthly breakdown (same values as annual, per month)
    _print_monthly_breakdown(weather_data)

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


def get_weather_for_year(latitude, longitude, year):
    """
    Fetches weather data for a full calendar year and prints annual summary
    plus every month (Jan–Dec) for that year.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        year: Calendar year (e.g. 2025)
    """
    start_date_str = f"{year}-01-01"
    end_date_str = f"{year}-12-31"

    print(f"\n{'='*70}")
    print(f"WEATHER ANALYSIS REPORT — FULL YEAR {year}")
    print(f"{'='*70}")
    print(f"Location: Latitude {latitude}°, Longitude {longitude}°")
    print(f"Period: {start_date_str} to {end_date_str} (full calendar year)")
    print(f"{'='*70}\n")

    weather_data = fetch_weather_data(latitude, longitude, start_date_str, end_date_str)

    if weather_data is None:
        print("❌ Could not fetch weather data.")
        return None

    actual_days = len(weather_data['time'])
    print(f"📊 Data Points Analyzed: {actual_days} days\n")

    # Reuse same stats logic via day indices for full period
    all_indices = list(range(actual_days))
    s = _compute_stats_for_days(weather_data, all_indices)
    if s is None:
        print("❌ No valid data to summarize.")
        return None

    n = actual_days
    pct = lambda x: (x / n * 100) if n else 0

    print(f"{'─'*70}")
    print("🌡️  TEMPERATURE (full year)")
    print(f"{'─'*70}")
    print(f"  Average Daily Maximum:     {s['avg_max_temp']:>8.2f}°C")
    print(f"  Average Daily Minimum:     {s['avg_min_temp']:>8.2f}°C")
    print(f"  Average Temperature:       {s['avg_temp']:>8.2f}°C")
    print(f"  Highest Recorded:          {s['max_temp']:>8.2f}°C")
    print(f"  Lowest Recorded:           {s['min_temp']:>8.2f}°C")
    print()

    print(f"{'─'*70}")
    print("💧 PRECIPITATION (full year)")
    print(f"{'─'*70}")
    print(f"  Total Precipitation:       {s['total_precipitation']:>8.2f} mm")
    print(f"  Average Daily:             {s['avg_daily_precipitation']:>8.2f} mm/day")
    print(f"  Rainy Days (>1mm):         {s['rainy_days']:>8} days ({pct(s['rainy_days']):.1f}%)")
    print()

    print(f"{'─'*70}")
    print("❄️  SNOWFALL (full year)")
    print(f"{'─'*70}")
    print(f"  Total Snowfall:            {s['total_snowfall']:>8.2f} cm")
    print(f"  Snowy Days (>0.1cm):       {s['snowy_days']:>8} days ({pct(s['snowy_days']):.1f}%)")
    print()

    print(f"{'─'*70}")
    print("☀️  SUNSHINE (full year)")
    print(f"{'─'*70}")
    print(f"  Total Sunshine Duration:   {s['total_sunshine_hours']:>8.2f} hours")
    print(f"  Average Daily:             {s['avg_daily_sunshine']:>8.2f} hours/day")
    print()

    print(f"{'─'*70}")
    print("💨 WIND SPEED (full year)")
    print(f"{'─'*70}")
    print(f"  Average Daily Maximum:     {s['avg_max_windspeed']:>8.1f} km/h ({s['avg_max_windspeed']/3.6:.1f} m/s)")
    print(f"  Highest Recorded:          {s['max_wind']:>8.1f} km/h ({s['max_wind']/3.6:.1f} m/s)")
    print()

    print(f"{'─'*70}")
    print("🌡️  TEMPERATURE EXTREMES (full year)")
    print(f"{'─'*70}")
    print(f"  Days Below Freezing:       {s['days_below_freezing']:>8} days ({pct(s['days_below_freezing']):.1f}%)")
    print(f"  Days Above 30°C:           {s['days_above_30']:>8} days ({pct(s['days_above_30']):.1f}%)")
    print(f"  Pleasant Days (15-25°C):   {s['days_pleasant_temp']:>8} days ({pct(s['days_pleasant_temp']):.1f}%)")
    print()

    print(f"{'='*70}\n")

    # Every month in the year (Jan..Dec)
    _print_monthly_breakdown(weather_data, for_year=year)

    return {
        'avg_max_temp': s['avg_max_temp'],
        'avg_min_temp': s['avg_min_temp'],
        'avg_temp': s['avg_temp'],
        'total_precipitation': s['total_precipitation'],
        'avg_daily_precipitation': s['avg_daily_precipitation'],
        'rainy_days': s['rainy_days'],
        'total_snowfall': s['total_snowfall'],
        'snowy_days': s['snowy_days'],
        'total_sunshine_hours': s['total_sunshine_hours'],
        'avg_daily_sunshine': s['avg_daily_sunshine'],
        'avg_max_windspeed': s['avg_max_windspeed'],
        'days_below_freezing': s['days_below_freezing'],
        'days_pleasant_temp': s['days_pleasant_temp'],
    }


def get_annual_weather_plot_data(latitude: float, longitude: float, year: int):
    """
    Returns annual weather data for a full calendar year with per-month stats,
    suitable for the Monthly Weather Distribution chart (sunny, pleasant, rainy, snow days)
    and all other monthly metrics.

    Returns None if fetch fails; otherwise a dict with:
      - location: { lat, lon }
      - year, start_date, end_date
      - annual: { sunny_days, pleasant_days, rainy_days, snow_days, ... }
      - monthly: list of 12 objects, one per month (Jan–Dec), each with
        month, month_name, days, sunny_days, pleasant_days, rainy_days, snow_days, and full stats.
    """
    start_date_str = f"{year}-01-01"
    end_date_str = f"{year}-12-31"
    weather_data = fetch_weather_data(latitude, longitude, start_date_str, end_date_str)
    if weather_data is None:
        return None

    times = weather_data["time"]
    month_to_indices = defaultdict(list)
    for i, t in enumerate(times):
        month_key = t[:7]
        month_to_indices[month_key].append(i)

    # Annual totals (all days)
    all_indices = list(range(len(times)))
    annual_s = _compute_stats_for_days(weather_data, all_indices)
    if annual_s is None:
        return None

    monthly = []
    for m in range(1, 13):
        month_key = f"{year}-{m:02d}"
        indices = month_to_indices.get(month_key, [])
        s = _compute_stats_for_days(weather_data, indices) if indices else None
        month_name = MONTH_NAMES[m - 1]
        if s is None:
            monthly.append({
                "month": m,
                "month_name": month_name,
                "month_key": month_key,
                "days": 0,
                "sunny_days": 0,
                "pleasant_days": 0,
                "rainy_days": 0,
                "snow_days": 0,
                "avg_temp_c": None,
                "total_precipitation_mm": None,
                "total_snowfall_cm": None,
                "total_sunshine_hours": None,
                "avg_max_wind_kmh": None,
                "days_below_freezing": None,
                "days_above_30c": None,
            })
        else:
            monthly.append({
                "month": m,
                "month_name": month_name,
                "month_key": month_key,
                "days": s["days"],
                "sunny_days": s["sunny_days"],
                "pleasant_days": s["days_pleasant_temp"],
                "rainy_days": s["rainy_days"],
                "snow_days": s["snowy_days"],
                "avg_temp_c": round(s["avg_temp"], 2),
                "total_precipitation_mm": round(s["total_precipitation"], 2),
                "total_snowfall_cm": round(s["total_snowfall"], 2),
                "total_sunshine_hours": round(s["total_sunshine_hours"], 2),
                "avg_max_wind_kmh": round(s["avg_max_windspeed"], 2),
                "days_below_freezing": s["days_below_freezing"],
                "days_above_30c": s["days_above_30"],
                "avg_daily_sunshine_hours": round(s["avg_daily_sunshine"], 2),
                "max_temp_c": round(s["max_temp"], 2),
                "min_temp_c": round(s["min_temp"], 2),
            })

    return {
        "location": {"lat": latitude, "lon": longitude},
        "year": year,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "annual": {
            "days": annual_s["days"],
            "sunny_days": annual_s["sunny_days"],
            "pleasant_days": annual_s["days_pleasant_temp"],
            "rainy_days": annual_s["rainy_days"],
            "snow_days": annual_s["snowy_days"],
            "avg_temp_c": round(annual_s["avg_temp"], 2),
            "total_precipitation_mm": round(annual_s["total_precipitation"], 2),
            "total_snowfall_cm": round(annual_s["total_snowfall"], 2),
            "total_sunshine_hours": round(annual_s["total_sunshine_hours"], 2),
            "avg_max_wind_kmh": round(annual_s["avg_max_windspeed"], 2),
            "days_below_freezing": annual_s["days_below_freezing"],
            "days_above_30c": annual_s["days_above_30"],
        },
        "monthly": monthly,
    }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python weather_period.py <latitude> <longitude> <year>")
        print("  Use a 4-digit year for full calendar year (e.g. 2025 → Jan–Dec 2025).")
        print("Example: python weather_period.py 38.6808632 -87.5201897 2025")
        sys.exit(1)

    try:
        latitude = float(sys.argv[1])
        longitude = float(sys.argv[2])
        arg3 = sys.argv[3]
        year = int(arg3)
        if year < 1900 or year > 2100 or len(arg3) != 4:
            print("Error: year must be a 4-digit year (e.g. 2025)")
            sys.exit(1)

        result = get_weather_for_year(latitude, longitude, year)

        if result is None:
            sys.exit(1)

    except ValueError:
        print("Error: Invalid input. Latitude and longitude must be numbers, third arg must be a year (e.g. 2025).")
        sys.exit(1)
