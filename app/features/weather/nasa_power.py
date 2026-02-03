import requests
import pandas as pd
from datetime import datetime

# --- Configuration (same as before for location and period) ---
LATITUDE = 34.0522  # Example: Los Angeles
LONGITUDE = -118.2437 # Example: Los Angeles
START_YEAR = 2015
END_YEAR = 2024 # Inclusive

# NASA POWER wants dates in YYYYMMDD format
START_DATE_NASA = f"{START_YEAR}0101"
END_DATE_NASA = f"{END_YEAR}1231"

# Thresholds
RAIN_DAY_THRESHOLD_MM = 1.0
FREEZING_THRESHOLD_C = 0.0

# NASA POWER API Endpoint
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# --- Parameters to request from NASA POWER ---
# Find equivalents:
# PRECTOTCORR: Precipitation Corrected (mm/day) - includes rain, snow etc.
# T2M_MIN: Minimum Temperature at 2 Meters (°C)
# T2M_MAX: Maximum Temperature at 2 Meters (°C)
# ALLSKY_SFC_SW_DWN: All Sky Insolation Incident on a Horizontal Surface (kWh/m^2/day) - proxy for sunniness
# (No direct snowfall sum, PM2.5, PM10, Dust)
NASA_PARAMETERS = [
    "PRECTOTCORR",
    "T2M_MIN",
    "T2M_MAX",
    "ALLSKY_SFC_SW_DWN"
]

# --- Helper Function to Fetch NASA POWER Data ---
def fetch_nasa_power_data(latitude, longitude, start_date, end_date, parameters):
    """Fetches data from NASA POWER API and returns a Pandas DataFrame."""
    api_params = {
        "latitude": latitude,
        "longitude": longitude,
        "community": "RE",  # Renewable Energy community often has good meteorological data
        "parameters": ",".join(parameters),
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "user": "anonymous" # For basic use; register for an API key for higher limits
    }
    try:
        response = requests.get(NASA_POWER_API_URL, params=api_params)
        response.raise_for_status()
        data = response.json()

        # Check if data is present
        if "properties" not in data or "parameter" not in data["properties"]:
            print("Warning: 'properties.parameter' not found in NASA POWER response.")
            print(f"Response: {data}")
            return pd.DataFrame()

        # Extract data into a more usable format
        # The structure is data['properties']['parameter']['PARAM_NAME']['YYYYMMDD']
        processed_data = {}
        dates = None
        for param_name, date_values in data["properties"]["parameter"].items():
            if dates is None: # Get dates from the first parameter
                dates = sorted(date_values.keys())
                processed_data['time'] = [datetime.strptime(d, '%Y%m%d') for d in dates]

            # NASA POWER uses -999 for missing values
            param_values = [date_values.get(d, -999) for d in dates]
            processed_data[param_name] = [val if val != -999 else pd.NA for val in param_values]
            
        if not processed_data or 'time' not in processed_data:
             print("Warning: No data could be processed from NASA POWER response.")
             return pd.DataFrame()

        df = pd.DataFrame(processed_data)
        df.set_index('time', inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from NASA POWER: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        print(f"Error processing NASA POWER data: {e}")
        print(f"Response snippet: {str(data)[:500]}") # Print part of the response for debugging
        return pd.DataFrame()

# --- 1. Fetch NASA POWER Data ---
print(f"Fetching NASA POWER data for {START_DATE_NASA} to {END_DATE_NASA}...")
nasa_df = fetch_nasa_power_data(LATITUDE, LONGITUDE, START_DATE_NASA, END_DATE_NASA, NASA_PARAMETERS)

if nasa_df.empty:
    print("Could not fetch or process NASA POWER data. Exiting.")
    exit()

# --- 2. Calculate Annual Metrics from NASA POWER Data ---
nasa_annual_metrics_list = []

for year in range(START_YEAR, END_YEAR + 1):
    year_df = nasa_df[nasa_df.index.year == year]

    if year_df.empty:
        print(f"No NASA POWER data for year {year}. Skipping.")
        continue

    # PRECTOTCORR is daily precipitation
    total_precipitation_nasa = year_df['PRECTOTCORR'].sum(skipna=True)
    rainy_days_nasa = (year_df['PRECTOTCORR'] > RAIN_DAY_THRESHOLD_MM).sum()

    # T2M_MIN is daily min temperature
    days_below_freezing_nasa = (year_df['T2M_MIN'] < FREEZING_THRESHOLD_C).sum()

    # ALLSKY_SFC_SW_DWN is daily insolation (proxy for sunshine)
    total_annual_insolation_kwh_m2 = year_df['ALLSKY_SFC_SW_DWN'].sum(skipna=True)


    nasa_annual_metrics_list.append({
        "year": year,
        "total_precipitation_mm": total_precipitation_nasa,
        "rainy_days": rainy_days_nasa,
        # "total_snowfall_cm": Not directly available,
        # "snowy_days": Not directly available,
        "days_below_freezing": days_below_freezing_nasa,
        "total_annual_insolation_kwh_m2": total_annual_insolation_kwh_m2,
        # "avg_pm10_ug_m3": Not available,
        # "avg_pm2_5_ug_m3": Not available,
        # "avg_dust_ug_m3": Not available
    })

if not nasa_annual_metrics_list:
    print("No annual metrics could be calculated from NASA POWER data. Exiting.")
    exit()

nasa_annual_metrics_df = pd.DataFrame(nasa_annual_metrics_list)

# --- 3. Calculate Climatological Averages from NASA POWER ---
nasa_climatological_averages = nasa_annual_metrics_df.drop(columns=['year']).mean(skipna=True)

print("\n--- NASA POWER Climatological Averages ---")
print(f"Location: Lat={LATITUDE}, Lon={LONGITUDE}")
print(f"Period: {START_YEAR}-{END_YEAR} ({len(nasa_annual_metrics_df)} years of data considered)")
for metric, value in nasa_climatological_averages.items():
    print(f"{metric}: {value:.2f}")

print("\n--- Notes on NASA POWER Data ---")
print("1. NASA POWER does not provide PM2.5, PM10, or specific dust concentrations.")
print("2. 'total_precipitation_mm' from PRECTOTCORR includes all forms of precipitation (rain, snow, etc.).")
print("3. Direct 'snowfall_sum' or 'snowy_days' (as defined by daily fall) are not standard NASA POWER outputs.")
print("4. 'total_annual_insolation_kwh_m2' is used as a proxy for sunniness instead of 'sunshine_duration_hours'.")