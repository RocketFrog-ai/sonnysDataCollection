import os
import requests
import traceback
import pandas as pd
from urllib.parse import quote
from math import radians, sin, cos, sqrt, atan2


RFW_HOME = os.getenviron("RFW_HOME","")
REDIS_HOST = os.getenviron("REDIS_HOST", "")
REDIS_PORT = os.getenviron("REDIS_PORT", "")
REDIS_DB = os.getenviron("REDIS_DB", "")
REDIS_PASSWORD = os.getenviron("REDIS_PASSWORD", "")
CELERY_BROKER_URL = os.getenviron("CELERY_BROKER_URL", "")
CELERY_RESULT_BACKEND = os.getenviron("CELERY_RESULT_BACKEND", "")
FAST_API_HOST = os.getenviron("FAST_API_HOST", "")
FAST_API_PORT = os.getenviron("FAST_API_PORT", "")
GEMINI_API_KEY = os.getenviron("GEMINI_API_KEY","")
GOOGLE_MAPS_API_KEY = os.getenviron("GOOGLE_MAPS_API_KEY","")
AZURE_OPENAI_ENDPOINT = os.getenviron("AZURE_OPENAI_ENDPOINT","")
AZURE_OPENAI_API_KEY = os.getenviron("AZURE_OPENAI_API_KEY","")
AZURE_OPENAI_API_VERSION = os.getenviron("AZURE_OPENAI_API_VERSION","")
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenviron("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME","")
TOMTOM_GEOCODE_API_URL = os.getenviron("TOMTOM_GEOCODE_API_URL","")
TOMTOM_API_KEY = os.getenviron("TOMTOM_API_KEY","")
LOCAL_LLM_URL = os.getenviron("LOCAL_LLM_URL", "")
LOCAL_LLM_API_KEY = os.getenviron("LOCAL_LLM_API_KEY", "")


def get_lat_long(address):
    base_url = TOMTOM_GEOCODE_API_URL
    params = {"key" : TOMTOM_API_KEY}
    encoded_address = quote(address)
    url = f"{base_url}/{encoded_address}.json"
    max_retry=5
    retry=1
    while retry<=max_retry:
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get('results') and len(data['results']) > 0:
                result = data['results'][0]
                position = result['position']
                address_data = result.get('address', {})

                result = {
                    'lat': position['lat'],
                    'lon': position['lon'],
                    'city': address_data.get('municipality', address_data.get('municipalitySubdivision', 'Unknown')),
                    'state': address_data.get('countrySubdivision', 'Unknown'),
                    'formatted_address': address_data.get('freeformAddress', address)
                }
                return result
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(traceback.format_exc())
            print("Retrying...Holiday Lookup...")
            retry+=1


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points in miles."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None
    R = 3958.8

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance