import os
import requests
import traceback
import pandas as pd
from urllib.parse import quote
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
load_dotenv()

RFW_HOME = os.getenv("RFW_HOME","")
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = os.getenv("REDIS_PORT", "")
REDIS_DB = os.getenv("REDIS_DB", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "")
FAST_API_HOST = os.getenv("FAST_API_HOST", "")
FAST_API_PORT = os.getenv("FAST_API_PORT", "8002")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY","")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT","")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY","")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION","")
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME","")
TOMTOM_GEOCODE_API_URL = os.getenv("TOMTOM_GEOCODE_API_URL","")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY","")
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")

# Celery task retry settings (delay in seconds)
TASK_RETRY_DELAY = int(os.getenv("TASK_RETRY_DELAY", "60"))
TASK_MAX_RETRIES = int(os.getenv("TASK_MAX_RETRIES", "3"))

# External service (optional, for tasks)
EXTERNAL_SERVICE_URL = os.getenv("EXTERNAL_SERVICE_URL", "")
EXTERNAL_SERVICE_TIMEOUT = int(os.getenv("EXTERNAL_SERVICE_TIMEOUT", "30"))


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