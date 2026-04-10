import os
import requests
import traceback
import pandas as pd
from pathlib import Path
from urllib.parse import quote
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv

# Load .env from project root so LLM URL/API key work regardless of cwd (e.g. running from v3/)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")
load_dotenv()  # still allow override from current cwd

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
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")

# LLM Server — base URL resolves both endpoints.
# Set LLM_BASE_URL (e.g. http://10.110.98.5:8080) and the realtime/batch paths are derived automatically.
# Override individual endpoints with LLM_REALTIME_URL / LLM_BATCH_URL if needed.
_LLM_BASE = os.getenv("LLM_BASE_URL", os.getenv("LOCAL_LLM_URL", "")).rstrip("/")
LLM_REALTIME_URL = os.getenv("LLM_REALTIME_URL", f"{_LLM_BASE}/realtime/completions" if _LLM_BASE else "")
LLM_BATCH_URL    = os.getenv("LLM_BATCH_URL",    f"{_LLM_BASE}/batch/completions"    if _LLM_BASE else "")
# Legacy alias kept for any code still referencing LOCAL_LLM_URL directly
LOCAL_LLM_URL = LLM_REALTIME_URL

# Celery task retry settings (delay in seconds)
TASK_RETRY_DELAY = int(os.getenv("TASK_RETRY_DELAY", "60"))
TASK_MAX_RETRIES = int(os.getenv("TASK_MAX_RETRIES", "3"))

# External service (optional, for tasks)
EXTERNAL_SERVICE_URL = os.getenv("EXTERNAL_SERVICE_URL", "")
EXTERNAL_SERVICE_TIMEOUT = int(os.getenv("EXTERNAL_SERVICE_TIMEOUT", "30"))


def get_lat_long(address):
    """Geocode via TomTom Search v2. Address must be one URL path segment: encode `/` etc. (safe='')."""
    if not address or not str(address).strip():
        return None
    base_url = (TOMTOM_GEOCODE_API_URL or "").rstrip("/")
    if not base_url or not TOMTOM_API_KEY:
        return None
    params = {"key": TOMTOM_API_KEY}
    # quote(..., safe="") so `/` in names like "RACER/ALL AMERICAN CAR WASH" does not split the path
    # (TomTom returns 404 for a malformed multi-segment path).
    encoded_address = quote(str(address).strip(), safe="")
    url = f"{base_url}/{encoded_address}.json"
    max_retry = 5
    retry = 1
    while retry <= max_retry:
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if data.get("results") and len(data["results"]) > 0:
                result = data["results"][0]
                position = result["position"]
                address_data = result.get("address", {})

                result = {
                    "lat": position["lat"],
                    "lon": position["lon"],
                    "city": address_data.get(
                        "municipality", address_data.get("municipalitySubdivision", "Unknown")
                    ),
                    "state": address_data.get("countrySubdivision", "Unknown"),
                    "formatted_address": address_data.get("freeformAddress", address),
                }
                return result
            return None
        except requests.exceptions.HTTPError as e:
            resp = e.response
            code = resp.status_code if resp is not None else None
            # TomTom: empty search results are often 404; bad path (unencoded slash) is also 404.
            if code == 404:
                return None
            if code in (400, 403):
                return None
            print(traceback.format_exc())
            print("Retrying...Geocode...")
            retry += 1
        except requests.exceptions.RequestException:
            print(traceback.format_exc())
            print("Retrying...Geocode...")
            retry += 1
    return None


def resolve_lat_lon(address: str):
    """Geocode to (lat, lon) as floats. Raises ValueError if the address is empty or not found."""
    if not address or not str(address).strip():
        raise ValueError("Address is required")
    geo = get_lat_long(address)
    if not geo:
        raise ValueError("Could not geocode address (no results or API error)")
    lat = geo.get("lat")
    lon = geo.get("lon")
    if lat is None or lon is None:
        raise ValueError("Could not geocode address")
    return float(lat), float(lon)


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