import os
import traceback
import requests
from urllib.parse import quote

TOMTOM_GEOCODE_API_URL = os.environ.get("TOMTOM_GEOCODE_API_URL", "")
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "")

def get_lat_long(address):
    base_url = TOMTOM_GEOCODE_API_URL
    params = {"key" : TOMTOM_API_KEY}
    encoded_address = quote(address)
    url = f"{base_url}/{encoded_address}.json"
    max_retry=5
    retry=1
    while retry<=max_retry:
        try:
            response = requests.get(url, params=params, timeout=30)
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