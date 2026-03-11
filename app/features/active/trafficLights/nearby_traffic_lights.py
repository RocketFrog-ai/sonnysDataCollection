import json
import requests
from app.utils import common as calib

def get_nearby_traffic_lights(lat, lon, radius=3218.68):
    """
    Fetches nearby traffic lights from the Overpass API and sorts them by distance.

    Args:
        lat (float): Latitude of the query location.
        lon (float): Longitude of the query location.
        radius (int, optional): Search radius in meters. Defaults to 3218.68 (2 miles).

    Returns:
        list: A sorted list of traffic light dictionaries, each including a 'distance_miles' key.
    """
    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:300];
    node["highway"="traffic_signals"](around:{radius},{lat},{lon});
    out body;
    """
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=300)
    if not response.ok or not response.text or not response.text.strip():
        return []
    try:
        data = response.json()
    except json.JSONDecodeError:
        return []

    traffic_lights = data.get('elements', [])

    for light in traffic_lights:
        light_lat = light.get('lat')
        light_lon = light.get('lon')
        if light_lat is not None and light_lon is not None:
            distance = calib.calculate_distance(lat, lon, light_lat, light_lon)
            light['distance_miles'] = distance

    # Filter out lights that couldn't have distance calculated and sort
    sorted_lights = sorted(
        [light for light in traffic_lights if 'distance_miles' in light],
        key=lambda x: x['distance_miles']
    )

    return sorted_lights

def filter_duplicate_locations(sorted_lights, threshold_miles=0.05):
    """
    Filters out duplicate traffic light locations based on a distance threshold.

    Args:
        sorted_lights (list): A list of sorted traffic light dictionaries.
        threshold_miles (float, optional): The distance threshold in miles to consider
                                           locations as duplicates. Defaults to 0.05.

    Returns:
        list: A filtered list of traffic light dictionaries.
    """
    if not sorted_lights:
        return []

    unique_lights = []
    for light in sorted_lights:
        is_duplicate = False
        for unique_light in unique_lights:
            distance = calib.calculate_distance(
                light['lat'], light['lon'],
                unique_light['lat'], unique_light['lon']
            )
            if distance < threshold_miles:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_lights.append(light)

    return unique_lights

if __name__ == "__main__":
    # Query location
    QUERY_LAT = 34.5810125
    QUERY_LON = -92.5740888

    sorted_traffic_lights = get_nearby_traffic_lights(QUERY_LAT, QUERY_LON)
    
    unique_traffic_lights = filter_duplicate_locations(sorted_traffic_lights)

    print(json.dumps(unique_traffic_lights, indent=4))
    # print(json.dumps(sorted_traffic_lights, indent=4))
