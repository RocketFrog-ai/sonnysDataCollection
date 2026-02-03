import requests
import json
import math
import sys
import argparse

# The public Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth (in meters)."""
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c / 1609.34  # Convert to miles

def get_nearest_roads_with_speed(lat, lon, radius_meters):
    """
    Queries the Overpass API for roads with speed limits within a given radius,
    then sorts them by distance.
    """
    
    # 1. Build the Overpass QL query with the provided inputs
    # This query finds all 'ways' (roads) with a 'highway' and 'maxspeed' tag
    # that are within the specified radius of the lat/lon.
    query = f"""
    [out:json][timeout:120];
    (
      way(around:{radius_meters},{lat},{lon})["highway"]["maxspeed"];
    );
    out tags geom;
    """

    print("Querying Overpass API... (this may take a moment)")
    
    # 2. Make the API request
    try:
        response = requests.post(OVERPASS_URL, data=query, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error calling Overpass API: {e}")
        return None
        
    data = response.json()
    
    # 3. Process the results: calculate distance and store relevant info
    roads = data.get('elements', [])
    if not roads:
        print("No roads with a maxspeed tag found in that area.")
        return []
        
    roads_with_distance = []
    center_point = (lat, lon)
    
    for road in roads:
        # Skip if it's not a 'way' or has no geometry
        if road.get('type') != 'way' or not road.get('geometry'):
            continue

        min_dist_to_road = float('inf')
        
        # Find the closest point (node) on the road to our center point
        for point in road['geometry']:
            dist = haversine(center_point[0], center_point[1], point['lat'], point['lon'])
            if dist < min_dist_to_road:
                min_dist_to_road = dist
        
        # Use .get() for safe access to tags that might be missing
        tags = road.get('tags', {})
        roads_with_distance.append({
            'name': tags.get('name', 'Unnamed Road'),
            'maxspeed': tags.get('maxspeed', 'N/A'),
            'id': road.get('id'),
            'distance': min_dist_to_road
        })

    # 4. Sort the list of roads by the calculated distance
    sorted_roads = sorted(roads_with_distance, key=lambda x: x['distance'])
    
    return sorted_roads


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Find the max speed of the nearest roads using OpenStreetMap data."
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the center point.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the center point.")
    parser.add_argument("--radius", type=int, required=True, help="Search radius in meters.")

    args = parser.parse_args()

    # Call the main function with the user's input
    results = get_nearest_roads_with_speed(args.lat, args.lon, args.radius)

    if results is not None:
        print("\n--- Results (sorted by nearest first) ---\n")
        # To only show the closest road segment for duplicate road names, track closest roads
        closest_roads = {}  # road_name: {'id': road_id, 'distance': distance}
        for road in results[:5]:
            road_name = road['name']
            if road_name not in closest_roads:
                # First time seeing this road name
                closest_roads[road_name] = {'id': road['id'], 'distance': road['distance']}
                print(f"Road Name: {road['name']}")
                print(f"Max Speed: {road['maxspeed']}")
                print(f"Distance:  {road['distance']:.1f} miles away")
                print("-" * 20)
            else:
                # We've seen this road name before, check if this segment is closer
                if road['distance'] < closest_roads[road_name]['distance']:
                    closest_roads[road_name] = {'id': road['id'], 'distance': road['distance']}
                    print(f"Road Name: {road['name']}")
                    print(f"Max Speed: {road['maxspeed']}")
                    print(f"Distance:  {road['distance']:.1f} miles away")
                    print("-" * 20)
