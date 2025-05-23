import requests
import sys
import time

# Replace 'YOUR_API_KEY_HERE' with your actual Google Maps API key
API_KEY = 'AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA'

# Define the base URL for the Street View Static API
base_url = 'https://maps.googleapis.com/maps/api/streetview'

def get_street_view_image(latitude, longitude, heading, output_filepath):
    """
    Downloads a Street View image for a given location and heading.
    """
    location = f'{latitude},{longitude}'
    size = '400x400'
    pitch = 0  # Optional: tilt up/down
    fov = 90   # Optional: zoom level (75-120 recommended)

    params = {
        'location': location,
        'size': size,
        'heading': heading,
        'pitch': pitch,
        'fov': fov,
        'key': API_KEY
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        with open(output_filepath, 'wb') as f:
            f.write(response.content)
        print(f"✅ Saved: {output_filepath}")
    else:
        print(f"❌ Failed to get image for location {location}, heading {heading}: {response.status_code}")
        print("Response text:", response.text)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python streetViewStatic.py <latitude> <longitude> <heading> <output_filepath>")
        sys.exit(1)

    try:
        latitude = float(sys.argv[1])
        longitude = float(sys.argv[2])
        heading = int(sys.argv[3])
        output_filepath = sys.argv[4]

        get_street_view_image(latitude, longitude, heading, output_filepath)
    except ValueError:
        print("Invalid arguments provided. Latitude and longitude should be floats, heading an integer.")
        sys.exit(1)
