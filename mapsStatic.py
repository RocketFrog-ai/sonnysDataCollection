import requests
import sys
import os

# Replace with your Google Static Maps API key
API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mapsStatic.py <latitude> <longitude> <output_filepath>")
        sys.exit(1)

    try:
        latitude = float(sys.argv[1])
        longitude = float(sys.argv[2])
        output_filepath = sys.argv[3]
    except ValueError:
        print("Invalid latitude or longitude provided. Please provide numeric values.")
        sys.exit(1)

    # Google Static Maps API endpoint
    base_url = "https://maps.googleapis.com/maps/api/staticmap"

    # Parameters for the request
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": 20,               # Higher zoom = more detail (max ~21)
        "size": "640x640",        # Max allowed size without premium plan
        "maptype": "hybrid",   # Use satellite imagery
        "key": API_KEY
    }

    # Make the GET request
    response = requests.get(base_url, params=params)

    # Save the image if the request was successful
    if response.status_code == 200:
        # Ensure the directory exists before saving
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_filepath, "wb") as f:
            f.write(response.content)
        # print(f"Satellite image saved as '{output_filepath}'") # Avoid excessive output

    else:
        print(f"Failed to retrieve image for {output_filepath}. Status code: {response.status_code}")
        print(response.text)
