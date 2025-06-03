import requests
import sys
import os

# Replace with your Google Static Maps API key
API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

def get_static_map_image(latitude, longitude, output_filepath):
    """
    Fetches a static map image from Google Static Maps API and saves it.

    Args:
        latitude (float): The latitude of the map center.
        longitude (float): The longitude of the map center.
        output_filepath (str): The full path including filename where the image will be saved.
    """
    # Google Static Maps API endpoint
    base_url = "https://maps.googleapis.com/maps/api/staticmap"

    # Parameters for the request
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": 18,               # Higher zoom = more detail (max ~21)
        "size": "640x640",        # Max allowed size without premium plan
        "maptype": "hybrid",   # Use satellite imagery
        "key": API_KEY
    }

    # Make the GET request
    response = requests.get(base_url, params=params)

    # Save the image if the request was successful
    if response.status_code == 200:
        # Ensure the target directory exists before saving
        # os.makedirs can create intermediate directories and doesn't raise an error if the directory already exists (exist_ok=True)
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_filepath, "wb") as f:
            f.write(response.content)
        # print(f"Satellite image saved as '{output_filepath}'") # Avoid excessive output
        return True
    else:
        print(f"Failed to retrieve image for {output_filepath}. Status code: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mapsStatic.py <latitude> <longitude> <output_filename>")
        sys.exit(1)

    try:
        latitude = float(sys.argv[1])
        longitude = float(sys.argv[2])
        output_filename_arg = sys.argv[3]
    except ValueError:
        print("Invalid latitude or longitude provided. Please provide numeric values.")
        sys.exit(1)

    # Determine the directory where the script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Define the target output directory relative to the script's location
    TARGET_SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "satellite_images")

    # Extract just the filename part, in case the user accidentally provides a path
    base_output_filename = os.path.basename(output_filename_arg)

    # Construct the full path for the output image in the target directory
    final_output_filepath = os.path.join(TARGET_SAVE_DIR, base_output_filename)

    get_static_map_image(latitude, longitude, final_output_filepath)
