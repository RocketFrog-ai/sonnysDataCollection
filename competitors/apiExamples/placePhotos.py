import requests
import os
import re # Added for sanitize_filename

# --- Configuration ---
API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA"  # <--- REPLACE WITH YOUR ACTUAL API KEY
# Ensure this API key has "Places API" enabled in your Google Cloud Console.

# --- API Endpoints ---
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACE_PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"

def get_photo_references_and_name(place_id):
    """Gets photo references and the place name for a given place_id."""
    if not place_id:
        return [], None

    params = {
        "place_id": place_id,
        "fields": "photo,name",  # Request photo information and name
        "key": API_KEY
    }
    response = requests.get(PLACE_DETAILS_URL, params=params)
    response.raise_for_status() # Raise an exception for HTTP errors
    result = response.json()

    place_name = None
    photo_refs = []

    if result["status"] == "OK" and "result" in result:
        place_name = result["result"].get("name", place_id) # Default to place_id if name not found
        if "photos" in result["result"]:
            photo_refs = [photo["photo_reference"] for photo in result["result"]["photos"]]
            print(f"Found {len(photo_refs)} photo references for '{place_name}' (ID: {place_id}).")
        else:
            print(f"No photos found for '{place_name}' (ID: {place_id}).")
    else:
        print(f"Error fetching details for Place ID {place_id}: {result.get('status')}, {result.get('error_message', '')}")

    return photo_refs, place_name

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename or directory name by replacing non-alphanumeric
    characters (including spaces) with underscores. This ensures consistency for path creation.
    Handles None input by returning an empty string.
    """
    if name is None:
        return ""
    # Replace all non-alphanumeric characters (including spaces) with a single underscore
    # and strip leading/trailing underscores.
    return re.sub(r'[^a-zA-Z0-9]+', '_', str(name)).strip('_')

def download_photo(photo_reference, original_name, found_car_wash_name, index, image_base_dir, max_width=800):
    """Downloads a photo given its reference and saves it into a structured subfolder."""
    if not photo_reference:
        return

    params = {
        "photoreference": photo_reference,
        "maxwidth": max_width, # You can also use maxheight
        "key": API_KEY
    }
    response = requests.get(PLACE_PHOTO_URL, params=params, stream=True) # stream=True for image
    response.raise_for_status()

    # Sanitize original_name and found_car_wash_name for folder creation using the consistent function
    safe_original_name = sanitize_filename(original_name)
    safe_found_car_wash_name = sanitize_filename(found_car_wash_name)

    # Create nested directory path using the provided image_base_dir
    nested_dir = os.path.join(image_base_dir, safe_original_name, safe_found_car_wash_name)
    os.makedirs(nested_dir, exist_ok=True)

    filename = os.path.join(nested_dir, f"photo_{index + 1}.jpg") # Assuming JPEG

    try:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Successfully downloaded: {filename}")
    except IOError as e:
        print(f"Error saving photo {filename}: {e}")

if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY":
        print("ERROR: Please replace 'YOUR_API_KEY' with your actual Google Maps API key.")
    else:
        # --- PROVIDE YOUR PLACE IDS HERE ---
        # Example:
        # place_ids_to_process = [
        #     "ChIJN1t_tDeuEmsRUsoyG83frY4", # Sydney Opera House
        #     "ChIJe639pc9f0GGREGEC2YgOZBE", # Eiffel Tower
        #     "ChIJhdq0QaH2j4AR42j004b3C0Y"  # Another example place ID
        # ]
        
        # Or get them from user input:
        place_ids_input = input("Enter Place IDs, separated by commas: ")
        place_ids_to_process = [pid.strip() for pid in place_ids_input.split(',') if pid.strip()]

        if not place_ids_to_process:
            print("No Place IDs provided. Exiting.")
        else:
            print(f"\nProcessing {len(place_ids_to_process)} Place ID(s)...")
            for place_id in place_ids_to_process:
                print(f"\n--- Processing Place ID: {place_id} ---")
                photo_references, place_name = get_photo_references_and_name(place_id)

                # Use place_name for filename if available, otherwise use place_id
                filename_prefix = place_name if place_name else place_id

                if photo_references:
                    print(f"Downloading {len(photo_references)} photos for '{filename_prefix}'...")
                    for i, ref in enumerate(photo_references):
                        # In the __main__ block, we don't have original_name and found_car_wash_name
                        # so we'll use filename_prefix for both for demonstration purposes.
                        # This part will be updated in countCompetitors.py to pass correct values.
                        # Pass a dummy IMAGE_DIR for the __main__ block, as it's not relevant for countCompetitors.py
                        download_photo(ref, filename_prefix, filename_prefix, i, "place_images_temp")
                else:
                    print(f"No photos to download for Place ID: {place_id}")
