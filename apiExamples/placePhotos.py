import requests
import os

# --- Configuration ---
API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA"  # <--- REPLACE WITH YOUR ACTUAL API KEY
# Ensure this API key has "Places API" enabled in your Google Cloud Console.

# Directory to save downloaded images
IMAGE_DIR = "place_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

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

def download_photo(photo_reference, identifier_for_filename, index, max_width=800):
    """Downloads a photo given its reference and saves it."""
    if not photo_reference:
        return

    params = {
        "photoreference": photo_reference,
        "maxwidth": max_width, # You can also use maxheight
        "key": API_KEY
    }
    response = requests.get(PLACE_PHOTO_URL, params=params, stream=True) # stream=True for image
    response.raise_for_status()

    # Sanitize identifier for filename
    safe_identifier = "".join(c if c.isalnum() else "_" for c in identifier_for_filename)
    filename = os.path.join(IMAGE_DIR, f"{safe_identifier}_photo_{index + 1}.jpg") # Assuming JPEG

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
                    print(f"Downloading up to {len(photo_references)} photos for '{filename_prefix}'...")
                    for i, ref in enumerate(photo_references):
                        # You might want to limit the number of downloads
                        # if i >= 5: # Example: download only the first 5 photos
                        #     print("Reached download limit for this example.")
                        #     break
                        download_photo(ref, filename_prefix, i)
                else:
                    print(f"No photos to download for Place ID: {place_id}")