import os
import re
import glob

def sanitize_filename(name):
    """Sanitizes a string to be used as a filename or directory name by replacing non-alphanumeric
    characters (including spaces) with underscores. This ensures consistency for path creation.
    Handles None input by returning an empty string.
    """
    if name is None:
        return ""
    # Ensure name is a string before applying regex
    name_str = str(name)
    # Replace all non-alphanumeric characters (including spaces) with a single underscore
    # and strip leading/trailing underscores.
    return re.sub(r'[^a-zA-Z0-9]+', '_', name_str).strip('_')

def get_place_image_count(original_name, found_car_wash_name, image_dir):
    """Counts the number of place images for a given record."""
    safe_original_name = sanitize_filename(original_name)
    safe_found_car_wash_name = sanitize_filename(found_car_wash_name)
    
    place_images_path = os.path.join(image_dir, safe_original_name, safe_found_car_wash_name)
    
    if os.path.exists(place_images_path):
        # Count only .jpg files
        count = len(glob.glob(os.path.join(place_images_path, "*.jpg")))
        return count
    return 0
