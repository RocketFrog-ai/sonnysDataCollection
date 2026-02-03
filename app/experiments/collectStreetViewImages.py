import pandas as pd
import subprocess
import sys
import os
import json
import re

def sanitize_folder_name(name):
    """Sanitizes a string to be used as a folder name."""
    # Remove characters that are not allowed in folder names
    sanitized_name = re.sub(r'[^\w\s.-]', '', name)
    # Replace spaces with underscores
    sanitized_name = sanitized_name.replace(' ', '_')
    return sanitized_name

def process_record_by_index(target_index):
    """
    Reads a specific record by index from the dataset, calls countCompetitors.py,
    creates folders for not-found competitors, and uses streetViewStatic.py to get images.
    """
    df = pd.read_excel('/home/arpit/dataCollection/datasets/1mile_raw_data.xlsx', engine='openpyxl')

    if target_index < 0 or target_index >= len(df):
        print(f"Error: Index {target_index} is out of bounds. Please provide an index between 0 and {len(df) - 1}.")
        sys.exit(1)

    row = df.iloc[target_index]
    site_address = row.iloc[0] # Assuming the first column is the full site address
    latitude = row.iloc[1]
    longitude = row.iloc[2]

    print(f"Processing record {target_index}: {site_address}, Latitude: {latitude}, Longitude: {longitude}")

    # Define the parent directory for collected images
    parent_dir = os.path.join("/home/arpit/dataCollection", "streetViewImages")
    os.makedirs(parent_dir, exist_ok=True)

    # Sanitize site address for folder name
    folder_name = sanitize_folder_name(str(site_address))
    site_folder_path = os.path.join(parent_dir, folder_name)

    # Create the site folder if it doesn't exist
    os.makedirs(site_folder_path, exist_ok=True)
    print(f"Created folder: {site_folder_path}")

    # Call countCompetitors.py with latitude and longitude and capture output
    try:
        result = subprocess.run(
            ["python", "countCompetitors.py", str(latitude), str(longitude)],
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/arpit/dataCollection" # Ensure the command is run in the correct directory
        )
        count_competitors_output = result.stdout
        print("countCompetitors.py output:")
        print(count_competitors_output)

        # Extract and save the "Matching Results" to a text file
        matching_results_match = re.search(r"--- Matching Results ---\n(.*?)---", count_competitors_output, re.DOTALL)
        if matching_results_match:
            matching_results_content = matching_results_match.group(1).strip()
            matching_results_filepath = os.path.join(site_folder_path, "matching_results.txt")
            with open(matching_results_filepath, "w") as f:
                f.write(matching_results_content)
            print(f"Saved matching results to: {matching_results_filepath}")
        else:
            print("Could not find '--- Matching Results ---' in countCompetitors.py output.")


        # Parse the output to find the JSON string
        json_match = re.search(r"--- Not Found Competitor Details ---\n(.*)", count_competitors_output, re.DOTALL)
        not_found_competitor_details = []
        if json_match:
            json_string = json_match.group(1).strip()
            try:
                not_found_competitor_details = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from countCompetitors.py output: {e}")
                print(f"JSON string: {json_string}")

        # For each not-found competitor, call streetViewStatic.py to get an image
        for competitor in not_found_competitor_details:
            comp_latitude = competitor.get("latitude")
            comp_longitude = competitor.get("longitude")
            place_id = competitor.get("place_id")
            comp_name = competitor.get("name")

            if comp_latitude is not None and comp_longitude is not None and place_id and place_id != "N/A":
                # Call streetViewStatic.py for different degrees
                degrees = [0, 90, 180, 270]
                for degree in degrees:
                    image_filename = f"{place_id}-street-{degree}.png"
                    image_filepath = os.path.join(site_folder_path, image_filename)
                    print(f"Getting street view image for {comp_name} (Place ID: {place_id}) at Lat: {comp_latitude}, Lng: {comp_longitude}, Heading: {degree}")

                    # Call streetViewStatic.py with latitude, longitude, heading, and output path
                    try:
                        subprocess.run(
                            ["python", "apiExamples/streetViewStatic.py", str(comp_latitude), str(comp_longitude), str(degree), image_filepath],
                            check=True,
                            cwd="/home/arpit/dataCollection" # Ensure the command is run in the correct directory
                        )
                        # The streetViewStatic.py script now prints its own success/failure message
                        # print(f"Saved image: {image_filepath}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error running apiExamples/streetViewStatic.py for {comp_name} at {degree} degrees: {e}")
            else:
                print(f"Skipping image for {comp_name} due to missing location or place ID.")


    except subprocess.CalledProcessError as e:
        print(f"Error running countCompetitors.py for {site_address}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python collectStreetViewImages.py <index>")
        sys.exit(1)

    try:
        index_to_process = int(sys.argv[1])
        process_record_by_index(index_to_process)
    except ValueError:
        print("Invalid index provided. Please provide an integer.")
        sys.exit(1)
