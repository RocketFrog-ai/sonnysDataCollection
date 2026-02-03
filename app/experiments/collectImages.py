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

def process_records(num_records):
    """
    Reads a specified number of records from the dataset, calls countCompetitors.py,
    creates folders for not-found competitors, and uses mapsStatic.py to get images.
    """
    df = pd.read_excel('/home/arpit/dataCollection/datasets/1mile_raw_data.xlsx', engine='openpyxl')

    for index, row in df.head(num_records).iterrows():
        site_address = row.iloc[0] # Assuming the first column is the full site address
        latitude = row.iloc[1]
        longitude = row.iloc[2]

        print(f"Processing record for site: {site_address}, Latitude: {latitude}, Longitude: {longitude}")

        # Define the parent directory for collected images
        parent_dir = os.path.join("/home/arpit/dataCollection", "collectedImages")
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

            # For each not-found competitor, call mapsStatic.py to get an image
            for competitor in not_found_competitor_details:
                comp_latitude = competitor.get("latitude")
                comp_longitude = competitor.get("longitude")
                place_id = competitor.get("place_id")
                comp_name = competitor.get("name")

                if comp_latitude is not None and comp_longitude is not None and place_id and place_id != "N/A":
                    image_filename = f"{place_id}.png"
                    image_filepath = os.path.join(site_folder_path, image_filename)
                    print(f"Getting image for {comp_name} (Place ID: {place_id}) at Lat: {comp_latitude}, Lng: {comp_longitude}")

                    # Call mapsStatic.py with latitude, longitude, and output path
                    try:
                        subprocess.run(
                            ["python", "mapsStatic.py", str(comp_latitude), str(comp_longitude), image_filepath],
                            check=True,
                            cwd="/home/arpit/dataCollection" # Ensure the command is run in the correct directory
                        )
                        print(f"Saved image: {image_filepath}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error running mapsStatic.py for {comp_name}: {e}")
                else:
                    print(f"Skipping image for {comp_name} due to missing location or place ID.")


        except subprocess.CalledProcessError as e:
            print(f"Error running countCompetitors.py for {site_address}: {e}")


def process_records(start_index, end_index):
    """
    Reads a specified range of records from the dataset, calls countCompetitors.py,
    creates folders for not-found competitors, and uses mapsStatic.py to get images.
    """
    df = pd.read_excel('/home/arpit/dataCollection/datasets/1mile_raw_data.xlsx', engine='openpyxl')

    # Ensure indices are within the DataFrame bounds
    if start_index < 0 or end_index > len(df) or start_index >= end_index:
        print(f"Error: Invalid record range. Please provide a valid range between 0 and {len(df) -1}.")
        sys.exit(1)

    for index, row in df.iloc[start_index:end_index].iterrows():
        site_address = row.iloc[0] # Assuming the first column is the full site address
        latitude = row.iloc[1]
        longitude = row.iloc[2]

        print(f"Processing record {index}: {site_address}, Latitude: {latitude}, Longitude: {longitude}")

        # Define the parent directory for collected images
        parent_dir = os.path.join("/home/arpit/dataCollection", "collectedImages")
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

            # For each not-found competitor, call mapsStatic.py to get an image
            for competitor in not_found_competitor_details:
                comp_latitude = competitor.get("latitude")
                comp_longitude = competitor.get("longitude")
                place_id = competitor.get("place_id")
                comp_name = competitor.get("name")

                if comp_latitude is not None and comp_longitude is not None and place_id and place_id != "N/A":
                    image_filename = f"{place_id}.png"
                    image_filepath = os.path.join(site_folder_path, image_filename)
                    print(f"Getting image for {comp_name} (Place ID: {place_id}) at Lat: {comp_latitude}, Lng: {comp_longitude}")

                    # Call mapsStatic.py with latitude, longitude, and output path
                    try:
                        subprocess.run(
                            ["python", "mapsStatic.py", str(comp_latitude), str(comp_longitude), image_filepath],
                            check=True,
                            cwd="/home/arpit/dataCollection" # Ensure the command is run in the correct directory
                        )
                        print(f"Saved image: {image_filepath}")
                    except subprocess.CalledProcessError as e:
                        print(f"Error running mapsStatic.py for {comp_name}: {e}")
                else:
                    print(f"Skipping image for {comp_name} due to missing location or place ID.")


        except subprocess.CalledProcessError as e:
            print(f"Error running countCompetitors.py for {site_address}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python collectImages.py <start_index> <end_index>")
        sys.exit(1)

    try:
        start_index_to_process = int(sys.argv[1])
        end_index_to_process = int(sys.argv[2])
        process_records(start_index_to_process, end_index_to_process)
    except ValueError:
        print("Invalid start or end index provided. Please provide integers.")
        sys.exit(1)
