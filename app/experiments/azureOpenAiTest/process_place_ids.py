import csv
import os
from visionStructuredResponse import classify_car_wash_from_google_place_id, CarWashClassification

# Define file paths
INPUT_CSV_PATH = "azureOpenAiTest/place_ids.csv"
OUTPUT_CSV_PATH = "azureOpenAiTest/classification_output.csv"

# User prompt for the AI model (can be the same as in visionStructuredResponse.py)
USER_PROMPT_FOR_AI = "Analyze the provided images for a location and determine if it is an Express Tunnel Car Wash competitor based on the criteria."

def main():
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    place_ids_processed = set() # To avoid processing duplicate place_ids if any
    rows_written = 0

    print(f"Reading place IDs from: {INPUT_CSV_PATH}")
    print(f"Results will be written one by one to: {OUTPUT_CSV_PATH}")

    try:
        with open(INPUT_CSV_PATH, mode='r', newline='', encoding='utf-8') as infile, \
             open(OUTPUT_CSV_PATH, mode='w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            
            fieldnames = ["place_id", "Classification", "Justification"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader() # Write header at the beginning

            header_from_input = next(reader) # Skip header row from input
            if header_from_input[0].strip().lower() != 'place_id':
                print(f"Warning: Expected input header 'place_id', but found '{header_from_input[0]}'. Processing first column.")

            for row_number, row in enumerate(reader, 1):
                if not row:  # Skip empty rows
                    print(f"Skipping empty input row {row_number + 1}")
                    continue
                
                image_filename = row[0].strip()
                if not image_filename:
                    print(f"Skipping input row {row_number + 1} due to empty image filename.")
                    continue

                # Extract place_id by removing .jpg suffix
                if image_filename.lower().endswith(".jpg"):
                    place_id = image_filename[:-4]
                else:
                    print(f"Warning: Filename '{image_filename}' in input row {row_number + 1} does not end with .jpg. Using as is.")
                    place_id = image_filename
                
                if not place_id:
                    print(f"Skipping input row {row_number + 1} as extracted place_id is empty from '{image_filename}'.")
                    continue

                if place_id in place_ids_processed:
                    print(f"Skipping duplicate place_id: {place_id}")
                    continue
                
                print(f"\nProcessing Place ID: {place_id} (from {image_filename})")
                
                classification_result: CarWashClassification = classify_car_wash_from_google_place_id(place_id, USER_PROMPT_FOR_AI)
                
                current_result_row = {}
                if classification_result:
                    current_result_row = {
                        "place_id": place_id,
                        "Classification": classification_result.Classification,
                        "Justification": classification_result.Justification
                    }
                    print(f"Result for {place_id}: {classification_result.Classification}")
                else:
                    current_result_row = {
                        "place_id": place_id,
                        "Classification": "Error",
                        "Justification": "Classification failed or returned no result."
                    }
                    print(f"Failed to classify {place_id}.")
                
                writer.writerow(current_result_row)
                rows_written += 1
                place_ids_processed.add(place_id)

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if rows_written > 0:
        print(f"\nSuccessfully processed and wrote {rows_written} results to {OUTPUT_CSV_PATH}")
    else:
        print("\nNo place IDs were processed or no results obtained. Output CSV might be empty or contain only headers.")

if __name__ == "__main__":
    main()
