import pandas as pd
import argparse
import os
import sys

# Add the directory containing mapsStatic.py and visionModel.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'apiExamples'))
from apiExamples.mapsStatic import get_static_map_image
from apiExamples.visionModel import visionModelResponse

def process_excel_data(start_index, end_index):
    excel_file_path = 'siteAccesibility/datasets/1mile_raw_data.xlsx'
    satellite_image_output_dir = 'siteAccesibility/satellite_images'
    csv_output_dir = 'siteAccesibility/output_csv'
    output_csv_filename = os.path.join(csv_output_dir, 'accessibility_results.csv')

    # Create necessary directories
    if not os.path.exists(satellite_image_output_dir):
        os.makedirs(satellite_image_output_dir)
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)

    try:
        df = pd.read_excel(excel_file_path)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file_path}")
        return

    # Ensure indices are within bounds
    if start_index < 0:
        start_index = 0
    if end_index > len(df):
        end_index = len(df)
    if start_index >= end_index:
        print("Error: Start index must be less than end index.")
        return

    results = []

    for index in range(start_index, end_index):
        row = df.iloc[index]
        site_name = row.iloc[0]  # Column A
        latitude = row.iloc[1]   # Column B
        longitude = row.iloc[2]  # Column C

        # Sanitize site_name for filename
        safe_site_name = "".join([c for c in str(site_name) if c.isalnum() or c in (' ', '.', '_')]).rstrip()
        image_filename = os.path.join(satellite_image_output_dir, f"{safe_site_name}_{index}.png")

        print(f"Processing record {index}: Site Name='{site_name}', Lat={latitude}, Lon={longitude}")

        # 1. Get and save satellite image
        try:
            print(f"  - Attempting to save satellite image to {image_filename}")
            image_saved = get_static_map_image(latitude, longitude, image_filename)
            if image_saved:
                print(f"  - Saved image for '{site_name}' to {image_filename}")

                # 2. Send image to vision model
                print(f"  - Sending image to vision model: {image_filename}")
                vision_response = visionModelResponse(satellite_image_path=image_filename)

                accessibility_score = None
                justification = None

                if vision_response and "error" not in vision_response:
                    accessibility_score = vision_response.get("accessibility_score")
                    justification = vision_response.get("justification")
                    print(f"  - Vision Model Response: Score={accessibility_score}, Justification='{justification[:50]}...'")
                else:
                    print(f"  - Error from Vision Model: {vision_response.get('error', 'Unknown error')}")
                    accessibility_score = "Error"
                    justification = vision_response.get('error', 'Failed to get response')

                results.append({
                    "site_name": site_name,
                    "accessibility_score": accessibility_score,
                    "justification": justification
                })
            else:
                print(f"  - Failed to save image for '{site_name}'. Skipping vision model analysis.")
                results.append({
                    "site_name": site_name,
                    "accessibility_score": "Image Save Failed",
                    "justification": "Image could not be saved."
                })
        except Exception as e:
            print(f"Error processing record {index} ('{site_name}'): {e}")
            results.append({
                "site_name": site_name,
                "accessibility_score": "Processing Error",
                "justification": str(e)
            })

    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv_filename, index=False)
        print(f"\nSuccessfully processed {len(results)} records. Results saved to {output_csv_filename}")
    else:
        print("\nNo records were processed or no results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Excel data to generate satellite images and analyze accessibility.")
    parser.add_argument("--start", type=int, required=True, help="Starting index (inclusive) for processing.")
    parser.add_argument("--end", type=int, required=True, help="Ending index (exclusive) for processing.")
    args = parser.parse_args()

    process_excel_data(args.start, args.end)
