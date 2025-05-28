import os
import mimetypes
import json # Import json to parse the model's output
from google import genai
from google.genai import types

def get_mime_type(file_path):
    """Determines the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # Fallback for common image types if mimetypes can't guess
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        elif ext == '.gif':
            return 'image/gif'
        elif ext == '.webp':
            return 'image/webp'
        else:
            return 'application/octet-stream' # Generic binary, might cause issues if not an image
    return mime_type

def visionModelResponse(place_images_folder_path: str, satellite_image_path: str) -> dict:
    """
    Analyzes car wash images (customer/business uploads and satellite image)
    to determine if it's an Express Tunnel Car Wash, returning structured JSON.

    Args:
        place_images_folder_path: Path to a folder containing publicly available images
                                  (e.g., from Google Maps, Yelp) of the car wash.
        satellite_image_path: Path to a single satellite image of the car wash location.

    Returns:
        A dictionary containing the classification and justification from the vision model,
        or an error dictionary if issues occur.
    """
    client = genai.Client(
        api_key="AIzaSyAt59WZAmoN2FVj_FZM6wYvAdJa5Q3MFL0",
    )

    # Use the specified model
    model = "gemini-2.5-flash-preview-05-20"

    # --- Prepare image parts from both sources ---
    image_parts = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')

    # Add images from the 'place_images_folder_path'
    print(f"Loading images from: {place_images_folder_path}")
    try:
        if os.path.isdir(place_images_folder_path):
            for filename in os.listdir(place_images_folder_path):
                if filename.lower().endswith(supported_extensions):
                    file_path = os.path.join(place_images_folder_path, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            img_bytes = f.read()
                        image_parts.append(
                            types.Part.from_bytes(
                                data=img_bytes,
                                mime_type=get_mime_type(file_path)
                            )
                        )
                        print(f"  - Loaded: {filename}")
                    except Exception as e:
                        print(f"  - Error reading image {file_path}: {e}. Skipping.")
        else:
            print(f"Place images folder not found or is not a directory at '{place_images_folder_path}'. Skipping place images.")
    except Exception as e:
        print(f"Error processing place images folder '{place_images_folder_path}': {e}. Skipping place images.")

    # Add the satellite image
    print(f"Loading satellite image from: {satellite_image_path}")
    try:
        if not os.path.isfile(satellite_image_path):
            return {"error": f"Satellite image file not found at '{satellite_image_path}'. Please check the path."}
            
        with open(satellite_image_path, 'rb') as f:
            sat_img_bytes = f.read()
        image_parts.append(
            types.Part.from_bytes(
                data=sat_img_bytes,
                mime_type=get_mime_type(satellite_image_path)
            )
        )
        print(f"  - Loaded: {os.path.basename(satellite_image_path)}")
    except Exception as e:
        return {"error": f"Error reading satellite image '{satellite_image_path}': {e}"}

    if not image_parts:
        return {"error": "No valid images were found to analyze. Please ensure the paths are correct and images exist in the folder."}

    # --- Construct the text prompt ---
    classification_criteria_prompt = """
You are analyzing publicly available images of car wash locations (from Google or Yelp), either uploaded by the business or its customers, including satellite images. Your goal is to determine whether the location is an express car wash that uses a tunnel system.

Classify a car wash as an “Express Tunnel Car Wash” if the following indicators are present:
1. Tunnel Structure:
Look for images showing a long, narrow building or open-ended structure through which cars appear to enter and exit in a straight line.
The tunnel should:
- Have entry and exit arches or doors (often labeled "Enter" and "Exit").
- Sometimes show rollers, brushes, or overhead sprayers inside.
- May be visibly wet or shiny at the exit — this is a common sign of high-frequency washes.
- Inside the tunnel you will find many cleaning and drying equipments, soaps and brushes working on the exterior surface of the car.

2. Conveyor System (if visible):
Look for guide rails, rollers, or tracks on the ground inside the tunnel that cars align with — these indicate a conveyorized wash.
May see multiple cars lined up in a sequence inside or outside the tunnel.

3. Drive-Through Experience:
Customers typically stay in the vehicle during the wash. You might observe:
- Cars entering one by one.
- No staff manually cleaning during the wash phase.
- No interior cleaning being shown (no opened car doors, no attendants inside cars).

4. Branding or Signage Clues:
The business name or visible signage includes words like:
- "Express"
- "Exterior"
- "Tunnel Wash"
These are strong indicators of an express tunnel model.

5. Vacuum Station Nearby (Optional):
Rows of covered or uncovered self-serve vacuums with hoses may appear adjacent to the tunnel.
While common, this is not required for classification.

Based on ALL the provided images (place images and satellite image), classify this car wash location and provide a detailed justification.
Generate the response in JSON format according to the provided schema.
"""

    # The contents list starts with the text prompt, followed by all loaded image parts.
    contents = [
        types.Part.from_text(text=classification_criteria_prompt)
    ] + image_parts

    # --- Define the structured output schema ---
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["classification", "justification"],
            properties={
                "classification": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="The classification of the car wash: 'Express Tunnel Car Wash' or 'Not an Express Tunnel'.",
                    enum=["Express Tunnel Car Wash", "Not an Express Tunnel"] # Using enum for stricter classification
                ),
                "justification": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="A detailed explanation of why the car wash was classified as such, referencing visible features from the images."
                ),
            },
        ),
    )

    # --- Call the Generative Model ---
    print("\nSending request to Gemini API...")
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

        # The response.text will now contain a JSON string
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response from model: {e}", "raw_response": response.text}

    except Exception as e:
        return {"error": f"Error during AI model inference: {e}\nEnsure your GEMINI_API_KEY is set and the API is accessible."}

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # --- IMPORTANT: Configure your image paths here ---
    # Replace these with the actual paths to your car wash images and satellite image.
    # Make sure the folder exists and contains valid image files.
    # For example:
    # my_place_images_folder = "path/to/your/car_wash_photos"
    # my_satellite_image = "path/to/your/satellite_view.jpg"

    my_place_images_folder = "place_images/26_WEISS_GUYS_EXPRESS_13820_N_40TH_STREET_PHOENIX_AZ_85032/Elephant_Car_Wash" # <-- CHANGE THIS to your folder path!
    my_satellite_image = "satellite_images/ChIJ874Sh8tzK4cRnDHD-MbhtBU.jpg" # <-- CHANGE THIS to your satellite image path!

    print("\n--- Attempting to run visionModelResponse with provided paths ---")

    # NOTE: For this to work, you MUST have your GEMINI_API_KEY environment variable set.
    # On Linux/macOS: export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    # On Windows (CMD): set GEMINI_API_KEY="YOUR_API_KEY_HERE"
    # On Windows (PowerShell): $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"


    response_dict = visionModelResponse(
        place_images_folder_path=my_place_images_folder,
        satellite_image_path=my_satellite_image
    )
    print("\n--- Model Response (Structured JSON) ---")
    print(json.dumps(response_dict, indent=2)) # Pretty print the JSON output
