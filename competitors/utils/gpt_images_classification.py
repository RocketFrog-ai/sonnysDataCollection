import os
import base64
import mimetypes
import json
from typing import List, Dict, Optional
import openai
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-4o")

# --- Pydantic Model for Car Wash Classification ---
class CarWashClassification(BaseModel):
    classification: Literal["Competitor", "Not a Competitor"] = Field(
        description="Classification of the car wash location."
    )
    justification: str = Field(
        description="Detailed justification for the classification, mentioning visible features, adherence to criteria, or missing/ambiguous data."
    )

def get_mime_type(file_path):
    """Determines the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        else:
            return 'application/octet-stream'
    return mime_type

def image_to_data_url(file_path: str) -> Optional[str]:
    """Encodes an image file into a base64 data URL."""
    try:
        mime_type = get_mime_type(file_path)
        if not mime_type or not mime_type.startswith('image'):
            print(f"Warning: File is not a recognized image type: {file_path}. Skipping.")
            return None
            
        with open(file_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        print(f"Error encoding image {file_path} to data URL: {e}")
        return None

def visionModelResponse(place_images_folder_path: str, satellite_image_path: str) -> dict:
    """
    Analyzes car wash images (customer/business uploads and satellite image)
    to determine if it's an Express Tunnel Car Wash, returning structured JSON.
    """
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_DEPLOYMENT_NAME]):
        return {"error": "Azure OpenAI environment variables are not fully configured."}

    all_image_content_parts = []

    # 1. Load Satellite Image
    print(f"Loading satellite image from: {satellite_image_path}")
    if os.path.isfile(satellite_image_path):
        satellite_data_url = image_to_data_url(satellite_image_path)
        if satellite_data_url:
            all_image_content_parts.append({
                "type": "image_url",
                "image_url": {"url": satellite_data_url, "detail": "high"}
            })
            print(f"  - Loaded: {os.path.basename(satellite_image_path)}")
    else:
        return {"error": f"Satellite image file not found at '{satellite_image_path}'. Please check the path."}

    # 2. Load Place Photos (up to 9)
    print(f"Loading images from: {place_images_folder_path}")
    if os.path.isdir(place_images_folder_path):
        image_files = [f for f in os.listdir(place_images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for filename in image_files[:9]:
            file_path = os.path.join(place_images_folder_path, filename)
            place_data_url = image_to_data_url(file_path)
            if place_data_url:
                all_image_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": place_data_url, "detail": "high"}
                })
                print(f"  - Loaded: {filename}")
    else:
        print(f"Place images folder not found or is not a directory at '{place_images_folder_path}'. Skipping place images.")

    if not all_image_content_parts:
        return {"error": "No valid images were found to analyze. Please ensure the paths are correct and images exist."}

    system_prompt = """
You are an expert AI assistant specializing in analyzing car wash locations from publicly available images (satellite views, Google/Yelp photos, etc.). Your primary goal is to determine if a given car wash location is an "Express Tunnel Car Wash" and therefore a competitor, based on specific visual criteria.

You will be provided with a set of images for a location. Analyze these images carefully against the following criteria:

**Classification Criteria for "Express Tunnel Car Wash" (Competitor):**

1.  **Tunnel Structure (Mandatory):**
    *   Look for images showing a long, narrow building or an open-ended structure through which cars appear to enter and exit in a continuous, straight line.
    *   The tunnel must have discernible entry and exit points (e.g., arches, doors, often labeled "Enter" and "Exit").
    *   Interior views (if available) might show automated equipment like rollers, brushes, overhead sprayers, or drying systems.
    *   The tunnel interior should be equipped for automated exterior cleaning of a car. Manual cleaning by humans inside the tunnel means it's NOT an express tunnel system.
    *   A visibly wet or shiny surface at the tunnel exit can be an indicator of frequent use.
    *   Attempt to visually estimate if the tunnel length is substantial (e.g., appears to be at least 34 feet / 10 meters or longer). Shorter, rudimentary tunnels might not qualify.
    *   If a tunnel structure is not clearly visible or identifiable as automated, it's likely not an express tunnel.

2.  **Conveyor System (Strong Indicator, if visible):**
    *   Look for guide rails, rollers, or tracks on the ground inside or leading into the tunnel. These are characteristic of a conveyorized (automated) wash.
    *   Multiple cars lined up in sequence, either inside the tunnel or waiting to enter, can also suggest a conveyor system.

3.  **Drive-Through Experience (Typical):**
    *   Customers typically remain in their vehicles throughout the wash process.
    *   Observe if cars are entering the tunnel one by one.
    *   There should be no signs of staff manually cleaning the car's exterior during the main wash phase within the tunnel.
    *   Absence of interior cleaning services being performed simultaneously (e.g., no open car doors with attendants, no vacuuming inside the car by staff during the tunnel pass).

4.  **Branding or Signage Clues (Important):**
    *   Look for words like "Express," "Exterior," "Tunnel Wash," "Automatic Car Wash," or similar terms in the business name or on visible signage. These are strong indicators.
    *   If signage clearly indicates "Full Serve," "Hand Wash," or "Detailing" as the primary service, it's less likely to be an express tunnel model *unless* there's also a clear, separate, and substantial express tunnel operation visible.
    *   A "Full Serve" location might be considered a competitor *only if* it has a very clearly identifiable, automated express tunnel of significant length and automation, suggesting it *also* offers express exterior services. If the tunnel at a "Full Serve" is short, open-roofed, or lacks extensive automated equipment, it's not a competitor.

5.  **Exclusions:**
    *   Do NOT classify "Truck Washes" (e.g., "Blue Beacon") as competitors, even if they use a tunnel. Focus on passenger vehicle car washes.
    *   Services like "Window Tinting," "Oil Change," or "Gas Station" car washes are not competitors unless they feature a distinct, standalone, automated express tunnel car wash as described above.
    *   Mobile car washes are NOT competitors unless they also operate a fixed-location public car wash with a clearly visible, qualifying Express Tunnel.
    *   Self-service bays (where customers wash their own cars) are NOT express tunnels.

6.  **Vacuum Station Nearby (Optional, Supporting Evidence):**
    *   The presence of rows of self-serve vacuum stations (covered or uncovered) adjacent to the tunnel exit is common for express models but not a mandatory criterion.

**Your Response Format (Strictly Adhere):**

You MUST provide your response in the following JSON format:
```json
{
"classification": "Competitor" | "Not a Competitor",
"justification": "Detailed explanation covering: - Visible features supporting your classification (tunnel presence/type, entrance/exit, equipment, signage, conveyor, vacuum area if relevant). - How the location meets or fails the criteria."
}
"""
    user_query_prompt = "Analyze the provided images for the car wash location and determine if it is an Express Tunnel Car Wash competitor based on the criteria."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query_prompt}
            ] + all_image_content_parts
        }
    ]

    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )

        print(f"\nSending request to Azure OpenAI with {len(all_image_content_parts)} image(s)...")
        
        completion = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,
            messages=messages,
            response_format=CarWashClassification,
            max_tokens=1500,
        )
        
        parsed_output: CarWashClassification = completion.choices[0].message.parsed
        return parsed_output.model_dump()
    
    except openai.APIConnectionError as e:
        return {"error": f"The server could not be reached: {e.__cause__}"}
    except openai.RateLimitError as e:
        return {"error": f"A 429 status code was received; we should back off a bit: {e}"}
    except openai.APIStatusError as e:
        return {"error": f"Another non-200-range status code was received: {e.status_code}", "response": str(e.response)}
    except openai.BadRequestError as e:
        return {"error": f"Bad request (e.g. invalid schema, model output issue, model does not support structured outputs, invalid inputs): {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during OpenAI call: {e}"}

if __name__ == "__main__":
    # --- IMPORTANT: Configure your image paths here for testing ---
    # Create a .env file in your project root with:
    # AZURE_OPENAI_ENDPOINT="your_endpoint"
    # AZURE_OPENAI_API_KEY="your_api_key"
    # AZURE_OPENAI_MODEL_DEPLOYMENT_NAME="your_deployment_name"

    # Example paths:
    my_place_images_folder = "path/to/your/place_images"
    my_satellite_image = "path/to/your/satellite_image.jpg"

    # Check if the required env vars are set before running
    if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")]):
        print("Error: Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file.")
    else:
        print("\n--- Attempting to run visionModelResponse with example paths ---")
        # Note: This will fail if the example paths are not valid.
        # Replace them with actual paths to test.
        response_dict = visionModelResponse(
            place_images_folder_path=my_place_images_folder,
            satellite_image_path=my_satellite_image
        )
        print("\n--- Model Response (Structured JSON) ---")
        print(json.dumps(response_dict, indent=2))
