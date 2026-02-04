import os
import base64
import mimetypes
import json
from typing import List, Dict, Optional
import openai
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from app.utils import common as calib

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = calib.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = calib.AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION = calib.AZURE_OPENAI_API_VERSION
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = calib.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME

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
You are analyzing publicly available images of car wash locations (from Google or Yelp), either uploaded by the business or its customers. Your goal is to determine whether the location is an express car wash that uses a tunnel system.

Classify a car wash as a “Competitor” if the following indicators are present:
1. Tunnel Structure:
Look for images showing a long, narrow building or open-ended structure through which cars appear to enter and exit in a straight line.


The tunnel should:


Have entry and exit arches or doors (often labeled "Enter" and "Exit").
Sometimes show rollers, brushes, or overhead sprayers inside.
May be visibly wet or shiny at the exit — this is a common sign of high-frequency washes.
Inside the tunnel you will find many cleaning and drying equipment, soaps and brushes working specifically on the exterior surface of the car.
Make sure that inside the tunnel, the cleaning is done using automated equipments and NOT done manually by a human.
Try to figure out the length of the tunnel, it should be at least 34 feet. The more extensive the tunnel’s length is, the better.
If the tunnel wash is not happening via an automated express system, it is not a competitor.


2. Conveyor System (if visible):
Look for guide rails, rollers, or tracks on the ground inside the tunnel that cars align with — these indicate a conveyorized wash.
May see multiple cars lined up in a sequence inside or outside the tunnel.


3. Drive-Through Experience:
Customers typically stay in the vehicle during the wash. You might observe:


Cars entering one by one.
No staff manually cleaning during the wash phase.
No interior cleaning being shown (no opened car doors, no attendants inside cars).


4. Branding or Signage Clues:
The business name or visible signage includes words like:


"Express"


"Exterior"


"Tunnel Wash"


These are strong indicators of an express tunnel model.
If the signage says “Full serve”, we will not consider it as a competitor until and unless it has a lot of real estate area where it can expand its express wash tunnel and use that space for the tunnel experience. In this case we need to judge the length of the tunnel. If it is very short, does not have enough cleaning equipments, has like an open roof, then it will not be considered as a competitor.
Do not consider Truck washes, or truck wash companies like “Blue Beacon” as competitors.
Do not consider Window tinting services of a car wash as a competitor.
If a Mobile car wash offers Public car wash services too with a clearly visible Express Tunnel available for service, only then consider it as a competitor.
In any case, a tunnel is a must.


5. Vacuum Station Nearby (Optional):
Rows of covered or uncovered self-serve vacuums with hoses may appear adjacent to the tunnel.
While common, this is not required for classification.



📝 Response Format (per location):

Classification: (Express Tunnel Car Wash)Competitor / (Not an Express Tunnel)Not a Competitor

Justification:
- [Mention visible features: tunnel structure, entrance/exit, equipment, signage, conveyor, vacuum area, etc.]
- [If classification is unclear, explain what data was missing or ambiguous.]
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
