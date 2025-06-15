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
    classification: Literal["Corner", "Inside"] = Field(
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

def visionModelResponse(satellite_images_folder_path: str, car_wash_name: Optional[str] = None) -> dict:
    """
    Analyzes car wash images (satellite images)
    to determine if it's a corner or inside lot, returning structured JSON.
    """
    if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_DEPLOYMENT_NAME]):
        return {"error": "Azure OpenAI environment variables are not fully configured."}

    all_image_content_parts = []

    # Load Satellite Images
    print(f"Loading images from: {satellite_images_folder_path}")
    if os.path.isdir(satellite_images_folder_path):
        image_files = [f for f in os.listdir(satellite_images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        for filename in image_files:
            file_path = os.path.join(satellite_images_folder_path, filename)
            data_url = image_to_data_url(file_path)
            if data_url:
                all_image_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"}
                })
                print(f"  - Loaded: {filename}")
    else:
        return {"error": f"Satellite images folder not found or is not a directory at '{satellite_images_folder_path}'. Please check the path."}


    if not all_image_content_parts:
        return {"error": "No valid images were found to analyze. Please ensure the paths are correct and images exist."}

    system_prompt = """
You are analyzing satellite images of car wash location.
 Your task is to classify whether the car wash is located on a Corner Lot or an Inside Lot.

🖼️ Input Images:
For a car wash location, you will be provided with three satellite images sourced from the Google Static Maps API. These images will be at zoom levels 18, 19, and 20 respectively. Utilize these different zoom levels to get a comprehensive view: zoom 18 for broader context (e.g., identifying main roads and intersections) and zooms 19 and 20 for finer details (e.g., lane markings, access points, lot boundaries).

Use the following visual and contextual rules:

🛣️ Lot Type Definitions:

Corner Lot
A car wash is considered a Corner Lot if:
It is located at an intersection of two main roads (not small residential lanes).
At least two roads meet near or adjacent to the lot, forming a visible corner.
At least one side of the lot should be accessible from the road, even if the other side is blocked or restricted (due to walls, fences, one-way flow, etc.).
The roads must be visibly wide enough or show signs of being major roads (e.g., lane markings, road labels, crosswalks, or traffic flow features like turn arrows).
Diagonal or angled corner lots are also valid if road edges clearly meet at a point.

Inside Lot
A car wash is considered an Inside Lot if:
It is located mid-block or between other properties.
It has access from only one main road, with no visible road intersection adjacent to the lot.
Even if a tiny lane or alley runs behind or beside it, do not classify it as a Corner Lot unless it connects two major roads.

🔍 Use These Visual Clues
Look at road intersections: Two roads must visibly connect at a corner.
Ignore narrow alleyways or dead-end residential roads.
Look for sidewalks, markings, or driveways suggesting road access.
Use text in the image like road names or arrows to judge road importance and access.
"""
    user_query_prompt = "Analyze the provided images for the car wash location and determine if it is a corner lot or an inside lot based on the criteria."
    if car_wash_name:
        user_query_prompt += f"\nThe name of the car wash is '{car_wash_name}'."

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
