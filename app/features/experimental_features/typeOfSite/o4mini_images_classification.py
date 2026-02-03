import os
import base64
import mimetypes
import json
from typing import List, Dict, Optional
import re
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
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "o4-mini")

# --- Pydantic Model for Car Wash Classification ---
class CarWashClassification(BaseModel):
    classification: Literal["Corner", "Inside"] = Field(
        description="Classification of the car wash location."
    )
    justification: str = Field(
        description="Detailed justification for the classification, mentioning visible features, adherence to criteria, or missing/ambiguous data."
    )

def parse_json_from_string(json_string: str) -> Optional[Dict]:
    """
    Parses a JSON object from a string, handling cases where it's embedded in other text,
    including markdown code blocks.
    """
    if not isinstance(json_string, str):
        return None

    # First, try to find a JSON markdown block
    match = re.search(r'```json\s*(\{.*?\})\s*```', json_string, re.DOTALL)
    if match:
        json_part = match.group(1)
        try:
            return json.loads(json_part)
        except json.JSONDecodeError:
            # If markdown parsing fails, fall through to the next method
            pass

    # If no markdown block or parsing failed, try to find the first and last curly brace
    try:
        start_index = json_string.find('{')
        if start_index == -1:
            return None
        end_index = json_string.rfind('}')
        if end_index == -1:
            return None
        json_part = json_string[start_index:end_index + 1]
        return json.loads(json_part)
    except (json.JSONDecodeError, IndexError):
        return None

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

def visionModelResponse(satellite_images_folder_path: str) -> dict:
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
You are an AI assistant that analyzes satellite images of car wash locations.
Your task is to classify whether the car wash is located on a Corner Lot or an Inside Lot and provide a justification for your decision.


🖼️ Input Images:
For each car wash location, you will be provided with three satellite images (Google Static Maps API) at zoom levels 17, 18, and 19.

    Zoom 17: broader context (identify main roads, highways, and intersections)

    Zoom 18 & 19: finer details (lane markings, driveways, lot boundaries)

🚩 Red Circle:
Each image will have a red circle marking the lot of interest. Focus your analysis on that circled area.

🛣️ Lot Type Definitions:

Corner Lot
A car wash is considered a Corner Lot if:

    It sits at an intersection of two major roads or highways—not small residential streets, drives, lanes, service roads, or alleys.

    At least two wide roads meet adjacent to the red‑circled lot, forming a visible corner.

    One or more sides of the lot face these main roads, even if other sides are blocked by fences or one‑way restrictions.

    Look for signs of road importance: multiple lanes, turn arrows, crosswalks, highway shields, or clear road names.

Inside Lot
A car wash is considered an Inside Lot if:

    It lies mid‑block between properties or alongside only one major road.

    No intersection of two main roads is adjacent to the red‑circled lot.

    Ignore streets, service drives, alleys, private lanes, or dead‑end residential roads—even if they border the lot, they do not qualify it as a Corner Lot.

🔍 Visual Clues & Process:

    Identify and confirm major road status by width, lane markings, labels, or traffic features.

    Check for intersections of these major roads next to the red circle.

    Disregard any narrow or private access ways.

    Use higher zoom levels to verify driveway access, curb cuts, and lot boundaries.

Proceed by examining the red‑circled lot in each zoom image, determine its classification, and justify your answer with specific visual and contextual evidence.

Your response must be a JSON object with the following structure:
{
  "classification": "Corner" | "Inside",
  "justification": "A detailed explanation for the classification."
}
"""
    user_query_prompt = "Analyze the provided images for the car wash location and determine if it is a corner lot or an inside lot based on the criteria."

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
        
        completion = client.chat.completions.create(
            model=AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,
            messages=messages,
            max_completion_tokens=3000,
            reasoning_effort="high"
        )
        
        response_content = completion.choices[0].message.content

        if not response_content:
            return {"error": "Model returned an empty response.", "raw_response": response_content}
        
        # Attempt to parse the JSON from the response content
        parsed_output = parse_json_from_string(response_content)
        
        if parsed_output:
            # Validate with Pydantic
            try:
                validated_output = CarWashClassification(**parsed_output)
                return validated_output.model_dump()
            except Exception as e:
                return {"error": f"Pydantic validation failed: {e}", "raw_response": response_content}
        else:
            return {"error": "Failed to parse JSON from model response.", "raw_response": response_content}
    
    except openai.APIConnectionError as e:
        return {"error": f"The server could not be reached: {e.__cause__}"}
    except openai.RateLimitError as e:
        return {"error": f"A 429 status code was received; we should back off a bit: {e}"}
    except openai.APIStatusError as e:
        print(f"API Status Error: {e.status_code}")
        try:
            print(f"Response Body: {e.response.json()}")
        except Exception:
            print(f"Response Text: {e.response.text}")
        return {"error": f"Another non-200-range status code was received: {e.status_code}", "response": str(e.response)}
    except openai.BadRequestError as e:
        return {"error": f"Bad request (e.g. invalid schema, model output issue, model does not support structured outputs, invalid inputs): {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during OpenAI call: {e}"}
if __name__ == '__main__':
    print(visionModelResponse('typeOfSite/satellite_images/26 WEISS GUYS EXPRESS  13820 N 40TH STREET  PHOENIX  AZ  85032'))
