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
class StackupCapacity(BaseModel):
    stackup_capacity: int = Field(
        description="The estimated number of cars that can fit in the stack-up area."
    )
    justification: str = Field(
        description="Detailed justification for the estimation, mentioning visible features, car-length estimates, and any assumptions made."
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
    to determine it's entrance stackup area, returning structured JSON.
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
You are an AI assistant that analyzes satellite images of car wash locations to estimate their vehicle stack-up capacity. Your task is to provide a numerical estimate of how many cars can fit in the car wash's stack-up area (the space where cars queue) and to justify your decision.

🖼️ Input Images
For each car wash location, you will be provided with two satellite images (Google Static Maps API) at:

    Zoom 19: shows the lot in context with surrounding lots and access points.

    Zoom 20: gives very fine detail—lane markings, curb cuts, parking stalls, and vehicle queue lines.

Each image will have a red circle marking the car wash center; focus on the adjacent stack-up area (ingress/egress lanes and waiting zones).

🚗 Stack-Up Capacity Estimation

    Your primary goal is to estimate the total number of full-sized vehicles that can queue in the designated stack-up lanes.

🔍 Visual Clues & Process

    Measure car-lengths

        At zoom 20, estimate one car length (≈5 meters). Count how many would fit end-to-end in the visible queue lanes.

    Identify queue configuration

        Single long lane vs. multiple parallel lanes vs. overflow bays. Sum their capacities.

    Check entry/exit layout

        Note if cars stack in separate spur lanes alongside conveyors or wash bays.

    Disregard parking stalls

        Only count the queuing lanes intended for cars waiting to enter the wash, not customer parking.

Proceed by examining the red-circled area in each zoom image, estimate the total stack-up capacity as a number, and justify your answer with specific visual evidence and your car-length estimates.

Your response must be a JSON object with the following structure:
{
  "stackup_capacity": <number>,
  "justification": "A detailed explanation with car-length counts, lane descriptions, and any assumptions made."
}
"""
    user_query_prompt = "Analyze the provided images for the car wash location and determine the entrance stackup area based on the criteria."

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
            max_completion_tokens=5000,
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
                validated_output = StackupCapacity(**parsed_output)
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
    print(visionModelResponse('entranceStackup/satellite_images/ StoneWash  Car Care Center  818 Paris Rd  Mayfield  KY  42066'))
