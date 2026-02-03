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
class AccessibilityScore(BaseModel):
    accessibility_score: int = Field(
        description="A 1-10 integer score that defines how accessible the site is."
    )
    rationale: str = Field(
        description="Detailed justification for the estimation, mentioning visible features, and any assumptions made."
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
Role: You are an AI expert in geographic and logistical analysis, specializing in commercial site evaluation.

Objective:
Analyze the provided set of four satellite images, which show a single car wash location at different zoom levels. The specific car wash building, our region of interest, is marked with a red circle in all four images. Your goal is to evaluate the site's vehicular accessibility for a typical customer trying to enter and exit the property from the main road. Based on your analysis, you must provide a detailed rationale and a final accessibility score.

IMPORTANT: Your analysis should focus only on the ease of getting into and out of the site itself. Do not evaluate the on-site traffic flow, queuing space, or layout of the vacuum/finishing areas.

Analysis Criteria:
Synthesize information from all four zoom levels to evaluate the following factors. Use the zoomed-out images to understand the broader context and the zoomed-in images for specific site details.

    Primary Road Access & Visibility:

        Identify the type of primary road the site is on (e.g., major highway, multi-lane arterial road, quieter side street).

        How direct is the entrance from this primary road? Is it a simple, single turn or does it require navigating secondary roads or complex intersections?

        How visible would the entrance be to a driver approaching on the main road?

    Entrance & Exit Complexity:

        Assess the entrance and exit points. Are they clearly defined and easy to spot?

        Is the driveway dedicated solely to the car wash, or is it shared with adjacent businesses (e.g., a shared entrance for a shopping center)? Shared access can cause confusion and traffic conflicts.

        Look for physical barriers or road features that complicate access. Specifically, identify if there is a median on the main road that prevents left-hand turns into or out of the site, forcing complex U-turns or detours for drivers approaching from one direction.

    Surrounding Environment & Potential for Conflict:

        Describe the immediate surroundings. Is the site in a dense commercial corridor with heavy traffic, a standalone location, or a quieter area?

        How does this context impact the ease of entering and exiting the property? High cross-traffic from nearby businesses (like a fast-food drive-thru or busy retail store) right next to the entrance/exit can create significant delays and increase the difficulty of turning.

Scoring Rubric (1-10):
Assign your final score based on the overall ease of entering and exiting the property.

    1-3 (Poor Accessibility): Access is extremely difficult. The entrance is indirect, hidden, or requires navigating through a confusing, high-traffic commercial lot. Major physical barriers like medians severely restrict access from a primary direction.

    4-6 (Moderate Accessibility): Access is functional but has significant drawbacks. The entrance may be shared with several other businesses, causing potential conflicts, or a road median might force inconvenient U-turns for some customers.

    7-8 (Good Accessibility): Access is largely easy and direct. The entrance is clearly defined and visible from the main road. There may be minor issues, like a shared driveway with one other low-traffic business or a location on a busy street that can make turning difficult during peak hours.

    9-10 (Excellent Accessibility): Access is exceptionally easy and intuitive. The site features a dedicated, highly visible entrance/exit directly off a primary road, ideally with full access from both directions (no restrictive medians).

Required Output Format:
Your final output must be a single, valid JSON object following this exact structure:

{
  "accessibility_score": <a single integer from 1 to 10>,
  "rationale": "A detailed summary of your analysis, explaining how you arrived at the score by evaluating the primary road access, entrance/exit complexity, and surrounding environment."
}
"""
    user_query_prompt = "Analyze the provided images for the car wash location and determine the accessibility score based on the criteria."

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
                validated_output = AccessibilityScore(**parsed_output)
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
