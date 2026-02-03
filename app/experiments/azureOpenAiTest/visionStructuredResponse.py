import os
import base64
import requests # For all Google API calls
# from io import BytesIO # Not strictly needed if we get bytes directly

import openai # For OpenAI exceptions
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any, Tuple, Optional

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://soneastus2proformaai.openai.azure.com")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "5sPwzEUN6KlaevseHDUJ4CAt733wG7bJUuSTpssVV9GtB5Lyq7QKJQQJ99BDACHYHv6XJ3w3AAABACOGbWjz") # Your Azure Key
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "gpt-4o")

# --- Google API Configuration (Hardcoded as requested for testing) ---
GOOGLE_PLACES_API_KEY = "AIzaSyCXxpPx_liQXml0e6Wc0v4Zg_uvEhlOTcA"
GOOGLE_STATIC_MAPS_API_KEY = "AIzaSyCHIa_N__Q6wOe8LlLaJdArlqM8_HfedQg"

# --- Google API Endpoints ---
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACE_PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"
STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"

# --- NEW Pydantic Model for Car Wash Classification ---
class CarWashClassification(BaseModel):
    Classification: Literal["Competitor", "Not a Competitor"] = Field(
        description="Classification of the car wash location."
    )
    Justification: str = Field(
        description="Detailed justification for the classification, mentioning visible features, adherence to criteria, or missing/ambiguous data."
    )

# --- Helper Function to Encode Image BYTES to Base64 Data URL ---
def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = 'image/jpeg') -> str:
    """Encodes image bytes into a base64 data URL."""
    base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# --- Google API Fetching Functions (largely unchanged from previous version) ---

def get_place_details_for_images(place_id: str, api_key: str) -> Tuple[Optional[List[str]], Optional[Dict[str, float]], Optional[str]]:
    if not place_id:
        return None, None, None
    params = {
        "place_id": place_id,
        "fields": "photo,geometry/location,name",
        "key": api_key
    }
    try:
        response = requests.get(PLACE_DETAILS_URL, params=params)
        response.raise_for_status()
        result = response.json()
        if result.get("status") == "OK" and "result" in result:
            place_data = result["result"]
            place_name = place_data.get("name")
            photo_refs = [photo["photo_reference"] for photo in place_data.get("photos", [])]
            location = place_data.get("geometry", {}).get("location")
            print(f"Found {len(photo_refs)} photo references and location for '{place_name or place_id}'.")
            return photo_refs, location, place_name
        else:
            print(f"Error fetching details for Place ID {place_id}: {result.get('status')}, {result.get('error_message', '')}")
            return None, None, None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for Place Details (ID: {place_id}): {e}")
        return None, None, None
    except ValueError as e: # JSONDecodeError
        print(f"Failed to decode JSON for Place Details (ID: {place_id}): {e}")
        return None, None, None


def fetch_place_photo_bytes(photo_reference: str, api_key: str, max_width: int = 800) -> Optional[bytes]:
    if not photo_reference:
        return None
    params = {
        "photoreference": photo_reference,
        "maxwidth": max_width,
        "key": api_key
    }
    try:
        response = requests.get(PLACE_PHOTO_URL, params=params, stream=True)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Google Place Photo (ref: {photo_reference[:10]}...): {e}")
        return None


def fetch_static_map_bytes(lat: float, lng: float, api_key: str,
                           size: str = "640x640", zoom: int = 20, maptype: str = "hybrid") -> Optional[bytes]:
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": size,
        "maptype": maptype,
        "key": api_key
    }
    try:
        response = requests.get(STATIC_MAP_URL, params=params)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Google Static Map for {lat},{lng}: {e}")
        if response is not None:
            print(f"Static map response content: {response.text[:200]}")
        return None

# --- Main Function to Process Images from Google Place ID ---
def classify_car_wash_from_google_place_id(place_id: str, user_query_prompt: str) -> Optional[CarWashClassification]:
    all_image_content_parts = []

    # 1. Fetch Place Details
    photo_references, location_data, place_name = get_place_details_for_images(place_id, GOOGLE_PLACES_API_KEY)

    # 2. Fetch Satellite Image
    if location_data and 'lat' in location_data and 'lng' in location_data:
        lat, lng = location_data['lat'], location_data['lng']
        print(f"Fetching satellite image for {place_name or place_id} at ({lat}, {lng})...")
        satellite_image_bytes = fetch_static_map_bytes(lat, lng, GOOGLE_STATIC_MAPS_API_KEY)
        if satellite_image_bytes:
            data_url = image_bytes_to_data_url(satellite_image_bytes, mime_type='image/jpeg')
            all_image_content_parts.append({
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"} # Use high detail for satellite if important
            })
            print("Satellite image added.")
        else:
            print("Could not fetch satellite image.")
    else:
        print(f"No location data for {place_name or place_id}, skipping satellite image.")

    # 3. Fetch Place Photos
    if photo_references:
        num_photos_to_fetch = min(len(photo_references), 9 if all_image_content_parts else 10)
        print(f"Fetching up to {num_photos_to_fetch} place photos for {place_name or place_id}...")
        
        for i, ref in enumerate(photo_references[:num_photos_to_fetch]):
            photo_bytes = fetch_place_photo_bytes(ref, GOOGLE_PLACES_API_KEY)
            if photo_bytes:
                data_url = image_bytes_to_data_url(photo_bytes, mime_type='image/jpeg')
                all_image_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"} # Use high detail for place photos
                })
                print(f"Place photo {i+1} added.")
            else:
                print(f"Could not fetch place photo {i+1} (ref: {ref[:10]}...).")
    else:
        print(f"No photo references found for {place_name or place_id}.")


    if not all_image_content_parts:
        print("No images could be fetched from Google. Cannot proceed with classification.")
        return None

    print(f"Total images prepared for OpenAI: {len(all_image_content_parts)}")

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
        "Classification": "Competitor" | "Not a Competitor",
        "Justification": "Detailed explanation covering: - Visible features supporting your classification (tunnel presence/type, entrance/exit, equipment, signage, conveyor, vacuum area if relevant). - How the location meets or fails the criteria."
        }
        """
    # 4. Call Azure OpenAI
    try:
        client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
        )

        messages = [
        {"role": "system", "content": system_prompt}, # <<< THIS IS THE CRUCIAL FIX
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query_prompt}
            ] + all_image_content_parts
        }
    ]
    
        print(f"\nSending request to Azure OpenAI with {len(all_image_content_parts)} image(s)...")
        # ... rest of the try-except block ...
        completion = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,
            messages=messages,
            response_format=CarWashClassification, # Use the new Pydantic model
            max_tokens=1500, 
            # temperature=0.1 
        )
        
        parsed_output: CarWashClassification = completion.choices[0].message.parsed
        return parsed_output
    
    except openai.APIConnectionError as e:
        print(f"The server could not be reached: {e.__cause__}")
    except openai.RateLimitError as e:
        print(f"A 429 status code was received; we should back off a bit: {e}")
    except openai.APIStatusError as e:
        print(f"Another non-200-range status code was received: {e.status_code}")
        print(e.response)
    except openai.BadRequestError as e:
        print(f"Bad request (e.g. invalid schema, model output issue, model does not support structured outputs, invalid inputs): {e}")
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI call: {e}")
    return None
    
if __name__ == "__main__":
    test_place_id = "ChIJIeC83tSwToYRSK-jop82xCs" 

    # New user prompt for the task
    user_prompt_for_ai = f"Analyze the provided images for a location and determine if it is an Express Tunnel Car Wash competitor based on the criteria."


    result = classify_car_wash_from_google_place_id(test_place_id, user_prompt_for_ai)

    if result:
        print("\n--- Car Wash Classification Result ---")
        print(f"Classification: {result.Classification}")
        print(f"Justification:\n{result.Justification}")
        print("\n--- Raw Pydantic Model ---")
        print(result.model_dump_json(indent=2))
    else:
        print("\nClassification failed.")