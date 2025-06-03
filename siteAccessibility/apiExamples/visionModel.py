import os
import mimetypes
import json
from google import genai
from google.genai import types # Ensure types is imported directly

# --- Helper function to determine MIME type ---
def get_mime_type(file_path):
    """Determines the MIME type of a file based on its extension."""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
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
            return 'application/octet-stream'
    return mime_type

# --- Main function to get site accessibility score ---
def visionModelResponse(satellite_image_path: str) -> dict:
    """
    Analyzes a single satellite image of a car wash location to determine
    its site accessibility score (1-10) and provides a justification.

    Args:
        satellite_image_path: Path to a single satellite image of the car wash location.

    Returns:
        A dictionary containing the accessibility score and justification from the vision model,
        or an error dictionary if issues occur.
    """
    api_key = "AIzaSyDMRaUltpo6pBXoSVs46L51Js70pLAketo" # Replace with your actual API key or use environment variables
    if not api_key:
        return {"error": "GEMINI_API_KEY not found. Please set it as an environment variable or in the script."}

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to initialize Gemini Client: {e}"}

    # Use the same model name string as in your working example and logs
    model = "gemini-2.5-pro-preview-05-06"

    # --- Prepare the satellite image part ---
    image_parts = []
    print(f"Loading satellite image from: {satellite_image_path}")
    try:
        if not os.path.isfile(satellite_image_path):
            return {"error": f"Satellite image file not found at '{satellite_image_path}'. Please check the path."}

        with open(satellite_image_path, 'rb') as f:
            sat_img_bytes = f.read()
        mime_type = get_mime_type(satellite_image_path)
        if not mime_type.startswith("image/"):
             return {"error": f"File '{satellite_image_path}' is not a recognized image type (MIME: {mime_type})."}

        image_parts.append(
            types.Part.from_bytes(
                data=sat_img_bytes,
                mime_type=mime_type
            )
        )
        print(f"  - Loaded: {os.path.basename(satellite_image_path)} (MIME: {mime_type})")
    except Exception as e:
        return {"error": f"Error reading satellite image '{satellite_image_path}': {e}"}

    if not image_parts:
        return {"error": "No valid satellite image was processed."}

    # --- Construct the text prompt for Site Accessibility ---
    site_accessibility_prompt = """
You are an expert site selection analyst specializing in car wash businesses in the United States. Your task is to evaluate the **vehicular accessibility** of the car wash location shown in the provided satellite image and assign it a score from 1 to 10.

**Scoring Scale:**
*   **1:** Extremely Poor Accessibility (Very difficult to enter/exit, dangerous, hidden, extremely inconvenient)
*   **5:** Average Accessibility (Functional, but with some noticeable drawbacks or no significant advantages)
*   **10:** Excellent Accessibility (Easy, safe, and convenient multi-directional access, highly visible, ample maneuvering space)

**Based solely on the visual information in the satellite image, consider the following factors:**

1.  **Road Network & Ingress/Egress:**
    *   **Proximity and Type of Main Roads:** Is it directly on a major artery, a busy secondary road, or a quiet side street?
    *   **Number and Clarity of Entry/Exit Points:** Are there multiple, clearly defined access points? Is access one-way or two-way?
    *   **Ease of Turning:** How easy is it for vehicles to turn into and out of the site from likely directions of travel? Consider medians, turn lanes (if visible), and proximity to intersections.
    *   **Potential Traffic Conflicts:** Does the access seem to create conflicts with through-traffic, neighboring businesses, or pedestrian walkways?

2.  **On-Site Circulation & Layout:**
    *   **Maneuvering Space:** Is there adequate space for cars to queue without blocking public roads or internal pathways?
    *   **Internal Flow:** Does the layout appear logical for navigating to the wash entrance, pay stations (if discernible), and vacuum/finishing areas?
    *   **Parking/Waiting Areas (if applicable):** If self-serve bays or detailing areas are present, is there sufficient space?

3.  **Visibility & Prominence:**
    *   **Setback from Road:** Is the car wash structure highly visible from the primary access roads, or is it set back and potentially obscured?
    *   **Signage Potential (Inferred):** Does the location offer good opportunities for prominent signage visible to passing traffic (though the sign itself may not be clear)?

4.  **Surrounding Environment (as it pertains to access):**
    *   **Density:** Is it in a sparse, suburban, or dense urban/commercial area? This can influence traffic volume and complexity.
    *   **Obstructions:** Are there any significant physical obstructions (e.g., large trees directly at an entrance, awkward building placements nearby) that hinder access?
    *   **Proximity to Traffic Generators:** Being near other high-traffic retail or services can be good, but can also complicate access if not managed well.

**Output Requirements:**
1.  **Accessibility Score:** A single integer number from 1 to 10.
2.  **Justification (2-4 bullet points):** Briefly explain the key positive and negative factors observed in the image that led to your score. Focus on aspects directly impacting how easily a customer can drive into, use, and exit the car wash.

**Important Considerations:**
*   Focus *only* on what can be reasonably inferred from the satellite image.
*   Do not consider factors not visible (e.g., specific speed limits, time-of-day traffic patterns unless clearly indicated by road design, local zoning, or specific business performance).
*   Assume typical US driving behavior and vehicle sizes.

**Provide your evaluation for the car wash centered in this image.**
"""

    contents = [
        types.Part.from_text(text=site_accessibility_prompt)
    ] + image_parts

    # --- Define the structured output schema ---
    # This object defines the desired JSON structure and types for the response.
    generate_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=types.Schema(
            type=types.Type.OBJECT,
            required=["accessibility_score", "justification"],
            properties={
                "accessibility_score": types.Schema(
                    type=types.Type.INTEGER,
                    description="The site accessibility score from 1 (worst) to 10 (best)."
                ),
                "justification": types.Schema(
                    type=types.Type.STRING,
                    description="A brief explanation (2-4 bullet points) of the factors leading to the score, based on the satellite image."
                ),
            },
        ),
    )

    # --- Call the Generative Model ---
    print(f"\nSending request to Gemini API (model: {model}) with 1 satellite image part...")
    try:
        # Ensure the API call uses 'model' for the model name string,
        # and 'config' for the GenerateContentConfig object.
        response = client.models.generate_content(
            model=model,            # The model name string (e.g., "gemini-1.5-pro-preview-05-06")
            contents=contents,
            config=generate_config  # The GenerateContentConfig object passed as 'config'
        )

        if response is None or not hasattr(response, 'text') or not response.text:
            error_detail = "No response text."
            if response and hasattr(response, 'candidates') and response.candidates:
                error_detail = f"Reason: {response.candidates[0].finish_reason}. Safety Ratings: {response.candidates[0].safety_ratings}"
            return {"error": f"Gemini API response was empty or invalid. {error_detail}", "raw_response": str(response)}

        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response from model: {e}", "raw_response": response.text}

    except Exception as e:
        print(f"DEBUG: Exception type: {type(e).__name__}, Message: {str(e)}")
        if hasattr(e, 'message'):
             error_message = e.message
        else:
             error_message = str(e)
        return {"error": f"Error calling Gemini API: {error_message}", "exception_type": type(e).__name__}


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Use the path from your execution log
    example_satellite_image_path = "satellite_images/ StoneWash  Car Care Center  818 Paris Rd  Mayfield  KY  42066_3.png"

    print(f"\n--- Attempting to run visionModelResponse with satellite image: {example_satellite_image_path} ---")
    print("Please ensure you have:")
    print(f"1. Saved a satellite image to the path: '{example_satellite_image_path}'")
    print("2. Set your GEMINI_API_KEY in the script or as an environment variable.")

    if not os.path.exists(example_satellite_image_path):
        print(f"\nWARNING: The example image '{example_satellite_image_path}' does not exist.")
        print("Please ensure the path is correct.")
        print("Skipping API call without an image.")
    else:
        response_dict = visionModelResponse(
            satellite_image_path=example_satellite_image_path
        )
        print("\n--- Model Response (Structured JSON) ---")
        if response_dict:
            print(json.dumps(response_dict, indent=2))
        else:
            print("No response received or an error occurred.")