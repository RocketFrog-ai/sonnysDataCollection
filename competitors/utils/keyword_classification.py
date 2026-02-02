import base64
import os
from google import genai
from google.genai import types
import json
from dotenv import load_dotenv
import traceback
import time

load_dotenv()

def keywordclassifier(car_wash_name: str):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-3-flash-preview" # Consider 'gemini-1.5-flash' or 'gemini-1.5-pro' for more complex classification or longer inputs if needed.
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""You are a smart classifier for car wash businesses. Your goal is to decide whether a business is a Competitor or Not a Competitor based on its name or description.

Classification Logic:

Competitor: Businesses that emphasize automated, full-service, or drive-through-style washes, typically using words like:

"Express", "Xpress", "Flex Serve", "Quick Wash", "Tunnel", “Exterior”,  etc.

You may encounter variations, misspellings, or synonyms indicating fast, full-service, or automated experiences.

Not a Competitor: Businesses that emphasize manual, customer-operated, or value-added services, typically using terms like:

"Self Serve", “Full Serve”, "Hand Wash", "Mobile", “Truck Wash”, “Blue Beacon”, “Window Tinting” "Detailing", "Oil Change", etc.

These are more traditional or niche service providers, not direct competitors.

Can’t Say: If a car wash name is generic and cannot be identified using above criteria or it has the following keywords:
 “Lube”, “Auto”, etc.



⚠️ Important: If the input contains both types of keywords, default to Competitor — automation usually implies higher overlap.

Examples:
Input: "Drive-Thru Express Wash"
Output: Competitor
                                     
Input: "Eco Hand Wash & Detail"
Output: Not a Competitor

Input: "Quick Lube"
Output: Can’t Say

Input: "Flex Serve Tunnel Wash and Lube"
Output: Competitor
                                     
Input: "Self Serve Car Wash and Oil Center"
Output: Not a Competitor
                                     
Input: "Speedy Xpress Car Wash"
Output: Competitor
                                     
Input: "Downtown Detail & Hand Wash"
Output: Not a Competitor


Now classify this input:
{{""" + str(car_wash_name) + """}}"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["classification", "explanation"],
            properties = {
                "classification": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    enum = ["Competitor", "Not a Competitor", "Can't say"], # Added "Can't say"
                    description = "The classification of the car wash business.",
                ),
                "explanation": genai.types.Schema(
                    type = genai.types.Type.STRING,
                    description = "A brief explanation for the classification, mentioning the keywords found.",
                ),
            },
        ),
    )

    full_response_content = ""
    for attempt in range(3):
        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk and chunk.text:
                    full_response_content += chunk.text
            break  # Exit loop if successful
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Final attempt failed. Moving on.")
                return {"classification": "Error", "explanation": f"Streaming error: {e}"}
    
    # Assuming the response is a single JSON object
    try:
        return json.loads(full_response_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini API response: {e}")
        print(f"Raw response content: {full_response_content}")
        return {"classification": "Error", "explanation": f"JSON decoding error: {e}"}
