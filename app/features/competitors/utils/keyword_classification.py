import json
import time
from app.utils.llm import local_llm as llm


CLASSIFIER_PROMPT = """You are a smart classifier for car wash businesses. Your goal is to decide whether a business is a Competitor or Not a Competitor based on its name or description.

Classification Logic:

Competitor: Businesses that emphasize automated, full-service, or drive-through-style washes, typically using words like:

"Express", "Xpress", "Flex Serve", "Quick Wash", "Tunnel", "Exterior",  etc.

You may encounter variations, misspellings, or synonyms indicating fast, full-service, or automated experiences.

Not a Competitor: Businesses that emphasize manual, customer-operated, or value-added services, typically using terms like:

"Self Serve", "Full Serve", "Hand Wash", "Mobile", "Truck Wash", "Blue Beacon", "Window Tinting" "Detailing", "Oil Change", etc.

These are more traditional or niche service providers, not direct competitors.

Can't Say: If a car wash name is generic and cannot be identified using above criteria or it has the following keywords:
 "Lube", "Auto", etc.

⚠️ Important: If the input contains both types of keywords, default to Competitor — automation usually implies higher overlap.

Examples:
Input: "Drive-Thru Express Wash"
Output: Competitor

Input: "Eco Hand Wash & Detail"
Output: Not a Competitor

Input: "Quick Lube"
Output: Can't Say

Input: "Flex Serve Tunnel Wash and Lube"
Output: Competitor

Input: "Self Serve Car Wash and Oil Center"
Output: Not a Competitor

Input: "Speedy Xpress Car Wash"
Output: Competitor

Input: "Downtown Detail & Hand Wash"
Output: Not a Competitor

Respond with a single JSON object only, no other text, with keys "classification" (one of: "Competitor", "Not a Competitor", "Can't say") and "explanation" (brief explanation mentioning keywords found).

Now classify this input:
""" + "{{input}}"


def keywordclassifier(car_wash_name: str):
    prompt = CLASSIFIER_PROMPT.replace("{{input}}", str(car_wash_name))
    full_response_content = ""

    for attempt in range(3):
        try:
            response = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.3)
            full_response_content = response.get("generated_text", "")
            if full_response_content:
                break
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"classification": "Error", "explanation": str(e)}

    if not full_response_content:
        return {"classification": "Error", "explanation": "Empty LLM response"}

    # Extract JSON from response (handle markdown code blocks if present)
    text = full_response_content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[-1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[-1].split("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Raw response content: {full_response_content}")
        return {"classification": "Error", "explanation": f"JSON decoding error: {e}"}
