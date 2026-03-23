import textwrap
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List
from type_car_wash.config import AZURE_OPENAI_API_KEY

# Expanded classification types
VALID_TYPES = [
    "Express Tunnel",
    "Full Service",
    "In-Bay Automatic",
    "Self-Serve",
    "Touchless",
    "Hand Wash / Detail",
    "Flex",
    "Mobile",
    "Unknown"
]


class CarWashClassification(BaseModel):

    primary_type: str = Field(
        description="Primary classification type"
    )

    secondary_types: List[str] = Field(
        default_factory=list,
        description="Additional detected types"
    )

    confidence_score: float = Field(
        description="Confidence score between 0 and 1"
    )

    found_packages: List[str] = Field(
        description="Wash packages detected"
    )

    detected_markers: List[str] = Field(
        description="Keywords or infrastructure markers detected"
    )

    reasoning: str = Field(
        description="Explanation"
    )


def classify_car_wash_with_ai(scraped_text: str) -> Optional[CarWashClassification]:

    if not scraped_text:
        return None

    print("\n[*] Classifying with OpenAI (gpt-4o-mini)...")

    endpoint = "https://soneastus2proformaai-dev.openai.azure.com/openai/v1"
    deployment_name = "gpt-4o-mini"
    api_key = AZURE_OPENAI_API_KEY

    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )

    truncated = scraped_text[:100000]

    prompt = f"""
You are a world-class car wash classification AI used in a production data pipeline.

Analyze the following scraped website text and classify the car wash.

---

CLASSIFICATION TYPES:

Express Tunnel:
- conveyor
- tunnel
- express wash
- unlimited membership
- stay in vehicle

Full Service:
- interior cleaning
- vacuum included
- hand dry
- staff cleaning

In-Bay Automatic:
- pull into bay
- automatic wash bay
- touchless automatic
- rollover wash

Self-Serve:
- self serve
- coin operated
- wash bays
- manual spray wand

Touchless:
- touchless wash
- no brushes
- brushless

Hand Wash / Detail:
- hand wash
- detailing
- ceramic coating
- paint correction

Flex:
- express exterior + full service option

Mobile:
- mobile wash
- we come to you

---

TASK:

Return structured JSON containing:

1 primary_type
2 secondary_types
3 confidence_score (0 to 1)
4 found_packages
5 detected_markers
6 reasoning

---

SCRAPED TEXT:
{truncated}

Only return valid JSON string mapping exactly to the fields above.
"""

    try:

        completion = client.chat.completions.create(
            model=deployment_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        response_text = completion.choices[0].message.content
        if not response_text:
            return None

        import json
        return CarWashClassification.model_validate_json(response_text)

    except Exception as e:

        print("OpenAI error:", e)
        return None