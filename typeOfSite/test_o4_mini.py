import os
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_MODEL_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME", "o4-mini")


if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_MODEL_DEPLOYMENT_NAME]):
    print("Error: Azure OpenAI environment variables are not fully configured.")
else:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,
            messages=[
                {"role": "user", "content": "how many prime numbers are there in first 100 natural numbers?"},
            ],
            max_completion_tokens=500,
            reasoning_effort="high"
        )
        
        # print("\nSuccess! Received response:")
        # print(response.model_dump_json(indent=2))
        if response.choices:
            print("\n--- Text Response ---")
            print(response.choices[0].message.content)
            print("--------------------")

    except openai.APIConnectionError as e:
        print(f"\nError: The server could not be reached: {e.__cause__}")
    except openai.RateLimitError as e:
        print(f"\nError: A 429 status code was received; we should back off a bit: {e}")
    except openai.APIStatusError as e:
        print(f"\nError: Another non-200-range status code was received: {e.status_code}")
        print(f"Response: {e.response}")
    except openai.BadRequestError as e:
        print(f"\nError: Bad request: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
