import requests
import traceback
from typing import Any, Dict
from agentic_pipeline.config.settings import LOCAL_LLM_URL, LOCAL_LLM_API_KEY

def get_llm_response(user_content: str, reasoning_effort: str = "medium", temperature: float = 0.5, developer_prompt: str = "") -> Dict[str, Any]:
    headers = {"x-api-key": LOCAL_LLM_API_KEY}
    user_message = {"role": "user", "content": user_content}
    if developer_prompt != "":
        user_message.update({"developer_prompt": developer_prompt})
    payload = {
        "messages": [
            user_message,
        ],
        "reasoning": reasoning_effort,
        "max_new_tokens": 16384,
        "temperature": temperature
    }
    try:
        response = requests.post(LOCAL_LLM_URL, json=payload, headers=headers, timeout=700)
        response.raise_for_status()
        response_dict = response.json()
        return response_dict
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again later.")
    except requests.exceptions.RequestException as e:
        print(f"Error with traceback-> {traceback.format_exc()}")
        raise Exception(f"Error communicating with LLM server: {e}")
