import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "")
