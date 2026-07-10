import os
from dotenv import load_dotenv

# Load environment variables from the .env file at the repo root (two levels up:
# libs/carwash_type/ -> libs/ -> root). load_dotenv() no-ops silently on a bad path,
# so a wrong depth here surfaces only as the ValueError below.
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

JINA_API_KEY = os.getenv("JINA_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

if not AZURE_OPENAI_API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set. Please check your .env file.")
