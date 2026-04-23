import asyncio
import os
import json
from dotenv import load_dotenv

from app.site_analysis.server.routes import get_competitors_dynamics_endpoint
from app.site_analysis.server.models import CompetitorsDynamicsRequest

load_dotenv()

print("Testing direct dynamics endpoint with integrated classification...")

req = CompetitorsDynamicsRequest(
    address="36 Battery St, San Francisco, CA 94111",
    radius_miles=1.0,  # Keep it exceptionally small for the test to limit targets
    fetch_place_details=True
)

try:
    results = get_competitors_dynamics_endpoint(req)
    print("\nResults:")
    print(json.dumps(results, indent=2))
except Exception as e:
    print(f"Error during execution: {e}")
