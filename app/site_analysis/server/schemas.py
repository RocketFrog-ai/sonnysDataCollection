from pydantic import BaseModel, Field
from typing import Optional


class AnalyseRequest(BaseModel):
    """An address to geocode, for /traffic-lights and /nearby-stores.

    Named for the removed POST /analyze-site endpoint it used to kick off. Kept as-is rather than
    renamed, because renaming a pydantic model changes its OpenAPI component title -- a visible
    change for any client generating types from /openapi.json.
    """
    address: str = Field(..., description="Site address to geocode and fetch nearby data for.")


class SiteContextRequest(BaseModel):
    """Synchronous lat/lon site-analysis (the shared map pin). Provide latitude+longitude OR an address.

    Returns weather / competitors / retail anchors / gas stations + map markers + rule-based insights in ONE
    response (no task polling) — the lat/lon counterpart to the async /analyze-site pipeline."""
    address: Optional[str] = Field(None, description="Site address. Optional if latitude/longitude provided.")
    latitude: Optional[float] = Field(None, description="Pin latitude.")
    longitude: Optional[float] = Field(None, description="Pin longitude.")
    include_ai: bool = Field(False, description="Also rewrite each dimension's insight with the internal LLM (skipped if unreachable).")
    demo: bool = Field(False, description="Anonymized demo: hide the origin address on the markers.")


class SiteFeaturesRequest(BaseModel):
    """Nearest-site feature lookup over the precomputed dataset. Provide a lat/lon pin; the closest
    site's grouped features are returned (no external calls, no competitor info)."""
    latitude: float = Field(..., description="Pin latitude.")
    longitude: float = Field(..., description="Pin longitude.")
