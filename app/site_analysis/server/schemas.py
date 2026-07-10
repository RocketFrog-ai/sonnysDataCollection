from pydantic import BaseModel, Field
from typing import Optional


class AddressRequest(BaseModel):
    """An address to geocode. Used by POST /traffic-lights and POST /nearby-stores."""
    address: str = Field(..., description="Site address to geocode and fetch nearby data for.")


class SiteContextRequest(BaseModel):
    """Synchronous lat/lon site-analysis (the shared map pin). Provide latitude+longitude OR an address.

    Returns weather / competitors / retail anchors / gas stations + map markers + rule-based insights in ONE
    response."""
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
