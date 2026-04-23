# Nearby car wash competitors within 4-mile radius (Places API + Distance Matrix).
# get_nearby_competitors returns count and list with name, distance, rating, type (Place Details or classification).

from app.site_analysis.features.active.nearbyCompetitors.get_nearby_competitors import (
    get_nearby_competitors,
    DEFAULT_RADIUS_MILES,
)

__all__ = ["get_nearby_competitors", "DEFAULT_RADIUS_MILES"]
