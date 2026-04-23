"""
Gas station narrative config: v3 feature keys, direction for summaries.
"""
from typing import Optional

# metric_key -> (display_name, subtitle, v3 feature_analysis key)
GAS_NARRATIVE_METRICS = {
    "gas-distance": (
        "Distance to Nearest Gas Station",
        "miles",
        "nearest_gas_station_distance_miles",
    ),
    "gas-rating": (
        "Nearest Gas Station Rating",
        "Google rating",
        "nearest_gas_station_rating",
    ),
    "gas-review-count": (
        "Nearest Gas Station Review Count",
        "reviews",
        "nearest_gas_station_rating_count",
    ),
}

GAS_METRIC_KEYS_ORDER = [
    "gas-distance",
    "gas-rating",
    "gas-review-count",
]

GAS_METRIC_UNITS = {
    "gas-distance": "miles",
    "gas-rating": "stars",
    "gas-review-count": "reviews",
}

# lower distance = better; higher rating/count = better
GAS_METRIC_DIRECTION = {
    "gas-distance": "lower",
    "gas-rating": "higher",
    "gas-review-count": "higher",
}

GAS_IMPACT_CLASSIFICATION_SUFFIX = {
    "gas-distance": "miles",
    "gas-rating": "stars",
    "gas-review-count": "reviews",
}

# High-traffic fuel brands (checked lower-case)
HIGH_TRAFFIC_BRANDS = {
    "shell", "chevron", "exxon", "exxonmobil", "bp", "sunoco", "arco",
    "costco", "quiktrip", "wawa", "circle k", "marathon", "valero",
    "speedway", "pilot", "loves", "love's", "kwik trip",
}


def is_high_traffic_brand(name: Optional[str]) -> bool:
    if not name:
        return False
    name_lower = name.lower()
    return any(brand in name_lower for brand in HIGH_TRAFFIC_BRANDS)
