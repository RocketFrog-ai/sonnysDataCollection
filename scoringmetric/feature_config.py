# feature_config.py

FEATURE_DIRECTION = {
    # Higher is better
    "total_sunshine_hours": "higher",
    "Nearest StreetLight US Hourly-Ttl AADT": "higher",
    "Sum ChainXY": "higher",
    "total_weekly_operational_hours": "higher",

    # Lower is better
    "distance_from_nearest_costco": "lower",
    "distance_from_nearest_walmart": "lower",
    "competitors_count": "lower",
    "rainy_days": "lower",
}

# Explicitly mark ambiguous features (optional)
AMBIGUOUS_FEATURES = {
    "competitor_distance": "context_dependent"
}
