"""
Competition narrative config: display names, v3 feature keys, direction for summaries.
"""

# metric_key -> (display_name, subtitle, v3 feature_analysis key)
COMPETITION_NARRATIVE_METRICS = {
    "same-format-count": (
        "Nearby Same-format Car Washes",
        "Within 4 miles",
        "competitors_count_4miles",
    ),
    "distance-to-nearest": (
        "Distance to Nearest Same-format",
        "miles",
        "competitor_1_distance_miles",
    ),
    "nearest-google-rating": (
        "Nearest Competitor Google Rating",
        "rating",
        "competitor_1_google_rating",
    ),
    "nearest-review-count": (
        "Nearest Competitor Review Count",
        "reviews",
        "competitor_1_rating_count",
    ),
    "nearest-brand-strength": (
        "Nearest Same-format Brand Strength",
        "Rating and reviews",
        "competition_quality",
    ),
}

COMPETITION_METRIC_KEYS_ORDER = [
    "same-format-count",
    "distance-to-nearest",
    "nearest-google-rating",
    "nearest-review-count",
    "nearest-brand-strength",
]

COMPETITION_METRIC_UNITS = {
    "same-format-count": "competitors",
    "distance-to-nearest": "miles",
    "nearest-google-rating": "stars",
    "nearest-review-count": "reviews",
    "nearest-brand-strength": "—",
}

# higher = better for wash demand; lower = better (e.g. distance)
COMPETITION_METRIC_DIRECTION = {
    "same-format-count": "neutral",
    "distance-to-nearest": "lower",
    "nearest-google-rating": "higher",
    "nearest-review-count": "higher",
    "nearest-brand-strength": "higher",
}

COMPETITION_IMPACT_CLASSIFICATION_SUFFIX = {
    "same-format-count": "competitors",
    "distance-to-nearest": "miles",
    "nearest-google-rating": "stars",
    "nearest-review-count": "reviews",
    "nearest-brand-strength": "—",
}
