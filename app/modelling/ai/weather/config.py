"""
Weather narrative config: display names, v3 feature keys, fixed business impact.
Used by weather narrative agents.
"""

# API metric_key -> (display_name, subtitle, v3 feature_analysis key)
# Subtitle = short label for the metric (e.g. "Rainy Days", "Total Annual Snowfall").
WEATHER_NARRATIVE_METRICS = {
    "dirt-trigger-days": (
        "Dirt Trigger Days Window",
        "Rainy Days",
        "weather_rainy_days",
    ),
    "dirt-deposit-severity": (
        "Dirt Deposit Severity",
        "Total Annual Snowfall",
        "weather_total_snowfall_cm",
    ),
    "comfortable-washing-days": (
        "Comfortable Washing Days",
        "Days with 60–80°F temperatures",
        "weather_days_pleasant_temp",
    ),
    "shutdown-risk-days": (
        "Shutdown Risk Days",
        "Days Below Freezing (< 32°F)",
        "weather_days_below_freezing",
    ),
}

# Ordered list for consistent iteration (dirt trigger, deposit, comfortable, shutdown)
WEATHER_METRIC_KEYS_ORDER = [
    "dirt-trigger-days",
    "dirt-deposit-severity",
    "comfortable-washing-days",
    "shutdown-risk-days",
]

# Units for display (metric_key -> unit string)
WEATHER_METRIC_UNITS = {
    "dirt-trigger-days": "days/year",
    "dirt-deposit-severity": "cm snowfall/year",
    "comfortable-washing-days": "days/year",
    "shutdown-risk-days": "days/year",
}

# Fixed business impact per metric (not LLM-generated)
WEATHER_METRIC_BUSINESS_IMPACT = {
    "dirt-trigger-days": "Frequent dirt triggers create strong recurring wash demand.",
    "dirt-deposit-severity": "Some winter dirt accumulation creates occasional seasonal washing demand.",
    "comfortable-washing-days": "Plenty of comfortable weather supports steady customer visits and consistent wash activity.",
    "shutdown-risk-days": "Noticeable winter disruption; wash volumes may drop for short seasonal periods.",
}

# Suffix for impact classification range (metric_key -> e.g. "days" or "cm")
WEATHER_IMPACT_CLASSIFICATION_SUFFIX = {
    "dirt-trigger-days": "days",
    "dirt-deposit-severity": "cm",
    "comfortable-washing-days": "days",
    "shutdown-risk-days": "days",
}

# For summary rationale: "higher" = higher value is better for wash demand; "lower" = lower is better
WEATHER_METRIC_DIRECTION = {
    "dirt-trigger-days": "higher",       # more rainy days → more dirt triggers → more washes
    "dirt-deposit-severity": "lower",    # less snowfall → less shutdown risk
    "comfortable-washing-days": "higher",
    "shutdown-risk-days": "lower",       # fewer freezing days → better
}
