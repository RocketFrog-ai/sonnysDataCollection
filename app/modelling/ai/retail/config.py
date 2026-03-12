"""
Retail narrative config: anchor retailer types, v3 feature keys, direction for summaries.
"""

# metric_key -> (display_name, subtitle, v3 feature_analysis key)
RETAIL_NARRATIVE_METRICS = {
    "costco-distance": (
        "Warehouse Club Distance",
        "Distance to Nearest Costco / Sam's Club",
        "costco_enc",
    ),
    "walmart-distance": (
        "Big Box Distance",
        "Distance to Nearest Walmart",
        "distance_nearest_walmart(5 mile)",
    ),
    "target-distance": (
        "Discount Retail Distance",
        "Distance to Nearest Target",
        "distance_nearest_target (5 mile)",
    ),
    "grocery-count": (
        "Grocery Anchors Within 1 Mile",
        "Grocery stores within 1 mile",
        "other_grocery_count_1mile",
    ),
    "food-joint-count": (
        "Food & Beverage Anchors Within 0.5 Miles",
        "Restaurants / QSR within 0.5 miles",
        "count_food_joints_0_5miles (0.5 mile)",
    ),
}

RETAIL_METRIC_KEYS_ORDER = [
    "costco-distance",
    "walmart-distance",
    "target-distance",
    "grocery-count",
    "food-joint-count",
]

RETAIL_METRIC_UNITS = {
    "costco-distance": "miles",
    "walmart-distance": "miles",
    "target-distance": "miles",
    "grocery-count": "stores",
    "food-joint-count": "places",
}

# "higher" = higher value is better for wash demand; "lower" = closer is better
RETAIL_METRIC_DIRECTION = {
    "costco-distance": "lower",
    "walmart-distance": "lower",
    "target-distance": "lower",
    "grocery-count": "higher",
    "food-joint-count": "higher",
}

RETAIL_IMPACT_CLASSIFICATION_SUFFIX = {
    "costco-distance": "miles",
    "walmart-distance": "miles",
    "target-distance": "miles",
    "grocery-count": "stores",
    "food-joint-count": "places",
}

# Anchor retail type by keyword in name (checked lower-case)
ANCHOR_TYPE_BY_KEYWORD = {
    "costco": "Warehouse Club",
    "sam's club": "Warehouse Club",
    "bj's": "Warehouse Club",
    "walmart": "Supercenter",
    "target": "Big Box / Discount",
    "meijer": "Big Box",
    "kohl's": "Big Box",
    "home depot": "Home Improvement",
    "lowe's": "Home Improvement",
    "kroger": "Grocery",
    "publix": "Grocery",
    "safeway": "Grocery",
    "whole foods": "Grocery",
    "aldi": "Grocery",
    "trader joe's": "Grocery",
    "h-e-b": "Grocery",
    "mcdonald's": "Food & Beverage",
    "chick-fil-a": "Food & Beverage",
    "starbucks": "Food & Beverage",
    "dunkin": "Food & Beverage",
    "chipotle": "Food & Beverage",
    "panera": "Food & Beverage",
    "taco bell": "Food & Beverage",
    "burger king": "Food & Beverage",
    "wendy's": "Food & Beverage",
}

ANCHOR_CATEGORY_TYPE = {
    "Grocery": "Grocery Anchor",
    "Food Joint": "Food & Beverage",
}
