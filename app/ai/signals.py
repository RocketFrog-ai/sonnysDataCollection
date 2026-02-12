CUTOFF = 0.5
CORR_FLOOR = 0.04

POSITIVE_SIGNALS = {
    'sunny_days_per_year': {'score': 0.8523, 'corr': 0.1442, 'description': 'Sunny days per year'},
    'total_precipitation_mm': {'score': 0.7033, 'corr': 0.0441, 'description': 'Total precipitation'},
    'days_pleasant_temp': {'score': 0.5601, 'corr': 0.1307, 'description': 'Days with pleasant temperature'},
    'nearby_traffic_lights_count': {'score': 0.4658, 'corr': 0.1372, 'description': 'Count of nearby traffic lights'},
    'rainy_days': {'score': 0.3574, 'corr': 0.0399, 'description': 'Number of rainy days'},
    'count_of_costco_5miles': {'score': 0.2188, 'corr': 0.1097, 'description': 'Costco stores within 5 miles'},
    'count_of_walmart_5miles': {'score': 0.2188, 'corr': 0.1097, 'description': 'Walmart stores within 5 miles'},
    'count_of_gas_stations_5miles': {'score': 0.2188, 'corr': 0.1097, 'description': 'Gas stations within 2.5 miles'},
    'competitor_1_google_user_rating_count': {'score': 0.2182, 'corr': 0.0354, 'description': 'Top competitor rating count'},
    'competitors_count': {'score': 0.0432, 'corr': 0.0257, 'description': 'Total competitors count'},
}

NEGATIVE_SIGNALS = {
    'days_below_freezing': {'score': 0.5977, 'corr': -0.0900, 'description': 'Days below freezing'},
    'distance_nearest_traffic_light_3': {'score': 0.5014, 'corr': -0.0991, 'description': 'Distance to 3rd nearest traffic light'},
    'distance_nearest_traffic_light_9': {'score': 0.4776, 'corr': -0.1540, 'description': 'Distance to 9th nearest traffic light'},
    'distance_from_nearest_costco': {'score': 0.4476, 'corr': -0.1663, 'description': 'Distance from nearest Costco'},
    'distance_nearest_traffic_light_2': {'score': 0.4282, 'corr': -0.1041, 'description': 'Distance to 2nd nearest traffic light'},
    'total_snowfall_cm': {'score': 0.4131, 'corr': -0.0918, 'description': 'Total snowfall'},
    'distance_nearest_traffic_light_7': {'score': 0.4023, 'corr': -0.1268, 'description': 'Distance to 7th nearest traffic light'},
    'distance_from_nearest_walmart': {'score': 0.3863, 'corr': -0.0710, 'description': 'Distance from nearest Walmart'},
    'distance_from_nearest_gas_station': {'score': 0.3863, 'corr': -0.0710, 'description': 'Distance from nearest gas station'},
    'distance_nearest_traffic_light_4': {'score': 0.3714, 'corr': -0.1149, 'description': 'Distance to 4th nearest traffic light'},
    'avg_daily_max_windspeed_ms': {'score': 0.3616, 'corr': -0.0057, 'description': 'Average daily max wind speed'},
}

BANDED_FEATURES = (
    ('sunny_days_per_year', 'Sunshine days/year', '<120 days (-60% to -30%), 120–180 (-20% to +10%), 180–240 (+10% to +35%), 240–300 (+35% to +65%), >300 (+40% to +65%)'),
    ('days_pleasant_temp', 'pleasant days/year', '<60 (-50% to -25%), 60–120 (-20% to +10%), 120–200 (+15% to +45%), 200–260 (+40% to +65%), >260 (+50% to +70%)'),
    ('total_snowfall_cm', 'cm/year', '0 cm (0% to +5%), 1–20 (+5% to +20%), 20–100 (-10% to +15%), 100–250 (-25% to -5%), >250 (-50% to -20%)'),
    ('rainy_days', 'days/year', '<20 (+20% to +40%), 20–60 (+5% to +25%), 60–120 (-10% to +5%), 120–200 (-30% to -10%), >200 (-60% to -30%)'),
    ('days_below_freezing', 'days/year', '0 (+20% to +40%), 1–30 (+5% to +20%), 30–90 (-10% to +10%), 90–150 (-30% to -10%), >150 (-60% to -30%)'),
)

FEATURE_CATEGORY = {
    "weather": [
        "sunny_days_per_year",
        "total_precipitation_mm",
        "days_pleasant_temp",
        "rainy_days",
        "days_below_freezing",
        "total_snowfall_cm",
        "avg_daily_max_windspeed_ms",
    ],
    "traffic": [
        "nearby_traffic_lights_count",
        "distance_nearest_traffic_light_2",
        "distance_nearest_traffic_light_3",
        "distance_nearest_traffic_light_4",
        "distance_nearest_traffic_light_7",
        "distance_nearest_traffic_light_9",
    ],
    "competition": [
        "competitor_1_google_user_rating_count",
        "competitors_count",
    ],
    "site_accessibility": [
        "distance_from_nearest_costco",
        "distance_from_nearest_walmart",
        "distance_from_nearest_gas_station",
        "count_of_costco_5miles",
        "count_of_walmart_5miles",
        "count_of_gas_stations_5miles",
    ],
}

def get_feature_category(feature_name):
    for category, features in FEATURE_CATEGORY.items():
        if feature_name in features:
            return category
    return "site_operations"
