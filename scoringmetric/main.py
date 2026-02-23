# main.py

import pandas as pd
from feature_config import FEATURE_DIRECTION
from percentile_engine import PercentileEngine
from feature_analyzer import FeatureAnalyzer
from site_scorer import SiteScorer

# Load historical data (NO target column)
df = pd.read_excel("dataSET (1).xlsx")

engine = PercentileEngine(df, FEATURE_DIRECTION)
analyzer = FeatureAnalyzer(engine, df)
scorer = SiteScorer(analyzer)

# New site
site = {
    "total_sunshine_hours": 3200,
    "Nearest StreetLight US Hourly-Ttl AADT": 12500,
    "distance_from_nearest_costco": 2.4,
    "competitors_count": 4,
    "rainy_days": 110,
}

result = scorer.score_site(site)

print("Overall Score:", result["overall_score"])
print("\nTop Strengths:")
for f in result["strengths"]:
    print(f)

print("\nTop Weaknesses:")
for f in result["weaknesses"]:
    print(f)
