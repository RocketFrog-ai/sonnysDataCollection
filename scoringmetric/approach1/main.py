# main.py

import pandas as pd
import numpy as np
from pathlib import Path

from feature_config import FEATURE_DIRECTION
from percentile_engine import PercentileEngine
from feature_analyzer import FeatureAnalyzer
from site_scorer import SiteScorer
# Load historical data (NO target column)
base = Path(__file__).resolve().parent.parent
data_path = base / "dataSET (1).xlsx"
df = pd.read_excel(data_path, engine="openpyxl")

engine = PercentileEngine(df, FEATURE_DIRECTION)
analyzer = FeatureAnalyzer(engine, df)
scorer = SiteScorer(analyzer)

# New site: use mean (average) values - represents a "typical" site
site = {}
for c in FEATURE_DIRECTION:
    if c in df.columns:
        mean_val = df[c].mean()
        if pd.notna(mean_val):
            site[c] = float(mean_val)

result = scorer.score_site(site)

print("=" * 80)
print("APPROACH 1 - Config-Driven Percentile + Range Engine")
print("=" * 80)
print("""
WHAT THIS APPROACH DOES:
  - Uses a config file (FEATURE_DIRECTION) for 50 features
  - Scores via RAW PERCENTILE RANK (same formula as Approach 2)
  - Range engine derives boundaries: ≤6 unique → discrete; large gaps → split points
  - Ranges are for INTERPRETATION only; they do NOT affect the score

HOW IT DIFFERS FROM APPROACH 2: Same 50 features; has explicit ranges; no plots
HOW IT DIFFERS FROM APPROACH 3: No shape-based binning; uses raw percentile, not binned
""")
print("=" * 80)

print("\nOverall Score:", result["overall_score"])

print("\n" + "=" * 80)
print("RANGES DEFINED PER FEATURE (from derive_ranges)")
print("=" * 80)
print("Rules: ≤6 unique → discrete buckets; large gaps → split points; else None\n")

# Build tabular output
rows = []
for r in result["feature_analysis"]:
    feat = r["feature"]
    val = r["value"]
    score = r["percentile_score"]
    ranges = r["ranges"]
    rule = r.get("range_rule", "smooth" if ranges is None else "gap-based")
    if ranges is None:
        range_str = "None (smooth)"
    else:
        range_str = str([round(x, 2) if isinstance(x, float) else x for x in ranges])
    rows.append((feat[:45], round(val, 2), score, range_str[:60] + ("..." if len(range_str) > 60 else ""), rule))

# Table header
w1, w2, w3, w4, w5 = 48, 12, 10, 62, 10
header = f"{'Feature':<{w1}} {'Value':>{w2}} {'Score':>{w3}} {'Ranges':<{w4}} {'Rule':<{w5}}"
print(header)
print("-" * (w1 + w2 + w3 + w4 + w5 + 4))
for feat, val, score, rng, rule in rows:
    print(f"{feat:<{w1}} {val:>{w2}.2f} {score:>{w3}.2f} {rng:<{w4}} {rule:<{w5}}")

print("\n" + "=" * 80)
print("TOP 5 STRENGTHS")
print("=" * 80)
for f in result["strengths"]:
    print(f"  {f['feature']}: {f['percentile_score']} (value={f['value']})")

print("\n" + "=" * 80)
print("TOP 5 WEAKNESSES")
print("=" * 80)
for f in result["weaknesses"]:
    print(f"  {f['feature']}: {f['percentile_score']} (value={f['value']})")

# METHODOLOGY EXAMPLE
example_feat = "distance_from_nearest_costco"
example_val = site.get(example_feat, 1.0)
if example_feat in FEATURE_DIRECTION and example_feat in df.columns:
    data = df[example_feat].dropna().values
    from scipy.stats import percentileofscore
    raw_p = percentileofscore(data, example_val, kind="rank")
    direction = FEATURE_DIRECTION[example_feat]
    final_score = round(100 - raw_p, 2) if direction == "lower" else round(raw_p, 2)
    note = "Closer to Costco is better" if direction == "lower" else "Higher is better"
    print("\n" + "=" * 80)
    print(f"METHODOLOGY EXAMPLE: {example_feat}")
    print("=" * 80)
    print(f"\nYour Value: {example_val:.2f}")
    print(f"\nAll {len(df)} sites distribution:")
    print(f"  Min:    {float(data.min()):.2f}")
    print(f"  Median: {float(np.median(data)):.2f}")
    print(f"  Max:    {float(data.max()):.2f}")
    print(f"\nStep 1: Calculate Percentile Rank")
    print(f"  Your value ({example_val:.2f}) ranks at: {raw_p:.2f}%ile")
    print(f"  (You are better than {raw_p:.2f}% of sites on RAW value)")
    print(f"\nStep 2: Apply Business Logic - {'lower_is_better' if direction == 'lower' else 'higher_is_better'}")
    print(f"  {note}")
    if direction == "lower":
        print(f"\n  Since LOWER is better:")
        print(f"  - Being at {raw_p:.2f}%ile means you have HIGHER value than {raw_p:.2f}%")
        print(f"  - Which is BAD (because lower is better)")
        print(f"  - So we INVERT: Score = 100 - {raw_p:.2f} = {final_score:.2f}")
    else:
        print(f"\n  Since HIGHER is better:")
        print(f"  - Being at {raw_p:.2f}%ile means you have HIGHER value than {raw_p:.2f}%")
        print(f"  - Which is GOOD (because higher is better)")
        print(f"  - So Score = {raw_p:.2f}")
    print(f"\nStep 3: Final Score")
    print(f"  {final_score:.2f}/100")
    print("=" * 80)
