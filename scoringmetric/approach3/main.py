"""
APPROACH 3: Shape-Based Percentile Binning (Auto-decided)
=========================================================

Uses min, max, shape, and distribution to AUTO-DECIDE bins:
- binary (2 unique): 2 bins → 0 or 100
- discrete (3-6 unique): value-based buckets
- continuous: bin count/edges derived from shape + range + n
  - right/left-skewed: coarser bins (5 quintiles) – tail is sparse
  - symmetric: finer bins (10 deciles)
  - narrow range or small n: fewer bins (Sturges rule)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Any


# Reuse approach2 direction maps (subset - can extend)
HIGHER_IS_BETTER = {
    'total_sunshine_hours', 'days_pleasant_temp',
    'Nearest StreetLight US Hourly-Ttl AADT', '2nd Nearest StreetLight US Hourly-Ttl AADT',
    '3rd Nearest StreetLight US Hourly-Ttl AADT', '4th Nearest StreetLight US Hourly-Ttl AADT',
    '5th Nearest StreetLight US Hourly-Ttl AADT', '6th Nearest StreetLight US Hourly-Ttl AADT',
    'total_weekly_operational_hours', 'nearby_traffic_lights_count',
    'Count of ChainXY VT - Building Supplies', 'Count of ChainXY VT - Department Store',
    'Count of ChainXY VT - Grocery', 'Count of ChainXY VT - Mass Merchant',
    'Count of ChainXY VT - Real Estate Model', 'Sum ChainXY',
    'count_of_target_5miles', 'count_of_costco_5miles', 'count_of_walmart_5miles',
    'count_of_bestbuy_5miles', 'competitor_1_google_user_rating_count',
    'tunnel_length (in ft.)',
}
LOWER_IS_BETTER = {
    'total_precipitation_mm', 'rainy_days', 'total_snowfall_cm', 'snowy_days',
    'days_below_freezing', 'avg_daily_max_windspeed_ms', 'competitors_count',
    'competitor_1_distance_miles', 'distance_from_nearest_target', 'distance_from_nearest_costco',
    'distance_from_nearest_walmart', 'distance_from_nearest_bestbuy',
    'distance_nearest_traffic_light_1', 'distance_nearest_traffic_light_2',
    'distance_nearest_traffic_light_3', 'distance_nearest_traffic_light_4',
    'distance_nearest_traffic_light_5', 'distance_nearest_traffic_light_6',
    'distance_nearest_traffic_light_7', 'distance_nearest_traffic_light_8',
    'distance_nearest_traffic_light_9', 'distance_nearest_traffic_light_10',
}
# Add more from approach2 if needed - for unknown, default higher
for col in ['Nearest StreetLight US Hourly-ttl_breakfast', 'Nearest StreetLight US Hourly-ttl_lunch',
            'Nearest StreetLight US Hourly-ttl_afternoon', 'Nearest StreetLight US Hourly-ttl_dinner',
            'Nearest StreetLight US Hourly-ttl_night', 'Nearest StreetLight US Hourly-ttl_overnight']:
    HIGHER_IS_BETTER.add(col)


def _auto_decide_continuous_bins(
    data: np.ndarray,
    shape: str,
    min_val: float,
    max_val: float,
    n: int,
    skew_val: float,
) -> Tuple[np.ndarray, str]:
    """
    Auto-decide percentile cut points and bin strategy from min, max, shape, n.
    Returns (percentile_edges, strategy_description).
    """
    range_val = max_val - min_val
    # Sturges: k = 1 + log2(n) – max bins from sample size
    sturges = max(2, min(10, int(1 + np.log2(n))))

    if shape == "symmetric":
        n_bins = min(10, sturges)  # deciles if enough data
        pcts = np.linspace(100 / n_bins, 100, n_bins)
        desc = f"{n_bins} equal percentile bins (symmetric)"
    else:
        # right-skewed or left-skewed: use coarser bins (quintiles)
        n_bins = min(5, sturges)
        pcts = np.linspace(100 / n_bins, 100, n_bins)
        desc = f"{n_bins} quintile bins (skewed, tail sparse)"

    # If range is negligible, fall back to fewer bins
    mean_val = np.mean(data)
    if mean_val != 0 and range_val / (abs(mean_val) + 1e-12) < 0.01:
        n_bins = 3
        pcts = np.array([33.33, 66.67, 100.0])
        desc = "3 bins (narrow range)"
    elif n < 50:
        n_bins = min(5, sturges)
        pcts = np.linspace(100 / n_bins, 100, n_bins)
        desc = f"{n_bins} bins (small n={n})"

    edges = np.percentile(data, pcts)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([min_val, max_val])
    return edges, desc


class ShapeBasedProfiler:
    """
    Scores features using shape-defined bins. Continuous bins are AUTO-DECIDED
    from min, max, shape, and n (no fixed 10 deciles).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        exclude = ['cars_washed(Actual)', 'full_site_address']
        self.feature_cols = [c for c in df.columns
                            if c not in exclude and df[c].dtype in ['int64', 'float64']]
        self._build_bins()

    def _build_bins(self):
        """For each feature: compute shape and define percentile bins."""
        self.bins = {}
        for feat in self.feature_cols:
            data = self.df[feat].dropna().values
            n_unique = len(np.unique(data))
            skew_val = float(stats.skew(data)) if len(data) > 2 else 0.0

            if n_unique <= 2:
                shape = "binary"
            elif n_unique <= 6:
                shape = "discrete"
            elif skew_val > 0.5:
                shape = "right-skewed"
            elif skew_val < -0.5:
                shape = "left-skewed"
            else:
                shape = "symmetric"

            lower_better = feat in LOWER_IS_BETTER

            if shape == "binary":
                uniq = np.sort(np.unique(data))
                min_val, max_val = float(np.min(data)), float(np.max(data))
                if lower_better:
                    best, worst = uniq[0], uniq[-1]
                else:
                    best, worst = uniq[-1], uniq[0]
                bins_def = [{"range": (float(worst), float(worst)), "score": 0, "label": f"worst ({worst})"},
                           {"range": (float(best), float(best)), "score": 100, "label": f"best ({best})"}]
                self.bins[feat] = {"shape": shape, "lower_better": lower_better, "bins_def": bins_def,
                                  "bin_edges": uniq, "data": data, "min": min_val, "max": max_val}
                continue

            elif shape == "discrete":
                uniq = np.sort(np.unique(data))
                n = len(uniq)
                if lower_better:
                    scores = np.linspace(100, 0, n)
                else:
                    scores = np.linspace(0, 100, n)
                bins_def = [{"range": (float(u), float(u)), "score": float(s), "label": str(u)}
                            for u, s in zip(uniq, scores)]
                self.bins[feat] = {"shape": shape, "lower_better": lower_better, "bins_def": bins_def,
                                  "bin_edges": uniq, "data": data, "min": float(np.min(data)), "max": float(np.max(data))}
                continue

            else:
                # continuous: auto-decide from shape, min, max, n
                edges, strategy_desc = _auto_decide_continuous_bins(
                    data, shape,
                    float(np.min(data)), float(np.max(data)),
                    len(data), skew_val,
                )
                n_bins = len(edges)
                if lower_better:
                    bin_scores = np.linspace(95, 5, n_bins)
                else:
                    bin_scores = np.linspace(5, 95, n_bins)
                bins_def = []
                prev = np.min(data) - 1e-9
                for i, e in enumerate(edges):
                    s = bin_scores[min(i, n_bins - 1)]
                    p_lo = float(stats.percentileofscore(data, prev, kind="rank"))
                    p_hi = float(stats.percentileofscore(data, float(e), kind="rank"))
                    bins_def.append({"range": (prev, float(e)), "score": float(s),
                                    "label": f"p{p_lo:.0f}-p{p_hi:.0f}"})
                    prev = float(e)
                bin_edges = edges

            meta = {
                "shape": shape,
                "lower_better": lower_better,
                "bins_def": bins_def,
                "bin_edges": bin_edges,
                "data": data,
                "min": float(np.min(data)),
                "max": float(np.max(data)),
            }
            if shape in ("right-skewed", "left-skewed", "symmetric"):
                meta["strategy_desc"] = strategy_desc
            self.bins[feat] = meta

        print(f"✓ Shape-based profiler initialized: {len(self.df)} sites, {len(self.feature_cols)} features")

    def score_feature(self, feature: str, value: float) -> Dict:
        if feature not in self.bins:
            return {"error": "Feature not found"}
        b = self.bins[feature]
        data = b["data"]
        raw_p = float(stats.percentileofscore(data, value, kind="rank"))

        if b["shape"] == "binary":
            uniq = np.sort(np.unique(data))
            if b["lower_better"]:
                best = uniq[0]
                score = 100.0 if value <= best else 0.0
            else:
                best = uniq[-1]
                score = 100.0 if value >= best else 0.0
            bin_label = f"best ({best})" if score == 100 else f"worst ({uniq[uniq != best][0]})"

        elif b["shape"] == "discrete":
            uniq = np.sort(np.unique(data))
            idx = np.searchsorted(uniq, value)
            idx = min(idx, len(uniq) - 1)
            n = len(uniq)
            if b["lower_better"]:
                s = np.linspace(100, 0, n)
            else:
                s = np.linspace(0, 100, n)
            score = float(s[idx])
            bin_label = str(uniq[idx])

        else:
            # continuous: find which bin (from auto-decided edges) value falls in
            edges = b["bin_edges"]
            bin_idx = np.searchsorted(edges, value)
            bin_idx = min(bin_idx, len(b["bins_def"]) - 1)
            bin_info = b["bins_def"][bin_idx]
            score = bin_info["score"]
            bin_label = bin_info["label"]

        return {
            "feature": feature,
            "value": value,
            "raw_percentile": round(raw_p, 2),
            "final_score": round(score, 2),
            "shape": b["shape"],
            "bin_label": bin_label,
            "direction": "lower_is_better" if b["lower_better"] else "higher_is_better",
        }

    def score_location(self, location: Dict[str, float]) -> Dict:
        scores = []
        details = {}
        for feat, val in location.items():
            if feat in self.feature_cols:
                d = self.score_feature(feat, val)
                if "error" not in d:
                    details[feat] = d
                    scores.append(d["final_score"])
        overall = round(np.mean(scores), 2) if scores else 0
        sorted_items = sorted(details.items(), key=lambda x: -x[1]["final_score"])
        return {
            "overall_score": overall,
            "feature_scores": details,
            "top_strengths": sorted_items[:10],
            "top_weaknesses": list(reversed(sorted_items[-10:])),
        }

    def print_bin_definitions(self):
        print("\n" + "=" * 100)
        print("AUTO-DECIDED BIN DEFINITIONS (min, max, shape → bins)")
        print("=" * 100)
        for feat in self.feature_cols:
            b = self.bins[feat]
            print(f"\n  {feat}")
            print(f"    min={b.get('min', '-'):.2f}  max={b.get('max', '-'):.2f}  shape={b['shape']}  direction={'lower✓' if b['lower_better'] else 'higher✓'}")
            if b["shape"] == "binary":
                print(f"    bins: {b['bins_def']}")
            elif b["shape"] == "discrete":
                print(f"    value → score: {[(bd['label'], bd['score']) for bd in b['bins_def']]}")
            else:
                desc = b.get("strategy_desc", "continuous")
                print(f"    strategy: {desc}")
                print(f"    bin ranges → score: {[(bd['label'], bd['score']) for bd in b['bins_def']]}")

        print("\n" + "=" * 100)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    data_paths = [base / "dataSET (1).xlsx", Path("dataSET (1).xlsx")]
    df = None
    for p in data_paths:
        if p.exists():
            df = pd.read_excel(p, engine="openpyxl")
            break
    if df is None:
        raise FileNotFoundError("Could not find dataSET (1).xlsx")

    profiler = ShapeBasedProfiler(df)

    print("\n" + "=" * 80)
    print("APPROACH 3 - Shape-Based Auto Binning")
    print("=" * 80)
    print("""
WHAT THIS APPROACH DOES:
  - AUTO-DECIDES bins from min, max, shape, sample size
  - Binary (2 values) → 2 bins; Discrete (3-6) → value buckets
  - Skewed → 5 quintiles; Symmetric → 10 deciles
  - Score = WHICH BIN you fall into (e.g. quintile 3 → 50), not raw percentile

HOW IT DIFFERS FROM APPROACH 1: Shape-driven binning (affects score); no config
HOW IT DIFFERS FROM APPROACH 2: Binned scores, not raw percentile; no plots
""")
    print("=" * 80)

    profiler.print_bin_definitions()

    # Sample site: use mean (average) values - represents a "typical" site
    location = {}
    for c in profiler.feature_cols:
        mean_val = df[c].mean()
        if pd.notna(mean_val):
            location[c] = float(mean_val)
    result = profiler.score_location(location)

    print("\n" + "=" * 80)
    print("SCORING REPORT (Shape-Based Bins)")
    print("=" * 80)
    print(f"Overall Score: {result['overall_score']}/100\n")
    print(f"{'Feature':<42} {'Value':>10} {'Shape':>12} {'Bin':>25} {'Score':>8}")
    print("-" * 100)
    for feat, d in result["feature_scores"].items():
        print(f"{feat[:40]:<42} {d['value']:>10.1f} {d['shape']:>12} {str(d['bin_label'])[:24]:>25} {d['final_score']:>7.1f}")
    print("-" * 100)
    print("\nTop Strengths:")
    for feat, d in result["top_strengths"][:5]:
        print(f"  {feat}: {d['final_score']} ({d['bin_label']})")
    print("\nTop Weaknesses:")
    for feat, d in result["top_weaknesses"][:5]:
        print(f"  {feat}: {d['final_score']} ({d['bin_label']})")

    # METHODOLOGY EXAMPLE
    example_feat = "distance_from_nearest_costco"
    example_val = location.get(example_feat, 1.0)
    if example_feat in profiler.feature_cols:
        d = profiler.score_feature(example_feat, example_val)
        b = profiler.bins[example_feat]
        print("\n" + "=" * 80)
        print(f"METHODOLOGY EXAMPLE: {example_feat}")
        print("=" * 80)
        print(f"\nYour Value: {example_val:.2f}")
        print(f"\nAll {len(df)} sites distribution:")
        print(f"  Min:    {b['min']:.2f}")
        print(f"  Max:    {b['max']:.2f}")
        print(f"  Shape:  {b['shape']}  (direction: {'lower_is_better' if b['lower_better'] else 'higher_is_better'})")
        print(f"\nStep 1: Raw Percentile (for context)")
        print(f"  Your value ranks at: {d['raw_percentile']:.2f}%ile")
        print(f"\nStep 2: Shape-Based Binning")
        if b["shape"] == "binary":
            print(f"  Binary feature → 2 bins (best vs worst)")
        elif b["shape"] == "discrete":
            print(f"  Discrete ({len(np.unique(b['data']))} unique values) → value buckets")
        else:
            print(f"  {b.get('strategy_desc', 'Continuous')}")
            print(f"  Bin definitions: {[(bd['label'], bd['score']) for bd in b['bins_def']]}")
        print(f"\nStep 3: Your value falls in bin: {d['bin_label']}")
        print(f"  Score = {d['final_score']:.2f}/100 (from bin, not raw percentile)")
        print("=" * 80)

    print("=" * 80)
