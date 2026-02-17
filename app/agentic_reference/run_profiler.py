#!/usr/bin/env python3
"""
Car Wash Site Profiling — CLI Entry Point
==========================================

End-to-end pipeline:
  1. Loads historical dataset → builds QuantileProfiler
  2. Scores the input location (overall + per-dimension)
  3. Generates LLM-powered rationale (or fallback)
  4. Prints a full report

Usage:
  python run_profiler.py                   # runs built-in demo
  python run_profiler.py input.json        # reads features from JSON file
"""

from __future__ import annotations

import sys
import os
import json

# Ensure sibling modules resolve when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Ensure app root is on path for local_llm imports
_app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

from profiler_engine import QuantileProfiler
from dimension_profiler import DimensionProfiler


# ── Demo location (representative values from actual dataset columns) ─
DEMO_LOCATION = {
    # Weather
    "total_sunshine_hours": 3350.0,
    "days_pleasant_temp": 160.0,
    "total_precipitation_mm": 1400.0,
    "rainy_days": 120.0,
    "total_snowfall_cm": 5.0,
    "snowy_days": 3.0,
    "days_below_freezing": 15.0,
    "avg_daily_max_windspeed_ms": 16.0,

    # Traffic (AADT)
    "Nearest StreetLight US Hourly-Ttl AADT": 15000,
    "2nd Nearest StreetLight US Hourly-Ttl AADT": 12000,
    "3rd Nearest StreetLight US Hourly-Ttl AADT": 9000,
    # Time-of-day
    "Nearest StreetLight US Hourly-ttl_breakfast": 2500,
    "Nearest StreetLight US Hourly-ttl_lunch": 3000,
    "Nearest StreetLight US Hourly-ttl_afternoon": 3200,
    "Nearest StreetLight US Hourly-ttl_dinner": 2800,
    "Nearest StreetLight US Hourly-ttl_night": 1200,
    "Nearest StreetLight US Hourly-ttl_overnight": 300,
    # Traffic lights
    "nearby_traffic_lights_count": 14,
    "distance_nearest_traffic_light_1": 0.15,
    "distance_nearest_traffic_light_2": 0.20,
    "distance_nearest_traffic_light_3": 0.35,

    # Competition
    "competitors_count": 1,
    "competitor_1_distance_miles": 3.5,
    "competitor_1_google_user_rating_count": 200,

    # Infrastructure
    "tunnel_length (in ft.)": 115.0,
    "total_weekly_operational_hours": 80.0,

    # Retail Proximity
    "Count of ChainXY VT - Grocery": 2,
    "Count of ChainXY VT - Mass Merchant": 1,
    "Sum ChainXY": 5,
    "distance_from_nearest_walmart": 1.2,
    "count_of_walmart_5miles": 2,
    "count_of_costco_5miles": 1,
}


def run_pipeline(
    location_features: dict,
    use_llm: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: quantile scoring → dimension profiling → rationale.
    """
    # 1. Build profilers
    if verbose:
        print("━" * 60)
        print("  STEP 1 — Loading dataset & building profilers")
        print("━" * 60)
    profiler = QuantileProfiler()
    dim_profiler = DimensionProfiler(profiler)

    if verbose:
        print(f"  Loaded {len(profiler.feature_cols)} features from {len(profiler.df)} sites")
        for cat, stats in profiler.category_stats.items():
            print(f"    {cat}: {stats['count']} sites, "
                  f"median={stats['median']:,.0f} cars/yr")

    # 2. Overall prediction
    if verbose:
        print("\n" + "━" * 60)
        print("  STEP 2 — Overall quantile prediction")
        print("━" * 60)
    overall = profiler.predict(location_features, return_details=True)
    if verbose:
        p = overall
        print(f"  Tier      : {p['predicted_category']}")
        print(f"  Fit score : {p['fit_score']}%")
        ev = p["expected_volume"]
        print(f"  Volume    : {ev['conservative']:,}–{ev['likely']:,}–{ev['optimistic']:,} cars/yr")

    # 3. Dimension scoring
    if verbose:
        print("\n" + "━" * 60)
        print("  STEP 3 — Dimension-specific scoring")
        print("━" * 60)
    dim_results = dim_profiler.score_all_dimensions(location_features)
    if verbose:
        print(dim_profiler.format_dimension_report(dim_results, include_details=True))
        print()
        print(dim_profiler.format_strength_ranking())

    strengths, weaknesses, neutrals = dim_profiler.get_strengths_weaknesses(dim_results)
    vote = dim_profiler.majority_vote(dim_results)

    # 4. Rationale
    if verbose:
        print("\n" + "━" * 60)
        print("  STEP 4 — Generating rationale")
        print("━" * 60)

    if use_llm:
        from agentic_rationale import generate_rationale
        rationale = generate_rationale(
            location_features=location_features,
            overall_prediction=overall,
            dimension_results=dim_results,
            dimension_strength=dim_profiler.dimension_strength,
            vote_result=vote,
            strengths=strengths,
            weaknesses=weaknesses,
            neutrals=neutrals,
        )
    else:
        from agentic_rationale import _generate_fallback_rationale
        rationale = _generate_fallback_rationale(
            location_features=location_features,
            overall_prediction=overall,
            dimension_results=dim_results,
            dimension_strength=dim_profiler.dimension_strength,
            vote_result=vote,
            strengths=strengths,
            weaknesses=weaknesses,
            neutrals=neutrals,
        )

    if verbose:
        print("\n" + rationale)

    return {
        "overall_prediction": overall,
        "dimension_results": dim_results,
        "vote": vote,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "neutrals": neutrals,
        "rationale": rationale,
    }


# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print(f"Loading features from {json_path}")
        with open(json_path) as f:
            features = json.load(f)
    else:
        print("No input file — using built-in demo location\n")
        features = DEMO_LOCATION

    # For standalone runs, skip LLM (may not have server running)
    result = run_pipeline(features, use_llm=False, verbose=True)

    # Save outputs
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(out_dir, "last_report.json")
    serialisable = {k: v for k, v in result.items() if k != "rationale"}
    with open(out_json, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    out_md = os.path.join(out_dir, "last_report.md")
    with open(out_md, "w") as f:
        f.write(result["rationale"])

    print(f"\n✓ Results saved to {out_json}")
    print(f"✓ Report saved to {out_md}")
