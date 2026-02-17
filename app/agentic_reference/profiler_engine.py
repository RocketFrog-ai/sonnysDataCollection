"""
Quantile-Based Site Profiling Engine
=====================================

Core engine that implements the non-parametric discriminant analysis:
  1. Loads historical data (531 sites, 50 features)
  2. Splits sites into 3 equal performance tiers by cars_washed(Actual)
  3. Computes IQR (Q25/Q50/Q75) for every feature within each tier
  4. Scores new locations by measuring how well feature values fit each tier's IQR
  5. Predicts the most likely performance tier with a fit score

Statistical basis: quantile reference ranges — robust to outliers,
no distributional assumptions, no unstable regression coefficients.
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────

CATEGORY_LABELS = ["Low Performing", "Average Performing", "High Performing"]

TARGET_COL = "cars_washed(Actual)"

EXCLUDE_COLS = {"full_site_address", TARGET_COL, "performance_category"}

# Default dataset path (sibling file)
_DEFAULT_DATASET = Path(__file__).resolve().parent / "dataSET (1).xlsx"

# ── Dimension groups: 50 features → 5 business dimensions ───────────
# These match the reference guides exactly.

DIMENSION_GROUPS: Dict[str, List[str]] = {
    "Weather": [
        "total_sunshine_hours",
        "days_pleasant_temp",
        "total_precipitation_mm",
        "rainy_days",
        "total_snowfall_cm",
        "snowy_days",
        "days_below_freezing",
        "avg_daily_max_windspeed_ms",
    ],
    "Traffic": [
        # AADT volumes (6 nearest roads)
        "Nearest StreetLight US Hourly-Ttl AADT",
        "2nd Nearest StreetLight US Hourly-Ttl AADT",
        "3rd Nearest StreetLight US Hourly-Ttl AADT",
        "4th Nearest StreetLight US Hourly-Ttl AADT",
        "5th Nearest StreetLight US Hourly-Ttl AADT",
        "6th Nearest StreetLight US Hourly-Ttl AADT",
        # Time-of-day traffic breakdown
        "Nearest StreetLight US Hourly-ttl_breakfast",
        "Nearest StreetLight US Hourly-ttl_lunch",
        "Nearest StreetLight US Hourly-ttl_afternoon",
        "Nearest StreetLight US Hourly-ttl_dinner",
        "Nearest StreetLight US Hourly-ttl_night",
        "Nearest StreetLight US Hourly-ttl_overnight",
        # Traffic light accessibility
        "nearby_traffic_lights_count",
        "distance_nearest_traffic_light_1",
        "distance_nearest_traffic_light_2",
        "distance_nearest_traffic_light_3",
        "distance_nearest_traffic_light_4",
        "distance_nearest_traffic_light_5",
        "distance_nearest_traffic_light_6",
        "distance_nearest_traffic_light_7",
        "distance_nearest_traffic_light_8",
        "distance_nearest_traffic_light_9",
        "distance_nearest_traffic_light_10",
    ],
    "Competition": [
        "competitors_count",
        "competitor_1_distance_miles",
        "competitor_1_google_user_rating_count",
    ],
    "Infrastructure": [
        "tunnel_length (in ft.)",
        "total_weekly_operational_hours",
    ],
    "Retail Proximity": [
        # Retail chain counts (ChainXY)
        "Count of ChainXY VT - Building Supplies",
        "Count of ChainXY VT - Department Store",
        "Count of ChainXY VT - Grocery",
        "Count of ChainXY VT - Mass Merchant",
        "Count of ChainXY VT - Real Estate Model",
        "Sum ChainXY",
        # Named retailer distances and counts
        "distance_from_nearest_target",
        "count_of_target_5miles",
        "distance_from_nearest_costco",
        "count_of_costco_5miles",
        "distance_from_nearest_walmart",
        "count_of_walmart_5miles",
        "distance_from_nearest_bestbuy",
        "count_of_bestbuy_5miles",
    ],
}


class QuantileProfiler:
    """
    Non-parametric profiling engine.

    For each of the 50 features and each performance tier, stores
    the IQR (Q25, Q50, Q75).  A new location is scored by checking
    how well its feature values sit within each tier's IQR.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        historical_data: Optional[pd.DataFrame] = None,
        target_col: str = TARGET_COL,
    ):
        # Load data
        if historical_data is not None:
            self.df = historical_data.copy()
        else:
            path = dataset_path or str(_DEFAULT_DATASET)
            self.df = pd.read_excel(path)

        self.target_col = target_col
        self.category_labels = CATEGORY_LABELS

        # Identify numeric feature columns
        self.feature_cols: List[str] = [
            c
            for c in self.df.columns
            if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(self.df[c])
        ]

        # Create performance tiers (equal-frequency tertiles)
        self.df["performance_category"] = pd.qcut(
            self.df[self.target_col],
            q=3,
            labels=self.category_labels,
            duplicates="drop",
        )

        # Compute IQR ranges for every feature × tier
        self.feature_ranges: Dict[str, Dict[str, Dict[str, float]]] = (
            self._compute_feature_ranges()
        )

        # Compute per-tier volume statistics
        self.category_stats: Dict[str, Dict[str, float]] = (
            self._compute_category_stats()
        )

    # ── Internal: compute ranges ─────────────────────────────────

    def _compute_feature_ranges(self) -> Dict:
        """
        For each feature and each tier, compute Q25, Q50 (median), Q75.
        """
        ranges: Dict[str, Dict[str, Dict[str, float]]] = {}
        for feature in self.feature_cols:
            ranges[feature] = {}
            for cat in self.category_labels:
                subset = self.df.loc[
                    self.df["performance_category"] == cat, feature
                ].dropna()
                if len(subset) < 5:
                    continue
                ranges[feature][cat] = {
                    "q25": float(subset.quantile(0.25)),
                    "q50": float(subset.quantile(0.50)),
                    "q75": float(subset.quantile(0.75)),
                    "min": float(subset.min()),
                    "max": float(subset.max()),
                    "count": int(len(subset)),
                }
        return ranges

    def _compute_category_stats(self) -> Dict:
        """Volume stats per tier (for expected range output)."""
        stats: Dict[str, Dict[str, float]] = {}
        for cat in self.category_labels:
            volumes = self.df.loc[
                self.df["performance_category"] == cat, self.target_col
            ]
            stats[cat] = {
                "q25": float(volumes.quantile(0.25)),
                "median": float(volumes.quantile(0.50)),
                "q75": float(volumes.quantile(0.75)),
                "min": float(volumes.min()),
                "max": float(volumes.max()),
                "mean": float(volumes.mean()),
                "count": int(len(volumes)),
            }
        return stats

    # ── Scoring ──────────────────────────────────────────────────

    def score_feature(
        self, feature: str, value: float
    ) -> Dict[str, float]:
        """
        Score a single feature value against each tier's IQR.

        Scoring logic (from the Quantile Profiling Guide):
          - Inside IQR [Q25, Q75]  →  1.0  (perfect fit)
          - Outside IQR            →  max(0, 1 − distance / IQR_width)
            where distance = min(|value − Q25|, |value − Q75|)

        Returns dict:  { "Low Performing": 0.85, "Average Performing": 1.0, ... }
        """
        if feature not in self.feature_ranges:
            return {}

        scores: Dict[str, float] = {}
        for cat in self.category_labels:
            if cat not in self.feature_ranges[feature]:
                continue
            r = self.feature_ranges[feature][cat]
            q25, q75 = r["q25"], r["q75"]
            iqr_width = q75 - q25

            if q25 <= value <= q75:
                scores[cat] = 1.0
            elif iqr_width > 0:
                dist = min(abs(value - q25), abs(value - q75))
                scores[cat] = max(0.0, 1.0 - dist / iqr_width)
            else:
                # Zero-width IQR (constant feature in this tier)
                scores[cat] = 1.0 if value == q25 else 0.0

        return scores

    def score_location(
        self,
        location_features: Dict[str, float],
        feature_subset: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Score a location across all (or a subset of) features.
        Returns total scores per tier.
        """
        totals: Dict[str, float] = {cat: 0.0 for cat in self.category_labels}
        features_to_score = feature_subset or list(location_features.keys())

        for feat in features_to_score:
            if feat not in location_features or location_features[feat] is None:
                continue
            per_tier = self.score_feature(feat, float(location_features[feat]))
            for cat, s in per_tier.items():
                totals[cat] += s

        return totals

    def predict(
        self,
        location_features: Dict[str, float],
        return_details: bool = False,
    ) -> Dict:
        """
        Predict performance tier for a new location.

        Returns:
          {
            "predicted_category": "High Performing",
            "fit_score": 36.2,         # proportion fit (%)
            "category_scores": {...},
            "expected_volume": { "conservative", "likely", "optimistic" },
            "feature_details": {...}    # if return_details=True
          }
        """
        scores = self.score_location(location_features)
        total = sum(scores.values())
        predicted = max(scores, key=scores.get)
        fit_pct = round(100 * scores[predicted] / total, 1) if total > 0 else 0.0

        vol = self.category_stats[predicted]
        result: Dict = {
            "predicted_category": predicted,
            "fit_score": fit_pct,
            "category_scores": {
                k: round(v, 2) for k, v in scores.items()
            },
            "expected_volume": {
                "conservative": round(vol["q25"]),
                "likely": round(vol["median"]),
                "optimistic": round(vol["q75"]),
            },
        }

        if return_details:
            details: Dict[str, Dict] = {}
            for feat, val in location_features.items():
                if feat not in self.feature_ranges or val is None:
                    continue
                per_tier = self.score_feature(feat, float(val))
                best = max(per_tier, key=per_tier.get) if per_tier else "N/A"
                details[feat] = {
                    "value": val,
                    "best_fit": best,
                    "tier_scores": per_tier,
                    "ranges": {
                        cat: self.feature_ranges[feat].get(cat, {})
                        for cat in self.category_labels
                    },
                }
            result["feature_details"] = details

        return result

    # ── Persistence ──────────────────────────────────────────────

    def save_model(self, path: str = "profiler_model.pkl") -> None:
        data = {
            "feature_ranges": self.feature_ranges,
            "category_stats": self.category_stats,
            "category_labels": self.category_labels,
            "feature_cols": self.feature_cols,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_model(cls, path: str = "profiler_model.pkl") -> "QuantileProfiler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = object.__new__(cls)
        obj.feature_ranges = data["feature_ranges"]
        obj.category_stats = data["category_stats"]
        obj.category_labels = data["category_labels"]
        obj.feature_cols = data["feature_cols"]
        obj.df = None  # no DataFrame needed after loading
        obj.target_col = TARGET_COL
        return obj
