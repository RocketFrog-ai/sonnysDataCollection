"""
Quantile-Based Car Wash Count Predictor
========================================

Given a new location's 19 Proforma features, this module predicts which
quantile of the car wash count distribution (Q1–Q4) the location falls into,
explains per-feature quantile positioning, and shows which feature improvements
would shift the predicted car wash quantile upward.

Methodology
-----------
1. Merge feature data (Proforma Excel, 19 features × 1235 sites) with car wash
   count data (temp_extrapolated.csv, current_count × 1064 sites) via street+zip.
2. Bin car wash counts into Q1–Q4 (data-driven quartile boundaries).
3. Compute per-feature quartile boundaries from each feature's raw distribution.
4. Train a Random Forest classifier: features → car wash count quantile.
5. For a new location:
   a. Predict which car wash count quantile it falls into (with probability breakdown).
   b. Per-feature quantile position (direction-adjusted so Q4 = always best).
   c. Quantile shift analysis: which feature changes shift the wash quantile up.
   d. LLM-generated narrative and strengths/weaknesses (no fallbacks).

Quantile Definitions
--------------------
  Q1 (Bottom 25%)  — Low Performer
  Q2 (25–50%)      — Below Median
  Q3 (50–75%)      — Above Median
  Q4 (Top 25%)     — High Performer

Feature Quantiles (direction-adjusted, Q4 = always best)
---------------------------------------------------------
  For "higher is better" features: raw quantile == adjusted quantile
  For "lower is better" features : adjusted_q = 5 - raw_q
    e.g. precipitation at raw Q1 (very low rain) → adjusted Q4 (excellent)
"""

from __future__ import annotations

import re
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

QUANTILE_LABELS: Dict[int, str] = {
    1: "Q1 (Bottom 25%)",
    2: "Q2 (25–50%)",
    3: "Q3 (50–75%)",
    4: "Q4 (Top 25%)",
}
QUANTILE_TIER_NAMES: Dict[int, str] = {
    1: "Low Performer",
    2: "Below Median",
    3: "Above Median",
    4: "High Performer",
}

# Direction: "lower" means lower value is BETTER (e.g. fewer rainy days)
FEATURE_DIRECTIONS: Dict[str, str] = {
    "weather_total_precipitation_mm": "lower",
    "weather_rainy_days": "lower",
    "weather_total_snowfall_cm": "lower",
    "weather_days_below_freezing": "lower",
    "weather_avg_daily_max_windspeed_ms": "lower",
    "nearest_gas_station_distance_miles": "lower",
    "competitors_count_4miles": "lower",
    "competitor_1_distance_miles": "lower",
    "distance_nearest_costco(5 mile)": "lower",
    "distance_nearest_walmart(5 mile)": "lower",
    "distance_nearest_target (5 mile)": "lower",
    "weather_total_sunshine_hours": "higher",
    "weather_days_pleasant_temp": "higher",
    "nearest_gas_station_rating": "higher",
    "nearest_gas_station_rating_count": "higher",
    "competitor_1_google_rating": "higher",
    "competitor_1_rating_count": "higher",
    "other_grocery_count_1mile": "higher",
    "count_food_joints_0_5miles (0.5 mile)": "higher",
}

FEATURE_LABELS: Dict[str, str] = {
    "weather_total_precipitation_mm": "Annual Precipitation (mm)",
    "weather_rainy_days": "Rainy Days / Year",
    "weather_total_snowfall_cm": "Annual Snowfall (cm)",
    "weather_days_below_freezing": "Days Below Freezing",
    "weather_avg_daily_max_windspeed_ms": "Avg Max Wind Speed (m/s)",
    "weather_total_sunshine_hours": "Annual Sunshine Hours",
    "weather_days_pleasant_temp": "Pleasant Temp Days",
    "nearest_gas_station_distance_miles": "Nearest Gas Station (miles)",
    "nearest_gas_station_rating": "Nearest Gas Station Rating",
    "nearest_gas_station_rating_count": "Gas Station Review Count",
    "competitors_count_4miles": "Competitors within 4 Miles",
    "competitor_1_google_rating": "Nearest Competitor Rating",
    "competitor_1_distance_miles": "Nearest Competitor (miles)",
    "competitor_1_rating_count": "Competitor Review Count",
    "distance_nearest_costco(5 mile)": "Distance to Nearest Costco (mi)",
    "distance_nearest_walmart(5 mile)": "Distance to Nearest Walmart (mi)",
    "distance_nearest_target (5 mile)": "Distance to Nearest Target (mi)",
    "other_grocery_count_1mile": "Grocery Stores within 1 Mile",
    "count_food_joints_0_5miles (0.5 mile)": "Food Joints within 0.5 Mile",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _extract_zip5(addr: str) -> str:
    if pd.isna(addr):
        return ""
    m = re.search(r"(\d{5})(?:-\d{4})?\s*$", str(addr).strip())
    return m.group(1) if m else ""


def _extract_street_from_address(addr: str) -> str:
    """Pull the street portion out of 'CHAIN, STREET, CITY, ZIP' format."""
    if pd.isna(addr):
        return ""
    parts = [p.strip() for p in str(addr).split(",")]
    return parts[1].upper() if len(parts) >= 2 else parts[0].upper()


def _load_and_merge(excel_path: Path, csv_path: Path) -> pd.DataFrame:
    """
    Merge Proforma feature data (Excel) with car wash count data (CSV).
    Join key: normalized street address + 5-digit ZIP code.
    Returns a DataFrame with all 19 feature columns + current_count + location_id.
    """
    df_feat = pd.read_excel(excel_path, engine="openpyxl", header=1)
    df_cnt = pd.read_csv(csv_path)

    df_feat["_zip5"] = df_feat["Address"].apply(_extract_zip5)
    df_feat["_street"] = df_feat["Address"].apply(_extract_street_from_address)

    df_cnt["_zip5"] = df_cnt["zip"].astype(str).str[:5].str.strip()
    df_cnt["_street"] = df_cnt["street"].astype(str).str.upper().str.strip()

    merge_cols = ["_street", "_zip5", "current_count", "location_id"]
    if "client_id" in df_cnt.columns:
        merge_cols = ["_street", "_zip5", "current_count", "location_id", "client_id"]
    merged = df_feat.merge(
        df_cnt[merge_cols],
        on=["_street", "_zip5"],
        how="inner",
    ).drop(columns=["_zip5", "_street"])

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_shape(data: np.ndarray) -> str:
    if len(np.unique(data)) <= 6:
        return "discrete"
    skew_val = float(stats.skew(data))
    if skew_val > 0.5:
        return "right-skewed"
    elif skew_val < -0.5:
        return "left-skewed"
    return "symmetric"


def _quantile_boundaries(data: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Return n+1 boundary values using equal-frequency (standard) quartile percentiles.

    Quartile definition:
      Q1 = bottom 25%   (0th – 25th percentile)
      Q2 = 25th – 50th
      Q3 = 50th – 75th
      Q4 = top 25%      (75th – 100th percentile)

    Direction-adjusted, Q4 always means best:
      "higher is better": Q4 = top 25% by raw value.
      "lower is better":  Q4 = bottom 25% by raw value (lowest = best).

    So: if your adjusted percentile is 90% → you beat 90% of car wash sites
    on that feature (after direction flip) → Q4 (top 25%).
    """
    pcts = np.linspace(0, 100, n + 1)   # [0, 25, 50, 75, 100]
    return np.percentile(data, pcts)


def _assign_raw_quantile(value: float, boundaries: np.ndarray) -> int:
    """Return raw quantile (1–4) based on where value falls in the boundaries."""
    for q in range(1, len(boundaries)):
        if value <= boundaries[q]:
            return q
    return len(boundaries) - 1


def _adj_quantile(raw_q: int, direction: str) -> int:
    """Adjusted quantile: always Q4 = best, Q1 = worst."""
    return (5 - raw_q) if direction == "lower" else raw_q


def _next_better_boundary(boundaries: np.ndarray, adj_q: int, direction: str) -> Optional[float]:
    """
    Return the feature value threshold needed to reach the next better adjusted quantile.

    For "higher is better":
      current raw_q == adj_q; need value >= boundaries[adj_q] to reach adj_q+1
    For "lower is better":
      current raw_q = 5 - adj_q; need value <= boundaries[4 - adj_q] to reach adj_q+1
    """
    if adj_q >= 4:
        return None
    if direction == "higher":
        return float(boundaries[adj_q])
    else:
        return float(boundaries[4 - adj_q])


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class QuantilePredictor:
    """
    Quantile-based car wash count predictor.

    Usage
    -----
    predictor = QuantilePredictor()
    result = predictor.analyze({"weather_total_sunshine_hours": 6500, ...})
    predictor.print_report(result)
    predictor.plot_feature_quantiles(result, "my_location.png")
    """

    def __init__(
        self,
        excel_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        n_quantiles: int = 4,
        use_control_sites_only: bool = False,
    ):
        base = Path(__file__).resolve().parent
        excel_path = excel_path or base / "Proforma-v2-data-final (1).xlsx"
        csv_path = csv_path or base / "v2" / "temp_extrapolated.csv"

        if not excel_path.exists():
            raise FileNotFoundError(f"Excel not found: {excel_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.n_quantiles = n_quantiles
        self.use_control_sites_only = use_control_sites_only

        print("Loading and merging data…")
        self.df = _load_and_merge(excel_path, csv_path)

        if use_control_sites_only and "client_id" in self.df.columns:
            before = len(self.df)
            self.df = self.df[
                ~self.df["client_id"].astype(str).str.contains(
                    "Controls Training", case=False, na=False
                )
            ].copy()
            print(f"  Control-sites only: excluded non-control rows → {len(self.df)} sites (was {before})")

        self.feature_cols: List[str] = [
            c for c in self.df.columns
            if c not in ("Address", "current_count", "location_id")
            and self.df[c].dtype in ("float64", "int64")
        ]

        print(f"  {len(self.df)} sites with features + car wash counts")
        print(f"  {len(self.feature_cols)} predictive features")

        self._build_feature_distributions()
        self._build_wash_quantiles()
        self._build_quantile_profiles()
        self._train_classifier()
        print("✓ QuantilePredictor ready\n")

    # ── Initialization helpers ────────────────────────────────────────────────

    def _build_feature_distributions(self):
        """
        Pre-compute per-feature distribution statistics and equal-frequency quartile boundaries.

        Boundaries use standard [0, 25, 50, 75, 100] percentiles so that:
          Q4 = top 25% of car wash sites on that feature (direction-adjusted)
          Q3 = 50th–75th percentile
          Q2 = 25th–50th percentile
          Q1 = bottom 25%
        Shape is detected for display/reporting only; it does not affect the bin edges.
        """
        self.feature_dists: Dict[str, Dict] = {}
        for feat in self.feature_cols:
            data = self.df[feat].dropna().values
            shape = _detect_shape(data)
            boundaries = _quantile_boundaries(data, self.n_quantiles)
            direction = FEATURE_DIRECTIONS.get(feat, "higher")
            self.feature_dists[feat] = {
                "data": data,
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "shape": shape,
                "boundaries": boundaries,
                "direction": direction,
            }

    def _build_wash_quantiles(self):
        """
        Bin current_count into Q1–Q4 using equal-frequency (standard) quartiles.

          Q1 = bottom 25% of car wash counts  (lowest volume)
          Q2 = 25th–50th percentile
          Q3 = 50th–75th percentile
          Q4 = top 25% of car wash counts     (highest volume)
        """
        counts = self.df["current_count"].dropna().values
        self.wash_shape = _detect_shape(counts)
        self.wash_boundaries = _quantile_boundaries(counts, self.n_quantiles)

        self.df["wash_q"] = pd.cut(
            self.df["current_count"],
            bins=self.wash_boundaries,
            labels=list(range(1, self.n_quantiles + 1)),
            include_lowest=True,
        ).astype("Int64")

        self.wash_q_ranges: Dict[int, Tuple[float, float]] = {
            q: (self.wash_boundaries[q - 1], self.wash_boundaries[q])
            for q in range(1, self.n_quantiles + 1)
        }

        print(f"\nCar Wash Count Quantile Boundaries ({self.wash_shape}):")
        for q, (lo, hi) in self.wash_q_ranges.items():
            n = int((self.df["wash_q"] == q).sum())
            print(f"  Q{q}: {lo:>10,.0f} – {hi:>10,.0f} cars/yr  (n={n})")

    def _build_quantile_profiles(self):
        """
        For each wash quantile Q1–Q4, compute median and IQR of every feature.
        This is the 'typical profile' of sites in each performance tier.
        """
        self.quantile_profiles: Dict[int, Dict[str, Dict]] = {}
        for q in range(1, self.n_quantiles + 1):
            subset = self.df[self.df["wash_q"] == q]
            profile: Dict[str, Dict] = {}
            for feat in self.feature_cols:
                vals = subset[feat].dropna()
                if len(vals) > 0:
                    profile[feat] = {
                        "median": float(vals.median()),
                        "mean": float(vals.mean()),
                        "q25": float(vals.quantile(0.25)),
                        "q75": float(vals.quantile(0.75)),
                        "n": len(vals),
                    }
            self.quantile_profiles[q] = profile

    def _feature_to_wash_q(self, feat: str, val: float, direction: str) -> Dict:
        """
        Map a feature value to the car-wash-count quartile (Q1–Q4) whose
        typical profile (median within the group) it most resembles.

        Logic:
          - For each wash group (Q1–Q4 sites), compute that group's median for this feature.
          - Direction-adjust (for "lower is better", invert so "better" = numerically higher).
          - Find the group whose adjusted median is closest to the adjusted value.
          - Flag "exceeds_q4" when the value is BETTER than even the Q4 group median.

        Returns dict with:
          wash_q (int)       : matched wash-count quartile (1–4)
          group_medians (dict): {q: median} raw (not direction-adjusted)
          exceeds_q4 (bool)  : value better than Q4 group median
          q4_median (float)  : median of this feature within Q4 car wash sites
        """
        group_medians: Dict[int, float] = {}
        for q in range(1, self.n_quantiles + 1):
            med = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
            if med is not None:
                group_medians[q] = med

        if not group_medians:
            return {"wash_q": None, "group_medians": {}, "exceeds_q4": False, "q4_median": None}

        sign = -1 if direction == "lower" else 1
        adj_val = sign * val
        adj_medians = {q: sign * m for q, m in group_medians.items()}

        q4_med = group_medians.get(4)
        q4_adj = adj_medians.get(4)
        exceeds_q4 = q4_adj is not None and adj_val > q4_adj

        # If value is better than Q4's typical, assign WashQ=4 (not just "closest median")
        if exceeds_q4:
            matched_q = 4
        else:
            matched_q = min(adj_medians.items(), key=lambda x: abs(x[1] - adj_val))[0]

        return {
            "wash_q": matched_q,
            "group_medians": group_medians,
            "exceeds_q4": exceeds_q4,
            "q4_median": q4_med,
        }

    def _train_classifier(self):
        """
        Train RandomForest: features → wash_q (1–4).

        NaN values are imputed with per-feature medians so we retain all 1000+
        training rows rather than dropping to ~250 complete-case rows.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        # Compute and store per-feature medians used for imputation
        self._feature_medians: Dict[str, float] = {
            feat: float(self.df[feat].median()) for feat in self.feature_cols
        }

        X_raw = self.df[self.feature_cols].copy()
        # Impute NaNs with training-set medians
        for feat in self.feature_cols:
            X_raw[feat] = X_raw[feat].fillna(self._feature_medians[feat])

        y = self.df["wash_q"].copy()
        mask = y.notna()
        X_clean = X_raw[mask].values
        y_clean = y[mask].values.astype(int)

        self.clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.clf.fit(X_clean, y_clean)

        # 5-fold CV for exact and adjacent-quantile accuracy
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_exact = cross_val_score(self.clf, X_clean, y_clean, cv=skf, scoring="accuracy")
        self.cv_accuracy = float(cv_exact.mean())

        # Adjacent accuracy (prediction within ±1 quantile)
        all_preds, all_true = [], []
        for tr, te in skf.split(X_clean, y_clean):
            self.clf.fit(X_clean[tr], y_clean[tr])
            all_preds.extend(self.clf.predict(X_clean[te]))
            all_true.extend(y_clean[te])
        self.clf.fit(X_clean, y_clean)  # refit on full data
        adj = np.mean(np.abs(np.array(all_preds) - np.array(all_true)) <= 1)
        self.cv_adjacent_accuracy = float(adj)

        self.feature_importances: Dict[str, float] = dict(
            zip(self.feature_cols, self.clf.feature_importances_)
        )

        print(f"\nClassifier — 5-fold CV  exact: {self.cv_accuracy:.1%}  |  within-1-quantile: {self.cv_adjacent_accuracy:.1%}  (n={len(y_clean)})")
        top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 predictive features:")
        for feat, imp in top:
            print(f"    {FEATURE_LABELS.get(feat, feat)}: {imp:.1%}")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        location_features: Dict[str, float],
        llm_narrative: bool = True,
    ) -> Dict:
        """
        Full quantile analysis for a new location.

        Parameters
        ----------
        location_features : dict
            Feature name → value. Missing features are imputed with the median.
        llm_narrative : bool
            If True, call the local LLM for a prose narrative summary.

        Returns
        -------
        dict with keys:
          predicted_wash_quantile, predicted_wash_quantile_label,
          predicted_wash_tier, predicted_wash_range, quantile_probabilities,
          wash_count_distribution, feature_analysis, shift_opportunities,
          profile_comparison, model_cv_accuracy, features_available,
          narrative (if llm_narrative=True)
        """
        # ── 1. Build feature vector and per-feature analysis ──────────────
        feature_vector: List[float] = []
        feature_analysis: Dict[str, Dict] = {}

        for feat in self.feature_cols:
            raw_val = location_features.get(feat)
            dist = self.feature_dists[feat]
            direction = dist["direction"]

            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                val = float(raw_val)
                imputed = False
            else:
                # Use training-set median (same imputation strategy as the classifier)
                val = self._feature_medians.get(feat, dist["median"])
                imputed = True

            feature_vector.append(val)

            # Percentile vs full distribution (raw: % of sites below this value)
            raw_pct = float(stats.percentileofscore(dist["data"], val, kind="rank"))
            # Adjusted: higher = better (inverted for "lower is better" features)
            adj_pct = (100.0 - raw_pct) if direction == "lower" else raw_pct

            raw_q = _assign_raw_quantile(val, dist["boundaries"])
            adj_q = _adj_quantile(raw_q, direction)

            better_boundary = _next_better_boundary(dist["boundaries"], adj_q, direction)
            if better_boundary is not None:
                delta = abs(val - better_boundary)
                shift_dir = "increase" if direction == "higher" else "decrease"
            else:
                delta = None
                shift_dir = None

            # Wash-correlated Q: which car wash count group's feature profile does this value match?
            wq_info = self._feature_to_wash_q(feat, val, direction)

            feature_analysis[feat] = {
                "value": val if not imputed else None,
                "imputed": imputed,
                "label": FEATURE_LABELS.get(feat, feat),
                "direction": direction,
                "raw_percentile": round(raw_pct, 1),
                "adjusted_percentile": round(adj_pct, 1),
                "feature_quantile_raw": raw_q,
                "feature_quantile_adj": adj_q,
                # Wash-correlated Q: matches which car wash count tier?
                "wash_correlated_q": wq_info["wash_q"],
                "wash_correlated_exceeds_q4": wq_info["exceeds_q4"],
                "wash_q_group_medians": wq_info["group_medians"],
                "wash_q_q4_median": wq_info["q4_median"],
                "distribution_shape": dist["shape"],
                "dist_min": dist["min"],
                "dist_max": dist["max"],
                "dist_median": dist["median"],
                "quantile_boundaries": dist["boundaries"].tolist(),
                "shift_to_next_q_delta": round(delta, 2) if delta is not None else None,
                "shift_to_next_q_direction": shift_dir,
                "importance": round(self.feature_importances.get(feat, 0), 4),
            }

        # ── 2. Predict wash count quantile ────────────────────────────────
        X_new = np.array(feature_vector).reshape(1, -1)
        predicted_q = int(self.clf.predict(X_new)[0])
        proba_arr = self.clf.predict_proba(X_new)[0]
        classes = list(self.clf.classes_)
        proba = {int(c): round(float(p), 3) for c, p in zip(classes, proba_arr)}

        wash_range = self.wash_q_ranges[predicted_q]

        # ── 3. Quantile shift analysis ─────────────────────────────────────
        shift_opportunities = self._quantile_shift_analysis(
            feature_vector, feature_analysis, predicted_q
        )

        # ── 4. Profile comparison ──────────────────────────────────────────
        profile_comparison = self._profile_comparison(feature_analysis)

        # ── 4b. Strengths & Weaknesses (investment-report style) ─────────────
        strengths, weaknesses = self._strengths_weaknesses(feature_analysis)

        # ── 5. Build result dict ───────────────────────────────────────────
        result: Dict = {
            "predicted_wash_quantile": predicted_q,
            "predicted_wash_quantile_label": QUANTILE_LABELS[predicted_q],
            "predicted_wash_tier": QUANTILE_TIER_NAMES[predicted_q],
            "predicted_wash_range": {
                "min": round(wash_range[0]),
                "max": round(wash_range[1]),
                "label": f"{wash_range[0]:,.0f} – {wash_range[1]:,.0f} cars/yr",
            },
            "quantile_probabilities": proba,
            "wash_count_distribution": {
                q: {
                    "range": f"{self.wash_q_ranges[q][0]:,.0f} – {self.wash_q_ranges[q][1]:,.0f}",
                    "min": round(self.wash_q_ranges[q][0]),
                    "max": round(self.wash_q_ranges[q][1]),
                }
                for q in range(1, self.n_quantiles + 1)
            },
            "feature_analysis": feature_analysis,
            "shift_opportunities": shift_opportunities,
            "profile_comparison": profile_comparison,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "model_cv_accuracy": round(self.cv_accuracy, 3),
            "model_adj_accuracy": round(self.cv_adjacent_accuracy, 3),
            "features_available": sum(
                1 for fa in feature_analysis.values() if not fa.get("imputed", True)
            ),
        }

        # ── 6. LLM narrative and strengths/weaknesses (no fallbacks) ─────────
        if llm_narrative:
            result["narrative"] = self._generate_narrative(result)
            result["strengths_weaknesses_llm"] = self._generate_strengths_weaknesses_llm(result)

        return result

    def _strengths_weaknesses(
        self, feature_analysis: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Build investment-report style strengths (Q4-like) and weaknesses (Q1/Q2-like)."""
        strengths: List[Dict] = []
        weaknesses: List[Dict] = []
        for feat, fa in feature_analysis.items():
            if fa.get("imputed"):
                continue
            wq = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            label = fa.get("label", feat)
            val = fa.get("value")
            q4_med = fa.get("wash_q_q4_median")
            imp = fa.get("importance", 0)
            entry = {"label": label, "value": val, "q4_median": q4_med, "importance": imp}
            if wq == 4 or exceeds:
                entry["note"] = "exceeds Q4 typical" if exceeds else "matches Q4"
                strengths.append(entry)
            elif wq == 1 or wq == 2:
                entry["note"] = "Q1-level" if wq == 1 else "below median (Q2)"
                weaknesses.append(entry)
        strengths.sort(key=lambda x: -x["importance"])
        weaknesses.sort(key=lambda x: -x["importance"])
        return strengths, weaknesses

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _quantile_shift_analysis(
        self,
        base_vector: List[float],
        feature_analysis: Dict[str, Dict],
        current_q: int,
    ) -> List[Dict]:
        """
        For each feature not at its best adjusted quantile, simulate improving it
        to each higher quantile boundary (Q2→Q3→Q4 thresholds) and check whether
        the wash count prediction shifts up.  If no single-feature simulation shifts
        the prediction (common when ALL features are weak), fall back to showing the
        features ranked by how much the Q{current_q+1} probability increases.

        Returns up to 5 opportunities sorted by quantile gain, then importance.
        """
        if current_q >= self.n_quantiles:
            return []

        opportunities: List[Dict] = []
        # track best probability-lift even without a full quantile jump (fallback)
        prob_lifts: List[Dict] = []
        target_proba_class = min(current_q + 1, self.n_quantiles)
        target_cls_idx = (
            list(self.clf.classes_).index(target_proba_class)
            if target_proba_class in self.clf.classes_
            else None
        )
        base_proba_up = (
            self.clf.predict_proba(np.array(base_vector).reshape(1, -1))[0][target_cls_idx]
            if target_cls_idx is not None
            else 0.0
        )

        for i, feat in enumerate(self.feature_cols):
            fa = feature_analysis.get(feat, {})
            if fa.get("imputed", True):
                continue

            adj_q = fa.get("feature_quantile_adj", 2)
            if adj_q >= self.n_quantiles:
                continue

            dist = self.feature_dists[feat]
            direction = dist["direction"]
            val = fa["value"]

            # Try improving to each higher quantile boundary (Q→Q+1→Q+2→Q4)
            best_gain, best_sim_q, best_target_val = 0, current_q, val
            for target_adj_q in range(adj_q + 1, self.n_quantiles + 1):
                if direction == "higher":
                    target_val = float(dist["boundaries"][target_adj_q - 1])
                else:
                    target_val = float(dist["boundaries"][self.n_quantiles - target_adj_q + 1])

                sim_vector = base_vector.copy()
                sim_vector[i] = target_val
                sim_q = int(self.clf.predict(np.array(sim_vector).reshape(1, -1))[0])
                if sim_q - current_q > best_gain:
                    best_gain = sim_q - current_q
                    best_sim_q = sim_q
                    best_target_val = target_val
                    break  # take the first (smallest) improvement that shifts

            if best_gain > 0:
                change = abs(val - best_target_val)
                opportunities.append(
                    {
                        "feature": feat,
                        "label": FEATURE_LABELS.get(feat, feat),
                        "current_value": val,
                        "current_feature_quantile": adj_q,
                        "target_value": round(float(best_target_val), 2),
                        "change_needed": round(change, 2),
                        "change_direction": fa["shift_to_next_q_direction"],
                        "current_wash_q": current_q,
                        "simulated_wash_q": best_sim_q,
                        "q_gain": best_gain,
                        "importance": fa.get("importance", 0),
                    }
                )
            elif target_cls_idx is not None:
                # Fallback: measure probability lift toward next quantile
                sim_vector = base_vector.copy()
                if direction == "higher":
                    sim_vector[i] = float(dist["boundaries"][min(adj_q, self.n_quantiles - 1)])
                else:
                    sim_vector[i] = float(dist["boundaries"][max(self.n_quantiles - adj_q, 1)])
                sim_p = self.clf.predict_proba(np.array(sim_vector).reshape(1, -1))[0][target_cls_idx]
                lift = sim_p - base_proba_up
                if lift > 0.01:
                    prob_lifts.append(
                        {
                            "feature": feat,
                            "label": FEATURE_LABELS.get(feat, feat),
                            "current_value": val,
                            "current_feature_quantile": adj_q,
                            "target_value": round(float(sim_vector[i]), 2),
                            "change_needed": round(abs(val - sim_vector[i]), 2),
                            "change_direction": fa["shift_to_next_q_direction"],
                            "current_wash_q": current_q,
                            "simulated_wash_q": current_q,  # no full shift, but probability improves
                            "q_gain": 0,
                            "prob_lift": round(lift * 100, 1),
                            "importance": fa.get("importance", 0),
                        }
                    )

        opportunities.sort(key=lambda x: (-x["q_gain"], -x["importance"]))
        # If no hard quantile shifts found, show top probability-lift features
        if not opportunities and prob_lifts:
            prob_lifts.sort(key=lambda x: (-x["prob_lift"], -x["importance"]))
            for pl in prob_lifts[:5]:
                pl["note"] = (
                    f"Improves probability of Q{target_proba_class} by +{pl['prob_lift']:.1f}% "
                    f"(not enough alone to flip prediction, but directionally impactful)"
                )
            return prob_lifts[:5]

        return opportunities[:5]

    def _profile_comparison(self, feature_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        For each feature, show how the location value compares to the
        median profile of each wash count quantile tier.
        """
        comparison: Dict[str, Dict] = {}
        for feat, fa in feature_analysis.items():
            if fa.get("imputed", True) or fa.get("value") is None:
                continue
            val = fa["value"]
            direction = fa["direction"]
            feat_comp: Dict[str, Dict] = {}
            for q in range(1, self.n_quantiles + 1):
                prof_median = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
                if prof_median is None:
                    continue
                if direction == "higher":
                    align = "above" if val > prof_median else "below" if val < prof_median else "at"
                else:
                    align = "better" if val < prof_median else "worse" if val > prof_median else "at"
                feat_comp[f"Q{q}"] = {
                    "profile_median": round(float(prof_median), 2),
                    "alignment": align,
                }
            comparison[feat] = feat_comp
        return comparison

    # ── Narrative generation ──────────────────────────────────────────────────

    def _generate_narrative(self, result: Dict) -> str:
        """Generate rationale via LLM only. Returns unavailable message if LLM does not respond."""
        predicted_q = result["predicted_wash_quantile"]
        wash_range = result["predicted_wash_range"]["label"]
        proba = result["quantile_probabilities"]
        confidence = round(proba.get(predicted_q, 0) * 100, 1)

        # Top features by importance
        top_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("imputed")],
            key=lambda x: -x[1].get("importance", 0),
        )[:10]

        feat_lines = []
        for feat, fa in top_feats:
            label = fa["label"]
            val = fa["value"]
            wq = fa.get("wash_correlated_q", fa["feature_quantile_adj"])
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            pct = fa["adjusted_percentile"]
            q4_med = fa.get("wash_q_q4_median")
            q4_note = f" (Q4 typical: {q4_med:.1f})" if q4_med is not None else ""
            above_note = ", exceeds Q4 typical" if exceeds else ""
            feat_lines.append(
                f"  - {label}: {val:.1f} → WashQ{wq}{q4_note}{above_note} ({pct:.0f}th pctile)"
            )

        shift_lines = []
        for opp in result["shift_opportunities"]:
            shift_lines.append(
                f"  - {opp['label']}: {opp['change_direction']} by {opp['change_needed']:.1f} "
                f"({opp['current_value']:.1f} → {opp['target_value']:.1f}) "
                f"shifts wash prediction Q{opp['current_wash_q']} → Q{opp['simulated_wash_q']}"
            )

        # Q4 (high performer) typical values for comparison — no suggestions, just benchmarks
        q4_profile = self.quantile_profiles.get(4, {})
        q4_lines = []
        for feat, fa in top_feats:
            label = fa["label"]
            our_val = fa["value"]
            if feat in q4_profile:
                med = q4_profile[feat].get("median")
                q25 = q4_profile[feat].get("q25")
                q75 = q4_profile[feat].get("q75")
                if med is not None:
                    range_str = f"median {med:.1f}" if q25 is None or q75 is None else f"median {med:.1f} (typical range {q25:.1f}–{q75:.1f})"
                    q4_lines.append(f"  - {label}: this location {our_val:.1f} | Q4 high performers: {range_str}")
            else:
                q4_lines.append(f"  - {label}: this location {our_val:.1f} | Q4: n/a")

        q4_block = "\n".join(q4_lines) if q4_lines else "  (no Q4 profile)"

        adj_acc_pct = round(result.get("model_adj_accuracy", 0) * 100, 1)

        prompt = f"""You are a car wash site analyst. Write a short narrative that describes ONLY (1) which quartile this location is in and (2) how its feature values compare to what high-performing (Q4) sites typically have. Do NOT give recommendations, suggestions, or actionable improvements.

PREDICTION:
  Predicted Car Wash Quantile: Q{predicted_q} ({QUANTILE_LABELS[predicted_q]}) — {QUANTILE_TIER_NAMES[predicted_q]}
  Predicted Annual Car Wash Count: {wash_range}
  Model Confidence: {confidence}%
  Probabilities: Q1={proba.get(1,0)*100:.0f}%, Q2={proba.get(2,0)*100:.0f}%, Q3={proba.get(3,0)*100:.0f}%, Q4={proba.get(4,0)*100:.0f}%

THIS LOCATION'S TOP FEATURES:
{chr(10).join(feat_lines)}

Q4 (HIGH PERFORMER) TYPICAL VALUES — what high-performing car wash locations usually have:
{q4_block}

BENCHMARK CAR WASH COUNT RANGES:
  Q1: {result['wash_count_distribution'][1]['range']} cars/yr
  Q2: {result['wash_count_distribution'][2]['range']} cars/yr
  Q3: {result['wash_count_distribution'][3]['range']} cars/yr
  Q4: {result['wash_count_distribution'][4]['range']} cars/yr

Write 2–3 paragraphs of flowing prose (no bullet points, no headers). Content must be ONLY:
1. Which quartile this location is predicted to be in and what that means (expected car wash count range).
2. For each main feature: state this location's value, which car wash count tier (WashQ) that value matches (this tells us whether this feature level is typical of low or high-performing sites), and what Q4 sites' typical value is. Use the WashQ to say e.g. "this feature matches Q3 car wash sites' levels" or "exceeds even Q4 sites' typical level".
Do NOT include: recommendations, suggestions, actions to take, or what to do to improve."""

        text = self._call_llm(prompt, temperature=0.3)
        if text:
            return text
        return "[Rationale unavailable: LLM did not respond. Check LOCAL_LLM_URL and network.]"

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Call local LLM; return generated text or None on failure. No fallback."""
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from app.utils.llm import local_llm as llm_module

            response = llm_module.get_llm_response(
                prompt, reasoning_effort="medium", temperature=temperature
            )
            text = (response or {}).get("generated_text", "").strip()
            return text if text else None
        except Exception as exc:
            print(f"[LLM] Call failed: {exc}")
            return None

    def _generate_strengths_weaknesses_llm(self, result: Dict) -> str:
        """Generate strengths and weaknesses in investment-report style via LLM. No fallback."""
        strengths = result.get("strengths") or []
        weaknesses = result.get("weaknesses") or []
        pred_q = result["predicted_wash_quantile"]
        tier = result["predicted_wash_tier"]
        wash_range = result["predicted_wash_range"]["label"]

        s_lines = [
            f"  - {s['label']}: {s['value']:.1f} ({s['note']}" + (f", Q4 median {s['q4_median']:.1f})" if s.get("q4_median") is not None else ")")
            for s in strengths[:10]
        ]
        w_lines = [
            f"  - {w['label']}: {w['value']:.1f} ({w['note']}" + (f", Q4 median {w['q4_median']:.1f})" if w.get("q4_median") is not None else ")")
            for w in weaknesses[:10]
        ]

        prompt = f"""You are a car wash site investment analyst. Based on the data below, write a short investment-report-style summary with two clear sections: STRENGTHS and WEAKNESSES.

PREDICTION: This location is in quartile Q{pred_q} ({tier}), expected annual car wash count {wash_range}.

STRENGTHS (factors where this site matches or exceeds high-performing Q4 sites):
{chr(10).join(s_lines) if s_lines else "  (none)"}

WEAKNESSES (factors where this site is at Q1 or Q2 level vs. high performers):
{chr(10).join(w_lines) if w_lines else "  (none)"}

Write 2–4 short paragraphs total. First 1–2 paragraphs: key strengths in flowing prose (no bullet list in your output). Next 1–2 paragraphs: key weaknesses in flowing prose. Use a professional, analytical tone. Do not add recommendations or suggestions—only describe strengths and weaknesses. Output only the prose, no section headers like "Strengths:" or "Weaknesses:" if you can weave them naturally; otherwise use "Strengths:" and "Weaknesses:" as subheadings."""

        text = self._call_llm(prompt, temperature=0.3)
        if text:
            return text
        return "[Strengths & Weaknesses unavailable: LLM did not respond. Check LOCAL_LLM_URL and network.]"

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, result: Dict):
        """Print a structured quantile analysis report to stdout."""
        q = result["predicted_wash_quantile"]
        W = 80

        def header(title: str):
            print(f"\n{'─' * W}")
            print(f"  {title}")
            print(f"{'─' * W}")

        print("\n" + "═" * W)
        print("  QUANTILE PREDICTION REPORT")
        print("═" * W)

        print(f"\n  Predicted Car Wash Quantile : {result['predicted_wash_quantile_label']}")
        print(f"  Performance Tier            : {result['predicted_wash_tier']}")
        print(f"  Expected Annual Count       : {result['predicted_wash_range']['label']}")

        header("QUANTILE PROBABILITY DISTRIBUTION")
        for qi in range(1, self.n_quantiles + 1):
            p = result["quantile_probabilities"].get(qi, 0)
            bar = "█" * int(p * 32)
            marker = "  ◄ PREDICTED" if qi == q else ""
            label = f"Q{qi} {QUANTILE_TIER_NAMES[qi][:13]}"
            r = result["wash_count_distribution"][qi]["range"]
            print(f"  {label:<20s} [{bar:<32s}] {p * 100:5.1f}%  {r}{marker}")

        header(
            "FEATURE ANALYSIS — CORRELATED WITH CAR WASH COUNT\n"
            "  WashQ = which car wash performance tier your feature value matches\n"
            "  Q4 median = median value of this feature among TOP-25% car wash sites"
        )
        hdr = (
            f"  {'Feature':<36s} {'Value':>9s}  {'WashQ':>5s}  {'Pctile':>6s}"
            f"  {'Q4 median':>9s}  {'Importance':>10s}"
        )
        print(hdr)
        print("  " + "-" * (W - 2))

        sorted_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("imputed")],
            key=lambda x: -x[1].get("importance", 0),
        )
        for feat, fa in sorted_feats:
            label = fa["label"][:34]
            val = fa["value"]
            wq = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            pct = fa["adjusted_percentile"]
            imp = fa["importance"]
            q4_med = fa.get("wash_q_q4_median")
            wq_str = f"Q{wq}" if wq else " — "
            marker = "★" if exceeds else ("▲" if wq == 4 else ("▼" if wq == 1 else " "))
            q4_med_str = f"{q4_med:>9.1f}" if q4_med is not None else f"{'n/a':>9s}"
            print(
                f"  {label:<36s} {val:>9.1f}  {wq_str:>4s}{marker}  {pct:>5.1f}%"
                f"  {q4_med_str}  {imp:>10.1%}"
            )

        header("STRENGTHS & WEAKNESSES  (LLM-generated investment summary)")
        sw_llm = result.get("strengths_weaknesses_llm")
        if sw_llm:
            for para in sw_llm.split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()
        else:
            strengths = result.get("strengths") or []
            weaknesses = result.get("weaknesses") or []
            if strengths:
                print("  Strengths (Q4-level or better):")
                for s in strengths[:8]:
                    q4 = f", Q4 med {s['q4_median']:.1f}" if s.get("q4_median") is not None else ""
                    print(f"    • {s['label']}: {s['value']:.1f} ({s['note']}{q4})")
            if weaknesses:
                print("  Weaknesses (Q1/Q2-level vs. high performers):")
                for w in weaknesses[:8]:
                    q4 = f", Q4 med {w['q4_median']:.1f}" if w.get("q4_median") is not None else ""
                    print(f"    • {w['label']}: {w['value']:.1f} ({w['note']}{q4})")
            if not strengths and not weaknesses:
                print("  (Run with llm_narrative=True for LLM-generated strengths & weaknesses.)")

        if result["shift_opportunities"]:
            # Determine if these are hard quantile shifts or probability-lift fallbacks
            hard_shifts = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) > 0]
            soft_shifts = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) == 0]

            if hard_shifts:
                header("QUANTILE SHIFT OPPORTUNITIES")
                print(
                    "  Improving these features (individually) shifts the predicted wash quantile up:\n"
                )
                print(
                    f"  {'Feature':<35s} {'Current':>9s} {'Target':>9s} {'Change':>8s}  {'Wash Q Shift'}"
                )
                print("  " + "-" * (W - 2))
                for opp in hard_shifts:
                    label = opp["label"][:33]
                    dirn = "+" if opp["change_direction"] == "increase" else "−"
                    shift = f"Q{opp['current_wash_q']} → Q{opp['simulated_wash_q']}"
                    print(
                        f"  {label:<35s} {opp['current_value']:>9.1f} {opp['target_value']:>9.1f} "
                        f"{dirn}{opp['change_needed']:>7.1f}  {shift}"
                    )

            if soft_shifts:
                header("TOP IMPROVEMENT PRIORITIES  (probability-lift toward next quantile)")
                print(
                    "  No single feature alone shifts the prediction, but these improvements\n"
                    "  each increase the probability of a higher quantile:\n"
                )
                print(
                    f"  {'Feature':<35s} {'Current':>9s} {'Target':>9s} {'Change':>8s}  {'Prob Lift'}"
                )
                print("  " + "-" * (W - 2))
                for opp in soft_shifts:
                    label = opp["label"][:33]
                    dirn = "+" if opp["change_direction"] == "increase" else "−"
                    lift = opp.get("prob_lift", 0)
                    print(
                        f"  {label:<35s} {opp['current_value']:>9.1f} {opp['target_value']:>9.1f} "
                        f"{dirn}{opp['change_needed']:>7.1f}  +{lift:.1f}% prob"
                    )
        else:
            print("\n  No shift opportunities — location is in top quantile or already well-optimized.")

        header("FEATURE vs. QUANTILE PROFILE COMPARISON")
        print(
            f"  Shows how your feature value compares to the median of each car wash quantile tier.\n"
        )
        print(
            f"  {'Feature':<38s} {'Your Value':>10s}  "
            + "  ".join(f"{'Q'+str(qi)+' med':>8s}" for qi in range(1, self.n_quantiles + 1))
        )
        print("  " + "-" * (W - 2))
        for feat, fa in sorted_feats[:12]:
            label = fa["label"][:36]
            val = fa["value"]
            comp = result["profile_comparison"].get(feat, {})
            medians_str = "  ".join(
                f"{comp.get(f'Q{qi}', {}).get('profile_median', 0.0):>8.1f}"
                for qi in range(1, self.n_quantiles + 1)
            )
            print(f"  {label:<38s} {val:>10.1f}  {medians_str}")

        if result.get("narrative"):
            header("NARRATIVE SUMMARY")
            for para in result["narrative"].split("\n\n"):
                wrapped = textwrap.fill(para.strip(), width=W - 4, initial_indent="  ", subsequent_indent="  ")
                print(wrapped)
                print()

        print(
            f"\n  Model CV Accuracy: exact {result['model_cv_accuracy']:.1%}  |  "
            f"within-1-quantile {result.get('model_adj_accuracy', 0):.1%}  |  "
            f"Features Provided: {result['features_available']}/{len(self.feature_cols)}\n"
            f"  Note: Predictions are based on 19 environmental/location features. "
            f"Actual wash volume also depends on traffic, pricing, and operations."
        )
        print("═" * W)

    def report_to_string(self, result: Dict) -> str:
        """Build the full report (same content as print_report) as a single string."""
        lines: List[str] = []
        q = result["predicted_wash_quantile"]
        W = 80

        def header(title: str):
            lines.append("")
            lines.append("─" * W)
            lines.append(f"  {title}")
            lines.append("─" * W)

        lines.append("")
        lines.append("═" * W)
        lines.append("  QUANTILE PREDICTION REPORT")
        lines.append("═" * W)
        lines.append("")
        lines.append(f"  Predicted Car Wash Quantile : {result['predicted_wash_quantile_label']}")
        lines.append(f"  Performance Tier            : {result['predicted_wash_tier']}")
        lines.append(f"  Expected Annual Count       : {result['predicted_wash_range']['label']}")

        header("QUANTILE PROBABILITY DISTRIBUTION")
        for qi in range(1, self.n_quantiles + 1):
            p = result["quantile_probabilities"].get(qi, 0)
            bar = "█" * int(p * 32)
            marker = "  ◄ PREDICTED" if qi == q else ""
            label = f"Q{qi} {QUANTILE_TIER_NAMES[qi][:13]}"
            r = result["wash_count_distribution"][qi]["range"]
            lines.append(f"  {label:<20s} [{bar:<32s}] {p * 100:5.1f}%  {r}{marker}")

        header(
            "FEATURE ANALYSIS — CORRELATED WITH CAR WASH COUNT\n"
            "  WashQ = which car wash performance tier your feature value matches\n"
            "  Q4 median = median value of this feature among TOP-25% car wash sites"
        )
        lines.append(
            f"  {'Feature':<36s} {'Value':>9s}  {'WashQ':>5s}  {'Pctile':>6s}"
            f"  {'Q4 median':>9s}  {'Importance':>10s}"
        )
        lines.append("  " + "-" * (W - 2))

        sorted_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("imputed")],
            key=lambda x: -x[1].get("importance", 0),
        )
        for feat, fa in sorted_feats:
            label = fa["label"][:34]
            val = fa["value"]
            wq = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            pct = fa["adjusted_percentile"]
            imp = fa["importance"]
            q4_med = fa.get("wash_q_q4_median")
            wq_str = f"Q{wq}" if wq else " — "
            marker = "★" if exceeds else ("▲" if wq == 4 else ("▼" if wq == 1 else " "))
            q4_med_str = f"{q4_med:>9.1f}" if q4_med is not None else f"{'n/a':>9s}"
            lines.append(
                f"  {label:<36s} {val:>9.1f}  {wq_str:>4s}{marker}  {pct:>5.1f}%"
                f"  {q4_med_str}  {imp:>10.1%}"
            )

        header("STRENGTHS & WEAKNESSES  (LLM-generated investment summary)")
        sw_llm = result.get("strengths_weaknesses_llm")
        if sw_llm:
            for para in sw_llm.split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        lines.append(line)
                    lines.append("")
        else:
            strengths = result.get("strengths") or []
            weaknesses = result.get("weaknesses") or []
            if strengths:
                lines.append("  Strengths (Q4-level or better):")
                for s in strengths[:8]:
                    q4 = f", Q4 med {s['q4_median']:.1f}" if s.get("q4_median") is not None else ""
                    lines.append(f"    • {s['label']}: {s['value']:.1f} ({s['note']}{q4})")
            if weaknesses:
                lines.append("  Weaknesses (Q1/Q2-level vs. high performers):")
                for w in weaknesses[:8]:
                    q4 = f", Q4 med {w['q4_median']:.1f}" if w.get("q4_median") is not None else ""
                    lines.append(f"    • {w['label']}: {w['value']:.1f} ({w['note']}{q4})")
            if not strengths and not weaknesses:
                lines.append("  (Run with llm_narrative=True for LLM-generated strengths & weaknesses.)")

        if result["shift_opportunities"]:
            hard_shifts = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) > 0]
            soft_shifts = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) == 0]
            if hard_shifts:
                header("QUANTILE SHIFT OPPORTUNITIES")
                lines.append(
                    "  Improving these features (individually) shifts the predicted wash quantile up:"
                )
                lines.append("")
                lines.append(
                    f"  {'Feature':<35s} {'Current':>9s} {'Target':>9s} {'Change':>8s}  {'Wash Q Shift'}"
                )
                lines.append("  " + "-" * (W - 2))
                for opp in hard_shifts:
                    label = opp["label"][:33]
                    dirn = "+" if opp["change_direction"] == "increase" else "−"
                    shift = f"Q{opp['current_wash_q']} → Q{opp['simulated_wash_q']}"
                    lines.append(
                        f"  {label:<35s} {opp['current_value']:>9.1f} {opp['target_value']:>9.1f} "
                        f"{dirn}{opp['change_needed']:>7.1f}  {shift}"
                    )
            if soft_shifts:
                header("TOP IMPROVEMENT PRIORITIES  (probability-lift toward next quantile)")
                lines.append(
                    "  No single feature alone shifts the prediction, but these improvements"
                )
                lines.append("  each increase the probability of a higher quantile:")
                lines.append("")
                lines.append(
                    f"  {'Feature':<35s} {'Current':>9s} {'Target':>9s} {'Change':>8s}  {'Prob Lift'}"
                )
                lines.append("  " + "-" * (W - 2))
                for opp in soft_shifts:
                    label = opp["label"][:33]
                    dirn = "+" if opp["change_direction"] == "increase" else "−"
                    lift = opp.get("prob_lift", 0)
                    lines.append(
                        f"  {label:<35s} {opp['current_value']:>9.1f} {opp['target_value']:>9.1f} "
                        f"{dirn}{opp['change_needed']:>7.1f}  +{lift:.1f}% prob"
                    )
        else:
            lines.append("")
            lines.append(
                "  No shift opportunities — location is in top quantile or already well-optimized."
            )

        header("FEATURE vs. QUANTILE PROFILE COMPARISON")
        lines.append(
            "  Shows how your feature value compares to the median of each car wash quantile tier."
        )
        lines.append("")
        lines.append(
            f"  {'Feature':<38s} {'Your Value':>10s}  "
            + "  ".join(f"{'Q'+str(qi)+' med':>8s}" for qi in range(1, self.n_quantiles + 1))
        )
        lines.append("  " + "-" * (W - 2))
        for feat, fa in sorted_feats[:12]:
            label = fa["label"][:36]
            val = fa["value"]
            comp = result["profile_comparison"].get(feat, {})
            medians_str = "  ".join(
                f"{comp.get(f'Q{qi}', {}).get('profile_median', 0.0):>8.1f}"
                for qi in range(1, self.n_quantiles + 1)
            )
            lines.append(f"  {label:<38s} {val:>10.1f}  {medians_str}")

        if result.get("narrative"):
            header("NARRATIVE SUMMARY")
            for para in result["narrative"].split("\n\n"):
                wrapped = textwrap.fill(
                    para.strip(), width=W - 4, initial_indent="  ", subsequent_indent="  "
                )
                lines.append(wrapped)
                lines.append("")

        lines.append("")
        lines.append(
            f"  Model CV Accuracy: exact {result['model_cv_accuracy']:.1%}  |  "
            f"within-1-quantile {result.get('model_adj_accuracy', 0):.1%}  |  "
            f"Features Provided: {result['features_available']}/{len(self.feature_cols)}"
        )
        lines.append("")
        lines.append(
            "  Note: Predictions are based on 19 environmental/location features. "
            "Actual wash volume also depends on traffic, pricing, and operations."
        )
        lines.append("")
        lines.append("═" * W)
        return "\n".join(lines)

    def save_report(self, result: Dict, path: Optional[Path] = None) -> Path:
        """
        Save the full report (comparison, narrative, all sections) to a text file.

        Parameters
        ----------
        result : dict
            Output of analyze().
        path : Path or str, optional
            Output file path. Defaults to distribution_plots/quantile_report_<timestamp>.txt

        Returns
        -------
        Path to the saved file.
        """
        from datetime import datetime

        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(__file__).parent / "distribution_plots" / f"quantile_report_{ts}.txt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.report_to_string(result)
        path.write_text(content, encoding="utf-8")
        print(f"\n✓ Full report saved: {path}")
        return path

    # ── Visualization ─────────────────────────────────────────────────────────

    def plot_feature_quantiles(
        self,
        result: Dict,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (18, 14),
    ):
        """
        Plot a grid of feature distributions with the new location marked,
        plus a summary panel showing wash quantile probabilities.

        Parameters
        ----------
        result : dict
            Output of `analyze()`.
        output_path : Path, optional
            Where to save the figure. Defaults to distribution_plots/quantile_analysis.png.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("[Skip] matplotlib not installed.")
            return

        if output_path is None:
            output_path = Path(__file__).parent / "distribution_plots" / "quantile_analysis.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort features by importance
        sorted_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("imputed")],
            key=lambda x: -x[1].get("importance", 0),
        )

        n_feat = len(sorted_feats)
        cols = 4
        rows = (n_feat + cols - 1) // cols + 1  # +1 for summary row

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor("#f8f9fa")
        axes_flat = axes.flatten()

        # Colour scheme for quantile bands
        q_colours = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]  # Q1–Q4
        predicted_q = result["predicted_wash_quantile"]

        # ── Summary panel (top-left, spanning first row) ───────────────────
        # Probability bar chart in axes_flat[0]
        ax_prob = axes_flat[0]
        qs = list(range(1, self.n_quantiles + 1))
        probs = [result["quantile_probabilities"].get(q, 0) * 100 for q in qs]
        bars = ax_prob.bar(
            [f"Q{q}" for q in qs],
            probs,
            color=[q_colours[q - 1] for q in qs],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, p in zip(bars, probs):
            ax_prob.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{p:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        ax_prob.set_title(
            f"Predicted Wash Quantile: Q{predicted_q} — {QUANTILE_TIER_NAMES[predicted_q]}\n"
            f"({result['predicted_wash_range']['label']})",
            fontsize=10,
            fontweight="bold",
        )
        ax_prob.set_ylabel("Probability (%)", fontsize=8)
        ax_prob.set_ylim(0, 105)
        ax_prob.axhline(50, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_prob.tick_params(labelsize=9)
        ax_prob.set_facecolor("#f0f0f0")

        # Shift opportunities text panel in axes_flat[1]
        ax_shift = axes_flat[1]
        ax_shift.axis("off")
        shift_text = "QUANTILE SHIFT OPPORTUNITIES\n\n"
        for opp in result["shift_opportunities"][:4]:
            dirn = "▲" if opp["change_direction"] == "increase" else "▼"
            shift_text += (
                f"{dirn} {opp['label'][:28]}\n"
                f"  {opp['current_value']:.1f} → {opp['target_value']:.1f}  "
                f"(Q{opp['current_wash_q']}→Q{opp['simulated_wash_q']})\n\n"
            )
        if not result["shift_opportunities"]:
            shift_text += "No shift opportunities found.\nLocation already well-optimised."
        ax_shift.text(
            0.05, 0.95, shift_text,
            transform=ax_shift.transAxes,
            va="top", ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107"),
        )

        # Wash count quantile benchmark in axes_flat[2]
        ax_bench = axes_flat[2]
        ax_bench.axis("off")
        bench_text = "BENCHMARK RANGES (cars/yr)\n\n"
        for qi in range(1, self.n_quantiles + 1):
            r = result["wash_count_distribution"][qi]["range"]
            marker = " ◄ YOU" if qi == predicted_q else ""
            bench_text += f"Q{qi}: {r}{marker}\n"
        bench_text += f"\nModel CV Acc: {result['model_cv_accuracy']:.1%}"
        ax_bench.text(
            0.05, 0.95, bench_text,
            transform=ax_bench.transAxes,
            va="top", ha="left",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#d4edda", edgecolor="#28a745"),
        )

        # Hide 4th summary cell
        axes_flat[3].axis("off")

        # ── Feature distribution panels ────────────────────────────────────
        for idx, (feat, fa) in enumerate(sorted_feats):
            ax = axes_flat[cols + idx]
            dist = self.feature_dists[feat]
            data = dist["data"]
            val = fa["value"]
            direction = fa["direction"]
            wq = fa.get("wash_correlated_q", 2)
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            group_medians = fa.get("wash_q_group_medians", {})

            # Histogram (plain — shading via group median spans below)
            ax.hist(
                data,
                bins=min(30, max(10, len(np.unique(data)))),
                color="#aec7e8",
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
                zorder=1,
            )

            # Draw vertical lines for each car wash group median (Q1–Q4)
            for qi in range(1, self.n_quantiles + 1):
                gm = group_medians.get(qi)
                if gm is not None:
                    ax.axvline(
                        gm,
                        color=q_colours[qi - 1],
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.85,
                        zorder=2,
                        label=f"Q{qi} med:{gm:.1f}",
                    )

            # Mark the new location value (★ exceeds Q4 → use Q4 colour)
            eff_wq = 4 if exceeds else wq
            marker_colour = q_colours[eff_wq - 1] if eff_wq else "#333333"
            ax.axvline(
                val,
                color=marker_colour,
                linestyle="-",
                linewidth=2.8,
                zorder=4,
                label=f"You:{val:.1f}",
            )

            dir_arrow = "↑ hi=better" if direction == "higher" else "↓ lo=better"
            imp = fa["importance"]
            wq_label = f"WashQ{wq}{'★' if exceeds else ''}"

            ax.set_title(
                f"{fa['label'][:30]}\n"
                f"{wq_label} | {fa['adjusted_percentile']:.0f}th pctile | imp {imp:.1%}",
                fontsize=7.5,
                fontweight="bold" if imp > 0.06 else "normal",
            )
            ax.set_ylabel(f"n={len(data)}\n{dir_arrow}", fontsize=6.5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5.5, loc="upper right", ncol=1)
            ax.set_facecolor("#f8f9fa")

        # Hide unused axes
        for j in range(cols + n_feat, len(axes_flat)):
            axes_flat[j].axis("off")

        # Legend: dashed lines = group medians for each wash tier
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color=q_colours[q - 1], linestyle="--", linewidth=1.5,
                   label=f"Q{q} median ({QUANTILE_TIER_NAMES[q]})")
            for q in range(1, self.n_quantiles + 1)
        ] + [
            Line2D([0], [0], color="black", linestyle="-", linewidth=2.5, label="Your value")
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.01),
        )

        fig.suptitle(
            f"Feature Analysis Correlated with Car Wash Count  —  Predicted: Q{predicted_q} "
            f"({result['predicted_wash_range']['label']})\n"
            f"Dashed lines = median of that feature for Q1/Q2/Q3/Q4 car wash sites. "
            f"Solid line = your value. WashQ = tier your value matches.",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n✓ Quantile analysis plot saved: {output_path}")
        return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = QuantilePredictor()

    # New location: mixed profile (good weather, decent retail, some weak factors)
    example: Dict[str, float] = {}
    for feat in predictor.feature_cols:
        dist = predictor.feature_dists[feat]
        strong = {
            "weather_total_sunshine_hours",
            "weather_days_pleasant_temp",
            "other_grocery_count_1mile",
            "count_food_joints_0_5miles (0.5 mile)",
        }
        weak = {
            "weather_total_precipitation_mm",
            "weather_total_snowfall_cm",
            "weather_days_below_freezing",
        }
        if feat in strong:
            example[feat] = float(np.percentile(dist["data"], 75))
        elif feat in weak:
            example[feat] = float(np.percentile(dist["data"], 60))
        else:
            example[feat] = float(dist["median"])

    # LLM-only rationale and strengths/weaknesses (no fallbacks)
    result = predictor.analyze(example, llm_narrative=True)
    predictor.print_report(result)
    predictor.plot_feature_quantiles(result)

    # Save full report for new location (includes accuracy, rationale, LLM S/W)
    report_path = Path(__file__).parent / "distribution_plots" / "quantile_full_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(predictor.report_to_string(result), encoding="utf-8")
    print(f"\n✓ Full report (accuracy + rationale + S/W) saved: {report_path}")
