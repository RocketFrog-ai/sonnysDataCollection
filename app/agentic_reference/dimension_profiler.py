"""
Dimension-Specific Profiler
=============================

Wraps QuantileProfiler to score each business dimension independently,
then combines via majority voting for an overall verdict.

Dimensions (from reference guides):
  1. Weather (8 features)
  2. Traffic (23 features)
  3. Competition (3 features)
  4. Infrastructure (2 features)
  5. Retail Proximity (14 features)

Key insight: a site can be "Weather: High + Traffic: Low" — information
that is impossible to see with a single combined score.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

try:
    from .profiler_engine import QuantileProfiler, DIMENSION_GROUPS, CATEGORY_LABELS
except ImportError:
    from profiler_engine import QuantileProfiler, DIMENSION_GROUPS, CATEGORY_LABELS


class DimensionProfiler:
    """
    Per-dimension quantile profiling with majority voting.
    """

    def __init__(self, profiler: QuantileProfiler):
        self.profiler = profiler
        self.dimensions = DIMENSION_GROUPS
        self.category_labels = CATEGORY_LABELS

        # Pre-compute how well each dimension discriminates Low vs High
        self.dimension_strength: Dict[str, Dict] = self._compute_dimension_strength()

    # ── Dimension-level scoring ──────────────────────────────────

    def score_dimension(
        self, dimension: str, location_features: Dict[str, float]
    ) -> Dict:
        """
        Score a single dimension for the given location features.

        Returns:
          {
            "scores": {"Low Performing": 4.2, ...},
            "predicted": "High Performing",
            "fit_score": 38.5,
            "features_scored": 6,
            "feature_details": { feature_name: { value, best_fit, ranges } }
          }
        """
        dim_features = self.dimensions.get(dimension, [])
        # Only score features that (a) belong to this dimension and
        # (b) exist in both the input and the profiler's ranges
        relevant = [
            f
            for f in dim_features
            if f in location_features
            and location_features[f] is not None
            and f in self.profiler.feature_ranges
        ]

        if not relevant:
            return {
                "scores": {},
                "predicted": "Insufficient Data",
                "fit_score": 0.0,
                "features_scored": 0,
                "feature_details": {},
            }

        scores = self.profiler.score_location(location_features, feature_subset=relevant)
        total = sum(scores.values())
        predicted = max(scores, key=scores.get) if total > 0 else "Insufficient Data"
        fit_pct = round(100 * scores[predicted] / total, 1) if total > 0 else 0.0

        # Per-feature breakdown
        details: Dict[str, Dict] = {}
        for feat in relevant:
            val = location_features[feat]
            per_tier = self.profiler.score_feature(feat, float(val))
            best = max(per_tier, key=per_tier.get) if per_tier else "N/A"
            details[feat] = {
                "value": val,
                "best_fit": best,
                "tier_scores": per_tier,
                "ranges": {
                    cat: self.profiler.feature_ranges[feat].get(cat, {})
                    for cat in self.category_labels
                },
            }

        return {
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "predicted": predicted,
            "fit_score": fit_pct,
            "features_scored": len(relevant),
            "feature_details": details,
        }

    def score_all_dimensions(
        self, location_features: Dict[str, float]
    ) -> Dict[str, Dict]:
        """Score every dimension. Returns { dim_name: score_dict }."""
        return {
            dim: self.score_dimension(dim, location_features)
            for dim in self.dimensions
        }

    # ── Majority voting ──────────────────────────────────────────

    def majority_vote(self, dim_results: Dict[str, Dict]) -> Dict:
        """
        Each dimension with data casts one vote for its predicted tier.
        Returns majority tier + vote breakdown.
        """
        votes: Dict[str, int] = {cat: 0 for cat in self.category_labels}
        voting_dims: List[str] = []

        for dim, res in dim_results.items():
            pred = res.get("predicted", "")
            if pred in votes:
                votes[pred] += 1
                voting_dims.append(dim)

        winner = max(votes, key=votes.get) if any(votes.values()) else "Undetermined"
        total_votes = sum(votes.values())

        return {
            "category": winner,
            "agreement": votes.get(winner, 0),
            "total_dimensions": total_votes,
            "vote_breakdown": votes,
            "voting_dimensions": voting_dims,
        }

    # ── Strengths & weaknesses ───────────────────────────────────

    def get_strengths_weaknesses(
        self, dim_results: Dict[str, Dict]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Classify dimensions as strengths (High), weaknesses (Low),
        or neutral (Average / insufficient data).
        """
        strengths, weaknesses, neutrals = [], [], []
        for dim, res in dim_results.items():
            pred = res.get("predicted", "")
            if "High" in pred:
                strengths.append(dim)
            elif "Low" in pred:
                weaknesses.append(dim)
            elif "Average" in pred:
                neutrals.append(dim)
        return strengths, weaknesses, neutrals

    # ── Discriminatory power ─────────────────────────────────────

    def _compute_dimension_strength(self) -> Dict[str, Dict]:
        """
        Measure how well each dimension separates Low from High tiers
        in the historical data.

        Uses Cohen's d (effect size) between Low and High tier medians
        for each feature, then averages across features in the dimension.
        """
        strength: Dict[str, Dict] = {}

        for dim, features in self.dimensions.items():
            effect_sizes: List[float] = []

            for feat in features:
                if feat not in self.profiler.feature_ranges:
                    continue
                low_r = self.profiler.feature_ranges[feat].get("Low Performing")
                high_r = self.profiler.feature_ranges[feat].get("High Performing")
                if not low_r or not high_r:
                    continue

                diff = abs(high_r["q50"] - low_r["q50"])
                # Use pooled IQR as denominator for a robust effect size
                pooled_iqr = (
                    (high_r["q75"] - high_r["q25"])
                    + (low_r["q75"] - low_r["q25"])
                ) / 2
                if pooled_iqr > 0:
                    effect_sizes.append(diff / pooled_iqr)

            avg_es = sum(effect_sizes) / len(effect_sizes) if effect_sizes else 0.0
            strength[dim] = {
                "average_effect_size": round(avg_es, 4),
                "features_measured": len(effect_sizes),
            }

        # Assign rank
        ranked = sorted(
            strength.keys(),
            key=lambda d: strength[d]["average_effect_size"],
            reverse=True,
        )
        for i, dim in enumerate(ranked, 1):
            strength[dim]["rank"] = i

        return strength

    # ── Formatted reports ────────────────────────────────────────

    def format_dimension_report(
        self,
        dim_results: Dict[str, Dict],
        include_details: bool = False,
    ) -> str:
        """Human-readable dimension-by-dimension report."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  DIMENSION-SPECIFIC PROFILING RESULTS")
        lines.append("=" * 60)

        for dim, res in dim_results.items():
            pred = res.get("predicted", "N/A")
            fit = res.get("fit_score", 0)
            n = res.get("features_scored", 0)

            icon = {"High Performing": "✓", "Low Performing": "✗"}.get(pred, "○")
            lines.append(f"\n  {icon} {dim}: {pred} (fit: {fit:.1f}%, {n} features)")

            if include_details and res.get("feature_details"):
                for feat, fd in res["feature_details"].items():
                    val = fd["value"]
                    bf = fd["best_fit"]
                    rng = fd["ranges"].get(bf, {})
                    q25 = rng.get("q25")
                    q50 = rng.get("q50")
                    q75 = rng.get("q75")
                    if q25 is not None:
                        pos = (
                            "within IQR"
                            if q25 <= val <= q75
                            else ("below IQR" if val < q25 else "above IQR")
                        )
                        lines.append(
                            f"      {feat} = {val:,.1f} → {bf} "
                            f"({pos}, Q25={q25:,.1f}, Med={q50:,.1f}, Q75={q75:,.1f})"
                        )

        # Overall verdict
        vote = self.majority_vote(dim_results)
        strengths, weaknesses, neutrals = self.get_strengths_weaknesses(dim_results)

        lines.append("\n" + "=" * 60)
        lines.append("  OVERALL VERDICT")
        lines.append("=" * 60)
        lines.append(
            f"  Majority vote: {vote['category']} "
            f"({vote['agreement']}/{vote['total_dimensions']} dimensions agree)"
        )
        lines.append(f"  Strengths : {', '.join(strengths) if strengths else 'None'}")
        lines.append(f"  Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}")
        lines.append(f"  Neutral   : {', '.join(neutrals) if neutrals else 'None'}")

        return "\n".join(lines)

    def format_strength_ranking(self) -> str:
        """Formatted ranking of dimension discriminatory power."""
        lines = ["=" * 60, "  DIMENSION DISCRIMINATORY POWER", "=" * 60]
        ranked = sorted(
            self.dimension_strength.items(),
            key=lambda x: x[1]["average_effect_size"],
            reverse=True,
        )
        for dim, info in ranked:
            bar = "█" * int(info["average_effect_size"] * 20)
            lines.append(
                f"  #{info['rank']}  {dim:20s}  "
                f"effect_size={info['average_effect_size']:.3f}  {bar}"
            )
        return "\n".join(lines)
