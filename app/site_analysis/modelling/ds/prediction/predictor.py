"""ExtraTrees wash-volume tiers: train, `analyze()`, reporting via `QuantileReportingMixin`."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.site_analysis.modelling.ds.prediction.constants import (
    DISPLAY_ONLY_FEATURES,
    FEATURE_DIRECTIONS,
    FEATURE_LABELS,
    FEATURE_SIGNAL,
    ML_FEATURE_ORDER,
    QUANTILE_LABELS,
    QUANTILE_TIER_NAMES,
    SIGNAL_THRESHOLD,
    TIER_PRESETS,
)
from app.site_analysis.modelling.ds.prediction.data import _load_and_merge
from app.site_analysis.modelling.ds.prediction.distributions import (
    _adj_quantile,
    _assign_raw_quantile,
    _detect_shape,
    _next_better_boundary,
    _quantile_boundaries,
)
from app.site_analysis.modelling.ds.prediction.paths import datasets_dir
from app.site_analysis.modelling.ds.prediction.reporting import QuantileReportingMixin


class QuantilePredictorV4(QuantileReportingMixin):
    """Calibrated ExtraTrees tier classifier over portfolio features; use `analyze()` then reporting helpers."""

    def __init__(
        self,
        excel_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        tier_strategy: str = "4-class-wide-middle",
        n_quantiles: Optional[int] = None,  # inferred from strategy if None
        use_control_sites_only: bool = False,
    ):
        canonical_path = datasets_dir() / "final_merged_dataset.csv"
        
        if tier_strategy not in TIER_PRESETS:
            print(f"Warning: Unknown tier_strategy '{tier_strategy}'. Falling back to '4-class-wide-middle'.")
            tier_strategy = "4-class-wide-middle"
        
        self.tier_strategy = tier_strategy
        self.percentile_splits = TIER_PRESETS[tier_strategy]
        self.n_quantiles = n_quantiles or len(self.percentile_splits)

        # Single-source mode: canonical merged dataset only.
        if not canonical_path.exists():
            raise FileNotFoundError(f"Canonical dataset not found: {canonical_path}")


        print("Loading and merging data…")
        self.df = _load_and_merge(canonical_path)

        if use_control_sites_only and "client_id" in self.df.columns:
            before = len(self.df)
            self.df = self.df[
                ~self.df["client_id"].astype(str).str.contains(
                    "Controls Training", case=False, na=False
                )
            ].copy()
            print(f"  Control-sites only: {len(self.df)} sites (was {before})")

        # ML feature cols: canonical order from ML_FEATURE_ORDER, filtered to columns present.
        # carwash_type_encoded is DISPLAY_ONLY — already encoded by effective_capacity.
        # The canonical order matters for ExtraTrees random_state reproducibility:
        # feature index assignments affect which columns are sampled at each split.
        available = set(self.df.columns) - {
            "Address", "current_count", "location_id", "client_id",
            "street", "city", "zip", "region", "state",
            "wash_q", "_match_type", "primary_carwash_type",
            *DISPLAY_ONLY_FEATURES,
        }
        self.feature_cols: List[str] = [
            f for f in ML_FEATURE_ORDER if f in available and f in FEATURE_LABELS
        ]

        print(f"  {len(self.df)} sites  |  {len(self.feature_cols)} ML features  |  "
              f"{len(DISPLAY_ONLY_FEATURES)} display-only features")

        self._build_feature_distributions()
        self._build_wash_quantiles()
        self._build_quantile_profiles()
        self._train_classifier()
        self._train_volume_regressors()
        print("✓ QuantilePredictorV4 ready\n")

    # ── Build helpers ─────────────────────────────────────────────────────────

    def _build_feature_distributions(self):
        # Build distributions for ALL features in FEATURE_LABELS (including display-only)
        self.feature_dists: Dict[str, Dict] = {}
        all_display_feats = list(self.feature_cols) + [
            f for f in DISPLAY_ONLY_FEATURES if f in self.df.columns
        ]
        for feat in all_display_feats:
            data = self.df[feat].dropna().values.astype(float)
            if len(data) == 0:
                continue
            shape = _detect_shape(data)
            boundaries = _quantile_boundaries(data, self.n_quantiles)
            direction = FEATURE_DIRECTIONS.get(feat, "higher")
            signal = FEATURE_SIGNAL.get(feat, 0.0)
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
                "signal": signal,
            }

    def _build_wash_quantiles(self):
        counts = self.df["current_count"].dropna().values
        self.wash_shape = _detect_shape(counts)
        
        # Calculate boundaries based on percentile splits (e.g. 20-30-30-20)
        cum_percentiles = np.cumsum([0] + self.percentile_splits)
        self.wash_boundaries = np.percentile(counts, cum_percentiles)
        # Ensure exact min/max from dataset if percentile interpolation shifts them
        self.wash_boundaries[0] = np.min(counts)
        self.wash_boundaries[-1] = np.max(counts)

        self.df["wash_q"] = pd.cut(
            self.df["current_count"],
            bins=self.wash_boundaries,
            labels=list(range(1, self.n_quantiles + 1)),
            include_lowest=True,
        ).astype("Int64")

        self.wash_q_ranges: Dict[int, Tuple[float, float]] = {
            q: (float(self.wash_boundaries[q - 1]), float(self.wash_boundaries[q]))
            for q in range(1, self.n_quantiles + 1)
        }
        print(f"\nCar wash count tier ranges (strategy: {self.tier_strategy}):")
        for q, (lo, hi) in self.wash_q_ranges.items():
            n = int((self.df["wash_q"] == q).sum())
            print(f"  Q{q}: {lo:>10,.0f} - {hi:>10,.0f} cars/yr  (n={n})")

    def _build_quantile_profiles(self):
        # Build profiles for ALL features (including display-only) for analysis
        all_feats = list(self.feature_cols) + [
            f for f in DISPLAY_ONLY_FEATURES if f in self.df.columns
        ]
        self.quantile_profiles: Dict[int, Dict[str, Dict]] = {}
        self.tier_medians: Dict[int, float] = {}
        self.tier_means: Dict[int, float] = {}
        
        counts = self.df["current_count"].dropna()
        
        for q in range(1, self.n_quantiles + 1):
            mask = self.df["wash_q"] == q
            subset = self.df[mask]
            
            # Store median wash count for this tier (used for weighted predictions)
            q_counts = counts[mask]
            self.tier_medians[q] = float(q_counts.median()) if not q_counts.empty else 0.0
            self.tier_means[q] = float(q_counts.mean()) if not q_counts.empty else 0.0
            
            profile: Dict[str, Dict] = {}
            for feat in all_feats:
                vals = subset[feat].dropna()
                if len(vals) > 0:
                    profile[feat] = {
                        "median": float(vals.median()),
                        "mean":   float(vals.mean()),
                        "q25":    float(vals.quantile(0.25)),
                        "q75":    float(vals.quantile(0.75)),
                        "n":      len(vals),
                    }
            self.quantile_profiles[q] = profile

    def _train_classifier(self):
        """Fit KNN imputer + CalibratedClassifierCV(ExtraTrees); sets CV metrics and importances."""
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.impute import KNNImputer

        X_raw = self.df[self.feature_cols].copy()
        for col in X_raw.columns:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

        y = self.df["wash_q"].copy()
        mask = y.notna()
        X_raw_m = X_raw[mask]
        y_clean = y[mask].values.astype(int)

        imputer = KNNImputer(n_neighbors=5)
        X_clean = imputer.fit_transform(X_raw_m)
        self._knn_imputer = imputer
        self._feature_medians = {f: float(X_raw[f].median()) for f in self.feature_cols}
        self._X_train_imputed = X_clean
        self._y_train = y_clean
        self._counts_train = self.df.loc[mask, "current_count"].astype(float).values

        # ExtraTrees with Optuna-tuned hyperparameters
        # Optuna best (50 trials, 5-fold CV on 482 sites):
        # n_estimators=600, max_depth=8, min_samples_leaf=5, max_features='sqrt' → 63.1% CV
        base_clf = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.clf = CalibratedClassifierCV(base_clf, cv=skf_cal, method="isotonic")
        self.clf.fit(X_clean, y_clean)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_exact = cross_val_score(base_clf, X_clean, y_clean, cv=skf, scoring="accuracy")
        self.cv_accuracy = float(cv_exact.mean())

        all_preds, all_true = [], []
        for tr, te in skf.split(X_clean, y_clean):
            base_clf.fit(X_clean[tr], y_clean[tr])
            all_preds.extend(base_clf.predict(X_clean[te]))
            all_true.extend(y_clean[te])
        adj = np.mean(np.abs(np.array(all_preds) - np.array(all_true)) <= 1)
        self.cv_adjacent_accuracy = float(adj)

        base_clf.fit(X_clean, y_clean)
        self.feature_importances: Dict[str, float] = dict(
            zip(self.feature_cols, base_clf.feature_importances_)
        )

        print(
            f"\nClassifier v4 (ExtraTrees) — 5-fold CV  exact: {self.cv_accuracy:.1%}  "
            f"within-1: {self.cv_adjacent_accuracy:.1%}  (n={len(y_clean)})"
        )
        top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 features by importance:")
        for f, imp in top:
            print(f"    {FEATURE_LABELS.get(f, f)}: {imp:.1%}")

    def _train_volume_regressors(self):
        """Per-tier ExtraTrees on volume for conditional E[Y|X]; sparse tiers use tier mean."""
        from sklearn.ensemble import ExtraTreesRegressor

        self._volume_regressors: Dict[int, object] = {}
        self._volume_fallback_means: Dict[int, float] = dict(self.tier_means)

        X_clean = getattr(self, "_X_train_imputed", None)
        y_clean = getattr(self, "_y_train", None)
        counts = getattr(self, "_counts_train", None)
        if X_clean is None or y_clean is None or counts is None:
            return

        for q in range(1, self.n_quantiles + 1):
            mask_q = y_clean == q
            n_q = int(mask_q.sum())
            if n_q < 15:
                self._volume_regressors[q] = None
                continue
            reg = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
            reg.fit(X_clean[mask_q], counts[mask_q])
            self._volume_regressors[q] = reg

    # ── Feature-to-wash-Q mapping ─────────────────────────────────────────────

    def _feature_to_wash_q(self, feat: str, val: float, direction: str) -> Dict:
        group_medians: Dict[int, float] = {}
        for q in range(1, self.n_quantiles + 1):
            med = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
            if med is not None:
                group_medians[q] = med

        if not group_medians:
            return {"wash_q": None, "group_medians": {}, "exceeds_q4": False, "q4_median": None}

        signal = FEATURE_SIGNAL.get(feat, 0.0)
        if signal < SIGNAL_THRESHOLD:
            return {"wash_q": None, "group_medians": group_medians, "exceeds_q4": False,
                    "q4_median": group_medians.get(4), "low_signal": True}

        sign = -1 if direction == "lower" else 1
        adj_val = sign * val
        adj_medians = {q: sign * m for q, m in group_medians.items()}

        q4_med = group_medians.get(4)
        q4_adj = adj_medians.get(4)
        exceeds_q4 = q4_adj is not None and adj_val > q4_adj

        matched_q = 4 if exceeds_q4 else min(adj_medians.items(), key=lambda x: abs(x[1] - adj_val))[0]

        return {"wash_q": matched_q, "group_medians": group_medians,
                "exceeds_q4": exceeds_q4, "q4_median": q4_med, "low_signal": False}

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        location_features: Dict[str, float],
        llm_narrative: bool = True,
    ) -> Dict:
        from sklearn.impute import KNNImputer

        # Auto-compute derived features if not explicitly provided
        location_features = dict(location_features)  # copy to avoid mutating caller's dict
        if "effective_capacity" not in location_features:
            tc = location_features.get("tunnel_count")
            cw = location_features.get("carwash_type_encoded")
            if tc is not None:
                is_express = (float(cw) == 1.0) if cw is not None else True  # default Express
                location_features["effective_capacity"] = float(tc) * float(is_express)

        feature_vector_raw: List[Optional[float]] = []
        feature_analysis: Dict[str, Dict] = {}

        # ── Main ML feature loop ───────────────────────────────────────────────
        for feat in self.feature_cols:
            raw_val = location_features.get(feat)
            dist = self.feature_dists.get(feat)
            if dist is None:
                feature_vector_raw.append(np.nan)
                continue
            direction = dist["direction"]

            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                val = float(raw_val)
                imputed = False
            else:
                val = dist["median"]
                imputed = True

            feature_vector_raw.append(val if not imputed else np.nan)

            raw_pct = float(stats.percentileofscore(dist["data"], val, kind="rank"))
            if direction == "lower":
                adj_pct = 100.0 - raw_pct
            else:
                adj_pct = raw_pct

            raw_q  = _assign_raw_quantile(val, dist["boundaries"])
            adj_q  = _adj_quantile(raw_q, direction)

            better_boundary = _next_better_boundary(dist["boundaries"], adj_q, direction)
            delta    = abs(val - better_boundary) if better_boundary is not None else None
            shift_dir = ("increase" if direction == "higher" else "decrease") if better_boundary else None

            wq_info = self._feature_to_wash_q(feat, val, direction)

            feature_analysis[feat] = {
                "value":                        val if not imputed else None,
                "imputed":                      imputed,
                "label":                        FEATURE_LABELS.get(feat, feat),
                "direction":                    direction,
                "signal":                       FEATURE_SIGNAL.get(feat, 0.0),
                "raw_percentile":               round(raw_pct, 1),
                "adjusted_percentile":          round(adj_pct, 1),
                "feature_quantile_raw":         raw_q,
                "feature_quantile_adj":         adj_q,
                "wash_correlated_q":            wq_info.get("wash_q"),
                "wash_correlated_exceeds_q4":   wq_info.get("exceeds_q4", False),
                "wash_q_group_medians":         wq_info.get("group_medians", {}),
                "wash_q_q4_median":             wq_info.get("q4_median"),
                "wash_q_low_signal":            wq_info.get("low_signal", False),
                "distribution_shape":           dist["shape"],
                "dist_min":                     dist["min"],
                "dist_max":                     dist["max"],
                "dist_median":                  dist["median"],
                "quantile_boundaries":          dist["boundaries"].tolist(),
                "shift_to_next_q_delta":        round(delta, 2) if delta is not None else None,
                "shift_to_next_q_direction":    shift_dir,
                "importance":                   round(self.feature_importances.get(feat, 0), 4),
                "ml_feature":                   True,
            }

        # Impute using trained KNN
        X_df = pd.DataFrame([feature_vector_raw], columns=self.feature_cols)
        X_imputed = self._knn_imputer.transform(X_df)
        for i, feat in enumerate(self.feature_cols):
            if feat in feature_analysis and feature_analysis[feat].get("imputed"):
                feature_analysis[feat]["value"] = float(X_imputed[0, i])

        # ── Display-only feature loop (carwash_type_encoded etc.) ─────────────
        # These appear in the report but are NOT fed to the ML model.
        for feat in DISPLAY_ONLY_FEATURES:
            raw_val = location_features.get(feat)
            dist = self.feature_dists.get(feat)
            if dist is None:
                continue
            direction = dist["direction"]

            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                val = float(raw_val)
                imputed = False
            else:
                val = dist["median"]
                imputed = True

            raw_pct = float(stats.percentileofscore(dist["data"], val, kind="rank"))
            adj_pct = 100.0 - raw_pct if direction == "lower" else raw_pct
            raw_q   = _assign_raw_quantile(val, dist["boundaries"])
            adj_q   = _adj_quantile(raw_q, direction)
            wq_info = self._feature_to_wash_q(feat, val, direction)

            feature_analysis[feat] = {
                "value":                        val if not imputed else None,
                "imputed":                      imputed,
                "label":                        FEATURE_LABELS.get(feat, feat),
                "direction":                    direction,
                "signal":                       FEATURE_SIGNAL.get(feat, 0.0),
                "raw_percentile":               round(raw_pct, 1),
                "adjusted_percentile":          round(adj_pct, 1),
                "feature_quantile_raw":         raw_q,
                "feature_quantile_adj":         adj_q,
                "wash_correlated_q":            wq_info.get("wash_q"),
                "wash_correlated_exceeds_q4":   wq_info.get("exceeds_q4", False),
                "wash_q_group_medians":         wq_info.get("group_medians", {}),
                "wash_q_q4_median":             wq_info.get("q4_median"),
                "wash_q_low_signal":            wq_info.get("low_signal", False),
                "distribution_shape":           dist["shape"],
                "dist_min":                     dist["min"],
                "dist_max":                     dist["max"],
                "dist_median":                  dist["median"],
                "quantile_boundaries":          dist["boundaries"].tolist(),
                "shift_to_next_q_delta":        None,
                "shift_to_next_q_direction":    None,
                "importance":                   0.0,  # not in ML model
                "ml_feature":                   False,  # display only
                "display_only_note": (
                    "Captured via effective_capacity = tunnel_count × is_express. "
                    "Adding directly to ML model causes multicollinearity (−0.9% accuracy). "
                    "Shown here for interpretation only."
                ),
            }

        # Predict
        predicted_q   = int(self.clf.predict(X_imputed)[0])
        proba_arr     = self.clf.predict_proba(X_imputed)[0]
        classes       = list(self.clf.classes_)
        proba         = {int(c): round(float(p), 3) for c, p in zip(classes, proba_arr)}

        wash_range = self.wash_q_ranges[predicted_q]

        shift_opportunities = self._quantile_shift_analysis(
            X_imputed[0].tolist(), feature_analysis, predicted_q
        )
        profile_comparison = self._profile_comparison(feature_analysis)
        strengths, weaknesses = self._strengths_weaknesses(feature_analysis)

        # Weighted volume estimates:
        # - mean_by_tier: quick expectation baseline (uses E[Y|Q_i] as tier means)
        # - conditional_regression: best estimate from per-tier regressors E[Y|Q_i, X]
        weighted_vol_mean = sum(
            proba.get(q, 0.0) * self.tier_means.get(q, 0.0)
            for q in range(1, self.n_quantiles + 1)
        )
        conditional_expectations: Dict[int, float] = {}
        for q in range(1, self.n_quantiles + 1):
            reg = (getattr(self, "_volume_regressors", {}) or {}).get(q)
            if reg is None:
                conditional_expectations[q] = float(
                    (getattr(self, "_volume_fallback_means", {}) or {}).get(q, self.tier_means.get(q, 0.0))
                )
            else:
                conditional_expectations[q] = float(reg.predict(X_imputed)[0])
        weighted_vol_conditional = sum(
            proba.get(q, 0.0) * conditional_expectations.get(q, 0.0)
            for q in range(1, self.n_quantiles + 1)
        )
        weighted_vol = weighted_vol_conditional

        # Probabilistic uncertainty from class-probability spread around the
        # conditional expectation (same estimator used for point prediction).
        variance = sum(
            proba.get(q, 0.0) * ((conditional_expectations.get(q, 0.0) - weighted_vol) ** 2)
            for q in range(1, self.n_quantiles + 1)
        )
        sigma = float(np.sqrt(max(variance, 0.0)))
        confidence_score = float(max(proba.values())) if proba else None
        entropy = (
            float(-sum(p * np.log(max(p, 1e-12)) for p in proba.values()))
            if proba else None
        )

        result: Dict = {
            "predicted_wash_quantile":          predicted_q,
            "predicted_wash_quantile_label":    QUANTILE_LABELS[predicted_q] if predicted_q in QUANTILE_LABELS else f"Q{predicted_q}",
            "predicted_wash_tier":              QUANTILE_TIER_NAMES[predicted_q] if predicted_q in QUANTILE_TIER_NAMES else f"Tier {predicted_q}",
            "predicted_wash_range": {
                "min":   round(wash_range[0]),
                "max":   round(wash_range[1]),
                "label": f"{wash_range[0]:,.0f} - {wash_range[1]:,.0f} cars/yr",
            },
            "weighted_volume_prediction":       round(weighted_vol),
            "weighted_volume_prediction_mean_by_tier": round(weighted_vol_mean),
            "weighted_volume_prediction_method": "conditional_regression",
            "conditional_expectations_by_quantile": {
                f"Q{q}": round(v, 1) for q, v in conditional_expectations.items()
            },
            "volume_uncertainty": {
                "sigma": round(sigma),
                "low": round(weighted_vol - sigma),
                "high": round(weighted_vol + sigma),
                "method": "probability_variance",
            },
            "prediction_confidence": {
                "max_probability": round(confidence_score, 4) if confidence_score is not None else None,
                "entropy": round(entropy, 4) if entropy is not None else None,
            },
            "quantile_probabilities":           proba,
            "tier_metadata": {
                "strategy":   self.tier_strategy,
                "n_tiers":    self.n_quantiles,
                "boundaries": [float(b) for b in self.wash_boundaries],
                "medians":    self.tier_medians,
                "means":      self.tier_means,
                "labels":     [QUANTILE_TIER_NAMES.get(q, f"Tier {q}") for q in range(1, self.n_quantiles + 1)],
            },
            "wash_count_distribution": {
                q: {
                    "range": f"{self.wash_q_ranges[q][0]:,.0f} - {self.wash_q_ranges[q][1]:,.0f}",
                    "min":   round(self.wash_q_ranges[q][0]),
                    "max":   round(self.wash_q_ranges[q][1]),
                }
                for q in range(1, self.n_quantiles + 1)
            },
            "feature_analysis":     feature_analysis,
            "shift_opportunities":  shift_opportunities,
            "profile_comparison":   profile_comparison,
            "strengths":            strengths,
            "weaknesses":           weaknesses,
            "model_cv_accuracy":    round(self.cv_accuracy, 3),
            "model_adj_accuracy":   round(self.cv_adjacent_accuracy, 3),
            "features_available":   sum(
                1 for fa in feature_analysis.values()
                if not fa.get("imputed", True) and fa.get("ml_feature", True)
            ),
        }

        if llm_narrative:
            result["narrative"]                  = self._generate_narrative(result)
            result["strengths_weaknesses_llm"]   = self._generate_strengths_weaknesses_llm(result)

        return result

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _quantile_shift_analysis(
        self,
        base_vector: List[float],
        feature_analysis: Dict[str, Dict],
        current_q: int,
    ) -> List[Dict]:
        if current_q >= self.n_quantiles:
            return []

        opportunities: List[Dict] = []
        prob_lifts: List[Dict] = []

        classes = list(self.clf.classes_)
        target_cls_idx = classes.index(current_q + 1) if (current_q + 1) in classes else None
        base_proba_up  = self.clf.predict_proba(np.array(base_vector).reshape(1, -1))[0][target_cls_idx] if target_cls_idx is not None else 0.0

        for i, feat in enumerate(self.feature_cols):
            fa = feature_analysis.get(feat, {})
            if fa.get("imputed") or fa.get("direction") == "neutral":
                continue

            dist      = self.feature_dists.get(feat)
            val       = fa.get("value")
            adj_q     = fa.get("feature_quantile_adj", 2)
            direction = fa.get("direction")

            if dist is None or val is None or adj_q >= self.n_quantiles:
                continue

            best_sim_q, best_target_val, best_gain = current_q, val, 0

            for target_adj_q in range(adj_q + 1, self.n_quantiles + 1):
                raw_target_q = (5 - target_adj_q) if direction == "lower" else target_adj_q
                if raw_target_q < 1 or raw_target_q > self.n_quantiles:
                    continue
                boundary_idx = raw_target_q - 1
                if boundary_idx < 0 or boundary_idx >= len(dist["boundaries"]):
                    continue
                if direction == "higher":
                    target_val = dist["boundaries"][target_adj_q - 1] + 0.01
                else:
                    target_val = dist["boundaries"][self.n_quantiles - target_adj_q + 1] - 0.01
                    if target_val < 0:
                        target_val = max(0, dist["boundaries"][0])

                sim_vector = base_vector.copy()
                sim_vector[i] = target_val
                sim_q = int(self.clf.predict(np.array(sim_vector).reshape(1, -1))[0])

                if sim_q > best_sim_q:
                    best_sim_q, best_target_val, best_gain = sim_q, target_val, sim_q - current_q
                    break

            change = abs(val - best_target_val)
            if best_gain > 0 and change > 0.001:
                opportunities.append({
                    "feature":          feat,
                    "label":            FEATURE_LABELS.get(feat, feat),
                    "current_value":    val,
                    "target_value":     round(float(best_target_val), 2),
                    "change_needed":    round(change, 2),
                    "change_direction": "increase" if direction == "higher" else "decrease",
                    "current_wash_q":   current_q,
                    "simulated_wash_q": best_sim_q,
                    "q_gain":           best_gain,
                    "importance":       fa.get("importance", 0),
                })
            elif target_cls_idx is not None:
                sim_vector = base_vector.copy()
                if direction == "higher":
                    sim_vector[i] = float(dist["boundaries"][min(adj_q, self.n_quantiles - 1)])
                else:
                    sim_vector[i] = float(dist["boundaries"][max(self.n_quantiles - adj_q, 1)])
                sim_p = self.clf.predict_proba(np.array(sim_vector).reshape(1, -1))[0][target_cls_idx]
                lift  = sim_p - base_proba_up
                if lift > 0.01:
                    prob_lifts.append({
                        "feature":          feat,
                        "label":            FEATURE_LABELS.get(feat, feat),
                        "current_value":    val,
                        "target_value":     round(float(sim_vector[i]), 2),
                        "change_needed":    round(abs(val - sim_vector[i]), 2),
                        "change_direction": "increase" if direction == "higher" else "decrease",
                        "current_wash_q":   current_q,
                        "simulated_wash_q": current_q,
                        "q_gain":           0,
                        "prob_lift":        round(lift * 100, 1),
                        "importance":       fa.get("importance", 0),
                    })

        opportunities.sort(key=lambda x: (-x["q_gain"], -x["importance"]))
        if not opportunities and prob_lifts:
            prob_lifts.sort(key=lambda x: (-x["prob_lift"], -x["importance"]))
            return prob_lifts[:5]
        return opportunities[:5]

    def _profile_comparison(self, feature_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        comparison: Dict[str, Dict] = {}
        for feat, fa in feature_analysis.items():
            if fa.get("value") is None:
                continue
            val       = fa["value"]
            direction = fa["direction"]
            feat_comp: Dict[str, Dict] = {}
            for q in range(1, self.n_quantiles + 1):
                pm = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
                if pm is None:
                    continue
                if direction == "higher":
                    align = "above" if val > pm else "below" if val < pm else "at"
                elif direction == "lower":
                    align = "better" if val < pm else "worse" if val > pm else "at"
                else:
                    align = "near"
                feat_comp[f"Q{q}"] = {"profile_median": round(float(pm), 2), "alignment": align}
            comparison[feat] = feat_comp
        return comparison

    def _strengths_weaknesses(
        self, feature_analysis: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        strengths: List[Dict] = []
        weaknesses: List[Dict] = []
        for feat, fa in feature_analysis.items():
            if fa.get("imputed") or fa.get("wash_q_low_signal"):
                continue
            wq      = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            label   = fa.get("label", feat)
            val     = fa.get("value")
            q4_med  = fa.get("wash_q_q4_median")
            imp     = fa.get("importance", 0)
            signal  = fa.get("signal", 0.0)
            entry   = {"label": label, "value": val, "q4_median": q4_med,
                       "importance": imp, "signal": signal}
            if wq == 4 or exceeds:
                entry["note"] = "exceeds Q4 typical" if exceeds else "matches Q4"
                strengths.append(entry)
            elif wq in (1, 2):
                entry["note"] = "Q1-level (low performing)" if wq == 1 else "Q2-level (below median)"
                weaknesses.append(entry)
        strengths.sort(key=lambda x: -x["importance"])
        weaknesses.sort(key=lambda x: -x["importance"])
        return strengths, weaknesses


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = QuantilePredictorV4()

    example: Dict[str, float] = {
        "weather_rainy_days":                   130,
        "weather_days_pleasant_temp":           180,
        "weather_total_snowfall_cm":            5.0,
        "weather_days_below_freezing":          30,
        "weather_total_precipitation_mm":       950,
        "weather_total_sunshine_hours":         3400,
        "weather_avg_daily_max_windspeed_ms":   15.0,
        "nearest_gas_station_distance_miles":   0.15,
        "nearest_gas_station_rating":           4.2,
        "nearest_gas_station_rating_count":     120,
        "competitors_count_4miles":             7,
        "competitor_1_google_rating":           4.1,
        "competitor_1_distance_miles":          0.4,
        "competitor_1_rating_count":            300,
        "costco_enc":                           2.5,
        "distance_nearest_walmart(5 mile)":     1.0,
        "distance_nearest_target (5 mile)":     1.5,
        "other_grocery_count_1mile":            2,
        "count_food_joints_0_5miles (0.5 mile)":8,
        "age_on_30_sep_25":                     3.0,
        "region_enc":                           1.0,
        "state_enc":                            10.0,
        "tunnel_count":                         2.0,
        "carwash_type_encoded":                 1.0,
    }

    result = predictor.analyze(example, llm_narrative=False)
    predictor.print_report(result)
    predictor.plot_feature_quantiles(result)
    predictor.save_report(result)
