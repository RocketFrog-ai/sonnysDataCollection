# site_scorer.py

import numpy as np

class SiteScorer:
    def __init__(self, feature_analyzer):
        self.analyzer = feature_analyzer

    def score_site(self, site_features):
        results = []
        scores = []

        for feature, value in site_features.items():
            analysis = self.analyzer.analyze(feature, value)
            results.append(analysis)
            scores.append(analysis["percentile_score"])

        return {
            "overall_score": round(np.mean(scores), 2),
            "feature_analysis": results,
            "strengths": sorted(results, key=lambda x: -x["percentile_score"])[:5],
            "weaknesses": sorted(results, key=lambda x: x["percentile_score"])[:5],
        }
