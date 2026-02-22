# percentile_engine.py

import numpy as np
from scipy.stats import percentileofscore

class PercentileEngine:
    def __init__(self, df, feature_direction):
        self.df = df
        self.feature_direction = feature_direction
        self.distributions = self._build_distributions()

    def _build_distributions(self):
        dist = {}
        for feature in self.feature_direction:
            dist[feature] = self.df[feature].dropna().values
        return dist

    def score(self, feature, value):
        values = self.distributions[feature]
        p = percentileofscore(values, value, kind="rank")

        if self.feature_direction[feature] == "lower":
            return round(100 - p, 2)

        return round(p, 2)
