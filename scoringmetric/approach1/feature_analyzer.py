# feature_analyzer.py

from range_engine import derive_ranges

class FeatureAnalyzer:
    def __init__(self, percentile_engine, df):
        self.engine = percentile_engine
        self.df = df

    def analyze(self, feature, value):
        values = self.df[feature].dropna().values

        ranges, rule = derive_ranges(values)
        return {
            "feature": feature,
            "value": value,
            "percentile_score": self.engine.score(feature, value),
            "ranges": ranges,
            "range_rule": rule
        }
