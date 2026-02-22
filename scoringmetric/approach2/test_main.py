"""
Tests for CTO-Exact Profiler (approach2)
========================================

Tests the percentile-based feature scoring methodology.
Uses synthetic data to avoid external file dependencies.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add approach2 to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import CTOExactProfiler


def make_sample_df():
    """Create a minimal sample DataFrame with features used by CTOExactProfiler."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        # Higher is better
        'total_sunshine_hours': np.random.uniform(2000, 3500, n),
        'count_of_costco_5miles': np.random.randint(0, 5, n),
        # Lower is better
        'distance_from_nearest_costco': np.random.uniform(0.5, 15, n),  # km
        'total_precipitation_mm': np.random.uniform(200, 1200, n),
    })


class TestCTOExactProfiler(unittest.TestCase):
    """Test cases for CTOExactProfiler."""

    def setUp(self):
        """Set up profiler with sample data before each test."""
        self.df = make_sample_df()
        self.profiler = CTOExactProfiler(self.df)

    def test_initialization(self):
        """Profiler initializes with correct feature count and excludes expected columns."""
        self.assertIsNotNone(self.profiler.df)
        self.assertIn('total_sunshine_hours', self.profiler.feature_cols)
        self.assertIn('distance_from_nearest_costco', self.profiler.feature_cols)
        self.assertIn('count_of_costco_5miles', self.profiler.feature_cols)
        self.assertGreater(len(self.profiler.feature_cols), 0)
        self.assertEqual(len(self.profiler.distributions), len(self.profiler.feature_cols))

    def test_score_feature_higher_is_better(self):
        """For higher-is-better features, higher percentile = higher score."""
        # Score at max value should be near 100
        max_val = self.df['total_sunshine_hours'].max()
        result = self.profiler.score_feature_cto_method('total_sunshine_hours', max_val)
        self.assertNotIn('error', result)
        self.assertEqual(result['direction'], 'higher_is_better')
        self.assertGreaterEqual(result['final_score'], 90)

        # Score at min value should be low
        min_val = self.df['total_sunshine_hours'].min()
        result = self.profiler.score_feature_cto_method('total_sunshine_hours', min_val)
        self.assertLessEqual(result['final_score'], 20)

    def test_score_feature_lower_is_better(self):
        """For lower-is-better features, lower percentile = higher score (inverted)."""
        # Minimum distance = best (lowest raw percentile) -> high final score
        min_val = self.df['distance_from_nearest_costco'].min()
        result = self.profiler.score_feature_cto_method('distance_from_nearest_costco', min_val)
        self.assertNotIn('error', result)
        self.assertEqual(result['direction'], 'lower_is_better')
        self.assertGreaterEqual(result['final_score'], 80)

        # Maximum distance = worst (high raw percentile) -> low final score
        max_val = self.df['distance_from_nearest_costco'].max()
        result = self.profiler.score_feature_cto_method('distance_from_nearest_costco', max_val)
        self.assertLessEqual(result['final_score'], 20)

    def test_score_feature_cto_costco_example(self):
        """
        Replicate CTO's Costco example logic:
        - 1 km at 94.3 percentile for distance (lower is better)
        - Final score = 100 - 94.3 = 5.7
        """
        # Create data where 1.0 km sits at high percentile (far)
        data = np.concatenate([
            np.linspace(0.1, 0.9, 90),  # Most sites 0.1-0.9 km
            np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])  # Few far sites
        ])
        df_small = pd.DataFrame({
            'distance_from_nearest_costco': data,
            'total_sunshine_hours': np.random.uniform(2000, 3500, len(data))
        })
        profiler = CTOExactProfiler(df_small)

        result = profiler.score_feature_cto_method('distance_from_nearest_costco', 1.0)
        self.assertEqual(result['direction'], 'lower_is_better')
        # Raw percentile should be high (1km is far in this distribution)
        self.assertGreater(result['raw_percentile'], 70)
        # Final score = 100 - raw_percentile, so low
        self.assertAlmostEqual(result['final_score'], 100 - result['raw_percentile'], places=1)

    def test_score_feature_unknown_feature_returns_error(self):
        """Unknown feature returns error dict."""
        result = self.profiler.score_feature_cto_method('nonexistent_feature', 42.0)
        self.assertIn('error', result)

    def test_score_location(self):
        """score_location returns overall score, feature_scores, strengths, and weaknesses."""
        location = {
            'total_sunshine_hours': 2800,
            'distance_from_nearest_costco': 3.0,
            'count_of_costco_5miles': 2,
            'total_precipitation_mm': 600,
        }
        result = self.profiler.score_location(location)

        self.assertIn('overall_score', result)
        self.assertIn('feature_scores', result)
        self.assertIn('top_strengths', result)
        self.assertIn('top_weaknesses', result)
        self.assertIn('total_features', result)

        self.assertIsInstance(result['overall_score'], (int, float))
        self.assertGreaterEqual(result['overall_score'], 0)
        self.assertLessEqual(result['overall_score'], 100)

        self.assertEqual(len(result['feature_scores']), len(location))
        self.assertLessEqual(len(result['top_strengths']), 10)
        self.assertLessEqual(len(result['top_weaknesses']), 10)

    def test_score_location_ignores_unknown_features(self):
        """score_location ignores features not in profiler."""
        location = {
            'total_sunshine_hours': 2800,
            'unknown_feature': 999,
        }
        result = self.profiler.score_location(location)
        self.assertIn('total_sunshine_hours', result['feature_scores'])
        self.assertNotIn('unknown_feature', result['feature_scores'])

    def test_distributions_have_expected_keys(self):
        """Each distribution has min, max, mean, median, std, count, data."""
        for feature, dist in self.profiler.distributions.items():
            for key in ('min', 'max', 'mean', 'median', 'std', 'count', 'data'):
                self.assertIn(key, dist, msg=f"Missing key {key} for {feature}")

    def test_interpretation_strings(self):
        """Score results include non-empty interpretation strings."""
        result = self.profiler.score_feature_cto_method('total_sunshine_hours', 3000)
        self.assertIn('interpretation', result)
        self.assertIsInstance(result['interpretation'], str)
        self.assertGreater(len(result['interpretation']), 0)


if __name__ == '__main__':
    unittest.main()
