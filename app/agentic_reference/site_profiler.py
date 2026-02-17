"""
Quantile-Based Site Profiling System
=====================================

A robust alternative to traditional regression for car wash site evaluation.

Author: Your Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class SiteProfiler:
    """
    Quantile-based site profiling system that classifies locations into
    performance tiers based on feature similarity to historical data.
    
    Instead of predicting exact volumes with unstable coefficients,
    this approach:
    1. Divides historical sites into performance categories
    2. Computes typical feature ranges (IQR) for each category
    3. Scores new locations based on fit to each category's ranges
    4. Predicts category and provides realistic volume ranges
    """
    
    def __init__(
        self, 
        historical_data: pd.DataFrame, 
        target_col: str = 'cars_washed(Actual)',
        n_categories: int = 3,
        category_labels: Optional[List[str]] = None
    ):
        """
        Initialize profiler with historical data.
        
        Parameters:
        -----------
        historical_data : pd.DataFrame
            Historical site data with features and target
        target_col : str
            Name of target variable column
        n_categories : int
            Number of performance tiers (default: 3)
        category_labels : list of str, optional
            Labels for categories (default: Low/Average/High)
        """
        self.df = historical_data.copy()
        self.target = target_col
        self.n_categories = n_categories
        
        if category_labels is None:
            if n_categories == 3:
                category_labels = ['Low Performing', 'Average Performing', 'High Performing']
            elif n_categories == 5:
                category_labels = ['Very Low', 'Low', 'Average', 'High', 'Very High']
            else:
                category_labels = [f'Tier {i+1}' for i in range(n_categories)]
        
        self.category_labels = category_labels
        
        # Create performance categories
        self.df['performance_category'] = pd.qcut(
            self.df[target_col], 
            q=n_categories, 
            labels=category_labels,
            duplicates='drop'
        )
        
        # Identify feature columns (exclude non-numeric and target)
        self.feature_cols = self._identify_features()
        
        # Compute feature ranges for each category
        self.feature_ranges = self._compute_feature_ranges()
        
        # Compute category statistics
        self.category_stats = self._compute_category_stats()
        
        print(f"✓ Profiler initialized with {len(self.df)} sites")
        print(f"✓ {len(self.feature_cols)} features")
        print(f"✓ {n_categories} performance categories")
        
    def _identify_features(self) -> List[str]:
        """Identify feature columns (numeric, excluding target and ID columns)"""
        exclude = [self.target, 'performance_category', 'full_site_address']
        features = [col for col in self.df.columns 
                   if col not in exclude and self.df[col].dtype in ['int64', 'float64']]
        return features
    
    def _compute_feature_ranges(self) -> Dict:
        """Compute IQR ranges for each feature by category"""
        ranges = {}
        
        for feature in self.feature_cols:
            ranges[feature] = {}
            
            for category in self.category_labels:
                data = self.df[self.df['performance_category'] == category][feature].dropna()
                
                if len(data) > 0:
                    ranges[feature][category] = {
                        'q25': float(data.quantile(0.25)),
                        'q50': float(data.quantile(0.50)),
                        'q75': float(data.quantile(0.75)),
                        'mean': float(data.mean()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        'max': float(data.max()),
                        'count': int(len(data))
                    }
                else:
                    ranges[feature][category] = None
        
        return ranges
    
    def _compute_category_stats(self) -> Dict:
        """Compute statistics for target variable by category"""
        stats = {}
        
        for category in self.category_labels:
            data = self.df[self.df['performance_category'] == category][self.target]
            
            stats[category] = {
                'count': int(len(data)),
                'min': float(data.min()),
                'q25': float(data.quantile(0.25)),
                'median': float(data.quantile(0.50)),
                'q75': float(data.quantile(0.75)),
                'max': float(data.max()),
                'mean': float(data.mean()),
                'std': float(data.std())
            }
        
        return stats
    
    def score_feature(
        self, 
        feature: str, 
        value: float, 
        category: str
    ) -> float:
        """
        Score how well a feature value fits a category's typical range.
        
        Returns:
        --------
        float : Score from 0.0 (poor fit) to 1.0 (perfect fit)
        """
        if feature not in self.feature_ranges:
            return 0.0
        
        ranges = self.feature_ranges[feature].get(category)
        if ranges is None:
            return 0.0
        
        q25, q75 = ranges['q25'], ranges['q75']
        
        # Perfect fit: value within IQR
        if q25 <= value <= q75:
            return 1.0
        
        # Partial fit: value near IQR
        dist_from_range = min(abs(value - q25), abs(value - q75))
        range_width = q75 - q25
        
        if range_width > 0:
            # Score decreases linearly with distance
            score = max(0.0, 1.0 - (dist_from_range / range_width))
            return score
        
        return 0.0
    
    def score_location(
        self, 
        location_features: Dict[str, float],
        feature_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Score a location against all performance categories.
        
        Parameters:
        -----------
        location_features : dict
            Feature name -> value mapping
        feature_weights : dict, optional
            Feature name -> weight mapping (default: equal weights)
        
        Returns:
        --------
        dict : Category -> total score
        """
        if feature_weights is None:
            feature_weights = {f: 1.0 for f in self.feature_cols}
        
        scores = {cat: 0.0 for cat in self.category_labels}
        total_weight = 0.0
        
        for feature, value in location_features.items():
            if feature not in self.feature_cols:
                continue
            
            weight = feature_weights.get(feature, 1.0)
            
            for category in self.category_labels:
                feature_score = self.score_feature(feature, value, category)
                scores[category] += feature_score * weight
            
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            scores = {cat: score / total_weight for cat, score in scores.items()}
        
        return scores
    
    def predict(
        self, 
        location_features: Dict[str, float],
        feature_weights: Optional[Dict[str, float]] = None,
        return_details: bool = False
    ) -> Dict:
        """
        Predict performance category and expected volume for a new location.
        
        Parameters:
        -----------
        location_features : dict
            Feature name -> value mapping
        feature_weights : dict, optional
            Feature importance weights
        return_details : bool
            If True, return detailed scoring breakdown
        
        Returns:
        --------
        dict : Prediction results with category, confidence, expected volumes
        """
        # Score against each category
        scores = self.score_location(location_features, feature_weights)
        
        # Determine predicted category
        predicted_category = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = (scores[predicted_category] / total_score * 100) if total_score > 0 else 0
        
        # Get expected volume range
        category_volumes = self.category_stats[predicted_category]
        
        result = {
            'predicted_category': predicted_category,
            'confidence': confidence,
            'category_scores': scores,
            'expected_volume': {
                'conservative': category_volumes['q25'],
                'likely': category_volumes['median'],
                'optimistic': category_volumes['q75'],
                'range_min': category_volumes['min'],
                'range_max': category_volumes['max']
            },
            'category_stats': {
                'count': category_volumes['count'],
                'mean': category_volumes['mean'],
                'std': category_volumes['std']
            }
        }
        
        if return_details:
            # Add feature-by-feature scoring
            feature_scores = {}
            for feature, value in location_features.items():
                if feature in self.feature_cols:
                    feature_scores[feature] = {
                        'value': value,
                        'scores': {
                            cat: self.score_feature(feature, value, cat) 
                            for cat in self.category_labels
                        },
                        'best_fit': max(
                            self.category_labels,
                            key=lambda cat: self.score_feature(feature, value, cat)
                        )
                    }
            
            result['feature_details'] = feature_scores
        
        return result
    
    def compare_locations(
        self, 
        locations: List[Dict[str, float]],
        location_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple candidate locations.
        
        Parameters:
        -----------
        locations : list of dict
            List of feature dictionaries
        location_names : list of str, optional
            Names for each location
        
        Returns:
        --------
        pd.DataFrame : Comparison table sorted by expected median volume
        """
        if location_names is None:
            location_names = [f'Location {i+1}' for i in range(len(locations))]
        
        results = []
        
        for name, features in zip(location_names, locations):
            pred = self.predict(features)
            
            results.append({
                'Location': name,
                'Predicted Category': pred['predicted_category'],
                'Confidence (%)': pred['confidence'],
                'Expected Volume (Median)': pred['expected_volume']['likely'],
                'Conservative': pred['expected_volume']['conservative'],
                'Optimistic': pred['expected_volume']['optimistic']
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Expected Volume (Median)', ascending=False)
        
        return df_results
    
    def get_feature_ranges(self, feature: str) -> pd.DataFrame:
        """
        Get typical ranges for a specific feature across all categories.
        
        Parameters:
        -----------
        feature : str
            Feature name
        
        Returns:
        --------
        pd.DataFrame : Range table by category
        """
        if feature not in self.feature_ranges:
            raise ValueError(f"Feature '{feature}' not found")
        
        data = []
        for category in self.category_labels:
            ranges = self.feature_ranges[feature].get(category)
            if ranges:
                data.append({
                    'Category': category,
                    'Min': ranges['min'],
                    '25th %ile': ranges['q25'],
                    'Median': ranges['q50'],
                    '75th %ile': ranges['q75'],
                    'Max': ranges['max'],
                    'Mean': ranges['mean'],
                    'Sites': ranges['count']
                })
        
        return pd.DataFrame(data)
    
    def export_ranges(self, filepath: str):
        """Export all feature ranges to CSV"""
        data = []
        
        for feature in self.feature_cols:
            for category in self.category_labels:
                ranges = self.feature_ranges[feature].get(category)
                if ranges:
                    data.append({
                        'feature': feature,
                        'category': category,
                        **ranges
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"✓ Feature ranges exported to {filepath}")
    
    def save_model(self, filepath: str):
        """Save model to JSON file"""
        model_data = {
            'target': self.target,
            'n_categories': self.n_categories,
            'category_labels': self.category_labels,
            'feature_cols': self.feature_cols,
            'feature_ranges': self.feature_ranges,
            'category_stats': self.category_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✓ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model from JSON file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Create empty profiler
        profiler = cls.__new__(cls)
        profiler.target = model_data['target']
        profiler.n_categories = model_data['n_categories']
        profiler.category_labels = model_data['category_labels']
        profiler.feature_cols = model_data['feature_cols']
        profiler.feature_ranges = model_data['feature_ranges']
        profiler.category_stats = model_data['category_stats']
        
        print(f"✓ Model loaded from {filepath}")
        return profiler


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Load data
    df = pd.read_excel('dataSET__1_.xlsx')
    
    # Initialize profiler
    profiler = SiteProfiler(df, target_col='cars_washed(Actual)', n_categories=3)
    
    # Example 1: Predict for a single location
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Location Prediction")
    print("="*80)
    
    new_location = {
        'total_sunshine_hours': 3200,
        'tunnel_length (in ft.)': 115,
        'days_pleasant_temp': 145,
        'total_precipitation_mm': 1250,
        'Nearest StreetLight US Hourly-Ttl AADT': 12000,
        'total_weekly_operational_hours': 80,
        'competitors_count': 1,
        'distance_from_nearest_walmart': 2.5,
        'distance_from_nearest_target': 3.0,
        'nearby_traffic_lights_count': 8
    }
    
    prediction = profiler.predict(new_location, return_details=True)
    
    print(f"\nPredicted Category: {prediction['predicted_category']}")
    print(f"Confidence: {prediction['confidence']:.1f}%")
    print(f"\nExpected Annual Volume:")
    print(f"  Conservative (25th %ile): {prediction['expected_volume']['conservative']:>10,.0f} cars/year")
    print(f"  Likely (Median):          {prediction['expected_volume']['likely']:>10,.0f} cars/year")
    print(f"  Optimistic (75th %ile):   {prediction['expected_volume']['optimistic']:>10,.0f} cars/year")
    
    print(f"\nCategory Scores:")
    for cat, score in prediction['category_scores'].items():
        print(f"  {cat:20s}: {score:.3f}")
    
    # Example 2: Compare multiple locations
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Multiple Candidate Locations")
    print("="*80)
    
    locations = [
        {
            'total_sunshine_hours': 3400,
            'tunnel_length (in ft.)': 130,
            'days_pleasant_temp': 160,
            'competitors_count': 0
        },
        {
            'total_sunshine_hours': 3100,
            'tunnel_length (in ft.)': 100,
            'days_pleasant_temp': 130,
            'competitors_count': 2
        },
        {
            'total_sunshine_hours': 3250,
            'tunnel_length (in ft.)': 120,
            'days_pleasant_temp': 145,
            'competitors_count': 1
        }
    ]
    
    location_names = ['Site A (Optimal)', 'Site B (Budget)', 'Site C (Standard)']
    comparison = profiler.compare_locations(locations, location_names)
    
    print("\n", comparison.to_string(index=False))
    
    # Example 3: Get feature ranges
    print("\n" + "="*80)
    print("EXAMPLE 3: Feature Range Reference")
    print("="*80)
    
    feature = 'total_sunshine_hours'
    ranges = profiler.get_feature_ranges(feature)
    print(f"\n{feature}:")
    print(ranges.to_string(index=False))
    
    # Export and save
    profiler.export_ranges('feature_ranges_export.csv')
    profiler.save_model('site_profiler_model.json')
    
    print("\n" + "="*80)
    print("✓ All examples completed successfully!")
    print("="*80)
