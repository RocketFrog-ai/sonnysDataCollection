"""Portfolio percentile scoring vs Proforma: raw percentile, invert when lower-is-better."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json


class PercentileProfiler:
    """Percentile-of-portfolio scoring per feature; invert score when lower values are better."""

    def __init__(self, df: pd.DataFrame):
        """Load numeric feature columns and precompute distributions."""
        self.df = df.copy()
        
        # Identify features
        exclude = ['cars_washed(Actual)', 'full_site_address', 'Address']
        self.feature_cols = [col for col in df.columns 
                            if col not in exclude and df[col].dtype in ['int64', 'float64']]
        
        # Define directionality (higher vs lower is better per feature)
        self._define_directions()
        
        # Precompute distributions
        self._analyze_distributions()
        
        print(f"✓ Percentile Profiler initialized")
        print(f"  - {len(self.df)} sites")
        print(f"  - {len(self.feature_cols)} features")
    
    def _define_directions(self):
        """Populate higher_is_better / lower_is_better sets from business rules."""
        # HIGHER IS BETTER
        self.higher_is_better = {
            # Weather - good conditions
            'total_sunshine_hours': 'More sunshine is better',
            'days_pleasant_temp': 'More pleasant days is better',
            
            # Traffic - more traffic is better
            'Nearest StreetLight US Hourly-Ttl AADT': 'More traffic is better',
            '2nd Nearest StreetLight US Hourly-Ttl AADT': 'More traffic is better',
            '3rd Nearest StreetLight US Hourly-Ttl AADT': 'More traffic is better',
            '4th Nearest StreetLight US Hourly-Ttl AADT': 'More traffic is better',
            '5th Nearest StreetLight US Hourly-Ttl AADT': 'More traffic is better',
            '6th Nearest StreetLight US Hourly-Ttl AADT': 'More traffic is better',
            'Nearest StreetLight US Hourly-ttl_breakfast': 'More traffic is better',
            'Nearest StreetLight US Hourly-ttl_lunch': 'More traffic is better',
            'Nearest StreetLight US Hourly-ttl_afternoon': 'More traffic is better',
            'Nearest StreetLight US Hourly-ttl_dinner': 'More traffic is better',
            'Nearest StreetLight US Hourly-ttl_night': 'More traffic is better',
            'Nearest StreetLight US Hourly-ttl_overnight': 'More traffic is better',
            'nearby_traffic_lights_count': 'More traffic signals = more traffic',
            
            # Infrastructure
            'tunnel_length (in ft.)': 'Longer tunnel is better',
            'total_weekly_operational_hours': 'More hours is better',
            
            # Retail proximity - more stores nearby
            'Count of ChainXY VT - Building Supplies': 'More stores is better',
            'Count of ChainXY VT - Department Store': 'More stores is better',
            'Count of ChainXY VT - Grocery': 'More stores is better',
            'Count of ChainXY VT - Mass Merchant': 'More stores is better',
            'Count of ChainXY VT - Real Estate Model': 'More stores is better',
            'Sum ChainXY': 'More stores is better',
            'count_of_target_5miles': 'More Targets nearby is better',
            'count_of_costco_5miles': 'More Costcos nearby is better',
            'count_of_walmart_5miles': 'More Walmarts nearby is better',
            'count_of_bestbuy_5miles': 'More Best Buys nearby is better',
            
            # Competition rating
            'competitor_1_google_user_rating_count': 'More reviews = more established area',
            # Proforma-v2 columns
            'weather_total_sunshine_hours': 'More sunshine is better',
            'weather_days_pleasant_temp': 'More pleasant days is better',
            'nearest_gas_station_rating': 'Higher rating is better',
            'nearest_gas_station_rating_count': 'More reviews is better',
            'competitor_1_google_rating': 'Higher competitor rating',
            'competitor_1_rating_count': 'More reviews = more established',
            # Proforma retailer counts (higher is better)
            'other_grocery_count_1mile': 'More grocery options nearby is better',
            'count_food_joints_0_5miles (0.5 mile)': 'More food options nearby is better',
        }
        
        # LOWER IS BETTER (e.g. distance to anchor: closer is better)
        self.lower_is_better = {
            # Weather - bad conditions (less is better)
            'total_precipitation_mm': 'Less rain is better',
            'rainy_days': 'Fewer rainy days is better',
            'total_snowfall_cm': 'Less snow is better',
            'snowy_days': 'Fewer snowy days is better',
            'days_below_freezing': 'Fewer freezing days is better',
            'avg_daily_max_windspeed_ms': 'Less wind is better',
            
            # Competition - fewer/farther competitors
            'competitors_count': 'Fewer competitors is better',
            'competitor_1_distance_miles': 'Actually this is tricky - farther might mean less competition OR less traffic',
            
            # Distance to amenities — closer is better
            'distance_from_nearest_target': 'Closer to Target is better',
            'distance_from_nearest_costco': 'Closer to Costco is better',
            'distance_from_nearest_walmart': 'Closer to Walmart is better',
            'distance_from_nearest_bestbuy': 'Closer to Best Buy is better',
            
            # Distance to traffic lights - closer is better
            'distance_nearest_traffic_light_1': 'Closer to signals is better',
            'distance_nearest_traffic_light_2': 'Closer to signals is better',
            'distance_nearest_traffic_light_3': 'Closer to signals is better',
            'distance_nearest_traffic_light_4': 'Closer to signals is better',
            'distance_nearest_traffic_light_5': 'Closer to signals is better',
            'distance_nearest_traffic_light_6': 'Closer to signals is better',
            'distance_nearest_traffic_light_7': 'Closer to signals is better',
            'distance_nearest_traffic_light_8': 'Closer to signals is better',
            'distance_nearest_traffic_light_9': 'Closer to signals is better',
            'distance_nearest_traffic_light_10': 'Closer to signals is better',
            # Proforma-v2 columns
            'weather_total_precipitation_mm': 'Less rain is better',
            'weather_rainy_days': 'Fewer rainy days is better',
            'weather_total_snowfall_cm': 'Less snow is better',
            'weather_days_below_freezing': 'Fewer freezing days is better',
            'weather_avg_daily_max_windspeed_ms': 'Less wind is better',
            'nearest_gas_station_distance_miles': 'Closer to gas station is better',
            'competitors_count_4miles': 'Fewer competitors is better',
            'competitor_1_distance_miles': 'Farther from competitor is better',
            # Proforma retailer distances (closer is better)
            'distance_nearest_costco(5 mile)': 'Closer to Costco is better',
            'distance_nearest_walmart(5 mile)': 'Closer to Walmart is better',
            'distance_nearest_target (5 mile)': 'Closer to Target is better',
        }
    
    def _analyze_distributions(self):
        """For each feature: min, max, full distribution for percentiles, and summary stats."""
        self.distributions = {}
        
        for feature in self.feature_cols:
            data = self.df[feature].dropna().values

            skew_val = float(stats.skew(data)) if len(data) > 2 else 0.0
            if len(np.unique(data)) <= 6:
                shape = "discrete"
            elif skew_val > 0.5:
                shape = "right-skewed"
            elif skew_val < -0.5:
                shape = "left-skewed"
            elif abs(skew_val) <= 0.5:
                shape = "symmetric"
            else:
                shape = "symmetric"

            self.distributions[feature] = {
                'data': data,
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'median': float(np.median(data)),
                'std': float(np.std(data)),
                'count': len(data),
                'skew': skew_val,
                'shape': shape
            }
    
    def score_feature_percentile(self, feature: str, value: float) -> Dict:
        """
        Rank the value in the portfolio distribution (percentile), then map to a 0–100 score.
        For lower-is-better features, invert: score = 100 - raw percentile.
        """
        if feature not in self.distributions:
            return {'error': 'Feature not found'}
        
        dist = self.distributions[feature]
        data = dist['data']
        
        raw_percentile = stats.percentileofscore(data, value, kind='rank')
        
        if feature in self.lower_is_better:
            direction = 'lower_is_better'
            # High raw percentile = worse (e.g. farther when closer is better) → invert
            final_score = 100 - raw_percentile
            
            interpretation = self._interpret_lower_is_better(raw_percentile)
            
        else:  # higher_is_better
            direction = 'higher_is_better'
            # If you're at 94.3%ile and higher is better, that's GOOD
            # Score = 94.3
            final_score = raw_percentile
            
            interpretation = self._interpret_higher_is_better(raw_percentile)
        
        return {
            'feature': feature,
            'value': value,
            'min': dist['min'],
            'max': dist['max'],
            'median': dist['median'],
            'raw_percentile': round(raw_percentile, 2),
            'final_score': round(final_score, 2),  # This is what we use
            'direction': direction,
            'interpretation': interpretation,
            'direction_note': self.lower_is_better.get(feature) or self.higher_is_better.get(feature, '')
        }
    
    def _interpret_lower_is_better(self, percentile: float) -> str:
        """
        For "lower is better" features.
        
        High percentile = you have HIGH value = BAD
        Low percentile = you have LOW value = GOOD
        
        Example: distance_from_costco
        - 10th percentile: You're at 10% → You're CLOSER than 90% → EXCELLENT
        - 90th percentile: You're at 90% → You're FARTHER than 90% → POOR
        """
        if percentile <= 10:
            return "Excellent (top 10% - lowest values)"
        elif percentile <= 25:
            return "Very Good (top 25% - low values)"
        elif percentile <= 50:
            return "Good (below median)"
        elif percentile <= 75:
            return "Fair (above median)"
        elif percentile <= 90:
            return "Poor (top 25% - high values)"
        else:
            return "Very Poor (top 10% - highest values)"
    
    def _interpret_higher_is_better(self, percentile: float) -> str:
        """
        For "higher is better" features.
        
        High percentile = you have HIGH value = GOOD
        Low percentile = you have LOW value = BAD
        """
        if percentile >= 90:
            return "Excellent (top 10%)"
        elif percentile >= 75:
            return "Very Good (top 25%)"
        elif percentile >= 50:
            return "Good (above median)"
        elif percentile >= 25:
            return "Fair (below median)"
        elif percentile >= 10:
            return "Poor (bottom 25%)"
        else:
            return "Very Poor (bottom 10%)"
    
    def score_location(self, location_features: Dict[str, float]) -> Dict:
        """
        Score all features for a location.
        
        Per-feature percentile-based scores, then an overall mean.
        """
        feature_scores = {}
        
        for feature, value in location_features.items():
            if feature in self.feature_cols:
                score_info = self.score_feature_percentile(feature, value)
                feature_scores[feature] = score_info
        
        # Calculate overall score
        scores = [fs['final_score'] for fs in feature_scores.values() 
                 if 'final_score' in fs]
        overall_score = np.mean(scores) if scores else 0
        
        # Top strengths / weaknesses by final score
        sorted_features = sorted(feature_scores.items(), 
                                key=lambda x: x[1].get('final_score', 0), 
                                reverse=True)
        
        return {
            'overall_score': round(overall_score, 2),
            'feature_scores': feature_scores,
            'top_strengths': sorted_features[:10],
            'top_weaknesses': list(reversed(sorted_features[-10:])),
            'total_features': len(feature_scores)
        }
    
    def print_detailed_report(self, result: Dict):
        """
        Print a text report of strengths and weaknesses.
        """
        print("="*80)
        print("SCORING REPORT")
        print("="*80)
        print(f"\nOVERALL SCORE: {result['overall_score']:.2f}/100")
        print(f"(Average of all {result['total_features']} feature scores)")
        
        print("\n" + "="*80)
        print("TOP 10 STRENGTHS")
        print("="*80)
        print(f"{'Feature':<40s} {'Value':>10s} {'Percentile':>12s} {'Score':>8s} {'Status'}")
        print("-"*80)
        
        for feature, details in result['top_strengths']:
            print(f"{feature[:38]:<40s} {details['value']:>10,.1f} {details['raw_percentile']:>11.2f}% {details['final_score']:>7.1f} {details['interpretation']}")
        
        print("\n" + "="*80)
        print("TOP 10 WEAKNESSES")
        print("="*80)
        print(f"{'Feature':<40s} {'Value':>10s} {'Percentile':>12s} {'Score':>8s} {'Status'}")
        print("-"*80)
        
        for feature, details in result['top_weaknesses']:
            print(f"{feature[:38]:<40s} {details['value']:>10,.1f} {details['raw_percentile']:>11.2f}% {details['final_score']:>7.1f} {details['interpretation']}")
    
    def explain_scoring_with_example(self, feature: str, value: float):
        """Print a step-by-step walkthrough for one feature and value."""
        score_info = self.score_feature_percentile(feature, value)
        
        print("\n" + "="*80)
        print(f"METHODOLOGY EXAMPLE: {feature}")
        print("="*80)
        
        print(f"\nYour Value: {value:.2f}")
        print(f"\nAll {len(self.df)} sites distribution:")
        print(f"  Min:    {score_info['min']:.2f}")
        print(f"  Median: {score_info['median']:.2f}")
        print(f"  Max:    {score_info['max']:.2f}")
        
        print(f"\nStep 1: Calculate Percentile Rank")
        print(f"  Your value ({value:.2f}) ranks at: {score_info['raw_percentile']:.2f}%ile")
        print(f"  (You are better than {score_info['raw_percentile']:.2f}% of sites on RAW value)")
        
        print(f"\nStep 2: Apply Business Logic - {score_info['direction']}")
        print(f"  {score_info['direction_note']}")
        
        if score_info['direction'] == 'lower_is_better':
            print(f"\n  Since LOWER is better:")
            print(f"  - Being at {score_info['raw_percentile']:.2f}%ile means you have HIGHER value than {score_info['raw_percentile']:.2f}%")
            print(f"  - Which is BAD (because lower is better)")
            print(f"  - So we INVERT: Score = 100 - {score_info['raw_percentile']:.2f} = {score_info['final_score']:.2f}")
        else:
            print(f"\n  Since HIGHER is better:")
            print(f"  - Being at {score_info['raw_percentile']:.2f}%ile means you have HIGHER value than {score_info['raw_percentile']:.2f}%")
            print(f"  - Which is GOOD (because higher is better)")
            print(f"  - So Score = {score_info['raw_percentile']:.2f}")
        
        print(f"\nStep 3: Final Score")
        print(f"  {score_info['final_score']:.2f}/100 - {score_info['interpretation']}")
        
        print("\n" + "="*80)

    def print_distribution_stats(self):
        """
        Print distribution statistics (min, max, mean, median, std) for all features.
        """
        print("\n" + "="*140)
        print("DISTRIBUTION STATISTICS - ALL FEATURES")
        print("="*140)
        print(f"{'Feature':<42} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'Std':>10} {'Count':>6} {'Dir':>8} {'Shape':>14}")
        print("-"*140)

        for feature in self.feature_cols:
            d = self.distributions[feature]
            direction = "lower✓" if feature in self.lower_is_better else "higher✓"
            shape = d.get("shape", "—")
            name = (feature[:39] + "..") if len(feature) > 42 else feature
            print(f"{name:<42} {d['min']:>10.2f} {d['max']:>10.2f} {d['mean']:>10.2f} {d['median']:>10.2f} {d['std']:>10.2f} {d['count']:>6} {direction:>8} {shape:>14}")

        print("-"*140)
        print("Shape: discrete (≤6 unique) | right-skewed (skew>0.5) | left-skewed (skew<-0.5) | symmetric")
        print("Direction: lower✓ = lower is better, higher✓ = higher is better")
        print("="*140)

    def plot_all_distributions(self, output_dir: str = "distribution_plots", figsize: Tuple[int, int] = (6, 4)):
        """
        Plot histogram distributions for all features in a single grid image.
        Saves all_features_distributions.png.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[Skip] matplotlib not installed. Run: pip install matplotlib")
            return

        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n = len(self.feature_cols)
        if n == 0:
            print("\n[Skip] No features to plot.")
            return
        cols = 5
        rows = (n + cols - 1) // cols

        # Single figure with all subplots (better for overview)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        axes_flat = axes.flatten()

        for i, feature in enumerate(self.feature_cols):
            ax = axes_flat[i]
            d = self.distributions[feature]
            data = d["data"]
            shape = d.get("shape", "—")

            ax.hist(data, bins=min(30, max(10, len(np.unique(data)))), edgecolor="white", alpha=0.8)
            ax.axvline(d["median"], color="red", linestyle="--", linewidth=2, label="median")
            ax.axvline(d["mean"], color="orange", linestyle=":", linewidth=1.5, label="mean")
            title = f"{feature[:35]}{'..' if len(feature) > 35 else ''} [{shape}]"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("")
            ax.legend(fontsize=6)
            ax.tick_params(axis="both", labelsize=7)
            direction = "↓" if feature in self.lower_is_better else "↑"
            ax.set_ylabel(f"n={d['count']} {direction}", fontsize=7)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle("Feature Distributions (↓ lower is better, ↑ higher is better)", fontsize=12, y=1.02)
        plt.tight_layout()
        grid_path = out / "all_features_distributions.png"
        plt.savefig(grid_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n✓ Saved distribution grid: {grid_path}")
        print("="*80)

