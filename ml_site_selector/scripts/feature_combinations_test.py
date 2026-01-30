"""
Feature Combination Testing for Car Wash Prediction Model
Tests if adding interaction features improves model performance
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Project root (ml_site_selector) - script lives in scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataSET__1_.xlsx"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*100)
print(" FEATURE COMBINATION TESTING - CAR WASH PREDICTION MODEL")
print("="*100)
print()

# Load data
print("Loading data...")
df = pd.read_excel(DATA_PATH)
print(f"Dataset: {df.shape[0]} locations, {df.shape[1]} features")
print()

# Prepare baseline data
target = 'cars_washed(Actual)'
X_baseline = df.drop(columns=[target, 'full_site_address'])
y = df[target]

# Handle missing values
X_baseline_processed = X_baseline.copy()
for col in X_baseline_processed.columns:
    if X_baseline_processed[col].dtype in ['float64', 'int64']:
        if X_baseline_processed[col].isnull().any():
            median_val = X_baseline_processed[col].median()
            X_baseline_processed[col] = X_baseline_processed[col].fillna(median_val)

print("Target statistics:")
print(f"  Mean: {y.mean():,.0f} cars/year")
print(f"  Median: {y.median():,.0f} cars/year")
print(f"  Std Dev: {y.std():,.0f} cars/year")
print(f"  Range: {y.min():,.0f} to {y.max():,.0f}")
print()

# ============================================================================
# CREATE INTERACTION FEATURES
# ============================================================================

print("="*100)
print(" CREATING INTERACTION FEATURES")
print("="*100)
print()

X_with_interactions = X_baseline_processed.copy()

# Based on feature importance, create meaningful interactions
print("Adding interaction features based on domain knowledge:")
print()

# 1. OPERATIONAL EFFICIENCY: Tunnel length × Operating hours
X_with_interactions['tunnel_x_hours'] = (
    X_baseline_processed['tunnel_length (in ft.)'] * 
    X_baseline_processed['total_weekly_operational_hours']
)
print("- tunnel_x_hours: Tunnel length × Operating hours")
print("  Rationale: Longer tunnels with more hours = higher throughput capacity")

# 2. WEATHER FAVORABILITY: Sunshine × Pleasant days
X_with_interactions['sunshine_x_pleasant'] = (
    X_baseline_processed['total_sunshine_hours'] * 
    X_baseline_processed['days_pleasant_temp']
)
print("- sunshine_x_pleasant: Sunshine hours × Pleasant temperature days")
print("  Rationale: Combined weather quality drives car wash demand")

# 3. WEATHER ADVERSITY: Freezing days × Precipitation
X_with_interactions['freezing_x_precip'] = (
    X_baseline_processed['days_below_freezing'] * 
    X_baseline_processed['total_precipitation_mm']
)
print("- freezing_x_precip: Freezing days × Precipitation")
print("  Rationale: Winter salt/dirt creates higher wash frequency need")

# 4. TRAFFIC ACCESSIBILITY: Main traffic × Operating hours
X_with_interactions['traffic_x_hours'] = (
    X_baseline_processed['Nearest StreetLight US Hourly-Ttl AADT'] * 
    X_baseline_processed['total_weekly_operational_hours']
)
print("- traffic_x_hours: Main traffic count × Operating hours")
print("  Rationale: High traffic is only valuable if you're open to capture it")

# 5. RETAIL GRAVITY: Combined retail proximity score
# Inverse distance (closer = better)
X_with_interactions['retail_gravity'] = (
    1 / (X_baseline_processed['distance_from_nearest_costco'] + 0.1) +
    1 / (X_baseline_processed['distance_from_nearest_walmart'] + 0.1) +
    1 / (X_baseline_processed['distance_from_nearest_target'] + 0.1)
)
print("- retail_gravity: Combined retail proximity score")
print("  Rationale: Proximity to multiple retail anchors compounds foot traffic")

# 6. COMPETITION PRESSURE: Competitors × Their ratings
X_with_interactions['competition_pressure'] = (
    X_baseline_processed['competitors_count'] * 
    (X_baseline_processed['competitor_1_google_user_rating_count'] + 1)
)
print("- competition_pressure: Competitor count × Their popularity")
print("  Rationale: More established competitors = higher market saturation")

# 7. TRAFFIC LIGHT ACCESSIBILITY: Traffic lights × Distance
X_with_interactions['traffic_light_access'] = (
    X_baseline_processed['nearby_traffic_lights_count'] / 
    (X_baseline_processed['distance_nearest_traffic_light_1'] + 0.1)
)
print("- traffic_light_access: Traffic lights count / Distance")
print("  Rationale: More nearby lights = easier access from multiple directions")

# 8. WEATHER DEMAND DRIVER: Rainy + Snowy days (dirtier cars)
X_with_interactions['weather_demand'] = (
    X_baseline_processed['rainy_days'] + 
    X_baseline_processed['snowy_days'] * 2  # Snow creates more dirt
)
print("- weather_demand: Rainy days + (Snowy days × 2)")
print("  Rationale: Adverse weather = dirtier cars = more wash frequency")

# 9. TUNNEL EFFICIENCY RATIO: Tunnel per operating hour
X_with_interactions['tunnel_efficiency'] = (
    X_baseline_processed['tunnel_length (in ft.)'] / 
    (X_baseline_processed['total_weekly_operational_hours'] + 1)
)
print("- tunnel_efficiency: Tunnel length / Operating hours")
print("  Rationale: Measures throughput potential per hour open")

# 10. TRAFFIC CONCENTRATION: Primary vs secondary traffic ratio
X_with_interactions['traffic_concentration'] = (
    X_baseline_processed['Nearest StreetLight US Hourly-Ttl AADT'] / 
    (X_baseline_processed['2nd Nearest StreetLight US Hourly-Ttl AADT'] + 1)
)
print("- traffic_concentration: Primary / Secondary traffic")
print("  Rationale: Concentrated traffic on main road = better visibility")

# 11. PREMIUM LOCATION SCORE: Tunnel × Traffic × Retail proximity
X_with_interactions['premium_score'] = (
    X_baseline_processed['tunnel_length (in ft.)'] * 
    X_baseline_processed['Nearest StreetLight US Hourly-Ttl AADT'] * 
    (1 / (X_baseline_processed['distance_from_nearest_costco'] + 0.5))
)
print("- premium_score: Tunnel × Traffic × (1/Costco distance)")
print("  Rationale: Premium locations combine infrastructure, traffic, and retail")

# 12. WEATHER WASH INDEX: Pleasant days / Total precipitation
X_with_interactions['weather_wash_index'] = (
    X_baseline_processed['days_pleasant_temp'] / 
    (X_baseline_processed['total_precipitation_mm'] + 1)
)
print("- weather_wash_index: Pleasant days / Precipitation")
print("  Rationale: High ratio = ideal car washing weather more frequently")

print()
print(f"Total features created: {len(X_with_interactions.columns) - len(X_baseline_processed.columns)}")
print(f"New dataset shape: {X_with_interactions.shape}")
print()

# ============================================================================
# MODEL EVALUATION FUNCTION
# ============================================================================

def evaluate_model(X, y, model_name, verbose=True):
    """Evaluate model with 5-fold stratified cross-validation"""
    
    # Create stratified folds
    volume_quartiles = pd.qcut(y, q=4, labels=False, duplicates='drop')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        max_features=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # Cross-validation predictions
    y_pred = cross_val_predict(model, X, y, cv=cv.split(X, volume_quartiles))
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    correlation = np.corrcoef(y, y_pred)[0, 1]
    
    # Within-percentage accuracy
    errors = np.abs((y - y_pred) / y)
    within_10 = (errors <= 0.10).mean() * 100
    within_20 = (errors <= 0.20).mean() * 100
    within_30 = (errors <= 0.30).mean() * 100
    within_50 = (errors <= 0.50).mean() * 100
    
    results = {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'within_10': within_10,
        'within_20': within_20,
        'within_30': within_30,
        'within_50': within_50,
        'n_features': X.shape[1],
        'predictions': y_pred,
        'model': model
    }
    
    if verbose:
        print(f"Model: {model_name}")
        print(f"  Features: {X.shape[1]}")
        print(f"  MAE: {mae:,.0f} cars/year")
        print(f"  RMSE: {rmse:,.0f} cars/year")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {correlation:.4f} ({correlation*100:.2f}%)")
        print(f"  Within 10%: {within_10:.2f}%")
        print(f"  Within 20%: {within_20:.2f}%")
        print(f"  Within 30%: {within_30:.2f}%")
        print(f"  Within 50%: {within_50:.2f}%")
        print()
    
    return results

# ============================================================================
# COMPARE MODELS
# ============================================================================

print("="*100)
print(" MODEL COMPARISON: BASELINE vs WITH INTERACTION FEATURES")
print("="*100)
print()

print("Training BASELINE model (original features only)...")
print("-" * 80)
baseline_results = evaluate_model(X_baseline_processed, y, "Baseline (50 features)", verbose=True)

print("Training model WITH INTERACTION FEATURES...")
print("-" * 80)
interaction_results = evaluate_model(X_with_interactions, y, "With Interactions (62 features)", verbose=True)

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

print("="*100)
print(" RESULTS COMPARISON")
print("="*100)
print()

comparison_data = {
    'Metric': [
        'Number of Features',
        'MAE (cars/year)',
        'RMSE (cars/year)',
        'R² Score',
        'Correlation (%)',
        'Within 10% Accuracy',
        'Within 20% Accuracy',
        'Within 30% Accuracy',
        'Within 50% Accuracy'
    ],
    'Baseline': [
        baseline_results['n_features'],
        f"{baseline_results['mae']:,.0f}",
        f"{baseline_results['rmse']:,.0f}",
        f"{baseline_results['r2']:.4f}",
        f"{baseline_results['correlation']*100:.2f}%",
        f"{baseline_results['within_10']:.2f}%",
        f"{baseline_results['within_20']:.2f}%",
        f"{baseline_results['within_30']:.2f}%",
        f"{baseline_results['within_50']:.2f}%"
    ],
    'With Interactions': [
        interaction_results['n_features'],
        f"{interaction_results['mae']:,.0f}",
        f"{interaction_results['rmse']:,.0f}",
        f"{interaction_results['r2']:.4f}",
        f"{interaction_results['correlation']*100:.2f}%",
        f"{interaction_results['within_10']:.2f}%",
        f"{interaction_results['within_20']:.2f}%",
        f"{interaction_results['within_30']:.2f}%",
        f"{interaction_results['within_50']:.2f}%"
    ]
}

# Calculate improvements
improvements = {
    'Improvement': [
        f"+{interaction_results['n_features'] - baseline_results['n_features']}",
        f"{baseline_results['mae'] - interaction_results['mae']:+,.0f}",
        f"{baseline_results['rmse'] - interaction_results['rmse']:+,.0f}",
        f"{interaction_results['r2'] - baseline_results['r2']:+.4f}",
        f"{(interaction_results['correlation'] - baseline_results['correlation'])*100:+.2f}%",
        f"{interaction_results['within_10'] - baseline_results['within_10']:+.2f}%",
        f"{interaction_results['within_20'] - baseline_results['within_20']:+.2f}%",
        f"{interaction_results['within_30'] - baseline_results['within_30']:+.2f}%",
        f"{interaction_results['within_50'] - baseline_results['within_50']:+.2f}%"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Improvement'] = improvements['Improvement']

print(comparison_df.to_string(index=False))
print()

# ============================================================================
# VERDICT
# ============================================================================

print("="*100)
print(" VERDICT: SHOULD YOU USE INTERACTION FEATURES?")
print("="*100)
print()

corr_improvement = (interaction_results['correlation'] - baseline_results['correlation']) * 100
mae_improvement_pct = ((baseline_results['mae'] - interaction_results['mae']) / baseline_results['mae']) * 100
within30_improvement = interaction_results['within_30'] - baseline_results['within_30']

if corr_improvement > 1.0 and within30_improvement > 2.0:
    verdict = "YES - Significant improvement!"
    recommendation = "The interaction features provide meaningful improvements. Use this enhanced model."
elif corr_improvement > 0.5 and mae_improvement_pct > 1.0:
    verdict = "YES - Moderate improvement"
    recommendation = "The interaction features provide modest but real improvements. Worth using."
elif corr_improvement > 0:
    verdict = "MARGINAL - Minor improvement"
    recommendation = "Small improvement, but adds complexity. Use if you need every edge."
else:
    verdict = "NO - No improvement"
    recommendation = "Interaction features don't help. Stick with baseline model."

print(f"VERDICT: {verdict}")
print()
print("Key Changes:")
print(f"  - Correlation: {baseline_results['correlation']*100:.2f}% -> {interaction_results['correlation']*100:.2f}% ({corr_improvement:+.2f}%)")
print(f"  - MAE: {baseline_results['mae']:,.0f} -> {interaction_results['mae']:,.0f} ({mae_improvement_pct:+.2f}%)")
print(f"  - Within 30%: {baseline_results['within_30']:.2f}% -> {interaction_results['within_30']:.2f}% ({within30_improvement:+.2f}%)")
print()
print(f"RECOMMENDATION: {recommendation}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("="*100)
print(" GENERATING COMPARISON VISUALIZATIONS")
print("="*100)
print()

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Actual vs Predicted - Baseline
axes[0, 0].scatter(y, baseline_results['predictions'], alpha=0.5, s=30, color='blue')
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Cars Washed', fontsize=11)
axes[0, 0].set_ylabel('Predicted Cars Washed', fontsize=11)
axes[0, 0].set_title(f'Baseline Model\nCorr: {baseline_results["correlation"]*100:.2f}%, MAE: {baseline_results["mae"]:,.0f}', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted - With Interactions
axes[0, 1].scatter(y, interaction_results['predictions'], alpha=0.5, s=30, color='green')
axes[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Cars Washed', fontsize=11)
axes[0, 1].set_ylabel('Predicted Cars Washed', fontsize=11)
axes[0, 1].set_title(f'With Interaction Features\nCorr: {interaction_results["correlation"]*100:.2f}%, MAE: {interaction_results["mae"]:,.0f}', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals - Baseline
residuals_baseline = y - baseline_results['predictions']
axes[1, 0].scatter(baseline_results['predictions'], residuals_baseline, alpha=0.5, s=30, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Cars Washed', fontsize=11)
axes[1, 0].set_ylabel('Residuals', fontsize=11)
axes[1, 0].set_title('Baseline Model - Residuals', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals - With Interactions
residuals_interaction = y - interaction_results['predictions']
axes[1, 1].scatter(interaction_results['predictions'], residuals_interaction, alpha=0.5, s=30, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Cars Washed', fontsize=11)
axes[1, 1].set_ylabel('Residuals', fontsize=11)
axes[1, 1].set_title('With Interactions - Residuals', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "comparison_plots.png", dpi=150, bbox_inches='tight')
print("Saved: comparison_plots.png")

# 5. Feature Importance Comparison
print("Calculating feature importance for both models...")
baseline_model = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_leaf=5, 
    max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1
)
baseline_model.fit(X_baseline_processed, y)

interaction_model = RandomForestRegressor(
    n_estimators=300, max_depth=20, min_samples_leaf=5,
    max_features=0.5, random_state=RANDOM_STATE, n_jobs=-1
)
interaction_model.fit(X_with_interactions, y)

# Get top features from interaction model
interaction_importances = pd.DataFrame({
    'feature': X_with_interactions.columns,
    'importance': interaction_model.feature_importances_
}).sort_values('importance', ascending=False)

# Identify which are new interaction features
new_features = [col for col in X_with_interactions.columns if col not in X_baseline_processed.columns]
interaction_importances['is_new'] = interaction_importances['feature'].isin(new_features)

# Plot top 20 features with new ones highlighted
fig, ax = plt.subplots(figsize=(12, 8))
top_features = interaction_importances.head(20)
colors = ['#2ecc71' if is_new else '#3498db' for is_new in top_features['is_new']]
bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=9)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Top 20 Features (Green = New Interaction Features)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='New Interaction Features'),
    Patch(facecolor='#3498db', label='Original Features')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance_with_interactions.png", dpi=150, bbox_inches='tight')
print("Saved: feature_importance_with_interactions.png")

# 6. Performance metrics bar chart
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Correlation\n(%)', 'Within 10%\n(%)', 'Within 20%\n(%)', 'Within 30%\n(%)', 'Within 50%\n(%)']
baseline_values = [
    baseline_results['correlation'] * 100,
    baseline_results['within_10'],
    baseline_results['within_20'],
    baseline_results['within_30'],
    baseline_results['within_50']
]
interaction_values = [
    interaction_results['correlation'] * 100,
    interaction_results['within_10'],
    interaction_results['within_20'],
    interaction_results['within_30'],
    interaction_results['within_50']
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db')
bars2 = ax.bar(x + width/2, interaction_values, width, label='With Interactions', color='#2ecc71')

ax.set_ylabel('Percentage', fontsize=11)
ax.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "performance_comparison.png", dpi=150, bbox_inches='tight')
print("Saved: performance_comparison.png")

print()
print("="*100)
print(" ANALYSIS COMPLETE!")
print("="*100)
print()
print("Generated files (in figures/):")
print("  1. comparison_plots.png - Actual vs Predicted & Residuals")
print("  2. feature_importance_with_interactions.png - Top features highlighted")
print("  3. performance_comparison.png - Metrics side-by-side")
print()
print("RECOMMENDATION:")
print(recommendation)
print()
