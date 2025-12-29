"""

Example 2: Builder Pattern & Feature Selection
==============================================

This example demonstrates the fluent builder pattern for pipeline configuration
and the TreeBasedFeatureSelector for dimensionality reduction.

Topics covered:

- TSForecastingBuilder fluent integration pipeline
- Method chaining configuration
- TreeBasedFeatureSelector usage
- Feature importance analysis
- Integration with pipeline

"""

import warnings
warnings.filterwarnings("ignore", category=Warning)

from tsforecasting import (
    TSForecastingBuilder,
    TreeBasedFeatureSelector,
    TimeSeriesDatasetGenerator,
    Processing,
    model_configurations,
)


# =============================================================================
# 1. BUILDER PATTERN
# =============================================================================

print("=" * 70)
print("1. BUILDER PATTERN")
print("=" * 70)

# Generate data
data = TimeSeriesDatasetGenerator.generate(
    n_samples=250,
    granularity="1mo",
    patterns=["trend", "seasonal"],
    random_state=42,
)

print(f"Data shape: {data.shape}")


# -----------------------------------------------------------------------------
# 1.1 Basic Builder Usage
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.1 Basic Builder Usage")
print("-" * 70)

# Fluent configuration with method chaining
pipeline = (
    TSForecastingBuilder()
    .with_train_size(0.80)
    .with_lags(12)
    .with_horizon(6)
    .with_sliding_size(6)
    .with_models(['RandomForest', 'XGBoost'])
    .with_granularity('1mo')
    .with_metric('MAE')
    .build()
)

print("Pipeline built with builder pattern:")
print(f"  train_size: {pipeline.train_size}")
print(f"  lags: {pipeline.lags}")
print(f"  horizon: {pipeline.horizon}")

# -----------------------------------------------------------------------------
# 1.2 Full Builder Configuration
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.2 Full Builder Configuration")
print("-" * 70)

# Get hyperparameters
hparams = model_configurations()
hparams["RandomForest"]["n_estimators"] = 30
hparams["XGBoost"]["n_estimators"] = 30

# Full configuration
pipeline = (
    TSForecastingBuilder()
    
    # Data split
    .with_train_size(0.80)
    
    # Time series parameters
    .with_lags(10)
    .with_horizon(5)
    .with_sliding_size(5)
    
    # Model selection
    .with_models(['RandomForest', 'GBR', 'XGBoost'])
    
    # Hyperparameters
    .with_hparameters(hparams)
    
    # Evaluation settings
    .with_granularity('1mo')
    .with_metric('MAE')
    
    # Preprocessing options
    .with_preprocessing(
        scaler='standard',       # Options: 'standard', 'minmax', 'robust'
        datetime_features=True,  # Add date-based features
    )
    
    # Build the pipeline
    .build()
)

print("Full pipeline configuration:")
print(f"  train_size: {pipeline.train_size}")
print(f"  lags: {pipeline.lags}")
print(f"  horizon: {pipeline.horizon}")
print(f"  sliding_size: {pipeline.sliding_size}")
print(f"  granularity: {pipeline.granularity}")
print(f"  metric: {pipeline.metric}")

# Fit the pipeline
print("\n Fitting pipeline...")
pipeline.fit_forecast(data)

print(f"\n Best model: {pipeline.selected_model}")
print(pipeline.history().leaderboard)

# -----------------------------------------------------------------------------
# 1.3 Builder Methods Reference
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.3 Builder Methods Reference")
print("-" * 70)

print("""
Available builder methods:

  .with_train_size(float)       # Range: 0.3 to 0.95
  .with_lags(int)               # Number of lagged features
  .with_horizon(int)            # Forecast horizon
  .with_sliding_size(int)       # Window expansion size
  .with_models(list)            # List of model names
  .with_hparameters(dict)       # Model hyperparameters
  .with_granularity(str)        # '1m', '30m', '1h', '1d', '1wk', '1mo'
  .with_metric(str)             # 'MAE', 'MAPE', 'MSE'
  .with_preprocessing(...)      # Scaler and feature options
  .build()                      # Create TSForecasting instance
""")


# =============================================================================
# 2. FEATURE SELECTION (MULTIVARIATE)
# =============================================================================

print("\n" + "=" * 70)
print("2. FEATURE SELECTION (MULTIVARIATE)")
print("=" * 70)

print("""
TreeBasedFeatureSelector computes feature importances across ALL forecast
horizons, using attention-weighted aggregation where closer horizons
(h1, h2, ...) receive higher weight than distant ones.

Weight formula: w_h = decay^(h-1), normalized to sum to 1
Default decay=0.8: h1=0.36, h2=0.29, h3=0.23, h4=0.18, h5=0.15 (for 5 horizons)
""")


# -----------------------------------------------------------------------------
# 2.1 Prepare Data for Feature Selection
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.1 Prepare Data for Feature Selection")
print("-" * 70)

# Generate larger dataset
data = TimeSeriesDatasetGenerator.generate(
    n_samples=300,
    granularity="1mo",
    patterns=["trend", "seasonal", "regime_change"],
    random_state=42,
)

# Transform to supervised format
processor = Processing()
timeseries = processor.make_timeseries(
    dataset=data,
    window_size=15,      # More lags = more features
    horizon=5,
    datetime_engineering=True,
)

# Remove NaN rows
target_cols = [c for c in timeseries.columns if c.startswith('y_horizon_')]
valid_mask = ~timeseries[target_cols].isna().any(axis=1)
timeseries = timeseries[valid_mask].copy()

# Separate features and targets (ALL horizons)
feature_cols = [c for c in timeseries.columns 
                if not c.startswith('y_horizon_') and c != 'Date']
X = timeseries[feature_cols]
y = timeseries[target_cols]  # All horizons as DataFrame

print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape} (multi-horizon)")
print(f"\nTarget columns: {list(y.columns)}")
print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols[:10]):
    print(f"  {col}")
if len(feature_cols) > 10:
    print(f"  ... and {len(feature_cols) - 10} more")


# -----------------------------------------------------------------------------
# 2.2 TreeBasedFeatureSelector Configuration
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.2 TreeBasedFeatureSelector Configuration")
print("-" * 70)

print("""
Available algorithms: 'RandomForest', 'ExtraTrees', 'GBR'

Parameters:
  algorithm           - Tree model for importance calculation
  n_estimators        - Number of trees
  relevance_threshold - Cumulative importance to retain (0.5 to 1.0)
  horizon_decay       - Weight decay for horizons (0.5 to 1.0)
                        Lower = more emphasis on near-term horizons
  random_state        - Reproducibility seed
""")

# Create selector with multivariate support
selector = TreeBasedFeatureSelector(
    algorithm="ExtraTrees",
    n_estimators=100,
    relevance_threshold=0.95,
    horizon_decay=0.8,  # h1 weighted ~2x more than h5
    random_state=42,
)

print("\nSelector configured:")
print("  algorithm: ExtraTrees")
print("  n_estimators: 100")
print("  relevance_threshold: 0.95")
print("  horizon_decay: 0.8")


# -----------------------------------------------------------------------------
# 2.3 Fit and Analyze Feature Importance (Multi-Horizon)
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.3 Fit and Analyze Feature Importance (Multi-Horizon)")
print("-" * 70)

# Fit selector with ALL horizons
selector.fit(X, y)

# Get horizon weights
print(f"\nHorizon weights (decay={selector._horizon_decay}):")
weights = selector.horizon_weights
for i, w in enumerate(weights):
    print(f"  Horizon {i+1}: {w:.4f} ({w*100:.1f}%)")

# Get aggregated importances
print("\nAggregated Feature Importances (top 15):")
importances = selector.feature_importances
print(importances.head(15).to_string(index=False))

# Get per-horizon importances
print("\nImportance by Horizon (top 10 features):")
comparison = selector.get_importance_comparison()
print(comparison.head(10).to_string(index=False))

# Selected features
selected = selector.selected_features
print(f"\n✓ Selected features: {len(selected)} / {len(feature_cols)}")
print(f"  Features retained: {selected}")

# Selection summary
print("\nSelection Summary:")
summary = selector.get_selection_summary()
for k, v in summary.items():
    print(f"  {k}: {v}")


# -----------------------------------------------------------------------------
# 2.4 Transform Data
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.4 Transform Data")
print("-" * 70)

# Transform to selected features only
X_selected = selector.transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_selected.shape}")
print(f"Dimensionality reduction: {X.shape[1]} → {X_selected.shape[1]} features")
print(f"Reduction: {100 * (1 - X_selected.shape[1] / X.shape[1]):.1f}%")


# -----------------------------------------------------------------------------
# 2.5 Feature Selection Strategies
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.5 Feature Selection Strategies")
print("-" * 70)

# Strategy 1: High threshold (keep most features)
selector_high = TreeBasedFeatureSelector(
    algorithm="ExtraTrees",
    relevance_threshold=0.99,
    horizon_decay=0.8,
)
selector_high.fit(X, y)
print(f"Threshold 0.99, decay 0.8: {len(selector_high.selected_features)} features")



aggregated = selector.feature_importances         # Aggregated Importances (weighted mean)
by_horizon = selector.importances_by_horizon      # Importances by Horizon
comparison = selector.get_importance_comparison() # Side-by-Side Comparison (horizons + weighted mean)
weights = selector.horizon_weights                # Horizon Weights Used


# Strategy 2: Medium threshold (balanced)
selector_med = TreeBasedFeatureSelector(
    algorithm="ExtraTrees",
    relevance_threshold=0.90,
    horizon_decay=0.8,
)
selector_med.fit(X, y)
print(f"Threshold 0.90, decay 0.8: {len(selector_med.selected_features)} features")

# Strategy 3: Low threshold (aggressive reduction)
selector_low = TreeBasedFeatureSelector(
    algorithm="ExtraTrees",
    relevance_threshold=0.80,
    horizon_decay=0.8,
)
selector_low.fit(X, y)
print(f"Threshold 0.80, decay 0.8: {len(selector_low.selected_features)} features")

# Compare decay factors
print("\nHorizon decay comparison (threshold=0.95):")
for decay in [0.5, 0.7, 0.8, 0.9, 1.0]:
    sel = TreeBasedFeatureSelector(
        algorithm="ExtraTrees",
        relevance_threshold=0.95,
        horizon_decay=decay,
    )
    sel.fit(X, y)
    weights_str = ", ".join([f"h{i+1}={w:.2f}" for i, w in enumerate(sel.horizon_weights)])
    print(f"  decay={decay}: {len(sel.selected_features)} features (weights: {weights_str})")

# Compare algorithms
print("\nAlgorithm comparison (threshold=0.95, decay=0.8):")
for algo in ["RandomForest", "ExtraTrees", "GBR"]:
    sel = TreeBasedFeatureSelector(
        algorithm=algo,
        relevance_threshold=0.95,
        horizon_decay=0.8,
    )
    sel.fit(X, y)
    print(f"  {algo}: {len(sel.selected_features)} features")


# =============================================================================
# 3. INTEGRATED WORKFLOW
# =============================================================================

print("\n" + "=" * 70)
print("3. INTEGRATED WORKFLOW")
print("=" * 70)

print("""
Combining builder pattern with multivariate feature selection.
""")

# Step 1: Generate data
data = TimeSeriesDatasetGenerator.generate(
    n_samples=300,
    granularity="1mo",
    patterns=["trend", "seasonal"],
    random_state=42,
)

# Step 2: Build pipeline with builder
pipeline = (
    TSForecastingBuilder()
    .with_train_size(0.80)
    .with_lags(12)
    .with_horizon(5)
    .with_sliding_size(5)
    .with_models(['RandomForest', 'XGBoost'])
    .with_metric('MAE')
    .with_granularity('1mo')
    .build()
)

# Step 3: Fit and forecast
pipeline.fit_forecast(data)

# Step 4: Results
print("\nWorkflow Results:")
print(f"  Best model: {pipeline.selected_model}")
print("\n  Leaderboard:")
print(pipeline.history().leaderboard.to_string(index=False))

forecast = pipeline.forecast(interval_method="ensemble")
print("\n  Forecast:")
print(forecast[['Date', 'y', 'y_lower_90', 'y_upper_90']].to_string(index=False))


# =============================================================================
# 4. QUICK REFERENCE
# =============================================================================

print("\n" + "=" * 70)
print("4. QUICK REFERENCE")
print("=" * 70)

print("""
BUILDER PATTERN:
----------------
from tsforecasting import TSForecastingBuilder

pipeline = (
    TSForecastingBuilder()
    .with_train_size(0.80)
    .with_lags(12)
    .with_horizon(6)
    .with_models(['RandomForest', 'XGBoost'])
    .with_metric('MAE')
    .build()
)
pipeline.fit_forecast(data)


MULTIVARIATE FEATURE SELECTION:
-------------------------------
from tsforecasting import TreeBasedFeatureSelector

# Single horizon (backward compatible)
selector = TreeBasedFeatureSelector(relevance_threshold=0.95)
selector.fit(X, y_horizon_1)

# Multi-horizon with attention weighting
selector = TreeBasedFeatureSelector(
    algorithm="ExtraTrees",
    relevance_threshold=0.95,
    horizon_decay=0.8,  # h1 weighted more than h5
)
selector.fit(X, y_all_horizons)  # y can be DataFrame with multiple columns

# Analyze
print(selector.feature_importances)       # Aggregated (weighted mean)
print(selector.importances_by_horizon)    # Per-horizon breakdown
print(selector.get_importance_comparison())  # Side-by-side comparison
print(selector.get_selection_summary())   # Full summary with weights

# Transform
X_reduced = selector.transform(X)


HORIZON DECAY EXPLAINED:
------------------------
decay=1.0  → Equal weight to all horizons
decay=0.8  → h1=36%, h2=29%, h3=23%, h4=18%, h5=15% (for 5 horizons)
decay=0.5  → h1=52%, h2=26%, h3=13%, h4=6%, h5=3% (aggressive near-term focus)
""")













