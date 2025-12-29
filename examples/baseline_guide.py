"""
Example 1: Baseline Pipeline
============================

This example demonstrates the main TSForecasting pipeline with all available
options for quick, high-level time series forecasting.

Topics covered:
- Data generation with TimeSeriesDatasetGenerator
- Model hyperparameter configuration
- TSForecasting pipeline setup and fitting
- Results extraction and forecasting
"""

import warnings
warnings.filterwarnings("ignore", category=Warning)

from tsforecasting import (
    TSForecasting,
    model_configurations,
    TimeSeriesDatasetGenerator,
)

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

print("=" * 70)
print("1. DATA GENERATION")
print("=" * 70)

# List available options
print("\nAvailable granularities:")
for g in TimeSeriesDatasetGenerator.list_available_granularities():
    print(f"  {g}")

print("\nAvailable patterns:")
for p in TimeSeriesDatasetGenerator.list_available_patterns():
    print(f"  {p}")

# Generate synthetic data
data = TimeSeriesDatasetGenerator.generate(
    n_samples=500,
    granularity="1mo",               # Options: "1m", "30m", "1h", "1d", "1wk", "1mo"
    patterns=["trend", "seasonal"],  # Options: "trend", "seasonal", "mixed", "random_walk"
                                     #          "multi_seasonal", "regime_change"
    trend_strength=0.3,              # Range: 0.0 to 1.0
    trend_type="linear",             # Options: "linear", "exponential"
    seasonality_period=12,           # Period in time steps
    seasonality_strength=1.5,        # Amplitude multiplier
    noise_level=0.1,                 # Range: 0.0 to 1.0
    start_date="2000-01-01",
    base_value=100.0,
    random_state=42,
)

print(f"\nGenerated data: {data.shape}")
print(data.head())

# Alternative: Quick generators
# data_2 = TimeSeriesDatasetGenerator.quick_monthly(n_samples=120)
# data_3 = TimeSeriesDatasetGenerator.quick_daily(n_samples=365)
# data_4 = TimeSeriesDatasetGenerator.quick_hourly(n_samples=720)

# =============================================================================
# 2. MODEL HYPERPARAMETERS
# =============================================================================

print("\n" + "=" * 70)
print("2. MODEL HYPERPARAMETERS")
print("=" * 70)

# Get default configurations
hparameters = model_configurations()

print("\nDefault configurations:")
for model, params in hparameters.items():
    print(f"\n  {model}:")
    for k, v in params.items():
        print(f"    {k}: {v}")

# Customize hyperparameters
hparameters["RandomForest"]["n_estimators"] = 50
hparameters["XGBoost"]["n_estimators"] = 50
hparameters["Catboost"]["iterations"] = 50
hparameters["KNN"]["n_neighbors"] = 5
hparameters["GBR"]["learning_rate"] = 0.05

print("\n Hyperparameters customized")

# =============================================================================
# 3. PIPELINE CONFIGURATION
# =============================================================================

print("\n" + "=" * 70)
print("3. PIPELINE CONFIGURATION")
print("=" * 70)

# Available models
AVAILABLE_MODELS = [
    'RandomForest',   # Ensemble of decision trees
    'ExtraTrees',     # Extremely randomized trees
    'GBR',            # Gradient Boosting Regressor
    'KNN',            # K-Nearest Neighbors
    'GeneralizedLR',  # Tweedie Regressor (GLM)
    'XGBoost',        # XGBoost
    #'Catboost',       # CatBoost
    #'AutoGluon',      # AutoGluon
]

# Available metrics
AVAILABLE_METRICS = ['MAE', 'MAPE', 'MSE']

# Available granularities
AVAILABLE_GRANULARITIES = ['1m', '30m', '1h', '1d', '1wk', '1mo']

print(f"Available models: {AVAILABLE_MODELS}")
print(f"Available metrics: {AVAILABLE_METRICS}")
print(f"Available granularities: {AVAILABLE_GRANULARITIES}")

# Configure pipeline
tsf = TSForecasting(
    train_size=0.80,            # Range: 0.3 to 0.95 (initial train proportion)
    # Time series parameters
    lags=12,                    # Number of lagged features (window size)
    horizon=6,                  # Number of future steps to predict
    sliding_size=6,             # Window expansion size per iteration
    
    # Model selection
    models=[
        'RandomForest',
        'GBR',
        'KNN',
        'XGBoost',
        #'Catboost',
        #'AutoGluon',
    ],
    
    # Hyperparameters
    hparameters=hparameters,
    
    # Evaluation settings
    granularity='1mo',          # Time granularity
    metric='MAE',               # Evaluation metric: 'MAE', 'MAPE', 'MSE'
)

print("\nPipeline configured:")
print(f"  train_size: {tsf.train_size}")
print(f"  lags: {tsf.lags}")
print(f"  horizon: {tsf.horizon}")
print(f"  sliding_size: {tsf.sliding_size}")
print(f"  metric: {tsf.metric}")


# =============================================================================
# 4. FIT AND EVALUATE
# =============================================================================

print("\n" + "=" * 70)
print("4. FIT AND EVALUATE")
print("=" * 70)

# Fit the pipeline (expanding window evaluation)
tsf.fit_forecast(dataset=data)

print("\n Pipeline fitted")
print(f"  Best model: {tsf.selected_model}")

# =============================================================================
# 5. RESULTS EXTRACTION
# =============================================================================

print("\n" + "=" * 70)
print("5. RESULTS EXTRACTION")
print("=" * 70)

# Get history
history = tsf.history()

print(f"\nHistory type: {type(history).__name__}")

# 5.1 Predictions DataFrame
print("\n5.1 Predictions (sample):")
predictions = history.predictions
print(f"  Shape: {predictions.shape}")
print(predictions.head())

# 5.2 Complete Performance (all windows, all horizons)
print("\n5.2 Performance Complete (sample):")
performance_complete = history.performance_complete
print(f"  Shape: {performance_complete.shape}")
print(performance_complete.head(10))

# 5.3 Performance by Horizon
print("\n5.3 Performance by Horizon:")
performance_horizon = history.performance_by_horizon
print(performance_horizon)

# 5.4 Leaderboard (model ranking)
print("\n5.4 Leaderboard:")
leaderboard = history.leaderboard
print(leaderboard.to_string(index=False))

# 5.5 Selected Model
print(f"\n5.5 Selected Model: {history.selected_model}")

# =============================================================================
# 6. GENERATE FORECAST
# =============================================================================

print("\n" + "=" * 70)
print("6. GENERATE FORECAST")
print("=" * 70)

# Generate future forecast
forecast = tsf.forecast(dataset=data,
                        interval_method="ensemble") # Options: "quantile", "conformal", "gaussian" 
                                                    # Interval_method: Method for prediction intervals
                                                    # - "ensemble": Average of all methods (default, most robust)
                                                    # - "quantile": Empirical percentiles (captures asymmetry)
                                                    # - "conformal": Coverage guarantee (symmetric)
                                                    # - "gaussian": Parametric mean ± z*std

print(f"\nForecast ({tsf.horizon} steps ahead):")
print(forecast[['Date','y']].to_string(index=False))

# =============================================================================
# 7. QUICK REFERENCE
# =============================================================================

print("\n" + "=" * 70)
print("7. QUICK REFERENCE")
print("=" * 70)

print("""
MINIMAL USAGE:
--------------
from tsforecasting import TSForecasting, TimeSeriesDatasetGenerator

data = TimeSeriesDatasetGenerator.quick_monthly(n_samples=200)
tsf = TSForecasting(lags=12, horizon=6, models=['RandomForest', 'XGBoost'])
tsf.fit_forecast(data)
forecast = tsf.forecast()


FULL CONFIGURATION:
-------------------
tsf = TSForecasting(
    train_size=0.80,            # Initial train proportion
    lags=12,                    # Lagged features count
    horizon=6,                  # Forecast steps
    sliding_size=10,            # Window expansion size
    models=['RandomForest', 'XGBoost'],
    hparameters=model_configurations(),
    granularity='1mo',          # '1m', '30m', '1h', '1d', '1wk', '1mo'
    metric='MAE',               # 'MAE', 'MAPE', 'MSE'
)


DATA GENERATION:
----------------
# Synthetic with patterns
data = TimeSeriesDatasetGenerator.generate(
    n_samples=300,
    granularity='1mo',
    patterns=['trend', 'seasonal'],
)

# Quick generators
data = TimeSeriesDatasetGenerator.quick_monthly()
data = TimeSeriesDatasetGenerator.quick_daily()
data = TimeSeriesDatasetGenerator.quick_hourly()
""")

print("=" * 70)
print("END OF EXAMPLE 1: BASELINE PIPELINE")
print("=" * 70)