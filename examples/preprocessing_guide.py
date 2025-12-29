"""
Example 3: Preprocessing & Decomposed Functionalities
=====================================================

This example demonstrates the internal preprocessing mechanisms and
decomposed components of TSForecasting for advanced usage.

Topics covered:
- Processing class and time series transformation
- Lag feature generation (window creation)
- Horizon target generation (multi-step)
- DateTime feature engineering
- Expanding window evaluation logic
- Direct model usage with BaseForecaster
- ModelRegistry and FORECASTER_CLASSES
- Metric strategies and evaluation
"""

import warnings
warnings.filterwarnings("ignore", category=Warning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsforecasting import (
    Processing,
    TimeSeriesDatasetGenerator,
)
from tsforecasting.models import (
    #BaseForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    FORECASTER_CLASSES,
)
from tsforecasting.evaluation.metrics import METRIC_STRATEGIES


# =============================================================================
# 1. TIME SERIES TRANSFORMATION
# =============================================================================

print("=" * 70)
print("1. TIME SERIES TRANSFORMATION")
print("=" * 70)

print("""
The Processing class transforms raw time series data into a supervised
learning format with lag features and horizon targets.

Raw data:       Date, y
                ↓
Transformed:    Date, y_lag_1, y_lag_2, ..., y_horizon_1, y_horizon_2, ...
""")


# -----------------------------------------------------------------------------
# 1.1 Generate Raw Data
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.1 Generate Raw Data")
print("-" * 70)

data = TimeSeriesDatasetGenerator.generate(
    n_samples=100,
    granularity="1mo",
    patterns=["trend", "seasonal"],
    random_state=42,
)

print(f"Raw data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(data.head())


# -----------------------------------------------------------------------------
# 1.2 Processing Class
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.2 Processing Class")
print("-" * 70)

processor = Processing()

# Configuration
LAGS = 10       # Number of lag features (window_size)
HORIZON = 5     # Number of future steps to predict
GRANULARITY = "1mo"

print("Configuration:")
print(f"  LAGS (window_size): {LAGS}")
print(f"  HORIZON: {HORIZON}")
print(f"  GRANULARITY: {GRANULARITY}")


# -----------------------------------------------------------------------------
# 1.3 make_timeseries() Transformation
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.3 make_timeseries() Transformation")
print("-" * 70)

timeseries = processor.make_timeseries(
    dataset=data,
    window_size=LAGS,           # Number of lags
    horizon=HORIZON,            # Forecast steps
    datetime_engineering=True,  # Add date features
)

print(f"Transformed shape: {timeseries.shape}")
print(f"Rows: {len(data)} → {len(timeseries)} (reduced due to windowing)")


# -----------------------------------------------------------------------------
# 1.4 Understanding Generated Columns
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("1.4 Understanding Generated Columns")
print("-" * 70)

columns = timeseries.columns.tolist()

# Categorize columns
lag_cols = [c for c in columns if c.startswith('y_lag_')]
horizon_cols = [c for c in columns if c.startswith('y_horizon_')]
date_cols = [c for c in columns if c.startswith('Date_')]
other_cols = [c for c in columns if c not in lag_cols + horizon_cols + date_cols and c != 'Date']

print(f"""
COLUMN STRUCTURE:
-----------------

LAG FEATURES ({len(lag_cols)} columns):
  {lag_cols}
  → Previous values: y_lag_1 = t-1, y_lag_2 = t-2, ..., y_lag_{LAGS} = t-{LAGS}

HORIZON TARGETS ({len(horizon_cols)} columns):
  {horizon_cols}
  → Future values: y_horizon_1 = t+1, y_horizon_2 = t+2, ..., y_horizon_{HORIZON} = t+{HORIZON}

DATETIME FEATURES ({len(date_cols)} columns):
  {date_cols[:5]}...
  → Extracted from Date column: month, day_of_week, year, etc.

OTHER ({len(other_cols)} columns):
  {other_cols}
""")

# Visual representation
print("Sample transformation:")
sample_cols = ['y_lag_3', 'y_lag_2', 'y_lag_1', 'y_horizon_1', 'y_horizon_2', 'y_horizon_3']
available = [c for c in sample_cols if c in timeseries.columns]
print(timeseries[available].head(5).to_string(index=False))


# =============================================================================
# 2. FEATURE AND TARGET SEPARATION
# =============================================================================

print("\n" + "=" * 70)
print("2. FEATURE AND TARGET SEPARATION")
print("=" * 70)


# -----------------------------------------------------------------------------
# 2.1 Define Input/Output Columns
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.1 Define Input/Output Columns")
print("-" * 70)

# Input features (X): everything except horizons and Date
input_cols = [c for c in timeseries.columns 
              if not c.startswith('y_horizon_') and c != 'Date']

# Target variables (y): horizon columns
target_cols = [c for c in timeseries.columns if c.startswith('y_horizon_')]

print(f"Input features: {len(input_cols)} columns")
print(f"  Includes: {len(lag_cols)} lags + {len(date_cols)} datetime features")

print(f"\nTarget variables: {len(target_cols)} columns")
print(f"  {target_cols}")


# -----------------------------------------------------------------------------
# 2.2 Handle NaN Values
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.2 Handle NaN Values")
print("-" * 70)

# The last row(s) have NaN in horizons (future forecasting row)
nan_count = timeseries[target_cols].isna().any(axis=1).sum()
print(f"Rows with NaN in targets: {nan_count}")

# Filter valid rows for training
valid_mask = ~timeseries[target_cols].isna().any(axis=1)
timeseries_clean = timeseries[valid_mask].copy()

print(f"Clean data: {len(timeseries_clean)} rows")


# -----------------------------------------------------------------------------
# 2.3 Create NumPy Arrays
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("2.3 Create NumPy Arrays")
print("-" * 70)

X = timeseries_clean[input_cols].values
y = timeseries_clean[target_cols].values

print(f"X shape: {X.shape} (samples × features)")
print(f"y shape: {y.shape} (samples × horizons)")


# =============================================================================
# 3. EXPANDING WINDOW EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("3. EXPANDING WINDOW EVALUATION")
print("=" * 70)

print("""
Expanding Window evaluation progressively increases the training set:

  Iteration 1: Train [0:80%], Test [80%:100%]
  Iteration 2: Train [0:80%+slide], Test [80%+slide:100%]
  ...
  
This simulates realistic forecasting where you train on all available
historical data and evaluate on future unseen data.
""")


# -----------------------------------------------------------------------------
# 3.1 Configure Window Parameters
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("3.1 Configure Window Parameters")
print("-" * 70)

TRAIN_SIZE = 0.80
SLIDING_SIZE = 10

total_samples = len(timeseries_clean)
initial_train = int(total_samples * TRAIN_SIZE)
test_size = total_samples - initial_train
iterations = test_size // SLIDING_SIZE

print("Window Configuration:")
print(f"  Total samples: {total_samples}")
print(f"  Initial train: {initial_train} ({TRAIN_SIZE*100:.0f}%)")
print(f"  Test pool: {test_size}")
print(f"  Sliding size: {SLIDING_SIZE}")
print(f"  Iterations: {iterations}")


# -----------------------------------------------------------------------------
# 3.2 Window Evolution
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("3.2 Window Evolution")
print("-" * 70)

print("Train/Test splits per iteration:")
for i in range(iterations):
    train_end = initial_train + (i * SLIDING_SIZE)
    test_start = train_end
    test_end = total_samples
    print(f"  Iter {i+1}: Train [0:{train_end}], Test [{test_start}:{test_end}] ({test_end - test_start} samples)")


# -----------------------------------------------------------------------------
# 3.3 Manual Expanding Window Implementation
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("3.3 Manual Expanding Window Implementation")
print("-" * 70)

def expanding_window_evaluate(X, y, model_class, model_params, 
                               train_size, sliding_size):
    """Manual expanding window evaluation."""
    total = len(X)
    initial = int(total * train_size)
    iterations = (total - initial) // sliding_size
    
    results = []
    
    for i in range(iterations):
        current_train = initial + (i * sliding_size)
        
        # Split
        X_train, X_test = X[:current_train], X[current_train:]
        y_train, y_test = y[:current_train], y[current_train:]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'window': i + 1,
            'train_size': current_train,
            'test_size': len(X_test),
            'mae': mae
        })
    
    return pd.DataFrame(results)

# Run evaluation
print("Running expanding window evaluation...")
results = expanding_window_evaluate(
    X, y,
    model_class=RandomForestForecaster,
    model_params={'n_estimators': 50, 'random_state': 42},
    train_size=TRAIN_SIZE,
    sliding_size=SLIDING_SIZE,
)

print(results.to_string(index=False))
print(f"\nMean MAE: {results['mae'].mean():.4f}")
print(f"Std MAE: {results['mae'].std():.4f}")


# =============================================================================
# 4. DIRECT MODEL USAGE
# =============================================================================

print("\n" + "=" * 70)
print("4. DIRECT MODEL USAGE")
print("=" * 70)

print("""
Models can be used directly without the pipeline for custom workflows.
All models inherit from BaseForecaster providing a unified interface.
""")


# -----------------------------------------------------------------------------
# 4.1 BaseForecaster Interface
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("4.1 BaseForecaster Interface")
print("-" * 70)

print("""
BaseForecaster provides:
  - fit(X, y) → Self        # Train model
  - predict(X) → np.ndarray # Generate predictions
  - get_params() → Dict     # Get hyperparameters
  - set_params(**params)    # Update parameters
  - is_fitted → bool        # Check fitted state
""")


# -----------------------------------------------------------------------------
# 4.2 Direct Instantiation
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("4.2 Direct Instantiation")
print("-" * 70)

# Prepare data
train_idx = int(len(X) * 0.8)
X_train, X_test = X[:train_idx], X[train_idx:]
y_train, y_test = y[:train_idx], y[train_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Direct instantiation
rf = RandomForestForecaster(n_estimators=50, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

print("RandomForestForecaster:")
print(f"  Params: {rf.get_params()}")
print(f"  is_fitted: {rf.is_fitted}")
print(f"  Predictions shape: {rf_pred.shape}")

xgb = XGBoostForecaster(n_estimators=50)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)

print("\nXGBoostForecaster:")
print(f"  Predictions shape: {xgb_pred.shape}")

# -----------------------------------------------------------------------------
# 4.3 FORECASTER_CLASSES Dictionary
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("4.4 FORECASTER_CLASSES Dictionary")
print("-" * 70)

print(f"Available classes: {list(FORECASTER_CLASSES.keys())}")

# Dynamic model selection
for name in ['RandomForest', 'GBR', 'XGBoost']:
    model_class = FORECASTER_CLASSES[name]
    model = model_class(n_estimators=30 if name != 'GBR' else 30)
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, pred)
    print(f"  {name}: MAE = {mae:.4f}")


# =============================================================================
# 5. METRICS AND EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("5. METRICS AND EVALUATION")
print("=" * 70)


# -----------------------------------------------------------------------------
# 5.1 Metric Strategies
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("5.1 Metric Strategies")
print("-" * 70)

print(f"Available metrics: {list(METRIC_STRATEGIES.keys())}")

# Use metric strategies
for name, strategy in METRIC_STRATEGIES.items():
    value = strategy.compute(y_test.flatten(), rf_pred.flatten())
    print(f"  {name} ({strategy.name}): {value:.4f}")


# -----------------------------------------------------------------------------
# 5.2 Per-Horizon Evaluation
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("5.2 Per-Horizon Evaluation")
print("-" * 70)

print("MAE per forecast horizon:")
for h in range(y_test.shape[1]):
    mae = mean_absolute_error(y_test[:, h], rf_pred[:, h])
    print(f"  Horizon {h+1}: {mae:.4f}")


# -----------------------------------------------------------------------------
# 5.3 Multi-Model Comparison
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("5.3 Multi-Model Comparison")
print("-" * 70)

predictions_dict = {
    'RandomForest': rf_pred,
    'XGBoost': xgb_pred,
}

print("Model comparison:")
print("-" * 40)
print(f"{'Model':<15} {'MAE':>10} {'MSE':>12} {'RMSE':>10}")
print("-" * 40)

for name, pred in predictions_dict.items():
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    print(f"{name:<15} {mae:>10.4f} {mse:>12.4f} {rmse:>10.4f}")


# =============================================================================
# 6. DATETIME FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 70)
print("6. DATETIME FEATURE ENGINEERING")
print("=" * 70)


# -----------------------------------------------------------------------------
# 6.1 engin_date() Method
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("6.1 engin_date() Method")
print("-" * 70)

# Create fresh processor
processor = Processing()

# Apply datetime engineering
data_with_features = processor.engin_date(data, drop=False)

print(f"Original columns: {list(data.columns)}")
print(f"After engin_date: {list(data_with_features.columns)}")

print("\nGenerated datetime features:")
date_features = [c for c in data_with_features.columns if c.startswith('Date_')]
for feat in date_features:
    sample_val = data_with_features[feat].iloc[0]
    print(f"  {feat}: {sample_val}")


# -----------------------------------------------------------------------------
# 6.2 future_timestamps() Method
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("6.2 future_timestamps() Method")
print("-" * 70)

# Generate future timestamps
last_date = data['Date'].iloc[-1]
future_dates = processor.future_timestamps(
    dataset=data,
    granularity="1mo",
    horizon=6,
)

print(f"Last date in data: {last_date}")
print(f"Future timestamps ({len(future_dates)}):")
for date in future_dates:
    print(f"  {date}")

# =============================================================================
# 7. QUICK REFERENCE
# =============================================================================

print("\n" + "=" * 70)
print("7. QUICK REFERENCE")
print("=" * 70)

print("""
PROCESSING:
-----------
from tsforecasting import Processing

processor = Processing()

# Transform to supervised format
timeseries = processor.make_timeseries(
    dataset=data,         # DataFrame with 'Date' and 'y' as default columns
    window_size=10,       # Number of lags
    horizon=5,            # Forecast steps
    datetime_engineering=True,
)

# DateTime features only
data_features = processor.engin_date(data, drop=False)

# Future timestamps
future = processor.future_timestamps(data, granularity="1mo", horizon=6)


DIRECT MODEL USAGE:
-------------------
from tsforecasting import ModelRegistry
from tsforecasting.models import RandomForestForecaster, FORECASTER_CLASSES

# Option 1: Direct instantiation
model = RandomForestForecaster(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Option 2: ModelRegistry
model = ModelRegistry.get("XGBoost", n_estimators=100)

# Option 3: FORECASTER_CLASSES
model_class = FORECASTER_CLASSES["RandomForest"]
model = model_class(n_estimators=100)


EXPANDING WINDOW:
-----------------
total = len(X)
initial_train = int(total * 0.8)
sliding_size = 10
iterations = (total - initial_train) // sliding_size

for i in range(iterations):
    train_end = initial_train + (i * sliding_size)
    X_train, X_test = X[:train_end], X[train_end:]
    y_train, y_test = y[:train_end], y[train_end:]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


COLUMN NAMING:
--------------
- y_lag_N    : Value at time t-N (input feature)
- y_horizon_N: Value at time t+N (target variable)
- Date_*     : Datetime-derived features
""")

print("=" * 70)
print("END OF EXAMPLE 3: PREPROCESSING & DECOMPOSED FUNCTIONALITIES")
print("=" * 70)