"""
Example 4: Synthetic Time Series Data Generation
================================================

This example demonstrates the TimeSeriesDatasetGenerator for creating
synthetic time series datasets with various patterns and configurations.

Topics covered:
- Available granularities and patterns
- generate() method with all parameters
- Quick generators (monthly, daily, hourly)
- Example datasets with random_generation
- Benchmark datasets for evaluation
"""

import warnings
warnings.filterwarnings("ignore", category=Warning)

from tsforecasting import TimeSeriesDatasetGenerator


# =============================================================================
# 1. AVAILABLE OPTIONS
# =============================================================================

print("=" * 70)
print("1. AVAILABLE OPTIONS")
print("=" * 70)

print("\nGranularities:")
for g in TimeSeriesDatasetGenerator.list_available_granularities():
    print(f"  {g}")

print("\nPatterns:")
for p in TimeSeriesDatasetGenerator.list_available_patterns():
    print(f"  {p}")

print("\nExample Datasets:")
for d in TimeSeriesDatasetGenerator.list_available_datasets():
    print(f"  {d}")


# =============================================================================
# 2. GENERATE METHOD
# =============================================================================

print("\n" + "=" * 70)
print("2. GENERATE METHOD")
print("=" * 70)

# Full configuration
data = TimeSeriesDatasetGenerator.generate(
    n_samples=300,              # Number of time steps
    granularity="1mo",          # Options: "1m", "30m", "1h", "1d", "1wk", "1mo"
    patterns=["trend", "seasonal"],  # Options: "trend", "seasonal", "mixed", "random_walk", "multi_seasonal", "regime_change"
    trend_strength=0.3,         # Range: 0.0 to 1.0
    trend_type="linear",        # Options: "linear", "exponential"
    seasonality_period=12,      # Period in time steps
    seasonality_strength=1.5,   # Amplitude multiplier
    noise_level=0.1,            # Range: 0.0 to 1.0
    start_date="2000-01-01",    # Start date
    base_value=100.0,           # Base value
    random_state=42,            # Seed for reproducibility
)

print(f"\nGenerated: {data.shape}")
print(data.head(10))


# =============================================================================
# 3. PATTERN EXAMPLES
# =============================================================================

print("\n" + "=" * 70)
print("3. PATTERN EXAMPLES")
print("=" * 70)

# Trend only
trend_data = TimeSeriesDatasetGenerator.generate(
    n_samples=100,
    patterns=["trend"],
    trend_strength=0.5,
    trend_type="exponential",
)
print(f"\nTrend only: {trend_data.shape}")

# Seasonal only
seasonal_data = TimeSeriesDatasetGenerator.generate(
    n_samples=100,
    patterns=["seasonal"],
    seasonality_period=12,
    seasonality_strength=2.0,
)
print(f"Seasonal only: {seasonal_data.shape}")

# Mixed (trend + seasonal + noise)
mixed_data = TimeSeriesDatasetGenerator.generate(
    n_samples=100,
    patterns=["mixed"],
)
print(f"Mixed: {mixed_data.shape}")

# Random walk
rw_data = TimeSeriesDatasetGenerator.generate(
    n_samples=100,
    patterns=["random_walk"],
    noise_level=0.05,
)
print(f"Random walk: {rw_data.shape}")

# Multi-seasonal
multi_data = TimeSeriesDatasetGenerator.generate(
    n_samples=100,
    granularity="1h",
    patterns=["multi_seasonal"],
    seasonality_period=24,
    secondary_period=168,
)
print(f"Multi-seasonal: {multi_data.shape}")

# Regime change
regime_data = TimeSeriesDatasetGenerator.generate(
    n_samples=300,
    patterns=["trend", "regime_change"],
    regime_change_points=[100, 200],
)
print(f"Regime change: {regime_data.shape}")


# =============================================================================
# 4. QUICK GENERATORS
# =============================================================================

print("\n" + "=" * 70)
print("4. QUICK GENERATORS")
print("=" * 70)

# Monthly (default: seasonal, period=12)
monthly = TimeSeriesDatasetGenerator.quick_monthly(n_samples=120)
print(f"\nquick_monthly: {monthly.shape}")

# Daily (default: mixed, period=7)
daily = TimeSeriesDatasetGenerator.quick_daily(n_samples=365)
print(f"quick_daily: {daily.shape}")

# Hourly (default: multi_seasonal, period=24, secondary=168)
hourly = TimeSeriesDatasetGenerator.quick_hourly(n_samples=720)
print(f"quick_hourly: {hourly.shape}")

# =============================================================================
# 5. BENCHMARK DATASETS
# =============================================================================

print("\n" + "=" * 70)
print("5. BENCHMARK DATASETS")
print("=" * 70)

benchmarks = TimeSeriesDatasetGenerator.benchmark_datasets()

print("\nBenchmark datasets for model evaluation:")
for name, df in benchmarks.items():
    print(f"  {name}: {df.shape}")


# =============================================================================
# 6. QUICK REFERENCE
# =============================================================================

print("\n" + "=" * 70)
print("7. QUICK REFERENCE")
print("=" * 70)

print("""
GENERATE:
---------
data = TimeSeriesDatasetGenerator.generate(
    n_samples=300,
    granularity="1mo",      # "1m", "30m", "1h", "1d", "1wk", "1mo"
    patterns=["trend", "seasonal"],
    random_state=42,
)


QUICK GENERATORS:
-----------------
data = TimeSeriesDatasetGenerator.quick_monthly(n_samples=120)
data = TimeSeriesDatasetGenerator.quick_daily(n_samples=365)
data = TimeSeriesDatasetGenerator.quick_hourly(n_samples=720)

PATTERNS:
---------
"trend"          - Linear or exponential trend
"seasonal"       - Single sinusoidal seasonality
"mixed"          - Trend + Seasonal + Noise
"random_walk"    - Non-stationary cumulative steps
"multi_seasonal" - Two seasonality periods
"regime_change"  - Structural breaks
""")

print("=" * 70)
print("END OF EXAMPLE 4: SYNTHETIC DATA GENERATION")
print("=" * 70)