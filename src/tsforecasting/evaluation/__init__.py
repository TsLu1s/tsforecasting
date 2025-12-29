"""Evaluation module for TSForecasting package."""

from tsforecasting.evaluation.engine import EvaluationEngine
from tsforecasting.evaluation.intervals import (
    ConformalIntervalEstimator,
    GaussianIntervalEstimator,
    IntervalEstimator,
    IntervalEstimatorFactory,
    IntervalMethod,
    QuantileIntervalEstimator,
    compute_intervals,
)
from tsforecasting.evaluation.metrics import (
    MAEStrategy,
    MAPEStrategy,
    MaxErrorStrategy,
    METRIC_STRATEGIES,
    MetricStrategy,
    MSEStrategy,
    aggregate_performance,
    best_model,
    get_all_strategies,
    get_metric_strategy,
    metrics_regression,
    vertical_performance,
)

__all__ = [
    # Engine
    "EvaluationEngine",
    # Interval estimation
    "IntervalMethod",
    "IntervalEstimator",
    "QuantileIntervalEstimator",
    "ConformalIntervalEstimator",
    "GaussianIntervalEstimator",
    "IntervalEstimatorFactory",
    "compute_intervals",
    # Metric strategies
    "MetricStrategy",
    "MAEStrategy",
    "MAPEStrategy",
    "MSEStrategy",
    "MaxErrorStrategy",
    "METRIC_STRATEGIES",
    "get_metric_strategy",
    "get_all_strategies",
    # Utilities
    "metrics_regression",
    "vertical_performance",
    "best_model",
    "aggregate_performance",
]