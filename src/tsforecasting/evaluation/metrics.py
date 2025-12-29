from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from tsforecasting.core.enums import EvaluationMetric


# =============================================================================
# Metric Strategy Classes
# =============================================================================

class MetricStrategy(ABC):
    """Abstract base for metric calculation strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name."""
        ...
    
    @property
    @abstractmethod
    def key(self) -> str:
        """Short key for the metric."""
        ...
    
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute metric value.
        
        Args:
            y_true: True target values.
            y_pred: Predicted values.
            
        Returns:
            Computed metric value.
        """
        ...


class MAEStrategy(MetricStrategy):
    """Mean Absolute Error strategy."""
    
    @property
    def name(self) -> str:
        return "Mean Absolute Error"
    
    @property
    def key(self) -> str:
        return "mae"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_error(y_true, y_pred))


class MAPEStrategy(MetricStrategy):
    """Mean Absolute Percentage Error strategy."""
    
    @property
    def name(self) -> str:
        return "Mean Absolute Percentage Error"
    
    @property
    def key(self) -> str:
        return "mape"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_percentage_error(y_true, y_pred)) * 100


class MSEStrategy(MetricStrategy):
    """Mean Squared Error strategy."""
    
    @property
    def name(self) -> str:
        return "Mean Squared Error"
    
    @property
    def key(self) -> str:
        return "mse"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_squared_error(y_true, y_pred))


class MaxErrorStrategy(MetricStrategy):
    """Maximum Error strategy."""
    
    @property
    def name(self) -> str:
        return "Max Error"
    
    @property
    def key(self) -> str:
        return "max_error"
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(max_error(y_true, y_pred))


# Strategy registry
METRIC_STRATEGIES: Dict[str, MetricStrategy] = {
    "mae": MAEStrategy(),
    "mape": MAPEStrategy(),
    "mse": MSEStrategy(),
    "max_error": MaxErrorStrategy(),
}


def get_metric_strategy(metric: EvaluationMetric | str) -> MetricStrategy:
    """Get metric strategy by enum or string.
    
    Args:
        metric: Metric enum or string key.
        
    Returns:
        MetricStrategy instance.
    """
    if isinstance(metric, EvaluationMetric):
        key = metric.value
    else:
        key = metric.lower()
    return METRIC_STRATEGIES[key]


def get_all_strategies() -> List[MetricStrategy]:
    """Get all metric strategies."""
    return [MAEStrategy(), MAPEStrategy(), MSEStrategy(), MaxErrorStrategy()]


# =============================================================================
# Utility Functions
# =============================================================================

def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all regression metrics.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary of metric names to values.
    """
    return {
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Mean Absolute Percentage Error": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "Mean Squared Error": mean_squared_error(y_true, y_pred),
        "Max Error": max_error(y_true, y_pred),
    }


def vertical_performance(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Evaluate performance across models, windows, and horizons.
    
    Args:
        forecasts: DataFrame with actual and predicted values.
        horizon: Number of forecast horizons.
        
    Returns:
        DataFrame with performance metrics per model/window/horizon.
    """
    df = forecasts.copy()
    horizons = list(range(1, horizon + 1))
    results = []
    
    for model in df["Model"].unique():
        for window in df["Window"].unique():
            model_window_data = df[(df["Model"] == model) & (df["Window"] == window)]
            
            for h in horizons:
                y_true = model_window_data[f"y_horizon_{h}"]
                y_pred = model_window_data[f"y_forecast_horizon_{h}"]
                
                meta = {"Model": model, "Window": window, "Horizon": h}
                metrics = metrics_regression(y_true.values, y_pred.values)
                results.append({**meta, **metrics})
    
    results_df = pd.DataFrame(results)
    return results_df.sort_values(by=["Model", "Window", "Horizon"])


def best_model(results: pd.DataFrame, metric: str = "MAE") -> str:
    """Identify best performing model based on metric.
    
    Args:
        results: Performance results DataFrame.
        metric: Metric to use for comparison (MAE, MAPE, MSE).
        
    Returns:
        Name of best performing model.
    """
    metric_mapping = {
        "MAE": "Mean Absolute Error",
        "MAPE": "Mean Absolute Percentage Error",
        "MSE": "Mean Squared Error",
    }
    _metric = metric_mapping.get(metric.upper(), metric)
    
    # Aggregate by model and horizon
    aggregated = results.groupby(["Model", "Horizon"])[_metric].mean().reset_index()
    
    # Get mean across all horizons per model
    mean_by_model = aggregated.groupby("Model")[_metric].mean().reset_index()
    
    # Find best model
    best = mean_by_model.loc[mean_by_model[_metric].idxmin()]
    
    print(
        f"The model with the best performance was {best['Model']} "
        f"with an (mean) {_metric} of {round(best[_metric], 4)}"
    )
    
    return best["Model"]


def aggregate_performance(
    results: pd.DataFrame,
    metric: str = "MAE"
) -> Dict[str, pd.DataFrame]:
    """Aggregate performance results."""
    metric_mapping = {
        "MAE": "Mean Absolute Error",
        "MAPE": "Mean Absolute Percentage Error",
        "MSE": "Mean Squared Error",
    }
    _metric = metric_mapping.get(metric.upper(), metric)
    
    by_horizon = results.groupby(["Model", "Horizon"]).agg({
        "Mean Absolute Error": "mean",
        "Mean Absolute Percentage Error": "mean",
        "Mean Squared Error": "mean",
        "Max Error": "max",
    }).reset_index()
    
    leaderboard = (
        results.groupby("Model")
        .mean(numeric_only=True)
        .reset_index()
        .drop(columns=["Window", "Horizon"], errors="ignore")
        .sort_values(by=_metric, ascending=True)
    )
    
    return {"by_horizon": by_horizon, "leaderboard": leaderboard}