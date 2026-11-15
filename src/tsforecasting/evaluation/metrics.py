from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from tsforecasting.core.enums import EvaluationMetric


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
    return list(METRIC_STRATEGIES.values())


# Short metric keys -> column names used in the performance tables.
METRIC_COLUMNS: Dict[str, str] = {
    "MAE": "Mean Absolute Error",
    "MAPE": "Mean Absolute Percentage Error",
    "MSE": "Mean Squared Error",
    "MAX_ERROR": "Max Error",
}

# How each metric column is summarised across windows.
_AGGREGATIONS: Dict[str, str] = {
    "Mean Absolute Error": "mean",
    "Mean Absolute Percentage Error": "mean",
    "Mean Squared Error": "mean",
    "Max Error": "max",
}


def resolve_metric_column(metric: Union[EvaluationMetric, str]) -> str:
    """Translate a metric identifier into its performance-table column name.

    Args:
        metric: Metric enum, short key ("MAE", "mape", ...) or full column name.

    Returns:
        Column name, e.g. "Mean Absolute Error".

    Raises:
        ValueError: If the metric is not recognised.
    """
    if isinstance(metric, EvaluationMetric):
        return metric.display_name

    key = str(metric).strip()
    normalised = key.upper().replace(" ", "_")
    if normalised in METRIC_COLUMNS:
        return METRIC_COLUMNS[normalised]
    for column in METRIC_COLUMNS.values():
        if column.lower() == key.lower():
            return column

    raise ValueError(f"Unknown metric: '{metric}'. Options: 'MAE', 'MAPE', 'MSE'")


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all regression metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric names to values. MAPE is expressed in percent and
        is not meaningful when ``y_true`` contains values at or near zero.
    """
    return {strategy.name: strategy.compute(y_true, y_pred) for strategy in get_all_strategies()}


def vertical_performance(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Evaluate performance across models, windows, and horizons.

    Args:
        forecasts: DataFrame with actual and predicted values.
        horizon: Number of forecast horizons.

    Returns:
        DataFrame with performance metrics per model/window/horizon.
    """
    if forecasts is None or forecasts.empty:
        raise ValueError("No forecasts available to evaluate.")

    horizons = list(range(1, horizon + 1))
    results = []

    for (model, window), group in forecasts.groupby(["Model", "Window"], sort=False):
        for h in horizons:
            y_true = group[f"y_horizon_{h}"].to_numpy()
            y_pred = group[f"y_forecast_horizon_{h}"].to_numpy()

            meta = {"Model": model, "Window": window, "Horizon": h}
            results.append({**meta, **metrics_regression(y_true, y_pred)})

    results_df = pd.DataFrame(results)
    return results_df.sort_values(by=["Model", "Window", "Horizon"])


def best_model(results: pd.DataFrame, metric: str = "MAE", verbose: bool = True) -> str:
    """Identify best performing model based on metric.

    Args:
        results: Performance results DataFrame.
        metric: Metric to use for comparison (MAE, MAPE, MSE).
        verbose: Whether to print the selected model.

    Returns:
        Name of best performing model.
    """
    _metric = resolve_metric_column(metric)

    aggregated = results.groupby(["Model", "Horizon"])[_metric].mean().reset_index()

    mean_by_model = aggregated.groupby("Model")[_metric].mean().reset_index()

    if mean_by_model[_metric].isna().all():
        raise ValueError(f"Cannot select a model: '{_metric}' is undefined for every model.")

    best = mean_by_model.loc[mean_by_model[_metric].idxmin()]

    if verbose:
        print(
            f"The model with the best performance was {best['Model']} "
            f"with an (mean) {_metric} of {round(best[_metric], 4)}"
        )

    return str(best["Model"])


def aggregate_performance(results: pd.DataFrame, metric: str = "MAE") -> Dict[str, pd.DataFrame]:
    """Aggregate performance results.

    Args:
        results: Performance results DataFrame (one row per model/window/horizon).
        metric: Metric used to rank the leaderboard (MAE, MAPE, MSE).

    Returns:
        Dictionary with:
        - "by_horizon": metrics per model and horizon, aggregated across windows.
        - "leaderboard": metrics per model, best model first.

        Error metrics are averaged; "Max Error" reports the largest error observed.
    """
    _metric = resolve_metric_column(metric)
    aggregations = {col: how for col, how in _AGGREGATIONS.items() if col in results.columns}

    by_horizon = results.groupby(["Model", "Horizon"]).agg(aggregations).reset_index()

    leaderboard = (
        results.groupby("Model")
        .agg(aggregations)
        .reset_index()
        .sort_values(by=_metric, ascending=True)
        .reset_index(drop=True)
    )

    return {"by_horizon": by_horizon, "leaderboard": leaderboard}
