"""Prediction interval estimation strategies for forecast uncertainty quantification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from statistics import NormalDist
from typing import List, Optional, Tuple

import numpy as np


class IntervalMethod(str, Enum):
    """Available interval estimation methods."""

    QUANTILE = "quantile"
    CONFORMAL = "conformal"
    GAUSSIAN = "gaussian"
    ENSEMBLE = "ensemble"


def _prepare_residual_inputs(
    y_true: np.ndarray, y_pred: np.ndarray, min_samples: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and align historical actuals and predictions.

    One-dimensional inputs are read as ``n_samples`` observations of a single
    horizon. Samples containing missing values are discarded.

    Returns:
        Tuple of float arrays with shape (n_samples, n_horizons).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}."
        )

    valid = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1))
    y_true, y_pred = y_true[valid], y_pred[valid]

    if y_true.shape[0] < min_samples:
        raise ValueError(
            f"At least {min_samples} historical sample(s) are required to estimate "
            f"prediction intervals, got {y_true.shape[0]}."
        )

    return y_true, y_pred


class IntervalEstimator(ABC):
    """Abstract base class for prediction interval estimation.

    Prediction intervals quantify forecast uncertainty by providing
    lower and upper bounds around point predictions.

    Intervals are calibrated on out-of-sample residuals, so the coverage that
    can be resolved depends on how many residuals are available: with ``n``
    samples per horizon, empirical bounds cannot go beyond the most extreme
    residual observed, which caps the attainable coverage near ``n / (n + 1)``.
    """

    def __init__(self, coverage: float = 0.90):
        """Initialize estimator.

        Args:
            coverage: Desired coverage level (0.0 to 1.0).
                      E.g., 0.90 means 90% of true values should fall within bounds.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"Coverage must be in (0, 1), got {coverage}")
        self.coverage = coverage
        self._is_fitted = False
        self._n_horizons: Optional[int] = None

    def _check_forecast(self, forecast: np.ndarray) -> np.ndarray:
        """Validate fitted state and forecast shape before computing bounds."""
        if not self._is_fitted:
            raise RuntimeError("Estimator not fitted. Call fit() first.")

        forecast = np.atleast_1d(np.asarray(forecast, dtype=float))
        n_horizons = self._n_horizons
        if n_horizons not in (None, 1) and forecast.shape[-1] != n_horizons:
            raise ValueError(
                f"Forecast has {forecast.shape[-1]} horizon(s) but the estimator was "
                f"fitted on {n_horizons}."
            )
        return forecast

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "IntervalEstimator":
        """Fit estimator using historical predictions and actuals.

        Args:
            y_true: Actual values, shape (n_samples, n_horizons).
            y_pred: Predicted values, shape (n_samples, n_horizons).

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals.

        Args:
            forecast: Point forecasts, shape (n_horizons,) or (n_samples, n_horizons).

        Returns:
            Tuple of (lower_bounds, upper_bounds) with same shape as forecast.
        """
        ...

    @property
    def method_name(self) -> str:
        """Return method identifier."""
        return self.__class__.__name__.replace("IntervalEstimator", "").lower()


class QuantileIntervalEstimator(IntervalEstimator):
    """Empirical quantile-based prediction intervals.

    Uses historical prediction errors to compute percentile-based bounds.
    Non-parametric approach that captures asymmetric error distributions.

    How it works:
        1. Compute residuals: r = y_true - y_pred for each horizon
        2. Find empirical quantiles at (1-coverage)/2 and (1+coverage)/2
        3. Apply: lower = forecast + q_low, upper = forecast + q_high

    Pros:
        - Captures asymmetric errors (e.g., consistent under/over-prediction)
        - No distributional assumptions
        - Simple to implement and interpret

    Cons:
        - Requires sufficient samples per horizon for stable quantiles
        - Fixed coverage (what you specify is approximately what you get)
        - Sensitive to outliers in small samples

    Example:
        estimator = QuantileIntervalEstimator(coverage=0.90)
        estimator.fit(y_true, y_pred)
        lower, upper = estimator.predict(forecast)
    """

    def __init__(self, coverage: float = 0.90):
        super().__init__(coverage)
        self._quantiles_low: Optional[np.ndarray] = None
        self._quantiles_high: Optional[np.ndarray] = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "QuantileIntervalEstimator":
        """Fit by computing empirical quantiles of residuals per horizon."""
        y_true, y_pred = _prepare_residual_inputs(y_true, y_pred)

        residuals = y_true - y_pred  # Shape: (n_samples, n_horizons)
        self._n_horizons = residuals.shape[1]

        alpha = 1 - self.coverage
        q_low = alpha / 2 * 100  # e.g., 5th percentile for 90% coverage
        q_high = (1 - alpha / 2) * 100  # e.g., 95th percentile

        self._quantiles_low = np.percentile(residuals, q_low, axis=0)
        self._quantiles_high = np.percentile(residuals, q_high, axis=0)
        self._is_fitted = True

        return self

    def predict(self, forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply quantile-based intervals to forecast."""
        forecast = self._check_forecast(forecast)
        lower = forecast + self._quantiles_low
        upper = forecast + self._quantiles_high

        return lower, upper


class ConformalIntervalEstimator(IntervalEstimator):
    """Conformal prediction intervals with coverage guarantees.

    Provides distribution-free prediction intervals with theoretical
    coverage guarantees under the assumption of exchangeable errors.

    How it works:
        1. Compute nonconformity scores: score = |y_true - y_pred| per horizon
        2. Find the (1-alpha) quantile of scores for desired coverage
        3. Apply: lower = forecast - q, upper = forecast + q

    Pros:
        - Theoretical coverage guarantee (if errors are exchangeable)
        - Distribution-free, minimal assumptions
        - Well-established statistical framework

    Cons:
        - Basic version produces symmetric, constant-width intervals
        - Requires calibration set (reduces effective training data)
        - May be conservative (wider than necessary)
        - With n calibration samples the guarantee only holds for coverage
          levels up to n / (n + 1); above that the widest observed error is used

    Note:
        This implements split conformal prediction. For adaptive-width
        intervals, consider Conformalized Quantile Regression (CQR).

    Example:
        estimator = ConformalIntervalEstimator(coverage=0.90)
        estimator.fit(y_true, y_pred)
        lower, upper = estimator.predict(forecast)
    """

    def __init__(self, coverage: float = 0.90):
        super().__init__(coverage)
        self._conformity_quantiles: Optional[np.ndarray] = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalIntervalEstimator":
        """Fit by computing conformity score quantiles per horizon."""
        y_true, y_pred = _prepare_residual_inputs(y_true, y_pred)

        # Nonconformity scores (absolute errors)
        scores = np.abs(y_true - y_pred)  # Shape: (n_samples, n_horizons)
        self._n_horizons = scores.shape[1]

        # Conformal quantile with finite-sample correction: the k-th smallest
        # score with k = ceil((n + 1) * coverage). It is taken as an observed
        # order statistic (no interpolation), which is what the coverage
        # guarantee relies on. When k exceeds n the guarantee would require an
        # unbounded interval; the largest observed score is used instead.
        n = scores.shape[0]
        rank = int(np.ceil((n + 1) * self.coverage - 1e-9))
        rank = min(max(rank, 1), n)

        self._conformity_quantiles = np.sort(scores, axis=0)[rank - 1]
        self._is_fitted = True

        return self

    def predict(self, forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply conformal intervals to forecast."""
        forecast = self._check_forecast(forecast)
        lower = forecast - self._conformity_quantiles
        upper = forecast + self._conformity_quantiles

        return lower, upper


class GaussianIntervalEstimator(IntervalEstimator):
    """Parametric Gaussian prediction intervals.

    Assumes residuals follow a normal distribution and uses
    mean ± z * std for interval construction.

    How it works:
        1. Compute residuals: r = y_true - y_pred per horizon
        2. Estimate mean (bias) and std of residuals
        3. Find z-score for desired coverage from normal distribution
        4. Apply: lower = forecast + mean - z*std, upper = forecast + mean + z*std

    Pros:
        - Computationally efficient
        - Works well when errors are approximately normal
        - Accounts for bias (systematic over/under-prediction)

    Cons:
        - Assumes symmetric, normal errors (often violated)
        - Sensitive to outliers affecting std estimation
        - May undercover if true distribution has heavy tails

    Example:
        estimator = GaussianIntervalEstimator(coverage=0.90)
        estimator.fit(y_true, y_pred)
        lower, upper = estimator.predict(forecast)
    """

    def __init__(self, coverage: float = 0.90):
        super().__init__(coverage)
        self._bias: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._z_score: Optional[float] = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "GaussianIntervalEstimator":
        """Fit by estimating residual mean and std per horizon."""
        y_true, y_pred = _prepare_residual_inputs(y_true, y_pred)

        residuals = y_true - y_pred
        self._n_horizons = residuals.shape[1]

        self._bias = np.mean(residuals, axis=0)
        # A single residual carries no spread information: fall back to zero width.
        ddof = 1 if residuals.shape[0] > 1 else 0
        self._std = np.std(residuals, axis=0, ddof=ddof)

        # Z-score for desired coverage (two-tailed)
        alpha = 1 - self.coverage
        self._z_score = NormalDist().inv_cdf(1 - alpha / 2)
        self._is_fitted = True

        return self

    def predict(self, forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian intervals to forecast."""
        forecast = self._check_forecast(forecast)
        margin = self._z_score * self._std

        lower = forecast + self._bias - margin
        upper = forecast + self._bias + margin

        return lower, upper


class EnsembleIntervalEstimator(IntervalEstimator):
    """Ensemble prediction intervals averaging quantile, conformal, and gaussian methods.

    Combines all three interval estimation approaches and averages the bounds,
    providing more robust uncertainty estimates.

    How it works:
        1. Fit quantile, conformal, and gaussian estimators
        2. Generate intervals from each method
        3. Average lower bounds and upper bounds across methods

    Pros:
        - More robust than any single method
        - Balances asymmetry (quantile) with guarantees (conformal) and efficiency (gaussian)
        - Reduces method-specific biases

    Cons:
        - Computationally heavier (3x fitting/prediction)
        - May dilute strengths of individual methods

    Example:
        estimator = EnsembleIntervalEstimator(coverage=0.90)
        estimator.fit(y_true, y_pred)
        lower, upper = estimator.predict(forecast)
    """

    def __init__(self, coverage: float = 0.90):
        super().__init__(coverage)
        self._estimators: List[IntervalEstimator] = [
            QuantileIntervalEstimator(coverage),
            ConformalIntervalEstimator(coverage),
            GaussianIntervalEstimator(coverage),
        ]

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "EnsembleIntervalEstimator":
        """Fit all underlying estimators."""
        for estimator in self._estimators:
            estimator.fit(y_true, y_pred)
        self._n_horizons = self._estimators[0]._n_horizons
        self._is_fitted = True
        return self

    def predict(self, forecast: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Average intervals from all estimators."""
        forecast = self._check_forecast(forecast)

        lowers, uppers = [], []
        for estimator in self._estimators:
            lower, upper = estimator.predict(forecast)
            lowers.append(lower)
            uppers.append(upper)

        avg_lower = np.mean(lowers, axis=0)
        avg_upper = np.mean(uppers, axis=0)

        return avg_lower, avg_upper


class IntervalEstimatorFactory:
    """Factory for creating interval estimators."""

    _estimators = {
        IntervalMethod.QUANTILE: QuantileIntervalEstimator,
        IntervalMethod.CONFORMAL: ConformalIntervalEstimator,
        IntervalMethod.GAUSSIAN: GaussianIntervalEstimator,
        IntervalMethod.ENSEMBLE: EnsembleIntervalEstimator,
    }

    @classmethod
    def create(
        cls,
        method: IntervalMethod | str = IntervalMethod.QUANTILE,
        coverage: float = 0.90,
    ) -> IntervalEstimator:
        """Create an interval estimator.

        Args:
            method: Estimation method ("quantile", "conformal", "gaussian", "ensemble").
            coverage: Desired coverage level (0.0 to 1.0).

        Returns:
            Configured IntervalEstimator instance.

        Raises:
            ValueError: If the method is not recognised.
        """
        if not isinstance(method, IntervalMethod):
            try:
                method = IntervalMethod(str(method).lower().strip())
            except ValueError:
                raise ValueError(
                    f"Unknown interval method: '{method}'. Available: {cls.list_methods()}"
                ) from None

        return cls._estimators[method](coverage=coverage)

    @classmethod
    def list_methods(cls) -> List[str]:
        """List available interval estimation methods."""
        return [m.value for m in IntervalMethod]


def compute_intervals(
    forecast: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "quantile",
    coverage: float = 0.90,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute prediction intervals in one call.

    Args:
        forecast: Point forecasts for which to compute intervals.
        y_true: Historical actual values.
        y_pred: Historical predictions.
        method: Interval method ("quantile", "conformal", "gaussian", "ensemble").
        coverage: Desired coverage level.

    Returns:
        Tuple of (lower_bounds, upper_bounds).
    """
    estimator = IntervalEstimatorFactory.create(method, coverage)
    estimator.fit(y_true, y_pred)
    return estimator.predict(forecast)
