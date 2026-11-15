from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from tsforecasting.core.base import (
    BasePipeline,
    BuilderPipeline,
    TSForecastingBuilder,
    format_validation_error,
    resolve_model_names,
)
from tsforecasting.core.exceptions import (
    ConfigurationError,
    InsufficientDataError,
    NotFittedError,
)
from tsforecasting.core.schemas import (
    ForecastConfig,
    PerformanceHistory,
    PipelineConfig,
    PreprocessingConfig,
)
from tsforecasting.evaluation.engine import EvaluationEngine, get_scaler
from tsforecasting.evaluation.intervals import IntervalEstimatorFactory
from tsforecasting.evaluation.metrics import (
    aggregate_performance,
    best_model,
    vertical_performance,
)
from tsforecasting.models.base import BaseForecaster
from tsforecasting.models.registry import ModelRegistry, model_configurations
from tsforecasting.processing.processor import Processing


class TSForecasting(BasePipeline, BuilderPipeline):
    """Automated Time Series Forecasting Pipeline.

    Implements multivariate multi-horizon forecasting using an Expanding Window
    evaluation approach with multiple regression models.

    The pipeline supports:
    - Multiple forecasting models (RandomForest, XGBoost, etc.)
    - Configurable lag features and forecast horizons
    - Expanding window cross-validation, free of look-ahead leakage
    - Automatic model selection based on performance metrics
    - Prediction interval estimation

    The input dataset needs a datetime ``Date`` column and a numeric ``y``
    column, with one row per timestamp at regular spacing. Rows are sorted
    chronologically when needed; missing values and duplicated timestamps are
    rejected with a ``DatasetError``. Additional columns are ignored.

    Example:
        # Basic usage
        tsf = TSForecasting(
            train_size=0.8,
            lags=10,
            horizon=5,
            models=['RandomForest', 'XGBoost']
        )
        tsf.fit_forecast(data)
        forecast = tsf.forecast()

        # Using builder methods
        tsf = (TSForecasting(train_size=0.8, lags=10, horizon=5)
               .with_models(['RandomForest', 'XGBoost'])
               .with_preprocessing(scaler='robust'))
        tsf.fit_forecast(data)

        # Using standalone builder
        tsf = (TSForecastingBuilder()
               .with_train_size(0.8)
               .with_lags(10)
               .with_horizon(5)
               .with_models(['RandomForest', 'XGBoost'])
               .build())

    Attributes:
        train_size: Proportion of data for training (0.3 up to, excluding, 1.0).
        lags: Number of lag features (window size).
        horizon: Number of future steps to forecast.
        sliding_size: Window slide size for expanding evaluation.
        models: List of model names to evaluate.
        hparameters: Model hyperparameter configurations.
        granularity: Time granularity of data.
        metric: Evaluation metric for model selection.
        selected_model: Best performing model (after fitting).
        failed_models: Models skipped because they raised during evaluation.
    """

    # Built-in models for reference (see ModelRegistry.list_available() for all)
    AVAILABLE_MODELS = [
        "RandomForest",
        "ExtraTrees",
        "GBR",
        "KNN",
        "GeneralizedLR",
        "XGBoost",
        "Catboost",
        "AutoGluon",
    ]

    _COVERAGE_LEVELS = (0.80, 0.90, 0.95, 0.99)

    def __init__(
        self,
        train_size: float = 0.8,
        lags: int = 7,
        horizon: int = 15,
        sliding_size: int = 15,
        models: Optional[List[str]] = None,
        hparameters: Optional[Dict[str, Dict[str, Any]]] = None,
        granularity: str = "1d",
        metric: str = "MAE",
        scaler: str = "standard",
        datetime_features: bool = True,
        verbose: bool = True,
    ):
        """Initialize TSForecasting pipeline.

        Args:
            train_size: Training data proportion (0.3 up to, excluding, 1.0).
            lags: Number of lag features.
            horizon: Forecast horizon (steps ahead).
            sliding_size: Sliding window size for evaluation.
            models: List of model names to evaluate (case-insensitive). Any
                model registered in ``ModelRegistry`` can be used.
            hparameters: Custom hyperparameters dict.
            granularity: Time granularity (1m, 30m, 1h, 1d, 1wk, 1mo).
            metric: Evaluation metric (MAE, MAPE, MSE).
            scaler: Feature scaler (standard, minmax, robust).
            datetime_features: Whether to engineer datetime features.
            verbose: Whether to print evaluation progress.

        Raises:
            ConfigurationError: If a parameter is out of range or not recognised.
            ModelNotFoundError: If a model name is not registered.
        """
        super().__init__()

        if models is None:
            models = ["RandomForest", "GBR", "XGBoost"]

        if hparameters is not None and not isinstance(hparameters, dict):
            raise ConfigurationError(
                "hparameters must be a dictionary mapping model names to parameter dictionaries."
            )

        try:
            self._config = PipelineConfig(
                forecast=ForecastConfig(
                    train_size=train_size,
                    lags=lags,
                    horizon=horizon,
                    sliding_size=sliding_size,
                    granularity=granularity,
                    metric=metric,
                ),
                preprocessing=PreprocessingConfig(
                    scaler=scaler,
                    datetime_features=datetime_features,
                ),
                models=resolve_model_names(models),
            )
        except PydanticValidationError as exc:
            raise ConfigurationError(format_validation_error(exc)) from exc

        self._hparameters = hparameters if hparameters is not None else model_configurations()
        self._verbose = bool(verbose)

        self._processor = Processing()
        self._evaluation_engine: Optional[EvaluationEngine] = None
        self._model_instances: Dict[str, BaseForecaster] = {}
        self._selected_model: Optional[str] = None
        self._fit_predictions: Optional[pd.DataFrame] = None
        self._complete_performance: Optional[pd.DataFrame] = None
        self._timeseries: Optional[pd.DataFrame] = None
        self._scaler = None
        self._failed_models: Dict[str, str] = {}
        self._final_forecast: Optional[Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]] = None

    @classmethod
    def builder(cls) -> TSForecastingBuilder:
        """Get a builder instance for advanced configuration.

        Returns:
            TSForecastingBuilder instance.
        """
        return TSForecastingBuilder()

    @property
    def train_size(self) -> float:
        return self._config.forecast.train_size

    @property
    def lags(self) -> int:
        return self._config.forecast.lags

    @property
    def horizon(self) -> int:
        return self._config.forecast.horizon

    @property
    def sliding_size(self) -> int:
        return self._config.forecast.sliding_size

    @property
    def granularity(self) -> str:
        return self._config.forecast.granularity.value

    @property
    def metric(self) -> str:
        return self._config.forecast.metric.value.upper()

    @property
    def selected_model(self) -> Optional[str]:
        return self._selected_model

    @property
    def models(self) -> List[str]:
        return self._config.selected_models

    @property
    def failed_models(self) -> Dict[str, str]:
        """Models skipped during the last fit, mapped to the error they raised."""
        return dict(self._failed_models)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def verbose(self) -> bool:
        return self._verbose

    def _on_config_change(self) -> None:
        """Invalidate fitted state: a changed configuration requires a new fit."""
        if self._is_fitted:
            self._is_fitted = False
            self._selected_model = None
            self._final_forecast = None

    def _params_for(self, name: str) -> Dict[str, Any]:
        """Hyperparameters supplied for a model (name matched case-insensitively)."""
        hparameters = self._hparameters or {}
        if name in hparameters:
            return hparameters[name] or {}
        for key, params in hparameters.items():
            if str(key).lower() == name.lower():
                return params or {}
        return {}

    def _preprocess(self, dataset: pd.DataFrame) -> None:
        """Preprocess dataset and initialize models."""
        self._dataset = dataset
        self._failed_models = {}
        self._final_forecast = None
        self._selected_model = None

        self._evaluation_engine = EvaluationEngine(
            config=self._config.forecast,
            preprocessing=self._config.preprocessing,
            verbose=self._verbose,
        )

        self._model_instances = {}
        for name in self._config.selected_models:
            self._model_instances[name] = ModelRegistry.get(name, self._params_for(name))

    def _run_evaluation(self) -> None:
        """Run expanding window evaluation for all models."""
        self._fit_predictions = self._evaluation_engine.evaluate(
            dataset=self._dataset,
            models=self._model_instances,
        )
        self._failed_models = self._evaluation_engine.failed_models

        self._complete_performance = vertical_performance(
            self._fit_predictions, self.horizon
        ).reset_index(drop=True)

    def _select_best_model(self) -> None:
        """Select best model based on metric."""
        self._selected_model = best_model(
            self._complete_performance, metric=self.metric, verbose=self._verbose
        )

    def history(self) -> PerformanceHistory:
        """Get historical performance data.

        Returns:
            PerformanceHistory dataclass containing:
            - predictions: Raw forecast DataFrame
            - performance_complete: Detailed metrics per window/horizon
            - performance_by_horizon: Aggregated by horizon
            - leaderboard: Model rankings
            - selected_model: Best performing model name

            The DataFrames are copies: editing them does not affect the pipeline.

        Raises:
            NotFittedError: If pipeline not fitted.
        """
        if not self._is_fitted:
            raise NotFittedError("Pipeline not fitted. Call fit_forecast first.")

        aggregated = aggregate_performance(
            self._complete_performance,
            metric=self.metric,
        )

        return PerformanceHistory(
            predictions=self._fit_predictions.copy(),
            performance_complete=self._complete_performance.copy(),
            performance_by_horizon=aggregated["by_horizon"],
            leaderboard=aggregated["leaderboard"],
            selected_model=self._selected_model,
        )

    def forecast(
        self,
        dataset: Optional[pd.DataFrame] = None,
        interval_method: str = "ensemble",
    ) -> pd.DataFrame:
        """Generate future forecasts using best model.

        The selected model is refitted on every row with complete targets and
        used to predict the ``horizon`` steps after the last timestamp. The
        fit is cached, so calling ``forecast`` again on the same data (e.g. to
        compare interval methods) does not retrain the model.

        Args:
            dataset: Input data (uses fit data if not provided).
            interval_method: Method for prediction intervals.
                - "ensemble": Average of all methods (default, most robust)
                - "quantile": Empirical percentiles (captures asymmetry)
                - "conformal": Coverage guarantee (symmetric)
                - "gaussian": Parametric mean ± z*std

        Returns:
            DataFrame with columns:
            - Date: Future timestamps
            - y: Point forecasts
            - y_lower_80, y_upper_80: 80% prediction interval
            - y_lower_90, y_upper_90: 90% prediction interval
            - y_lower_95, y_upper_95: 95% prediction interval
            - y_lower_99, y_upper_99: 99% prediction interval

        Raises:
            NotFittedError: If pipeline not fitted.
            ValueError: If ``interval_method`` is not recognised.
        """
        if not self._is_fitted:
            raise NotFittedError("Pipeline not fitted. Call fit_forecast first.")

        # Fail fast on an unknown interval method, before any model is trained
        IntervalEstimatorFactory.create(interval_method, self._COVERAGE_LEVELS[0])

        if dataset is None:
            X = self._dataset
        else:
            X = self._prepare_dataset(dataset, announce=False)

        forecast_df, point_forecast = self._point_forecast(X)

        for coverage in self._COVERAGE_LEVELS:
            lower, upper = self._compute_intervals(
                forecast=point_forecast,
                method=interval_method,
                coverage=coverage,
            )
            coverage_pct = int(round(coverage * 100))
            forecast_df[f"y_lower_{coverage_pct}"] = lower
            forecast_df[f"y_upper_{coverage_pct}"] = upper

        return forecast_df

    def _point_forecast(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Fit the selected model on all available data and predict the future.

        Args:
            X: Prepared dataset (validated and sorted).

        Returns:
            Tuple of (DataFrame with future ``Date`` and ``y``, point forecast array).
        """
        if self._final_forecast is not None:
            cached_data, cached_forecast, cached_frame = self._final_forecast
            if cached_data is X or cached_data.equals(X):
                return cached_frame.copy(), cached_forecast.copy()

        forecast_df = self._processor.future_timestamps(X, self.horizon, self.granularity)

        timeseries = self._processor.make_timeseries(
            dataset=X,
            window_size=self.lags,
            horizon=self.horizon,
            datetime_engineering=self._config.preprocessing.datetime_features,
        )
        self._timeseries = timeseries.copy()

        # Split for final prediction: every row with complete targets vs. the last origin
        train = timeseries.iloc[: -self.horizon]
        test = timeseries.tail(1)

        if train.empty:
            raise InsufficientDataError(
                f"Insufficient data to forecast: {len(X)} observations cannot provide a "
                f"training sample for lags={self.lags} and horizon={self.horizon}."
            )

        input_cols = self._evaluation_engine.input_cols
        target_cols = self._evaluation_engine.target_cols

        self._scaler = self._get_scaler()
        self._scaler.fit(train[input_cols])
        train_scaled = pd.DataFrame(
            self._scaler.transform(train[input_cols]), columns=input_cols, index=train.index
        )
        test_scaled = pd.DataFrame(
            self._scaler.transform(test[input_cols]), columns=input_cols, index=test.index
        )
        train_scaled[target_cols] = train[target_cols]

        best_model_instance = self._model_instances[self._selected_model]

        predictions = self._evaluation_engine.single_fit_predict(
            train=train_scaled,
            test=test_scaled,
            model=best_model_instance,
            model_name=self._selected_model,
        )

        point_forecast = predictions.to_numpy().flatten()[: self.horizon]
        forecast_df["y"] = point_forecast

        self._final_forecast = (X, point_forecast.copy(), forecast_df.copy())
        return forecast_df, point_forecast

    def _compute_intervals(
        self,
        forecast: np.ndarray,
        method: str = "ensemble",
        coverage: float = 0.90,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals using historical residuals.

        Args:
            forecast: Point forecasts array.
            method: Interval method ("ensemble", "quantile", "conformal", "gaussian").
            coverage: Desired coverage level.

        Returns:
            Tuple of (lower_bounds, upper_bounds).
        """

        y_true, y_pred = self._get_historical_residuals()

        estimator = IntervalEstimatorFactory.create(method, coverage)
        estimator.fit(y_true, y_pred)

        return estimator.predict(forecast)

    def _get_historical_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract historical y_true and y_pred from fit predictions.

        Returns:
            Tuple of (y_true, y_pred) arrays, shape (n_samples, n_horizons).
        """
        df = self._fit_predictions

        model_df = df[df["Model"] == self._selected_model]

        actual_cols = [f"y_horizon_{i+1}" for i in range(self.horizon)]
        pred_cols = [f"y_forecast_horizon_{i+1}" for i in range(self.horizon)]

        y_true = model_df[actual_cols].values
        y_pred = model_df[pred_cols].values

        return y_true, y_pred

    def _get_scaler(self):
        """Get scaler based on configuration."""
        return get_scaler(self._config.preprocessing.scaler)

    def get_config(self) -> PipelineConfig:
        """Get current pipeline configuration."""
        return self._config

    def get_model(self, name: str) -> Optional[BaseForecaster]:
        """Get a specific model instance.

        Returns:
            BaseForecaster instance or None.
        """
        if name in self._model_instances:
            return self._model_instances[name]
        for key, model in self._model_instances.items():
            if key.lower() == str(name).lower():
                return model
        return None

    def __repr__(self) -> str:
        models_str = ", ".join(self.models[:3])
        if len(self.models) > 3:
            models_str += f", ... ({len(self.models)} total)"

        return (
            f"TSForecasting("
            f"train_size={self.train_size}, "
            f"lags={self.lags}, "
            f"horizon={self.horizon}, "
            f"models=[{models_str}])"
        )


__all__ = ["TSForecasting", "TSForecastingBuilder", "model_configurations"]
