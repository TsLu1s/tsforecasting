from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from tsforecasting.core.base import BasePipeline, BuilderPipeline, TSForecastingBuilder
from tsforecasting.core.enums import EvaluationMetric, Granularity, ModelName, ScalerType
from tsforecasting.core.exceptions import NotFittedError
from tsforecasting.core.schemas import (
    ForecastConfig,
    #ForecastOutput,
    PerformanceHistory,
    PipelineConfig,
    PreprocessingConfig,
)
from tsforecasting.evaluation.intervals import IntervalEstimatorFactory
from tsforecasting.evaluation.engine import EvaluationEngine
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
    - Expanding window cross-validation
    - Automatic model selection based on performance metrics
    - Confidence interval estimation
    
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
        train_size: Proportion of data for training (0.3 to 1.0).
        lags: Number of lag features (window size).
        horizon: Number of future steps to forecast.
        sliding_size: Window slide size for expanding evaluation.
        models: List of model names to evaluate.
        hparameters: Model hyperparameter configurations.
        granularity: Time granularity of data.
        metric: Evaluation metric for model selection.
    """
    
    # Available models for reference
    AVAILABLE_MODELS = [
        "RandomForest", "ExtraTrees", "GBR", "KNN", "GeneralizedLR",
        "XGBoost", "Catboost", "AutoGluon"
    ]
    
    # Fixed coverage levels for prediction intervals
    _COVERAGE_LEVELS = (0.80, 0.90, 0.95, 0.99)
    
    def __init__(
        self,
        train_size: float = 0.8,
        lags: int = 7,
        horizon: int = 15,
        sliding_size: int = 15,
        models: List[str] = None,
        hparameters: Optional[Dict[str, Dict[str, Any]]] = None,
        granularity: str = "1d",
        metric: str = "MAE",
        scaler: str = "standard",
        datetime_features: bool = True,
    ):
        """Initialize TSForecasting pipeline.
        
        Args:
            train_size: Training data proportion (0.3 to 1.0).
            lags: Number of lag features.
            horizon: Forecast horizon (steps ahead).
            sliding_size: Sliding window size for evaluation.
            models: List of model names to evaluate.
            hparameters: Custom hyperparameters dict.
            granularity: Time granularity (1m, 30m, 1h, 1d, 1wk, 1mo).
            metric: Evaluation metric (MAE, MAPE, MSE).
            scaler: Feature scaler (standard, minmax, robust).
            datetime_features: Whether to engineer datetime features.
        """
        super().__init__()
        
        # Default models
        if models is None:
            models = ["RandomForest", "GBR", "XGBoost"]
        
        # Build configuration
        self._config = PipelineConfig(
            forecast=ForecastConfig(
                train_size=train_size,
                lags=lags,
                horizon=horizon,
                sliding_size=sliding_size,
                granularity=Granularity(granularity),
                metric=EvaluationMetric(metric.lower()),
            ),
            preprocessing=PreprocessingConfig(
                scaler=ScalerType(scaler),
                datetime_features=datetime_features,
            ),
            models=[ModelName(m) for m in models],
        )
        
        # Hyperparameters
        self._hparameters = hparameters or model_configurations()
        
        # Internal state
        self._processor = Processing()
        self._evaluation_engine: Optional[EvaluationEngine] = None
        self._model_instances: Dict[str, BaseForecaster] = {}
        self._selected_model: Optional[str] = None
        self._fit_predictions: Optional[pd.DataFrame] = None
        self._complete_performance: Optional[pd.DataFrame] = None
        self._timeseries: Optional[pd.DataFrame] = None
        self._scaler = None
    
    @classmethod
    def builder(cls) -> TSForecastingBuilder:
        """Get a builder instance for advanced configuration.
        
        Returns:
            TSForecastingBuilder instance.
        """
        return TSForecastingBuilder()
    
    # =========================================================================
    # Properties for backward compatibility
    # =========================================================================
    
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
        return [m.value for m in self._config.models]
    
    # =========================================================================
    # Abstract method implementations
    # =========================================================================
    
    def _preprocess(self, dataset: pd.DataFrame) -> None:
        """Preprocess dataset and initialize models."""
        self._dataset = dataset.copy()
        
        # Initialize evaluation engine
        self._evaluation_engine = EvaluationEngine(
            config=self._config.forecast,
            preprocessing=self._config.preprocessing,
        )
        
        # Initialize model instances
        self._model_instances = {}
        for model_name in self._config.models:
            name = model_name.value
            if name in self.AVAILABLE_MODELS:
                params = self._hparameters.get(name, {})
                self._model_instances[name] = ModelRegistry.get(name, params)
    
    def _run_evaluation(self) -> None:
        """Run expanding window evaluation for all models."""
        self._fit_predictions = self._evaluation_engine.evaluate(
            dataset=self._dataset,
            models=self._model_instances,
        )
        
        # Compute performance metrics
        self._complete_performance = vertical_performance(
            self._fit_predictions,
            self.horizon
        ).reset_index(drop=True)
    
    def _select_best_model(self) -> None:
        """Select best model based on metric."""
        self._selected_model = best_model(
            self._complete_performance,
            metric=self.metric
        )
    
    # =========================================================================
    # Public methods
    # =========================================================================
    
    def history(self) -> PerformanceHistory:
        """Get historical performance data.
        
        Returns:
            PerformanceHistory dataclass containing:
            - predictions: Raw forecast DataFrame
            - performance_complete: Detailed metrics per window/horizon
            - performance_by_horizon: Aggregated by horizon
            - leaderboard: Model rankings
            - selected_model: Best performing model name
            
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
            predictions=self._fit_predictions,
            performance_complete=self._complete_performance,
            performance_by_horizon=aggregated["by_horizon"],
            leaderboard=aggregated["leaderboard"],
            selected_model=self._selected_model,
        )
    
    def forecast(
        self,
        dataset: Optional[pd.DataFrame] = None,
        interval_method: str = "ensemble",  # Options: "ensemble", "quantile", "conformal", "gaussian"
    ) -> pd.DataFrame:
        """Generate future forecasts using best model.
        
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
        """
        if not self._is_fitted:
            raise NotFittedError("Pipeline not fitted. Call fit_forecast first.")
        
        X = dataset.copy() if dataset is not None else self._dataset.copy()
        self._validate_dataset(X)
        
        # Generate future timestamps
        forecast_df = self._processor.future_timestamps(
            X, self.horizon, self.granularity
        )
        
        # Prepare time series
        timeseries = self._processor.make_timeseries(
            dataset=X,
            window_size=self.lags,
            horizon=self.horizon,
        )
        self._timeseries = timeseries.copy()
        
        # Split for final prediction
        train = timeseries.iloc[:-self.horizon].copy()
        test = timeseries.tail(1).copy()
        
        # Get column lists
        input_cols = self._evaluation_engine.input_cols
        
        # Scale
        self._scaler = self._get_scaler()
        self._scaler.fit(train[input_cols])
        train[input_cols] = self._scaler.transform(train[input_cols])
        test[input_cols] = self._scaler.transform(test[input_cols])
        
        # Get best model and predict
        best_model_instance = self._model_instances[self._selected_model]
        
        predictions = self._evaluation_engine.single_fit_predict(
            train=train,
            test=test,
            model=best_model_instance,
            model_name=self._selected_model,
        )
        
        # Extract point forecasts
        point_forecast = predictions.values.flatten()[:self.horizon]
        forecast_df["y"] = point_forecast
        
        # Compute prediction intervals for fixed coverage levels
        for coverage in self._COVERAGE_LEVELS:
            lower, upper = self._compute_intervals(
                forecast=point_forecast,
                method=interval_method,
                coverage=coverage,
            )
            coverage_pct = int(coverage * 100)
            forecast_df[f"y_lower_{coverage_pct}"] = lower
            forecast_df[f"y_upper_{coverage_pct}"] = upper
        
        return forecast_df
    
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
        
        # Extract historical actuals and predictions for the selected model
        y_true, y_pred = self._get_historical_residuals()
        
        estimator = IntervalEstimatorFactory.create(method, coverage)
        estimator.fit(y_true, y_pred)
        
        return estimator.predict(forecast)
    
    def _get_historical_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract historical y_true and y_pred from fit predictions.
        
        Returns:
            Tuple of (y_true, y_pred) arrays, shape (n_samples, n_horizons).
        """
        df = self._fit_predictions.copy()
        
        # Filter for selected model
        model_df = df[df["Model"] == self._selected_model]
        
        # Extract actual and predicted columns
        actual_cols = [f"y_horizon_{i+1}" for i in range(self.horizon)]
        pred_cols = [f"y_forecast_horizon_{i+1}" for i in range(self.horizon)]
        
        y_true = model_df[actual_cols].values
        y_pred = model_df[pred_cols].values
        
        return y_true, y_pred
    
    def _get_scaler(self):
        """Get scaler based on configuration."""
        scalers = {
            ScalerType.STANDARD: StandardScaler,
            ScalerType.MINMAX: MinMaxScaler,
            ScalerType.ROBUST: RobustScaler,
        }
        return scalers[self._config.preprocessing.scaler]()
    
    def get_config(self) -> PipelineConfig:
        """Get current pipeline configuration."""
        return self._config
    
    def get_model(self, name: str) -> Optional[BaseForecaster]:
        """Get a specific model instance.

        Returns:
            BaseForecaster instance or None.
        """
        return self._model_instances.get(name)
    
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