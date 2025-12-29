"""Base classes for TSForecasting pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from tsforecasting.core.enums import (
    EvaluationMetric,
    Granularity,
    ModelName,
    ScalerType,
)

from tsforecasting.core.exceptions import DatasetError, ValidationError
from tsforecasting.core.schemas import PipelineConfig
from tsforecasting.core.schemas import PerformanceHistory #ForecastOutput

class BasePipeline(ABC):
    """
    Abstract base class for forecasting pipelines.
    
    Defines the template method pattern for the forecasting workflow.
    """
    
    REQUIRED_COLUMNS = ["Date", "y"]
    
    def __init__(self):
        self._config: Optional[PipelineConfig] = None
        self._is_fitted: bool = False
        self._input_schema: Optional[List[str]] = None
        self._dataset: Optional[pd.DataFrame] = None
    
    def fit_forecast(self, dataset: pd.DataFrame):
        """Execute the forecasting pipeline.

        Returns:
            Self for method chaining.
        """
        self._validate_dataset(dataset)
        self._store_input_schema(dataset)
        self._preprocess(dataset)
        self._run_evaluation()
        self._select_best_model()
        self._is_fitted = True
        return self
    
    def _validate_dataset(self, dataset: pd.DataFrame) -> None:
        """Validate dataset has required columns and format."""
        missing = set(self.REQUIRED_COLUMNS) - set(dataset.columns)
        if missing:
            raise DatasetError(
                f"Dataset missing required columns: {missing}. "
                f"Expected columns: {self.REQUIRED_COLUMNS}"
            )
        
        if not pd.api.types.is_datetime64_any_dtype(dataset["Date"]):
            raise DatasetError(
                "Column 'Date' must be datetime type. "
                "Use pd.to_datetime() to convert."
            )
        
        if dataset["y"].isna().all():
            raise DatasetError("Column 'y' contains only NaN values.")
    
    def _store_input_schema(self, dataset: pd.DataFrame) -> None:
        """Store input schema for validation during transform."""
        self._input_schema = list(dataset.columns)
        self._dataset = dataset.copy()
    
    def _validate_input_schema(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate and align input schema with training schema."""
        if self._input_schema is None:
            raise ValidationError("Pipeline not fitted. Call fit_forecast first.")
        
        missing = set(self.REQUIRED_COLUMNS) - set(dataset.columns)
        if missing:
            raise DatasetError(f"Missing required columns: {missing}")
        
        return dataset
    
    @abstractmethod
    def _preprocess(self, dataset: pd.DataFrame) -> None:
        """Preprocess the dataset."""
        ...
    
    @abstractmethod
    def _run_evaluation(self) -> None:
        """Run model evaluation across windows."""
        ...
    
    @abstractmethod
    def _select_best_model(self) -> None:
        """Select the best performing model."""
        ...
    
    @abstractmethod
    def history(self) -> "PerformanceHistory":
        """Return historical performance data."""
        ...
    
    @abstractmethod
    def forecast(
        self,
        dataset: Optional[pd.DataFrame] = None,
        interval_method: str = "quantile",
    ) -> pd.DataFrame:
        """Generate future forecasts with prediction intervals."""
        ...


class BuilderPipeline:
    """Mixin providing builder-style configuration methods.
    
    Enables fluent interface for pipeline configuration when used
    with direct inheritance pattern.
    """
    
    def with_train_size(self, train_size: float):
        """Set training data proportion."""
        self._config.forecast.train_size = train_size
        return self
    
    def with_lags(self, lags: int):
        """Set number of lag features."""
        self._config.forecast.lags = lags
        return self
    
    def with_horizon(self, horizon: int):
        """Set forecast horizon."""
        self._config.forecast.horizon = horizon
        return self
    
    def with_sliding_size(self, sliding_size: int):
        """Set sliding window size."""
        self._config.forecast.sliding_size = sliding_size
        return self
    
    def with_granularity(self, granularity: Granularity | str):
        """Set time granularity."""
        if isinstance(granularity, str):
            granularity = Granularity(granularity)
        self._config.forecast.granularity = granularity
        return self
    
    def with_metric(self, metric: EvaluationMetric | str):
        """Set evaluation metric."""
        if isinstance(metric, str):
            metric = EvaluationMetric(metric.lower())
        self._config.forecast.metric = metric
        return self
    
    def with_models(self, models: List[str | ModelName]):
        """Set models to evaluate."""
        parsed = []
        for m in models:
            if isinstance(m, str):
                parsed.append(ModelName(m))
            else:
                parsed.append(m)
        self._config.models = parsed
        return self
    
    def with_preprocessing(
        self,
        scaler: ScalerType | str = ScalerType.STANDARD,
        datetime_features: bool = True
    ):
        """Configure preprocessing options."""
        if isinstance(scaler, str):
            scaler = ScalerType(scaler)
        self._config.preprocessing.scaler = scaler
        self._config.preprocessing.datetime_features = datetime_features
        return self
    
    def with_scaler(self, scaler: ScalerType | str):
        """Set scaler type."""
        if isinstance(scaler, str):
            scaler = ScalerType(scaler)
        self._config.preprocessing.scaler = scaler
        return self
    
    def with_datetime_features(self, enabled: bool = True):
        """Enable/disable datetime feature engineering."""
        self._config.preprocessing.datetime_features = enabled
        return self
    
    def with_hyperparameters(self, hparameters: Dict[str, Dict[str, Any]]):
        """Set custom hyperparameters for models."""
        self._hparameters = hparameters
        return self
    
    # Alias for convenience
    with_hparameters = with_hyperparameters


class TSForecastingBuilder:
    """
    Standalone builder for advanced pipeline customization.
    
    Provides full control over pipeline construction for advanced users.
    
    Example:
        builder = TSForecastingBuilder()
        tsf = (builder
               .with_train_size(0.8)
               .with_lags(10)
               .with_horizon(5)
               .with_models(["RandomForest", "XGBoost"])
               .with_preprocessing(scaler="robust")
               .build())
    """
    
    def __init__(self):
        self._config = PipelineConfig()
        self._hparameters: Optional[Dict[str, Dict[str, Any]]] = None
    
    def with_train_size(self, train_size: float):
        """Set training data proportion (0.3 to 1.0)."""
        self._config.forecast.train_size = train_size
        return self
    
    def with_lags(self, lags: int):
        """Set number of lag features (window size)."""
        self._config.forecast.lags = lags
        return self
    
    def with_horizon(self, horizon: int):
        """Set forecast horizon (number of future steps)."""
        self._config.forecast.horizon = horizon
        return self
    
    def with_sliding_size(self, sliding_size: int):
        """Set sliding window size for expanding window evaluation."""
        self._config.forecast.sliding_size = sliding_size
        return self
    
    def with_granularity(self, granularity: Granularity | str):
        """Set time granularity (1m, 30m, 1h, 1d, 1wk, 1mo)."""
        if isinstance(granularity, str):
            granularity = Granularity(granularity)
        self._config.forecast.granularity = granularity
        return self
    
    def with_metric(self, metric: EvaluationMetric | str):
        """Set evaluation metric (mae, mape, mse)."""
        if isinstance(metric, str):
            metric = EvaluationMetric(metric.lower())
        self._config.forecast.metric = metric
        return self
    
    def with_models(self, models: List[str | ModelName]):
        """Set models to evaluate."""
        parsed = []
        for m in models:
            if isinstance(m, str):
                parsed.append(ModelName(m))
            else:
                parsed.append(m)
        self._config.models = parsed
        return self
    
    def with_preprocessing(
        self,
        scaler: ScalerType | str = ScalerType.STANDARD,
        datetime_features: bool = True
    ):
        """Configure preprocessing options."""
        if isinstance(scaler, str):
            scaler = ScalerType(scaler)
        self._config.preprocessing.scaler = scaler
        self._config.preprocessing.datetime_features = datetime_features
        return self
    
    def with_scaler(self, scaler: ScalerType | str):
        """Set scaler type (standard, minmax, robust)."""
        if isinstance(scaler, str):
            scaler = ScalerType(scaler)
        self._config.preprocessing.scaler = scaler
        return self
    
    def with_hyperparameters(self, hparameters: Dict[str, Dict[str, Any]]):
        """Set model hyperparameters."""
        self._hparameters = hparameters
        return self
    
    # Alias for convenience
    def with_hparameters(self, hparameters: Dict[str, Dict[str, Any]]):
        """Alias for with_hyperparameters."""
        return self.with_hyperparameters(hparameters)
    
    def with_datetime_features(self, enabled: bool = True):
        """Enable/disable datetime feature engineering."""
        self._config.preprocessing.datetime_features = enabled
        return self
    
    def build(self): #-> "TSForecasting":
        """Build and return configured TSForecasting instance."""
        from tsforecasting.pipeline.forecasting import TSForecasting
        # Convert config to constructor arguments
        models_list = [m.value for m in self._config.models]
        
        return TSForecasting(
            train_size=self._config.forecast.train_size,
            lags=self._config.forecast.lags,
            horizon=self._config.forecast.horizon,
            sliding_size=self._config.forecast.sliding_size,
            models=models_list,
            hparameters=self._hparameters,
            granularity=self._config.forecast.granularity.value,
            metric=self._config.forecast.metric.value.upper(),
            scaler=self._config.preprocessing.scaler.value,
            datetime_features=self._config.preprocessing.datetime_features,
        )
    
    def get_config(self) -> PipelineConfig:
        """Return current configuration."""
        return self._config






























