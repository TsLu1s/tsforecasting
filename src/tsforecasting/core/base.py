"""Base classes for TSForecasting pipeline."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from tsforecasting.core.enums import (
    EvaluationMetric,
    Granularity,
    ModelName,
    ScalerType,
)
from tsforecasting.core.exceptions import (
    ConfigurationError,
    DatasetError,
    NotFittedError,
)
from tsforecasting.core.schemas import PerformanceHistory, PipelineConfig


def format_validation_error(error: PydanticValidationError) -> str:
    """Render a pydantic validation error as a single readable message."""
    parts = []
    for item in error.errors():
        location = ".".join(str(loc) for loc in item.get("loc", ()))
        message = item.get("msg", "invalid value")
        parts.append(f"{location}: {message}" if location else message)
    return "Invalid configuration - " + "; ".join(parts)


def resolve_model_names(models: Any) -> List[str]:
    """Resolve model identifiers against the registry (case-insensitive).

    Args:
        models: A model name, ``ModelName`` member, or a list of them.

    Returns:
        Registered model names, without duplicates and in the given order.

    Raises:
        ConfigurationError: If no model is given.
        ModelNotFoundError: If a name is not registered.
    """
    # Imported lazily: the registry lives in the models layer, above core.
    from tsforecasting.models.registry import ModelRegistry

    if isinstance(models, (str, ModelName)):
        models = [models]
    if models is None or len(models) == 0:
        raise ConfigurationError("At least one model must be provided.")

    resolved: List[str] = []
    for model in models:
        name = ModelRegistry.resolve_name(model)
        if name not in resolved:
            resolved.append(name)
    return resolved


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
        self._is_fitted = False
        dataset = self._prepare_dataset(dataset)
        self._store_input_schema(dataset)
        self._preprocess(dataset)
        self._run_evaluation()
        self._select_best_model()
        self._is_fitted = True
        return self

    def _validate_dataset(self, dataset: pd.DataFrame) -> None:
        """Validate dataset has required columns and format."""
        if not isinstance(dataset, pd.DataFrame):
            raise DatasetError(f"Dataset must be a pandas DataFrame, got {type(dataset).__name__}.")

        missing = set(self.REQUIRED_COLUMNS) - set(dataset.columns)
        if missing:
            raise DatasetError(
                f"Dataset missing required columns: {missing}. "
                f"Expected columns: {self.REQUIRED_COLUMNS}"
            )

        if not pd.api.types.is_datetime64_any_dtype(dataset["Date"]):
            raise DatasetError(
                "Column 'Date' must be datetime type. " "Use pd.to_datetime() to convert."
            )

        if dataset.empty:
            raise DatasetError("Dataset is empty.")

        if dataset["Date"].isna().any():
            raise DatasetError(
                f"Column 'Date' contains {int(dataset['Date'].isna().sum())} missing "
                f"timestamp(s) (NaT). Remove or repair them before fitting."
            )

        if dataset["y"].isna().all():
            raise DatasetError("Column 'y' contains only NaN values.")

        if pd.api.types.is_bool_dtype(dataset["y"]) or not pd.api.types.is_numeric_dtype(
            dataset["y"]
        ):
            raise DatasetError(
                f"Column 'y' must be numeric, got dtype '{dataset['y'].dtype}'. "
                f"Use pd.to_numeric() to convert."
            )

        n_missing = int(dataset["y"].isna().sum())
        if n_missing:
            raise DatasetError(
                f"Column 'y' contains {n_missing} missing value(s) (NaN). Impute them "
                f"(e.g. dataset['y'].interpolate()) or drop those rows before fitting."
            )

        if not np.isfinite(dataset["y"].to_numpy(dtype=float)).all():
            raise DatasetError("Column 'y' contains infinite values.")

        n_duplicates = int(dataset["Date"].duplicated().sum())
        if n_duplicates:
            raise DatasetError(
                f"Column 'Date' contains {n_duplicates} duplicated timestamp(s). "
                f"Each timestamp must appear once: aggregate or drop the duplicates."
            )

    def _prepare_dataset(self, dataset: pd.DataFrame, announce: bool = True) -> pd.DataFrame:
        """Validate the dataset and return a clean, chronologically ordered copy.

        Args:
            dataset: Raw input DataFrame.
            announce: Whether to emit advisory warnings (ignored columns,
                reordering, granularity mismatch).

        Returns:
            Copy of the dataset sorted by ``Date`` with a fresh integer index.
        """
        self._validate_dataset(dataset)
        prepared = dataset.copy()

        if not prepared["Date"].is_monotonic_increasing:
            prepared = prepared.sort_values("Date", kind="stable")
            if announce:
                warnings.warn(
                    "Dataset was not ordered by 'Date'; it has been sorted chronologically.",
                    UserWarning,
                    stacklevel=3,
                )
        prepared = prepared.reset_index(drop=True)

        if announce:
            ignored = [c for c in prepared.columns if c not in self.REQUIRED_COLUMNS]
            if ignored:
                warnings.warn(
                    f"Columns {ignored} are not used: the pipeline models the target "
                    f"'y' from its own lagged values and date features.",
                    UserWarning,
                    stacklevel=3,
                )
            self._check_granularity(prepared["Date"])

        return prepared

    def _check_granularity(self, dates: pd.Series) -> None:
        """Warn when the configured granularity disagrees with the data spacing."""
        if self._config is None or len(dates) < 3:
            return

        observed = pd.Timedelta(dates.diff().dropna().median()).total_seconds()
        if not observed > 0:
            return

        configured = self._config.forecast.granularity
        if abs(math.log(observed / configured.approx_seconds)) <= math.log(1.15):
            return

        closest = min(Granularity, key=lambda g: abs(math.log(observed / g.approx_seconds)))
        hint = ""
        if abs(math.log(observed / closest.approx_seconds)) <= math.log(1.15):
            hint = f" The data looks like granularity '{closest.value}'."
        warnings.warn(
            f"Configured granularity '{configured.value}' does not match the spacing of "
            f"'Date' (median step: {pd.Timedelta(seconds=observed)}).{hint} Forecast "
            f"timestamps are generated with the configured granularity.",
            UserWarning,
            stacklevel=4,
        )

    def _store_input_schema(self, dataset: pd.DataFrame) -> None:
        """Store input schema for validation during transform."""
        self._input_schema = list(dataset.columns)
        self._dataset = dataset

    def _validate_input_schema(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Validate and align input schema with training schema."""
        if self._input_schema is None:
            raise NotFittedError("Pipeline not fitted. Call fit_forecast first.")

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
        interval_method: str = "ensemble",
    ) -> pd.DataFrame:
        """Generate future forecasts with prediction intervals."""
        ...


class BuilderPipeline:
    """Mixin providing builder-style configuration methods.

    Enables fluent interface for pipeline configuration. Every setter is
    validated: an out-of-range or unknown value raises ``ConfigurationError``
    (or ``ModelNotFoundError`` for an unknown model) instead of being stored.
    """

    _config: PipelineConfig
    _hparameters: Optional[Dict[str, Dict[str, Any]]]
    _verbose: bool

    def _on_config_change(self) -> None:
        """Hook invoked after a configuration value changes."""

    def _set_option(self, section: Any, field: str, value: Any):
        """Assign a validated configuration value."""
        try:
            setattr(section, field, value)
        except PydanticValidationError as exc:
            raise ConfigurationError(format_validation_error(exc)) from exc
        self._on_config_change()
        return self

    def with_train_size(self, train_size: float):
        """Set training data proportion (0.3 up to, but excluding, 1.0)."""
        return self._set_option(self._config.forecast, "train_size", train_size)

    def with_lags(self, lags: int):
        """Set number of lag features (window size)."""
        return self._set_option(self._config.forecast, "lags", lags)

    def with_horizon(self, horizon: int):
        """Set forecast horizon (number of future steps)."""
        return self._set_option(self._config.forecast, "horizon", horizon)

    def with_sliding_size(self, sliding_size: int):
        """Set sliding window size for expanding window evaluation."""
        return self._set_option(self._config.forecast, "sliding_size", sliding_size)

    def with_granularity(self, granularity: Granularity | str):
        """Set time granularity (1m, 30m, 1h, 1d, 1wk, 1mo)."""
        return self._set_option(self._config.forecast, "granularity", granularity)

    def with_metric(self, metric: EvaluationMetric | str):
        """Set evaluation metric (MAE, MAPE, MSE)."""
        return self._set_option(self._config.forecast, "metric", metric)

    def with_models(self, models: List[str | ModelName]):
        """Set models to evaluate."""
        return self._set_option(self._config, "models", resolve_model_names(models))

    def with_preprocessing(
        self, scaler: ScalerType | str = ScalerType.STANDARD, datetime_features: bool = True
    ):
        """Configure preprocessing options."""
        self._set_option(self._config.preprocessing, "scaler", scaler)
        return self._set_option(self._config.preprocessing, "datetime_features", datetime_features)

    def with_scaler(self, scaler: ScalerType | str):
        """Set scaler type (standard, minmax, robust)."""
        return self._set_option(self._config.preprocessing, "scaler", scaler)

    def with_datetime_features(self, enabled: bool = True):
        """Enable/disable datetime feature engineering."""
        return self._set_option(self._config.preprocessing, "datetime_features", enabled)

    def with_hyperparameters(self, hparameters: Dict[str, Dict[str, Any]]):
        """Set custom hyperparameters for models."""
        if hparameters is not None and not isinstance(hparameters, dict):
            raise ConfigurationError(
                "hparameters must be a dictionary mapping model names to parameter dictionaries."
            )
        self._hparameters = hparameters
        self._on_config_change()
        return self

    def with_hparameters(self, hparameters: Dict[str, Dict[str, Any]]):
        """Alias for with_hyperparameters."""
        return self.with_hyperparameters(hparameters)

    def with_verbose(self, verbose: bool = True):
        """Enable/disable progress output during fitting."""
        self._verbose = bool(verbose)
        return self


class TSForecastingBuilder(BuilderPipeline):
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
        self._verbose: bool = True

    def build(self):
        """Build and return configured TSForecasting instance."""
        from tsforecasting.pipeline.forecasting import TSForecasting

        return TSForecasting(
            train_size=self._config.forecast.train_size,
            lags=self._config.forecast.lags,
            horizon=self._config.forecast.horizon,
            sliding_size=self._config.forecast.sliding_size,
            models=self._config.selected_models,
            hparameters=self._hparameters,
            granularity=self._config.forecast.granularity.value,
            metric=self._config.forecast.metric.value.upper(),
            scaler=self._config.preprocessing.scaler.value,
            datetime_features=self._config.preprocessing.datetime_features,
            verbose=self._verbose,
        )

    def get_config(self) -> PipelineConfig:
        """Return current configuration."""
        return self._config
