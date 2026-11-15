"""Pydantic models and dataclasses for TSForecasting package."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from tsforecasting.core.enums import (
    EvaluationMetric,
    Granularity,
    ModelName,
    ScalerType,
)


class ForecastConfig(BaseModel):
    """Configuration for forecasting pipeline."""

    train_size: float = Field(default=0.8, ge=0.3, lt=1.0)
    lags: int = Field(default=7, gt=0)
    horizon: int = Field(default=15, gt=0)
    sliding_size: int = Field(default=15, gt=0)
    granularity: Granularity = Field(default=Granularity.DAILY)
    metric: EvaluationMetric = Field(default=EvaluationMetric.MAE)

    # validate_assignment keeps the constraints above enforced when a field is
    # changed after construction (e.g. through the builder ``with_*`` methods).
    model_config = {"use_enum_values": False, "validate_assignment": True}

    @field_validator("granularity", mode="before")
    @classmethod
    def _parse_granularity(cls, value: Any) -> Any:
        return Granularity.from_string(value) if isinstance(value, str) else value

    @field_validator("metric", mode="before")
    @classmethod
    def _parse_metric(cls, value: Any) -> Any:
        return EvaluationMetric.from_string(value) if isinstance(value, str) else value


class ModelConfig(BaseModel):
    """Configuration for a single forecasting model."""

    name: ModelName
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": False}


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing steps."""

    scaler: ScalerType = Field(default=ScalerType.STANDARD)
    datetime_features: bool = Field(default=True)

    model_config = {"use_enum_values": False, "validate_assignment": True}

    @field_validator("scaler", mode="before")
    @classmethod
    def _parse_scaler(cls, value: Any) -> Any:
        return ScalerType.from_string(value) if isinstance(value, str) else value


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    # Built-in models are stored as ``ModelName`` members; models added through
    # ``ModelRegistry.register`` are stored under their registered string name.
    models: List[Union[ModelName, str]] = Field(
        default_factory=lambda: [ModelName.RANDOM_FOREST, ModelName.GBR, ModelName.XGBOOST]
    )

    model_config = {"use_enum_values": False, "validate_assignment": True}

    @field_validator("models", mode="before")
    @classmethod
    def _normalise_models(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return value

        parsed: List[Any] = []
        for model in value:
            if isinstance(model, str) and not isinstance(model, ModelName):
                try:
                    model = ModelName(model)
                except ValueError:
                    model = model.strip()
            if model not in parsed:
                parsed.append(model)
        return parsed

    @field_validator("models")
    @classmethod
    def _require_models(cls, value: List[Union[ModelName, str]]) -> List[Union[ModelName, str]]:
        if not value:
            raise ValueError("At least one model must be provided.")
        return value

    @property
    def selected_models(self) -> List[str]:
        """Model identifiers as plain strings."""
        return [m.value if isinstance(m, ModelName) else str(m) for m in self.models]


@dataclass(frozen=True)
class WindowResult:
    """Result from a single evaluation window."""

    window: int
    model: str
    horizon: int
    y_true: tuple  # Using tuple for immutability (from np.ndarray)
    y_pred: tuple  # Using tuple for immutability (from np.ndarray)
    metrics: Dict[str, float]

    @classmethod
    def from_arrays(
        cls,
        window: int,
        model: str,
        horizon: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, float],
    ) -> "WindowResult":
        """Create WindowResult from numpy arrays."""
        return cls(
            window=window,
            model=model,
            horizon=horizon,
            y_true=tuple(y_true.flatten()),
            y_pred=tuple(y_pred.flatten()),
            metrics=dict(metrics),
        )

    @property
    def y_true_array(self) -> np.ndarray:
        """Return y_true as numpy array."""
        return np.array(self.y_true)

    @property
    def y_pred_array(self) -> np.ndarray:
        """Return y_pred as numpy array."""
        return np.array(self.y_pred)


@dataclass
class EvaluationResult:
    """Aggregated results from model evaluation."""

    window_results: List[WindowResult] = field(default_factory=list)
    predictions_df: Optional[pd.DataFrame] = None

    def add_result(self, result: WindowResult) -> None:
        """Add a window result."""
        self.window_results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        records = []
        for wr in self.window_results:
            records.append(
                {"Window": wr.window, "Model": wr.model, "Horizon": wr.horizon, **wr.metrics}
            )
        return pd.DataFrame(records)

    @property
    def models(self) -> List[str]:
        """Return unique models in results."""
        return list(set(wr.model for wr in self.window_results))


@dataclass
class ForecastOutput:
    """Output from forecast method."""

    predictions: pd.DataFrame
    selected_model: str
    confidence_intervals: Optional[pd.DataFrame] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Combine predictions and intervals into single DataFrame."""
        if self.confidence_intervals is None:
            return self.predictions.copy()
        return self.predictions.merge(self.confidence_intervals, on="Date", how="left")

    def __repr__(self) -> str:
        return (
            f"ForecastOutput(model='{self.selected_model}', " f"horizons={len(self.predictions)})"
        )


@dataclass
class PerformanceHistory:
    """Historical performance data from fitting."""

    predictions: pd.DataFrame
    performance_complete: pd.DataFrame
    performance_by_horizon: pd.DataFrame
    leaderboard: pd.DataFrame
    selected_model: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (backward compatibility)."""
        return {
            "Predictions": self.predictions,
            "Performance Complete": self.performance_complete,
            "Performance by Horizon": self.performance_by_horizon,
            "Leaderboard": self.leaderboard,
        }

    def __repr__(self) -> str:
        return (
            f"PerformanceHistory(models={len(self.leaderboard)}, "
            f"selected='{self.selected_model}')"
        )
