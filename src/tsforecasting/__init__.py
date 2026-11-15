"""TSForecasting: Automated Time Series Forecasting Framework.

A comprehensive pipeline for multivariate multi-horizon time series forecasting
using an Expanding Window evaluation approach with multiple regression models.

Example:
    from tsforecasting import TSForecasting, model_configurations

    # Load data
    data = pd.read_csv('timeseries.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    # Configure and fit
    tsf = TSForecasting(
        train_size=0.8,
        lags=10,
        horizon=5,
        models=['RandomForest', 'XGBoost']
    )
    tsf.fit_forecast(data)

    # Get results
    performance = tsf.history()
    forecast = tsf.forecast()
"""

__version__ = "2.1.0"
__author__ = "Luis Santos"

from tsforecasting.core.base import TSForecastingBuilder
from tsforecasting.core.enums import (
    EvaluationMetric,
    Granularity,
    ModelName,
    ScalerType,
    TimeSeriesPattern,
)
from tsforecasting.core.exceptions import (
    ConfigurationError,
    DatasetError,
    InsufficientDataError,
    ModelFitError,
    ModelNotFoundError,
    NotFittedError,
    TSForecastingError,
    ValidationError,
)
from tsforecasting.core.schemas import (
    ForecastConfig,
    PerformanceHistory,
    PipelineConfig,
    PreprocessingConfig,
)
from tsforecasting.data.generator import TimeSeriesDatasetGenerator
from tsforecasting.evaluation.intervals import IntervalEstimatorFactory, compute_intervals
from tsforecasting.evaluation.metrics import (
    best_model,
    metrics_regression,
    vertical_performance,
)
from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.models.forecasters import (
    FORECASTER_CLASSES,
    CatBoost_Forecasting,
    CatBoostForecaster,
    ExtraTrees_Forecasting,
    ExtraTreesForecaster,
    GBR_Forecasting,
    GBRForecaster,
    GeneralizedLR_Forecasting,
    GeneralizedLRForecaster,
    KNN_Forecasting,
    KNNForecaster,
    RandomForest_Forecasting,
    RandomForestForecaster,
    XGBoost_Forecasting,
    XGBoostForecaster,
)
from tsforecasting.models.registry import ModelRegistry, model_configurations
from tsforecasting.pipeline.forecasting import TSForecasting
from tsforecasting.processing.feature_selection import TreeBasedFeatureSelector
from tsforecasting.processing.processor import Processing

__all__ = [
    "__version__",
    "TSForecasting",
    "TSForecastingBuilder",
    "ModelRegistry",
    "model_configurations",
    "BaseForecaster",
    "AutoGluonForecaster",
    "RandomForestForecaster",
    "ExtraTreesForecaster",
    "GBRForecaster",
    "KNNForecaster",
    "GeneralizedLRForecaster",
    "XGBoostForecaster",
    "CatBoostForecaster",
    "FORECASTER_CLASSES",
    "RandomForest_Forecasting",
    "ExtraTrees_Forecasting",
    "GBR_Forecasting",
    "KNN_Forecasting",
    "GeneralizedLR_Forecasting",
    "XGBoost_Forecasting",
    "CatBoost_Forecasting",
    "Processing",
    "TreeBasedFeatureSelector",
    "TimeSeriesDatasetGenerator",
    "vertical_performance",
    "best_model",
    "metrics_regression",
    "IntervalEstimatorFactory",
    "compute_intervals",
    "Granularity",
    "EvaluationMetric",
    "ModelName",
    "TimeSeriesPattern",
    "ScalerType",
    "ForecastConfig",
    "PreprocessingConfig",
    "PipelineConfig",
    "PerformanceHistory",
    "TSForecastingError",
    "NotFittedError",
    "ValidationError",
    "ConfigurationError",
    "ModelNotFoundError",
    "ModelFitError",
    "DatasetError",
    "InsufficientDataError",
]
