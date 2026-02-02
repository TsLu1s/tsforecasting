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

__version__ = "2.0.0"
__author__ = "Luis Santos"

# Builder
from tsforecasting.core.base import TSForecastingBuilder

# Enums for type-safe configuration
from tsforecasting.core.enums import (
    EvaluationMetric,
    Granularity,
    ModelName,
    ScalerType,
    TimeSeriesPattern,
)

# Exceptions
from tsforecasting.core.exceptions import (
    DatasetError,
    ModelNotFoundError,
    NotFittedError,
    TSForecastingError,
    ValidationError,
)

# Schemas for advanced usage
from tsforecasting.core.schemas import (
    ForecastConfig,
    PipelineConfig,
    PreprocessingConfig,
)

# Data generation
from tsforecasting.data.generator import TimeSeriesDatasetGenerator

# Evaluation utilities
from tsforecasting.evaluation.metrics import (
    best_model,
    metrics_regression,
    vertical_performance,
)

# Base forecaster and AutoGluon
from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster

# Forecasting models (legacy naming convention for backward compatibility)
# Forecasting models (new naming convention)
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

# Configuration utilities
from tsforecasting.models.registry import ModelRegistry, model_configurations

# Core pipeline
from tsforecasting.pipeline.forecasting import TSForecasting
from tsforecasting.processing.feature_selection import TreeBasedFeatureSelector

# Processing utilities
from tsforecasting.processing.processor import Processing

__all__ = [
    # Version
    "__version__",
    # Main classes
    "TSForecasting",
    "TSForecastingBuilder",
    # Registry
    "ModelRegistry",
    "model_configurations",
    # Base forecaster
    "BaseForecaster",
    "AutoGluonForecaster",
    # Forecasters (new naming)
    "RandomForestForecaster",
    "ExtraTreesForecaster",
    "GBRForecaster",
    "KNNForecaster",
    "GeneralizedLRForecaster",
    "XGBoostForecaster",
    "CatBoostForecaster",
    "FORECASTER_CLASSES",
    # Forecasters (legacy naming for backward compatibility)
    "RandomForest_Forecasting",
    "ExtraTrees_Forecasting",
    "GBR_Forecasting",
    "KNN_Forecasting",
    "GeneralizedLR_Forecasting",
    "XGBoost_Forecasting",
    "CatBoost_Forecasting",
    # Processing
    "Processing",
    "TreeBasedFeatureSelector",
    # Data
    "TimeSeriesDatasetGenerator",
    # Evaluation
    "vertical_performance",
    "best_model",
    "metrics_regression",
    # Enums
    "Granularity",
    "EvaluationMetric",
    "ModelName",
    "TimeSeriesPattern",
    "ScalerType",
    # Schemas
    "ForecastConfig",
    "PreprocessingConfig",
    "PipelineConfig",
    # Exceptions
    "TSForecastingError",
    "NotFittedError",
    "ValidationError",
    "ModelNotFoundError",
    "DatasetError",
]
