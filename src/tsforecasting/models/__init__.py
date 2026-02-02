"""Models module for TSForecasting package.

This module provides:
- BaseForecaster: Abstract base class for all forecasters
- AutoGluonForecaster: Special handler for AutoGluon
- Individual forecaster classes (RandomForestForecaster, etc.)
- Legacy aliases (RandomForest_Forecasting, etc.) for backward compatibility
- ModelRegistry: Central registry for model lookup and configuration
"""

from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.models.forecasters import (  # New naming convention; Legacy naming convention (backward compatibility)
    FORECASTER_CLASSES,
    LEGACY_FORECASTER_CLASSES,
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

__all__ = [
    # Base classes
    "BaseForecaster",
    "AutoGluonForecaster",
    # New naming convention
    "RandomForestForecaster",
    "ExtraTreesForecaster",
    "GBRForecaster",
    "KNNForecaster",
    "GeneralizedLRForecaster",
    "XGBoostForecaster",
    "CatBoostForecaster",
    # Legacy naming convention (backward compatibility)
    "RandomForest_Forecasting",
    "ExtraTrees_Forecasting",
    "GBR_Forecasting",
    "KNN_Forecasting",
    "GeneralizedLR_Forecasting",
    "XGBoost_Forecasting",
    "CatBoost_Forecasting",
    # Registries
    "FORECASTER_CLASSES",
    "LEGACY_FORECASTER_CLASSES",
    # Registry and configuration
    "ModelRegistry",
    "model_configurations",
]
