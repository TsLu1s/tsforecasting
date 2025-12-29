"""Models module for TSForecasting package.

This module provides:
- BaseForecaster: Abstract base class for all forecasters
- AutoGluonForecaster: Special handler for AutoGluon
- Individual forecaster classes (RandomForestForecaster, etc.)
- Legacy aliases (RandomForest_Forecasting, etc.) for backward compatibility
- ModelRegistry: Central registry for model lookup and configuration
"""

from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.models.forecasters import (
    # New naming convention
    CatBoostForecaster,
    ExtraTreesForecaster,
    FORECASTER_CLASSES,
    GBRForecaster,
    GeneralizedLRForecaster,
    KNNForecaster,
    LEGACY_FORECASTER_CLASSES,
    RandomForestForecaster,
    XGBoostForecaster,
    # Legacy naming convention (backward compatibility)
    RandomForest_Forecasting,
    ExtraTrees_Forecasting,
    GBR_Forecasting,
    KNN_Forecasting,
    GeneralizedLR_Forecasting,
    XGBoost_Forecasting,
    CatBoost_Forecasting,
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