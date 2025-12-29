from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from tsforecasting.core.enums import ModelName
from tsforecasting.core.exceptions import ModelNotFoundError
from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.models.forecasters import (
    CatBoostForecaster,
    ExtraTreesForecaster,
    GBRForecaster,
    GeneralizedLRForecaster,
    KNNForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)


class ModelRegistry:
    """Registry for forecasting models.
    
    Provides centralized model management with lazy instantiation
    and default hyperparameter configurations.
    
    Example:
        # Get default configurations
        configs = ModelRegistry.get_default_configurations()
        
        # Create a model instance
        model = ModelRegistry.get("RandomForest", n_estimators=200)
        
        # List available models
        available = ModelRegistry.list_available()
    """
    
    _registry: Dict[str, Type[BaseForecaster]] = {
        ModelName.RANDOM_FOREST.value: RandomForestForecaster,
        ModelName.EXTRA_TREES.value: ExtraTreesForecaster,
        ModelName.GBR.value: GBRForecaster,
        ModelName.KNN.value: KNNForecaster,
        ModelName.GENERALIZED_LR.value: GeneralizedLRForecaster,
        ModelName.XGBOOST.value: XGBoostForecaster,
        ModelName.CATBOOST.value: CatBoostForecaster,
        ModelName.AUTOGLUON.value: AutoGluonForecaster,
    }
    
    _default_configurations: Dict[str, Dict[str, Any]] = {
        "RandomForest": {
            "n_estimators": 100,
            "random_state": 42,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        "ExtraTrees": {
            "n_estimators": 100,
            "random_state": 42,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        "GBR": {
            "n_estimators": 100,
            "criterion": "friedman_mse",
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "loss": "squared_error",
        },
        "KNN": {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
        },
        "GeneralizedLR": {
            "power": 1,
            "alpha": 0.5,
            "link": "log",
            "fit_intercept": True,
            "max_iter": 100,
            "warm_start": False,
        },
        "XGBoost": {
            "objective": "reg:squarederror",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "reg_lambda": 1,
            "reg_alpha": 0,
            "subsample": 1,
            "colsample_bytree": 1,
        },
        "Catboost": {
            "iterations": 100,
            "loss_function": "RMSE",
            "depth": 8,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3,
            "border_count": 254,
            "subsample": 1,
        },
        "AutoGluon": {
            "eval_metric": "mean_squared_error",
            "verbosity": 0,
            "presets": "medium_quality",
            "time_limit": 10,
        },
    }
    
    @classmethod
    def register(cls, name: str, forecaster_class: Type[BaseForecaster]) -> None:
        """Register a new forecaster class.
        
        Args:
            name: Model identifier string.
            forecaster_class: BaseForecaster subclass.
        """
        cls._registry[name] = forecaster_class
    
    @classmethod
    def get(
        cls,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseForecaster:
        """Get a forecaster instance.
        
        Args:
            name: Model identifier string.
            params: Optional hyperparameters dict.
            **kwargs: Additional hyperparameters (override params).
            
        Returns:
            Configured BaseForecaster instance.
            
        Raises:
            ModelNotFoundError: If model name not in registry.
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ModelNotFoundError(
                f"Model '{name}' not found. Available models: {available}"
            )
        
        # Merge default config with provided params
        default_params = cls._default_configurations.get(name, {}).copy()
        if params:
            default_params.update(params)
        default_params.update(kwargs)
        
        forecaster_class = cls._registry[name]
        return forecaster_class(**default_params)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered model names.
        
        Returns:
            List of model identifier strings.
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_default_configurations(cls) -> Dict[str, Dict[str, Any]]:
        """Get default hyperparameter configurations for all models.
        
        Returns:
            Dictionary mapping model names to their default parameters.
        """
        return cls._default_configurations.copy()
    
    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """Get default configuration for a specific model.
        
        Args:
            name: Model identifier string.
            
        Returns:
            Dictionary of default hyperparameters.
            
        Raises:
            ModelNotFoundError: If model name not found.
        """
        if name not in cls._default_configurations:
            raise ModelNotFoundError(f"No default config for model '{name}'")
        return cls._default_configurations[name].copy()
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.
        
        Args:
            name: Model identifier string.
            
        Returns:
            True if model is registered.
        """
        return name in cls._registry


def model_configurations() -> Dict[str, Dict[str, Any]]:
    """Get default model configurations.
    
    Convenience function for backward compatibility.
    
    Returns:
        Dictionary mapping model names to their default parameters.
    """
    return ModelRegistry.get_default_configurations()