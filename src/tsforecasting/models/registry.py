from __future__ import annotations

import copy
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

        # Register a custom model and use it in the pipeline
        ModelRegistry.register("Ridge", RidgeForecaster, default_params={"alpha": 1.0})
        tsf = TSForecasting(models=["Ridge", "RandomForest"])
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
    def register(
        cls,
        name: str,
        forecaster_class: Type[BaseForecaster],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new forecaster class.

        Registered models can be selected by name in ``TSForecasting(models=[...])``.

        Args:
            name: Model identifier string.
            forecaster_class: BaseForecaster subclass.
            default_params: Optional default hyperparameters for the model.

        Raises:
            TypeError: If ``forecaster_class`` is not a BaseForecaster subclass.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name must be a non-empty string.")
        if not (
            isinstance(forecaster_class, type) and issubclass(forecaster_class, BaseForecaster)
        ):
            raise TypeError("forecaster_class must be a subclass of BaseForecaster.")

        name = name.strip()
        cls._registry[name] = forecaster_class
        if default_params is not None:
            cls._default_configurations[name] = copy.deepcopy(default_params)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a model (and its default configuration) from the registry.

        Raises:
            ModelNotFoundError: If model name not in registry.
        """
        resolved = cls.resolve_name(name)
        del cls._registry[resolved]
        cls._default_configurations.pop(resolved, None)

    @classmethod
    def resolve_name(cls, name: str) -> str:
        """Resolve a model name to its registered spelling (case-insensitive).

        Args:
            name: Model identifier, e.g. "xgboost", "CatBoost" or ``ModelName.KNN``.

        Returns:
            Registered model identifier string.

        Raises:
            ModelNotFoundError: If model name not in registry.
        """
        key = name.value if isinstance(name, ModelName) else str(name).strip()
        if key in cls._registry:
            return key

        lowered = key.lower()
        for registered in cls._registry:
            if registered.lower() == lowered:
                return registered

        available = ", ".join(cls._registry.keys())
        raise ModelNotFoundError(f"Model '{name}' not found. Available models: {available}")

    @classmethod
    def get(cls, name: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> BaseForecaster:
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
        name = cls.resolve_name(name)

        default_params = copy.deepcopy(cls._default_configurations.get(name, {}))
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
            Dictionary mapping model names to their default parameters. The
            result is an independent deep copy: editing it never changes the
            registry defaults.
        """
        return copy.deepcopy(cls._default_configurations)

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
        try:
            name = cls.resolve_name(name)
        except ModelNotFoundError:
            raise ModelNotFoundError(f"No default config for model '{name}'") from None
        if name not in cls._default_configurations:
            raise ModelNotFoundError(f"No default config for model '{name}'")
        return copy.deepcopy(cls._default_configurations[name])

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Model identifier string.

        Returns:
            True if model is registered.
        """
        try:
            cls.resolve_name(name)
        except ModelNotFoundError:
            return False
        return True


def model_configurations() -> Dict[str, Dict[str, Any]]:
    """Get default model configurations.

    Convenience function for backward compatibility.

    Returns:
        Dictionary mapping model names to their default parameters (a fresh
        deep copy on every call, safe to customise).
    """
    return ModelRegistry.get_default_configurations()
