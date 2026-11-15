"""Tests for tsforecasting.models module."""

import warnings

import numpy as np
import pytest

from tsforecasting.core.exceptions import ModelNotFoundError, NotFittedError
from tsforecasting.models.base import BaseForecaster
from tsforecasting.models.forecasters import (  # Legacy aliases
    FORECASTER_CLASSES,
    CatBoost_Forecasting,
    CatBoostForecaster,
    ExtraTreesForecaster,
    GBRForecaster,
    GeneralizedLRForecaster,
    KNNForecaster,
    RandomForest_Forecasting,
    RandomForestForecaster,
    XGBoostForecaster,
)
from tsforecasting.models.registry import ModelRegistry, model_configurations


@pytest.fixture
def regression_data():
    """Simple regression dataset for model tests."""
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.column_stack([X[:, 0] * 2 + 1, X[:, 1] - X[:, 2]])
    return X, y


class TestForecasterFitPredict:
    @pytest.mark.parametrize(
        "forecaster_cls",
        [
            RandomForestForecaster,
            ExtraTreesForecaster,
            GBRForecaster,
            KNNForecaster,
            XGBoostForecaster,
            CatBoostForecaster,
        ],
    )
    def test_fit_predict_cycle(self, forecaster_cls, regression_data):
        X, y = regression_data
        model = forecaster_cls()
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5, 2)
        assert model.is_fitted

    def test_generalized_lr_fit_predict(self, regression_data):
        """GeneralizedLR needs positive y for Tweedie with log link."""
        X, _ = regression_data
        y = np.abs(np.random.randn(50, 2)) + 1  # positive values
        model = GeneralizedLRForecaster()
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert preds.shape == (5, 2)


class TestForecasterNotFitted:
    def test_predict_before_fit_raises(self):
        model = RandomForestForecaster()
        with pytest.raises(NotFittedError):
            model.predict(np.zeros((5, 3)))


class TestForecasterParams:
    def test_get_params(self):
        model = RandomForestForecaster(n_estimators=200)
        params = model.get_params()
        assert params["n_estimators"] == 200

    def test_set_params_resets_fitted(self, regression_data):
        X, y = regression_data
        model = RandomForestForecaster()
        model.fit(X, y)
        assert model.is_fitted
        model.set_params(n_estimators=50)
        assert not model.is_fitted

    def test_repr(self):
        model = KNNForecaster(n_neighbors=3)
        r = repr(model)
        assert "KNNForecaster" in r
        assert "n_neighbors=3" in r


class TestLegacyAliases:
    def test_random_forest_alias(self):
        assert RandomForest_Forecasting is RandomForestForecaster

    def test_catboost_alias(self):
        assert CatBoost_Forecasting is CatBoostForecaster


class TestForecasterClasses:
    def test_registry_dict(self):
        assert "RandomForest" in FORECASTER_CLASSES
        assert "XGBoost" in FORECASTER_CLASSES
        assert "Catboost" in FORECASTER_CLASSES
        assert len(FORECASTER_CLASSES) == 7  # Excludes AutoGluon


class TestModelRegistry:
    def test_list_available(self):
        available = ModelRegistry.list_available()
        assert "RandomForest" in available
        assert "XGBoost" in available
        assert "AutoGluon" in available
        assert len(available) == 8

    def test_get_model(self):
        model = ModelRegistry.get("RandomForest")
        assert isinstance(model, RandomForestForecaster)

    def test_get_with_custom_params(self):
        model = ModelRegistry.get("RandomForest", params={"n_estimators": 500})
        assert model.get_params()["n_estimators"] == 500

    def test_get_unknown_model(self):
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.get("NonExistent")

    def test_is_registered(self):
        assert ModelRegistry.is_registered("RandomForest")
        assert not ModelRegistry.is_registered("FakeModel")

    def test_get_default_configurations(self):
        configs = ModelRegistry.get_default_configurations()
        assert "RandomForest" in configs
        assert "n_estimators" in configs["RandomForest"]

    def test_get_default_config_single(self):
        config = ModelRegistry.get_default_config("XGBoost")
        assert "learning_rate" in config

    def test_get_default_config_unknown(self):
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.get_default_config("Unknown")

    def test_register_custom(self):
        """Test registering a custom model class."""

        class DummyForecaster(BaseForecaster):
            def _create_base_estimator(self):
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

        ModelRegistry.register("Dummy", DummyForecaster)
        assert ModelRegistry.is_registered("Dummy")
        ModelRegistry.unregister("Dummy")


class TestModelConfigurations:
    def test_convenience_function(self):
        configs = model_configurations()
        assert isinstance(configs, dict)
        assert "RandomForest" in configs


class TestForecasterInputs:
    def test_one_dimensional_target(self, regression_data):
        X, y = regression_data
        model = KNNForecaster().fit(X, y[:, 0])
        assert model.predict(X[:4]).shape == (4, 1)

    def test_get_params_accepts_deep(self):
        model = KNNForecaster(n_neighbors=3)
        assert model.get_params(deep=True) == model.get_params()
        assert model.get_params(deep=False)["n_neighbors"] == 3

    def test_get_params_returns_copy(self):
        model = KNNForecaster(n_neighbors=3)
        model.get_params()["n_neighbors"] = 99
        assert model.get_params()["n_neighbors"] == 3


class TestGBRCriterion:
    def test_default_configuration_emits_no_deprecation(self, regression_data):
        X, y = regression_data
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            ModelRegistry.get("GBR", n_estimators=5).fit(X, y)
            GBRForecaster(n_estimators=5).fit(X, y)

    def test_criterion_not_forwarded_by_default(self):
        assert "criterion" not in GBRForecaster().get_params()
        assert "criterion" not in model_configurations()["GBR"]

    def test_explicit_criterion_is_forwarded(self):
        assert GBRForecaster(criterion="squared_error").get_params()["criterion"] == "squared_error"


class TestRegistryIntegrity:
    def test_default_configurations_are_deep_copies(self):
        configs = ModelRegistry.get_default_configurations()
        configs["XGBoost"]["max_depth"] = 99
        configs["KNN"].clear()

        assert ModelRegistry.get_default_configurations()["XGBoost"]["max_depth"] == 3
        assert ModelRegistry.get_default_config("KNN")["n_neighbors"] == 5

    def test_single_config_is_a_copy(self):
        ModelRegistry.get_default_config("XGBoost")["max_depth"] = 99
        assert ModelRegistry.get_default_config("XGBoost")["max_depth"] == 3

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("xgboost", "XGBoost"),
            ("CATBOOST", "Catboost"),
            (" KNN ", "KNN"),
            ("autogluon", "AutoGluon"),
        ],
    )
    def test_resolve_name_is_case_insensitive(self, name, expected):
        assert ModelRegistry.resolve_name(name) == expected
        assert ModelRegistry.is_registered(name)

    def test_resolve_name_accepts_enum(self):
        from tsforecasting.core.enums import ModelName

        assert ModelRegistry.resolve_name(ModelName.GBR) == "GBR"

    def test_unknown_name_lists_available(self):
        with pytest.raises(ModelNotFoundError, match="Available models"):
            ModelRegistry.resolve_name("Prophet")

    def test_model_not_found_is_value_error(self):
        with pytest.raises(ValueError):
            ModelRegistry.get("Prophet")

    def test_register_with_default_params_and_unregister(self):
        class RidgeForecaster(BaseForecaster):
            def _create_base_estimator(self):
                from sklearn.linear_model import Ridge

                return Ridge(**self._params)

        ModelRegistry.register("Ridge", RidgeForecaster, default_params={"alpha": 2.0})
        try:
            assert ModelRegistry.get("ridge").get_params() == {"alpha": 2.0}
            assert ModelRegistry.get("Ridge", alpha=5.0).get_params() == {"alpha": 5.0}
            assert model_configurations()["Ridge"] == {"alpha": 2.0}
        finally:
            ModelRegistry.unregister("Ridge")

        assert not ModelRegistry.is_registered("Ridge")
        assert "Ridge" not in model_configurations()

    def test_register_rejects_non_forecasters(self):
        with pytest.raises(TypeError, match="BaseForecaster"):
            ModelRegistry.register("Bad", dict)
        with pytest.raises(ValueError, match="non-empty"):
            ModelRegistry.register("  ", RandomForestForecaster)

    def test_unregister_unknown(self):
        with pytest.raises(ModelNotFoundError):
            ModelRegistry.unregister("Prophet")
