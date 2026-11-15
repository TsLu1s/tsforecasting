"""Tests for input validation, configuration handling and pipeline state."""

import warnings

import numpy as np
import pandas as pd
import pytest

from tsforecasting import (
    ConfigurationError,
    DatasetError,
    ModelNotFoundError,
    NotFittedError,
    TSForecasting,
    TSForecastingBuilder,
    model_configurations,
)
from tsforecasting.data.generator import TimeSeriesDatasetGenerator
from tsforecasting.models.base import BaseForecaster
from tsforecasting.models.registry import ModelRegistry


@pytest.fixture
def data():
    return TimeSeriesDatasetGenerator.generate(n_samples=200, random_state=42)


def quick_pipeline(**kwargs):
    params = dict(lags=5, horizon=3, sliding_size=10, models=["KNN"], verbose=False)
    params.update(kwargs)
    return TSForecasting(**params)


class CountingForecaster(BaseForecaster):
    """Linear model that counts how many times it is fitted."""

    def __init__(self, **params):
        super().__init__(**params)
        self.n_fits = 0

    def _create_base_estimator(self):
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    def fit(self, X, y):
        self.n_fits += 1
        return super().fit(X, y)


@pytest.fixture
def counting_model():
    ModelRegistry.register("Counting", CountingForecaster)
    yield "Counting"
    ModelRegistry.unregister("Counting")


class TestDatasetValidation:
    def test_not_a_dataframe(self):
        with pytest.raises(DatasetError, match="pandas DataFrame"):
            quick_pipeline().fit_forecast([1, 2, 3])

    def test_missing_values_in_target(self, data):
        data.loc[50, "y"] = np.nan
        with pytest.raises(DatasetError, match="1 missing value"):
            quick_pipeline().fit_forecast(data)

    def test_infinite_values_in_target(self, data):
        data.loc[50, "y"] = np.inf
        with pytest.raises(DatasetError, match="infinite"):
            quick_pipeline().fit_forecast(data)

    def test_non_numeric_target(self, data):
        data["y"] = data["y"].astype(str)
        with pytest.raises(DatasetError, match="must be numeric"):
            quick_pipeline().fit_forecast(data)

    def test_missing_timestamps(self, data):
        data.loc[10, "Date"] = pd.NaT
        with pytest.raises(DatasetError, match="NaT"):
            quick_pipeline().fit_forecast(data)

    def test_duplicated_timestamps(self, data):
        duplicated = pd.concat([data, data.iloc[:20]])
        with pytest.raises(DatasetError, match="20 duplicated timestamp"):
            quick_pipeline().fit_forecast(duplicated)

    def test_dataset_errors_are_value_errors(self, data):
        data.loc[50, "y"] = np.nan
        with pytest.raises(ValueError):
            quick_pipeline().fit_forecast(data)

    def test_input_dataset_is_not_modified(self, data):
        shuffled = data.sample(frac=1, random_state=0)
        snapshot = shuffled.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quick_pipeline().fit_forecast(shuffled)
        pd.testing.assert_frame_equal(shuffled, snapshot)


class TestChronologicalOrder:
    def test_unsorted_dataset_is_sorted_with_warning(self, data):
        shuffled = data.sample(frac=1, random_state=0)
        tsf = quick_pipeline()
        with pytest.warns(UserWarning, match="sorted chronologically"):
            tsf.fit_forecast(shuffled)
        forecast = tsf.forecast()
        assert forecast["Date"].iloc[0] == data["Date"].max() + pd.Timedelta(days=1)

    def test_unsorted_and_sorted_inputs_give_same_forecast(self, data):
        reference = quick_pipeline().fit_forecast(data).forecast()
        tsf = quick_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsf.fit_forecast(data.sample(frac=1, random_state=1))
        pd.testing.assert_frame_equal(tsf.forecast(), reference)

    def test_sorted_dataset_does_not_warn(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quick_pipeline().fit_forecast(data)


class TestAdvisoryWarnings:
    def test_extra_columns_are_reported(self, data):
        data["exogenous"] = np.arange(len(data))
        with pytest.warns(UserWarning, match="exogenous"):
            quick_pipeline().fit_forecast(data)

    def test_granularity_mismatch_is_reported(self):
        monthly = TimeSeriesDatasetGenerator.generate(n_samples=200, granularity="1mo")
        with pytest.warns(UserWarning, match="looks like granularity '1mo'"):
            quick_pipeline(granularity="1d").fit_forecast(monthly)

    @pytest.mark.parametrize("granularity", ["1m", "30m", "1h", "1d", "1wk", "1mo"])
    def test_matching_granularity_does_not_warn(self, granularity):
        series = TimeSeriesDatasetGenerator.generate(n_samples=200, granularity=granularity)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            quick_pipeline(granularity=granularity).fit_forecast(series)

    def test_month_end_series_matches_monthly_granularity(self):
        series = pd.DataFrame(
            {
                "Date": pd.date_range("2005-01-31", periods=200, freq="ME"),
                "y": np.sin(np.arange(200) / 6.0) * 10 + 100,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tsf = quick_pipeline(granularity="1mo").fit_forecast(series)
        assert tsf.forecast()["Date"].dt.is_month_end.all()


class TestConfigurationValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train_size": 5},
            {"train_size": 0.1},
            {"lags": 0},
            {"horizon": -1},
            {"sliding_size": 0},
            {"metric": "RMSE"},
            {"granularity": "fortnightly"},
            {"scaler": "quantile"},
            {"models": []},
            {"hparameters": ["not", "a", "dict"]},
        ],
    )
    def test_invalid_constructor_arguments(self, kwargs):
        with pytest.raises(ConfigurationError):
            TSForecasting(**kwargs)

    def test_configuration_errors_are_value_errors(self):
        with pytest.raises(ValueError):
            TSForecasting(train_size=5)

    def test_unknown_model_lists_available_models(self):
        with pytest.raises(ModelNotFoundError, match="Available models: RandomForest"):
            TSForecasting(models=["Prophet"])

    @pytest.mark.parametrize(
        "given, expected",
        [
            ("xgboost", "XGBoost"),
            ("CatBoost", "Catboost"),
            ("knn", "KNN"),
            ("randomforest", "RandomForest"),
        ],
    )
    def test_model_names_are_case_insensitive(self, given, expected):
        assert TSForecasting(models=[given]).models == [expected]

    def test_duplicate_models_are_collapsed(self):
        assert TSForecasting(models=["KNN", "knn", "GBR"]).models == ["KNN", "GBR"]

    def test_aliases_are_accepted(self):
        tsf = TSForecasting(granularity="monthly", metric="mape", scaler="MinMax")
        assert tsf.granularity == "1mo"
        assert tsf.metric == "MAPE"
        assert tsf.get_config().preprocessing.scaler.value == "minmax"

    @pytest.mark.parametrize(
        "method, value",
        [
            ("with_train_size", 5),
            ("with_lags", -3),
            ("with_horizon", 0),
            ("with_sliding_size", 0),
            ("with_metric", "RMSE"),
            ("with_granularity", "yearly"),
            ("with_scaler", "quantile"),
        ],
    )
    def test_builder_setters_validate(self, method, value):
        with pytest.raises(ConfigurationError):
            getattr(TSForecastingBuilder(), method)(value)

    def test_pipeline_setters_validate(self):
        with pytest.raises(ConfigurationError):
            TSForecasting().with_train_size(5)

    def test_builder_rejects_unknown_model(self):
        with pytest.raises(ModelNotFoundError):
            TSForecastingBuilder().with_models(["Prophet"])

    def test_builder_propagates_verbose(self):
        tsf = TSForecastingBuilder().with_models(["KNN"]).with_verbose(False).build()
        assert tsf.verbose is False


class TestHyperparameters:
    def test_defaults_are_not_shared_state(self):
        """Editing the returned dict (as the README does) must not alter the registry."""
        hparameters = model_configurations()
        hparameters["RandomForest"]["n_estimators"] = 7

        assert model_configurations()["RandomForest"]["n_estimators"] == 100
        assert ModelRegistry.get("RandomForest").get_params()["n_estimators"] == 100

    def test_custom_hyperparameters_reach_the_model(self, data):
        tsf = quick_pipeline(hparameters={"KNN": {"n_neighbors": 3}})
        tsf.fit_forecast(data)
        assert tsf.get_model("KNN").get_params()["n_neighbors"] == 3

    def test_hyperparameter_keys_are_case_insensitive(self, data):
        tsf = quick_pipeline(models=["Catboost"], hparameters={"CatBoost": {"iterations": 5}})
        tsf.fit_forecast(data)
        assert tsf.get_model("Catboost").get_params()["iterations"] == 5


class TestCustomModels:
    def test_registered_model_can_be_used_in_pipeline(self, data, counting_model):
        tsf = quick_pipeline(models=[counting_model, "KNN"])
        tsf.fit_forecast(data)
        assert set(tsf.history().leaderboard["Model"]) == {"Counting", "KNN"}
        assert len(tsf.forecast()) == 3

    def test_unregistered_model_is_rejected(self, counting_model):
        ModelRegistry.unregister(counting_model)
        try:
            with pytest.raises(ModelNotFoundError):
                TSForecasting(models=[counting_model])
        finally:
            ModelRegistry.register(counting_model, CountingForecaster)


class TestPipelineState:
    def test_changing_configuration_invalidates_fit(self, data):
        tsf = quick_pipeline().fit_forecast(data)
        tsf.with_horizon(7)
        assert not tsf.is_fitted
        with pytest.raises(NotFittedError):
            tsf.forecast()
        assert len(tsf.fit_forecast(data).forecast()) == 7

    def test_history_returns_copies(self, data):
        tsf = quick_pipeline().fit_forecast(data)
        history = tsf.history()
        history.predictions["Model"] = "tampered"
        history.performance_complete["Model"] = "tampered"
        assert set(tsf.history().predictions["Model"]) == {"KNN"}
        assert set(tsf.history().performance_complete["Model"]) == {"KNN"}

    def test_refit_replaces_previous_results(self, data):
        tsf = quick_pipeline().fit_forecast(data)
        first = tsf.forecast()
        tsf.fit_forecast(data.iloc[:150])
        second = tsf.forecast()
        assert second["Date"].iloc[0] < first["Date"].iloc[0]

    def test_verbose_false_is_silent(self, data, capsys):
        quick_pipeline(models=["KNN", "GBR"]).fit_forecast(data).forecast()
        assert capsys.readouterr().out == ""

    def test_verbose_true_reports_progress(self, data, capsys):
        quick_pipeline(verbose=True).fit_forecast(data)
        output = capsys.readouterr().out
        assert "Algorithm Evaluation: KNN" in output
        assert "best performance" in output


class TestForecast:
    def test_final_fit_is_cached(self, data, counting_model):
        tsf = quick_pipeline(models=[counting_model]).fit_forecast(data)
        model = tsf.get_model(counting_model)
        fits_after_evaluation = model.n_fits

        results = [tsf.forecast(interval_method=m) for m in ("ensemble", "quantile", "conformal")]
        results.append(tsf.forecast(dataset=data, interval_method="gaussian"))

        assert model.n_fits == fits_after_evaluation + 1
        for result in results[1:]:
            pd.testing.assert_series_equal(result["y"], results[0]["y"])

    def test_new_dataset_triggers_refit(self, data, counting_model):
        tsf = quick_pipeline(models=[counting_model]).fit_forecast(data.iloc[:150])
        model = tsf.get_model(counting_model)
        tsf.forecast()
        fits = model.n_fits

        forecast = tsf.forecast(dataset=data)
        assert model.n_fits == fits + 1
        assert forecast["Date"].iloc[0] == data["Date"].max() + pd.Timedelta(days=1)

    def test_cached_forecast_is_not_mutable(self, data):
        tsf = quick_pipeline().fit_forecast(data)
        first = tsf.forecast()
        first["y"] = 0.0
        assert (tsf.forecast()["y"] != 0.0).all()

    def test_unknown_interval_method(self, data):
        tsf = quick_pipeline().fit_forecast(data)
        with pytest.raises(ValueError, match="Unknown interval method"):
            tsf.forecast(interval_method="bootstrap")

    @pytest.mark.parametrize("method", ["quantile", "conformal", "gaussian", "ensemble"])
    def test_intervals_are_nested_and_ordered(self, data, method):
        forecast = quick_pipeline().fit_forecast(data).forecast(interval_method=method)
        levels = [80, 90, 95, 99]
        for narrow, wide in zip(levels[:-1], levels[1:]):
            assert (forecast[f"y_lower_{wide}"] <= forecast[f"y_lower_{narrow}"] + 1e-9).all()
            assert (forecast[f"y_upper_{wide}"] >= forecast[f"y_upper_{narrow}"] - 1e-9).all()
        for level in levels:
            assert (forecast[f"y_lower_{level}"] <= forecast[f"y_upper_{level}"]).all()

    def test_forecast_respects_disabled_datetime_features(self, data):
        tsf = quick_pipeline(datetime_features=False).fit_forecast(data)
        assert not [c for c in tsf._evaluation_engine.input_cols if c.startswith("Date")]
        forecast = tsf.forecast()
        assert len(forecast) == 3
        assert np.isfinite(forecast["y"]).all()

    def test_timezone_aware_dates(self, data):
        data["Date"] = data["Date"].dt.tz_localize("UTC")
        forecast = quick_pipeline().fit_forecast(data).forecast()
        assert forecast["Date"].iloc[0] == pd.Timestamp("2020-07-19")
