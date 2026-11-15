"""Tests for tsforecasting.pipeline.forecasting and core.base (Builder) modules."""

import numpy as np
import pandas as pd
import pytest

from tsforecasting.core.base import TSForecastingBuilder
from tsforecasting.core.enums import ScalerType
from tsforecasting.core.exceptions import DatasetError, NotFittedError
from tsforecasting.core.schemas import PerformanceHistory, PipelineConfig
from tsforecasting.data.generator import TimeSeriesDatasetGenerator
from tsforecasting.pipeline.forecasting import TSForecasting


class TestTSForecastingInit:
    def test_defaults(self):
        tsf = TSForecasting()
        assert tsf.train_size == 0.8
        assert tsf.lags == 7
        assert tsf.horizon == 15
        assert tsf.sliding_size == 15
        assert tsf.granularity == "1d"
        assert tsf.metric == "MAE"
        assert "RandomForest" in tsf.models

    def test_custom_params(self):
        tsf = TSForecasting(
            train_size=0.7,
            lags=10,
            horizon=5,
            sliding_size=5,
            models=["KNN", "GBR"],
            granularity="1mo",
            metric="MSE",
            scaler="robust",
            datetime_features=False,
        )
        assert tsf.train_size == 0.7
        assert tsf.lags == 10
        assert tsf.horizon == 5
        assert tsf.models == ["KNN", "GBR"]
        assert tsf.granularity == "1mo"
        assert tsf.metric == "MSE"

    def test_repr(self):
        tsf = TSForecasting(models=["RandomForest", "XGBoost"])
        r = repr(tsf)
        assert "TSForecasting" in r
        assert "RandomForest" in r

    def test_get_config(self):
        tsf = TSForecasting()
        config = tsf.get_config()
        assert isinstance(config, PipelineConfig)

    def test_builder_classmethod(self):
        builder = TSForecasting.builder()
        assert isinstance(builder, TSForecastingBuilder)


class TestTSForecastingValidation:
    def test_missing_date_column(self):
        df = pd.DataFrame({"y": [1, 2, 3]})
        tsf = TSForecasting()
        with pytest.raises(DatasetError, match="missing required columns"):
            tsf.fit_forecast(df)

    def test_missing_y_column(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=3)})
        tsf = TSForecasting()
        with pytest.raises(DatasetError, match="missing required columns"):
            tsf.fit_forecast(df)

    def test_non_datetime_date_column(self):
        df = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "y": [1.0, 2.0]})
        tsf = TSForecasting()
        with pytest.raises(DatasetError, match="datetime type"):
            tsf.fit_forecast(df)

    def test_all_nan_y(self):
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=5),
                "y": [np.nan] * 5,
            }
        )
        tsf = TSForecasting()
        with pytest.raises(DatasetError, match="NaN"):
            tsf.fit_forecast(df)


class TestTSForecastingNotFitted:
    def test_history_before_fit(self):
        tsf = TSForecasting()
        with pytest.raises(NotFittedError):
            tsf.history()

    def test_forecast_before_fit(self):
        tsf = TSForecasting()
        with pytest.raises(NotFittedError):
            tsf.forecast()

    def test_get_model_before_fit(self):
        tsf = TSForecasting()
        assert tsf.get_model("RandomForest") is None


@pytest.fixture
def fitted_pipeline():
    """Pre-fitted pipeline for test reuse."""
    data = TimeSeriesDatasetGenerator.generate(
        n_samples=200, granularity="1d", patterns=["mixed"], random_state=42
    )
    tsf = TSForecasting(
        train_size=0.8,
        lags=5,
        horizon=3,
        sliding_size=10,
        models=["KNN", "RandomForest"],
        metric="MAE",
        scaler="standard",
    )
    tsf.fit_forecast(data)
    return tsf, data


class TestTSForecastingFitForecast:
    def test_fit_marks_fitted(self, fitted_pipeline):
        tsf, _ = fitted_pipeline
        assert tsf._is_fitted

    def test_selected_model_set(self, fitted_pipeline):
        tsf, _ = fitted_pipeline
        assert tsf.selected_model in ["KNN", "RandomForest"]

    def test_history_returns_dataclass(self, fitted_pipeline):
        tsf, _ = fitted_pipeline
        history = tsf.history()
        assert isinstance(history, PerformanceHistory)
        assert isinstance(history.predictions, pd.DataFrame)
        assert isinstance(history.leaderboard, pd.DataFrame)
        assert history.selected_model in ["KNN", "RandomForest"]

    def test_history_leaderboard_has_all_models(self, fitted_pipeline):
        tsf, _ = fitted_pipeline
        history = tsf.history()
        models_in_lb = set(history.leaderboard["Model"].values)
        assert models_in_lb == {"KNN", "RandomForest"}

    def test_forecast_returns_dataframe(self, fitted_pipeline):
        tsf, data = fitted_pipeline
        forecast = tsf.forecast(data)
        assert isinstance(forecast, pd.DataFrame)
        assert "Date" in forecast.columns
        assert "y" in forecast.columns
        assert len(forecast) == 3  # horizon

    def test_forecast_has_intervals(self, fitted_pipeline):
        tsf, data = fitted_pipeline
        forecast = tsf.forecast(data, interval_method="ensemble")
        assert "y_lower_80" in forecast.columns
        assert "y_upper_80" in forecast.columns
        assert "y_lower_90" in forecast.columns
        assert "y_upper_90" in forecast.columns
        assert "y_lower_95" in forecast.columns
        assert "y_upper_95" in forecast.columns
        assert "y_lower_99" in forecast.columns
        assert "y_upper_99" in forecast.columns

    def test_interval_ordering(self, fitted_pipeline):
        tsf, data = fitted_pipeline
        forecast = tsf.forecast(data)
        # Wider coverage = wider intervals
        for i in range(len(forecast)):
            assert forecast["y_lower_99"].iloc[i] <= forecast["y_lower_90"].iloc[i]
            assert forecast["y_upper_99"].iloc[i] >= forecast["y_upper_90"].iloc[i]

    @pytest.mark.parametrize("method", ["quantile", "conformal", "gaussian", "ensemble"])
    def test_all_interval_methods(self, fitted_pipeline, method):
        tsf, data = fitted_pipeline
        forecast = tsf.forecast(data, interval_method=method)
        assert "y_lower_90" in forecast.columns
        assert "y_upper_90" in forecast.columns

    def test_get_model_after_fit(self, fitted_pipeline):
        tsf, _ = fitted_pipeline
        model = tsf.get_model("RandomForest")
        assert model is not None

    def test_fit_forecast_returns_self(self):
        data = TimeSeriesDatasetGenerator.generate(n_samples=200, random_state=42)
        tsf = TSForecasting(lags=5, horizon=3, sliding_size=10, models=["KNN"])
        result = tsf.fit_forecast(data)
        assert result is tsf

    def test_history_to_dict_backward_compat(self, fitted_pipeline):
        tsf, _ = fitted_pipeline
        history = tsf.history()
        d = history.to_dict()
        assert "Predictions" in d
        assert "Performance Complete" in d
        assert "Performance by Horizon" in d
        assert "Leaderboard" in d


class TestTSForecastingBuilder:
    def test_method_chaining(self):
        builder = TSForecastingBuilder()
        result = (
            builder.with_train_size(0.7)
            .with_lags(10)
            .with_horizon(5)
            .with_sliding_size(5)
            .with_granularity("1mo")
            .with_metric("mse")
            .with_models(["KNN", "GBR"])
            .with_scaler("robust")
            .with_datetime_features(False)
        )
        assert result is builder

    def test_build_creates_tsforecasting(self):
        tsf = (
            TSForecastingBuilder()
            .with_train_size(0.7)
            .with_lags(10)
            .with_horizon(5)
            .with_models(["KNN"])
            .build()
        )
        assert isinstance(tsf, TSForecasting)
        assert tsf.train_size == 0.7
        assert tsf.lags == 10
        assert tsf.horizon == 5
        assert tsf.models == ["KNN"]

    def test_get_config(self):
        builder = TSForecastingBuilder()
        builder.with_train_size(0.6)
        config = builder.get_config()
        assert isinstance(config, PipelineConfig)
        assert config.forecast.train_size == 0.6

    def test_with_hyperparameters(self):
        hparams = {"KNN": {"n_neighbors": 10}}
        builder = TSForecastingBuilder()
        builder.with_hyperparameters(hparams)
        assert builder._hparameters == hparams

    def test_with_hparameters_alias(self):
        hparams = {"KNN": {"n_neighbors": 10}}
        builder = TSForecastingBuilder()
        builder.with_hparameters(hparams)
        assert builder._hparameters == hparams

    def test_with_preprocessing(self):
        builder = TSForecastingBuilder()
        builder.with_preprocessing(scaler="minmax", datetime_features=False)
        config = builder.get_config()
        assert config.preprocessing.scaler == ScalerType.MINMAX
        assert config.preprocessing.datetime_features is False

    def test_builder_end_to_end(self):
        data = TimeSeriesDatasetGenerator.generate(n_samples=200, random_state=42)
        tsf = (
            TSForecastingBuilder()
            .with_train_size(0.8)
            .with_lags(5)
            .with_horizon(3)
            .with_sliding_size(10)
            .with_models(["KNN"])
            .build()
        )
        tsf.fit_forecast(data)
        assert tsf._is_fitted
        forecast = tsf.forecast(data)
        assert len(forecast) == 3
