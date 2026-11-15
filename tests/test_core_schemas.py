"""Tests for tsforecasting.core.schemas module."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from tsforecasting.core.enums import EvaluationMetric, Granularity, ModelName, ScalerType
from tsforecasting.core.schemas import (
    EvaluationResult,
    ForecastConfig,
    ForecastOutput,
    ModelConfig,
    PerformanceHistory,
    PipelineConfig,
    PreprocessingConfig,
    WindowResult,
)


class TestForecastConfig:
    def test_defaults(self):
        config = ForecastConfig()
        assert config.train_size == 0.8
        assert config.lags == 7
        assert config.horizon == 15
        assert config.sliding_size == 15
        assert config.granularity == Granularity.DAILY
        assert config.metric == EvaluationMetric.MAE

    def test_custom_values(self):
        config = ForecastConfig(train_size=0.7, lags=10, horizon=5)
        assert config.train_size == 0.7
        assert config.lags == 10
        assert config.horizon == 5

    def test_train_size_validation_too_low(self):
        with pytest.raises(PydanticValidationError):
            ForecastConfig(train_size=0.1)

    def test_train_size_validation_too_high(self):
        with pytest.raises(PydanticValidationError):
            ForecastConfig(train_size=1.0)

    def test_lags_must_be_positive(self):
        with pytest.raises(PydanticValidationError):
            ForecastConfig(lags=0)

    def test_horizon_must_be_positive(self):
        with pytest.raises(PydanticValidationError):
            ForecastConfig(horizon=0)


class TestPreprocessingConfig:
    def test_defaults(self):
        config = PreprocessingConfig()
        assert config.scaler == ScalerType.STANDARD
        assert config.datetime_features is True

    def test_custom(self):
        config = PreprocessingConfig(scaler=ScalerType.ROBUST, datetime_features=False)
        assert config.scaler == ScalerType.ROBUST
        assert config.datetime_features is False


class TestModelConfig:
    def test_basic(self):
        config = ModelConfig(name=ModelName.RANDOM_FOREST, params={"n_estimators": 200})
        assert config.name == ModelName.RANDOM_FOREST
        assert config.params["n_estimators"] == 200

    def test_empty_params(self):
        config = ModelConfig(name=ModelName.XGBOOST)
        assert config.params == {}


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert isinstance(config.forecast, ForecastConfig)
        assert isinstance(config.preprocessing, PreprocessingConfig)
        assert len(config.models) == 3

    def test_default_models(self):
        config = PipelineConfig()
        model_values = [m.value for m in config.models]
        assert "RandomForest" in model_values
        assert "GBR" in model_values
        assert "XGBoost" in model_values


class TestWindowResult:
    def test_frozen(self):
        wr = WindowResult(
            window=1,
            model="RF",
            horizon=3,
            y_true=(1.0, 2.0),
            y_pred=(1.1, 2.1),
            metrics={"mae": 0.1},
        )
        with pytest.raises(AttributeError):
            wr.window = 2

    def test_from_arrays(self):
        wr = WindowResult.from_arrays(
            window=1,
            model="RF",
            horizon=2,
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.1, 2.1]),
            metrics={"mae": 0.1},
        )
        assert wr.y_true == (1.0, 2.0)
        assert wr.y_pred == (1.1, 2.1)

    def test_array_properties(self):
        wr = WindowResult(
            window=1,
            model="RF",
            horizon=2,
            y_true=(1.0, 2.0),
            y_pred=(1.1, 2.1),
            metrics={"mae": 0.1},
        )
        np.testing.assert_array_equal(wr.y_true_array, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(wr.y_pred_array, np.array([1.1, 2.1]))


class TestEvaluationResult:
    def test_add_and_list(self):
        er = EvaluationResult()
        wr = WindowResult(
            window=1, model="RF", horizon=1, y_true=(1.0,), y_pred=(1.1,), metrics={"mae": 0.1}
        )
        er.add_result(wr)
        assert len(er.window_results) == 1

    def test_to_dataframe(self):
        er = EvaluationResult()
        er.add_result(
            WindowResult(
                window=1, model="RF", horizon=1, y_true=(1.0,), y_pred=(1.1,), metrics={"mae": 0.1}
            )
        )
        er.add_result(
            WindowResult(
                window=1, model="XGB", horizon=1, y_true=(1.0,), y_pred=(0.9,), metrics={"mae": 0.1}
            )
        )
        df = er.to_dataframe()
        assert len(df) == 2
        assert "Model" in df.columns
        assert "mae" in df.columns

    def test_models_property(self):
        er = EvaluationResult()
        er.add_result(
            WindowResult(window=1, model="RF", horizon=1, y_true=(1.0,), y_pred=(1.1,), metrics={})
        )
        er.add_result(
            WindowResult(window=1, model="XGB", horizon=1, y_true=(1.0,), y_pred=(0.9,), metrics={})
        )
        assert set(er.models) == {"RF", "XGB"}


class TestForecastOutput:
    def test_repr(self):
        fo = ForecastOutput(
            predictions=pd.DataFrame({"Date": [1, 2], "y": [10, 20]}),
            selected_model="RF",
        )
        assert "RF" in repr(fo)

    def test_to_dataframe_without_intervals(self):
        preds = pd.DataFrame({"Date": [1, 2], "y": [10, 20]})
        fo = ForecastOutput(predictions=preds, selected_model="RF")
        df = fo.to_dataframe()
        assert list(df.columns) == ["Date", "y"]

    def test_to_dataframe_with_intervals(self):
        preds = pd.DataFrame({"Date": [1, 2], "y": [10, 20]})
        intervals = pd.DataFrame({"Date": [1, 2], "lower": [8, 18], "upper": [12, 22]})
        fo = ForecastOutput(predictions=preds, selected_model="RF", confidence_intervals=intervals)
        df = fo.to_dataframe()
        assert "lower" in df.columns
        assert "upper" in df.columns


class TestPerformanceHistory:
    def test_to_dict(self):
        ph = PerformanceHistory(
            predictions=pd.DataFrame(),
            performance_complete=pd.DataFrame(),
            performance_by_horizon=pd.DataFrame(),
            leaderboard=pd.DataFrame({"Model": ["RF"]}),
            selected_model="RF",
        )
        d = ph.to_dict()
        assert "Predictions" in d
        assert "Leaderboard" in d

    def test_repr(self):
        ph = PerformanceHistory(
            predictions=pd.DataFrame(),
            performance_complete=pd.DataFrame(),
            performance_by_horizon=pd.DataFrame(),
            leaderboard=pd.DataFrame({"Model": ["RF", "XGB"]}),
            selected_model="RF",
        )
        assert "RF" in repr(ph)
        assert "2" in repr(ph)


class TestAssignmentValidation:
    """Constraints must hold when fields are changed after construction."""

    def test_forecast_config_rejects_invalid_assignment(self):
        config = ForecastConfig()
        with pytest.raises(PydanticValidationError):
            config.train_size = 5
        with pytest.raises(PydanticValidationError):
            config.lags = 0
        assert config.train_size == 0.8
        assert config.lags == 7

    def test_forecast_config_parses_strings(self):
        config = ForecastConfig(granularity="monthly", metric="MAPE")
        assert config.granularity == Granularity.MONTHLY
        assert config.metric == EvaluationMetric.MAPE
        config.granularity = "1h"
        assert config.granularity == Granularity.HOURLY

    def test_preprocessing_config_parses_strings(self):
        config = PreprocessingConfig(scaler="MinMax")
        assert config.scaler == ScalerType.MINMAX
        with pytest.raises(PydanticValidationError):
            config.scaler = "quantile"


class TestPipelineConfigModels:
    def test_strings_become_enum_members(self):
        config = PipelineConfig(models=["KNN", "xgboost"])
        assert config.models == [ModelName.KNN, ModelName.XGBOOST]
        assert config.selected_models == ["KNN", "XGBoost"]

    def test_custom_names_are_kept_as_strings(self):
        config = PipelineConfig(models=["KNN", "MyModel"])
        assert config.models == [ModelName.KNN, "MyModel"]
        assert config.selected_models == ["KNN", "MyModel"]

    def test_single_name_and_duplicates(self):
        assert PipelineConfig(models="KNN").selected_models == ["KNN"]
        assert PipelineConfig(models=["KNN", "KNN", "GBR"]).selected_models == ["KNN", "GBR"]

    def test_empty_models_rejected(self):
        with pytest.raises(PydanticValidationError):
            PipelineConfig(models=[])

    def test_models_validated_on_assignment(self):
        config = PipelineConfig()
        config.models = ["catboost"]
        assert config.models == [ModelName.CATBOOST]
        with pytest.raises(PydanticValidationError):
            config.models = []
