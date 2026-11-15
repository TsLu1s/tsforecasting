"""Tests for tsforecasting.core.enums module."""

import pytest

from tsforecasting.core.enums import (
    EvaluationMetric,
    Granularity,
    ModelName,
    ScalerType,
    TimeSeriesPattern,
    TrendType,
)


class TestGranularity:
    def test_values(self):
        assert Granularity.MINUTE.value == "1m"
        assert Granularity.HALF_HOUR.value == "30m"
        assert Granularity.HOURLY.value == "1h"
        assert Granularity.DAILY.value == "1d"
        assert Granularity.WEEKLY.value == "1wk"
        assert Granularity.MONTHLY.value == "1mo"

    def test_from_string_primary(self):
        assert Granularity.from_string("1d") == Granularity.DAILY
        assert Granularity.from_string("1h") == Granularity.HOURLY
        assert Granularity.from_string("1mo") == Granularity.MONTHLY

    def test_from_string_aliases(self):
        assert Granularity.from_string("daily") == Granularity.DAILY
        assert Granularity.from_string("day") == Granularity.DAILY
        assert Granularity.from_string("hourly") == Granularity.HOURLY
        assert Granularity.from_string("weekly") == Granularity.WEEKLY
        assert Granularity.from_string("monthly") == Granularity.MONTHLY
        assert Granularity.from_string("minute") == Granularity.MINUTE

    def test_from_string_case_insensitive(self):
        assert Granularity.from_string("DAILY") == Granularity.DAILY
        assert Granularity.from_string("Daily") == Granularity.DAILY

    def test_from_string_passthrough_enum(self):
        assert Granularity.from_string(Granularity.DAILY) == Granularity.DAILY

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid granularity"):
            Granularity.from_string("invalid")

    def test_to_timedelta_kwargs(self):
        assert Granularity.to_timedelta_kwargs(Granularity.DAILY) == {"days": 1}
        assert Granularity.to_timedelta_kwargs(Granularity.HOURLY) == {"hours": 1}
        assert Granularity.to_timedelta_kwargs(Granularity.WEEKLY) == {"weeks": 1}
        assert Granularity.to_timedelta_kwargs(Granularity.MINUTE) == {"minutes": 1}

    def test_is_string_enum(self):
        assert isinstance(Granularity.DAILY, str)
        assert Granularity.DAILY == "1d"


class TestEvaluationMetric:
    def test_values(self):
        assert EvaluationMetric.MAE.value == "mae"
        assert EvaluationMetric.MAPE.value == "mape"
        assert EvaluationMetric.MSE.value == "mse"

    def test_display_name(self):
        assert EvaluationMetric.MAE.display_name == "Mean Absolute Error"
        assert EvaluationMetric.MAPE.display_name == "Mean Absolute Percentage Error"
        assert EvaluationMetric.MSE.display_name == "Mean Squared Error"


class TestModelName:
    def test_all_models_present(self):
        names = [m.value for m in ModelName]
        assert "RandomForest" in names
        assert "ExtraTrees" in names
        assert "GBR" in names
        assert "KNN" in names
        assert "GeneralizedLR" in names
        assert "XGBoost" in names
        assert "Catboost" in names
        assert "AutoGluon" in names

    def test_count(self):
        assert len(ModelName) == 8


class TestTimeSeriesPattern:
    def test_from_string_primary(self):
        assert TimeSeriesPattern.from_string("trend") == TimeSeriesPattern.TREND
        assert TimeSeriesPattern.from_string("seasonal") == TimeSeriesPattern.SEASONAL
        assert TimeSeriesPattern.from_string("mixed") == TimeSeriesPattern.MIXED

    def test_from_string_aliases(self):
        assert TimeSeriesPattern.from_string("rw") == TimeSeriesPattern.RANDOM_WALK
        assert TimeSeriesPattern.from_string("randomwalk") == TimeSeriesPattern.RANDOM_WALK
        assert TimeSeriesPattern.from_string("regime") == TimeSeriesPattern.REGIME_CHANGE
        assert TimeSeriesPattern.from_string("structural_break") == TimeSeriesPattern.REGIME_CHANGE

    def test_from_string_passthrough(self):
        assert TimeSeriesPattern.from_string(TimeSeriesPattern.TREND) == TimeSeriesPattern.TREND

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid pattern"):
            TimeSeriesPattern.from_string("nonexistent")


class TestTrendType:
    def test_from_string(self):
        assert TrendType.from_string("linear") == TrendType.LINEAR
        assert TrendType.from_string("exponential") == TrendType.EXPONENTIAL
        assert TrendType.from_string("lin") == TrendType.LINEAR
        assert TrendType.from_string("exp") == TrendType.EXPONENTIAL

    def test_from_string_passthrough(self):
        assert TrendType.from_string(TrendType.LINEAR) == TrendType.LINEAR

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid trend_type"):
            TrendType.from_string("quadratic")


class TestScalerType:
    def test_values(self):
        assert ScalerType.STANDARD.value == "standard"
        assert ScalerType.MINMAX.value == "minmax"
        assert ScalerType.ROBUST.value == "robust"


class TestFlexibleParsing:
    @pytest.mark.parametrize("value", ["xgboost", "XGBOOST", " XGBoost "])
    def test_model_name_is_case_insensitive(self, value):
        assert ModelName(value) is ModelName.XGBOOST

    def test_model_name_common_spelling(self):
        assert ModelName("CatBoost") is ModelName.CATBOOST

    def test_model_name_unknown(self):
        with pytest.raises(ValueError):
            ModelName("Prophet")

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("MAE", EvaluationMetric.MAE),
            ("mape", EvaluationMetric.MAPE),
            ("Mean Squared Error", EvaluationMetric.MSE),
            (EvaluationMetric.MAE, EvaluationMetric.MAE),
        ],
    )
    def test_metric_from_string(self, value, expected):
        assert EvaluationMetric.from_string(value) is expected

    def test_metric_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid metric"):
            EvaluationMetric.from_string("RMSE")

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("standard", ScalerType.STANDARD),
            ("MinMax", ScalerType.MINMAX),
            ("min_max", ScalerType.MINMAX),
            ("RobustScaler", ScalerType.ROBUST),
            (ScalerType.ROBUST, ScalerType.ROBUST),
        ],
    )
    def test_scaler_from_string(self, value, expected):
        assert ScalerType.from_string(value) is expected

    def test_scaler_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid scaler"):
            ScalerType.from_string("quantile")

    def test_monthly_offset_kwargs(self):
        assert Granularity.to_timedelta_kwargs(Granularity.MONTHLY) == {"months": 1}

    def test_approx_seconds_are_increasing(self):
        seconds = [g.approx_seconds for g in Granularity]
        assert seconds == sorted(seconds)
        assert Granularity.DAILY.approx_seconds == 86400.0
