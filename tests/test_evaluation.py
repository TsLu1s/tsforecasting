"""Tests for tsforecasting.evaluation module (metrics + intervals)."""

import numpy as np
import pandas as pd
import pytest

from tsforecasting.core.enums import EvaluationMetric
from tsforecasting.evaluation.intervals import (
    ConformalIntervalEstimator,
    EnsembleIntervalEstimator,
    GaussianIntervalEstimator,
    IntervalEstimatorFactory,
    IntervalMethod,
    QuantileIntervalEstimator,
    compute_intervals,
)
from tsforecasting.evaluation.metrics import (
    METRIC_STRATEGIES,
    MAEStrategy,
    MAPEStrategy,
    MaxErrorStrategy,
    MSEStrategy,
    aggregate_performance,
    best_model,
    get_all_strategies,
    get_metric_strategy,
    metrics_regression,
    resolve_metric_column,
    vertical_performance,
)


class TestMetricStrategies:
    def test_mae(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        strategy = MAEStrategy()
        assert strategy.compute(y_true, y_pred) == pytest.approx(0.5)
        assert strategy.key == "mae"
        assert strategy.name == "Mean Absolute Error"

    def test_mse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        strategy = MSEStrategy()
        assert strategy.compute(y_true, y_pred) == pytest.approx(1.0 / 3)

    def test_mape(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        strategy = MAPEStrategy()
        result = strategy.compute(y_true, y_pred)
        assert result == pytest.approx(7.5)  # (10/100 + 10/200) / 2 * 100

    def test_max_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        strategy = MaxErrorStrategy()
        assert strategy.compute(y_true, y_pred) == pytest.approx(2.0)


class TestMetricRegistry:
    def test_get_by_string(self):
        s = get_metric_strategy("mae")
        assert isinstance(s, MAEStrategy)

    def test_get_by_enum(self):
        s = get_metric_strategy(EvaluationMetric.MAE)
        assert isinstance(s, MAEStrategy)

    def test_get_all(self):
        strategies = get_all_strategies()
        assert len(strategies) == 4

    def test_all_keys_present(self):
        assert "mae" in METRIC_STRATEGIES
        assert "mape" in METRIC_STRATEGIES
        assert "mse" in METRIC_STRATEGIES
        assert "max_error" in METRIC_STRATEGIES


class TestMetricsRegression:
    def test_returns_all_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        result = metrics_regression(y_true, y_pred)
        assert "Mean Absolute Error" in result
        assert "Mean Absolute Percentage Error" in result
        assert "Mean Squared Error" in result
        assert "Max Error" in result


class TestVerticalPerformance:
    def test_basic(self):
        df = pd.DataFrame(
            {
                "Model": ["RF", "RF"],
                "Window": [1, 1],
                "y_horizon_1": [10.0, 20.0],
                "y_forecast_horizon_1": [11.0, 19.0],
            }
        )
        result = vertical_performance(df, horizon=1)
        assert "Model" in result.columns
        assert "Horizon" in result.columns
        assert "Mean Absolute Error" in result.columns


class TestBestModel:
    def test_selects_best(self, capsys):
        df = pd.DataFrame(
            {
                "Model": ["RF", "RF", "XGB", "XGB"],
                "Window": [1, 1, 1, 1],
                "Horizon": [1, 2, 1, 2],
                "Mean Absolute Error": [1.0, 2.0, 0.5, 1.0],
            }
        )
        result = best_model(df, metric="MAE")
        assert result == "XGB"

    def test_with_mse(self, capsys):
        df = pd.DataFrame(
            {
                "Model": ["RF", "RF", "XGB", "XGB"],
                "Window": [1, 1, 1, 1],
                "Horizon": [1, 2, 1, 2],
                "Mean Squared Error": [4.0, 4.0, 1.0, 1.0],
            }
        )
        result = best_model(df, metric="MSE")
        assert result == "XGB"


class TestAggregatePerformance:
    def test_returns_both_keys(self):
        df = pd.DataFrame(
            {
                "Model": ["RF", "RF"],
                "Window": [1, 1],
                "Horizon": [1, 2],
                "Mean Absolute Error": [1.0, 2.0],
                "Mean Absolute Percentage Error": [5.0, 10.0],
                "Mean Squared Error": [1.0, 4.0],
                "Max Error": [1.5, 3.0],
            }
        )
        result = aggregate_performance(df, metric="MAE")
        assert "by_horizon" in result
        assert "leaderboard" in result


@pytest.fixture
def interval_data():
    """Historical actuals and predictions for interval testing."""
    np.random.seed(42)
    n, h = 30, 3
    y_true = np.random.randn(n, h) * 10 + 100
    y_pred = y_true + np.random.randn(n, h) * 2
    forecast = np.array([100.0, 101.0, 102.0])
    return y_true, y_pred, forecast


class TestQuantileInterval:
    def test_fit_predict(self, interval_data):
        y_true, y_pred, forecast = interval_data
        est = QuantileIntervalEstimator(coverage=0.90)
        est.fit(y_true, y_pred)
        lower, upper = est.predict(forecast)
        assert lower.shape == forecast.shape
        assert upper.shape == forecast.shape
        assert np.all(lower < upper)

    def test_not_fitted_raises(self):
        est = QuantileIntervalEstimator()
        with pytest.raises(RuntimeError, match="not fitted"):
            est.predict(np.array([1.0]))


class TestConformalInterval:
    def test_fit_predict(self, interval_data):
        y_true, y_pred, forecast = interval_data
        est = ConformalIntervalEstimator(coverage=0.90)
        est.fit(y_true, y_pred)
        lower, upper = est.predict(forecast)
        assert np.all(lower < upper)
        # Conformal intervals are symmetric
        np.testing.assert_array_almost_equal(forecast - lower, upper - forecast)


class TestGaussianInterval:
    def test_fit_predict(self, interval_data):
        y_true, y_pred, forecast = interval_data
        est = GaussianIntervalEstimator(coverage=0.90)
        est.fit(y_true, y_pred)
        lower, upper = est.predict(forecast)
        assert np.all(lower < upper)


class TestEnsembleInterval:
    def test_fit_predict(self, interval_data):
        y_true, y_pred, forecast = interval_data
        est = EnsembleIntervalEstimator(coverage=0.90)
        est.fit(y_true, y_pred)
        lower, upper = est.predict(forecast)
        assert np.all(lower < upper)


class TestIntervalEstimatorFactory:
    @pytest.mark.parametrize("method", ["quantile", "conformal", "gaussian", "ensemble"])
    def test_create_all_methods(self, method):
        est = IntervalEstimatorFactory.create(method, coverage=0.95)
        assert est is not None

    def test_create_from_enum(self):
        est = IntervalEstimatorFactory.create(IntervalMethod.QUANTILE, coverage=0.90)
        assert isinstance(est, QuantileIntervalEstimator)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            IntervalEstimatorFactory.create("invalid")

    def test_list_methods(self):
        methods = IntervalEstimatorFactory.list_methods()
        assert len(methods) == 4
        assert "quantile" in methods


class TestComputeIntervals:
    def test_convenience_function(self, interval_data):
        y_true, y_pred, forecast = interval_data
        lower, upper = compute_intervals(forecast, y_true, y_pred, method="quantile")
        assert np.all(lower < upper)


class TestIntervalCoverage:
    def test_invalid_coverage(self):
        with pytest.raises(ValueError, match="Coverage must be"):
            QuantileIntervalEstimator(coverage=0.0)
        with pytest.raises(ValueError, match="Coverage must be"):
            QuantileIntervalEstimator(coverage=1.0)
        with pytest.raises(ValueError, match="Coverage must be"):
            ConformalIntervalEstimator(coverage=-0.1)

    def test_wider_intervals_with_higher_coverage(self, interval_data):
        y_true, y_pred, forecast = interval_data
        est_90 = QuantileIntervalEstimator(coverage=0.90)
        est_99 = QuantileIntervalEstimator(coverage=0.99)
        est_90.fit(y_true, y_pred)
        est_99.fit(y_true, y_pred)
        l90, u90 = est_90.predict(forecast)
        l99, u99 = est_99.predict(forecast)
        width_90 = u90 - l90
        width_99 = u99 - l99
        assert np.all(width_99 >= width_90 - 1e-10)


class TestResolveMetricColumn:
    @pytest.mark.parametrize(
        "metric, column",
        [
            ("MAE", "Mean Absolute Error"),
            ("mape", "Mean Absolute Percentage Error"),
            (EvaluationMetric.MSE, "Mean Squared Error"),
            ("Max Error", "Max Error"),
            ("mean absolute error", "Mean Absolute Error"),
        ],
    )
    def test_known_metrics(self, metric, column):
        assert resolve_metric_column(metric) == column

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            resolve_metric_column("RMSE")


class TestAggregationSemantics:
    @pytest.fixture
    def results(self):
        return pd.DataFrame(
            {
                "Model": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "Window": [1, 1, 2, 2, 1, 1, 2, 2],
                "Horizon": [1, 2, 1, 2, 1, 2, 1, 2],
                "Mean Absolute Error": [1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5],
                "Mean Absolute Percentage Error": [10.0, 20.0, 30.0, 40.0, 5.0, 5.0, 5.0, 5.0],
                "Mean Squared Error": [1.0, 4.0, 9.0, 16.0, 0.25, 0.25, 0.25, 0.25],
                "Max Error": [2.0, 3.0, 9.0, 5.0, 1.0, 1.0, 1.0, 1.0],
            }
        )

    def test_leaderboard_is_ranked_best_first(self, results):
        leaderboard = aggregate_performance(results, metric="MAE")["leaderboard"]
        assert list(leaderboard["Model"]) == ["B", "A"]
        assert list(leaderboard.index) == [0, 1]
        assert leaderboard.loc[1, "Mean Absolute Error"] == pytest.approx(2.5)

    def test_max_error_is_the_largest_error(self, results):
        aggregated = aggregate_performance(results, metric="MAE")
        leaderboard = aggregated["leaderboard"].set_index("Model")
        assert leaderboard.loc["A", "Max Error"] == 9.0
        by_horizon = aggregated["by_horizon"].set_index(["Model", "Horizon"])
        assert by_horizon.loc[("A", 1), "Max Error"] == 9.0
        assert by_horizon.loc[("A", 2), "Max Error"] == 5.0

    def test_best_model_agrees_with_leaderboard(self, results, capsys):
        for metric in ("MAE", "MAPE", "MSE"):
            leaderboard = aggregate_performance(results, metric=metric)["leaderboard"]
            assert best_model(results, metric=metric) == leaderboard["Model"].iloc[0]

    def test_best_model_can_be_silent(self, results, capsys):
        best_model(results, metric="MAE", verbose=False)
        assert capsys.readouterr().out == ""

    def test_vertical_performance_handles_uneven_windows(self):
        """Models evaluated on different windows must not break the aggregation."""
        df = pd.DataFrame(
            {
                "Model": ["A", "A", "B"],
                "Window": [1, 2, 1],
                "y_horizon_1": [10.0, 20.0, 10.0],
                "y_forecast_horizon_1": [11.0, 18.0, 10.5],
            }
        )
        result = vertical_performance(df, horizon=1)
        assert len(result) == 3
        assert list(result["Model"]) == ["A", "A", "B"]

    def test_vertical_performance_rejects_empty_input(self):
        with pytest.raises(ValueError, match="No forecasts"):
            vertical_performance(pd.DataFrame(), horizon=1)


class TestIntervalInputs:
    @pytest.mark.parametrize(
        "estimator_cls",
        [
            QuantileIntervalEstimator,
            ConformalIntervalEstimator,
            GaussianIntervalEstimator,
            EnsembleIntervalEstimator,
        ],
    )
    def test_one_dimensional_history_is_a_single_horizon(self, estimator_cls):
        rng = np.random.RandomState(0)
        y_true = rng.randn(200)
        y_pred = y_true + rng.randn(200)
        lower, upper = estimator_cls(coverage=0.9).fit(y_true, y_pred).predict(np.array([5.0]))
        width = float(upper[0] - lower[0])
        assert np.isfinite(width)
        assert 2.5 < width < 4.5  # ~2 * 1.645 for unit-variance residuals

    def test_rows_with_missing_values_are_ignored(self, interval_data):
        y_true, y_pred, forecast = interval_data
        clean = QuantileIntervalEstimator(0.9).fit(y_true, y_pred).predict(forecast)

        y_true_nan = np.vstack([y_true, [np.nan, 1.0, 1.0]])
        y_pred_nan = np.vstack([y_pred, [1.0, 1.0, 1.0]])
        with_nan = QuantileIntervalEstimator(0.9).fit(y_true_nan, y_pred_nan).predict(forecast)

        np.testing.assert_array_equal(clean[0], with_nan[0])
        np.testing.assert_array_equal(clean[1], with_nan[1])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            QuantileIntervalEstimator().fit(np.zeros((5, 2)), np.zeros((5, 3)))

    def test_no_history(self):
        with pytest.raises(ValueError, match="historical sample"):
            ConformalIntervalEstimator().fit(np.empty((0, 2)), np.empty((0, 2)))

    def test_forecast_with_wrong_number_of_horizons(self, interval_data):
        y_true, y_pred, _ = interval_data
        est = GaussianIntervalEstimator().fit(y_true, y_pred)
        with pytest.raises(ValueError, match="horizon"):
            est.predict(np.zeros(5))

    def test_two_dimensional_forecast(self, interval_data):
        y_true, y_pred, forecast = interval_data
        est = EnsembleIntervalEstimator().fit(y_true, y_pred)
        lower, upper = est.predict(np.tile(forecast, (4, 1)))
        assert lower.shape == upper.shape == (4, 3)

    def test_gaussian_single_sample_has_zero_width(self):
        est = GaussianIntervalEstimator().fit(np.array([[1.0, 2.0]]), np.array([[0.5, 2.5]]))
        lower, upper = est.predict(np.array([10.0, 10.0]))
        np.testing.assert_allclose(lower, upper)
        np.testing.assert_allclose(lower, [10.5, 9.5])

    @pytest.mark.parametrize(
        "coverage, z", [(0.80, 1.2816), (0.90, 1.6449), (0.95, 1.9600), (0.99, 2.5758)]
    )
    def test_gaussian_z_scores(self, coverage, z, interval_data):
        y_true, y_pred, _ = interval_data
        est = GaussianIntervalEstimator(coverage=coverage).fit(y_true, y_pred)
        assert est._z_score == pytest.approx(z, abs=1e-4)


class TestConformalCalibration:
    def test_quantile_is_an_observed_score(self):
        """Finite-sample validity needs a real score, not an interpolated one."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        est = ConformalIntervalEstimator(coverage=0.8).fit(scores, np.zeros(10))
        # ceil((10 + 1) * 0.8) = 9th smallest score
        assert est._conformity_quantiles[0] == 9.0

    def test_small_samples_fall_back_to_largest_score(self):
        scores = np.array([1.0, 5.0, 2.0])
        est = ConformalIntervalEstimator(coverage=0.99).fit(scores, np.zeros(3))
        assert est._conformity_quantiles[0] == 5.0

    @pytest.mark.parametrize("coverage", [0.8, 0.9, 0.95])
    def test_empirical_coverage_on_exchangeable_errors(self, coverage):
        """Split conformal intervals must cover at least the nominal rate on average."""
        rng = np.random.RandomState(123)
        hits, trials, n_calibration = 0, 400, 60
        for _ in range(trials):
            errors = rng.standard_t(df=3, size=n_calibration + 1)  # heavy tails
            est = ConformalIntervalEstimator(coverage=coverage)
            est.fit(errors[:n_calibration], np.zeros(n_calibration))
            lower, upper = est.predict(np.array([0.0]))
            hits += int(lower[0] <= errors[-1] <= upper[0])
        assert hits / trials >= coverage - 0.04


class TestFactoryMessages:
    def test_unknown_method_lists_options(self):
        with pytest.raises(
            ValueError, match="Available: \\['quantile', 'conformal', 'gaussian', 'ensemble'\\]"
        ):
            IntervalEstimatorFactory.create("bootstrap")

    def test_method_names_are_case_insensitive(self):
        assert isinstance(IntervalEstimatorFactory.create("Conformal"), ConformalIntervalEstimator)
