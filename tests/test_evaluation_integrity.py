"""Tests guarding the statistical integrity of the expanding-window evaluation.

These cover look-ahead leakage, window planning, per-model failure isolation
and the AutoGluon wrapper (through a stub, so AutoGluon itself is not required).
"""

import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from tsforecasting.core.exceptions import InsufficientDataError, ModelFitError
from tsforecasting.core.schemas import ForecastConfig, PreprocessingConfig
from tsforecasting.evaluation.engine import EvaluationEngine
from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.models.registry import ModelRegistry
from tsforecasting.pipeline.forecasting import TSForecasting

START = pd.Timestamp("2020-01-01")


def index_series(n: int) -> pd.DataFrame:
    """Daily series where y equals its own time index (0, 1, 2, ...)."""
    return pd.DataFrame(
        {"Date": pd.date_range(START, periods=n, freq="D"), "y": np.arange(n, dtype=float)}
    )


class SpyForecaster(BaseForecaster):
    """Records what it is trained on and predicts zeros."""

    def __init__(self, **params):
        super().__init__(**params)
        self.max_target_seen = []
        self.train_rows = []
        self.n_fits = 0

    def _create_base_estimator(self):
        return None

    def fit(self, X, y):
        self.n_fits += 1
        self.max_target_seen.append(float(np.max(y)))
        self.train_rows.append(len(y))
        self._n_outputs = y.shape[1]
        self._is_fitted = True
        return self

    def predict(self, X):
        return np.zeros((len(X), self._n_outputs))


class FailingForecaster(BaseForecaster):
    def _create_base_estimator(self):
        return None

    def fit(self, X, y):
        raise RuntimeError("boom")


def make_engine(lags=4, horizon=5, sliding_size=5, train_size=0.6, datetime_features=True):
    return EvaluationEngine(
        config=ForecastConfig(
            train_size=train_size, lags=lags, horizon=horizon, sliding_size=sliding_size
        ),
        preprocessing=PreprocessingConfig(datetime_features=datetime_features),
        verbose=False,
    )


class TestNoLookAheadLeakage:
    @pytest.mark.parametrize("horizon", [1, 3, 5, 8])
    @pytest.mark.parametrize("datetime_features", [True, False])
    def test_training_targets_never_exceed_forecast_origin(self, horizon, datetime_features):
        """Every target used for training must already be observed at the forecast origin."""
        engine = make_engine(
            horizon=horizon, sliding_size=horizon, datetime_features=datetime_features
        )
        spy = SpyForecaster()
        predictions = engine.evaluate(index_series(120), {"Spy": spy})

        windows = sorted(predictions["Window"].unique())
        assert len(windows) == len(spy.max_target_seen) > 1

        for window, max_target in zip(windows, spy.max_target_seen):
            first_origin = predictions.loc[predictions["Window"] == window, "Date"].iloc[0]
            origin_index = (pd.Timestamp(first_origin) - START).days
            # y equals the time index, so the largest training target is the latest
            # observation the model saw. It must not lie after the forecast origin...
            assert max_target <= origin_index
            # ...and the purge must not discard more history than necessary.
            assert max_target == origin_index

    def test_training_set_grows_by_sliding_size(self):
        engine = make_engine(horizon=3, sliding_size=7)
        spy = SpyForecaster()
        engine.evaluate(index_series(150), {"Spy": spy})
        assert len(spy.train_rows) >= 2
        assert set(np.diff(spy.train_rows)) == {7}

    def test_each_window_evaluates_horizon_rows(self):
        engine = make_engine(horizon=4, sliding_size=4)
        predictions = engine.evaluate(index_series(120), {"Spy": SpyForecaster()})
        assert (predictions.groupby("Window").size() == 4).all()

    def test_actuals_in_predictions_are_future_values(self):
        """y_horizon_k of a row dated t must be the observation at t + k."""
        engine = make_engine(horizon=3, sliding_size=3)
        predictions = engine.evaluate(index_series(100), {"Spy": SpyForecaster()})
        origin = (pd.to_datetime(predictions["Date"]) - START).dt.days.to_numpy()
        for k in (1, 2, 3):
            np.testing.assert_array_equal(predictions[f"y_horizon_{k}"].to_numpy(), origin + k)

    def test_pipeline_forecast_uses_all_complete_rows(self):
        """The final fit keeps every row whose targets are fully observed."""
        ModelRegistry.register("Spy", SpyForecaster)
        try:
            n, lags, horizon = 100, 4, 5
            tsf = TSForecasting(
                lags=lags, horizon=horizon, sliding_size=5, models=["Spy"], verbose=False
            )
            tsf.fit_forecast(index_series(n))
            spy = tsf.get_model("Spy")
            fits_during_evaluation = spy.n_fits
            tsf.forecast()
            assert spy.n_fits == fits_during_evaluation + 1
            assert spy.train_rows[-1] == n - lags - horizon + 1
            assert spy.max_target_seen[-1] == n - 1
        finally:
            ModelRegistry.unregister("Spy")


class TestWindowPlanning:
    def test_insufficient_data_raises_clear_error(self):
        engine = make_engine(lags=7, horizon=15, sliding_size=15, train_size=0.8)
        with pytest.raises(InsufficientDataError, match="Insufficient data for evaluation"):
            engine.evaluate(index_series(30), {"Spy": SpyForecaster()})

    @pytest.mark.parametrize(
        "n, lags, horizon, sliding_size, train_size",
        [(40, 5, 10, 2, 0.8), (60, 5, 15, 3, 0.9)],
    )
    def test_no_valid_window_raises_instead_of_keyerror(
        self, n, lags, horizon, sliding_size, train_size
    ):
        """sliding_size < horizon used to end in ``KeyError: 'Model'``."""
        tsf = TSForecasting(
            lags=lags,
            horizon=horizon,
            sliding_size=sliding_size,
            train_size=train_size,
            models=["KNN"],
            verbose=False,
        )
        with pytest.raises(InsufficientDataError):
            tsf.fit_forecast(index_series(n))

    def test_series_shorter_than_lags(self):
        tsf = TSForecasting(lags=7, models=["KNN"], verbose=False)
        with pytest.raises(InsufficientDataError, match="lags length"):
            tsf.fit_forecast(index_series(5))

    def test_insufficient_data_is_value_error(self):
        assert issubclass(InsufficientDataError, ValueError)


class TestModelFailureIsolation:
    def test_failing_model_is_skipped(self):
        engine = make_engine()
        with pytest.warns(RuntimeWarning, match="Broken"):
            predictions = engine.evaluate(
                index_series(120), {"Broken": FailingForecaster(), "Spy": SpyForecaster()}
            )
        assert set(predictions["Model"]) == {"Spy"}
        assert "boom" in engine.failed_models["Broken"]

    def test_all_models_failing_raises(self):
        engine = make_engine()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ModelFitError, match="No model could be evaluated"):
                engine.evaluate(index_series(120), {"Broken": FailingForecaster()})

    def test_non_finite_predictions_are_rejected(self):
        class NaNForecaster(SpyForecaster):
            def predict(self, X):
                return np.full((len(X), self._n_outputs), np.nan)

        engine = make_engine()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ModelFitError, match="non-finite"):
                engine.evaluate(index_series(120), {"NaN": NaNForecaster()})

    def test_pipeline_continues_when_one_model_fails(self):
        """GeneralizedLR (Poisson deviance) cannot fit negative targets."""
        data = index_series(160)
        data["y"] = np.sin(np.arange(160) / 5.0) * 10  # crosses zero
        tsf = TSForecasting(
            lags=5, horizon=3, sliding_size=10, models=["KNN", "GeneralizedLR"], verbose=False
        )
        with pytest.warns(RuntimeWarning, match="GeneralizedLR"):
            tsf.fit_forecast(data)
        assert tsf.selected_model == "KNN"
        assert list(tsf.failed_models) == ["GeneralizedLR"]
        assert set(tsf.history().leaderboard["Model"]) == {"KNN"}
        assert len(tsf.forecast()) == 3


class _StubPredictor:
    """Stand-in for autogluon.tabular.TabularPredictor recording the columns it sees."""

    instances = []

    def __init__(self, label, **kwargs):
        self.label = label
        self.fit_columns = None
        self.predict_columns = []
        _StubPredictor.instances.append(self)

    def fit(self, frame, **kwargs):
        self.fit_columns = list(frame.columns)
        self._mean = float(frame[self.label].mean())
        return self

    def predict(self, frame):
        self.predict_columns.append(list(frame.columns))
        return pd.Series(np.full(len(frame), self._mean), index=frame.index)


@pytest.fixture
def stub_autogluon(monkeypatch):
    _StubPredictor.instances = []
    tabular = types.ModuleType("autogluon.tabular")
    tabular.TabularPredictor = _StubPredictor
    package = types.ModuleType("autogluon")
    package.tabular = tabular
    monkeypatch.setitem(sys.modules, "autogluon", package)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", tabular)
    return _StubPredictor


class TestAutoGluonForecaster:
    def test_each_horizon_only_sees_its_own_target(self, stub_autogluon):
        X = pd.DataFrame({"y_lag_1": np.arange(10.0), "y_lag_2": np.arange(10.0) + 1})
        y = pd.DataFrame({"y_horizon_1": np.arange(10.0) + 2, "y_horizon_2": np.arange(10.0) + 3})
        model = AutoGluonForecaster()
        model.show_progress = False
        model.fit(X, y)

        assert len(stub_autogluon.instances) == 2
        for predictor in stub_autogluon.instances:
            other_targets = [
                c
                for c in predictor.fit_columns
                if c.startswith("y_horizon_") and c != predictor.label
            ]
            assert other_targets == []
            assert predictor.label in predictor.fit_columns

    def test_legacy_single_frame_call_drops_other_targets(self, stub_autogluon):
        frame = pd.DataFrame(
            {
                "y_lag_1": np.arange(10.0),
                "y_horizon_1": np.arange(10.0) + 1,
                "y_horizon_2": np.arange(10.0) + 2,
            }
        )
        model = AutoGluonForecaster(labels=["y_horizon_1", "y_horizon_2"])
        model.show_progress = False
        model.fit(frame)
        for predictor in stub_autogluon.instances:
            assert set(predictor.fit_columns) == {"y_lag_1", predictor.label}

    def test_predict_ignores_target_columns(self, stub_autogluon):
        frame = pd.DataFrame(
            {
                "y_lag_1": np.arange(10.0),
                "y_horizon_1": np.arange(10.0) + 1,
                "y_horizon_2": np.arange(10.0) + 2,
            }
        )
        model = AutoGluonForecaster(labels=["y_horizon_1", "y_horizon_2"])
        model.show_progress = False
        model.fit(frame)
        predictions = model.predict(frame)

        assert list(predictions.columns) == ["y_horizon_1", "y_horizon_2"]
        for predictor in stub_autogluon.instances:
            assert predictor.predict_columns == [["y_lag_1"]]

    def test_missing_labels_raise(self, stub_autogluon):
        model = AutoGluonForecaster()
        with pytest.raises(ValueError, match="labels must be provided"):
            model.fit(pd.DataFrame({"y_lag_1": [1.0, 2.0]}))

    def test_pipeline_end_to_end_is_leak_free(self, stub_autogluon):
        tsf = TSForecasting(lags=4, horizon=3, sliding_size=6, models=["AutoGluon"], verbose=False)
        tsf.fit_forecast(index_series(120))
        forecast = tsf.forecast()

        assert tsf.selected_model == "AutoGluon"
        assert len(forecast) == 3
        assert np.isfinite(forecast["y"]).all()
        for predictor in stub_autogluon.instances:
            assert not [
                c
                for c in predictor.fit_columns
                if c.startswith("y_horizon_") and c != predictor.label
            ]
            for columns in predictor.predict_columns:
                assert not [c for c in columns if c.startswith("y_horizon_")]

    def test_evaluation_fits_autogluon_once(self, stub_autogluon):
        tsf = TSForecasting(lags=4, horizon=3, sliding_size=6, models=["AutoGluon"], verbose=False)
        tsf.fit_forecast(index_series(120))
        n_windows = tsf.history().predictions["Window"].nunique()
        assert n_windows > 1
        assert len(stub_autogluon.instances) == 3  # one predictor per horizon, fitted once
