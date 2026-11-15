from __future__ import annotations

import warnings
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from tsforecasting.core.enums import ScalerType
from tsforecasting.core.exceptions import InsufficientDataError, ModelFitError
from tsforecasting.core.schemas import ForecastConfig, PreprocessingConfig
from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.processing.processor import Processing

_SCALERS = {
    ScalerType.STANDARD: StandardScaler,
    ScalerType.MINMAX: MinMaxScaler,
    ScalerType.ROBUST: RobustScaler,
}

# Smallest training set accepted for an evaluation window.
_MIN_TRAIN_ROWS = 2


def get_scaler(scaler: ScalerType):
    """Instantiate the feature scaler for a ScalerType."""
    return _SCALERS[ScalerType.from_string(scaler)]()


class EvaluationEngine:
    """Handles expanding window evaluation logic.

    Manages the complete evaluation workflow including:
    - Time series transformation
    - Train/test splitting across windows
    - Feature scaling
    - Model fitting and prediction
    - Metric computation and aggregation

    Leakage control:
        A supervised row built at forecast origin ``t`` carries the targets
        ``y[t+1] ... y[t+horizon]``. When a window is evaluated from origin
        ``t0``, the ``horizon - 1`` rows immediately before it have targets
        that were not yet observed at ``t0``. Those rows are purged from the
        training set, so every window is fitted only on information that was
        available at its forecast origin, which is exactly what the final
        ``forecast()`` call can use.
    """

    def __init__(
        self,
        config: ForecastConfig,
        preprocessing: PreprocessingConfig,
        verbose: bool = True,
    ):
        """Initialize evaluation engine.

        Args:
            config: Forecasting configuration.
            preprocessing: Preprocessing configuration.
            verbose: Whether to print evaluation progress.
        """
        self._config = config
        self._preprocessing = preprocessing
        self._verbose = verbose
        self._processor = Processing()
        self._results: List[pd.DataFrame] = []
        self._input_cols: List[str] = []
        self._target_cols: List[str] = []
        self._failed_models: Dict[str, str] = {}
        self._fitted_once: Set[int] = set()
        self._n_iterations: int = 0

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def _get_scaler(self):
        """Get scaler based on configuration."""
        return get_scaler(self._preprocessing.scaler)

    def _scale(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit the scaler on the training features and apply it to both sets."""
        scaler = self._get_scaler()
        scaler.fit(train[self._input_cols])
        X_train = pd.DataFrame(
            scaler.transform(train[self._input_cols]), columns=self._input_cols, index=train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(test[self._input_cols]), columns=self._input_cols, index=test.index
        )
        return X_train, X_test

    def _plan_windows(self, n_rows: int) -> List[Tuple[int, int, int]]:
        """Compute the expanding-window splits.

        Args:
            n_rows: Number of supervised rows with complete targets.

        Returns:
            List of ``(window_id, train_end, test_start)``: the window trains on
            rows ``[0, train_end)`` and is evaluated on rows
            ``[test_start, test_start + horizon)``.

        Raises:
            InsufficientDataError: If no window can be evaluated.
        """
        horizon = self._config.horizon
        sliding_size = self._config.sliding_size
        gap = horizon - 1

        train_size = int(self._config.train_size * n_rows)
        self._n_iterations = max((n_rows - train_size) // sliding_size, 0)

        windows = []
        for rolling_cycle in range(self._n_iterations):
            test_start = train_size + rolling_cycle * sliding_size
            train_end = test_start - gap
            if n_rows - test_start >= horizon and train_end >= _MIN_TRAIN_ROWS:
                windows.append((rolling_cycle + 1, train_end, test_start))

        if not windows:
            raise InsufficientDataError(
                f"Insufficient data for evaluation. {max(n_rows, 0)} supervised samples remain "
                f"after building lags={self._config.lags} and horizon={horizon}, which leaves "
                f"no test window of {horizon} rows (train_size={self._config.train_size}, "
                f"sliding_size={sliding_size}). Provide more observations or reduce "
                f"lags, horizon, train_size or sliding_size."
            )
        return windows

    def evaluate(
        self,
        dataset: pd.DataFrame,
        models: Dict[str, BaseForecaster],
    ) -> pd.DataFrame:
        """Run expanding window evaluation for all models.

        A model that raises during evaluation is skipped with a warning and
        reported in ``failed_models``; the remaining models are still evaluated.

        Args:
            dataset: Input DataFrame with Date and y columns.
            models: Dictionary mapping model names to forecaster instances.

        Returns:
            DataFrame with all forecast results.

        Raises:
            InsufficientDataError: If the dataset is too small for the configuration.
            ModelFitError: If every model failed.
        """
        self._results = []
        self._failed_models = {}
        self._fitted_once = set()

        timeseries = self._processor.make_timeseries(
            dataset=dataset,
            window_size=self._config.lags,
            horizon=self._config.horizon,
            datetime_engineering=self._preprocessing.datetime_features,
        )

        self._target_cols = [c for c in timeseries.columns if c.startswith("y_horizon_")]
        self._input_cols = [
            c for c in timeseries.columns if c not in self._target_cols and c != "Date"
        ]

        # Keep rows with complete targets (the last `horizon` rows are incomplete)
        frame = timeseries.iloc[: -self._config.horizon]
        windows = self._plan_windows(frame.shape[0])

        for model_name, model in models.items():
            try:
                model_results = self._evaluate_model(
                    frame=frame,
                    windows=windows,
                    model_name=model_name,
                    model=model,
                )
            except Exception as exc:  # noqa: BLE001 - one model must not abort the run
                self._failed_models[model_name] = f"{type(exc).__name__}: {exc}"
                self._log(f"Model {model_name} failed and was skipped: {type(exc).__name__}\n ")
                warnings.warn(
                    f"Model '{model_name}' failed during evaluation and was skipped "
                    f"({type(exc).__name__}: {exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            self._results.extend(model_results)

        if not self._results:
            details = "; ".join(f"{name}: {error}" for name, error in self._failed_models.items())
            raise ModelFitError(
                f"No model could be evaluated. {details}" if details else "No models to evaluate."
            )

        return pd.concat(self._results, ignore_index=True)

    def _evaluate_model(
        self,
        frame: pd.DataFrame,
        windows: List[Tuple[int, int, int]],
        model_name: str,
        model: BaseForecaster,
    ) -> List[pd.DataFrame]:
        """Evaluate single model across all windows.

        Args:
            frame: Supervised DataFrame restricted to rows with complete targets.
            windows: Window plan from ``_plan_windows``.
            model_name: Model identifier.
            model: Forecaster instance.

        Returns:
            List of DataFrames with results per window.
        """
        results = []
        horizon = self._config.horizon

        for window_id, train_end, test_start in windows:
            train = frame.iloc[:train_end]
            test = frame.iloc[test_start : test_start + horizon]

            self._log(
                f"Algorithm Evaluation: {model_name} || "
                f"Window Iteration: {window_id} of {self._n_iterations}"
            )
            self._log(f"Rows Train: {train.shape[0]}")
            self._log(f"Rows Test: {test.shape[0]}")

            if window_id == windows[-1][0]:
                self._log(" ")

            # Forecast origins of the evaluated rows
            test_dates = test.index if "Date" not in test.columns else test["Date"].values

            X_train, X_test = self._scale(train, test)

            y_pred = self._fit_predict(
                X_train=X_train,
                y_train=train[self._target_cols],
                X_test=X_test,
                model=model,
                reuse_fitted=True,
            )

            result_df = pd.concat(
                [test[self._target_cols].reset_index(drop=True), y_pred.reset_index(drop=True)],
                axis=1,
            )
            result_df.insert(0, "Date", test_dates)
            result_df["Window"] = window_id
            result_df["Model"] = model_name

            results.append(result_df)

        return results

    def _fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model: BaseForecaster,
        reuse_fitted: bool = False,
    ) -> pd.DataFrame:
        """Fit model and generate predictions.

        Args:
            X_train: Scaled training features.
            y_train: Training targets (one column per horizon).
            X_test: Scaled test features.
            model: Forecaster instance.
            reuse_fitted: AutoGluon only. When True the model is fitted on its
                first call and reused afterwards, which keeps AutoGluon's cost
                independent of the number of evaluation windows.

        Returns:
            DataFrame with predictions.
        """
        if isinstance(model, AutoGluonForecaster):
            model.show_progress = self._verbose
            already_fitted = reuse_fitted and model.is_fitted and id(model) in self._fitted_once
            if not already_fitted:
                self._log(" ")
                model.fit(X_train, y_train)
                self._fitted_once.add(id(model))
            y_pred = model.predict(X_test)[list(y_train.columns)].to_numpy(dtype=float)
        else:
            model.fit(X_train.to_numpy(), y_train.to_numpy())
            y_pred = np.asarray(model.predict(X_test.to_numpy()), dtype=float)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_pred.shape != (X_test.shape[0], y_train.shape[1]):
            raise ValueError(
                f"Model returned predictions with shape {y_pred.shape}, expected "
                f"{(X_test.shape[0], y_train.shape[1])}."
            )
        if not np.isfinite(y_pred).all():
            raise ValueError("Model returned non-finite predictions (NaN or infinite values).")

        forecast_cols = [f"y_forecast_horizon_{i + 1}" for i in range(y_pred.shape[1])]

        return pd.DataFrame(y_pred, columns=forecast_cols, index=X_test.index)

    def single_fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: BaseForecaster,
        model_name: str = "model",
    ) -> pd.DataFrame:
        """
        Single fit-predict cycle for final forecasting.

        Args:
            train: Training DataFrame (already transformed).
            test: Test DataFrame (already transformed).
            model: Forecaster instance.
            model_name: Model identifier.

        Returns:
            DataFrame with predictions.
        """
        return self._fit_predict(
            X_train=train[self._input_cols],
            y_train=train[self._target_cols],
            X_test=test[self._input_cols],
            model=model,
            reuse_fitted=False,
        )

    @property
    def input_cols(self) -> List[str]:
        """Return input column names."""
        return self._input_cols

    @property
    def target_cols(self) -> List[str]:
        """Return target column names."""
        return self._target_cols

    @property
    def failed_models(self) -> Dict[str, str]:
        """Models skipped during the last evaluation, mapped to their error."""
        return dict(self._failed_models)
