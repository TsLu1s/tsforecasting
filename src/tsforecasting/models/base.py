"""Base forecaster adapter for unified model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from tsforecasting.core.exceptions import NotFittedError


class BaseForecaster(ABC):
    """Unified interface for all forecasting models."""

    def __init__(self, **params):
        self._params = params
        self._model: Optional[MultiOutputRegressor] = None
        self._is_fitted: bool = False

    @abstractmethod
    def _create_base_estimator(self) -> Any:
        """Create the underlying sklearn-compatible estimator."""
        ...

    def _wrap_estimator(self, estimator: Any) -> MultiOutputRegressor:
        """Wrap estimator for multi-output prediction."""
        return MultiOutputRegressor(estimator)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit one estimator per forecast horizon.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Targets, shape (n_samples, n_horizons). A 1-D array is treated
               as a single horizon.

        Returns:
            Self for method chaining.
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        estimator = self._create_base_estimator()
        self._model = self._wrap_estimator(estimator)
        self._model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all horizons, shape (n_samples, n_horizons)."""
        self._check_fitted()
        return self._model.predict(X)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return a copy of the hyperparameters (``deep`` kept for sklearn parity)."""
        return self._params.copy()

    def set_params(self, **params):
        self._params.update(params)
        self._is_fitted = False
        return self

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"


class AutoGluonForecaster(BaseForecaster):
    """AutoGluon forecaster with special multi-output handling.

    One ``TabularPredictor`` is trained per forecast horizon. Each predictor
    only ever sees the input features plus its own target: the remaining
    horizon columns are removed before fitting and before predicting, since
    they are future values that are unknown at forecast time.
    """

    def __init__(self, labels: Optional[List[str]] = None, **params):
        super().__init__(**params)
        self._labels = list(labels) if labels else []
        self._predictors: Dict[str, Any] = {}
        self.show_progress: bool = True

    def _create_base_estimator(self) -> None:
        return None

    def _wrap_estimator(self, estimator: Any) -> None:
        return None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        labels: Optional[List[str]] = None,
    ):
        """Fit one AutoGluon predictor per label.

        Two calling conventions are supported:

        - ``fit(X, y)``: ``X`` holds the features and ``y`` the targets
          (one column per horizon).
        - ``fit(frame)``: a single DataFrame holding features *and* label
          columns, with the label names given at init, via ``set_labels`` or
          through ``labels``.

        Args:
            X: Feature DataFrame (or features + label columns).
            y: Optional targets, one column per horizon.
            labels: Optional label column names.

        Returns:
            Self for method chaining.
        """
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ImportError("AutoGluon is not installed. Install with: pip install autogluon")

        from tqdm import tqdm

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
            X.columns = [f"feature_{i + 1}" for i in range(X.shape[1])]

        if labels is not None:
            self._labels = list(labels)

        if y is not None:
            targets = self._targets_to_frame(y, X.index)
            self._labels = list(targets.columns)
            features = X.drop(columns=[c for c in self._labels if c in X.columns])
        else:
            if not self._labels:
                raise ValueError("labels must be provided either at init or fit time")
            missing = [c for c in self._labels if c not in X.columns]
            if missing:
                raise ValueError(f"Label columns not found in the training frame: {missing}")
            targets = X[self._labels]
            features = X.drop(columns=self._labels)

        self._predictors = {}

        for label in tqdm(
            self._labels, desc="Fitting AutoGluon", ncols=80, disable=not self.show_progress
        ):
            # Features + this horizon's target only: other horizons would leak the future.
            train_frame = features.copy()
            train_frame[label] = targets[label].to_numpy()

            predictor = TabularPredictor(
                label=label,
                eval_metric=self._params.get("eval_metric", "mean_squared_error"),
                verbosity=self._params.get("verbosity", 0),
            )
            predictor.fit(
                train_frame,
                presets=self._params.get("presets", "good_quality"),
                time_limit=self._params.get("time_limit", 30),
                save_space=self._params.get("save_space", False),
            )
            self._predictors[label] = predictor

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict every horizon; label columns present in ``X`` are ignored."""
        self._check_fitted()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
            X.columns = [f"feature_{i + 1}" for i in range(X.shape[1])]

        features = X.drop(columns=[c for c in self._labels if c in X.columns])
        predictions = pd.DataFrame(index=X.index)
        for label, predictor in self._predictors.items():
            predictions[label] = np.asarray(predictor.predict(features), dtype=float)
        return predictions

    def set_labels(self, labels: List[str]):
        self._labels = list(labels)
        return self

    def _targets_to_frame(
        self, y: Union[pd.DataFrame, pd.Series, np.ndarray], index: pd.Index
    ) -> pd.DataFrame:
        """Normalise targets to a DataFrame aligned with the feature index."""
        if isinstance(y, pd.DataFrame):
            targets = y.copy()
        elif isinstance(y, pd.Series):
            targets = y.to_frame(name=y.name if y.name is not None else "y_horizon_1")
        else:
            values = np.asarray(y)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if self._labels and len(self._labels) == values.shape[1]:
                columns = list(self._labels)
            else:
                columns = [f"y_horizon_{i + 1}" for i in range(values.shape[1])]
            targets = pd.DataFrame(values, columns=columns)

        if len(targets) != len(index):
            raise ValueError(
                f"X and y have inconsistent lengths: {len(index)} vs {len(targets)} rows."
            )
        targets.index = index
        targets.columns = [str(c) for c in targets.columns]
        return targets
