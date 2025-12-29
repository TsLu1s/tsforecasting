"""Base forecaster adapter for unified model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
        estimator = self._create_base_estimator()
        self._model = self._wrap_estimator(estimator)
        self._model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)
    
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted.")
    
    def get_params(self) -> Dict[str, Any]:
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
    """AutoGluon forecaster with special multi-output handling."""
    
    def __init__(self, labels: Optional[List[str]] = None, **params):
        super().__init__(**params)
        self._labels = labels or []
        self._predictors: Dict[str, Any] = {}
    
    def _create_base_estimator(self) -> None:
        return None
    
    def _wrap_estimator(self, estimator: Any) -> None:
        return None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, 
            labels: Optional[List[str]] = None):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ImportError("AutoGluon is not installed. Install with: pip install autogluon")
        
        from tqdm import tqdm
        
        if labels is not None:
            self._labels = labels
        
        if not self._labels:
            raise ValueError("labels must be provided either at init or fit time")
        
        for label in tqdm(self._labels, desc="Fitting AutoGluon", ncols=80):
            predictor = TabularPredictor(
                label=label,
                eval_metric=self._params.get("eval_metric", "mean_squared_error"),
                verbosity=self._params.get("verbosity", 0),
            )
            predictor.fit(
                X,
                presets=self._params.get("presets", "good_quality"),
                time_limit=self._params.get("time_limit", 30),
                save_space=self._params.get("save_space", False),
            )
            self._predictors[label] = predictor
        
        self._is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        predictions = pd.DataFrame(index=X.index)
        for label, predictor in self._predictors.items():
            predictions[label] = predictor.predict(X).astype(float)
        return predictions
    
    def set_labels(self, labels: List[str]):
        self._labels = labels
        return self