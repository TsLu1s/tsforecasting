"""Tree-based feature selection for time series forecasting."""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from tsforecasting.core.exceptions import NotFittedError


class TreeBasedFeatureSelector:
    """
    Tree-based feature importance selection for multi-horizon forecasting.
    
    Uses tree-based regression models to compute feature importances
    across multiple forecast horizons, with attention-weighted aggregation
    (closer horizons receive higher weight).
    
    For multivariate targets (multiple horizons), importances are computed
    per horizon and aggregated using exponential decay weights where
    horizon 1 has the highest weight.
    """
    
    ALGORITHMS = {
        "RandomForest": RandomForestRegressor,
        "ExtraTrees": ExtraTreesRegressor,
        "GBR": GradientBoostingRegressor,
    }
    
    def __init__(
        self,
        algorithm: Literal["RandomForest", "ExtraTrees", "GBR"] = "ExtraTrees",
        n_estimators: int = 250,
        relevance_threshold: float = 0.99,
        random_state: int = 42,
        horizon_decay: float = 0.8,
    ):
        """
        Initialize feature selector.
        
        Args:
            algorithm: Tree-based algorithm to use.
            n_estimators: Number of trees.
            relevance_threshold: Cumulative importance threshold (0.5 to 1.0).
            random_state: Random seed for reproducibility.
            horizon_decay: Exponential decay factor for horizon weights (0.5 to 1.0).
                           Lower values give more weight to earlier horizons.
                           Weight for horizon h = decay^(h-1), so horizon 1 always has weight 1.
        """
        if not 0.5 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0.5 and 1.0")
        
        if not 0.5 <= horizon_decay <= 1.0:
            raise ValueError("horizon_decay must be between 0.5 and 1.0")
        
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Available: {list(self.ALGORITHMS.keys())}"
            )
        
        self._algorithm = algorithm
        self._n_estimators = n_estimators
        self._relevance = relevance_threshold
        self._random_state = random_state
        self._horizon_decay = horizon_decay
        
        self._feature_importances: Optional[pd.DataFrame] = None
        self._importances_by_horizon: Optional[pd.DataFrame] = None
        self._selected_features: Optional[List[str]] = None
        self._is_fitted: bool = False
        self._n_horizons: int = 1
    
    def _compute_horizon_weights(self, n_horizons: int) -> np.ndarray:
        """
        Compute attention-based weights for each horizon.
        
        Earlier horizons get higher weights using exponential decay.
        
        Args:
            n_horizons: Number of forecast horizons.
            
        Returns:
            Normalized weight array.
        """
        weights = np.array([self._horizon_decay ** (h - 1) for h in range(1, n_horizons + 1)])
        return weights / weights.sum()  # Normalize to sum to 1
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, pd.DataFrame, np.ndarray]
    ):
        """
        Fit and compute feature importances across all horizons.
        
        For multivariate targets, computes importance per horizon and
        aggregates using attention-weighted mean (closer horizons weighted more).
        
        Args:
            X: Feature DataFrame.
            y: Target - can be Series (single horizon) or DataFrame/2D array (multi-horizon).
            
        Returns:
            Self for method chaining.
        """
        features = list(X.columns)
        X_values = X.values
        
        # Handle different y formats
        if isinstance(y, pd.Series):
            y_values = y.values.reshape(-1, 1)
        elif isinstance(y, pd.DataFrame):
            y_values = y.values
        else:
            y_values = np.atleast_2d(y)
            if y_values.shape[0] == 1:
                y_values = y_values.T
        
        self._n_horizons = y_values.shape[1]
        
        # Compute importances for each horizon
        importances_per_horizon = []
        model_class = self.ALGORITHMS[self._algorithm]
        
        for h in range(self._n_horizons):
            model = model_class(
                n_estimators=self._n_estimators,
                random_state=self._random_state
            )
            model.fit(X_values, y_values[:, h])
            importances_per_horizon.append(model.feature_importances_)
        
        # Create per-horizon importance DataFrame
        horizon_cols = {f"horizon_{h+1}": imp for h, imp in enumerate(importances_per_horizon)}
        self._importances_by_horizon = pd.DataFrame({
            "variable": features,
            **horizon_cols
        })
        
        # Compute weighted mean importance
        weights = self._compute_horizon_weights(self._n_horizons)
        importances_array = np.array(importances_per_horizon)  # Shape: (n_horizons, n_features)
        weighted_importances = np.average(importances_array, axis=0, weights=weights)
        
        # Create aggregated importance DataFrame
        self._feature_importances = pd.DataFrame({
            "variable": features,
            "percentage": weighted_importances
        }).sort_values(by="percentage", ascending=False).reset_index(drop=True)
        
        # Add cumulative percentage
        self._feature_importances["cumulative"] = self._feature_importances["percentage"].cumsum()
        
        # Select features by cumulative importance
        cumulative = 0.0
        selected = []
        
        for _, row in self._feature_importances.iterrows():
            selected.append(row["variable"])
            cumulative += row["percentage"]
            if cumulative >= self._relevance:
                break
        
        self._selected_features = selected
        self._is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on fitted importances.
        
        Args:
            X: Feature DataFrame.
            
        Returns:
            DataFrame with selected features only.
        """
        self._check_fitted()
        return X[self._selected_features].copy()
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Fit and transform in single call.
        
        Args:
            X: Feature DataFrame.
            y: Target Series, DataFrame, or 2D array.
            
        Returns:
            DataFrame with selected features.
        """
        return self.fit(X, y).transform(X)
    
    def _check_fitted(self) -> None:
        """Verify selector has been fitted."""
        if not self._is_fitted:
            raise NotFittedError(
                "TreeBasedFeatureSelector is not fitted. Call fit() first."
            )
    
    @property
    def feature_importances(self) -> pd.DataFrame:
        """
        Return aggregated feature importance DataFrame.
        
        Returns:
            DataFrame with 'variable', 'percentage', and 'cumulative' columns.
        """
        self._check_fitted()
        return self._feature_importances.copy()
    
    @property
    def importances_by_horizon(self) -> pd.DataFrame:
        """
        Return feature importances per horizon.
        
        Returns:
            DataFrame with 'variable' and 'horizon_N' columns for each horizon.
        """
        self._check_fitted()
        return self._importances_by_horizon.copy()
    
    @property
    def selected_features(self) -> List[str]:
        """Return list of selected feature names."""
        self._check_fitted()
        return self._selected_features.copy()
    
    @property
    def n_selected(self) -> int:
        """Return number of selected features."""
        self._check_fitted()
        return len(self._selected_features)
    
    @property
    def n_horizons(self) -> int:
        """Return number of horizons used in fitting."""
        self._check_fitted()
        return self._n_horizons
    
    @property
    def horizon_weights(self) -> np.ndarray:
        """Return the attention weights used for each horizon."""
        self._check_fitted()
        return self._compute_horizon_weights(self._n_horizons)
    
    def get_selection_summary(self) -> dict:
        """
        Get summary of feature selection.
        
        Returns:
            Dictionary with selection statistics.
        """
        self._check_fitted()
        
        selected_importance = self._feature_importances[
            self._feature_importances["variable"].isin(self._selected_features)
        ]["percentage"].sum()
        
        weights = self._compute_horizon_weights(self._n_horizons)
        
        return {
            "algorithm": self._algorithm,
            "threshold": self._relevance,
            "n_original": len(self._feature_importances),
            "n_selected": self.n_selected,
            "selected_importance": selected_importance,
            "n_horizons": self._n_horizons,
            "horizon_decay": self._horizon_decay,
            "horizon_weights": {f"h{i+1}": w for i, w in enumerate(weights)},
            "top_features": self._selected_features[:5],
        }
    
    def get_importance_comparison(self) -> pd.DataFrame:
        """
        Get side-by-side comparison of importances across horizons.
        
        Returns:
            DataFrame with variable, per-horizon importances, and weighted mean.
        """
        self._check_fitted()
        
        comparison = self._importances_by_horizon.copy()
        comparison["weighted_mean"] = self._feature_importances.set_index("variable").loc[
            comparison["variable"]
        ]["percentage"].values
        
        return comparison.sort_values("weighted_mean", ascending=False).reset_index(drop=True)