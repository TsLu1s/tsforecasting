from __future__ import annotations

from typing import Any, Dict, Optional

from tsforecasting.models.base import BaseForecaster


class RandomForestForecaster(BaseForecaster):
    """Random Forest regression model for multi-horizon forecasting.
    
    Wraps sklearn's RandomForestRegressor with MultiOutputRegressor for
    handling multiple forecast horizons simultaneously.
    
    Args:
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.
        criterion: Function to measure split quality.
        max_depth: Maximum depth of trees (None for unlimited).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Number of features to consider for best split.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **kwargs
        )
    
    def _create_base_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**self._params)


class ExtraTreesForecaster(BaseForecaster):
    """Extra Trees regression model for multi-horizon forecasting.
    
    Wraps sklearn's ExtraTreesRegressor with MultiOutputRegressor.
    Extra Trees uses random thresholds for splitting, providing additional
    randomization compared to Random Forest.
    
    Args:
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.
        criterion: Function to measure split quality.
        max_depth: Maximum depth of trees (None for unlimited).
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Number of features to consider for best split.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **kwargs
        )
    
    def _create_base_estimator(self) -> Any:
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(**self._params)


class GBRForecaster(BaseForecaster):
    """Gradient Boosting Regression model for multi-horizon forecasting.
    
    Wraps sklearn's GradientBoostingRegressor with MultiOutputRegressor.
    Uses sequential boosting to minimize loss functions.
    
    Args:
        n_estimators: Number of boosting stages.
        criterion: Function to measure split quality.
        learning_rate: Shrinks contribution of each tree.
        max_depth: Maximum depth of individual estimators.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        loss: Loss function to optimize.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "friedman_mse",
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        loss: str = "squared_error",
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            loss=loss,
            **kwargs
        )
    
    def _create_base_estimator(self) -> Any:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**self._params)


class KNNForecaster(BaseForecaster):
    """K-Nearest Neighbors regression model for multi-horizon forecasting.
    
    Wraps sklearn's KNeighborsRegressor with MultiOutputRegressor.
    Predictions are based on the k-nearest training samples.
    
    Args:
        n_neighbors: Number of neighbors to use.
        weights: Weight function ('uniform' or 'distance').
        algorithm: Algorithm for nearest neighbor search.
        leaf_size: Leaf size for BallTree/KDTree.
        p: Power parameter for Minkowski metric.
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        **kwargs
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            **kwargs
        )
    
    def _create_base_estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**self._params)


class GeneralizedLRForecaster(BaseForecaster):
    """Generalized Linear Regression (Tweedie) model for multi-horizon forecasting.
    
    Wraps sklearn's TweedieRegressor with MultiOutputRegressor.
    Supports various distributions through the power parameter.
    
    Args:
        power: Tweedie power (0=Normal, 1=Poisson, 2=Gamma).
        alpha: Regularization strength.
        link: Link function ('auto', 'identity', 'log').
        fit_intercept: Whether to fit intercept.
        max_iter: Maximum iterations for solver.
        warm_start: Reuse previous solution.
        verbose: Verbosity level.
    """
    
    def __init__(
        self,
        power: float = 1,
        alpha: float = 0.5,
        link: str = "log",
        fit_intercept: bool = True,
        max_iter: int = 100,
        warm_start: bool = False,
        verbose: int = 0,
        **kwargs
    ):
        # Note: verbose is not passed to TweedieRegressor as it doesn't accept it
        super().__init__(
            power=power,
            alpha=alpha,
            link=link,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            warm_start=warm_start,
            **kwargs
        )
        self._verbose = verbose
    
    def _create_base_estimator(self) -> Any:
        from sklearn.linear_model import TweedieRegressor
        # Remove verbose from params as TweedieRegressor doesn't accept it
        params = {k: v for k, v in self._params.items() if k != "verbose"}
        return TweedieRegressor(**params)


class XGBoostForecaster(BaseForecaster):
    """XGBoost regression model for multi-horizon forecasting.
    
    Wraps XGBoost's XGBRegressor with MultiOutputRegressor.
    Provides efficient gradient boosting with regularization.
    
    Args:
        n_estimators: Number of boosting rounds.
        objective: Learning objective.
        learning_rate: Boosting learning rate.
        max_depth: Maximum tree depth.
        reg_lambda: L2 regularization term.
        reg_alpha: L1 regularization term.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns per tree.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "reg:squarederror",
        learning_rate: float = 0.1,
        max_depth: int = 3,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        subsample: float = 1,
        colsample_bytree: float = 1,
        **kwargs
    ):
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **kwargs
        )
    
    def _create_base_estimator(self) -> Any:
        import xgboost as xgb
        params = {**self._params, "verbosity": 0}
        return xgb.XGBRegressor(**params)


class CatBoostForecaster(BaseForecaster):
    """CatBoost regression model for multi-horizon forecasting.
    
    Wraps CatBoost's CatBoostRegressor with MultiOutputRegressor.
    Handles categorical features natively and provides robust regularization.
    
    Args:
        iterations: Maximum number of trees.
        loss_function: Loss function to optimize.
        depth: Tree depth.
        learning_rate: Learning rate.
        l2_leaf_reg: L2 regularization coefficient.
        border_count: Number of splits for numerical features.
        subsample: Subsample ratio.
    """
    
    def __init__(
        self,
        iterations: int = 100,
        loss_function: str = "RMSE",
        depth: int = 8,
        learning_rate: float = 0.1,
        l2_leaf_reg: float = 3,
        border_count: int = 254,
        subsample: float = 1,
        **kwargs
    ):
        super().__init__(
            iterations=iterations,
            loss_function=loss_function,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            subsample=subsample,
            **kwargs
        )
    
    def _create_base_estimator(self) -> Any:
        from catboost import CatBoostRegressor
        params = {**self._params, "verbose": False, "save_snapshot": False}
        return CatBoostRegressor(**params)

# =============================================================================
# Legacy class aliases for backward compatibility
# =============================================================================

# These aliases match the original forecasting_models.py naming convention
RandomForest_Forecasting = RandomForestForecaster
ExtraTrees_Forecasting = ExtraTreesForecaster
GBR_Forecasting = GBRForecaster
KNN_Forecasting = KNNForecaster
GeneralizedLR_Forecasting = GeneralizedLRForecaster
XGBoost_Forecasting = XGBoostForecaster
CatBoost_Forecasting = CatBoostForecaster


# =============================================================================
# Class registry for model lookup
# =============================================================================

FORECASTER_CLASSES: Dict[str, type] = {
    "RandomForest": RandomForestForecaster,
    "ExtraTrees": ExtraTreesForecaster,
    "GBR": GBRForecaster,
    "KNN": KNNForecaster,
    "GeneralizedLR": GeneralizedLRForecaster,
    "XGBoost": XGBoostForecaster,
    "Catboost": CatBoostForecaster,
}

# Legacy naming convention mapping
LEGACY_FORECASTER_CLASSES: Dict[str, type] = {
    "RandomForest": RandomForest_Forecasting,
    "ExtraTrees": ExtraTrees_Forecasting,
    "GBR": GBR_Forecasting,
    "KNN": KNN_Forecasting,
    "GeneralizedLR": GeneralizedLR_Forecasting,
    "XGBoost": XGBoost_Forecasting,
    "Catboost": CatBoost_Forecasting,
}

