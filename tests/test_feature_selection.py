"""Tests for tsforecasting.processing.feature_selection module."""

import numpy as np
import pandas as pd
import pytest

from tsforecasting.core.exceptions import NotFittedError
from tsforecasting.processing.feature_selection import TreeBasedFeatureSelector


@pytest.fixture
def feature_data():
    """Feature matrix and multi-horizon target."""
    np.random.seed(42)
    n = 100
    X = pd.DataFrame(
        {
            "important_1": np.random.randn(n) * 10,
            "important_2": np.random.randn(n) * 5,
            "noise_1": np.random.randn(n) * 0.01,
            "noise_2": np.random.randn(n) * 0.01,
        }
    )
    y = pd.DataFrame(
        {
            "h1": X["important_1"] * 2 + np.random.randn(n) * 0.1,
            "h2": X["important_2"] * 3 + np.random.randn(n) * 0.1,
        }
    )
    return X, y


class TestTreeBasedFeatureSelector:
    def test_fit_sets_fitted(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        assert selector._is_fitted

    def test_selected_features_nonempty(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(relevance_threshold=0.99)
        selector.fit(X, y)
        assert len(selector.selected_features) > 0

    def test_transform_reduces_columns(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(relevance_threshold=0.8)
        selector.fit(X, y)
        X_transformed = selector.transform(X)
        assert X_transformed.shape[1] <= X.shape[1]

    def test_fit_transform(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(relevance_threshold=0.8)
        result = selector.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] <= X.shape[1]

    def test_not_fitted_raises(self, feature_data):
        X, _ = feature_data
        selector = TreeBasedFeatureSelector()
        with pytest.raises(NotFittedError):
            selector.transform(X)
        with pytest.raises(NotFittedError):
            _ = selector.selected_features
        with pytest.raises(NotFittedError):
            _ = selector.feature_importances

    def test_importances_dataframe(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        imp = selector.feature_importances
        assert "variable" in imp.columns
        assert "percentage" in imp.columns
        assert "cumulative" in imp.columns
        assert imp["percentage"].sum() == pytest.approx(1.0, abs=0.01)

    def test_importances_by_horizon(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        imp = selector.importances_by_horizon
        assert "horizon_1" in imp.columns
        assert "horizon_2" in imp.columns

    def test_n_horizons(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        assert selector.n_horizons == 2

    def test_horizon_weights(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(horizon_decay=0.8)
        selector.fit(X, y)
        weights = selector.horizon_weights
        assert len(weights) == 2
        assert weights[0] > weights[1]  # First horizon weighted more
        assert np.isclose(weights.sum(), 1.0)

    def test_selection_summary(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        summary = selector.get_selection_summary()
        assert "algorithm" in summary
        assert "n_selected" in summary
        assert "n_original" in summary
        assert summary["n_original"] == 4

    def test_importance_comparison(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        comp = selector.get_importance_comparison()
        assert "weighted_mean" in comp.columns

    def test_single_target(self):
        np.random.seed(42)
        X = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        y = pd.Series(X["a"] * 3 + np.random.randn(50) * 0.1)
        selector = TreeBasedFeatureSelector()
        selector.fit(X, y)
        assert selector.n_horizons == 1

    @pytest.mark.parametrize("algo", ["RandomForest", "ExtraTrees", "GBR"])
    def test_all_algorithms(self, algo, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(algorithm=algo, n_estimators=50)
        selector.fit(X, y)
        assert len(selector.selected_features) > 0

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            TreeBasedFeatureSelector(algorithm="Invalid")

    def test_invalid_relevance_threshold(self):
        with pytest.raises(ValueError, match="relevance_threshold"):
            TreeBasedFeatureSelector(relevance_threshold=0.3)

    def test_invalid_horizon_decay(self):
        with pytest.raises(ValueError, match="horizon_decay"):
            TreeBasedFeatureSelector(horizon_decay=0.2)


class TestSelectorInputValidation:
    def test_requires_dataframe(self, feature_data):
        X, y = feature_data
        with pytest.raises(TypeError, match="DataFrame"):
            TreeBasedFeatureSelector(n_estimators=10).fit(X.to_numpy(), y)

    def test_length_mismatch(self, feature_data):
        X, y = feature_data
        with pytest.raises(ValueError, match="inconsistent lengths"):
            TreeBasedFeatureSelector(n_estimators=10).fit(X, y.iloc[:-5])

    def test_missing_targets(self, feature_data):
        X, y = feature_data
        y = y.copy()
        y.iloc[-1, 0] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            TreeBasedFeatureSelector(n_estimators=10).fit(X, y)

    def test_one_dimensional_array_target(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(n_estimators=10).fit(X, y["h1"].to_numpy())
        assert selector.n_horizons == 1

    def test_two_dimensional_array_target(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(n_estimators=10).fit(X, y.to_numpy())
        assert selector.n_horizons == 2

    def test_transform_reports_missing_columns(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(n_estimators=10, relevance_threshold=1.0).fit(X, y)
        with pytest.raises(ValueError, match="missing selected feature"):
            selector.transform(X.drop(columns=[selector.selected_features[0]]))

    def test_invalid_n_estimators(self):
        with pytest.raises(ValueError, match="n_estimators"):
            TreeBasedFeatureSelector(n_estimators=0)

    def test_informative_features_rank_first(self, feature_data):
        X, y = feature_data
        selector = TreeBasedFeatureSelector(n_estimators=50, relevance_threshold=0.9).fit(X, y)
        assert set(selector.selected_features) == {"important_1", "important_2"}
