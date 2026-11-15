"""Shared fixtures for TSForecasting test suite."""

import numpy as np
import pandas as pd
import pytest

from tsforecasting.data.generator import TimeSeriesDatasetGenerator


@pytest.fixture
def daily_dataset():
    """Small daily dataset for fast tests."""
    return TimeSeriesDatasetGenerator.generate(
        n_samples=200,
        granularity="1d",
        patterns=["mixed"],
        random_state=42,
    )


@pytest.fixture
def monthly_dataset():
    """Monthly dataset for testing."""
    return TimeSeriesDatasetGenerator.generate(
        n_samples=120,
        granularity="1mo",
        patterns=["trend", "seasonal"],
        random_state=42,
    )


@pytest.fixture
def small_dataset():
    """Minimal dataset for unit tests that need speed."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({"Date": dates, "y": values})


@pytest.fixture
def multivariate_dataset():
    """Dataset with extra feature columns."""
    dates = pd.date_range("2020-01-01", periods=150, freq="D")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Date": dates,
            "y": 100 + np.cumsum(np.random.randn(150) * 0.5),
            "feature_1": np.random.randn(150),
            "feature_2": np.random.randn(150) * 2,
        }
    )
