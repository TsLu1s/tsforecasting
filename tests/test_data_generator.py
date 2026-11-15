"""Tests for tsforecasting.data.generator module."""

import numpy as np
import pandas as pd
import pytest

from tsforecasting.core.enums import TimeSeriesPattern
from tsforecasting.data.generator import TimeSeriesDatasetGenerator


class TestGenerate:
    def test_default_output(self):
        df = TimeSeriesDatasetGenerator.generate()
        assert isinstance(df, pd.DataFrame)
        assert "Date" in df.columns
        assert "y" in df.columns
        assert len(df) == 1000

    def test_custom_n_samples(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=50)
        assert len(df) == 50

    def test_date_column_is_datetime(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=10)
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_reproducibility(self):
        df1 = TimeSeriesDatasetGenerator.generate(random_state=42)
        df2 = TimeSeriesDatasetGenerator.generate(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = TimeSeriesDatasetGenerator.generate(n_samples=50, random_state=1)
        df2 = TimeSeriesDatasetGenerator.generate(n_samples=50, random_state=2)
        assert not np.allclose(df1["y"].values, df2["y"].values)


class TestGranularities:
    @pytest.mark.parametrize("gran", ["1m", "30m", "1h", "1d", "1wk", "1mo"])
    def test_all_granularities(self, gran):
        df = TimeSeriesDatasetGenerator.generate(n_samples=20, granularity=gran)
        assert len(df) == 20
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_daily_frequency(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=10, granularity="1d")
        diffs = df["Date"].diff().dropna()
        assert all(d == pd.Timedelta(days=1) for d in diffs)

    def test_hourly_frequency(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=10, granularity="1h")
        diffs = df["Date"].diff().dropna()
        assert all(d == pd.Timedelta(hours=1) for d in diffs)


class TestPatterns:
    def test_trend_pattern(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=100, patterns=["trend"], trend_strength=0.8, noise_level=0.0
        )
        # With a strong trend and no noise, values should generally increase
        assert df["y"].iloc[-1] > df["y"].iloc[0]

    def test_seasonal_pattern(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=100, patterns=["seasonal"], seasonality_period=10, noise_level=0.0
        )
        # Should have periodic behavior
        assert df["y"].std() > 0

    def test_mixed_pattern(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=100, patterns=["mixed"])
        assert len(df) == 100

    def test_random_walk_pattern(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=100, patterns=["random_walk"])
        assert len(df) == 100

    def test_multi_seasonal_pattern(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=200, patterns=["multi_seasonal"], seasonality_period=24, secondary_period=168
        )
        assert len(df) == 200

    def test_regime_change_pattern(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=100, patterns=["regime_change"], regime_change_points=[30, 70]
        )
        assert len(df) == 100

    def test_combined_patterns(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=200, patterns=["trend", "seasonal", "regime_change"]
        )
        assert len(df) == 200

    def test_string_and_enum_patterns_equivalent(self):
        df_str = TimeSeriesDatasetGenerator.generate(
            n_samples=50, patterns=["trend"], random_state=42
        )
        df_enum = TimeSeriesDatasetGenerator.generate(
            n_samples=50, patterns=[TimeSeriesPattern.TREND], random_state=42
        )
        pd.testing.assert_frame_equal(df_str, df_enum)


class TestTrendTypes:
    def test_linear(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=50, patterns=["trend"], trend_type="linear", noise_level=0.0
        )
        assert len(df) == 50

    def test_exponential(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=50, patterns=["trend"], trend_type="exponential", noise_level=0.0
        )
        assert len(df) == 50


class TestConvenienceMethods:
    def test_quick_monthly(self):
        df = TimeSeriesDatasetGenerator.quick_monthly(60)
        assert len(df) == 60

    def test_quick_daily(self):
        df = TimeSeriesDatasetGenerator.quick_daily(100)
        assert len(df) == 100

    def test_quick_hourly(self):
        df = TimeSeriesDatasetGenerator.quick_hourly(48)
        assert len(df) == 48

    def test_quick_monthly_with_patterns(self):
        df = TimeSeriesDatasetGenerator.quick_monthly(60, patterns=["trend", "seasonal"])
        assert len(df) == 60


class TestBenchmarkDatasets:
    def test_returns_dict(self):
        datasets = TimeSeriesDatasetGenerator.benchmark_datasets()
        assert isinstance(datasets, dict)
        assert len(datasets) == 6

    def test_all_are_dataframes(self):
        datasets = TimeSeriesDatasetGenerator.benchmark_datasets()
        for name, df in datasets.items():
            assert isinstance(df, pd.DataFrame), f"{name} is not a DataFrame"
            assert "Date" in df.columns
            assert "y" in df.columns


class TestListMethods:
    def test_list_patterns(self):
        patterns = TimeSeriesDatasetGenerator.list_available_patterns()
        assert len(patterns) == 6

    def test_list_granularities(self):
        grans = TimeSeriesDatasetGenerator.list_available_granularities()
        assert len(grans) == 6


class TestRandomState:
    def test_global_numpy_rng_is_untouched(self):
        np.random.seed(123)
        expected = np.random.rand()

        np.random.seed(123)
        TimeSeriesDatasetGenerator.generate(n_samples=50, patterns=["mixed", "random_walk"])
        assert np.random.rand() == expected

    def test_values_are_stable_across_releases(self):
        """Seeded output must not change: downstream experiments depend on it."""
        df = TimeSeriesDatasetGenerator.generate(n_samples=6, random_state=42)
        np.testing.assert_allclose(
            df["y"].to_numpy(),
            [104.967142, 107.784024, 123.470473, 137.730299, 122.985387, 123.491964],
            atol=1e-6,
        )
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=6,
            patterns=["random_walk", "regime_change"],
            regime_change_points=[2, 4],
            random_state=7,
        )
        np.testing.assert_allclose(
            df["y"].to_numpy(),
            [106.778816, 104.227081, 88.867823, 91.544743, 94.135358, 90.304144],
            atol=1e-6,
        )

    def test_unseeded_generation(self):
        df1 = TimeSeriesDatasetGenerator.generate(n_samples=50, random_state=None)
        df2 = TimeSeriesDatasetGenerator.generate(n_samples=50, random_state=None)
        assert not np.allclose(df1["y"].to_numpy(), df2["y"].to_numpy())


class TestGeneratorValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_samples": 0},
            {"n_samples": -5},
            {"n_samples": 10.5},
            {"seasonality_period": 0},
            {"secondary_period": 0},
            {"noise_level": -0.1},
        ],
    )
    def test_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            TimeSeriesDatasetGenerator.generate(**kwargs)

    def test_out_of_range_change_points_are_ignored(self):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=50,
            patterns=["regime_change"],
            regime_change_points=[-5, 500],
            noise_level=0.0,
        )
        assert df["y"].nunique() == 1


class TestDates:
    @pytest.mark.parametrize("gran", ["1m", "30m", "1h", "1d", "1wk"])
    def test_series_starts_at_start_date(self, gran):
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=5, granularity=gran, start_date="2020-01-01"
        )
        assert df["Date"].iloc[0] == pd.Timestamp("2020-01-01")

    def test_weekly_frequency(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=10, granularity="1wk")
        diffs = df["Date"].diff().dropna()
        assert all(d == pd.Timedelta(weeks=1) for d in diffs)

    def test_monthly_is_month_start(self):
        df = TimeSeriesDatasetGenerator.generate(n_samples=10, granularity="1mo")
        assert df["Date"].dt.is_month_start.all()


class TestQuickGeneratorOverrides:
    def test_monthly_defaults_can_be_overridden(self):
        """Used to fail with "got multiple values for keyword argument"."""
        df = TimeSeriesDatasetGenerator.quick_monthly(60, seasonality_period=6, noise_level=0.0)
        reference = TimeSeriesDatasetGenerator.generate(
            n_samples=60,
            granularity="1mo",
            patterns=["seasonal"],
            seasonality_period=6,
            noise_level=0.0,
        )
        pd.testing.assert_frame_equal(df, reference)

    def test_daily_and_hourly_overrides(self):
        assert len(TimeSeriesDatasetGenerator.quick_daily(30, seasonality_period=14)) == 30
        assert len(TimeSeriesDatasetGenerator.quick_hourly(48, secondary_period=12)) == 48


class TestDatasetCatalogue:
    def test_list_available_datasets_matches_benchmarks(self):
        listed = [
            entry.split(" - ")[0].strip()
            for entry in TimeSeriesDatasetGenerator.list_available_datasets()
        ]
        assert listed == list(TimeSeriesDatasetGenerator.benchmark_datasets().keys())
