"""Tests for tsforecasting.processing.processor module."""

import numpy as np
import pandas as pd
import pytest

from tsforecasting.core.exceptions import DatasetError
from tsforecasting.processing.processor import Processing


@pytest.fixture
def processor():
    return Processing()


@pytest.fixture
def simple_ts():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.DataFrame({"Date": dates, "y": np.arange(50, dtype=float)})


class TestSliceTimestamp:
    def test_sets_date_index(self, processor, simple_ts):
        result = processor.slice_timestamp(simple_ts)
        assert result.index.name == "Date" or (result.index == result["Date"]).all()

    def test_detects_date_cols(self, processor, simple_ts):
        processor.slice_timestamp(simple_ts)
        assert "Date" in processor.date_cols

    def test_non_datetime_cols_ignored(self, processor):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5), "y": [1, 2, 3, 4, 5]})
        processor.slice_timestamp(df)
        assert "y" not in processor.date_cols


class TestEnginDate:
    def test_creates_date_features(self, processor, simple_ts):
        result = processor.engin_date(simple_ts)
        assert "Date_day_of_month" in result.columns
        assert "Date_day_of_week" in result.columns
        assert "Date_is_weekend" in result.columns
        assert "Date_month" in result.columns
        assert "Date_year" in result.columns
        assert "Date_hour" in result.columns

    def test_drops_date_column(self, processor, simple_ts):
        result = processor.engin_date(simple_ts, drop=True)
        assert "Date" not in result.columns

    def test_keeps_date_column(self, processor, simple_ts):
        result = processor.engin_date(simple_ts, drop=False)
        assert "Date" in result.columns

    def test_weekend_detection(self, processor):
        # 2020-01-04 is a Saturday, 2020-01-05 is Sunday
        dates = pd.date_range("2020-01-03", periods=4, freq="D")  # Fri, Sat, Sun, Mon
        df = pd.DataFrame({"Date": dates, "y": [1, 2, 3, 4]})
        result = processor.engin_date(df)
        weekend_vals = result["Date_is_weekend"].values
        assert weekend_vals[0] == 0  # Friday
        assert weekend_vals[1] == 1  # Saturday
        assert weekend_vals[2] == 1  # Sunday
        assert weekend_vals[3] == 0  # Monday


class TestMakeTimeseries:
    def test_output_shape(self, processor, simple_ts):
        result = processor.make_timeseries(simple_ts, window_size=5, horizon=3)
        assert result.shape[0] > 0
        # Should have lag columns + horizon columns + date features
        lag_cols = [c for c in result.columns if "lag" in c]
        horizon_cols = [c for c in result.columns if "horizon" in c]
        assert len(lag_cols) == 5
        assert len(horizon_cols) == 3

    def test_lag_column_naming(self, processor, simple_ts):
        result = processor.make_timeseries(simple_ts, window_size=3, horizon=2)
        assert "y_lag_1" in result.columns
        assert "y_lag_2" in result.columns
        assert "y_lag_3" in result.columns

    def test_horizon_column_naming(self, processor, simple_ts):
        result = processor.make_timeseries(simple_ts, window_size=3, horizon=2)
        assert "y_horizon_1" in result.columns
        assert "y_horizon_2" in result.columns

    def test_column_groups_set(self, processor, simple_ts):
        processor.make_timeseries(simple_ts, window_size=5, horizon=3)
        assert len(processor.target_cols) == 3
        assert all("horizon" in c for c in processor.target_cols)
        assert len(processor.input_cols) > 0

    def test_without_datetime_engineering(self, processor, simple_ts):
        result = processor.make_timeseries(
            simple_ts, window_size=5, horizon=3, datetime_engineering=False
        )
        assert "Date_day_of_month" not in result.columns
        assert "Date" in result.columns

    def test_window_size_validation(self, processor, simple_ts):
        with pytest.raises(ValueError, match="lags length"):
            processor.make_timeseries(simple_ts, window_size=100, horizon=3)

    def test_values_correctness(self, processor):
        """Verify lag values are correct sliding windows."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Date": dates, "y": np.arange(10, dtype=float)})
        result = processor.make_timeseries(df, window_size=3, horizon=2, datetime_engineering=False)
        # First row should have lags [0, 1, 2] and horizons [3, 4]
        first_row = result.iloc[0]
        assert first_row["y_lag_1"] == 0.0
        assert first_row["y_lag_2"] == 1.0
        assert first_row["y_lag_3"] == 2.0
        assert first_row["y_horizon_1"] == 3.0
        assert first_row["y_horizon_2"] == 4.0


class TestFutureTimestamps:
    def test_daily(self, processor, simple_ts):
        result = processor.future_timestamps(simple_ts, horizon=5, granularity="1d")
        assert len(result) == 5
        assert "Date" in result.columns
        # First future date should be day after last date
        last_date = simple_ts["Date"].iloc[-1]
        assert result["Date"].iloc[0] == last_date + pd.Timedelta(days=1)

    def test_monthly(self, processor):
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        df = pd.DataFrame({"Date": dates, "y": range(12)})
        result = processor.future_timestamps(df, horizon=3, granularity="1mo")
        assert len(result) == 3

    def test_hourly(self, processor):
        dates = pd.date_range("2020-01-01", periods=24, freq="h")
        df = pd.DataFrame({"Date": dates, "y": range(24)})
        result = processor.future_timestamps(df, horizon=5, granularity="1h")
        assert len(result) == 5
        diff = result["Date"].iloc[1] - result["Date"].iloc[0]
        assert diff == pd.Timedelta(hours=1)

    def test_invalid_granularity(self, processor, simple_ts):
        with pytest.raises(ValueError, match="Unsupported granularity"):
            processor.future_timestamps(simple_ts, horizon=5, granularity="invalid")


class TestMakeTimeseriesStructure:
    def test_row_count_and_forecast_row(self, processor, simple_ts):
        """n - window + 1 rows; the last one is the forecast row with empty targets."""
        result = processor.make_timeseries(simple_ts, window_size=5, horizon=3)
        assert len(result) == len(simple_ts) - 5 + 1

        horizon_cols = [c for c in result.columns if c.startswith("y_horizon_")]
        assert result[horizon_cols].iloc[-1].isna().all()
        # Exactly `horizon` trailing rows have incomplete targets
        assert result[horizon_cols].isna().any(axis=1).sum() == 3
        assert not result[horizon_cols].iloc[:-3].isna().any().any()

    def test_last_row_holds_most_recent_window(self, processor, simple_ts):
        result = processor.make_timeseries(
            simple_ts, window_size=4, horizon=2, datetime_engineering=False
        )
        last = result.iloc[-1]
        assert [last[f"y_lag_{i}"] for i in range(1, 5)] == [46.0, 47.0, 48.0, 49.0]
        assert last["Date"] == simple_ts["Date"].iloc[-1]

    def test_matches_reference_implementation(self, processor):
        """The vectorised windowing must reproduce a plain loop."""
        rng = np.random.RandomState(0)
        n, window, horizon = 60, 7, 4
        values = rng.randn(n)
        df = pd.DataFrame({"Date": pd.date_range("2021-01-01", periods=n, freq="D"), "y": values})
        result = processor.make_timeseries(df, window, horizon, datetime_engineering=False)

        for i in range(n - window + 1):
            np.testing.assert_array_equal(
                result.iloc[i][[f"y_lag_{k + 1}" for k in range(window)]].to_numpy(dtype=float),
                values[i : i + window],
            )
            expected = np.full(horizon, np.nan)
            available = values[i + window : i + window + horizon]
            expected[: len(available)] = available
            np.testing.assert_array_equal(
                result.iloc[i][[f"y_horizon_{k + 1}" for k in range(horizon)]].to_numpy(
                    dtype=float
                ),
                expected,
            )

    def test_input_cols_exclude_date_and_targets(self, processor, simple_ts):
        processor.make_timeseries(simple_ts, window_size=5, horizon=3, datetime_engineering=False)
        assert processor.input_cols == [f"y_lag_{i}" for i in range(1, 6)]

    def test_integer_target(self, processor):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=20), "y": np.arange(20)})
        result = processor.make_timeseries(df, window_size=3, horizon=2, datetime_engineering=False)
        assert result["y_horizon_1"].iloc[0] == 3

    def test_non_numeric_target(self, processor):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5), "y": list("abcde")})
        with pytest.raises(DatasetError, match="must be numeric"):
            processor.make_timeseries(df, window_size=2, horizon=1)

    def test_missing_columns(self, processor):
        with pytest.raises(DatasetError, match="missing required columns"):
            processor.make_timeseries(
                pd.DataFrame({"y": [1.0, 2.0, 3.0]}), window_size=2, horizon=1
            )

    def test_invalid_horizon(self, processor, simple_ts):
        with pytest.raises(ValueError, match="horizon"):
            processor.make_timeseries(simple_ts, window_size=3, horizon=0)

    def test_timezone_aware_dates_keep_wall_clock_time(self, processor):
        dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="Europe/Lisbon")
        df = pd.DataFrame({"Date": dates, "y": np.arange(10.0)})
        result = processor.make_timeseries(df, window_size=3, horizon=2, datetime_engineering=False)
        assert result["Date"].iloc[0] == pd.Timestamp("2020-01-03")


class TestFutureTimestampsCalendar:
    def test_month_end_series_stays_on_month_ends(self, processor):
        """Jan 31 -> Feb 28 must not drag the following months to the 28th."""
        df = pd.DataFrame(
            {"Date": pd.date_range("2020-01-31", periods=24, freq="ME"), "y": range(24)}
        )
        result = processor.future_timestamps(df, horizon=4, granularity="1mo")
        assert [str(d.date()) for d in result["Date"]] == [
            "2022-01-31",
            "2022-02-28",
            "2022-03-31",
            "2022-04-30",
        ]

    def test_month_start_series(self, processor):
        df = pd.DataFrame(
            {"Date": pd.date_range("2020-01-01", periods=12, freq="MS"), "y": range(12)}
        )
        result = processor.future_timestamps(df, horizon=3, granularity="1mo")
        assert [str(d.date()) for d in result["Date"]] == ["2021-01-01", "2021-02-01", "2021-03-01"]

    def test_mid_month_series_keeps_its_day(self, processor):
        dates = pd.to_datetime(["2021-10-30", "2021-11-30", "2021-12-30", "2022-01-30"])
        df = pd.DataFrame({"Date": dates, "y": range(4)})
        result = processor.future_timestamps(df, horizon=3, granularity="1mo")
        assert [str(d.date()) for d in result["Date"]] == ["2022-02-28", "2022-03-30", "2022-04-30"]

    def test_weekly(self, processor, simple_ts):
        result = processor.future_timestamps(simple_ts, horizon=2, granularity="1wk")
        last = simple_ts["Date"].iloc[-1]
        assert list(result["Date"]) == [last + pd.Timedelta(weeks=1), last + pd.Timedelta(weeks=2)]

    def test_minute_and_half_hour(self, processor):
        df = pd.DataFrame(
            {"Date": pd.date_range("2020-01-01", periods=5, freq="min"), "y": range(5)}
        )
        assert processor.future_timestamps(df, 1, "1m")["Date"].iloc[0] == pd.Timestamp(
            "2020-01-01 00:05:00"
        )
        assert processor.future_timestamps(df, 1, "30m")["Date"].iloc[0] == pd.Timestamp(
            "2020-01-01 00:34:00"
        )

    def test_granularity_aliases(self, processor, simple_ts):
        by_code = processor.future_timestamps(simple_ts, horizon=3, granularity="1d")
        by_alias = processor.future_timestamps(simple_ts, horizon=3, granularity="daily")
        pd.testing.assert_frame_equal(by_code, by_alias)

    def test_timestamps_are_strictly_increasing(self, processor):
        df = pd.DataFrame(
            {"Date": pd.date_range("2019-08-31", periods=30, freq="ME"), "y": range(30)}
        )
        result = processor.future_timestamps(df, horizon=24, granularity="1mo")
        assert result["Date"].is_monotonic_increasing
        assert result["Date"].is_unique
        assert result["Date"].dt.is_month_end.all()
