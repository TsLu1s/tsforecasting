"""Data processing utilities for time series transformation."""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from tsforecasting.core.enums import Granularity
from tsforecasting.core.exceptions import DatasetError, InsufficientDataError

_FIXED_STEPS = {
    Granularity.MINUTE: pd.Timedelta(minutes=1),
    Granularity.HALF_HOUR: pd.Timedelta(minutes=30),
    Granularity.HOURLY: pd.Timedelta(hours=1),
    Granularity.DAILY: pd.Timedelta(days=1),
    Granularity.WEEKLY: pd.Timedelta(weeks=1),
}


class Processing:
    """Time series data processing and transformation.

    Handles conversion of raw time series data into windowed format
    suitable for multi-horizon forecasting models.

    Column naming produced by :meth:`make_timeseries`, for a row whose
    forecast origin is time ``t`` and a window of ``w`` lags:

    - ``y_lag_1`` is the *oldest* value in the window (``t - w + 1``) and
      ``y_lag_w`` the most recent one (``t``).
    - ``y_horizon_k`` is the value ``k`` steps ahead (``t + k``).
    - ``Date`` (or the ``Date_*`` features) refer to the forecast origin ``t``.
    """

    def __init__(self):
        self.target = "y"
        self.date_cols: List[str] = []
        self.input_cols: List[str] = []
        self.target_cols: List[str] = []
        self.lag_cols: List[str] = []
        self.horizon_cols: List[str] = []

    @staticmethod
    def _naive_seconds(values: pd.Series) -> pd.Series:
        """Drop timezone (keeping wall-clock time) and sub-second precision."""
        if getattr(values.dt, "tz", None) is not None:
            values = values.dt.tz_localize(None)
        return values.dt.floor("s")

    def slice_timestamp(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Detect and format datetime columns.

        Datetime columns are normalised to timezone-naive, second precision
        timestamps. When a ``Date`` column exists it also becomes the index.

        Args:
            dataset: Input DataFrame.

        Returns:
            DataFrame with formatted datetime columns.
        """
        X = dataset.copy()
        self.date_cols = []

        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                self.date_cols.append(col)

        for col in self.date_cols:
            X[col] = self._naive_seconds(X[col])

        if "Date" in self.date_cols:
            X.index = X["Date"]

        return X

    def engin_date(self, dataset: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        """Engineer date-related features from datetime columns.

        Args:
            dataset: Input DataFrame.
            drop: Whether to drop original datetime columns.

        Returns:
            DataFrame with engineered datetime features.
        """
        X = self.slice_timestamp(dataset).copy()
        X_ = pd.DataFrame(index=X.index)

        for col in self.date_cols:
            X_[f"{col}_day_of_month"] = X[col].dt.day
            X_[f"{col}_day_of_week"] = X[col].dt.day_of_week + 1
            X_[f"{col}_is_weekend"] = (X[col].dt.day_of_week >= 5).astype(int)
            X_[f"{col}_month"] = X[col].dt.month
            X_[f"{col}_day_of_year"] = X[col].dt.day_of_year
            X_[f"{col}_year"] = X[col].dt.year
            X_[f"{col}_hour"] = X[col].dt.hour
            X_[f"{col}_minute"] = X[col].dt.minute
            X_[f"{col}_second"] = X[col].dt.second

            if drop:
                X = X.drop(col, axis=1)

        return pd.concat([X_, X], axis=1)

    def _validate_window_size(self, window_size: int, dataset_length: int) -> None:
        """Validate window size against dataset length."""
        if window_size < 1:
            raise ValueError("The lags length (window_size) must be a positive integer.")
        if window_size > dataset_length:
            raise InsufficientDataError(
                "The lags length (window_size) cannot exceed the length of the time_series."
            )

    def _create_lag_windows(
        self,
        time_series: np.ndarray,
        date_series: np.ndarray,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding windows of lag features."""
        windows = sliding_window_view(time_series, window_size).copy()
        window_dates = date_series[window_size - 1 :]
        return windows, window_dates

    def _create_horizon_targets(
        self,
        time_series: np.ndarray,
        window_size: int,
        horizon: int,
    ) -> np.ndarray:
        """Create horizon targets with NaN padding for incomplete windows."""
        n_rows = len(time_series) - window_size + 1
        padded = np.concatenate([time_series.astype(float), np.full(horizon, np.nan)])
        return sliding_window_view(padded, horizon)[window_size : window_size + n_rows].copy()

    def _build_dataframe(
        self,
        windows: np.ndarray,
        horizons: np.ndarray,
        window_dates: np.ndarray,
        window_size: int,
        horizon: int,
    ) -> pd.DataFrame:
        """Build DataFrame from windows and horizons."""
        self.lag_cols = [f"y_lag_{i + 1}" for i in range(window_size)]
        self.horizon_cols = [f"y_horizon_{i + 1}" for i in range(horizon)]

        lag_df = pd.DataFrame(windows, columns=self.lag_cols)
        horizon_df = pd.DataFrame(horizons, columns=self.horizon_cols)

        X = pd.concat([lag_df, horizon_df], axis=1).reset_index(drop=True)
        X.insert(0, "Date", window_dates)
        return X

    def _set_column_groups(self, X: pd.DataFrame) -> None:
        """Set target and input column groups."""
        self.target_cols = [c for c in X.columns if c.startswith("y_horizon_")]
        self.input_cols = [c for c in X.columns if c not in self.target_cols and c != "Date"]

    def make_timeseries(
        self,
        dataset: pd.DataFrame,
        window_size: int,
        horizon: int,
        datetime_engineering: bool = True,
    ) -> pd.DataFrame:
        """Transform DataFrame into windowed time series format.

        Creates lag features and horizon targets for multi-step forecasting.
        The dataset must already be sorted by ``Date``. The last ``horizon``
        rows of the result have incomplete (NaN) targets; the very last row is
        the one used to forecast beyond the end of the data.

        Args:
            dataset: DataFrame with 'y' and 'Date' columns.
            window_size: Number of lag time steps (features).
            horizon: Number of future time steps to predict.
            datetime_engineering: Whether to add datetime features.

        Returns:
            DataFrame with lag features and horizon targets.
        """
        missing = [c for c in ("Date", self.target) if c not in dataset.columns]
        if missing:
            raise DatasetError(f"Dataset missing required columns: {missing}")
        if horizon < 1:
            raise ValueError("The horizon must be a positive integer.")

        self._validate_window_size(window_size, len(dataset))

        time_series = dataset[self.target].to_numpy()
        if not (np.issubdtype(time_series.dtype, np.number)):
            raise DatasetError(f"Column '{self.target}' must be numeric, got {time_series.dtype}.")

        dates = dataset["Date"]
        if pd.api.types.is_datetime64_any_dtype(dates) and getattr(dates.dt, "tz", None):
            dates = dates.dt.tz_localize(None)
        date_series = dates.to_numpy()

        windows, window_dates = self._create_lag_windows(time_series, date_series, window_size)
        horizons = self._create_horizon_targets(time_series, window_size, horizon)

        X = self._build_dataframe(windows, horizons, window_dates, window_size, horizon)

        if datetime_engineering:
            X = self.engin_date(dataset=X, drop=True)

        self._set_column_groups(X)

        return X

    def future_timestamps(
        self,
        dataset: pd.DataFrame,
        horizon: int,
        granularity: Union[Granularity, str] = "1d",
    ) -> pd.DataFrame:
        """Generate future timestamps for forecasting.

        Monthly steps keep the day-of-month convention of the series: month-end
        series stay on month ends (Jan 31 -> Feb 28 -> Mar 31) instead of
        drifting to the shortest month seen along the way.

        Args:
            dataset: DataFrame with 'Date' column, sorted ascending.
            horizon: Number of future timestamps.
            granularity: Time granularity (1m, 30m, 1h, 1d, 1wk, 1mo).

        Returns:
            DataFrame with future Date column.
        """
        try:
            step = Granularity.from_string(granularity)
        except ValueError:
            raise ValueError(f"Unsupported granularity: {granularity}") from None

        if horizon < 1:
            raise ValueError("The horizon must be a positive integer.")

        dates = pd.DatetimeIndex(self._naive_seconds(pd.to_datetime(dataset["Date"])))
        last_date = dates[-1]

        if step is Granularity.MONTHLY:
            anchor_day = int(dates.day.max())
            time_of_day = last_date - last_date.normalize()
            base_period = last_date.to_period("M")
            timestamps = []
            for k in range(1, horizon + 1):
                period = base_period + k
                day = min(anchor_day, period.days_in_month)
                timestamps.append(pd.Timestamp(period.year, period.month, day) + time_of_day)
        else:
            delta = _FIXED_STEPS[step]
            timestamps = [last_date + k * delta for k in range(1, horizon + 1)]

        return pd.DataFrame({"Date": pd.DatetimeIndex(timestamps)})
