"""Data processing utilities for time series transformation."""

from __future__ import annotations

import datetime
from typing import List

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


class Processing:
    """Time series data processing and transformation.
    
    Handles conversion of raw time series data into windowed format
    suitable for multi-horizon forecasting models.
    """
    
    def __init__(self):
        self.target = "y"
        self.date_cols: List[str] = []
        self.input_cols: List[str] = []
        self.target_cols: List[str] = []
        self.lag_cols: List[str] = []
        self.horizon_cols: List[str] = []
    
    def slice_timestamp(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Detect and format datetime columns.
        
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
            X[col] = pd.to_datetime(X[col].dt.strftime("%Y-%m-%d %H:%M:%S"))
        
        if "Date" in self.date_cols:
            X.index = X["Date"]
        
        return X
    
    def engin_date(
        self,
        dataset: pd.DataFrame,
        drop: bool = True
    ) -> pd.DataFrame:
        """Engineer date-related features from datetime columns.
        
        Args:
            dataset: Input DataFrame.
            drop: Whether to drop original datetime columns.
            
        Returns:
            DataFrame with engineered datetime features.
        """
        X = self.slice_timestamp(dataset).copy()
        X_ = pd.DataFrame(index=X.index)
        
        for col in set(self.date_cols):
            X_[f"{col}_day_of_month"] = X[col].dt.day
            X_[f"{col}_day_of_week"] = X[col].dt.dayofweek + 1
            X_[f"{col}_is_weekend"] = (X[col].dt.dayofweek >= 5).astype(int)
            X_[f"{col}_month"] = X[col].dt.month
            X_[f"{col}_day_of_year"] = X[col].dt.dayofyear
            X_[f"{col}_year"] = X[col].dt.year
            X_[f"{col}_hour"] = X[col].dt.hour
            X_[f"{col}_minute"] = X[col].dt.minute
            X_[f"{col}_second"] = X[col].dt.second
            
            if drop:
                X = X.drop(col, axis=1)
        
        return pd.concat([X_, X], axis=1)
        
    def _validate_window_size(self, window_size: int, dataset_length: int) -> None:
        """Validate window size against dataset length."""
        if window_size > dataset_length:
            raise ValueError(
                "The lags length (window_size) cannot exceed the length of the time_series."
            )
    
    def _create_lag_windows(
        self,
        time_series: np.ndarray,
        date_series: np.ndarray,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding windows of lag features."""
        max_idx = len(time_series) - window_size
        windows = np.array([time_series[i:i + window_size] for i in range(max_idx + 1)])
        window_dates = date_series[window_size - 1:]
        return windows, window_dates
    
    def _create_horizon_targets(
        self,
        time_series: np.ndarray,
        window_size: int,
        horizon: int,
    ) -> np.ndarray:
        """Create horizon targets with NaN padding for incomplete windows."""
        max_idx = len(time_series) - window_size
        n_samples = len(time_series)
        
        horizons = []
        for i in range(max_idx + 1):
            start, end = i + window_size, i + window_size + horizon
            if end <= n_samples:
                horizons.append(time_series[start:end])
            else:
                available = time_series[start:]
                padding = np.full(end - n_samples, np.nan)
                horizons.append(np.concatenate([available, padding]))
        
        return np.array(horizons)
    
    def _append_forecast_row(
        self,
        time_series: np.ndarray,
        date_series: np.ndarray,
        windows: np.ndarray,
        horizons: np.ndarray,
        window_dates: np.ndarray,
        window_size: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Append forecast row with NaN horizons for future prediction."""
        future_window = time_series[-window_size:].reshape(1, -1)
        future_horizon = np.full((1, horizon), np.nan)
        future_date = np.array([pd.to_datetime(date_series[-1])])
        
        windows = np.concatenate([windows, future_window])
        horizons = np.concatenate([horizons, future_horizon])
        window_dates = pd.to_datetime(np.concatenate([window_dates, future_date]))
        
        return windows, horizons, window_dates
    
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
        self.target_cols = [c for c in X.columns if "horizon" in c]
        self.input_cols = [c for c in X.columns if c not in self.target_cols]
    
    def make_timeseries(
        self,
        dataset: pd.DataFrame,
        window_size: int,
        horizon: int,
        datetime_engineering: bool = True,
    ) -> pd.DataFrame:
        """Transform DataFrame into windowed time series format.
        
        Creates lag features and horizon targets for multi-step forecasting.
        
        Args:
            dataset: DataFrame with 'y' and 'Date' columns.
            window_size: Number of lag time steps (features).
            horizon: Number of future time steps to predict.
            datetime_engineering: Whether to add datetime features.
            
        Returns:
            DataFrame with lag features and horizon targets.
        """
        self._validate_window_size(window_size, len(dataset))
        
        time_series = dataset[self.target].values
        date_series = dataset["Date"].values
        
        windows, window_dates = self._create_lag_windows(time_series, date_series, window_size)
        horizons = self._create_horizon_targets(time_series, window_size, horizon)
        
        windows, horizons, window_dates = self._append_forecast_row(
            time_series, date_series, windows, horizons, window_dates, window_size, horizon
        )
        
        X = self._build_dataframe(windows, horizons, window_dates, window_size, horizon)
        
        if datetime_engineering:
            X = self.engin_date(dataset=X, drop=True)
        
        X = X.head(X.shape[0] - 1)
        self._set_column_groups(X)
        
        return X

    def future_timestamps(
        self,
        dataset: pd.DataFrame,
        horizon: int,
        granularity: str = "1d"
    ) -> pd.DataFrame:
        """Generate future timestamps for forecasting.
        
        Args:
            dataset: DataFrame with 'Date' column.
            horizon: Number of future timestamps.
            granularity: Time granularity (1m, 30m, 1h, 1d, 1wk, 1mo).
            
        Returns:
            DataFrame with future Date column.
        """
        X = self.slice_timestamp(dataset).copy()
        last_date = pd.to_datetime(X["Date"].iloc[-1])
        
        delta_map = {
            "1m": datetime.timedelta(minutes=1),
            "30m": datetime.timedelta(minutes=30),
            "1h": datetime.timedelta(hours=1),
            "1d": datetime.timedelta(days=1),
            "1wk": datetime.timedelta(weeks=1),
            "1mo": relativedelta(months=1),
        }
        
        if granularity not in delta_map:
            raise ValueError(f"Unsupported granularity: {granularity}")
        
        delta = delta_map[granularity]
        timestamps = []
        
        for _ in range(horizon):
            last_date = last_date + delta
            timestamps.append(last_date)
        
        return pd.DataFrame({"Date": timestamps})
