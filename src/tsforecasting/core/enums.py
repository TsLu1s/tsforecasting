from enum import Enum
from typing import Any, Union


class Granularity(str, Enum):
    """Time granularity options for time series data."""

    MINUTE = "1m"
    HALF_HOUR = "30m"
    HOURLY = "1h"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"

    @classmethod
    def from_string(cls, value: Union["Granularity", str]) -> "Granularity":
        """Parse string to Granularity with flexible aliases."""
        if isinstance(value, cls):
            return value

        key = str(value).lower().strip()
        aliases = {
            "1m": cls.MINUTE,
            "30m": cls.HALF_HOUR,
            "1h": cls.HOURLY,
            "1d": cls.DAILY,
            "1wk": cls.WEEKLY,
            "1mo": cls.MONTHLY,
            "minute": cls.MINUTE,
            "min": cls.MINUTE,
            "m": cls.MINUTE,
            "half_hour": cls.HALF_HOUR,
            "30min": cls.HALF_HOUR,
            "halfhour": cls.HALF_HOUR,
            "hourly": cls.HOURLY,
            "hour": cls.HOURLY,
            "h": cls.HOURLY,
            "daily": cls.DAILY,
            "day": cls.DAILY,
            "d": cls.DAILY,
            "weekly": cls.WEEKLY,
            "week": cls.WEEKLY,
            "w": cls.WEEKLY,
            "monthly": cls.MONTHLY,
            "month": cls.MONTHLY,
            "mo": cls.MONTHLY,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Invalid granularity: '{value}'. Options: {list(aliases.keys())[:12]}")

    @classmethod
    def to_timedelta_kwargs(cls, granularity: "Granularity") -> dict:
        """Convert granularity to offset keyword arguments.

        The result is valid for ``pandas.DateOffset`` and
        ``dateutil.relativedelta``. Note that ``datetime.timedelta`` has no
        ``months`` argument, so the monthly entry cannot be passed to it.
        """
        mapping = {
            cls.MINUTE: {"minutes": 1},
            cls.HALF_HOUR: {"minutes": 30},
            cls.HOURLY: {"hours": 1},
            cls.DAILY: {"days": 1},
            cls.WEEKLY: {"weeks": 1},
            cls.MONTHLY: {"months": 1},
        }
        return mapping[granularity]

    @property
    def approx_seconds(self) -> float:
        """Approximate length of one step in seconds (a month counts as 30.44 days)."""
        seconds = {
            Granularity.MINUTE: 60.0,
            Granularity.HALF_HOUR: 1800.0,
            Granularity.HOURLY: 3600.0,
            Granularity.DAILY: 86400.0,
            Granularity.WEEKLY: 604800.0,
            Granularity.MONTHLY: 2629800.0,
        }
        return seconds[self]


class EvaluationMetric(str, Enum):
    """Evaluation metrics for model performance."""

    MAE = "mae"
    MAPE = "mape"
    MSE = "mse"

    @classmethod
    def from_string(cls, value: Union["EvaluationMetric", str]) -> "EvaluationMetric":
        """Parse string to EvaluationMetric (case-insensitive, accepts display names)."""
        if isinstance(value, cls):
            return value

        key = str(value).lower().strip()
        aliases = {
            "mae": cls.MAE,
            "mean absolute error": cls.MAE,
            "mape": cls.MAPE,
            "mean absolute percentage error": cls.MAPE,
            "mse": cls.MSE,
            "mean squared error": cls.MSE,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Invalid metric: '{value}'. Options: 'MAE', 'MAPE', 'MSE'")

    @property
    def display_name(self) -> str:
        """Return human-readable metric name."""
        names = {
            self.MAE: "Mean Absolute Error",
            self.MAPE: "Mean Absolute Percentage Error",
            self.MSE: "Mean Squared Error",
        }
        return names[self]


class ModelName(str, Enum):
    """Available forecasting model identifiers."""

    RANDOM_FOREST = "RandomForest"
    EXTRA_TREES = "ExtraTrees"
    GBR = "GBR"
    KNN = "KNN"
    GENERALIZED_LR = "GeneralizedLR"
    XGBOOST = "XGBoost"
    CATBOOST = "Catboost"
    AUTOGLUON = "AutoGluon"

    @classmethod
    def _missing_(cls, value: Any) -> Any:
        """Resolve names case-insensitively (e.g. 'xgboost', 'CatBoost')."""
        if isinstance(value, str):
            key = value.strip().lower()
            for member in cls:
                if member.value.lower() == key:
                    return member
        return None


class TimeSeriesPattern(str, Enum):
    """Synthetic time series pattern types."""

    TREND = "trend"
    SEASONAL = "seasonal"
    MIXED = "mixed"
    RANDOM_WALK = "random_walk"
    MULTI_SEASONAL = "multi_seasonal"
    REGIME_CHANGE = "regime_change"

    @classmethod
    def from_string(cls, value: Union["TimeSeriesPattern", str]) -> "TimeSeriesPattern":
        """Parse string to TimeSeriesPattern with flexible aliases."""
        if isinstance(value, cls):
            return value

        key = str(value).lower().strip()
        aliases = {
            "trend": cls.TREND,
            "seasonal": cls.SEASONAL,
            "mixed": cls.MIXED,
            "random_walk": cls.RANDOM_WALK,
            "multi_seasonal": cls.MULTI_SEASONAL,
            "regime_change": cls.REGIME_CHANGE,
            "randomwalk": cls.RANDOM_WALK,
            "rw": cls.RANDOM_WALK,
            "walk": cls.RANDOM_WALK,
            "multiseasonal": cls.MULTI_SEASONAL,
            "multi": cls.MULTI_SEASONAL,
            "regimechange": cls.REGIME_CHANGE,
            "regime": cls.REGIME_CHANGE,
            "structural_break": cls.REGIME_CHANGE,
            "break": cls.REGIME_CHANGE,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Invalid pattern: '{value}'. Options: {[p.value for p in cls]}")


class ScalerType(str, Enum):
    """Preprocessing scaler options."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"

    @classmethod
    def from_string(cls, value: Union["ScalerType", str]) -> "ScalerType":
        """Parse string to ScalerType (case-insensitive)."""
        if isinstance(value, cls):
            return value

        key = str(value).lower().strip().replace("_", "").replace("-", "")
        aliases = {
            "standard": cls.STANDARD,
            "standardscaler": cls.STANDARD,
            "minmax": cls.MINMAX,
            "minmaxscaler": cls.MINMAX,
            "robust": cls.ROBUST,
            "robustscaler": cls.ROBUST,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Invalid scaler: '{value}'. Options: 'standard', 'minmax', 'robust'")


class TrendType(str, Enum):
    """Trend pattern types for data generation."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"

    @classmethod
    def from_string(cls, value: Union["TrendType", str]) -> "TrendType":
        """Parse string to TrendType with flexible aliases."""
        if isinstance(value, cls):
            return value

        key = str(value).lower().strip()
        aliases = {
            "linear": cls.LINEAR,
            "lin": cls.LINEAR,
            "exponential": cls.EXPONENTIAL,
            "exp": cls.EXPONENTIAL,
        }
        if key in aliases:
            return aliases[key]
        raise ValueError(f"Invalid trend_type: '{value}'. Options: 'linear', 'exponential'")
