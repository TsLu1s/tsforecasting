from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from tsforecasting.core.enums import Granularity, TimeSeriesPattern, TrendType


class TimeSeriesDatasetGenerator:
    """Generate synthetic time series datasets for forecasting evaluation.
    
    Supports multiple pattern types that can be combined for realistic
    time series simulation. All parameters accept both enum values and
    string equivalents for convenience.
    
    Example:
        # Using enums
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=500,
            granularity=Granularity.MONTHLY,
            patterns=[TimeSeriesPattern.TREND, TimeSeriesPattern.SEASONAL],
        )
        
        # Using strings (equivalent)
        df = TimeSeriesDatasetGenerator.generate(
            n_samples=500,
            granularity="1mo",
            patterns=["trend", "seasonal"],
        )
    """
    
    @classmethod
    def _parse_granularity(cls, value: Union[Granularity, str]) -> Granularity:
        """Convert string or enum to Granularity enum."""
        return Granularity.from_string(value)
    
    @classmethod
    def _parse_pattern(cls, value: Union[TimeSeriesPattern, str]) -> TimeSeriesPattern:
        """Convert string or enum to TimeSeriesPattern enum."""
        return TimeSeriesPattern.from_string(value)
    
    @classmethod
    def _parse_trend_type(cls, value: Union[TrendType, str]) -> TrendType:
        """Convert string or enum to TrendType enum."""
        return TrendType.from_string(value)
    
    @staticmethod
    def generate(
        n_samples: int = 1000,
        granularity: Union[Granularity, str] = "1d",  # Options: "1m", "30m", "1h", "1d", "1wk", "1mo"
        patterns: Optional[List[Union[TimeSeriesPattern, str]]] = None,  # Options: "trend", "seasonal", "mixed", "random_walk", "multi_seasonal", "regime_change"
        # Trend parameters
        trend_strength: float = 0.5,  # Range: 0.0 to 1.0 (0=no trend, 1=strong trend)
        trend_type: Union[TrendType, str] = "linear",  # Options: "linear", "exponential"
        # Seasonality parameters
        seasonality_period: int = 12,  # Period length in time steps (e.g., 12 for monthly annual cycle)
        seasonality_strength: float = 1.0,  # Range: 0.0 to 2.0 (amplitude multiplier)
        secondary_period: Optional[int] = None,  # Secondary period for multi_seasonal pattern
        # Noise parameters
        noise_level: float = 0.1,  # Range: 0.0 to 1.0 (std dev as fraction of base_value)
        # Regime change parameters
        regime_change_points: Optional[List[int]] = None,  # Indices where structural breaks occur
        # General parameters
        start_date: str = "2020-01-01",  # Start date in "YYYY-MM-DD" format
        base_value: float = 100.0,  # Base value around which series fluctuates
        random_state: int = 42,  # Random seed for reproducibility
    ) -> pd.DataFrame:
        """Generate synthetic time series with configurable patterns.
        
        Creates periodic, linear timestamps without gaps (suitable for 
        financial/operational data that doesn't skip weekends).
        
        Args:
            n_samples: Number of time steps to generate.
            
            granularity: Time granularity between observations.
                Options: "1m" (minute), "30m" (half-hour), "1h" (hourly),
                         "1d" (daily), "1wk" (weekly), "1mo" (monthly)
                         
            patterns: List of patterns to combine (additive).
                Options:
                - "trend": Linear or exponential trend component
                - "seasonal": Single sinusoidal seasonality
                - "mixed": Trend + Seasonal + Noise combined
                - "random_walk": Non-stationary cumulative random steps
                - "multi_seasonal": Two seasonality periods (e.g., daily + weekly)
                - "regime_change": Structural breaks/level shifts
                Default: ["mixed"]
                
            trend_strength: Trend magnitude relative to base_value.
                Range: 0.0 (no trend) to 1.0 (strong trend)
                
            trend_type: Shape of trend component.
                Options: "linear", "exponential"
                
            seasonality_period: Number of time steps per seasonal cycle.
                Examples: 12 (monthly data, annual cycle), 
                          7 (daily data, weekly cycle),
                          24 (hourly data, daily cycle)
                          
            seasonality_strength: Amplitude of seasonal component.
                Range: 0.0 to 2.0+ (multiplier on seasonal wave)
                
            secondary_period: Optional second seasonality period.
                Used with "multi_seasonal" pattern.
                Default: seasonality_period * 4
                
            noise_level: Random noise standard deviation.
                Range: 0.0 to 1.0 (as fraction of base_value)
                
            regime_change_points: Indices where level shifts occur.
                Default: [n_samples//3, 2*n_samples//3]
                
            start_date: Series start date ("YYYY-MM-DD" format).
            
            base_value: Central value around which series fluctuates.
            
            random_state: Seed for reproducible random generation.
            
        Returns:
            DataFrame with columns:
            - 'Date': datetime index (periodic, no gaps)
            - 'y': target values
            
        Examples:
            # Simple trend + seasonality
            df = TimeSeriesDatasetGenerator.generate(
                n_samples=300,
                granularity="1mo",
                patterns=["trend", "seasonal"],
                trend_strength=0.3,
                seasonality_period=12,
            )
            
            # Random walk
            df = TimeSeriesDatasetGenerator.generate(
                n_samples=500,
                patterns=["random_walk"],
                noise_level=0.05,
            )
            
            # Complex multi-seasonal with regime changes
            df = TimeSeriesDatasetGenerator.generate(
                n_samples=1000,
                granularity="1h",
                patterns=["trend", "multi_seasonal", "regime_change"],
                seasonality_period=24,
                secondary_period=168,
                regime_change_points=[300, 700],
            )
        """
        np.random.seed(random_state)
        
        # Parse granularity
        granularity = TimeSeriesDatasetGenerator._parse_granularity(granularity)
        
        # Parse trend type
        trend_type = TimeSeriesDatasetGenerator._parse_trend_type(trend_type)
        
        # Default to MIXED pattern
        if patterns is None:
            patterns = [TimeSeriesPattern.MIXED]
        
        # Parse patterns (convert strings to enums)
        patterns = [
            TimeSeriesDatasetGenerator._parse_pattern(p) for p in patterns
        ]
        
        # Generate base time index
        t = np.arange(n_samples)
        
        # Initialize series with base value
        y = np.full(n_samples, base_value, dtype=float)
        
        # Apply patterns
        for pattern in patterns:
            if pattern == TimeSeriesPattern.TREND:
                y += TimeSeriesDatasetGenerator._generate_trend(
                    t, trend_strength, trend_type, base_value
                )
            
            elif pattern == TimeSeriesPattern.SEASONAL:
                y += TimeSeriesDatasetGenerator._generate_seasonal(
                    t, seasonality_period, seasonality_strength, base_value
                )
            
            elif pattern == TimeSeriesPattern.MIXED:
                y += TimeSeriesDatasetGenerator._generate_trend(
                    t, trend_strength * 0.5, trend_type, base_value
                )
                y += TimeSeriesDatasetGenerator._generate_seasonal(
                    t, seasonality_period, seasonality_strength * 0.5, base_value
                )
                y += TimeSeriesDatasetGenerator._generate_noise(
                    n_samples, noise_level, base_value
                )
            
            elif pattern == TimeSeriesPattern.RANDOM_WALK:
                y += TimeSeriesDatasetGenerator._generate_random_walk(
                    n_samples, noise_level, base_value
                )
            
            elif pattern == TimeSeriesPattern.MULTI_SEASONAL:
                y += TimeSeriesDatasetGenerator._generate_seasonal(
                    t, seasonality_period, seasonality_strength * 0.6, base_value
                )
                sec_period = secondary_period or seasonality_period * 4
                y += TimeSeriesDatasetGenerator._generate_seasonal(
                    t, sec_period, seasonality_strength * 0.4, base_value
                )
            
            elif pattern == TimeSeriesPattern.REGIME_CHANGE:
                y += TimeSeriesDatasetGenerator._generate_regime_changes(
                    n_samples, regime_change_points, base_value
                )
        
        # Add base noise if not MIXED (MIXED already includes noise)
        if TimeSeriesPattern.MIXED not in patterns:
            y += TimeSeriesDatasetGenerator._generate_noise(
                n_samples, noise_level * 0.5, base_value
            )
        
        # Generate dates
        dates = TimeSeriesDatasetGenerator._generate_dates(
            start_date, n_samples, granularity
        )
        
        return pd.DataFrame({"Date": dates, "y": y})
    
    @staticmethod
    def _generate_trend(
        t: np.ndarray,
        strength: float,
        trend_type: TrendType,
        base_value: float,
    ) -> np.ndarray:
        """Generate trend component."""
        if trend_type == TrendType.LINEAR:
            return strength * base_value * t / len(t)
        else:  # EXPONENTIAL
            return base_value * (np.exp(strength * t / len(t)) - 1)
    
    @staticmethod
    def _generate_seasonal(
        t: np.ndarray,
        period: int,
        strength: float,
        base_value: float,
    ) -> np.ndarray:
        """Generate seasonal component."""
        return strength * base_value * 0.2 * np.sin(2 * np.pi * t / period)
    
    @staticmethod
    def _generate_noise(
        n_samples: int,
        level: float,
        base_value: float,
    ) -> np.ndarray:
        """Generate noise component."""
        return np.random.normal(0, level * base_value, n_samples)
    
    @staticmethod
    def _generate_random_walk(
        n_samples: int,
        level: float,
        base_value: float,
    ) -> np.ndarray:
        """Generate random walk component."""
        steps = np.random.normal(0, level * base_value * 0.1, n_samples)
        return np.cumsum(steps)
    
    @staticmethod
    def _generate_regime_changes(
        n_samples: int,
        change_points: Optional[List[int]],
        base_value: float,
    ) -> np.ndarray:
        """Generate regime change component."""
        if change_points is None:
            # Default: changes at 1/3 and 2/3 of series
            change_points = [n_samples // 3, 2 * n_samples // 3]
        
        result = np.zeros(n_samples)
        shifts = np.random.uniform(-0.2, 0.2, len(change_points)) * base_value
        
        for i, cp in enumerate(change_points):
            if cp < n_samples:
                result[cp:] += shifts[i]
        
        return result
    
    @staticmethod
    def _generate_dates(
        start_date: str,
        n_samples: int,
        granularity: Granularity,
    ) -> pd.DatetimeIndex:
        """Generate date range with periodic linear timestamps (no gaps)."""
        freq_map = {
            Granularity.MINUTE: "min",      # Updated from "T"
            Granularity.HALF_HOUR: "30min", # Updated from "30T"
            Granularity.HOURLY: "h",        # Updated from "H"
            Granularity.DAILY: "D",
            Granularity.WEEKLY: "W",
            Granularity.MONTHLY: "MS",
        }
        return pd.date_range(
            start=start_date,
            periods=n_samples,
            freq=freq_map[granularity]
        )
    
    @staticmethod
    def quick_monthly(
        n_samples: int = 120,  # Default: 10 years of monthly data
        patterns: Optional[List[Union[TimeSeriesPattern, str]]] = None,  # Options: "trend", "seasonal", "mixed", "random_walk", "multi_seasonal", "regime_change"
        **kwargs
    ) -> pd.DataFrame:
        """Generate monthly data with configurable patterns.
        
        Convenience method with monthly-appropriate defaults.
        
        Args:
            n_samples: Number of months (default: 120 = 10 years).
            patterns: Patterns to apply.
                Options: "trend", "seasonal", "mixed", "random_walk",
                         "multi_seasonal", "regime_change"
                Default: ["seasonal"]
            **kwargs: Additional parameters for generate().
            
        Returns:
            DataFrame with monthly time series (Date, y).
        """
        if patterns is None:
            patterns = ["seasonal"]
        
        return TimeSeriesDatasetGenerator.generate(
            n_samples=n_samples,
            granularity="1mo",
            patterns=patterns,
            seasonality_period=12,  # Annual cycle
            **kwargs
        )
    
    @staticmethod
    def quick_daily(
        n_samples: int = 365,  # Default: 1 year of daily data
        patterns: Optional[List[Union[TimeSeriesPattern, str]]] = None,  # Options: "trend", "seasonal", "mixed", "random_walk", "multi_seasonal", "regime_change"
        **kwargs
    ) -> pd.DataFrame:
        """Generate daily data with configurable patterns.
        
        Convenience method with daily-appropriate defaults.
        Creates continuous daily timestamps (no weekend gaps).
        
        Args:
            n_samples: Number of days (default: 365 = 1 year).
            patterns: Patterns to apply.
                Options: "trend", "seasonal", "mixed", "random_walk",
                         "multi_seasonal", "regime_change"
                Default: ["mixed"]
            **kwargs: Additional parameters for generate().
            
        Returns:
            DataFrame with daily time series (Date, y).
        """
        if patterns is None:
            patterns = ["mixed"]
        
        return TimeSeriesDatasetGenerator.generate(
            n_samples=n_samples,
            granularity="1d",
            patterns=patterns,
            seasonality_period=7,  # Weekly cycle
            **kwargs
        )
    
    @staticmethod
    def quick_hourly(
        n_samples: int = 720,  # Default: 30 days of hourly data
        patterns: Optional[List[Union[TimeSeriesPattern, str]]] = None,  # Options: "trend", "seasonal", "mixed", "random_walk", "multi_seasonal", "regime_change"
        **kwargs
    ) -> pd.DataFrame:
        """Generate hourly data with configurable patterns.
        
        Convenience method with hourly-appropriate defaults.
        
        Args:
            n_samples: Number of hours (default: 720 = 30 days).
            patterns: Patterns to apply.
                Options: "trend", "seasonal", "mixed", "random_walk",
                         "multi_seasonal", "regime_change"
                Default: ["multi_seasonal"]
            **kwargs: Additional parameters for generate().
            
        Returns:
            DataFrame with hourly time series (Date, y).
        """
        if patterns is None:
            patterns = ["multi_seasonal"]
        
        return TimeSeriesDatasetGenerator.generate(
            n_samples=n_samples,
            granularity="1h",
            patterns=patterns,
            seasonality_period=24,   # Daily cycle
            secondary_period=168,    # Weekly cycle (24 * 7)
            **kwargs
        )
    
    @staticmethod
    def list_available_patterns() -> List[str]:
        """List all available pattern types.
        
        Returns:
            List of pattern names with descriptions.
        """
        return [
            "trend         - Linear or exponential trend component",
            "seasonal      - Single sinusoidal seasonality",
            "mixed         - Trend + Seasonal + Noise combined",
            "random_walk   - Non-stationary cumulative random steps",
            "multi_seasonal- Two seasonality periods (e.g., daily + weekly)",
            "regime_change - Structural breaks/level shifts",
        ]
    
    @staticmethod
    def list_available_granularities() -> List[str]:
        """List all available granularity options.
        
        Returns:
            List of granularity options with descriptions.
        """
        return [
            "1m   / minute  - Minute-level data",
            "30m  / half_hour - 30-minute intervals",
            "1h   / hourly  - Hourly data",
            "1d   / daily   - Daily data",
            "1wk  / weekly  - Weekly data",
            "1mo  / monthly - Monthly data",
        ]
    
    @staticmethod
    def benchmark_datasets() -> dict:
        """Generate multiple datasets for benchmarking.
        
        Returns:
            Dictionary mapping dataset names to DataFrames.
        """
        return {
            "monthly_seasonal": TimeSeriesDatasetGenerator.quick_monthly(120),
            "monthly_trend": TimeSeriesDatasetGenerator.quick_monthly(
                120, patterns=["trend"]
            ),
            "daily_mixed": TimeSeriesDatasetGenerator.quick_daily(365),
            "daily_random_walk": TimeSeriesDatasetGenerator.quick_daily(
                365, patterns=["random_walk"]
            ),
            "hourly_multi_seasonal": TimeSeriesDatasetGenerator.quick_hourly(720),
            "regime_change": TimeSeriesDatasetGenerator.generate(
                n_samples=300,
                patterns=["trend", "regime_change"],
                regime_change_points=[100, 200],
            ),
        }