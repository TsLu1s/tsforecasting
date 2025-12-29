"""Custom exceptions for TSForecasting package."""


class TSForecastingError(Exception):
    """Base exception for TSForecasting package."""
    pass


class NotFittedError(TSForecastingError):
    """Raised when predict is called before fit."""
    pass


class ValidationError(TSForecastingError):
    """Raised when validation fails."""
    pass


class ConfigurationError(TSForecastingError):
    """Raised when configuration is invalid."""
    pass


class ModelNotFoundError(TSForecastingError):
    """Raised when requested model is not in registry."""
    pass


class DatasetError(TSForecastingError):
    """Raised when dataset format is invalid."""
    pass


class InsufficientDataError(TSForecastingError):
    """Raised when dataset is too small for the requested configuration."""
    pass
