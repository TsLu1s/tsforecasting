"""Custom exceptions for TSForecasting package.

Every exception derives from :class:`TSForecastingError`, so a single
``except TSForecastingError`` catches anything raised by the package.

Exceptions describing invalid *input* (configuration, dataset, model name)
additionally derive from :class:`ValueError`, which keeps ``except ValueError``
handlers written against earlier releases working.
"""


class TSForecastingError(Exception):
    """Base exception for TSForecasting package."""

    pass


class NotFittedError(TSForecastingError):
    """Raised when predict is called before fit."""

    pass


class ValidationError(TSForecastingError, ValueError):
    """Raised when validation fails."""

    pass


class ConfigurationError(ValidationError):
    """Raised when configuration is invalid."""

    pass


class ModelNotFoundError(TSForecastingError, ValueError):
    """Raised when requested model is not in registry."""

    pass


class ModelFitError(TSForecastingError, RuntimeError):
    """Raised when no candidate model could be fitted and evaluated."""

    pass


class DatasetError(TSForecastingError, ValueError):
    """Raised when dataset format is invalid."""

    pass


class InsufficientDataError(DatasetError):
    """Raised when dataset is too small for the requested configuration."""

    pass
