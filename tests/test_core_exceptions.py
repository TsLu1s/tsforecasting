"""Tests for tsforecasting.core.exceptions module."""

import pytest

from tsforecasting.core.exceptions import (
    ConfigurationError,
    DatasetError,
    InsufficientDataError,
    ModelFitError,
    ModelNotFoundError,
    NotFittedError,
    TSForecastingError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        with pytest.raises(TSForecastingError):
            raise TSForecastingError("test")

    def test_not_fitted_is_subclass(self):
        assert issubclass(NotFittedError, TSForecastingError)

    def test_validation_is_subclass(self):
        assert issubclass(ValidationError, TSForecastingError)

    def test_configuration_is_subclass(self):
        assert issubclass(ConfigurationError, TSForecastingError)

    def test_model_not_found_is_subclass(self):
        assert issubclass(ModelNotFoundError, TSForecastingError)

    def test_dataset_is_subclass(self):
        assert issubclass(DatasetError, TSForecastingError)

    def test_insufficient_data_is_subclass(self):
        assert issubclass(InsufficientDataError, TSForecastingError)

    def test_catch_all_with_base(self):
        """All specific exceptions should be catchable via the base."""
        for exc_class in [
            NotFittedError,
            ValidationError,
            ConfigurationError,
            ModelNotFoundError,
            DatasetError,
            InsufficientDataError,
        ]:
            with pytest.raises(TSForecastingError):
                raise exc_class("test message")

    def test_message_preserved(self):
        msg = "custom error message"
        try:
            raise DatasetError(msg)
        except DatasetError as e:
            assert str(e) == msg


class TestBackwardCompatibleBases:
    """Input-related errors stay catchable as ValueError, as in earlier releases."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ValidationError,
            ConfigurationError,
            ModelNotFoundError,
            DatasetError,
            InsufficientDataError,
        ],
    )
    def test_input_errors_are_value_errors(self, exc_class):
        assert issubclass(exc_class, ValueError)
        with pytest.raises(ValueError):
            raise exc_class("test message")

    def test_configuration_is_validation_error(self):
        assert issubclass(ConfigurationError, ValidationError)

    def test_insufficient_data_is_dataset_error(self):
        assert issubclass(InsufficientDataError, DatasetError)

    def test_model_fit_error(self):
        assert issubclass(ModelFitError, TSForecastingError)
        assert issubclass(ModelFitError, RuntimeError)

    def test_not_fitted_is_not_a_value_error(self):
        assert not issubclass(NotFittedError, ValueError)
