from __future__ import annotations


class ChokkhuError(Exception):
    """Base exception for all Chokkhu errors."""

    pass


class DataLoadError(ChokkhuError):
    """Raised when data loading fails."""

    pass


class InvalidFormatError(ChokkhuError):
    """Raised when an input or file format is invalid."""

    pass


class NotFittedError(ChokkhuError, ValueError, AttributeError):
    """Exception raised when an estimator is used before fitting."""

    pass


class ConvergenceWarning(UserWarning):
    """Warning raised when an iterative optimization algorithm does not converge."""

    pass
