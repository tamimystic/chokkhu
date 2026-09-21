"""Enterprise logging and progress utilities for Chokkhu."""

from __future__ import annotations

import logging
from typing import Any, Iterable

_logger = logging.getLogger("chokkhu")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())


class Logger:
    """Standardized enterprise logger for Chokkhu."""

    @staticmethod
    def info(msg: str) -> None:
        _logger.info(msg)

    @staticmethod
    def error(msg: str) -> None:
        _logger.error(msg)

    @staticmethod
    def warning(msg: str) -> None:
        _logger.warning(msg)

    @staticmethod
    def debug(msg: str) -> None:
        _logger.debug(msg)


def get_logger(name: str = "chokkhu") -> logging.Logger:
    """Get the standard Chokkhu logger."""
    return logging.getLogger(name)


def get_progress_bar(
    iterable: Iterable[Any], desc: str, disable: bool = True
) -> Iterable[Any]:
    """Return a progress bar if tqdm is available, else the raw iterable."""
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, leave=False, disable=disable)
    except ImportError:
        return iterable
