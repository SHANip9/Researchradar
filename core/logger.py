"""
core/logger.py — Structured Logging for ResearchRadar.

Provides a pre-configured logger with consistent formatting, a ``@timed``
decorator for automatic latency tracking, and a context-aware log helper.
"""

import logging
import sys
import time
from functools import wraps
from typing import Callable, TypeVar

from core.config import APP_NAME, APP_VERSION, DEBUG

F = TypeVar("F", bound=Callable)

# ─── Formatter ────────────────────────────────────────────────
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ─── Logger Factory ──────────────────────────────────────────
def _create_logger(name: str) -> logging.Logger:
    """Creates and configures a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        logger.propagate = False
    return logger


# Global application logger
log = _create_logger(APP_NAME)


def get_logger(module_name: str) -> logging.Logger:
    """Returns a child logger scoped to a specific module.

    Usage::

        from core.logger import get_logger
        log = get_logger(__name__)
    """
    return _create_logger(f"{APP_NAME}.{module_name}")


# ─── Performance Decorator ───────────────────────────────────
def timed(func: F) -> F:
    """Decorator that logs function execution time in milliseconds."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info("%s completed in %.1fms", func.__name__, elapsed_ms)
            return result
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.error("%s failed after %.1fms", func.__name__, elapsed_ms)
            raise

    return wrapper  # type: ignore[return-value]


log.info("ResearchRadar %s logger initialized (debug=%s)", APP_VERSION, DEBUG)

