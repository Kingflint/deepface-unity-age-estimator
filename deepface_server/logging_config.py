"""Logging configuration helpers."""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a sensible default formatter."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    # Replace any previously attached handlers so reconfiguring is idempotent.
    root.handlers = [handler]
    root.setLevel(numeric)

    # Quiet down chatty third party loggers.
    for noisy in ("werkzeug", "tensorflow", "deepface"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
