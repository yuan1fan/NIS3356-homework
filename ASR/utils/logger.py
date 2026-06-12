"""Logging utilities."""

from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str = "asr_app", level: int = logging.INFO) -> logging.Logger:
    """Create and return configured logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
