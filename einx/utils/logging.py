# -*- coding: utf-8 -*-
"""Logging utilities for EINX."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "einx", level: str = "INFO") -> logging.Logger:
    """Get a configured logger.  Idempotent — safe to call many times."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
