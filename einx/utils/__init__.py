# -*- coding: utf-8 -*-
"""EINX utilities: hardware detection, seeds, logging, registry."""

from einx.utils.hardware import detect_device, get_device_info, is_cuda_available
from einx.utils.seeds import set_seed
from einx.utils.logging import get_logger

__all__ = [
    "detect_device",
    "get_device_info",
    "is_cuda_available",
    "set_seed",
    "get_logger",
]
