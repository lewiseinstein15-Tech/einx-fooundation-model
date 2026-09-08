# -*- coding: utf-8 -*-
"""Hardware detection for EINX.

Detects the best available compute device (CUDA > MPS > CPU) so the
training/inference code can run on any machine without hardcoding.
The project never assumes a powerful GPU exists — on a laptop CPU,
everything still works, just slower.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about the selected compute device."""

    device: str            # "cpu" | "cuda" | "mps"
    name: str              # human-readable name
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "name": self.name,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "mps_available": self.mps_available,
        }


def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _is_mps_available() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def detect_device(preference: str = "auto") -> str:
    """Pick the best device given the environment.

    ``preference`` may be:
      * "auto" — best available (cuda > mps > cpu)
      * "cpu" / "cuda" / "mps" — forced (validated; falls back to CPU
        if the requested device is unavailable).
    """
    if preference == "auto":
        if is_cuda_available():
            return "cuda"
        if _is_mps_available():
            return "mps"
        return "cpu"
    if preference == "cuda" and not is_cuda_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return "cpu"
    if preference == "mps" and not _is_mps_available():
        logger.warning("MPS requested but not available — falling back to CPU")
        return "cpu"
    return preference


def get_device_info(preference: str = "auto") -> DeviceInfo:
    device = detect_device(preference)
    name = "CPU"
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
    elif device == "mps":
        name = "Apple Silicon GPU (MPS)"
    return DeviceInfo(
        device=device,
        name=name,
        cuda_available=is_cuda_available(),
        cuda_device_count=(_cuda_device_count() if is_cuda_available() else 0),
        mps_available=_is_mps_available(),
    )


def _cuda_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0
