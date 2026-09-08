# -*- coding: utf-8 -*-
"""Reproducibility — seed management for EINX."""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Note: full determinism is not guaranteed on CUDA (cuDNN non-determinism),
    but this gets us as close as practical.  Same seed + same hardware +
    same config = same training run on CPU.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort deterministic mode — costs some performance on GPU
        # but is essential for reproducible research.
        # Comment out if you need max speed.
        # torch.use_deterministic_algorithms(True)
    except ImportError:
        pass
