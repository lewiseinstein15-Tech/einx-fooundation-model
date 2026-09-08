# -*- coding: utf-8 -*-
"""EINX training package.

A real PyTorch training loop with:
  * AdamW optimizer
  * Cosine / linear / constant learning-rate schedule with warmup
  * Gradient accumulation
  * Gradient clipping
  * Mixed precision (fp16 / bf16 — CUDA only; fp32 default on CPU)
  * Periodic evaluation
  * Periodic checkpoint saving (with keep-last-N rotation)
  * Resumable: load checkpoint + optimizer state + step counter
  * Reproducible via seed
"""

from einx.training.trainer import EINXTrainer, TrainingState
from einx.training.optimizers import build_optimizer
from einx.training.schedulers import build_scheduler, LRScheduler

__all__ = [
    "EINXTrainer",
    "TrainingState",
    "build_optimizer",
    "build_scheduler",
    "LRScheduler",
]
