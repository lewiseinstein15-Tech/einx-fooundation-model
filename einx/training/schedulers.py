# -*- coding: utf-8 -*-
"""Learning-rate schedulers for EINX."""

from __future__ import annotations

import math
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR


class LRScheduler:
    """Wrapper around torch's LambdaLR with a friendly interface.

    Schedules supported:
      * ``cosine`` — half-cosine decay to ``min_lr_ratio * lr``
      * ``linear`` — linear decay to ``min_lr_ratio * lr``
      * ``constant`` — no decay (still warmup)
    """

    def __init__(
        self,
        optimizer,
        *,
        schedule: str = "cosine",
        warmup_steps: int = 100,
        total_steps: int = 2000,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.schedule = schedule
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            # 1. Warmup: linear from 0 to 1 over warmup_steps
            if step < self.warmup_steps:
                return float(step) / float(self.warmup_steps)
            # 2. After warmup: decay according to the chosen schedule
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            if self.schedule == "cosine":
                cos = 0.5 * (1.0 + math.cos(math.pi * progress))
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos
            if self.schedule == "linear":
                return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - progress)
            if self.schedule == "constant":
                return 1.0
            raise ValueError(f"unknown schedule: {self.schedule!r}")

        self._scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(self) -> None:
        self._scheduler.step()

    def get_last_lr(self) -> list:
        return self._scheduler.get_last_lr()

    def state_dict(self) -> dict:
        return self._scheduler.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._scheduler.load_state_dict(state)


def build_scheduler(
    optimizer,
    *,
    schedule: str = "cosine",
    warmup_steps: int = 100,
    total_steps: int = 2000,
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """Factory: build an LR scheduler by config."""
    return LRScheduler(
        optimizer,
        schedule=schedule,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )
