# -*- coding: utf-8 -*-
"""Performance measurement for EINX (spec §28).

Records real metrics — never fabricated:
  * tokens/sec (training throughput)
  * steps/sec
  * peak memory (CUDA only)
  * training loss (live)
  * validation loss (periodic)

Use ``PerformanceMonitor`` to wrap a training loop and get a final
``PerformanceReport`` at the end.  Optional ``--profile`` flag runs
PyTorch's profiler for one step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Final performance report from a training run."""

    n_steps: int = 0
    n_tokens: int = 0
    total_seconds: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_steps_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    initial_train_loss: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    device: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """Track throughput + memory across a training run.

    Usage:
        monitor = PerformanceMonitor(device="cpu")
        monitor.start()
        for step in training_loop:
            monitor.step(n_tokens_in_batch, train_loss=loss.item())
        report = monitor.finish()
        print(report.to_dict())
    """

    def __init__(self, *, device: str = "cpu"):
        self.device = device
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._step_times: List[float] = []
        self._token_counts: List[int] = []
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._peak_memory_mb: float = 0.0
        self._n_steps: int = 0
        self._n_tokens: int = 0

    def start(self) -> None:
        self._start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def step(
        self,
        n_tokens: int,
        *,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        self._n_steps += 1
        self._n_tokens += n_tokens
        self._step_times.append(time.time())
        self._token_counts.append(n_tokens)
        if train_loss is not None:
            self._train_losses.append(train_loss)
        if val_loss is not None:
            self._val_losses.append(val_loss)
        # Update peak memory (cheap on CUDA, no-op on CPU)
        if torch.cuda.is_available():
            self._peak_memory_mb = max(
                self._peak_memory_mb,
                torch.cuda.max_memory_allocated() / (1024 * 1024),
            )

    def finish(self, *, final_val_loss: Optional[float] = None) -> PerformanceReport:
        self._end_time = time.time()
        elapsed = (self._end_time or 0) - (self._start_time or 0)
        avg_tps = self._n_tokens / elapsed if elapsed > 0 else 0.0
        avg_sps = self._n_steps / elapsed if elapsed > 0 else 0.0
        return PerformanceReport(
            n_steps=self._n_steps,
            n_tokens=self._n_tokens,
            total_seconds=elapsed,
            avg_tokens_per_second=avg_tps,
            avg_steps_per_second=avg_sps,
            peak_memory_mb=self._peak_memory_mb,
            initial_train_loss=self._train_losses[0] if self._train_losses else None,
            final_train_loss=self._train_losses[-1] if self._train_losses else None,
            final_val_loss=final_val_loss if final_val_loss is not None else (
                self._val_losses[-1] if self._val_losses else None
            ),
            device=self.device,
        )


def profile_one_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    output_path: str = "experiments/profile.json",
) -> Dict[str, Any]:
    """Run PyTorch's profiler for one forward + backward step.

    Returns a small summary dict.  The full profiler trace is written
    to ``output_path`` for inspection with chrome://tracing.

    Honest: this is best-effort.  PyTorch's profiler output shape can
    change between versions; we capture what's stably available.
    """
    from torch.profiler import profile, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU] + (
            [ProfilerActivity.CUDA] if torch.cuda.is_available() else []
        ),
    ) as prof:
        logits, loss = model(input_ids, targets=targets)
        loss.backward()

    summary: Dict[str, Any] = {
        "n_events": 0,
        "cpu_time_total_us": 0,
        "cuda_time_total_us": 0,
    }
    try:
        events = prof.key_averages()
        summary["n_events"] = len(events)
        summary["cpu_time_total_us"] = sum(e.cpu_time_total for e in events)
        if torch.cuda.is_available():
            summary["cuda_time_total_us"] = sum(e.cuda_time_total for e in events)
    except Exception as exc:
        summary["warning"] = f"could not collect profiler events: {exc}"

    try:
        import os
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(output_path)
        summary["trace_path"] = output_path
    except Exception as exc:
        summary["trace_export_error"] = str(exc)

    return summary
