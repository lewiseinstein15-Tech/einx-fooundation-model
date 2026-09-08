# -*- coding: utf-8 -*-
"""Experiment tracking for EINX (spec §16).

Every training run produces a machine-readable ``experiment.json`` file
in the checkpoint directory, recording everything needed for
reproducibility + audit:

  * timestamp (start + end)
  * model configuration
  * training configuration
  * dataset path + version (if available)
  * tokenizer path + version
  * hardware (device, dtype, CUDA version)
  * software versions (Python, PyTorch, NumPy, EINX)
  * random seed
  * training metrics over time (step, loss, lr, val_loss)

Never fabricate metrics.  Every metric in the experiment record comes
from a real training step.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """One row of training metrics — appended to the experiment record
    every time we log."""

    step: int
    train_loss: float
    learning_rate: float
    val_loss: Optional[float] = None
    elapsed_seconds: float = 0.0
    tokens_per_second: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    """Full experiment record — one per training run."""

    run_name: str
    model_name: str
    started_at: str = ""
    finished_at: str = ""
    status: str = "running"             # running | completed | failed | interrupted

    # Configurations
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    eval_config: Dict[str, Any] = field(default_factory=dict)

    # Data
    dataset_path: str = ""
    val_dataset_path: str = ""
    tokenizer_path: str = ""
    tokenizer_version: str = ""

    # Environment
    hardware: Dict[str, Any] = field(default_factory=dict)
    software: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    # Metrics (time series)
    metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Final summary
    final_step: int = 0
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: float = float("inf")
    total_tokens: int = 0

    # Error info (if status == "failed")
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class ExperimentTracker:
    """Track and persist an experiment record.

    Usage:
        tracker = ExperimentTracker("checkpoints/EINX-Experimental/run-1", cfg)
        tracker.start()
        ...
        tracker.log_metric(step=10, train_loss=8.4, learning_rate=3e-5)
        ...
        tracker.finish(final_train_loss=7.1, final_val_loss=7.3)
    """

    def __init__(
        self,
        experiment_dir: str | Path,
        *,
        run_name: str = "run",
        model_name: str = "einx-experimental",
        model_config: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        eval_config: Optional[Dict] = None,
        dataset_path: str = "",
        tokenizer_path: str = "",
        tokenizer_version: str = "",
        seed: int = 42,
    ):
        self.dir = Path(experiment_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.record = ExperimentRecord(
            run_name=run_name,
            model_name=model_name,
            model_config=model_config or {},
            training_config=training_config or {},
            eval_config=eval_config or {},
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            tokenizer_version=tokenizer_version,
            seed=seed,
        )
        self._start_time: Optional[float] = None
        self._path = self.dir / "experiment.json"

    def start(self) -> None:
        """Mark the experiment as started."""
        self._start_time = datetime.now(timezone.utc).timestamp()
        self.record.started_at = datetime.now(timezone.utc).isoformat()
        self.record.status = "running"
        self._capture_environment()
        self._save()

    def log_metric(
        self,
        *,
        step: int,
        train_loss: float,
        learning_rate: float,
        val_loss: Optional[float] = None,
        elapsed_seconds: float = 0.0,
        tokens_per_second: Optional[float] = None,
    ) -> None:
        """Append a metrics row.  Safe to call many times."""
        m = ExperimentMetrics(
            step=step,
            train_loss=train_loss,
            learning_rate=learning_rate,
            val_loss=val_loss,
            elapsed_seconds=elapsed_seconds,
            tokens_per_second=tokens_per_second,
        )
        self.record.metrics.append(m.to_dict())
        # Update the running best
        if val_loss is not None and val_loss < self.record.best_val_loss:
            self.record.best_val_loss = val_loss
        self._save()

    def finish(
        self,
        *,
        final_step: int,
        final_train_loss: Optional[float] = None,
        final_val_loss: Optional[float] = None,
        total_tokens: int = 0,
        status: str = "completed",
    ) -> None:
        """Mark the experiment as finished.  Persists the final record."""
        self.record.finished_at = datetime.now(timezone.utc).isoformat()
        self.record.status = status
        self.record.final_step = final_step
        self.record.final_train_loss = final_train_loss
        self.record.final_val_loss = final_val_loss
        self.record.total_tokens = total_tokens
        self._save()
        logger.info("experiment record saved to %s", self._path)

    def fail(self, error: str) -> None:
        """Mark the experiment as failed with an error message."""
        self.record.status = "failed"
        self.record.error = error
        self.record.finished_at = datetime.now(timezone.utc).isoformat()
        self._save()

    # ------------------------------------------------------------------
    def _capture_environment(self) -> None:
        """Record hardware + software versions for reproducibility."""
        import torch

        self.record.software = {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": _safe_version("numpy"),
            "platform": platform.platform(),
            "einx": _safe_version("einx"),
        }
        self.record.hardware = {
            "device": _detect_device_str(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mps_available": (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
            "cpu_count": os.cpu_count(),
        }

    def _save(self) -> None:
        """Atomic write — temp file + rename so a crash never leaves a
        half-written experiment.json."""
        tmp = self._path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.record.to_json())
        os.replace(tmp, self._path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


def _detect_device_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"
