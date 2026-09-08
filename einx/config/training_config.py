# -*- coding: utf-8 -*-
"""Training configuration for EINX."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainingConfig:
    """Every hyperparameter of a training run."""

    # --- Run identity ---
    run_name: str = "einx-experimental-run-1"
    model_name: str = "einx-experimental"

    # --- Data ---
    dataset_path: str = "data/processed/train.jsonl"
    val_dataset_path: str = "data/processed/val.jsonl"
    tokenizer_path: str = "data/tokenized/einx-bpe.json"

    # --- Optimisation ---
    batch_size: int = 8
    grad_accum_steps: int = 4          # effective batch = batch_size * grad_accum
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # --- Learning-rate schedule ---
    warmup_steps: int = 100
    lr_schedule: str = "cosine"          # "cosine" | "linear" | "constant"
    min_lr_ratio: float = 0.1             # min_lr = learning_rate * min_lr_ratio

    # --- Training length ---
    max_steps: int = 2000
    max_epochs: int = 0                   # 0 = use max_steps, else stop at this many epochs

    # --- Mixed precision ---
    # "fp32" | "fp16" | "bf16" — fp16/bf16 only on CUDA
    precision: str = "fp32"

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 500
    keep_last_n_checkpoints: int = 3
    resume_from: Optional[str] = None     # checkpoint path or None

    # --- Evaluation ---
    eval_every_steps: int = 250
    eval_steps: int = 50

    # --- Logging ---
    log_every_steps: int = 10
    log_level: str = "INFO"

    # --- Reproducibility ---
    seed: int = 42

    # --- Hardware ---
    device: str = "auto"                  # "auto" | "cpu" | "cuda" | "mps"

    # --- Experiment tracking (optional) ---
    wandb_project: str = ""
    wandb_run_name: str = ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {self.grad_accum_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.max_steps <= 0 and self.max_epochs <= 0:
            raise ValueError("must set either max_steps or max_epochs")
        if self.lr_schedule not in ("cosine", "linear", "constant"):
            raise ValueError(
                f"lr_schedule must be cosine|linear|constant, got {self.lr_schedule!r}"
            )
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(
                f"precision must be fp32|fp16|bf16, got {self.precision!r}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_yaml())


def get_training_config(profile: str = "experimental") -> TrainingConfig:
    """Look up a built-in training profile."""
    if profile == "experimental":
        return TrainingConfig(
            run_name="einx-experimental-run-1",
            model_name="einx-experimental",
            batch_size=8,
            grad_accum_steps=4,
            learning_rate=3e-4,
            max_steps=2000,
            eval_every_steps=250,
            save_every_steps=500,
            precision="fp32",
        )
    if profile == "small":
        return TrainingConfig(
            run_name="einx-small-run-1",
            model_name="einx-small",
            batch_size=4,
            grad_accum_steps=8,
            learning_rate=3e-4,
            max_steps=5000,
            precision="fp32",
        )
    raise KeyError(
        f"unknown training profile: {profile!r}. defined: experimental, small"
    )
