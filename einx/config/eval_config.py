# -*- coding: utf-8 -*-
"""Evaluation configuration for EINX."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    model_name: str = "einx-experimental"
    checkpoint_path: str = ""                # required at eval time
    tokenizer_path: str = "data/tokenized/einx-bpe.json"
    eval_dataset_path: str = "data/processed/test.jsonl"

    # Metrics to compute (subset of: perplexity, loss, latency, memory)
    metrics: List[str] = None

    # Generation evaluation
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # Eval batch size (can be different from training batch size)
    batch_size: int = 4

    # Hardware
    device: str = "auto"

    # Where to write the eval results
    output_path: str = "experiments/eval_results.json"

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["perplexity", "loss", "latency"]

    def validate(self) -> None:
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for evaluation")
        for m in self.metrics:
            if m not in ("perplexity", "loss", "latency", "memory"):
                raise ValueError(f"unknown metric: {m!r}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


def get_eval_config(profile: str = "experimental") -> EvalConfig:
    if profile == "experimental":
        return EvalConfig(
            model_name="einx-experimental",
            metrics=["perplexity", "loss", "latency"],
            max_new_tokens=128,
        )
    raise KeyError(f"unknown eval profile: {profile!r}")
