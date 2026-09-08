# -*- coding: utf-8 -*-
"""EINX — Lewis Einstein's experimental intelligence architecture.

A modular foundation-model project.  Real engineering, not a UI
wrapper.  See ``README.md`` for the full project overview.

Public API:
    from einx import get_model_config, EINXTransformer, BPETokenizer, EINXGenerator

Submodules are lazy-imported to keep top-level import fast (we only
import torch when actually building a model).
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export the public API at the top level for convenience.
from einx.config import (
    EINXModelConfig,
    TrainingConfig,
    EvalConfig,
    get_model_config,
    get_training_config,
    get_eval_config,
)


def __getattr__(name: str):
    """Lazy-import heavy submodules so ``import einx`` is fast."""
    if name == "EINXTransformer":
        from einx.model.transformer import EINXTransformer
        return EINXTransformer
    if name == "BPETokenizer":
        from einx.tokenizer.bpe import BPETokenizer
        return BPETokenizer
    if name == "EINXGenerator":
        from einx.inference.generator import EINXGenerator
        return EINXGenerator
    if name == "EINXTrainer":
        from einx.training.trainer import EINXTrainer
        return EINXTrainer
    if name == "EINXEvaluator":
        from einx.evaluation.evaluator import EINXEvaluator
        return EINXEvaluator
    raise AttributeError(f"module 'einx' has no attribute {name!r}")


__all__ = [
    "EINXModelConfig",
    "TrainingConfig",
    "EvalConfig",
    "get_model_config",
    "get_training_config",
    "get_eval_config",
    # Lazy-loaded:
    "EINXTransformer",
    "BPETokenizer",
    "EINXGenerator",
    "EINXTrainer",
    "EINXEvaluator",
    "__version__",
]
