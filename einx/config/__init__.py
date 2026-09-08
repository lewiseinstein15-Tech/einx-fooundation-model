# -*- coding: utf-8 -*-
"""EINX configuration package.

Typed dataclasses for model, training, and evaluation configuration.
Every EINX subsystem reads its settings from these objects — no
hardcoded magic numbers anywhere else in the codebase.
"""

from einx.config.model_config import EINXModelConfig, get_model_config
from einx.config.training_config import TrainingConfig, get_training_config
from einx.config.eval_config import EvalConfig, get_eval_config
from einx.config.runtime_config import RuntimeConfig, ResolvedRuntime, get_default_runtime_config

__all__ = [
    "EINXModelConfig",
    "TrainingConfig",
    "EvalConfig",
    "RuntimeConfig",
    "ResolvedRuntime",
    "get_model_config",
    "get_training_config",
    "get_eval_config",
    "get_default_runtime_config",
]
