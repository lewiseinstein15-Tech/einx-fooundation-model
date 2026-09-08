# -*- coding: utf-8 -*-
"""Factory: build a model from a config name or path."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from einx.config.model_config import EINXModelConfig, get_model_config
from einx.model.transformer import EINXTransformer


def build_model_from_config(
    config: Optional[EINXModelConfig] = None,
    *,
    name: str = "einx-experimental",
    config_path: Optional[str] = None,
) -> EINXTransformer:
    """Build an EINX transformer from a config.

    Three call styles:
        build_model_from_config(cfg)            # explicit config
        build_model_from_config(name="...")    # built-in name
        build_model_from_config(config_path="...")  # YAML file
    """
    if config is None:
        if config_path:
            config = EINXModelConfig.from_yaml(config_path)
        else:
            config = get_model_config(name)
    return EINXTransformer(config)
