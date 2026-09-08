# -*- coding: utf-8 -*-
"""EINX model package — Transformer architecture.

A real, configurable decoder-only transformer for language modeling.
Built on PyTorch — no auto-regressive wrappers, no HuggingFace
abstractions.  Every component is inspectable and replaceable.
"""

from einx.model.transformer import EINXTransformer
from einx.model.layers import (
    RMSNorm,
    LayerNorm,
    RotaryPositionEmbedding,
    MultiHeadAttention,
    FeedForward,
)
from einx.model.config_builder import build_model_from_config

__all__ = [
    "EINXTransformer",
    "RMSNorm",
    "LayerNorm",
    "RotaryPositionEmbedding",
    "MultiHeadAttention",
    "FeedForward",
    "build_model_from_config",
]
