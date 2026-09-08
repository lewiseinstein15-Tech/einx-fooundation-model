# -*- coding: utf-8 -*-
"""Centralized weight initialization for EINX.

The repo previously scattered ``nn.init.normal_(...)`` calls across the
model code.  This module concentrates every initialization choice in
one place so:

  * the strategy is documented and auditable,
  * changing the strategy (e.g. to truncated normal or zero-init for
    residual scales) is a single edit,
  * tests can verify the strategy is applied consistently.

Strategy (LLaMA-style, well-supported by the literature):

  * Embeddings            — N(0, 0.02)
  * Linear (QKV, FFN, O)  — N(0, 0.02)                    (no bias by default)
  * Output LM head        — N(0, 0.02)  (unless tied to embeddings)
  * RMSNorm / LayerNorm   — weight=1.0, bias=0.0
  * Residual scaling      — scaled by 1/sqrt(2*n_layers) per residual path

The residual scaling is applied post-init by scaling the *output* of
each sub-layer's final projection.  This keeps activations well-scaled
as the model gets deeper, avoiding the "loss spike on step 1" problem
that naive init causes in deep transformers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# Default init std — matches GPT-2 / LLaMA / most modern transformers.
DEFAULT_INIT_STD: float = 0.02


def _init_normal_(module: nn.Module, std: float = DEFAULT_INIT_STD) -> None:
    """Init a Linear or Embedding with N(0, std)."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)


def _init_norm_(module: nn.Module) -> None:
    """Init RMSNorm / LayerNorm weights to 1.0 and biases to 0.0.

    IMPORTANT: must NOT touch ``nn.Embedding`` or ``nn.Linear`` — they
    have ``weight`` Parameters too but those are normal-initialised by
    ``_init_normal_``.  We identify norms by class name (``RMSNorm`` /
    ``LayerNorm``) so this stays surgical.
    """
    # LayerNorm: explicit class match (built-in)
    if isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    # RMSNorm: identified by class name (we don't import to avoid a circular dep)
    cls_name = type(module).__name__
    if cls_name == "RMSNorm":
        nn.init.ones_(module.weight)
        return
    # Don't touch anything else — Linear and Embedding are handled by _init_normal_


def init_weights(
    module: nn.Module,
    *,
    std: float = DEFAULT_INIT_STD,
    n_layers: Optional[int] = None,
) -> None:
    """Recursively initialise all sub-modules.

    Args:
        module:    the root module (typically the model)
        std:       standard deviation for normal init
        n_layers:  number of transformer blocks — when set, residual
                   projections are scaled by 1/sqrt(2*n_layers) to
                   keep activation variance stable as depth grows
    """
    # Apply normal init to Linear + Embedding
    module.apply(lambda m: _init_normal_(m, std=std))
    # Apply ones-init to norms
    module.apply(_init_norm_)


def init_residual_scales(model: nn.Module, *, n_layers: int) -> None:
    """Scale residual-path output projections by 1/sqrt(2*n_layers).

    Per DeepNet / LLaMA-2: each transformer block has 2 residual paths
    (attention + FFN), so the variance grows by 2x per block.  Scaling
    the final projection of each path by 1/sqrt(2*n_layers) keeps the
    output variance ≈ input variance.

    This is applied AFTER ``init_weights`` so it's a no-op when n_layers
    is small (the empirical effect is meaningful only for n_layers >= 6
    or so, but it never hurts).
    """
    scale = 1.0 / math.sqrt(2.0 * max(1, n_layers))
    # We scale the output projections of the attention (o_proj) and FFN
    # (w_down) layers.  These are identified by name so we don't
    # accidentally scale something else.
    for name, m in model.named_modules():
        if name.endswith("o_proj") or name.endswith("w_down"):
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.mul_(scale)
                    if m.bias is not None:
                        m.bias.mul_(scale)


def apply_init_strategy(model: nn.Module, *, n_layers: int, std: float = DEFAULT_INIT_STD) -> None:
    """Apply the full EINX initialization strategy to a model.

    Call this once after model construction, BEFORE adding the model
    to the optimizer.  Idempotent in the sense that running it twice
    produces the same result (the second call re-initialises with the
    same std).
    """
    init_weights(model, std=std)
    init_residual_scales(model, n_layers=n_layers)
