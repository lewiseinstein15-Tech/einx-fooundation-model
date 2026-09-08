# -*- coding: utf-8 -*-
"""Optimizer factory for EINX."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    *,
    kind: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Build an optimizer by name.

    Currently supported: ``"adamw"`` (default), ``"adam"``, ``"sgd"``.
    Weight decay is applied to all parameters — modern transformers
    typically decay only the 2D weights (embeddings, linear layers)
    but not biases / layer norms.  We follow that convention.
    """
    # Separate parameters into "decay" (2D weights) and "no-decay"
    # (biases, 1D norms).  This is the LLaMA / GPT-2 convention.
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    kind = kind.lower()
    if kind == "adamw":
        return torch.optim.AdamW(
            param_groups, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
    if kind == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    if kind == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {kind!r} (supported: adamw, adam, sgd)")
