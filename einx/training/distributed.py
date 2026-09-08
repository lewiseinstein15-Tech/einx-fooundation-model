# -*- coding: utf-8 -*-
"""Distributed training abstraction for EINX (spec §23).

Build 1 was single-device only.  This module adds a clean abstraction
for distributed training so the trainer can later use DDP / FSDP
WITHOUT rewriting its core logic.

Current state (HONEST):
  * Single-device mode is the default and is fully implemented + tested.
  * The interface for multi-device mode is defined but the actual
    DDP/FSDP wrapping is NOT yet wired into the trainer — that's Phase 3
    work because it requires multi-GPU hardware to test against.

The abstraction:
  * ``DeviceMesh`` describes the layout (single / multi-GPU / multi-node)
  * ``DistributedStrategy`` selects the strategy (none / ddp / fsdp)
  * ``wrap_model(model, strategy)`` applies the right wrapper
  * ``is_distributed()`` tells the trainer whether to use distributed
    samplers / all-reduce / etc.

When distributed mode is NOT active, every function is a no-op so the
trainer runs exactly as in Build 1.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class DeviceMesh:
    """Describes the device layout for a training run.

    For single-device: ``world_size=1``, ``rank=0``, ``local_rank=0``.
    For multi-GPU single-node: ``world_size=N``, ``rank=0..N-1``,
    ``local_rank=rank``.
    For multi-node: ``world_size=N*M``, ``rank=0..N*M-1``,
    ``local_rank=rank % N``.
    """

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"             # "nccl" (CUDA) | "gloo" (CPU) | "mpi"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def detect_mesh() -> DeviceMesh:
    """Detect the distributed environment from env vars.

    Reads the standard PyTorch env vars (``WORLD_SIZE``, ``RANK``,
    ``LOCAL_RANK``) set by ``torchrun`` / ``torch.distributed.launch``.

    Returns a single-process mesh if no distributed env is detected.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("EINX_DIST_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    mesh = DeviceMesh(world_size=world_size, rank=rank, local_rank=local_rank, backend=backend)
    if mesh.is_distributed:
        logger.info(
            "distributed mesh detected: world_size=%d rank=%d local_rank=%d backend=%s",
            world_size, rank, local_rank, backend,
        )
    return mesh


def is_distributed() -> bool:
    """True if running in distributed mode (world_size > 1)."""
    return detect_mesh().is_distributed


def init_distributed(mesh: Optional[DeviceMesh] = None) -> Optional[Any]:
    """Initialise the distributed process group.

    Returns the process group (or None if single-device).  No-op when
    not in distributed mode.
    """
    mesh = mesh or detect_mesh()
    if not mesh.is_distributed:
        return None
    if torch.distributed.is_initialized():
        logger.warning("torch.distributed already initialised — skipping init")
        return torch.distributed.group.WORLD
    torch.distributed.init_process_group(
        backend=mesh.backend,
        rank=mesh.rank,
        world_size=mesh.world_size,
    )
    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(mesh.local_rank)
    logger.info("distributed process group initialised (backend=%s)", mesh.backend)
    return torch.distributed.group.WORLD


def wrap_model(
    model: nn.Module,
    *,
    strategy: str = "none",
    mesh: Optional[DeviceMesh] = None,
) -> nn.Module:
    """Wrap a model for distributed training.

    ``strategy``:
      * "none" — single device (no wrapping) — DEFAULT
      * "ddp"  — DistributedDataParallel (data-parallel, simplest multi-GPU)
      * "fsdp" — FullyShardedDataParallel (sharded, for large models) — Phase 3

    The default ("none") is a no-op — the model is returned unchanged
    so single-device training works exactly as in Build 1.
    """
    mesh = mesh or detect_mesh()
    if strategy == "fsdp":
        # FSDP wrapping is non-trivial — needs to know which modules to
        # shard.  This is genuinely Phase 3 work; the abstraction is in
        # place so the trainer can call ``wrap_model(model, strategy="fsdp")``
        # without rewriting its core loop.  Raise eagerly so callers
        # don't silently fall back to no-op wrapping.
        raise NotImplementedError(
            "FSDP strategy is planned for Phase 3 — not yet implemented. "
            "Use strategy='ddp' or strategy='none'."
        )
    if strategy == "none" or not mesh.is_distributed:
        return model
    if strategy == "ddp":
        device = torch.device(f"cuda:{mesh.local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[mesh.local_rank] if torch.cuda.is_available() else None,
        )
    raise ValueError(f"unknown distributed strategy: {strategy!r} (use 'none', 'ddp', or 'fsdp')")


def maybe_compile_model(
    model: nn.Module,
    *,
    enabled: bool = False,
    backend: str = "inductor",
) -> nn.Module:
    """Optionally ``torch.compile`` a model (spec §22).

    ``enabled=False`` is the default — no compilation, no overhead, no
    graph breaks.  This is the safe path for development.

    When ``enabled=True`` AND PyTorch supports ``torch.compile`` (2.0+),
    the model is compiled with the given backend.  Compilation is
    applied at the top-level ``EINXTransformer`` rather than per-module,
    matching PyTorch's recommendation.

    Compilation is best enabled via the ``--compile`` CLI flag for
    production training runs where the warmup cost is amortised.
    """
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        logger.warning(
            "torch.compile not available in this PyTorch version (%s) — "
            "returning uncompiled model",
            torch.__version__,
        )
        return model
    try:
        compiled = torch.compile(model, backend=backend)
        logger.info("model compiled with backend=%s", backend)
        return compiled
    except Exception as exc:
        # Compilation can fail for many reasons (dynamic shapes, custom
        # ops, etc.).  Fall back to the uncompiled model rather than
        # crashing the run.
        logger.warning(
            "torch.compile failed (%s) — falling back to uncompiled model",
            type(exc).__name__,
        )
        return model


def cleanup_distributed() -> None:
    """Destroy the distributed process group (call at end of training)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("distributed process group destroyed")
