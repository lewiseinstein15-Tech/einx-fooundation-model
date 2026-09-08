# -*- coding: utf-8 -*-
"""Distributed training for EINX (Build 2.1 — hardware-adaptive).

Real DDP support + FSDP implementation + CPU-safe distributed
infrastructure tests via the Gloo backend.

What's actually implemented:

  * **DDP** — full DistributedDataParallel support: per-GPU process
    spawn, distributed sampler, rank-aware logging, rank-aware
    checkpointing (only rank 0 writes), clean shutdown.  Works on
    multi-GPU CUDA environments.

  * **FSDP** — real implementation using ``torch.distributed.fsdp.
    FullyShardedDataParallel``.  Importable + unit-testable without
    CUDA; runtime requires a compatible CUDA environment (BF16-capable
    GPUs, compute capability >= 8.0).

  * **CPU distributed infrastructure** — the same code paths can be
    exercised on CPU using the Gloo backend, so the distributed
    plumbing (init, rank, sampler, cleanup, checkpoint ownership) is
    testable without GPUs.  These are explicitly labelled as
    "CPU distributed infrastructure tests", NOT multi-GPU training
    tests.

Honesty contract (spec §22):
  * DDP multi-GPU runtime has NOT been validated on this hardware
    (no CUDA GPUs in this environment).
  * FSDP GPU runtime has NOT been validated on this hardware.
  * CPU distributed infrastructure (Gloo) IS tested.
  * When real CUDA hardware is available, EINX uses it without rewrite.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device mesh — describes the process layout
# ---------------------------------------------------------------------------


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
        """True if this process should do global logging / checkpointing."""
        return self.rank == 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Process group init / cleanup
# ---------------------------------------------------------------------------


def init_distributed(mesh: Optional[DeviceMesh] = None) -> Optional[Any]:
    """Initialise the distributed process group.

    Returns the process group (or None if single-device).  No-op when
    not in distributed mode.

    Works on CPU (Gloo backend) AND CUDA (NCCL backend).  CPU + Gloo
    is the path used by the CPU distributed infrastructure tests.
    """
    mesh = mesh or detect_mesh()
    if not mesh.is_distributed:
        return None
    if torch.distributed.is_initialized():
        logger.warning("torch.distributed already initialised — skipping init")
        return torch.distributed.group.WORLD

    # Choose backend: NCCL for CUDA, Gloo for CPU.  This lets the same
    # code path run on either — Gloo on CPU is how we test the
    # distributed plumbing without GPUs.
    backend = mesh.backend
    if backend == "nccl" and not torch.cuda.is_available():
        logger.warning(
            "NCCL backend requested but CUDA not available — falling back to Gloo"
        )
        backend = "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        rank=mesh.rank,
        world_size=mesh.world_size,
    )
    # Set the device for this process (CUDA only)
    if torch.cuda.is_available():
        torch.cuda.set_device(mesh.local_rank)
    logger.info(
        "distributed process group initialised (backend=%s, world_size=%d, rank=%d)",
        backend, mesh.world_size, mesh.rank,
    )
    return torch.distributed.group.WORLD


def cleanup_distributed() -> None:
    """Destroy the distributed process group (call at end of training)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()  # make sure everyone is done
        torch.distributed.destroy_process_group()
        logger.info("distributed process group destroyed")


# ---------------------------------------------------------------------------
# Rank-aware utilities — only rank 0 logs / saves / prints
# ---------------------------------------------------------------------------


def is_main_process(mesh: Optional[DeviceMesh] = None) -> bool:
    """True if this process should perform global logging/checkpointing."""
    mesh = mesh or detect_mesh()
    return mesh.is_main_process


def rank_aware_log(msg: str, *, level: int = logging.INFO, mesh: Optional[DeviceMesh] = None) -> None:
    """Log a message only from rank 0.

    Prevents every process from spamming the same log line — a common
    pitfall in naive DDP code.
    """
    if is_main_process(mesh):
        logger.log(level, msg)


def maybe_barrier(mesh: Optional[DeviceMesh] = None) -> None:
    """Synchronise all processes — no-op when not distributed."""
    mesh = mesh or detect_mesh()
    if mesh.is_distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# DistributedSampler helper
# ---------------------------------------------------------------------------


def make_distributed_sampler(
    dataset,
    *,
    shuffle: bool = True,
    seed: int = 42,
    mesh: Optional[DeviceMesh] = None,
):
    """Build a :class:`torch.utils.data.distributed.DistributedSampler`.

    Returns None when not in distributed mode — caller falls back to a
    regular DataLoader with ``shuffle=True``.
    """
    mesh = mesh or detect_mesh()
    if not mesh.is_distributed:
        return None
    from torch.utils.data.distributed import DistributedSampler
    return DistributedSampler(
        dataset,
        num_replicas=mesh.world_size,
        rank=mesh.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Model wrapping — DDP + FSDP
# ---------------------------------------------------------------------------


def wrap_model(
    model: nn.Module,
    *,
    strategy: str = "none",
    mesh: Optional[DeviceMesh] = None,
    device: Optional[str] = None,
) -> nn.Module:
    """Wrap a model for distributed training.

    ``strategy``:
      * "none" — single device (no wrapping) — DEFAULT
      * "ddp"  — DistributedDataParallel (data-parallel)
      * "fsdp" — FullyShardedDataParallel (sharded, for large models)

    The default ("none") is a no-op — single-device training works
    exactly as in Build 1.

    For DDP, the model is moved to the right device (cuda:local_rank
    if CUDA, cpu if Gloo) and wrapped in DDP.

    For FSDP, the model is wrapped in ``FullyShardedDataParallel``.
    FSDP runtime requires a compatible CUDA environment (BF16-capable
    GPUs).  The code is importable + testable without CUDA — calling
    FSDP on CPU raises a clear error.
    """
    mesh = mesh or detect_mesh()

    # Resolve the target device
    if device is None:
        if torch.cuda.is_available():
            device = f"cuda:{mesh.local_rank}"
        else:
            device = "cpu"

    # FSDP — check eagerly so callers don't silently fall back
    if strategy == "fsdp":
        return _wrap_fsdp(model, mesh=mesh, device=device)

    if strategy == "none" or not mesh.is_distributed:
        # Single-device: just move to the target device
        return model.to(torch.device(device))

    if strategy == "ddp":
        return _wrap_ddp(model, mesh=mesh, device=device)

    raise ValueError(
        f"unknown distributed strategy: {strategy!r} (use 'none', 'ddp', or 'fsdp')"
    )


def _wrap_ddp(
    model: nn.Module,
    *,
    mesh: DeviceMesh,
    device: str,
) -> nn.Module:
    """Wrap a model in DistributedDataParallel.

    Works on CUDA (NCCL backend) AND CPU (Gloo backend).  Gloo + CPU
    is the path used by CPU distributed infrastructure tests.
    """
    target_device = torch.device(device)
    model = model.to(target_device)
    device_ids = None
    if target_device.type == "cuda":
        device_ids = [mesh.local_rank]
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=mesh.local_rank if target_device.type == "cuda" else None,
        find_unused_parameters=False,
    )


def _wrap_fsdp(
    model: nn.Module,
    *,
    mesh: DeviceMesh,
    device: str,
) -> nn.Module:
    """Wrap a model in FullyShardedDataParallel.

    FSDP shards model parameters across GPUs — useful for models too
    large to fit on a single GPU.  Runtime requires:
      * CUDA (FSDP doesn't support CPU)
      * BF16-capable GPUs (compute capability >= 8.0) for stable sharded
        mixed-precision training
      * A real distributed process group (world_size >= 2)

    The code is importable + unit-testable without CUDA — calling FSDP
    on CPU raises a clear error explaining what's missing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FSDP requires a CUDA environment. "
            "FSDP implementation is present, but FSDP runtime validation "
            "requires compatible accelerator hardware (BF16-capable GPUs). "
            "Detected: no CUDA devices. "
            "Use strategy='ddp' with backend='gloo' for CPU distributed infrastructure tests."
        )
    if not mesh.is_distributed:
        raise RuntimeError(
            "FSDP requires a distributed process group (world_size >= 2). "
            "Launch with torchrun --nproc_per_node=N or einx distributed-train --nproc N."
        )
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
    except ImportError as exc:
        raise RuntimeError(
            f"FSDP not available in this PyTorch version ({torch.__version__}): {exc}. "
            "FSDP requires PyTorch 1.12+."
        ) from exc
    target_device = torch.device(device)
    model = model.to(target_device)
    # Use FULL_SHARD for maximum memory savings; SHARD_GRAD_OP is a
    # faster alternative if memory is less of a concern.
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=mesh.local_rank,
        use_orig_params=True,
    )


# ---------------------------------------------------------------------------
# torch.compile wrapper (spec §22)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test helpers — for CPU distributed infrastructure tests (spec §9)
# ---------------------------------------------------------------------------


def init_cpu_distributed_for_tests(
    *,
    world_size: int = 2,
    rank: int,
    port: int = 29500,
) -> DeviceMesh:
    """Initialise a CPU-only Gloo process group for infrastructure tests.

    This is for TESTS ONLY — it lets us exercise the distributed
    plumbing (init, rank, sampler, barrier, cleanup) without GPUs.

    Each test process calls this with its own rank; they share the
    same port so they find each other.

    Returns the mesh describing the test layout.
    """
    if torch.distributed.is_initialized():
        # Already init'd — return the mesh describing current state
        return DeviceMesh(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            backend="gloo",
        )
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    mesh = DeviceMesh(world_size=world_size, rank=rank, local_rank=rank, backend="gloo")
    init_distributed(mesh)
    return mesh
