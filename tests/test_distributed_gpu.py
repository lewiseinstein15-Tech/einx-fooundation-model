# -*- coding: utf-8 -*-
"""Conditional multi-GPU integration tests (spec §10).

These tests run ONLY when >=2 CUDA GPUs are actually available.  On
this CPU-only environment they are SKIPPED with an explicit reason —
NOT marked as passed or failed.

Per spec §10:
  Expected result on current hardware:
  SKIPPED
  Reason: Requires >=2 CUDA GPUs. Detected: 0.

This is correct behavior.  Do not mark as passed.  Do not mark as
failed.  Mark as skipped.

When a real multi-GPU CUDA machine becomes available, these tests
will automatically run — no rewrite needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from einx.utils.hardware import is_cuda_available, _cuda_device_count
from einx.config import EINXModelConfig
from einx.model.transformer import EINXTransformer


# Skip markers — these are the heart of the spec §10 contract.

requires_cuda = pytest.mark.skipif(
    not is_cuda_available(),
    reason=f"Requires >=1 CUDA GPU. Detected: {_cuda_device_count()} GPU(s).",
)

requires_multi_gpu = pytest.mark.skipif(
    not is_cuda_available() or _cuda_device_count() < 2,
    reason=(
        f"Requires >=2 CUDA GPUs for DDP. Detected: "
        f"{_cuda_device_count()} CUDA GPU(s)."
    ),
)


@requires_cuda
def test_single_gpu_forward_pass():
    """Test 1-CUDA-GPU forward pass — runs only when CUDA exists."""
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg).to("cuda")
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device="cuda")
    logits, loss = model(input_ids, targets=input_ids)
    assert logits.device.type == "cuda"
    assert loss.item() > 0


@requires_multi_gpu
def test_ddp_multi_gpu_initialization():
    """Real DDP multi-GPU test — runs only when >=2 CUDA GPUs exist.

    This test verifies that DDP can be initialised across multiple
    GPUs.  It requires the test to be launched via ``torchrun`` with
    ``--nproc_per_node=N`` where N = GPU count.

    SKIPPED on this CPU-only environment.
    """
    # The actual DDP test runs here only on multi-GPU hardware.
    # We use init_distributed + wrap_model(strategy="ddp").
    from einx.training.distributed import detect_mesh, init_distributed, wrap_model, cleanup_distributed
    mesh = detect_mesh()
    assert mesh.is_distributed, "must be launched with torchrun --nproc_per_node=N"
    init_distributed(mesh)
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    model = wrap_model(model, strategy="ddp", mesh=mesh)
    # Verify the model is wrapped in DDP
    assert "DistributedDataParallel" in type(model).__name__
    cleanup_distributed()


@requires_multi_gpu
def test_ddp_multi_gpu_one_step_training():
    """Real DDP multi-GPU training step — runs only when >=2 CUDA GPUs exist.

    Verifies that one step of DDP training works: forward, backward,
    gradient sync, optimizer step.  SKIPPED on this CPU-only environment.
    """
    from einx.training.distributed import detect_mesh, init_distributed, wrap_model, cleanup_distributed
    from einx.training.optimizers import build_optimizer
    mesh = detect_mesh()
    assert mesh.is_distributed
    init_distributed(mesh)
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    model = wrap_model(model, strategy="ddp", mesh=mesh)
    optimizer = build_optimizer(model, lr=1e-3)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=f"cuda:{mesh.local_rank}")
    targets = torch.randint(0, cfg.vocab_size, (2, 16), device=f"cuda:{mesh.local_rank}")
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    cleanup_distributed()


# A test that explicitly reports what was skipped — useful when running
# with -v to see the reason.


def test_multi_gpu_availability_report():
    """Report (don't assert) how many CUDA GPUs are available.

    This test ALWAYS runs — it just reports the detected hardware so
    users can see why the DDP tests above were skipped.
    """
    n_gpus = _cuda_device_count()
    cuda_ok = is_cuda_available()
    if not cuda_ok:
        print(f"\n  CUDA: unavailable")
        print(f"  GPU count: 0")
        print(f"  Multi-GPU DDP tests: SKIPPED (requires >=2 CUDA GPUs)")
        print(f"  FSDP tests: SKIPPED (requires compatible CUDA GPUs)")
    elif n_gpus == 1:
        print(f"\n  CUDA: available")
        print(f"  GPU count: 1")
        print(f"  Multi-GPU DDP tests: SKIPPED (requires >=2 GPUs, got 1)")
        print(f"  FSDP tests: SKIPPED (requires >=2 BF16-capable GPUs)")
    else:
        print(f"\n  CUDA: available")
        print(f"  GPU count: {n_gpus}")
        print(f"  Multi-GPU DDP tests: AVAILABLE — will run")
        # FSDP requires BF16-capable GPUs
        from einx.utils.hardware import _list_gpus
        gpus = _list_gpus()
        bf16_capable = all(g.supports_bf16 for g in gpus)
        if bf16_capable:
            print(f"  FSDP tests: AVAILABLE — will run")
        else:
            print(f"  FSDP tests: SKIPPED (requires BF16-capable GPUs, compute capability >= 8.0)")
