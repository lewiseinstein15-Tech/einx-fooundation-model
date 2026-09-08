# -*- coding: utf-8 -*-
"""Conditional FSDP integration tests (spec §11).

These tests run ONLY when a compatible CUDA environment exists:
  * >=2 CUDA GPUs
  * All GPUs support BF16 (compute capability >= 8.0, Ampere+)

On this CPU-only environment they are SKIPPED with an explicit reason.

Per spec §22:
  "FSDP implementation: present.
   FSDP runtime validation requires compatible accelerator hardware."

Per spec §11:
  "if compatible CUDA environment:
      run FSDP integration tests
   else:
      skip"

The test output identifies why it was skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from einx.utils.hardware import is_cuda_available, _cuda_device_count, _list_gpus
from einx.config import EINXModelConfig
from einx.model.transformer import EINXTransformer


def _fsdp_compatible() -> bool:
    """True iff FSDP can actually run: >=2 CUDA GPUs all BF16-capable."""
    if not is_cuda_available():
        return False
    if _cuda_device_count() < 2:
        return False
    gpus = _list_gpus()
    return all(g.supports_bf16 for g in gpus)


def _skip_reason() -> str:
    if not is_cuda_available():
        return "FSDP requires CUDA. Detected: 0 CUDA GPUs."
    n = _cuda_device_count()
    if n < 2:
        return f"FSDP requires >=2 CUDA GPUs. Detected: {n}."
    gpus = _list_gpus()
    non_bf16 = [g for g in gpus if not g.supports_bf16]
    if non_bf16:
        names = ", ".join(f"{g.name}(cc {g.major}.{g.minor})" for g in non_bf16)
        return (
            f"FSDP requires BF16-capable GPUs (compute capability >= 8.0). "
            f"Non-BF16 GPUs detected: {names}"
        )
    return ""


requires_fsdp = pytest.mark.skipif(
    not _fsdp_compatible(),
    reason=_skip_reason() or "FSDP-compatible environment not detected",
)


def test_fsdp_implementation_present():
    """Verify the FSDP code path is importable (spec §11 + §22).

    This test ALWAYS runs — it doesn't need CUDA.  It just verifies
    that calling wrap_model(strategy="fsdp") on CPU raises the expected
    RuntimeError (not ImportError), proving the FSDP implementation
    is present in the codebase.
    """
    from einx.training.distributed import wrap_model
    from einx.utils.errors import EINXConfigError  # noqa: F401
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    if not is_cuda_available():
        # On CPU: FSDP raises RuntimeError with a clear message
        with pytest.raises(RuntimeError, match="FSDP requires a CUDA environment"):
            wrap_model(model, strategy="fsdp")
    else:
        # On CUDA without distributed init: FSDP raises RuntimeError
        # about needing a process group.  We don't actually init the
        # group here — that's the job of the conditional test below.
        with pytest.raises(RuntimeError):
            wrap_model(model, strategy="fsdp")


@requires_fsdp
def test_fsdp_multi_gpu_initialization():
    """Real FSDP multi-GPU test — runs ONLY with compatible CUDA hardware.

    Verifies that FSDP can wrap a model across multiple BF16-capable
    GPUs.  Requires launch via ``torchrun --nproc_per_node=N``.

    SKIPPED on this environment.
    """
    from einx.training.distributed import detect_mesh, init_distributed, wrap_model, cleanup_distributed
    mesh = detect_mesh()
    assert mesh.is_distributed, "must be launched with torchrun --nproc_per_node=N"
    init_distributed(mesh)
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
        precision="bf16",
    )
    model = EINXTransformer(cfg)
    model = wrap_model(model, strategy="fsdp", mesh=mesh)
    # Verify the model is wrapped in FSDP
    assert "FullySharded" in type(model).__name__ or "FSDP" in type(model).__name__
    cleanup_distributed()


@requires_fsdp
def test_fsdp_multi_gpu_one_step_training():
    """Real FSDP multi-GPU training step — runs ONLY with compatible CUDA.

    Verifies that one step of FSDP training works: forward, backward,
    gradient sync (sharded all-reduce), optimizer step.

    SKIPPED on this environment.
    """
    from einx.training.distributed import detect_mesh, init_distributed, wrap_model, cleanup_distributed
    from einx.training.optimizers import build_optimizer
    mesh = detect_mesh()
    assert mesh.is_distributed
    init_distributed(mesh)
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
        precision="bf16",
    )
    model = EINXTransformer(cfg)
    model = wrap_model(model, strategy="fsdp", mesh=mesh)
    optimizer = build_optimizer(model, lr=1e-3)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16), device=f"cuda:{mesh.local_rank}")
    targets = torch.randint(0, cfg.vocab_size, (2, 16), device=f"cuda:{mesh.local_rank}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    cleanup_distributed()


def test_fsdp_availability_report():
    """Always-runs test that reports whether FSDP can run in this env.

    Per spec §22:
      "FSDP implementation: present.
       FSDP runtime validation requires compatible accelerator hardware."
    """
    print("\n  FSDP implementation: present")
    if _fsdp_compatible():
        print("  FSDP runtime: AVAILABLE — will run integration tests")
    else:
        print(f"  FSDP runtime: NOT TESTED")
        print(f"  Reason: {_skip_reason()}")
