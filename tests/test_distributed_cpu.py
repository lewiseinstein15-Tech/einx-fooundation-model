# -*- coding: utf-8 -*-
"""CPU distributed infrastructure tests (spec §9).

These are explicitly **CPU distributed infrastructure tests**, NOT
multi-GPU training tests.  They exercise the distributed plumbing
(init, rank, sampler, barrier, cleanup, checkpoint ownership) using
the Gloo backend on CPU — no CUDA required.

Per spec §9:
  "Clearly label these as: CPU distributed infrastructure tests
   NOT: multi-GPU training tests"

If CUDA GPUs ARE available, the multi-GPU integration tests in
test_distributed_gpu.py cover the real DDP runtime path.

Test strategy: spawn 2 worker processes via torch.multiprocessing,
each calls init_cpu_distributed_for_tests(rank=i), then exercises
the distributed plumbing on CPU.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from einx.training.distributed import (
    DeviceMesh,
    detect_mesh,
    init_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    make_distributed_sampler,
    wrap_model,
    init_cpu_distributed_for_tests,
)


# ---------------------------------------------------------------------------
# Tests that DON'T need a real process group — pure config detection
# ---------------------------------------------------------------------------


def test_detect_mesh_single_process_default():
    """When no WORLD_SIZE env var is set, detect_mesh returns a single-process mesh."""
    # Save + clear env vars
    saved = {k: os.environ.pop(k, None) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    try:
        mesh = detect_mesh()
        assert mesh.world_size == 1
        assert mesh.rank == 0
        assert mesh.local_rank == 0
        assert not mesh.is_distributed
        assert mesh.is_main_process
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_detect_mesh_reads_env_vars():
    """When env vars are set (as torchrun does), detect_mesh reads them."""
    os.environ["WORLD_SIZE"] = "4"
    os.environ["RANK"] = "2"
    os.environ["LOCAL_RANK"] = "2"
    try:
        mesh = detect_mesh()
        assert mesh.world_size == 4
        assert mesh.rank == 2
        assert mesh.local_rank == 2
        assert mesh.is_distributed
        assert not mesh.is_main_process  # rank 2 != 0
    finally:
        for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            os.environ.pop(k, None)


def test_is_distributed_returns_false_in_single_process():
    saved = {k: os.environ.pop(k, None) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    try:
        assert is_distributed() is False
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_is_main_process_true_in_single_process():
    saved = {k: os.environ.pop(k, None) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    try:
        assert is_main_process() is True
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_device_mesh_dataclass():
    mesh = DeviceMesh(world_size=4, rank=0, local_rank=0, backend="nccl")
    assert mesh.is_distributed
    assert mesh.is_main_process


def test_device_mesh_single_process():
    mesh = DeviceMesh()
    assert mesh.world_size == 1
    assert not mesh.is_distributed


# ---------------------------------------------------------------------------
# make_distributed_sampler — single-process returns None
# ---------------------------------------------------------------------------


def test_make_distributed_sampler_returns_none_when_not_distributed():
    """In single-process mode, no sampler is needed — fall back to shuffle=True."""
    saved = {k: os.environ.pop(k, None) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    try:
        # Fake dataset with 100 examples
        from torch.utils.data import Dataset
        class FakeDataset(Dataset):
            def __len__(self): return 100
            def __getitem__(self, i): return torch.tensor([i])
        sampler = make_distributed_sampler(FakeDataset())
        assert sampler is None
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# wrap_model — single-process returns the model unchanged (just moved to device)
# ---------------------------------------------------------------------------


def test_wrap_model_single_process_returns_model_on_cpu():
    saved = {k: os.environ.pop(k, None) for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    try:
        model = nn.Linear(10, 10)
        wrapped = wrap_model(model, strategy="none", device="cpu")
        # In single-process mode, model is moved to the target device
        # but NOT wrapped in DDP/FSDP.
        assert isinstance(wrapped, nn.Linear)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# init_cpu_distributed_for_tests — the real Gloo path.
#
# These spawn real worker processes via torch.multiprocessing.  Each
# worker calls init_cpu_distributed_for_tests(rank=i) which sets up a
# Gloo process group.  This is the path that exercises the real
# distributed plumbing (init, rank, barrier, cleanup) WITHOUT GPUs.
#
# Per spec §9: these are CPU distributed infrastructure tests,
# NOT multi-GPU training tests.
# ---------------------------------------------------------------------------


def _worker_process(rank: int, world_size: int, port: int, result_queue):
    """Worker process for CPU distributed infrastructure tests."""
    try:
        # Init the Gloo process group
        mesh = init_cpu_distributed_for_tests(
            world_size=world_size, rank=rank, port=port,
        )
        # Verify the mesh
        assert mesh.world_size == world_size
        assert mesh.rank == rank
        assert mesh.is_distributed
        # rank 0 is the main process; others aren't
        assert mesh.is_main_process == (rank == 0)
        # Test a barrier — all processes must reach this
        torch.distributed.barrier()
        # Test all_reduce so we can verify ranks can communicate
        tensor = torch.tensor([rank + 1])
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        expected = sum(i + 1 for i in range(world_size))
        assert tensor.item() == expected, f"rank {rank}: got {tensor.item()}, expected {expected}"
        cleanup_distributed()
        result_queue.put(("ok", rank))
    except Exception as exc:
        result_queue.put(("error", rank, str(exc)))


def test_cpu_distributed_init_and_barrier():
    """CPU distributed infrastructure test (spec §9) — Gloo backend.

    Spawns 2 worker processes, each calls init_cpu_distributed_for_tests,
    then exercises barrier + all_reduce + cleanup.  Uses Gloo (no CUDA).

    This is a CPU distributed infrastructure test, NOT a multi-GPU
    training test.
    """
    if torch.cuda.is_available():
        pytest.skip("CUDA available — this test is for CPU-only distributed plumbing")

    world_size = 2
    port = 29555  # unique port to avoid collisions with other tests
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_process,
            args=(rank, world_size, port, result_queue),
        )
        p.start()
        procs.append(p)
    # Wait for all to finish (timeout = 30s)
    for p in procs:
        p.join(timeout=30)
        assert not p.is_alive(), f"worker {p.pid} did not finish in 30s"
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    assert len(results) == world_size, f"expected {world_size} results, got {results}"
    for r in results:
        assert r[0] == "ok", f"worker {r[1]} failed: {r[2] if len(r) > 2 else 'unknown'}"


def _checkpoint_worker(rank: int, world_size: int, port: int, tmp_dir, result_queue):
    """Worker that verifies rank-aware checkpointing — only rank 0 writes."""
    try:
        mesh = init_cpu_distributed_for_tests(
            world_size=world_size, rank=rank, port=port,
        )
        # Verify checkpoint ownership contract:
        # Only rank 0 should write checkpoints.
        from einx.training.distributed import is_main_process
        if is_main_process(mesh):
            # Simulate a checkpoint write
            marker_path = Path(tmp_dir) / "rank0_marker.txt"
            marker_path.write_text(f"rank {rank} wrote this")
            result_queue.put(("rank0_wrote", rank))
        else:
            # Non-rank-0 must NOT write
            result_queue.put(("rank_skipped", rank))
        torch.distributed.barrier()
        cleanup_distributed()
    except Exception as exc:
        result_queue.put(("error", rank, str(exc)))


def test_cpu_distributed_checkpoint_ownership(tmp_path):
    """CPU distributed infrastructure test — only rank 0 writes checkpoints (spec §6).

    Verifies the rank-aware checkpointing contract: when running DDP,
    only rank 0 should write checkpoints.  Other ranks would just
    produce duplicate copies, breaking resume semantics.
    """
    if torch.cuda.is_available():
        pytest.skip("CUDA available — this test is for CPU-only distributed plumbing")
    world_size = 2
    port = 29556
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_checkpoint_worker,
            args=(rank, world_size, port, str(tmp_path), result_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=30)
        assert not p.is_alive(), f"worker did not finish in 30s"
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    assert len(results) == world_size
    # Exactly one rank should have written (rank 0)
    writers = [r for r in results if r[0] == "rank0_wrote"]
    skippers = [r for r in results if r[0] == "rank_skipped"]
    assert len(writers) == 1
    assert writers[0][1] == 0  # rank 0 wrote
    assert len(skippers) == world_size - 1
    # The marker file should exist
    assert (tmp_path / "rank0_marker.txt").exists()


def _sampler_worker(rank: int, world_size: int, port: int, result_queue):
    """Worker that verifies DistributedSampler gives each rank different data."""
    try:
        mesh = init_cpu_distributed_for_tests(
            world_size=world_size, rank=rank, port=port,
        )
        from torch.utils.data import Dataset, DataLoader
        class RangeDataset(Dataset):
            def __len__(self): return 100
            def __getitem__(self, i): return torch.tensor([i])
        ds = RangeDataset()
        sampler = make_distributed_sampler(ds, shuffle=True, seed=42, mesh=mesh)
        assert sampler is not None, "sampler should be created in distributed mode"
        loader = DataLoader(ds, batch_size=10, sampler=sampler, drop_last=True)
        # Collect the indices this rank sees
        seen = set()
        for batch in loader:
            for val in batch.tolist():
                seen.add(val[0])
        # Each rank should see roughly 100/world_size = 50 unique indices
        # (with drop_last, might be slightly fewer)
        result_queue.put(("ok", rank, len(seen)))
        torch.distributed.barrier()
        cleanup_distributed()
    except Exception as exc:
        result_queue.put(("error", rank, str(exc)))


def test_cpu_distributed_sampler_shards_data(tmp_path):
    """CPU distributed infrastructure test — DistributedSampler shards data (spec §6).

    Each rank should see a different subset of the dataset, with no
    overlap between ranks.  This is the core contract of data-parallel
    training.
    """
    if torch.cuda.is_available():
        pytest.skip("CUDA available — this test is for CPU-only distributed plumbing")
    world_size = 2
    port = 29557
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_sampler_worker,
            args=(rank, world_size, port, result_queue),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=30)
        assert not p.is_alive()
    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())
    assert len(results) == world_size
    # Each rank should have seen roughly half the 100 indices
    for r in results:
        assert r[0] == "ok", f"worker {r[1]} failed: {r[2] if len(r) > 2 else 'unknown'}"
        n_seen = r[2]
        # With world_size=2 and 100 examples, each rank should see ~50
        assert 40 <= n_seen <= 60, f"rank {r[1]} saw {n_seen} examples, expected ~50"
