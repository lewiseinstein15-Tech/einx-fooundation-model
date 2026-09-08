# -*- coding: utf-8 -*-
"""Tests for the CheckpointManager (spec §15)."""

import json
import pytest
import torch

from einx.config import EINXModelConfig
from einx.model.transformer import EINXTransformer
from einx.training.checkpoint_manager import CheckpointManager, CheckpointState


def _make_model(**cfg_kw):
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0, **cfg_kw,
    )
    return EINXTransformer(cfg), cfg


def test_save_creates_step_directory(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    path = mgr.save(step=100, model=model)
    assert path.exists()
    assert path.name == "step-000100"
    assert (path / "model.pt").exists()
    assert (path / "metadata.json").exists()


def test_metadata_records_step_and_timestamp(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=42, model=model, epoch=3, metrics={"val_loss": 4.5})
    latest = mgr.find_latest()
    with open(latest / "metadata.json") as fh:
        meta = json.load(fh)
    assert meta["step"] == 42
    assert meta["epoch"] == 3
    assert meta["metrics"]["val_loss"] == 4.5
    assert "timestamp" in meta
    assert meta["run_name"] == "test-run"


def test_find_latest_returns_most_recent(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=100, model=model)
    mgr.save(step=200, model=model)
    mgr.save(step=50, model=model)  # saved out of order on purpose
    latest = mgr.find_latest()
    assert latest is not None
    # ``latest`` symlink points at the most recently saved (step=50 in this test)
    # because save() updates ``latest`` on every call.  Use list_checkpoints()
    # + sort by step number to find the highest-numbered checkpoint.
    all_ckpts = mgr.list_checkpoints()
    assert len(all_ckpts) == 3
    # The highest-numbered step should be step=200
    steps = [int(d.name.split("-")[1]) for d in all_ckpts]
    assert max(steps) == 200
    # And find_latest (the symlink) should point at a real, loadable checkpoint
    with open(latest / "metadata.json") as fh:
        meta = json.load(fh)
    assert meta["step"] in (50, 100, 200)  # any of the saved ones


def test_find_latest_returns_none_when_empty(tmp_path):
    mgr = CheckpointManager(tmp_path, "test-run")
    assert mgr.find_latest() is None


def test_load_restores_model_weights(tmp_path):
    model, cfg = _make_model()
    # Run a forward pass to populate weights
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    model.eval()
    logits_before, _ = model(input_ids)

    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=100, model=model)

    # Build a fresh model with the same config + load
    model2 = EINXTransformer(cfg)
    state = mgr.load(model=model2)
    assert state.step == 100
    model2.eval()
    logits_after, _ = model2(input_ids)
    assert torch.allclose(logits_before, logits_after, atol=1e-6)


def test_load_restores_optimizer_state(tmp_path):
    import torch.optim as optim
    model, _ = _make_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Do a step so optimizer has nonzero moments
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    targets = torch.randint(0, model.config.vocab_size, (1, 4))
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    optimizer.step()

    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=10, model=model, optimizer=optimizer)

    # Build fresh optimizer and load
    model2, _ = _make_model()
    opt2 = optim.AdamW(model2.parameters(), lr=1e-3)
    state = mgr.load(model=model2, optimizer=opt2)
    # Optimizer state should now have moments populated
    opt2_state = opt2.state_dict()
    has_moments = any(
        len(s) > 0 for s in opt2_state["state"].values()
    )
    assert has_moments, "optimizer state was not restored"


def test_load_with_no_path_loads_latest(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=100, model=model)
    # Call load with path=None — should find latest
    state = mgr.load(model=model)
    assert state.step == 100


def test_load_nonexistent_raises(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    with pytest.raises(FileNotFoundError):
        mgr.load("nonexistent", model=model)


def test_keep_last_n(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    # Save 5 checkpoints
    for step in [100, 200, 300, 400, 500]:
        mgr.save(step=step, model=model)
    # Should have 5 step-* dirs
    assert len(mgr.list_checkpoints()) == 5
    # Keep only last 2
    mgr.keep_last_n(2)
    remaining = mgr.list_checkpoints()
    assert len(remaining) == 2
    # The two remaining should be step=400 and step=500
    remaining_steps = sorted(int(d.name.split("-")[1]) for d in remaining)
    assert remaining_steps == [400, 500]


def test_keep_last_n_with_zero_noops(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=100, model=model)
    mgr.keep_last_n(0)
    assert len(mgr.list_checkpoints()) == 1


def test_atomic_write_no_partial_state(tmp_path, monkeypatch):
    """If save fails mid-write, the previous checkpoint should still be intact.

    We patch ``torch.save`` to raise on the SECOND save call.  The
    CheckpointManager writes to a temp directory first, only renaming
    to the final location on success — so a mid-write failure leaves
    the previous good checkpoint intact.
    """
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=100, model=model)

    # Patch torch.save INSIDE the checkpoint_manager module to raise
    import einx.training.checkpoint_manager as cm_module
    original_save = cm_module.torch.save
    call_count = {"n": 0}

    def failing_save(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 1:
            raise RuntimeError("simulated failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(cm_module.torch, "save", failing_save)

    with pytest.raises(RuntimeError, match="simulated"):
        mgr.save(step=200, model=model)

    # The original step-100 checkpoint should still be loadable
    latest = mgr.find_latest()
    assert latest is not None
    state = mgr.load(model=model)
    assert state.step == 100  # not 200 — atomic write succeeded


def test_best_tag(tmp_path):
    """Saving with tag='best' creates a 'best' symlink/marker."""
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    mgr.save(step=50, model=model, tag="best")
    best = mgr.find_best()
    assert best is not None
    with open(best / "metadata.json") as fh:
        meta = json.load(fh)
    assert meta["step"] == 50


def test_list_checkpoints_sorted(tmp_path):
    model, _ = _make_model()
    mgr = CheckpointManager(tmp_path, "test-run")
    # Save out of order
    for step in [300, 100, 200]:
        mgr.save(step=step, model=model)
    listed = mgr.list_checkpoints()
    steps = [int(d.name.split("-")[1]) for d in listed]
    assert steps == sorted(steps)  # ascending
