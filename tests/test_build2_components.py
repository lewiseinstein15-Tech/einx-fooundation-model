# -*- coding: utf-8 -*-
"""Tests for sequence packing, streaming dataset, init, errors, distributed, performance, experiment tracker (Build 2 components)."""

import json
import pytest
import torch

from einx.config import EINXModelConfig
from einx.tokenizer.bpe import BPETokenizer
from einx.data.dataset import PackedDataset, StreamingTextDataset, write_jsonl
from einx.model.transformer import EINXTransformer
from einx.model.init import (
    init_weights,
    init_residual_scales,
    apply_init_strategy,
    DEFAULT_INIT_STD,
)
from einx.utils.errors import (
    EINXConfigError,
    validate_model_config_friendly,
    validate_training_config_friendly,
    EINXRuntimeError,
)
from einx.training.distributed import (
    DeviceMesh,
    detect_mesh,
    is_distributed,
    wrap_model,
    maybe_compile_model,
)
from einx.utils.performance import PerformanceMonitor, PerformanceReport
from einx.training.experiment_tracker import ExperimentTracker, ExperimentRecord


# ---------------------------------------------------------------------------
# PackedDataset
# ---------------------------------------------------------------------------


def test_packed_dataset_basic():
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(20)]
    tok.train(texts, vocab_size=400, verbose=False)
    ds = PackedDataset(texts, tok, context_length=32)
    assert len(ds) > 0
    input_ids, targets = ds[0]
    assert input_ids.shape == (32,)
    assert targets.shape == (32,)
    # Target should be input shifted by 1
    assert torch.equal(input_ids[1:], targets[:-1])


def test_packed_dataset_efficiency():
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(20)]
    tok.train(texts, vocab_size=400, verbose=False)
    ds = PackedDataset(texts, tok, context_length=32)
    # Packing efficiency is 1.0 because we never pad — only truncate the tail
    assert ds.packing_efficiency() == 1.0


def test_packed_dataset_n_tokens():
    tok = BPETokenizer()
    texts = ["the cat sat on the mat", "the dog ran in the park"] * 20
    tok.train(texts, vocab_size=400, verbose=False)
    ds = PackedDataset(texts, tok, context_length=16)
    assert ds.n_tokens() > 0


# ---------------------------------------------------------------------------
# StreamingTextDataset
# ---------------------------------------------------------------------------


def test_streaming_dataset_iterates(tmp_path):
    path = tmp_path / "stream.jsonl"
    write_jsonl([{"text": "first"}, {"text": "second"}, {"text": "third"}], path)
    ds = StreamingTextDataset(path)
    texts = list(ds)
    assert texts == ["first", "second", "third"]


def test_streaming_dataset_skips_malformed(tmp_path):
    path = tmp_path / "stream.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"text": "good"}) + "\n")
        fh.write("not json\n")
        fh.write(json.dumps({"no_text": "missing"}) + "\n")
        fh.write(json.dumps({"text": "good2"}) + "\n")
    ds = StreamingTextDataset(path)
    texts = list(ds)
    assert texts == ["good", "good2"]


def test_streaming_dataset_count_lines(tmp_path):
    path = tmp_path / "stream.jsonl"
    write_jsonl([{"text": "a"}, {"text": "b"}, {"text": "c"}], path)
    ds = StreamingTextDataset(path)
    assert ds.count_lines() == 3


def test_streaming_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        StreamingTextDataset(tmp_path / "nonexistent.jsonl")


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------


def test_init_weights_normal_std():
    """After init_weights, Linear weights should follow N(0, std)."""
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    # Embedding weight should have std ~0.02 (within tolerance)
    emb_std = model.token_embedding.weight.std().item()
    assert 0.015 < emb_std < 0.025


def test_init_residual_scales_projections():
    """init_residual_scales should scale o_proj and w_down by 1/sqrt(2*L)."""
    import math
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=4, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    # Manually reset o_proj weights to all 1.0
    with torch.no_grad():
        model.blocks[0].attn.o_proj.weight.fill_(1.0)
    init_residual_scales(model, n_layers=4)
    expected_scale = 1.0 / math.sqrt(2.0 * 4.0)
    actual = model.blocks[0].attn.o_proj.weight[0, 0].item()
    assert abs(actual - expected_scale) < 1e-5


def test_apply_init_strategy_idempotent():
    """Running init twice produces the same statistical distribution."""
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=2, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    apply_init_strategy(model, n_layers=2)
    std1 = model.token_embedding.weight.std().item()
    apply_init_strategy(model, n_layers=2)
    std2 = model.token_embedding.weight.std().item()
    # Re-init produces similar std (different actual values, similar spread)
    assert abs(std1 - std2) < 0.005


# ---------------------------------------------------------------------------
# Friendly errors
# ---------------------------------------------------------------------------


def test_friendly_error_for_bad_head_dim():
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=64, n_heads=4, head_dim=17,  # 4*17 != 64
        max_context_length=32, ffn_dim=128, dropout=0.0,
    )
    with pytest.raises(EINXConfigError) as exc:
        validate_model_config_friendly(cfg)
    assert "hidden_dim" in str(exc.value)
    assert "hint" in str(exc.value).lower()


def test_friendly_error_for_bad_precision():
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
        precision="int8",
    )
    with pytest.raises(EINXConfigError) as exc:
        validate_model_config_friendly(cfg)
    assert "precision" in str(exc.value).lower()


def test_friendly_error_for_unknown_positional():
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
        positional_encoding="invalid",
    )
    with pytest.raises(EINXConfigError) as exc:
        validate_model_config_friendly(cfg)
    assert "rope" in str(exc.value).lower()


def test_friendly_error_for_bad_batch_size():
    from einx.config import TrainingConfig
    cfg = TrainingConfig(batch_size=0, max_steps=10)
    with pytest.raises(EINXConfigError) as exc:
        validate_training_config_friendly(cfg)
    assert "batch_size" in str(exc.value).lower()


def test_friendly_error_for_bad_lr():
    from einx.config import TrainingConfig
    cfg = TrainingConfig(learning_rate=-0.1, max_steps=10)
    with pytest.raises(EINXConfigError) as exc:
        validate_training_config_friendly(cfg)
    assert "learning_rate" in str(exc.value).lower()


def test_einx_runtime_error_with_hint():
    err = EINXRuntimeError("disk full", hint="free up space and retry")
    assert "disk full" in str(err)
    assert "free up space" in str(err)


# ---------------------------------------------------------------------------
# Distributed abstraction
# ---------------------------------------------------------------------------


def test_device_mesh_single_process():
    mesh = DeviceMesh(world_size=1, rank=0, local_rank=0)
    assert not mesh.is_distributed
    assert mesh.is_main_process


def test_detect_mesh_defaults_single():
    # No env vars set → single process
    import os
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(k, None)
    mesh = detect_mesh()
    assert mesh.world_size == 1
    assert not mesh.is_distributed


def test_is_distributed_false_in_tests():
    assert is_distributed() is False


def test_wrap_model_none_strategy_is_noop():
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    # In single-device mode, wrap_model returns the model unchanged
    out = wrap_model(model, strategy="none")
    assert out is model


def test_maybe_compile_model_disabled_by_default():
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    out = maybe_compile_model(model, enabled=False)
    assert out is model  # unchanged when disabled


def test_wrap_model_fsdp_not_implemented():
    cfg = EINXModelConfig(
        vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    with pytest.raises(NotImplementedError, match="Phase 3"):
        wrap_model(model, strategy="fsdp")


# ---------------------------------------------------------------------------
# Performance monitor
# ---------------------------------------------------------------------------


def test_performance_monitor_basic():
    mon = PerformanceMonitor(device="cpu")
    mon.start()
    mon.step(100, train_loss=8.0)
    mon.step(100, train_loss=7.5)
    report = mon.finish()
    assert report.n_steps == 2
    assert report.n_tokens == 200
    assert report.initial_train_loss == 8.0
    assert report.final_train_loss == 7.5


def test_performance_report_to_dict():
    # Construct with all fields explicitly — avg_tokens_per_second is
    # computed in PerformanceMonitor.finish(), not in the dataclass init.
    report = PerformanceReport(
        n_steps=5,
        n_tokens=1000,
        total_seconds=10.0,
        avg_tokens_per_second=100.0,
    )
    d = report.to_dict()
    assert d["n_steps"] == 5
    assert d["n_tokens"] == 1000
    assert d["avg_tokens_per_second"] == 100.0


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------


def test_experiment_tracker_writes_json(tmp_path):
    tracker = ExperimentTracker(
        tmp_path / "exp",
        run_name="test-run",
        model_name="einx-experimental",
    )
    tracker.start()
    tracker.log_metric(step=10, train_loss=8.4, learning_rate=3e-5)
    tracker.log_metric(step=20, train_loss=8.0, learning_rate=6e-5)
    tracker.finish(final_step=20, final_train_loss=8.0)
    path = tmp_path / "exp" / "experiment.json"
    assert path.exists()
    with open(path) as fh:
        record = json.load(fh)
    assert record["run_name"] == "test-run"
    assert record["status"] == "completed"
    assert len(record["metrics"]) == 2
    assert record["metrics"][0]["step"] == 10
    assert record["final_train_loss"] == 8.0


def test_experiment_tracker_captures_environment(tmp_path):
    tracker = ExperimentTracker(tmp_path / "exp", run_name="test")
    tracker.start()
    path = tmp_path / "exp" / "experiment.json"
    with open(path) as fh:
        record = json.load(fh)
    assert "hardware" in record
    assert "software" in record
    assert record["software"]["pytorch"].startswith("2.")
    assert record["hardware"]["device"] in ("cpu", "cuda", "mps")


def test_experiment_tracker_fail_marks_status(tmp_path):
    tracker = ExperimentTracker(tmp_path / "exp", run_name="test")
    tracker.start()
    tracker.fail("OOM")
    with open(tmp_path / "exp" / "experiment.json") as fh:
        record = json.load(fh)
    assert record["status"] == "failed"
    assert record["error"] == "OOM"


def test_experiment_record_to_json_serialisable():
    rec = ExperimentRecord(run_name="x", model_name="y")
    j = rec.to_json()
    parsed = json.loads(j)
    assert parsed["run_name"] == "x"
