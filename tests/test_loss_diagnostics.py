# -*- coding: utf-8 -*-
"""Tests for loss logger + overfitting detection (spec §22, §24)."""

import json
import math
import pytest
from pathlib import Path

from einx.training.loss_logger import LossLogger
from einx.training.diagnostics import (
    OverfittingDetector,
    DiagnosticsReport,
    DiagnosticWarning,
)


# ---------------------------------------------------------------------------
# LossLogger
# ---------------------------------------------------------------------------


def test_loss_logger_writes_jsonl(tmp_path):
    """Each log() call writes one JSON object per line (spec §22)."""
    path = tmp_path / "loss_log.jsonl"
    with LossLogger(path) as logger:
        logger.log(step=10, loss=5.5, tokens_seen=1000, learning_rate=3e-4)
        logger.log(step=20, loss=5.0, tokens_seen=2000, learning_rate=3e-4)
        logger.log(step=30, loss=4.8, tokens_seen=3000, learning_rate=2.5e-4,
                    val_loss=5.2)

    entries = LossLogger.load(path)
    assert len(entries) == 3
    assert entries[0]["step"] == 10
    assert entries[0]["loss"] == 5.5
    assert entries[0]["tokens_seen"] == 1000
    assert entries[0]["lr"] == 3e-4
    # val_loss only present when provided
    assert "val_loss" not in entries[0]
    assert "val_loss" in entries[2]
    assert entries[2]["val_loss"] == 5.2


def test_loss_logger_entries_contain_timestamp(tmp_path):
    """Each entry must have a timestamp (spec §22)."""
    path = tmp_path / "loss_log.jsonl"
    with LossLogger(path) as logger:
        logger.log(step=1, loss=5.0, tokens_seen=100, learning_rate=1e-3)
    entries = LossLogger.load(path)
    assert "timestamp" in entries[0]
    assert len(entries[0]["timestamp"]) > 10  # ISO format


def test_loss_logger_is_durable(tmp_path):
    """The log is flushed after every write so it survives a crash."""
    path = tmp_path / "loss_log.jsonl"
    logger = LossLogger(path)
    logger.log(step=1, loss=5.0, tokens_seen=100, learning_rate=1e-3)
    # DON'T call close — simulate a crash
    # (The __del__ method will close it, but the file should already be
    # flushed on disk)
    entries = LossLogger.load(path)
    assert len(entries) == 1


def test_loss_logger_append_mode(tmp_path):
    """Resumed runs append to the existing log (spec §22)."""
    path = tmp_path / "loss_log.jsonl"
    with LossLogger(path) as logger:
        logger.log(step=1, loss=5.0, tokens_seen=100, learning_rate=1e-3)
    # Simulate a resume — open again
    with LossLogger(path) as logger:
        logger.log(step=2, loss=4.5, tokens_seen=200, learning_rate=1e-3)
    entries = LossLogger.load(path)
    assert len(entries) == 2  # both the old + new entries


def test_loss_logger_load_as_dict(tmp_path):
    """Load as column-oriented dict for plotting."""
    path = tmp_path / "loss_log.jsonl"
    with LossLogger(path) as logger:
        for i in range(5):
            logger.log(step=i, loss=5.0 - i * 0.1, tokens_seen=i * 100,
                       learning_rate=3e-4)
    data = LossLogger.load_as_dict(path)
    assert len(data["step"]) == 5
    assert len(data["loss"]) == 5
    assert data["loss"][0] > data["loss"][-1]  # decreasing


def test_loss_logger_load_missing_file_returns_empty(tmp_path):
    entries = LossLogger.load(tmp_path / "nonexistent.jsonl")
    assert entries == []


# ---------------------------------------------------------------------------
# OverfittingDetector
# ---------------------------------------------------------------------------


def test_detector_nan_loss():
    """NaN loss is detected + reported (spec §24)."""
    detector = OverfittingDetector()
    detector.check_loss(step=10, loss=float("nan"))
    report = detector.report()
    assert report.n_nan == 1
    assert report.has_warnings
    assert report.warnings[0].kind == "nan_loss"


def test_detector_inf_loss():
    """Inf loss is detected (spec §24)."""
    detector = OverfittingDetector()
    detector.check_loss(step=10, loss=float("inf"))
    report = detector.report()
    assert report.n_nan == 1


def test_detector_abnormally_large_loss():
    """Loss above the threshold is flagged (spec §24)."""
    detector = OverfittingDetector(nan_threshold=100.0)
    detector.check_loss(step=10, loss=200.0)
    report = detector.report()
    assert report.n_nan == 1
    assert "abnormally large" in report.warnings[0].message


def test_detector_normal_loss_no_warning():
    """Normal loss values should NOT trigger any warning."""
    detector = OverfittingDetector()
    detector.check_loss(step=10, loss=5.5)
    report = detector.report()
    assert not report.has_warnings


def test_detector_exploding_gradients():
    """Grad norm above threshold triggers a warning (spec §24)."""
    detector = OverfittingDetector(exploding_grad_threshold=100.0)
    detector.check_gradients(step=10, grad_norm=500.0)
    report = detector.report()
    assert report.n_exploding == 1


def test_detector_normal_gradients_no_warning():
    detector = OverfittingDetector(exploding_grad_threshold=100.0)
    detector.check_gradients(step=10, grad_norm=1.5)
    report = detector.report()
    assert not report.has_warnings


def test_detector_nan_gradients():
    detector = OverfittingDetector()
    detector.check_gradients(step=10, grad_norm=float("nan"))
    report = detector.report()
    assert report.n_exploding == 1


def test_detector_overfitting_detected():
    """Train loss ↓ while val loss ↑ for 3 consecutive evals = overfitting."""
    detector = OverfittingDetector(overfitting_patience=3, overfitting_min_delta=0.01)
    # Simulate: train loss decreasing, val loss increasing
    for i in range(5):
        train_loss = 5.0 - i * 0.1  # decreasing
        val_loss = 5.0 + i * 0.05    # increasing
        detector.check_val(step=(i + 1) * 100, train_loss=train_loss, val_loss=val_loss)
    report = detector.report()
    assert report.n_overfitting >= 1
    assert "overfitting" in report.warnings[0].message.lower()


def test_detector_no_overfitting_when_val_decreases():
    """If val loss also decreases, no overfitting warning."""
    detector = OverfittingDetector(overfitting_patience=3)
    for i in range(5):
        train_loss = 5.0 - i * 0.1
        val_loss = 5.5 - i * 0.1  # also decreasing
        detector.check_val(step=(i + 1) * 100, train_loss=train_loss, val_loss=val_loss)
    report = detector.report()
    assert report.n_overfitting == 0


def test_detector_unstable_validation():
    """Wildly oscillating val loss triggers an unstable warning (spec §24)."""
    detector = OverfittingDetector(unstable_window=5, unstable_threshold=0.1)
    # Oscillating val losses
    val_losses = [5.0, 10.0, 3.0, 8.0, 2.0]
    for i, vl in enumerate(val_losses):
        detector.check_val(step=(i + 1) * 100, train_loss=5.0, val_loss=vl)
    report = detector.report()
    assert report.n_unstable >= 1


def test_detector_does_not_auto_adjust():
    """The detector only reports — it never changes training parameters (spec §24)."""
    detector = OverfittingDetector()
    detector.check_loss(step=10, loss=float("nan"))
    detector.check_gradients(step=10, grad_norm=500.0)
    detector.check_val(step=10, train_loss=5.0, val_loss=6.0)
    detector.check_val(step=20, train_loss=4.0, val_loss=6.5)
    detector.check_val(step=30, train_loss=3.0, val_loss=7.0)
    report = detector.report()
    # The report has warnings but NO suggested parameter changes
    d = report.to_dict()
    assert "auto_adjust" not in d
    assert "parameter_change" not in d
    assert "recommended_lr" not in d


def test_diagnostics_report_summary():
    report = DiagnosticsReport(warnings=[
        DiagnosticWarning(step=100, kind="overfitting", message="val rising"),
        DiagnosticWarning(step=200, kind="nan_loss", message="loss is NaN"),
    ])
    summary = report.summary()
    assert "EINX TRAINING DIAGNOSTICS" in summary
    assert "Overfitting:" in summary
    assert "NaN loss:" in summary
    assert "Total warnings:" in summary


def test_diagnostics_report_to_dict():
    report = DiagnosticsReport(warnings=[
        DiagnosticWarning(step=10, kind="overfitting", message="test", detail={"a": 1}),
    ])
    d = report.to_dict()
    assert d["n_warnings"] == 1
    assert d["n_overfitting"] == 1
    assert len(d["warnings"]) == 1
    assert d["warnings"][0]["kind"] == "overfitting"


def test_diagnostics_report_empty():
    report = DiagnosticsReport()
    assert not report.has_warnings
    assert report.n_overfitting == 0
    assert report.n_nan == 0


# ---------------------------------------------------------------------------
# Integration: trainer writes a loss log file
# ---------------------------------------------------------------------------


def test_trainer_writes_loss_log(tmp_path):
    """A real training run must produce a loss_log.jsonl file (spec §22)."""
    from einx.config import EINXModelConfig, TrainingConfig
    from einx.data.dataset import TokenisedDataset
    from einx.data.synthetic import generate_synthetic_corpus
    from einx.tokenizer.bpe import BPETokenizer
    from einx.model.transformer import EINXTransformer
    from einx.training.trainer import EINXTrainer

    tok = BPETokenizer()
    texts = generate_synthetic_corpus(50, seed=42)
    tok.train(texts, vocab_size=300, verbose=False)

    cfg = EINXModelConfig(
        name="loss-test", vocab_size=tok.vocab_size(),
        hidden_dim=32, n_layers=2, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64, dropout=0.0,
    )
    model = EINXTransformer(cfg)
    ds = TokenisedDataset(texts, tok, context_length=32)

    train_cfg = TrainingConfig(
        run_name="loss-test", model_name="loss-test",
        batch_size=2, grad_accum_steps=1, learning_rate=1e-3,
        max_steps=10, warmup_steps=0,
        save_every_steps=10, eval_every_steps=5, eval_steps=2,
        log_every_steps=2, log_level="WARNING",
        checkpoint_dir=str(tmp_path / "ckpts"),
        device="cpu", seed=42,
    )
    trainer = EINXTrainer(model, train_cfg, ds, val_dataset=ds)
    result = trainer.train()

    # The loss log must exist
    loss_log_path = Path(result["loss_log_path"])
    assert loss_log_path.exists()

    # Load and verify entries
    entries = LossLogger.load(loss_log_path)
    assert len(entries) > 0
    # Each entry must have step + loss + tokens_seen (spec §22)
    for e in entries:
        assert "step" in e
        assert "loss" in e
        assert "tokens_seen" in e
    # At least some entries should have val_loss (eval_every_steps=5)
    val_entries = [e for e in entries if "val_loss" in e]
    assert len(val_entries) > 0

    # Diagnostics should be in the result
    assert "diagnostics" in result
    assert result["diagnostics"]["n_warnings"] >= 0
