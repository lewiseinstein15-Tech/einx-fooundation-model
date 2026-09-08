# -*- coding: utf-8 -*-
"""Tests for the training loop."""

import pytest
import torch

from einx.config import EINXModelConfig, TrainingConfig
from einx.model.transformer import EINXTransformer
from einx.training.trainer import EINXTrainer, TrainingState
from einx.training.optimizers import build_optimizer
from einx.training.schedulers import build_scheduler, LRScheduler
from einx.data.dataset import TokenisedDataset
from einx.tokenizer.bpe import BPETokenizer


def _setup_small_trainer(tmp_path):
    """Build a tiny trainer for tests."""
    # Tiny tokenizer
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(50)]
    tok.train(texts, vocab_size=400, verbose=False)

    # Tiny model
    cfg = EINXModelConfig(
        name="test-model",
        vocab_size=tok.vocab_size(),
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        max_context_length=32,
        ffn_dim=64,
        dropout=0.0,
    )
    model = EINXTransformer(cfg)

    # Tiny dataset
    ds = TokenisedDataset(texts, tok, context_length=32)

    # Tiny training config
    train_cfg = TrainingConfig(
        run_name="test-run",
        model_name="test-model",
        batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-3,
        max_steps=5,
        warmup_steps=2,
        eval_every_steps=0,
        save_every_steps=5,
        log_every_steps=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
        precision="fp32",
    )
    return tok, model, ds, train_cfg


def test_training_state_defaults():
    s = TrainingState()
    assert s.step == 0
    assert s.best_val_loss == float("inf")


def test_build_optimizer_adamw():
    from einx.model.transformer import EINXTransformer
    cfg = EINXModelConfig(vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16, max_context_length=32, ffn_dim=64)
    model = EINXTransformer(cfg)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_unknown_raises():
    cfg = EINXModelConfig(vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16, max_context_length=32, ffn_dim=64)
    model = EINXTransformer(cfg)
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(model, kind="rmsprop")


def test_lr_scheduler_cosine():
    cfg = EINXModelConfig(vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16, max_context_length=32, ffn_dim=64)
    model = EINXTransformer(cfg)
    opt = build_optimizer(model, lr=1e-3)
    sched = build_scheduler(opt, schedule="cosine", warmup_steps=10, total_steps=100)
    assert isinstance(sched, LRScheduler)
    # Step through warmup
    for _ in range(10):
        sched.step()
    # LR should be near peak after warmup
    lr_after_warmup = sched.get_last_lr()[0]
    assert lr_after_warmup > 0


def test_lr_scheduler_linear():
    cfg = EINXModelConfig(vocab_size=100, hidden_dim=32, n_layers=1, n_heads=2, head_dim=16, max_context_length=32, ffn_dim=64)
    model = EINXTransformer(cfg)
    opt = build_optimizer(model, lr=1e-3)
    sched = build_scheduler(opt, schedule="linear", warmup_steps=0, total_steps=100)
    # Step to the end
    for _ in range(100):
        sched.step()
    final_lr = sched.get_last_lr()[0]
    # Linear should decay to near min_lr_ratio * lr
    assert final_lr < 1e-3


def test_trainer_initialization(tmp_path):
    tok, model, ds, train_cfg = _setup_small_trainer(tmp_path)
    trainer = EINXTrainer(model, train_cfg, ds)
    assert trainer.device.type == "cpu"
    assert trainer.state.step == 0


def test_trainer_runs_short_loop(tmp_path):
    """A real 5-step training loop must complete without error."""
    tok, model, ds, train_cfg = _setup_small_trainer(tmp_path)
    trainer = EINXTrainer(model, train_cfg, ds)
    result = trainer.train()
    assert result["final_step"] == 5
    assert len(result["train_losses"]) == 5
    # Loss should be finite
    assert all(isinstance(l, float) for l in result["train_losses"])


def test_trainer_saves_checkpoint(tmp_path):
    tok, model, ds, train_cfg = _setup_small_trainer(tmp_path)
    trainer = EINXTrainer(model, train_cfg, ds)
    trainer.train()
    # Should have saved final + latest + step-5
    ckpt_dir = tmp_path / "checkpoints" / "test-run"
    assert (ckpt_dir / "final.pt").exists()
    assert (ckpt_dir / "latest.pt").exists()


def test_trainer_resume_from_checkpoint(tmp_path):
    """Training should be resumable — load state and continue."""
    tok, model, ds, train_cfg = _setup_small_trainer(tmp_path)
    # First run: 5 steps
    trainer = EINXTrainer(model, train_cfg, ds)
    result1 = trainer.train()
    assert result1["final_step"] == 5

    # Second run: resume from latest, run 5 more steps
    train_cfg.max_steps = 10
    train_cfg.resume_from = str(tmp_path / "checkpoints" / "test-run" / "latest.pt")
    trainer2 = EINXTrainer(model, train_cfg, ds)
    assert trainer2.state.step == 5  # resumed
    result2 = trainer2.train()
    assert result2["final_step"] == 10


def test_trainer_evaluate(tmp_path):
    tok, model, ds, train_cfg = _setup_small_trainer(tmp_path)
    trainer = EINXTrainer(model, train_cfg, ds, val_dataset=ds)
    loss = trainer.evaluate(n_batches=3)
    assert loss > 0
    assert isinstance(loss, float)
