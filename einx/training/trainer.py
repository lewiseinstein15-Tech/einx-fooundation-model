# -*- coding: utf-8 -*-
"""The EINX training loop.

A real, resumable training loop.  Not a toy — runs a real forward
pass, real backprop, real checkpoint save/load, real evaluation.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from einx.config.training_config import TrainingConfig
from einx.training.optimizers import build_optimizer
from einx.training.schedulers import build_scheduler
from einx.utils.hardware import detect_device
from einx.utils.seeds import set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Everything needed to resume a training run.

    Stored in every checkpoint so we can pick up exactly where we
    left off — model weights, optimizer state, scheduler state, and
    the global step counter.  A failed run should never destroy the
    previous checkpoint.
    """

    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    model_state: Dict[str, Any] = field(default_factory=dict)
    optimizer_state: Dict[str, Any] = field(default_factory=dict)
    scheduler_state: Dict[str, Any] = field(default_factory=dict)
    rng_state: Optional[Any] = None
    cuda_rng_state: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EINXTrainer:
    """The full training loop.

    Usage:
        trainer = EINXTrainer(model, config, train_dataset, val_dataset)
        trainer.train()

    The trainer handles:
      * device placement (auto-detected, can be overridden)
      * mixed precision (AMP) when precision != "fp32" and CUDA is available
      * gradient accumulation
      * gradient clipping
      * periodic evaluation on the validation set
      * periodic checkpoint saving (keep-last-N rotation)
      * resuming from a checkpoint
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset,
        val_dataset=None,
        *,
        tokenizer=None,
    ):
        config.validate()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Device
        self.device = torch.device(detect_device(config.device))
        self.model.to(self.device)
        logger.info("training device: %s", self.device)

        # Optimizer + scheduler
        self.optimizer = build_optimizer(
            self.model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            schedule=config.lr_schedule,
            warmup_steps=config.warmup_steps,
            total_steps=config.max_steps,
            min_lr_ratio=config.min_lr_ratio,
        )

        # Mixed precision
        self.use_amp = config.precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = (
            torch.float16 if config.precision == "fp16" else torch.bfloat16
        ) if self.use_amp else torch.float32
        if self.use_amp:
            logger.info("mixed precision training: %s", config.precision)
        elif config.precision in ("fp16", "bf16") and self.device.type != "cuda":
            logger.warning(
                "%s requested but CUDA not available — falling back to fp32",
                config.precision,
            )

        # State
        self.state = TrainingState(config=config.to_dict())

        # Checkpoint dir
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume immediately if a checkpoint was specified in config.
        # This makes ``trainer.state.step`` reflect the resumed step
        # even before ``train()`` is called.
        if config.resume_from:
            self._load_checkpoint(config.resume_from)
            logger.info("resumed from step %d", self.state.step)

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """Run the training loop.  Returns final metrics + state."""
        set_seed(self.config.seed)
        # NOTE: checkpoint resume (if config.resume_from was set) already
        # happened in __init__ — no need to redo it here.

        # DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # Training loop
        total_steps = self.config.max_steps
        log_every = self.config.log_every_steps
        eval_every = self.config.eval_every_steps
        save_every = self.config.save_every_steps

        train_losses: List[float] = []
        val_losses: List[Tuple[int, float]] = []

        epoch = self.state.epoch
        step = self.state.step
        while step < total_steps:
            for batch in train_loader:
                if step >= total_steps:
                    break
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)

                # Forward with optional AMP
                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                        _, loss = self.model(input_ids, targets=targets)
                    # Scale loss by grad_accum_steps so the effective batch
                    # size is batch_size * grad_accum_steps.
                    loss = loss / self.config.grad_accum_steps
                    loss.backward()
                else:
                    _, loss = self.model(input_ids, targets=targets)
                    loss = loss / self.config.grad_accum_steps
                    loss.backward()

                # Step the optimizer every grad_accum_steps micro-batches
                if (step + 1) % self.config.grad_accum_steps == 0:
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                train_losses.append(loss.item() * self.config.grad_accum_steps)

                step += 1
                self.state.step = step

                # Logging
                if step % log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    avg_loss = sum(train_losses[-log_every:]) / min(log_every, len(train_losses))
                    logger.info(
                        "step %d/%d  loss=%.4f  lr=%.2e",
                        step, total_steps, avg_loss, lr,
                    )

                # Periodic eval
                if eval_every > 0 and step % eval_every == 0 and self.val_dataset is not None:
                    val_loss = self.evaluate(self.config.eval_steps)
                    val_losses.append((step, val_loss))
                    logger.info("eval step %d  val_loss=%.4f", step, val_loss)
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                        self._save_checkpoint(step, tag="best")

                # Periodic save
                if save_every > 0 and step % save_every == 0:
                    self._save_checkpoint(step, tag=f"step-{step}")
                    self._save_checkpoint(step, tag="latest")
                    self._rotate_checkpoints()

            epoch += 1
            self.state.epoch = epoch
            if self.config.max_epochs > 0 and epoch >= self.config.max_epochs:
                break

        # Final save
        self._save_checkpoint(step, tag="latest")
        self._save_checkpoint(step, tag="final")

        return {
            "final_step": step,
            "final_epoch": epoch,
            "best_val_loss": self.state.best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": self.config.to_dict(),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, n_batches: int = 50) -> float:
        """Evaluate on the validation set.  Returns average loss."""
        if self.val_dataset is None:
            return float("nan")
        self.model.eval()
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                with torch.autocast(
                    device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
                ):
                    _, loss = self.model(input_ids, targets=targets)
                total_loss += loss.item()
                n += 1
        self.model.train()
        return total_loss / max(1, n)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, *, tag: str = "latest") -> Path:
        """Save a checkpoint to ``<checkpoint_dir>/<tag>.pt``."""
        path = self.checkpoint_dir / f"{tag}.pt"
        # Update state with current weights
        self.state.model_state = self.model.state_dict()
        self.state.optimizer_state = self.optimizer.state_dict()
        self.state.scheduler_state = self.scheduler.state_dict()
        self.state.rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.state.cuda_rng_state = torch.cuda.get_rng_state_all()
        torch.save(
            {
                "step": step,
                "epoch": self.state.epoch,
                "best_val_loss": self.state.best_val_loss,
                "model_state_dict": self.state.model_state,
                "optimizer_state_dict": self.state.optimizer_state,
                "scheduler_state_dict": self.state.scheduler_state,
                "rng_state": self.state.rng_state,
                "cuda_rng_state": self.state.cuda_rng_state,
                "config": self.config.to_dict(),
                "einx_version": "0.1.0",
            },
            path,
        )
        logger.info("checkpoint saved: %s (step %d)", path, step)
        return path

    def _load_checkpoint(self, path: str) -> None:
        """Load a checkpoint and restore state."""
        path = Path(path)
        if not path.exists():
            # Try as a tag inside the run's checkpoint dir
            path = self.checkpoint_dir / f"{path}.pt"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("rng_state") is not None:
            torch.set_rng_state(ckpt["rng_state"])
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        self.state.step = ckpt.get("step", 0)
        self.state.epoch = ckpt.get("epoch", 0)
        self.state.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info("checkpoint loaded: %s (step %d)", path, self.state.step)

    def _rotate_checkpoints(self) -> None:
        """Keep only the last N step-* checkpoints.  Always keep best/latest/final."""
        keep_n = self.config.keep_last_n_checkpoints
        if keep_n <= 0:
            return
        step_ckpts = sorted(self.checkpoint_dir.glob("step-*.pt"))
        if len(step_ckpts) <= keep_n:
            return
        for ckpt in step_ckpts[:-keep_n]:
            try:
                ckpt.unlink()
                logger.info("rotated out old checkpoint: %s", ckpt)
            except OSError:
                pass

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<EINXTrainer device={self.device} "
            f"step={self.state.step} "
            f"best_val_loss={self.state.best_val_loss:.4f}>"
        )
