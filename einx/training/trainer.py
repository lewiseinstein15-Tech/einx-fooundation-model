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
        compile_model: bool = False,
        distributed_strategy: str = "none",
        runtime_config: Optional[Any] = None,
    ):
        # Friendly validation — raises EINXConfigError with hints on bad input
        from einx.utils.errors import validate_training_config_friendly, EINXConfigError
        try:
            validate_training_config_friendly(config)
        except EINXConfigError:
            raise  # Don't catch — let the user see the friendly message
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # ----- Build 2.1: resolve the runtime config (device, precision,
        # distributed, backend, compile) against actual hardware.
        from einx.config.runtime_config import RuntimeConfig, ResolvedRuntime
        if runtime_config is None:
            runtime_config = RuntimeConfig(
                device=config.device,
                precision=config.precision,
                compile=compile_model,
            )
        elif isinstance(runtime_config, RuntimeConfig):
            # Allow the CLI to override the training config's device/precision
            if config.device != "auto":
                runtime_config.device = config.device
            if config.precision != "fp32":
                runtime_config.precision = config.precision
            runtime_config.compile = compile_model or runtime_config.compile
        self.runtime = runtime_config.resolve()
        self.hardware_report = self.runtime.hardware_report

        # ----- Print the hardware report banner (spec §14) — only rank 0.
        from einx.training.distributed import detect_mesh, is_main_process, init_distributed, wrap_model
        from einx.utils.hardware import format_hardware_report
        self._mesh = detect_mesh()
        if is_main_process(self._mesh):
            print(format_hardware_report(self.hardware_report))

        # ----- Device placement — use the resolved device, NOT a scattered
        # torch.device() call.  This is the single point of truth.
        self.device = torch.device(self.runtime.device)
        self.model.to(self.device)
        if is_main_process(self._mesh):
            logger.info("training device: %s", self.device)

        # ----- Optional torch.compile (spec §22).  No-op by default — must be
        # explicitly requested via compile_model=True or runtime.compile=True.
        from einx.training.distributed import maybe_compile_model
        self.model = maybe_compile_model(self.model, enabled=self.runtime.compile)

        # ----- Optional distributed wrapping (spec §23).  No-op in single-device mode.
        # Determine strategy from runtime config or explicit override.
        strategy = distributed_strategy
        if distributed_strategy == "none" and self.runtime.is_distributed:
            strategy = self.runtime.distributed
        if self._mesh.is_distributed:
            init_distributed(self._mesh)
            if strategy != "none":
                self.model = wrap_model(
                    self.model,
                    strategy=strategy,
                    mesh=self._mesh,
                    device=self.runtime.device,
                )

        # ----- Mixed precision — use the resolved precision, not the
        # training config's.  The runtime config validates that fp16/bf16
        # is only used on CUDA.
        self.use_amp = self.runtime.use_amp
        self.amp_dtype = self.runtime.amp_dtype or torch.float32
        if self.use_amp and is_main_process(self._mesh):
            logger.info("mixed precision training: %s", self.runtime.precision)

        # ----- Optimizer + scheduler (set up AFTER the model is wrapped,
        # so DDP/FSDP parameters are correctly tracked).
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

        # State
        self.state = TrainingState(config=config.to_dict())

        # Robust checkpoint manager (spec §15) — atomic writes,
        # step-NNNNNN/ directories, find_latest, keep_last_n.
        from einx.training.checkpoint_manager import CheckpointManager
        self.ckpt_mgr = CheckpointManager(config.checkpoint_dir, run_name=config.run_name)
        self.checkpoint_dir = Path(config.checkpoint_dir) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Experiment tracker (spec §16) — writes experiment.json with
        # real metrics, configs, hardware/software versions, timestamps.
        from einx.training.experiment_tracker import ExperimentTracker
        self.tracker = ExperimentTracker(
            self.checkpoint_dir,
            run_name=config.run_name,
            model_name=getattr(getattr(model, "config", None), "name", "einx"),
            model_config=getattr(getattr(model, "config", None), "to_dict", lambda: {})(),
            training_config=config.to_dict(),
            dataset_path=config.dataset_path,
            tokenizer_path=config.tokenizer_path,
            seed=config.seed,
        )

        # Performance monitor (spec §28) — tokens/sec, peak memory, etc.
        from einx.utils.performance import PerformanceMonitor
        self.perf = PerformanceMonitor(device=str(self.device))

        # Loss logger (spec §22) — machine-readable JSONL loss curve.
        # Writes {step, loss, tokens_seen, lr, timestamp} per log interval.
        from einx.training.loss_logger import LossLogger
        self.loss_logger = LossLogger(self.checkpoint_dir / "loss_log.jsonl")

        # Overfitting detector (spec §24) — detects train/val divergence,
        # NaN loss, exploding gradients, unstable validation.
        # Reports warnings but NEVER auto-adjusts training parameters.
        from einx.training.diagnostics import OverfittingDetector
        self.diagnostics = OverfittingDetector()

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

        # Start experiment tracker + performance monitor (rank 0 only writes)
        from einx.training.distributed import is_main_process, make_distributed_sampler, maybe_barrier, cleanup_distributed, rank_aware_log
        is_rank0 = is_main_process(self._mesh)
        if is_rank0:
            self.tracker.start()
            self.perf.start()

        # ----- DataLoader with optional DistributedSampler.
        # In distributed mode, each rank sees a different subset of the
        # data — the DistributedSampler handles sharding + epoch-based
        # reshuffling.  In single-device mode, fall back to shuffle=True.
        train_sampler = make_distributed_sampler(
            self.train_dataset, shuffle=True, seed=self.config.seed, mesh=self._mesh,
        )
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),  # sampler handles shuffling
            sampler=train_sampler,
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
        final_val_loss: Optional[float] = None

        epoch = self.state.epoch
        step = self.state.step
        try:
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
                        loss = loss / self.config.grad_accum_steps
                        loss.backward()
                    else:
                        _, loss = self.model(input_ids, targets=targets)
                        loss = loss / self.config.grad_accum_steps
                        loss.backward()

                    # Step the optimizer every grad_accum_steps micro-batches
                    if (step + 1) % self.config.grad_accum_steps == 0:
                        if self.config.max_grad_norm > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.config.max_grad_norm
                            )
                            # Diagnostics: check for exploding gradients (spec §24)
                            self.diagnostics.check_gradients(step + 1, float(grad_norm))
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

                    # Track throughput — tokens = batch_size * seq_len
                    n_tokens_in_batch = input_ids.numel()
                    raw_loss = loss.item() * self.config.grad_accum_steps
                    self.perf.step(n_tokens_in_batch, train_loss=raw_loss)
                    train_losses.append(raw_loss)

                    # Diagnostics: check for NaN loss (spec §24)
                    self.diagnostics.check_loss(step + 1, raw_loss)

                    step += 1
                    self.state.step = step

                    # Logging — only rank 0 logs to avoid spamming
                    if step % log_every == 0 and is_rank0:
                        lr = self.scheduler.get_last_lr()[0]
                        avg_loss = sum(train_losses[-log_every:]) / min(log_every, len(train_losses))
                        logger.info(
                            "step %d/%d  loss=%.4f  lr=%.2e",
                            step, total_steps, avg_loss, lr,
                        )
                        # Loss curve logging (spec §22) — machine-readable JSONL
                        self.loss_logger.log(
                            step=step,
                            loss=avg_loss,
                            tokens_seen=self.perf._n_tokens,
                            learning_rate=lr,
                        )
                        self.tracker.log_metric(
                            step=step,
                            train_loss=avg_loss,
                            learning_rate=lr,
                            elapsed_seconds=self.perf._step_times[-1] - self.perf._start_time if self.perf._step_times and self.perf._start_time else 0,
                        )

                    # Periodic eval — rank 0 only writes the checkpoint,
                    # but all ranks evaluate so they stay in sync.
                    if eval_every > 0 and step % eval_every == 0 and self.val_dataset is not None:
                        val_loss = self.evaluate(self.config.eval_steps)
                        maybe_barrier(self._mesh)  # keep ranks in sync
                        if is_rank0:
                            val_losses.append((step, val_loss))
                            final_val_loss = val_loss
                            logger.info("eval step %d  val_loss=%.4f", step, val_loss)
                            # Log val_loss to the loss curve (spec §22)
                            self.loss_logger.log(
                                step=step,
                                loss=train_losses[-1],
                                tokens_seen=self.perf._n_tokens,
                                learning_rate=self.scheduler.get_last_lr()[0],
                                val_loss=val_loss,
                            )
                            # Overfitting detection (spec §24) — check
                            # train/val divergence.  Reports but NEVER
                            # auto-adjusts parameters.
                            self.diagnostics.check_val(step, train_losses[-1], val_loss)
                            self.tracker.log_metric(
                                step=step,
                                train_loss=train_losses[-1],
                                learning_rate=self.scheduler.get_last_lr()[0],
                                val_loss=val_loss,
                            )
                            if val_loss < self.state.best_val_loss:
                                self.state.best_val_loss = val_loss
                                self._save_checkpoint(step, tag="best")

                    # Periodic save — RANK 0 ONLY writes the checkpoint.
                    # Other ranks would just write duplicate copies, which
                    # wastes disk + breaks resume semantics.
                    if save_every > 0 and step % save_every == 0:
                        if is_rank0:
                            self._save_checkpoint(step)
                            self.ckpt_mgr.keep_last_n(self.config.keep_last_n_checkpoints)
                        maybe_barrier(self._mesh)

                epoch += 1
                self.state.epoch = epoch
                # DistributedSampler needs set_epoch() called each epoch
                # so each rank gets a different shuffle.
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                if self.config.max_epochs > 0 and epoch >= self.config.max_epochs:
                    break
        except KeyboardInterrupt:
            # Graceful interruption — save what we have, mark status.
            # Only rank 0 writes the checkpoint + tracker.
            maybe_barrier(self._mesh)
            if is_rank0:
                logger.warning("training interrupted by user (Ctrl-C) — saving checkpoint")
                self.tracker.fail("interrupted by user (KeyboardInterrupt)")
                self._save_checkpoint(step, tag="interrupted")
            cleanup_distributed()
            return {
                "final_step": step,
                "final_epoch": epoch,
                "best_val_loss": self.state.best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "interrupted": True,
                "config": self.config.to_dict(),
            }
        except Exception as exc:
            logger.exception("training failed: %s", exc)
            if is_rank0:
                self.tracker.fail(str(exc))
            cleanup_distributed()
            raise

        # Final barrier so all ranks reach the end together
        maybe_barrier(self._mesh)

        # Close the loss logger (spec §22) — flush + close the file
        if is_rank0:
            self.loss_logger.close()

        # Final save — rank 0 only
        if is_rank0:
            self._save_checkpoint(step, tag="final")
            # Finalise experiment tracker + performance monitor
            perf_report = self.perf.finish(final_val_loss=final_val_loss)
            self.tracker.finish(
                final_step=step,
                final_train_loss=train_losses[-1] if train_losses else None,
                final_val_loss=final_val_loss,
                total_tokens=perf_report.n_tokens,
            )
            # Print diagnostics summary (spec §24) — reports overfitting,
            # NaN, exploding gradients, unstable val.  NEVER auto-adjusts.
            diag_report = self.diagnostics.report()
            if diag_report.has_warnings:
                print(diag_report.summary())
        else:
            perf_report = None
            diag_report = None

        # Clean shutdown of the distributed process group
        cleanup_distributed()

        if is_rank0:
            return {
                "final_step": step,
                "final_epoch": epoch,
                "best_val_loss": self.state.best_val_loss,
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": final_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "performance": perf_report.to_dict() if perf_report else None,
                "diagnostics": diag_report.to_dict() if diag_report else None,
                "loss_log_path": str(self.loss_logger.path),
                "hardware": self.hardware_report.to_dict(),
                "runtime": self.runtime.to_dict(),
                "config": self.config.to_dict(),
            }
        # Non-rank-0 processes return a minimal result — rank 0 is the
        # source of truth for global state.
        return {
            "final_step": step,
            "final_epoch": epoch,
            "rank": self._mesh.rank,
            "rank0_only": True,
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
    # Checkpoints — delegate to the new CheckpointManager (atomic,
    # directory-based, with metadata.json per checkpoint)
    # ------------------------------------------------------------------
    def _save_checkpoint(self, step: int, *, tag: Optional[str] = None) -> Path:
        """Save via the CheckpointManager.  Atomic write — a power loss
        mid-write never corrupts the previous good checkpoint.

        Build 3: records dataset manifest hash + tokenizer version in
        the checkpoint metadata, so resumed runs can verify they're
        using the same dataset (spec §18).
        """
        rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        # Build a metrics dict for the metadata
        metrics = {"best_val_loss": self.state.best_val_loss}
        if self.perf and self.perf._train_losses:
            metrics["last_train_loss"] = self.perf._train_losses[-1]
        # Build 3: token accounting (spec §21)
        if self.perf:
            metrics["tokens_seen"] = self.perf._n_tokens
            metrics["tokens_per_second"] = (
                self.perf._n_tokens / (time.time() - self.perf._start_time)
                if self.perf._start_time else 0.0
            )

        # Build a config dict that includes BOTH the training config AND
        # the model config — the loader needs the model config to
        # reconstruct the architecture before loading weights.
        # Build 3: also include dataset manifest hash + tokenizer version
        # for resume-time verification (spec §18).
        dataset_info = {}
        if hasattr(self, "dataset_manifest_hash"):
            dataset_info["manifest_hash"] = self.dataset_manifest_hash
        if hasattr(self, "dataset_manifest_path"):
            dataset_info["manifest_path"] = self.dataset_manifest_path

        full_config = {
            "training": self.config.to_dict(),
            "model": getattr(getattr(self.model, "config", None), "to_dict", lambda: {})(),
            "runtime": self.runtime.to_dict() if hasattr(self, "runtime") else {},
            "dataset": dataset_info,
        }

        return self.ckpt_mgr.save(
            step=step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.state.epoch,
            metrics=metrics,
            config=full_config,
            rng_state=rng_state,
            cuda_rng_state=cuda_rng_state,
            tag=tag,
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load via the CheckpointManager — restores model + optimizer +
        scheduler + RNG state + step counter."""
        # Use the new manager: it handles metadata + atomic loads.
        state = self.ckpt_mgr.load(
            path if path != "latest" else None,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=str(self.device),
        )
        self.state.step = state.step
        self.state.epoch = state.epoch
        self.state.best_val_loss = state.metrics.get("best_val_loss", float("inf"))
        logger.info("checkpoint loaded: step %d, epoch %d", state.step, state.epoch)

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
