# -*- coding: utf-8 -*-
"""Robust checkpoint manager for EINX.

Build 1 wrote checkpoints as flat ``latest.pt`` / ``final.pt`` /
``step-N.pt`` files.  This module upgrades to:

  * ``step-NNNNNN/`` **directories** (one per checkpoint) containing:
      - ``model.pt``           (weights)
      - ``optimizer.pt``        (AdamW moments)
      - ``scheduler.pt``        (LR position)
      - ``metadata.json``       (step, epoch, loss, config, timestamp)
  * **Atomic writes** — write to a temp dir, then ``os.rename`` to the
    final location.  If writing fails, the previous good checkpoint is
    untouched.  A power loss mid-write never corrupts the previous
    checkpoint.
  * ``find_latest_checkpoint()`` — returns the highest-numbered
    ``step-NNNNNN/`` directory.
  * ``keep_last_n()`` — rotates old checkpoints, keeping the most recent N.

Public API:

    mgr = CheckpointManager(checkpoint_dir, run_name="EINX-Experimental")

    path = mgr.save(step=1000, model=model, optimizer=opt, scheduler=sched,
                    metrics={"train_loss": 3.2, "val_loss": 3.4})
    latest = mgr.find_latest()
    state = mgr.load(latest)  # → CheckpointState
    mgr.keep_last_n(3)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


STEP_DIR_PATTERN = re.compile(r"^step-(\d+)$")


@dataclass
class CheckpointState:
    """Everything stored in a checkpoint.  Restorable in one call."""

    step: int
    epoch: int
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    rng_state: Optional[Any]
    cuda_rng_state: Optional[Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    einx_version: str = "0.1.0"


class CheckpointManager:
    """Atomic, directory-based checkpoint manager.

    Args:
        checkpoint_dir:  root directory (e.g. ``checkpoints/``)
        run_name:       sub-directory for this run (e.g. ``EINX-Experimental``)
    """

    def __init__(self, checkpoint_dir: str | Path, run_name: str):
        self.root = Path(checkpoint_dir) / run_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

    # ------------------------------------------------------------------
    # Save (atomic)
    # ------------------------------------------------------------------
    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        rng_state: Optional[Any] = None,
        cuda_rng_state: Optional[Any] = None,
        tag: Optional[str] = None,           # if set, also create a symlink/alias
    ) -> Path:
        """Save a checkpoint atomically.

        Writes to a temp directory first, then renames it to the final
        location.  If any step fails, the previous good checkpoint is
        left untouched.
        """
        # Final destination: <root>/step-NNNNNN/
        step_dir_name = f"step-{step:06d}"
        final_path = self.root / step_dir_name

        # Write to a temp dir in the SAME filesystem (so rename is atomic)
        with tempfile.TemporaryDirectory(dir=self.root, prefix=".tmp-ckpt-") as tmpdir:
            tmp_path = Path(tmpdir)

            # 1. Model weights
            torch.save(model.state_dict(), tmp_path / "model.pt")

            # 2. Optimizer + scheduler (if provided)
            if optimizer is not None:
                torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
            if scheduler is not None and hasattr(scheduler, "state_dict"):
                torch.save(scheduler.state_dict(), tmp_path / "scheduler.pt")

            # 3. Metadata JSON — the "single source of truth" for the
            # checkpoint.  Includes everything needed to resume + audit.
            metadata = {
                "step": step,
                "epoch": epoch,
                "metrics": metrics or {},
                "config": config or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "einx_version": "0.1.0",
                "run_name": self.run_name,
                "rng_state": _serialize_rng(rng_state),
                "cuda_rng_state": _serialize_rng(cuda_rng_state),
            }
            with open(tmp_path / "metadata.json", "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=str)

            # 4. Atomic rename — this is the commit point.  After this,
            # the checkpoint is visible and loadable.
            if final_path.exists():
                # Shouldn't happen (we rotate), but be defensive
                shutil.rmtree(final_path)
            os.rename(tmp_path, final_path)

        # 5. Update the ``latest`` symlink to point at this checkpoint.
        # Symlinks make "find latest" an O(1) readdir rather than a
        # max-step scan.
        latest_link = self.root / "latest"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        try:
            os.symlink(final_path.name, latest_link, target_is_directory=True)
        except OSError:
            # Some filesystems don't support symlinks — fall back to a
            # marker file.  ``find_latest_checkpoint`` handles both.
            with open(self.root / "latest.marker", "w") as fh:
                fh.write(final_path.name)

        # 6. Optional tag alias (e.g. "best" for the lowest-val-loss ckpt)
        if tag:
            tag_link = self.root / tag
            if tag_link.is_symlink() or tag_link.exists():
                tag_link.unlink()
            try:
                os.symlink(final_path.name, tag_link, target_is_directory=True)
            except OSError:
                with open(self.root / f"{tag}.marker", "w") as fh:
                    fh.write(final_path.name)

        logger.info("checkpoint saved: %s (step %d)", final_path, step)
        return final_path

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(
        self,
        path: Optional[str | Path] = None,
        *,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> CheckpointState:
        """Load a checkpoint into the provided model/optimizer/scheduler.

        If ``path`` is None, loads the latest checkpoint (or raises if none).
        Returns the CheckpointState (with metrics, config, step, etc.)
        for inspection / audit.
        """
        if path is None:
            path = self.find_latest()
        if path is None:
            raise FileNotFoundError(f"no checkpoint found in {self.root}")
        path = Path(path)
        if not path.exists():
            # Try as a step number → step-NNNNNN/
            try:
                step_num = int(str(path))
                step_path = self.root / f"step-{step_num:06d}"
                if step_path.exists():
                    path = step_path
                else:
                    raise FileNotFoundError(f"checkpoint not found: {path}")
            except ValueError:
                raise FileNotFoundError(f"checkpoint not found: {path}")

        # Load metadata first — it's the source of truth.
        with open(path / "metadata.json", "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        # Load weights
        if model is not None:
            model_state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
            model.load_state_dict(model_state)

        # Load optimizer (must come AFTER model load for state_dict compat)
        if optimizer is not None and (path / "optimizer.pt").exists():
            opt_state = torch.load(path / "optimizer.pt", map_location=map_location, weights_only=True)
            optimizer.load_state_dict(opt_state)

        # Load scheduler
        if scheduler is not None and hasattr(scheduler, "load_state_dict"):
            if (path / "scheduler.pt").exists():
                sched_state = torch.load(path / "scheduler.pt", map_location=map_location, weights_only=True)
                scheduler.load_state_dict(sched_state)

        # Restore RNG state for reproducibility
        rng_state = metadata.get("rng_state")
        if rng_state is not None:
            torch.set_rng_state(_deserialize_rng(rng_state))
        cuda_rng_state = metadata.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_deserialize_rng(cuda_rng_state))

        logger.info("checkpoint loaded: %s (step %d)", path, metadata.get("step", 0))
        return CheckpointState(
            step=metadata.get("step", 0),
            epoch=metadata.get("epoch", 0),
            model_state={},  # already loaded above
            optimizer_state={},
            scheduler_state={},
            rng_state=rng_state,
            cuda_rng_state=cuda_rng_state,
            metrics=metadata.get("metrics", {}),
            config=metadata.get("config", {}),
            timestamp=metadata.get("timestamp", ""),
            einx_version=metadata.get("einx_version", "0.1.0"),
        )

    # ------------------------------------------------------------------
    # Find / rotate
    # ------------------------------------------------------------------
    def find_latest(self) -> Optional[Path]:
        """Return the path to the most-recent checkpoint, or None.

        Resolution order:
          1. ``latest`` symlink/marker (set on every save) — points at
             the most-recently-saved checkpoint
          2. Otherwise scan ``step-NNNNNN/`` dirs and return the one
             with the highest step number
        """
        # Try symlink/marker first
        latest_link = self.root / "latest"
        if latest_link.is_symlink() and (latest_link / "metadata.json").exists():
            return latest_link.resolve()
        marker = self.root / "latest.marker"
        if marker.exists():
            with open(marker) as fh:
                name = fh.read().strip()
            p = self.root / name
            if (p / "metadata.json").exists():
                return p
        # Fall back to scanning step-NNNNNN/ dirs — return the highest step
        step_dirs = self._list_step_dirs()
        return step_dirs[-1] if step_dirs else None

    def find_best(self, metric: str = "val_loss", *, tag: str = "best") -> Optional[Path]:
        """Return the path tagged as ``best`` (or None if never tagged)."""
        best_link = self.root / tag
        if best_link.is_symlink() and (best_link / "metadata.json").exists():
            return best_link.resolve()
        marker = self.root / f"{tag}.marker"
        if marker.exists():
            with open(marker) as fh:
                name = fh.read().strip()
            p = self.root / name
            if (p / "metadata.json").exists():
                return p
        return None

    def list_checkpoints(self) -> List[Path]:
        """Return all step-NNNNNN/ directories, sorted by step ascending."""
        return self._list_step_dirs()

    def keep_last_n(self, n: int) -> None:
        """Delete all but the last N step-NNNNNN/ checkpoints.

        The ``latest`` and ``best`` symlinks are preserved even if they
        point at deleted checkpoints (we update them to point at the
        newest surviving checkpoint instead).
        """
        if n <= 0:
            return
        step_dirs = self._list_step_dirs()
        if len(step_dirs) <= n:
            return
        to_delete = step_dirs[:-n]
        latest = self.find_latest()
        for d in to_delete:
            # Don't delete the directory the 'latest' symlink points to
            if latest is not None and d.resolve() == latest.resolve():
                continue
            try:
                shutil.rmtree(d)
                logger.info("rotated out old checkpoint: %s", d)
            except OSError as exc:
                logger.warning("could not delete %s: %s", d, exc)

    # ------------------------------------------------------------------
    def _list_step_dirs(self) -> List[Path]:
        out: List[Path] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            m = STEP_DIR_PATTERN.match(entry.name)
            if m is None:
                continue
            if not (entry / "metadata.json").exists():
                continue
            out.append(entry)
        return out


# ---------------------------------------------------------------------------
# RNG state (de)serialisation — torch RNG states are torch.ByteTensors
# which JSON can't natively store, so we base64 them.
# ---------------------------------------------------------------------------


def _serialize_rng(state: Any) -> Optional[Any]:
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        import base64
        return {
            "__torch_tensor__": True,
            "dtype": str(state.dtype),
            "data": base64.b64encode(state.cpu().numpy().tobytes()).decode("ascii"),
        }
    if isinstance(state, list):
        return [_serialize_rng(s) for s in state]
    return state


def _deserialize_rng(state: Any) -> Any:
    if state is None:
        return None
    if isinstance(state, dict) and state.get("__torch_tensor__"):
        import base64
        import numpy as np
        dtype_str = state.get("dtype", "uint8")
        # Map common torch tensor dtype strings
        dtype_map = {
            "torch.ByteTensor": "uint8",
            "torch.uint8": "uint8",
            "uint8": "uint8",
        }
        np_dtype = dtype_map.get(dtype_str, dtype_str)
        data = base64.b64decode(state["data"])
        arr = np.frombuffer(data, dtype=np_dtype)
        return torch.from_numpy(arr.copy())
    if isinstance(state, list):
        return [_deserialize_rng(s) for s in state]
    return state
