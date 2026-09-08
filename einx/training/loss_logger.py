# -*- coding: utf-8 -*-
"""EINX loss curve logging (spec §22).

Records training loss throughout training in a machine-readable JSONL
file.  Every entry has:

    {"step": 100, "loss": 5.21, "tokens_seen": 204800, "lr": 3e-4, "timestamp": "..."}

Values come from actual training — never fabricated.

The loss log is separate from the experiment.json tracker (which
records metadata + final metrics).  The loss log is a time-series
of every logged step, suitable for plotting loss curves.

Usage:
    logger = LossLogger("checkpoints/run/loss_log.jsonl")
    logger.log(step=100, loss=5.21, tokens_seen=204800, learning_rate=3e-4)
    logger.close()
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class LossLogger:
    """Machine-readable loss curve logger.

    Writes one JSON object per line to a JSONL file.  Each line is a
    complete record that can be loaded independently — no need to parse
    the whole file to get the latest entry.

    The file is flushed after every write so the log is durable even
    if training crashes mid-run.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so resumed runs continue the log
        self._fh = open(self.path, "a", encoding="utf-8")
        self._n_entries = 0
        logger.info("loss logger opened: %s", self.path)

    # ------------------------------------------------------------------
    def log(
        self,
        *,
        step: int,
        loss: float,
        tokens_seen: int = 0,
        learning_rate: float = 0.0,
        val_loss: Optional[float] = None,
    ) -> None:
        """Append one loss entry.  Flushed immediately for durability."""
        entry: Dict[str, Any] = {
            "step": step,
            "loss": round(float(loss), 6),
            "tokens_seen": tokens_seen,
            "lr": float(learning_rate),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if val_loss is not None:
            entry["val_loss"] = round(float(val_loss), 6)
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()
        self._n_entries += 1

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the log file.  Safe to call multiple times."""
        if self._fh and not self._fh.closed:
            self._fh.close()
            logger.info("loss logger closed: %s (%d entries)", self.path, self._n_entries)

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load all entries from a loss log file."""
        path = Path(path)
        if not path.exists():
            return []
        entries = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    # ------------------------------------------------------------------
    @staticmethod
    def load_as_dict(path: Union[str, Path]) -> Dict[str, List]:
        """Load the loss log as column-oriented dict (for plotting)."""
        entries = LossLogger.load(path)
        if not entries:
            return {"step": [], "loss": [], "val_loss": [], "tokens_seen": []}
        return {
            "step": [e["step"] for e in entries],
            "loss": [e["loss"] for e in entries],
            "val_loss": [e.get("val_loss") for e in entries],
            "tokens_seen": [e.get("tokens_seen", 0) for e in entries],
            "lr": [e.get("lr", 0) for e in entries],
        }

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
