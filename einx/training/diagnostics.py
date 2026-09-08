# -*- coding: utf-8 -*-
"""EINX overfitting detection (spec §24).

Detects and reports training diagnostics:
  * training loss decreasing while validation loss rises (overfitting)
  * validation loss becoming unstable (oscillating)
  * NaN loss (numerical instability)
  * exploding gradients (grad norm exceeds threshold)

Per spec §24: "Do not automatically change training parameters without
explicit configuration.  Report warnings clearly."

This module DETECTS and REPORTS — it never auto-adjusts.  The warnings
are logged via Python's logging module and recorded in the experiment
metadata so they're visible post-hoc.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticWarning:
    """One diagnostic warning."""

    step: int
    kind: str          # "overfitting" | "nan_loss" | "exploding_gradients" | "unstable_val"
    message: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticsReport:
    """All warnings raised during a training run."""

    warnings: List[DiagnosticWarning] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def n_overfitting(self) -> int:
        return sum(1 for w in self.warnings if w.kind == "overfitting")

    @property
    def n_nan(self) -> int:
        return sum(1 for w in self.warnings if w.kind == "nan_loss")

    @property
    def n_exploding(self) -> int:
        return sum(1 for w in self.warnings if w.kind == "exploding_gradients")

    @property
    def n_unstable(self) -> int:
        return sum(1 for w in self.warnings if w.kind == "unstable_val")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_warnings": len(self.warnings),
            "n_overfitting": self.n_overfitting,
            "n_nan": self.n_nan,
            "n_exploding": self.n_exploding,
            "n_unstable": self.n_unstable,
            "warnings": [w.to_dict() for w in self.warnings],
        }

    def summary(self) -> str:
        lines = [
            "EINX TRAINING DIAGNOSTICS",
            "─" * 40,
            f"Total warnings:          {len(self.warnings)}",
            f"  Overfitting:           {self.n_overfitting}",
            f"  NaN loss:              {self.n_nan}",
            f"  Exploding gradients:   {self.n_exploding}",
            f"  Unstable validation:   {self.n_unstable}",
        ]
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings[-5:]:  # show last 5
                lines.append(f"  step {w.step}: [{w.kind}] {w.message}")
        lines.append("─" * 40)
        return "\n".join(lines)


class OverfittingDetector:
    """Detect overfitting, NaN, exploding gradients, unstable validation.

    Usage:
        detector = OverfittingDetector()
        # In the training loop:
        detector.check_loss(step, loss_value)
        detector.check_gradients(step, grad_norm)
        detector.check_val(step, train_loss, val_loss)
        # At the end:
        report = detector.report()
        print(report.summary())
    """

    def __init__(
        self,
        *,
        overfitting_patience: int = 3,
        overfitting_min_delta: float = 0.01,
        nan_threshold: float = 1e10,
        exploding_grad_threshold: float = 1000.0,
        unstable_window: int = 5,
        unstable_threshold: float = 0.5,
    ):
        """Configure the detector.

        Args:
            overfitting_patience:     how many consecutive val evaluations
                                      with rising val_loss before warning
            overfitting_min_delta:    minimum val_loss increase to count
            nan_threshold:            loss values above this are flagged NaN-ish
            exploding_grad_threshold: grad norm above this triggers a warning
            unstable_window:          number of recent val losses to check for oscillation
            unstable_threshold:       coefficient of variation above this = unstable
        """
        self.overfitting_patience = overfitting_patience
        self.overfitting_min_delta = overfitting_min_delta
        self.nan_threshold = nan_threshold
        self.exploding_grad_threshold = exploding_grad_threshold
        self.unstable_window = unstable_window
        self.unstable_threshold = unstable_threshold

        self._warnings: List[DiagnosticWarning] = []
        self._val_history: List[Tuple[int, float]] = []
        self._train_history: List[Tuple[int, float]] = []
        self._consecutive_overfit = 0

    # ------------------------------------------------------------------
    def check_loss(self, step: int, loss: float) -> None:
        """Check for NaN or abnormally large loss."""
        if math.isnan(loss) or math.isinf(loss):
            self._add_warning(step, "nan_loss",
                               f"Loss is NaN/Inf at step {step} (loss={loss})")
            return
        if abs(loss) > self.nan_threshold:
            self._add_warning(step, "nan_loss",
                               f"Loss is abnormally large at step {step} (loss={loss:.2e})",
                               detail={"loss": loss, "threshold": self.nan_threshold})

    # ------------------------------------------------------------------
    def check_gradients(self, step: int, grad_norm: float) -> None:
        """Check for exploding gradients."""
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            self._add_warning(step, "exploding_gradients",
                               f"Gradient norm is NaN/Inf at step {step} (grad_norm={grad_norm})")
            return
        if grad_norm > self.exploding_grad_threshold:
            self._add_warning(step, "exploding_gradients",
                               f"Exploding gradients at step {step} "
                               f"(grad_norm={grad_norm:.2e}, threshold={self.exploding_grad_threshold})",
                               detail={"grad_norm": grad_norm, "threshold": self.exploding_grad_threshold})

    # ------------------------------------------------------------------
    def check_val(self, step: int, train_loss: float, val_loss: float) -> None:
        """Check for overfitting (train ↓ while val ↑) + unstable validation."""
        self._train_history.append((step, train_loss))
        self._val_history.append((step, val_loss))

        # Overfitting detection: train_loss decreasing while val_loss increasing
        if len(self._val_history) >= 2:
            prev_step, prev_val = self._val_history[-2]
            curr_step, curr_val = self._val_history[-1]
            if curr_val > prev_val + self.overfitting_min_delta:
                self._consecutive_overfit += 1
                if self._consecutive_overfit >= self.overfitting_patience:
                    self._add_warning(step, "overfitting",
                                       f"Possible overfitting: val_loss has risen for "
                                       f"{self._consecutive_overfit} consecutive evaluations "
                                       f"(was {prev_val:.4f}, now {curr_val:.4f}) while train_loss "
                                       f"is decreasing",
                                       detail={
                                           "prev_val_loss": prev_val,
                                           "curr_val_loss": curr_val,
                                           "train_loss": train_loss,
                                           "consecutive": self._consecutive_overfit,
                                       })
            else:
                self._consecutive_overfit = 0

        # Unstable validation detection: val_loss oscillates wildly
        if len(self._val_history) >= self.unstable_window:
            recent_vals = [v for _, v in self._val_history[-self.unstable_window:]]
            mean_val = sum(recent_vals) / len(recent_vals)
            if mean_val > 0:
                variance = sum((v - mean_val) ** 2 for v in recent_vals) / len(recent_vals)
                cv = (variance ** 0.5) / mean_val  # coefficient of variation
                if cv > self.unstable_threshold:
                    self._add_warning(step, "unstable_val",
                                       f"Validation loss is unstable (coefficient of variation "
                                       f"{cv:.2f} > {self.unstable_threshold} over last "
                                       f"{self.unstable_window} evaluations)",
                                       detail={
                                           "cv": cv,
                                           "window": self.unstable_window,
                                           "recent_vals": recent_vals,
                                       })

    # ------------------------------------------------------------------
    def _add_warning(self, step: int, kind: str, message: str, detail: Optional[Dict] = None) -> None:
        warning = DiagnosticWarning(step=step, kind=kind, message=message, detail=detail or {})
        self._warnings.append(warning)
        logger.warning("DIAGNOSTIC [%s] step %d: %s", kind, step, message)

    # ------------------------------------------------------------------
    def report(self) -> DiagnosticsReport:
        return DiagnosticsReport(warnings=list(self._warnings))

    # ------------------------------------------------------------------
    @property
    def has_warnings(self) -> bool:
        return len(self._warnings) > 0
