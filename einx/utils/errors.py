# -*- coding: utf-8 -*-
"""Friendly error messages for EINX (spec §29).

Wraps the most common config / runtime errors with actionable messages
so users get:

  "what went wrong"
  "what the value was"
  "how to fix it"

instead of a long Python traceback.
"""

from __future__ import annotations


class EINXConfigError(Exception):
    """A configuration error with a clear, actionable message."""

    def __init__(self, message: str, *, field: str = "", value=None, hint: str = ""):
        self.field = field
        self.value = value
        self.hint = hint
        parts = [message]
        if field:
            parts.append(f"\n  field: {field}")
        if value is not None:
            parts.append(f"  value: {value!r}")
        if hint:
            parts.append(f"  hint:  {hint}")
        super().__init__("".join(parts))


def validate_model_config_friendly(cfg) -> None:
    """Run model config validation with friendly error messages.

    Wraps the raw ``ValueError`` from ``cfg.validate()`` with field
    names + hints so the user knows exactly what to fix.
    """
    try:
        cfg.validate()
    except ValueError as exc:
        msg = str(exc)
        # Match common error patterns and add hints
        if "hidden_dim" in msg and "head_dim" in msg:
            raise EINXConfigError(
                "Model configuration is internally inconsistent.",
                field="hidden_dim / n_heads / head_dim",
                value=f"{cfg.hidden_dim} != {cfg.n_heads} * {cfg.head_dim}",
                hint=(
                    "hidden_dim must equal n_heads × head_dim. "
                    f"Either set hidden_dim={cfg.n_heads * cfg.head_dim}, "
                    f"or change n_heads/head_dim so they multiply to {cfg.hidden_dim}."
                ),
            ) from exc
        if "vocab_size" in msg:
            raise EINXConfigError(
                "Vocabulary size must be a positive integer.",
                field="vocab_size",
                value=cfg.vocab_size,
                hint="Set vocab_size to match your tokenizer (e.g. 4096 for the default BPE).",
            ) from exc
        if "positional_encoding" in msg:
            raise EINXConfigError(
                "Unknown positional encoding type.",
                field="positional_encoding",
                value=cfg.positional_encoding,
                hint="Use 'rope' (Rotary Position Embeddings) or 'learned'.",
            ) from exc
        if "norm_type" in msg:
            raise EINXConfigError(
                "Unknown normalization type.",
                field="norm_type",
                value=cfg.norm_type,
                hint="Use 'rms' (RMSNorm, recommended) or 'layer' (classical LayerNorm).",
            ) from exc
        if "precision" in msg:
            raise EINXConfigError(
                "Unknown precision type.",
                field="precision",
                value=cfg.precision,
                hint="Use 'fp32' (CPU-safe), 'fp16' (CUDA only), or 'bf16' (CUDA only).",
            ) from exc
        # Generic fallback
        raise EINXConfigError(
            f"Model configuration validation failed: {msg}",
            hint="See docs/training.md for the full configuration reference.",
        ) from exc


def validate_training_config_friendly(cfg) -> None:
    """Run training config validation with friendly error messages."""
    try:
        cfg.validate()
    except ValueError as exc:
        msg = str(exc)
        if "batch_size" in msg:
            raise EINXConfigError(
                "batch_size must be a positive integer.",
                field="batch_size",
                value=cfg.batch_size,
                hint="Try batch_size=8 for EINX-Experimental on CPU.",
            ) from exc
        if "learning_rate" in msg:
            raise EINXConfigError(
                "learning_rate must be positive.",
                field="learning_rate",
                value=cfg.learning_rate,
                hint="Typical range: 1e-5 (fine-tune) to 3e-4 (pretraining).",
            ) from exc
        if "max_steps" in msg or "max_epochs" in msg:
            raise EINXConfigError(
                "Must set either max_steps or max_epochs to a positive value.",
                field="max_steps / max_epochs",
                value=f"max_steps={cfg.max_steps}, max_epochs={cfg.max_epochs}",
                hint="Set max_steps=500 for a quick smoke test.",
            ) from exc
        if "precision" in msg:
            raise EINXConfigError(
                "Unknown precision.",
                field="precision",
                value=cfg.precision,
                hint="Use 'fp32' on CPU. 'fp16'/'bf16' require CUDA.",
            ) from exc
        raise EINXConfigError(
            f"Training configuration validation failed: {msg}",
            hint="See docs/training.md for the full configuration reference.",
        ) from exc


class EINXRuntimeError(Exception):
    """A runtime error with a clear, actionable message."""

    def __init__(self, message: str, *, hint: str = ""):
        parts = [message]
        if hint:
            parts.append(f"\n  hint: {hint}")
        super().__init__("".join(parts))
