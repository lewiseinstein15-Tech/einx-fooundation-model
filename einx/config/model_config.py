# -*- coding: utf-8 -*-
"""Model architecture configuration for EINX.

A single :class:`EINXModelConfig` dataclass carries every parameter of
the transformer architecture.  Configs are loaded from YAML files in
``configs/model/`` (one per model family member: einx-experimental,
einx-1b, etc.).

Defaults below define **EINX-Experimental** — a tiny research model
(~2M params) small enough to train on a laptop CPU in a few minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class EINXModelConfig:
    """Every architectural parameter of an EINX model.

    Frozen semantics are NOT enforced so we can load YAML into the
    dataclass, but the contract is: once a model is trained, its config
    is part of its identity and must NOT be mutated.
    """

    name: str = "einx-experimental"
    version: str = "0.1.0"
    arch: str = "decoder-only-transformer"

    # Vocabulary / tokenizer
    vocab_size: int = 4096

    # Core dimensions
    hidden_dim: int = 128        # d_model
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 32            # must equal hidden_dim // n_heads

    # Context
    max_context_length: int = 256

    # Feed-forward (intermediate) dimension — 4x d_model is the
    # classical default from the original Transformer paper.
    ffn_dim: int = 512

    # Regularization
    dropout: float = 0.1

    # Positional encoding: "rope" (Rotary Position Embeddings) or "learned"
    positional_encoding: str = "rope"

    # Normalization: "layer" or "rms"
    norm_type: str = "rms"

    # Precision (training): "fp32" | "fp16" | "bf16"
    # Only fp32 is guaranteed on CPU; fp16/bf16 require CUDA.
    precision: str = "fp32"

    # Tying: share weights between input embeddings and output LM head.
    tie_word_embeddings: bool = True

    # Special-token IDs (filled in by tokenizer, but defaults here so the
    # model can be initialised before a tokenizer is attached).
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Check the architecture is internally consistent.

        Raises ``ValueError`` on any inconsistency.  Called at model init
        time so a bad config fails fast rather than producing a broken
        model.
        """
        if self.head_dim * self.n_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal "
                f"n_heads ({self.n_heads}) * head_dim ({self.head_dim})"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.max_context_length <= 0:
            raise ValueError(
                f"max_context_length must be positive, got {self.max_context_length}"
            )
        if self.positional_encoding not in ("rope", "learned"):
            raise ValueError(
                f"positional_encoding must be 'rope' or 'learned', got {self.positional_encoding!r}"
            )
        if self.norm_type not in ("layer", "rms"):
            raise ValueError(
                f"norm_type must be 'layer' or 'rms', got {self.norm_type!r}"
            )
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(
                f"precision must be 'fp32' | 'fp16' | 'bf16', got {self.precision!r}"
            )

    # ------------------------------------------------------------------
    # Approximate parameter count (for sizing / logging)
    # ------------------------------------------------------------------
    def approx_param_count(self) -> int:
        """Rough parameter count for this configuration.

        Useful for sanity-checking a config without instantiating the
        model.  Numbers are estimates — embeddings + transformer blocks
        + LM head (if not tied).
        """
        v, d, l, f = self.vocab_size, self.hidden_dim, self.n_layers, self.ffn_dim
        emb = v * d
        # Each transformer block:
        #   attention: 4 * d^2 (Q, K, V, O projections)
        #   FFN: 2 * d * f (up + down)
        #   norms: small (d per norm, ~2d per block)
        per_block = 4 * d * d + 2 * d * f + 4 * d
        total = emb + l * per_block
        if not self.tie_word_embeddings:
            total += v * d  # output head
        return total

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EINXModelConfig":
        # Only keep fields the dataclass knows about — ignore extras so
        # configs from future EINX versions don't break older code.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EINXModelConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_yaml())


# ---------------------------------------------------------------------------
# Built-in configs (one per model family member)
# ---------------------------------------------------------------------------


def experimental() -> EINXModelConfig:
    """EINX-Experimental: ~2M param research model.

    Tiny enough to train on a laptop CPU in a few minutes.  Real
    transformer — just small.  This is the v0.1 milestone model.
    """
    return EINXModelConfig(
        name="einx-experimental",
        version="0.1.0",
        vocab_size=4096,
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        head_dim=32,
        max_context_length=256,
        ffn_dim=512,
        dropout=0.1,
    )


def small() -> EINXModelConfig:
    """EINX-Small: ~25M params.  For when you have a GPU."""
    return EINXModelConfig(
        name="einx-small",
        version="0.1.0",
        vocab_size=8192,
        hidden_dim=256,
        n_layers=6,
        n_heads=8,
        head_dim=32,
        max_context_length=512,
        ffn_dim=1024,
        dropout=0.1,
    )


def get_model_config(name: str = "einx-experimental") -> EINXModelConfig:
    """Look up a built-in model config by name.

    Currently defined (real): ``einx-experimental``, ``einx-small``.
    Future EINX-1B / 7B / 14B / 32B / MoE are NOT yet implemented —
    they will return a config with the right *shape* but with a clear
    ``version="planned"`` marker so nobody mistakes them for trained
    models.
    """
    table = {
        "einx-experimental": experimental,
        "einx-small": small,
    }
    factory = table.get(name)
    if factory is None:
        raise KeyError(
            f"unknown model name: {name!r}. "
            f"defined: {sorted(table.keys())}. "
            f"larger models (einx-1b/7b/14b/32b/moe) are PLANNED — see docs/roadmap.md."
        )
    return factory()
