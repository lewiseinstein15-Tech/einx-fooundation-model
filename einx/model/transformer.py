# -*- coding: utf-8 -*-
"""The full EINX Transformer model.

A decoder-only transformer for causal language modeling.  Combines
the layers in ``einx/model/layers.py`` into a complete model with
embeddings, transformer stack, and LM head.

The model is intentionally simple — no MoE, no multi-query attention,
no speculative decoding.  Those are Phase 3+ features.  This is the
v0.1 research prototype: real transformer, real forward pass, real
generation, but small and clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einx.config.model_config import EINXModelConfig
from einx.model.layers import (
    RotaryPositionEmbedding,
    TransformerBlock,
    build_norm,
)
from einx.model.init import apply_init_strategy, DEFAULT_INIT_STD
from einx.model.kv_cache import KVCacheStack


class EINXTransformer(nn.Module):
    """The EINX decoder-only transformer.

    Construction:
        cfg = EINXModelConfig(...)        # or get_model_config("einx-experimental")
        model = EINXTransformer(cfg)

    Forward:
        logits, loss = model(input_ids, targets=optional_targets)

    Generate:
        out_ids = model.generate(input_ids, max_new_tokens=64)

    The model is a real PyTorch ``nn.Module`` — fully inspectable,
    serialisable with ``torch.save``, and trainable with any standard
    PyTorch optimiser.
    """

    def __init__(self, config: EINXModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Token embeddings.  Scale by sqrt(d_model) (Transformer paper trick)
        # to keep the embedding magnitude comparable to the rest of the
        # activations after the residual connections.
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Positional encoding
        self.rope: Optional[RotaryPositionEmbedding] = None
        if config.positional_encoding == "rope":
            self.rope = RotaryPositionEmbedding(
                head_dim=config.head_dim,
                max_seq_len=config.max_context_length,
            )
        else:
            # Learned absolute positional embeddings
            self.positional_embedding = nn.Embedding(
                config.max_context_length, config.hidden_dim
            )
            nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)

        # Dropout on embeddings (applied after embedding + position)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final norm before the LM head
        self.norm_final = build_norm(config.norm_type, config.hidden_dim)

        # LM head: project hidden_dim -> vocab_size.
        # If ``tie_word_embeddings``, share weights with the token embedding.
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        # Number of parameters — logged at init time so it's obvious
        # when a config change accidentally doubled the model size.
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._n_params = n_params
        self._n_trainable = n_trainable

        # Apply the centralized initialization strategy — replaces the
        # scattered nn.init.normal_ calls from Build 1 with one auditable
        # call.  See einx/model/init.py for the strategy documentation.
        apply_init_strategy(self, n_layers=config.n_layers, std=DEFAULT_INIT_STD)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def n_trainable_params(self) -> int:
        return self._n_trainable

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCacheStack] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        ``input_ids``:  (batch, seq) LongTensor
        ``targets``:    (batch, seq) LongTensor or None
        ``kv_cache``:   optional KVCacheStack — when provided, the cache
                        is appended to and the model uses the full cached
                        K/V for attention.  Pass ``cache_position_offset``
                        implicitly via ``kv_cache.seq_len`` so RoPE applies
                        the correct rotation.

        Returns ``(logits, loss)`` where logits is (batch, seq, vocab)
        and loss is a scalar (or None when targets is None).
        """
        B, T = input_ids.size()
        if T > self.config.max_context_length:
            raise ValueError(
                f"input sequence length {T} exceeds max_context_length "
                f"({self.config.max_context_length})"
            )

        # 1. Token embeddings
        x = self.token_embedding(input_ids)  # (B, T, C)

        # 2. Positional encoding
        # When using a cache, the new tokens start at position
        # kv_cache.seq_len (the existing cached length), not 0.
        cache_position_offset = kv_cache.seq_len if kv_cache is not None else 0

        if self.rope is not None:
            # RoPE is applied inside attention; here we just keep x as-is.
            pass
        else:
            positions = torch.arange(
                cache_position_offset,
                cache_position_offset + T,
                device=input_ids.device,
            ).unsqueeze(0).expand(B, T)
            x = x + self.positional_embedding(positions)

        # 3. Embedding dropout
        x = self.emb_dropout(x)

        # 4. Transformer blocks — thread the per-layer cache through.
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = block(
                x,
                rope=self.rope,
                kv_cache=layer_cache,
                cache_position_offset=cache_position_offset,
            )

        # 5. Final norm
        x = self.norm_final(x)

        # 6. LM head
        logits = self.lm_head(x)  # (B, T, vocab)

        # 7. Loss (cross-entropy over the vocabulary, ignoring pad token)
        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Implements:
          * temperature scaling
          * top-k filtering
          * top-p (nucleus) filtering
          * early stop on EOS

        Greedy decoding when ``temperature <= 0`` or ``top_k == 1``.

        Returns the full sequence (input + generated) as a LongTensor.
        """
        self.eval()
        device = input_ids.device
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Crop context to max_context_length if needed
            context = input_ids[:, -self.config.max_context_length:]
            logits, _ = self.forward(context)
            next_logits = logits[:, -1, :]  # (B, vocab)

            # Temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                # Greedy — take argmax
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if eos_id is not None and (next_token == eos_id).all():
                    break
                continue

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                # Keep only the top-k logits, set the rest to -inf
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits >= threshold,
                    next_logits,
                    torch.full_like(next_logits, float("-inf")),
                )

            # Top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Always keep at least one token (the top-1)
                sorted_indices_to_remove[..., 0] = False
                # Scatter back to original indexing
                indices_to_remove = torch.zeros_like(next_logits, dtype=torch.bool)
                for b in range(next_logits.size(0)):
                    indices_to_remove[b].scatter_(
                        -1, sorted_indices[b], sorted_indices_to_remove[b]
                    )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Early stop on EOS
            if eos_id is not None and (next_token == eos_id).all():
                break

        return input_ids

    # ------------------------------------------------------------------
    # Checkpoint helpers — support both the legacy flat-file format
    # (a single .pt with model_state_dict + config) AND the new
    # CheckpointManager directory format (step-NNNNNN/model.pt +
    # metadata.json).  The inference path uses these.
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the model + config to a single checkpoint file.

        This is the *legacy* flat-file format used by the inference
        generator (and by tests).  The trainer uses the new directory
        format via ``CheckpointManager`` instead.
        """
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "n_params": self.n_params,
            "einx_version": "0.1.0",
        }, path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str = "cpu") -> "EINXTransformer":
        """Load a model from a checkpoint.

        ``path`` may be:
          * a flat ``.pt`` file (legacy format) — loads model_state_dict + config
          * a checkpoint directory (new format: ``step-NNNNNN/`` with
            ``model.pt`` + ``metadata.json``) — loads from ``model.pt``
            and reads config from ``metadata.json``
          * a directory containing both ``model.pt`` and ``metadata.json``
        """
        from pathlib import Path
        path = Path(path)

        # Case 1: directory (new CheckpointManager format)
        if path.is_dir():
            model_path = path / "model.pt"
            meta_path = path / "metadata.json"
            if not model_path.exists():
                raise FileNotFoundError(f"no model.pt in checkpoint dir: {path}")
            state = torch.load(model_path, map_location=map_location, weights_only=True)
            # Try to read config from metadata.json first; fall back to a
            # config.json sidecar (older format); fall back to a flat
            # ``config`` key in the state dict (oldest format).
            config_dict = None
            if meta_path.exists():
                import json
                with open(meta_path) as fh:
                    meta = json.load(fh)
                # New format: meta["config"]["model"] holds the model config
                # (the trainer writes both training + model configs)
                cfg_field = meta.get("config", {})
                if isinstance(cfg_field, dict):
                    if "model" in cfg_field and isinstance(cfg_field["model"], dict):
                        config_dict = cfg_field["model"]
                    elif "vocab_size" in cfg_field:
                        # Old format: config was just the model config
                        config_dict = cfg_field
            if config_dict is None and isinstance(state, dict) and "config" in state:
                config_dict = state["config"]
            if config_dict is None:
                raise ValueError(f"could not find model config in {path}")
            cfg = EINXModelConfig.from_dict(config_dict)
            model = cls(cfg)
            # state may be a raw state_dict (just weights) or a dict
            # containing ``model_state_dict``
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
            return model

        # Case 2: flat file (legacy format)
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = EINXModelConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<EINXTransformer name={self.config.name!r} "
            f"arch={self.config.arch!r} "
            f"n_params={self.n_params:,} "
            f"n_layers={self.config.n_layers} hidden={self.config.hidden_dim} "
            f"heads={self.config.n_heads} ctx={self.config.max_context_length}>"
        )
