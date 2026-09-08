# -*- coding: utf-8 -*-
"""Building blocks for the EINX transformer.

Each layer is intentionally small and inspectable.  No hidden state
mutations, no monolithic files — each layer has one job.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm — no bias, no mean subtraction.

    Slightly faster and empirically better-behaved than LayerNorm in
    modern transformer training (LLaMA, etc.).  When the model config
    says ``norm_type="rms"`` we use this.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability when the model
        # is in fp16/bf16.
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return (x.to(dtype)) * self.weight


class LayerNorm(nn.Module):
    """Classical LayerNorm with bias.  Used when config says ``norm_type="layer"``."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


def build_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    if norm_type == "rms":
        return RMSNorm(dim, eps=eps)
    if norm_type == "layer":
        return LayerNorm(dim, eps=eps)
    raise ValueError(f"unknown norm_type: {norm_type!r}")


# ---------------------------------------------------------------------------
# Positional encoding — Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Applied to Q and K before the attention computation.  Has the
    nice property that relative positions are encoded as rotations, so
    the model can generalise to longer contexts than seen in training
    (up to ``max_context_length``).

    The implementation below uses the standard formulation.  No
    complex numbers — we rotate pairs of dimensions directly.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # Pre-compute the inverse frequencies so we don't redo it every
        # forward pass.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # cos/sin: each of shape (max_seq_len, head_dim // 2)
        cos = freqs.cos()
        sin = freqs.sin()
        # Repeat along the last dim so cos/sin are (max_seq_len, head_dim)
        # — matches the "rotate_pairs" implementation below.
        self.register_buffer("cos_cached", torch.cat([cos, cos], dim=-1), persistent=False)
        self.register_buffer("sin_cached", torch.cat([sin, sin], dim=-1), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim: (x1, x2) -> (-x2, x1).

    Used in the RoPE rotation.  Equivalent to multiplying by i in the
    complex-number formulation.
    """
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K.

    ``q``, ``k`` shape: (batch, n_heads, seq, head_dim)
    ``cos``, ``sin`` shape: (seq, head_dim)
    """
    # Broadcast cos/sin to (1, 1, seq, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Multi-head self-attention (causal)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Supports RoPE (rotary) or learned absolute position embeddings.
    No KV cache yet (Phase 4 — efficient inference) — every forward
    recomputes from scratch, which is fine for training and small
    research models.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.1,
        bias: bool = False,                     # LLaMA convention: no bias in QKV
    ):
        super().__init__()
        assert hidden_dim == n_heads * head_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Fused QKV projection (single matmul instead of three)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope: Optional[RotaryPositionEmbedding] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (batch, seq, hidden_dim)
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)
        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rope is not None:
            cos, sin = rope(T, device=x.device, dtype=x.dtype)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal mask.
        # PyTorch 2.0+ has fused SDPA — use it when available.
        if mask is None:
            # Build a causal mask on the fly
            mask = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # we passed the mask explicitly
        )
        # (B, n_heads, T, head_dim) -> (B, T, hidden_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn)


# ---------------------------------------------------------------------------
# Feed-forward network (SwiGLU)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (LLaMA-style).

    FFN(x) = (W_down(silu(W_up(x)) * W_gate(x)))
    where silu is the Sigmoid Linear Unit (swish).

    Slightly more parameters than a vanilla ReLU FFN, but empirically
    better-behaved and the de-facto standard in modern transformers.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        # Note: with SwiGLU the effective intermediate size is 2/3 of
        # ffn_dim, so we multiply by 2/3 to keep the param count matched
        # to a vanilla ReLU FFN of size ffn_dim.  Many implementations
        # skip this; we keep it explicit.
        intermediate = int(ffn_dim * 2 / 3)
        # Round up to nearest multiple of 8 for tensor-core efficiency
        intermediate = ((intermediate + 7) // 8) * 8
        self.w_up = nn.Linear(hidden_dim, intermediate, bias=bias)
        self.w_gate = nn.Linear(hidden_dim, intermediate, bias=bias)
        self.w_down = nn.Linear(intermediate, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_up(x))
        return self.w_down(gate * self.w_gate(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """One decoder block: pre-norm attention + pre-norm FFN."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.norm1 = build_norm(cfg.norm_type, cfg.hidden_dim)
        self.attn = MultiHeadAttention(
            hidden_dim=cfg.hidden_dim,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            dropout=cfg.dropout,
        )
        self.norm2 = build_norm(cfg.norm_type, cfg.hidden_dim)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.ffn_dim)

    def forward(self, x: torch.Tensor, *, rope) -> torch.Tensor:
        # Pre-norm: x = x + sublayer(norm(x))
        h = self.norm1(x)
        h = self.attn(h, rope=rope)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h
