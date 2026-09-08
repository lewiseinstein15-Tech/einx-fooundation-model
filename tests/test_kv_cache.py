# -*- coding: utf-8 -*-
"""Tests for the KV cache (spec §19)."""

import pytest
import torch

from einx.config import EINXModelConfig
from einx.model.transformer import EINXTransformer
from einx.model.kv_cache import KVCache, KVCacheStack, new_cache_for_model
from einx.model.layers import MultiHeadAttention, RotaryPositionEmbedding


def _small_config(**kw) -> EINXModelConfig:
    defaults = dict(
        vocab_size=256,
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        head_dim=16,
        max_context_length=128,
        ffn_dim=128,
        dropout=0.0,
    )
    defaults.update(kw)
    return EINXModelConfig(**defaults)


def test_kv_cache_initial_state():
    cache = KVCache(batch_size=2, n_heads=4, head_dim=16, max_seq_len=128)
    assert cache.seq_len == 0
    k, v = cache.get()
    assert k is None
    assert v is None


def test_kv_cache_append_grows():
    cache = KVCache(batch_size=1, n_heads=2, head_dim=8, max_seq_len=64)
    new_k = torch.randn(1, 2, 3, 8)
    new_v = torch.randn(1, 2, 3, 8)
    k, v = cache.append(new_k, new_v)
    assert k.shape == (1, 2, 3, 8)
    assert cache.seq_len == 3

    # Append more
    more_k = torch.randn(1, 2, 2, 8)
    more_v = torch.randn(1, 2, 2, 8)
    k, v = cache.append(more_k, more_v)
    assert k.shape == (1, 2, 5, 8)
    assert cache.seq_len == 5


def test_kv_cache_max_seq_len_sliding_window():
    """When the cache exceeds max_seq_len, oldest tokens drop off."""
    cache = KVCache(batch_size=1, n_heads=2, head_dim=8, max_seq_len=4)
    # Append 3 tokens
    cache.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    assert cache.seq_len == 3
    # Append 3 more — total 6, exceeds 4 → drop 2 oldest
    cache.append(torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
    assert cache.seq_len == 4  # capped at max_seq_len


def test_kv_cache_reset():
    cache = KVCache(batch_size=1, n_heads=2, head_dim=8, max_seq_len=64)
    cache.append(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))
    assert cache.seq_len == 5
    cache.reset()
    assert cache.seq_len == 0


def test_kv_cache_stack():
    stack = KVCacheStack(
        n_layers=4, batch_size=2, n_heads=4, head_dim=16, max_seq_len=64,
    )
    assert len(stack) == 4
    assert stack.seq_len == 0
    # Each layer cache is independent
    for i in range(4):
        stack[i].append(torch.randn(2, 4, 1, 16), torch.randn(2, 4, 1, 16))
    assert stack.seq_len == 1
    # Reset all
    stack.reset()
    assert stack.seq_len == 0


def test_new_cache_for_model():
    cfg = _small_config()
    model = EINXTransformer(cfg)
    stack = new_cache_for_model(model, batch_size=2)
    assert len(stack) == cfg.n_layers
    assert stack[0].n_heads == cfg.n_heads
    assert stack[0].head_dim == cfg.head_dim
    assert stack[0].max_seq_len == cfg.max_context_length


def test_model_forward_with_cache_appends():
    """When a cache is provided, the model should append to it each forward."""
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    stack = new_cache_for_model(model, batch_size=1)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    # First forward: appends 4 tokens to each layer cache
    logits1, _ = model(input_ids, kv_cache=stack)
    assert stack.seq_len == 4
    assert logits1.shape == (1, 4, cfg.vocab_size)

    # Second forward with 1 new token: appends 1 → cache now has 5
    new_token = torch.randint(0, cfg.vocab_size, (1, 1))
    logits2, _ = model(new_token, kv_cache=stack)
    assert stack.seq_len == 5


def test_cached_vs_uncached_logits_match_for_first_token():
    """The logits for the *first* token should be IDENTICAL whether or not
    a cache is used (since the cache is empty at that point)."""
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))

    # Without cache
    torch.manual_seed(0)
    logits_no_cache, _ = model(input_ids)

    # With cache (empty at start) — should produce the same logits
    stack = new_cache_for_model(model, batch_size=1)
    torch.manual_seed(0)
    logits_cached, _ = model(input_ids, kv_cache=stack)

    # They should be identical for the first forward pass
    assert torch.allclose(logits_no_cache, logits_cached, atol=1e-5)


def test_cache_position_offset_with_rope():
    """When using RoPE + cache, the new tokens get the right position offset."""
    cfg = _small_config(positional_encoding="rope")
    model = EINXTransformer(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    stack = new_cache_for_model(model, batch_size=1)

    # Forward 1: 4 tokens at positions 0-3
    logits1, _ = model(input_ids, kv_cache=stack)
    assert stack.seq_len == 4

    # Forward 2: 1 new token at position 4
    new_token = torch.randint(0, cfg.vocab_size, (1, 1))
    logits2, _ = model(new_token, kv_cache=stack)
    # Logits shape should be (1, 1, vocab)
    assert logits2.shape == (1, 1, cfg.vocab_size)
    assert stack.seq_len == 5
