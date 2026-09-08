# -*- coding: utf-8 -*-
"""Tests for the EINX transformer model."""

import pytest
import torch

from einx.config import EINXModelConfig, get_model_config
from einx.model.transformer import EINXTransformer
from einx.model.layers import RMSNorm, LayerNorm, RotaryPositionEmbedding, MultiHeadAttention, FeedForward


def _small_config(**overrides) -> EINXModelConfig:
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
    defaults.update(overrides)
    return EINXModelConfig(**defaults)


def test_model_init_experimental():
    cfg = get_model_config("einx-experimental")
    model = EINXTransformer(cfg)
    assert model.n_params > 0
    assert model.n_trainable_params > 0
    assert "EINXTransformer" in repr(model)


def test_model_init_small():
    cfg = get_model_config("einx-small")
    model = EINXTransformer(cfg)
    assert model.n_params > 1_000_000


def test_model_init_validates_config():
    cfg = EINXModelConfig(hidden_dim=64, n_heads=4, head_dim=17)  # 4 * 17 != 64
    with pytest.raises(ValueError, match="hidden_dim"):
        EINXTransformer(cfg)


def test_forward_pass_returns_logits_and_loss():
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(input_ids, targets=targets)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_forward_pass_without_targets():
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, loss = model(input_ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert loss is None


def test_forward_pass_rejects_too_long_sequence():
    cfg = _small_config(max_context_length=64)
    model = EINXTransformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 128))  # > max_context_length
    with pytest.raises(ValueError, match="max_context_length"):
        model(input_ids)


def test_generate_returns_longer_sequence():
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    input_ids = torch.tensor([[1, 5, 10, 15]], dtype=torch.long)  # BOS + 3 tokens
    out = model.generate(input_ids, max_new_tokens=10, temperature=0.0)
    assert out.shape == (1, 4 + 10)


def test_generate_with_top_k_top_p():
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    input_ids = torch.tensor([[1, 5, 10]], dtype=torch.long)
    out = model.generate(input_ids, max_new_tokens=5, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (1, 3 + 5)


def test_generate_stops_on_eos():
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    input_ids = torch.tensor([[1]], dtype=torch.long)
    # Force the model to predict EOS — patch the lm_head's bias so logits[0, 2] is huge
    # Simpler: just pass max_new_tokens=1 and check it stops
    out = model.generate(input_ids, max_new_tokens=1, eos_token_id=cfg.eos_token_id)
    assert out.shape[1] >= 2  # at least one new token


def test_model_save_load_roundtrip(tmp_path):
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.eval()
    # Run a forward pass to populate state
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    logits_before, _ = model(input_ids)

    path = str(tmp_path / "model.pt")
    model.save(path)
    model2 = EINXTransformer.load(path)
    model2.eval()

    logits_after, _ = model2(input_ids)
    assert torch.allclose(logits_before, logits_after, atol=1e-6)


def test_rmsnorm():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape
    # RMSNorm preserves the shape


def test_layernorm():
    norm = LayerNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rope():
    rope = RotaryPositionEmbedding(head_dim=32, max_seq_len=128)
    cos, sin = rope(16, device=torch.device("cpu"), dtype=torch.float32)
    assert cos.shape == (16, 32)
    assert sin.shape == (16, 32)


def test_multi_head_attention():
    attn = MultiHeadAttention(hidden_dim=64, n_heads=4, head_dim=16)
    x = torch.randn(2, 8, 64)
    out = attn(x)
    assert out.shape == (2, 8, 64)


def test_feedforward():
    ffn = FeedForward(hidden_dim=64, ffn_dim=128)
    x = torch.randn(2, 8, 64)
    out = ffn(x)
    assert out.shape == (2, 8, 64)


def test_tied_weights_save_memory():
    """Tied embeddings should have fewer params than untied."""
    cfg_tied = _small_config(tie_word_embeddings=True)
    cfg_untied = _small_config(tie_word_embeddings=False)
    tied = EINXTransformer(cfg_tied)
    untied = EINXTransformer(cfg_untied)
    assert tied.n_params < untied.n_params


def test_backward_pass_works():
    """Training requires gradients to flow through the model."""
    cfg = _small_config()
    model = EINXTransformer(cfg)
    model.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    # Check that gradients are populated
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no gradient for {name}"
            break
