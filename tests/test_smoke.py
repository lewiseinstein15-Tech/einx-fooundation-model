# -*- coding: utf-8 -*-
"""EINX smoke test (spec §27) — full end-to-end integration.

This is the single most important automated test in the project.  It
runs the complete pipeline:

  1. Generate a tiny synthetic corpus
  2. Train a BPE tokenizer
  3. Build a tiny EINX model (einx-smoke config)
  4. Train for 5 steps
  5. Save a checkpoint
  6. Reload the checkpoint into a fresh model
  7. Verify the reloaded model produces identical logits
  8. Generate text from the reloaded model

If this test passes, the entire end-to-end pipeline works.

Marked ``slow`` because it does real training (a few seconds on CPU).
Run with: ``pytest tests/test_smoke.py -v -s``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from einx.config import EINXModelConfig, TrainingConfig
from einx.data.dataset import TokenisedDataset, write_jsonl
from einx.data.synthetic import generate_synthetic_corpus
from einx.model.transformer import EINXTransformer
from einx.tokenizer.bpe import BPETokenizer
from einx.training.trainer import EINXTrainer


@pytest.mark.slow
def test_full_pipeline_smoke(tmp_path):
    """End-to-end: corpus → tokenizer → model → train → save → reload → generate."""
    # 1. Synthetic corpus
    texts = generate_synthetic_corpus(n_records=50, seed=42)
    train_records = [{"text": t} for t in texts[:40]]
    val_records = [{"text": t} for t in texts[40:]]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    # 2. Tokenizer
    tok = BPETokenizer()
    tok.train(train_records[0]["text"] for _ in [0] for r in [None])  # placeholder
    # Re-train on the real texts (the above was a no-op to satisfy types)
    tok = BPETokenizer()
    tok.train([r["text"] for r in train_records], vocab_size=300, verbose=False)
    assert tok.is_trained()
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    # 3. Model
    cfg = EINXModelConfig(
        name="einx-smoke",
        vocab_size=tok.vocab_size(),
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        max_context_length=32,
        ffn_dim=64,
        dropout=0.0,
    )
    model = EINXTransformer(cfg)
    n_params = model.n_params
    assert n_params > 0

    # 4. Train 5 steps
    train_ds = TokenisedDataset(
        [r["text"] for r in train_records], tok, context_length=32,
    )
    train_cfg = TrainingConfig(
        run_name="smoke-test",
        model_name="einx-smoke",
        dataset_path=str(train_path),
        val_dataset_path=str(val_path),
        tokenizer_path=str(tok_path),
        batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-3,
        max_steps=5,
        warmup_steps=2,
        save_every_steps=5,
        eval_every_steps=0,
        log_every_steps=1,
        log_level="WARNING",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
        precision="fp32",
        seed=42,
    )
    trainer = EINXTrainer(model, train_cfg, train_ds)
    result = trainer.train()

    assert result["final_step"] == 5
    assert len(result["train_losses"]) == 5
    initial_loss = result["train_losses"][0]
    final_loss = result["train_losses"][-1]
    # Loss should be finite
    assert initial_loss > 0
    assert final_loss > 0
    # Loss should decrease (or at least not blow up catastrophically)
    # On a tiny corpus + 5 steps this isn't guaranteed, but the loss
    # should not be NaN or infinity.
    import math
    assert not math.isnan(final_loss)
    assert not math.isinf(final_loss)
    print(f"\nSMOKE TEST: initial_loss={initial_loss:.4f} final_loss={final_loss:.4f}")

    # 5. Checkpoint should exist
    from einx.training.checkpoint_manager import CheckpointManager
    mgr = CheckpointManager(tmp_path / "checkpoints", run_name="smoke-test")
    latest = mgr.find_latest()
    assert latest is not None, "no checkpoint saved"
    assert (latest / "model.pt").exists()
    assert (latest / "metadata.json").exists()

    # 6. Reload into a fresh model
    model2 = EINXTransformer.load(latest, map_location="cpu")
    assert model2.n_params == n_params
    model2.eval()

    # 7. Verify logits match — the reloaded model must produce the same
    #    outputs as the original (post-training) model.
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    model.eval()
    logits_original, _ = model(input_ids)
    logits_reloaded, _ = model2(input_ids)
    assert torch.allclose(logits_original, logits_reloaded, atol=1e-6), \
        "reloaded model produces different logits than the original"

    # 8. Generate text from the reloaded model
    from einx.inference.generator import EINXGenerator, GenerationConfig
    gen = EINXGenerator(model2, tok, device="cpu")
    result_gen = gen.generate(
        "the cat",
        GenerationConfig(max_new_tokens=5, temperature=0.0),
    )
    assert result_gen.n_output_tokens > 0
    assert isinstance(result_gen.text, str)
    print(f"SMOKE TEST generation: {result_gen.text!r}")

    # 9. Verify experiment.json was written
    exp_path = tmp_path / "checkpoints" / "smoke-test" / "experiment.json"
    assert exp_path.exists(), "experiment.json not written"
    with open(exp_path) as fh:
        exp = json.load(fh)
    assert exp["status"] == "completed"
    assert exp["final_step"] == 5
    assert "metrics" in exp
    assert len(exp["metrics"]) >= 1

    print("\n✓ FULL END-TO-END SMOKE TEST PASSED")
    print(f"  Model:      {cfg.name} ({n_params:,} params)")
    print(f"  Training:   5 steps")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Checkpoint:   {latest}")
    print(f"  Reload:       identical logits (atol=1e-6)")
    print(f"  Generation:   {result_gen.n_output_tokens} tokens")


@pytest.mark.slow
def test_kv_cache_generation_matches_uncached(tmp_path):
    """Generation with the KV cache should produce the same output as
    non-cached generation (since the cache is just an optimisation)."""
    cfg = EINXModelConfig(
        name="test",
        vocab_size=100,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        max_context_length=32,
        ffn_dim=64,
        dropout=0.0,
    )
    model = EINXTransformer(cfg)
    model.eval()
    input_ids = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)

    # Non-cached: feed full sequence every step
    out_uncached = model.generate(input_ids, max_new_tokens=3, temperature=0.0)

    # Cached: build cache from prompt, then feed 1 token at a time
    from einx.model.kv_cache import new_cache_for_model
    stack = new_cache_for_model(model, batch_size=1)
    # Prime the cache with the prompt
    model(input_ids, kv_cache=stack)
    cached_ids = input_ids
    for _ in range(3):
        next_input = cached_ids[:, -1:]
        logits, _ = model(next_input, kv_cache=stack)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        cached_ids = torch.cat([cached_ids, next_token], dim=1)

    # Both should produce the same generated tokens (greedy, deterministic)
    assert torch.equal(out_uncached[:, -3:], cached_ids[:, -3:]), (
        "KV-cached generation diverged from non-cached generation"
    )
    print("\n✓ KV CACHE GENERATION MATCHES UNCACHED (greedy, deterministic)")
