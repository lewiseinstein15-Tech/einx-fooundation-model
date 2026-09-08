# -*- coding: utf-8 -*-
"""EINX Build 3 — end-to-end pretraining integration test (spec §28).

Executes the COMPLETE pipeline on CPU:

    tiny raw dataset
    ↓
    validation
    ↓
    cleaning
    ↓
    deduplication
    ↓
    tokenization
    ↓
    packing
    ↓
    training
    ↓
    validation
    ↓
    checkpoint
    ↓
    reload
    ↓
    generation

This must execute on CPU.  The model + dataset are tiny enough for
automated testing (~10 seconds total).

This is the spec §28 "end-to-end pretraining test" — the most
important integration test in Build 3.
"""

from __future__ import annotations

import json
import pytest
import torch
from pathlib import Path

from einx.config import EINXModelConfig, TrainingConfig
from einx.data.pipeline import DataPipeline, PipelineConfig
from einx.data.shards import ShardDataset
from einx.data.manifest import DatasetManifest
from einx.tokenizer.bpe import BPETokenizer
from einx.model.transformer import EINXTransformer
from einx.training.trainer import EINXTrainer
from einx.inference.generator import EINXGenerator, GenerationConfig


@pytest.mark.slow
def test_end_to_end_pretraining_pipeline(tmp_path):
    """Full pipeline: raw → validate → clean → dedupe → tokenize → pack →
    shard → train → checkpoint → reload → generate.

    Every step uses REAL data + REAL processing — no fabrication.
    """
    print("\n" + "=" * 60)
    print("EINX Build 3 — End-to-End Pretraining Integration Test")
    print("=" * 60)

    # ---- 1. Create a tiny raw dataset with variation ----------------
    raw_path = tmp_path / "raw.jsonl"
    with open(raw_path, "w") as fh:
        # Multiple documents
        for i in range(30):
            fh.write(json.dumps({"text": f"the cat sat on the mat number {i}"}) + "\n")
        # Duplicates
        fh.write(json.dumps({"text": "the cat sat on the mat number 0"}) + "\n")
        fh.write(json.dumps({"text": "the cat sat on the mat number 1"}) + "\n")
        # Empty (should be removed)
        fh.write(json.dumps({"text": ""}) + "\n")
        # Malformed
        fh.write("this is not json\n")
    print(f"  ✓ Raw dataset: {raw_path} (35 lines, incl. duplicates + malformed)")

    # ---- 2. Train a tokenizer -------------------------------------
    tok = BPETokenizer()
    tok_texts = [f"the cat sat on the mat number {i}" for i in range(30)] * 5
    tok.train(tok_texts, vocab_size=300, verbose=False)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)
    print(f"  ✓ Tokenizer trained: vocab={tok.vocab_size()}, version={tok.VERSION}")

    # ---- 3. Run the full data pipeline -----------------------------
    output_dir = tmp_path / "processed"
    pipeline = DataPipeline(PipelineConfig(
        name="einx-build3-test",
        version="0.1.0",
        input_paths=[raw_path],
        text_field="text",
        tokenizer_path=str(tok_path),
        context_length=32,
        shard_size=100,
        validation_ratio=0.2,
        seed=42,
        output_dir=str(output_dir),
    ))
    result = pipeline.run()
    print(f"  ✓ Pipeline complete: {result.n_train_records} train + "
          f"{result.n_val_records} val records, "
          f"{result.n_tokens:,} tokens")
    assert result.n_train_records > 0
    assert result.n_val_records > 0
    assert result.n_tokens > 0
    assert result.manifest is not None

    # ---- 4. Verify manifest exists + is loadable -------------------
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    loaded_manifest = DatasetManifest.load(manifest_path)
    assert loaded_manifest.identity_hash == result.manifest.identity_hash
    print(f"  ✓ Manifest verified: identity={loaded_manifest.identity_hash[:16]}")

    # ---- 5. Build a tiny model -------------------------------------
    model_cfg = EINXModelConfig(
        name="einx-build3-test",
        vocab_size=tok.vocab_size(),
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        max_context_length=32,
        ffn_dim=64,
        dropout=0.0,
    )
    model = EINXTransformer(model_cfg)
    print(f"  ✓ Model built: {model.n_params:,} params")

    # ---- 6. Build datasets from shards ------------------------------
    train_ds = ShardDataset(output_dir / "train", context_length=32)
    val_ds = ShardDataset(output_dir / "val", context_length=32)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    print(f"  ✓ Shard datasets: train={len(train_ds)}, val={len(val_ds)}")

    # ---- 7. Train --------------------------------------------------
    train_cfg = TrainingConfig(
        run_name="build3-test",
        model_name="einx-build3-test",
        batch_size=2,
        grad_accum_steps=1,
        learning_rate=1e-3,
        max_steps=10,
        warmup_steps=2,
        save_every_steps=10,
        eval_every_steps=5,
        eval_steps=5,
        log_every_steps=5,
        log_level="WARNING",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
        precision="fp32",
        seed=42,
    )
    trainer = EINXTrainer(model, train_cfg, train_ds, val_ds, tokenizer=tok)
    result_train = trainer.train()
    initial_loss = result_train["train_losses"][0]
    final_loss = result_train["train_losses"][-1]
    val_loss = result_train.get("final_val_loss")
    tokens = result_train.get("performance", {}).get("n_tokens", 0)
    print(f"  ✓ Training: {result_train['final_step']} steps, "
          f"loss {initial_loss:.4f} → {final_loss:.4f}, "
          f"val_loss={val_loss}, tokens={tokens}")
    assert result_train["final_step"] == 10
    assert not (final_loss != final_loss)  # not NaN

    # ---- 8. Checkpoint exists + is loadable ------------------------
    from einx.training.checkpoint_manager import CheckpointManager
    mgr = CheckpointManager(tmp_path / "checkpoints", run_name="build3-test")
    latest = mgr.find_latest()
    assert latest is not None
    assert (latest / "model.pt").exists()
    assert (latest / "metadata.json").exists()
    print(f"  ✓ Checkpoint saved: {latest.name}")

    # ---- 9. Reload checkpoint into a fresh model -------------------
    model2 = EINXTransformer.load(latest, map_location="cpu")
    model2.eval()
    # Verify logits match
    input_ids = torch.randint(0, model_cfg.vocab_size, (1, 8))
    model.eval()
    logits_orig, _ = model(input_ids)
    logits_reloaded, _ = model2(input_ids)
    max_diff = (logits_orig - logits_reloaded).abs().max().item()
    print(f"  ✓ Checkpoint reloaded: max logits diff = {max_diff:.2e}")
    assert max_diff < 1e-6

    # ---- 10. Generate text from the reloaded model -----------------
    gen = EINXGenerator(model2, tok, device="cpu")
    gen_result = gen.generate(
        "the cat",
        GenerationConfig(max_new_tokens=5, temperature=0.8, seed=42),
    )
    print(f"  ✓ Generation: {gen_result.n_output_tokens} tokens, "
          f"{gen_result.tokens_per_second:.1f} tok/s")
    assert gen_result.n_output_tokens > 0

    # ---- Summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("BUILD 3 END-TO-END PRETRAINING TEST — PASSED")
    print("=" * 60)
    print(f"  Raw records:       35 (incl. dups + malformed)")
    print(f"  Valid records:     {result.n_valid_records}")
    print(f"  Unique records:    {result.n_unique_records}")
    print(f"  Train records:     {result.n_train_records}")
    print(f"  Val records:       {result.n_val_records}")
    print(f"  Total tokens:      {result.n_tokens:,}")
    print(f"  Shards:            {result.n_shards}")
    print(f"  Model params:      {model.n_params:,}")
    print(f"  Initial loss:      {initial_loss:.4f}")
    print(f"  Final loss:        {final_loss:.4f}")
    print(f"  Val loss:          {val_loss}")
    print(f"  Tokens processed:  {tokens}")
    print(f"  Checkpoint:        {latest.name}")
    print(f"  Reload diff:       {max_diff:.2e}")
    print(f"  Generation:        {gen_result.n_output_tokens} tokens")
    print("=" * 60)
