#!/usr/bin/env python3
"""EINX Pretraining Run 1 — real training on the generated corpus.

Trains EINX-Pretrain-1 (128-dim, 4-layer, ~390K params) on the 10K
document corpus for 2000 steps on CPU.  Tracks loss, perplexity, tokens/sec,
and writes a full benchmark report at the end.

Usage:
    python scripts/run_pretraining.py
"""
import sys
import json
import time
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from einx.config import EINXModelConfig, TrainingConfig, RuntimeConfig
from einx.data.shards import ShardDataset
from einx.model.transformer import EINXTransformer
from einx.tokenizer.bpe import BPETokenizer
from einx.training.trainer import EINXTrainer
from einx.inference.generator import EINXGenerator, GenerationConfig
from einx.evaluation.evaluator import EINXEvaluator, EvalConfig

def main():
    print("=" * 60)
    print("EINX PRETRAINING RUN 1")
    print("=" * 60)

    # ---- Load configs ----
    model_cfg = EINXModelConfig.from_yaml("configs/model/einx-pretrain-1.yaml")
    train_cfg = TrainingConfig.from_yaml("configs/training/pretrain-1.yaml")

    # Override dataset paths to use the processed shards
    tokenizer_path = "data/tokenized/einx-bpe-v2.json"
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"

    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer: vocab={tokenizer.vocab_size()}, version={tokenizer.VERSION}")

    # ---- Build datasets ----
    train_ds = ShardDataset(train_dir, context_length=model_cfg.max_context_length)
    val_ds = ShardDataset(val_dir, context_length=model_cfg.max_context_length)
    print(f"Dataset: train={len(train_ds)} samples, val={len(val_ds)} samples")

    # ---- Build model ----
    model = EINXTransformer(model_cfg)
    print(f"Model: {model}")
    print(f"  Parameters: {model.n_params:,}")
    print(f"  Architecture: {model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H ctx={model_cfg.max_context_length}")
    print()

    # ---- Train ----
    runtime_cfg = RuntimeConfig(device="cpu", precision="fp32", compile=False)
    trainer = EINXTrainer(
        model, train_cfg, train_ds, val_ds,
        tokenizer=tokenizer,
        runtime_config=runtime_cfg,
    )

    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    # ---- Report ----
    print()
    print("=" * 60)
    print("TRAINING COMPLETE — RESULTS")
    print("=" * 60)

    initial_loss = result["train_losses"][0] if result["train_losses"] else 0
    final_loss = result["train_losses"][-1] if result["train_losses"] else 0
    final_val_loss = result.get("final_val_loss")
    best_val_loss = result.get("best_val_loss")
    perf = result.get("performance", {})
    tokens_seen = perf.get("n_tokens", 0)
    tokens_per_sec = perf.get("avg_tokens_per_second", 0)
    steps_per_sec = perf.get("avg_steps_per_second", 0)

    # Compute perplexity from final val loss
    perplexity = math.exp(final_val_loss) if final_val_loss and final_val_loss < 20 else float("inf")

    print(f"Model:               EINX-Pretrain-1")
    print(f"Parameters:          {model.n_params:,}")
    print(f"Architecture:        {model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H ctx={model_cfg.max_context_length}")
    print(f"Vocab size:          {tokenizer.vocab_size()}")
    print(f"Tokenizer:           {tokenizer.VERSION}")
    print()
    print(f"Training steps:      {result['final_step']}")
    print(f"Training time:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Initial loss:        {initial_loss:.4f}")
    print(f"Final loss:          {final_loss:.4f}")
    print(f"Final val loss:      {final_val_loss:.4f}" if final_val_loss else "Final val loss:      N/A")
    print(f"Best val loss:       {best_val_loss:.4f}" if best_val_loss and best_val_loss != float("inf") else "Best val loss:       N/A")
    print(f"Perplexity (val):    {perplexity:.2f}")
    print(f"Tokens processed:    {tokens_seen:,}")
    print(f"Tokens/sec:          {tokens_per_sec:.1f}")
    print(f"Steps/sec:           {steps_per_sec:.2f}")
    print(f"Device:              CPU")
    print(f"Peak memory:         N/A (CPU)")
    print()

    # ---- Generation test ----
    print("=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    # Load the best checkpoint
    from einx.training.checkpoint_manager import CheckpointManager
    mgr = CheckpointManager("checkpoints", run_name="pretrain-1")
    best_ckpt = mgr.find_best() or mgr.find_latest()
    if best_ckpt:
        print(f"Loading checkpoint: {best_ckpt.name}")
        model_loaded = EINXTransformer.load(best_ckpt, map_location="cpu")
        gen = EINXGenerator(model_loaded, tokenizer, device="cpu")

        prompts = [
            "the cat",
            "at dawn",
            "what did the",
            "count:",
        ]

        for prompt in prompts:
            print(f"\nPrompt: {prompt!r}")
            for temp in [0.0, 0.5, 0.8]:
                result_gen = gen.generate(prompt, GenerationConfig(
                    max_new_tokens=30,
                    temperature=temp,
                    top_k=20,
                    top_p=0.9,
                    seed=42 if temp > 0 else None,
                ))
                label = "greedy" if temp == 0 else f"temp={temp}"
                print(f"  [{label}] {result_gen.text!r}")
                print(f"  ({result_gen.n_output_tokens} tokens, {result_gen.tokens_per_second:.0f} tok/s)")
    else:
        print("No checkpoint found!")

    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    benchmark = {
        "model": "EINX-Pretrain-1",
        "parameters": model.n_params,
        "architecture": f"{model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H",
        "vocab_size": tokenizer.vocab_size(),
        "context_length": model_cfg.max_context_length,
        "training_steps": result["final_step"],
        "training_time_seconds": round(elapsed, 1),
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "final_val_loss": round(final_val_loss, 4) if final_val_loss else None,
        "best_val_loss": round(best_val_loss, 4) if best_val_loss and best_val_loss != float("inf") else None,
        "perplexity": round(perplexity, 2),
        "tokens_processed": tokens_seen,
        "tokens_per_second": round(tokens_per_sec, 1),
        "steps_per_second": round(steps_per_sec, 2),
        "device": "CPU",
        "dataset_records": 9491,
        "dataset_tokens": 343163,
    }

    # Save benchmark
    bench_path = Path("experiments/benchmark_pretrain1.json")
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w") as fh:
        json.dump(benchmark, fh, indent=2)
    print(json.dumps(benchmark, indent=2))

    return 0

if __name__ == "__main__":
    sys.exit(main())
