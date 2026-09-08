#!/usr/bin/env python3
"""EINX Pretraining Run 2 — larger model, diverse corpus, more steps.

Trains EINX-Pretrain-2 (~2.5M params) on the 50K diverse corpus
(2.4M tokens) for 1500 steps on CPU. Context 256, vocab 2048.
"""
import sys, json, time, math
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
from einx.training.checkpoint_manager import CheckpointManager
from einx.training.loss_logger import LossLogger

def main():
    print("=" * 60)
    print("EINX PRETRAINING RUN 2 — DIVERSE CORPUS")
    print("=" * 60)

    model_cfg = EINXModelConfig.from_yaml("configs/model/einx-pretrain-2.yaml")
    # Fix vocab size to match tokenizer
    model_cfg.vocab_size = 1901  # actual tokenizer size
    train_cfg = TrainingConfig.from_yaml("configs/training/pretrain-2.yaml")

    tokenizer_path = "data/tokenized/einx-bpe-v3.json"
    train_dir = "data/processed_v2/train"
    val_dir = "data/processed_v2/val"

    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer: vocab={tokenizer.vocab_size()}, merges={len(tokenizer.merges)}, version={tokenizer.VERSION}")

    train_ds = ShardDataset(train_dir, context_length=model_cfg.max_context_length)
    val_ds = ShardDataset(val_dir, context_length=model_cfg.max_context_length)
    print(f"Dataset: train={len(train_ds)} samples, val={len(val_ds)} samples")

    model = EINXTransformer(model_cfg)
    print(f"Model: {model}")
    print(f"  Parameters: {model.n_params:,}")
    print(f"  Architecture: {model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H ctx={model_cfg.max_context_length}")
    print()

    runtime_cfg = RuntimeConfig(device="cpu", precision="fp32", compile=False)
    trainer = EINXTrainer(
        model, train_cfg, train_ds, val_ds,
        tokenizer=tokenizer,
        runtime_config=runtime_cfg,
    )

    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    # ---- Analyze loss log ----
    loss_entries = LossLogger.load(result.get("loss_log_path", ""))
    train_entries = [e for e in loss_entries if "val_loss" not in e]
    val_entries = [e for e in loss_entries if "val_loss" in e]

    initial_loss = train_entries[0]["loss"] if train_entries else 0
    final_loss = train_entries[-1]["loss"] if train_entries else 0
    final_val_loss = val_entries[-1]["val_loss"] if val_entries else None
    best_val_loss = result.get("best_val_loss")
    if best_val_loss == float("inf"):
        best_val_loss = None

    perf = result.get("performance", {})
    tokens_seen = perf.get("n_tokens", 0)
    tokens_per_sec = perf.get("avg_tokens_per_second", 0)
    steps_per_sec = perf.get("avg_steps_per_second", 0)

    ppl = math.exp(final_val_loss) if final_val_loss and final_val_loss < 20 else float("inf")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE — RESULTS")
    print("=" * 60)
    print(f"Model:               EINX-Pretrain-2")
    print(f"Parameters:          {model.n_params:,}")
    print(f"Architecture:        {model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H ctx={model_cfg.max_context_length}")
    print(f"Vocab size:          {tokenizer.vocab_size()}")
    print(f"Dataset:             42,559 unique docs, 2.4M tokens")
    print()
    print(f"Training steps:      {result['final_step']}")
    print(f"Training time:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Initial loss:        {initial_loss:.4f}")
    print(f"Final loss:          {final_loss:.4f}")
    if final_val_loss:
        print(f"Final val loss:      {final_val_loss:.4f}")
    if best_val_loss:
        print(f"Best val loss:       {best_val_loss:.4f}")
    print(f"Perplexity (val):    {ppl:.2f}")
    print(f"Tokens processed:    {tokens_seen:,}")
    print(f"Tokens/sec:          {tokens_per_sec:.1f}")
    print(f"Steps/sec:           {steps_per_sec:.2f}")
    print(f"Device:              CPU")

    # ---- Loss curve ----
    print()
    print("LOSS CURVE:")
    for e in train_entries:
        print(f"  step {e['step']:5d}  loss={e['loss']:.4f}  tokens={e['tokens_seen']:,}  lr={e['lr']:.2e}")
    if val_entries:
        print("VALIDATION:")
        for e in val_entries:
            vp = math.exp(e["val_loss"]) if e["val_loss"] < 20 else float("inf")
            print(f"  step {e['step']:5d}  val_loss={e['val_loss']:.4f}  perplexity={vp:.2f}")

    # ---- Generation test ----
    print()
    print("=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    mgr = CheckpointManager("checkpoints", run_name="pretrain-2")
    best_ckpt = mgr.find_best() or mgr.find_latest()
    if best_ckpt:
        print(f"Loading checkpoint: {best_ckpt.name}")
        model_loaded = EINXTransformer.load(best_ckpt, map_location="cpu")
        gen = EINXGenerator(model_loaded, tokenizer, device="cpu")

        prompts = [
            "the cat", "at dawn", "what did the", "count:",
            "the dragon", "how to", "on monday", "the old",
        ]
        for prompt in prompts:
            print(f"\nPrompt: {prompt!r}")
            for temp in [0.0, 0.7]:
                result_gen = gen.generate(prompt, GenerationConfig(
                    max_new_tokens=40, temperature=temp, top_k=30, top_p=0.9,
                    seed=42 if temp > 0 else None,
                ))
                label = "greedy" if temp == 0 else f"temp={temp}"
                print(f"  [{label}] {result_gen.text!r}")

    # ---- Evaluation ----
    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    model_loaded.eval()
    from torch.utils.data import DataLoader
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, drop_last=False)
    total_loss_val = 0.0
    n_val = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 50:
                break
            input_ids, targets = batch
            _, loss = model_loaded(input_ids, targets=targets)
            total_loss_val += loss.item()
            n_val += 1
    eval_loss = total_loss_val / max(1, n_val)
    eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    # Latency
    import time as _t
    timings = []
    for _ in range(3):
        start = _t.time()
        r = gen.generate("the cat", GenerationConfig(max_new_tokens=20, temperature=0.0, top_k=0, top_p=1.0))
        if r.n_output_tokens > 0:
            timings.append(r.elapsed_seconds / r.n_output_tokens)
    avg_latency = (sum(timings) / len(timings) * 1000) if timings else 0.0

    print(f"Evaluation loss:     {eval_loss:.4f}")
    print(f"Perplexity:          {eval_ppl:.2f}")
    print(f"Latency:             {avg_latency:.2f} ms/token")
    print(f"Throughput:          {1000/avg_latency:.1f} tok/s" if avg_latency > 0 else "")
    print(f"Val examples:        {n_val * 16}")

    # ---- Final benchmark ----
    print()
    print("=" * 60)
    print("FINAL BENCHMARK — EINX-Pretrain-2")
    print("=" * 60)
    benchmark = {
        "model": "EINX-Pretrain-2",
        "parameters": model.n_params,
        "architecture": f"{model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H ctx={model_cfg.max_context_length}",
        "vocab_size": tokenizer.vocab_size(),
        "context_length": model_cfg.max_context_length,
        "training_steps": result["final_step"],
        "training_time_seconds": round(elapsed, 1),
        "initial_train_loss": round(initial_loss, 4),
        "final_train_loss": round(final_loss, 4),
        "final_val_loss": round(eval_loss, 4),
        "best_val_loss": round(best_val_loss, 4) if best_val_loss else None,
        "perplexity": round(eval_ppl, 2),
        "loss_reduction_pct": round((1 - final_loss / initial_loss) * 100, 1) if initial_loss else 0,
        "tokens_processed": tokens_seen,
        "tokens_per_second": round(tokens_per_sec, 1),
        "latency_ms_per_token": round(avg_latency, 2),
        "device": "CPU",
        "dataset_records": 42559,
        "dataset_tokens": 2406991,
    }
    print(json.dumps(benchmark, indent=2))

    bench_path = Path("experiments/benchmark_pretrain2.json")
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bench_path, "w") as fh:
        json.dump(benchmark, fh, indent=2)
    print(f"\nSaved to {bench_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
