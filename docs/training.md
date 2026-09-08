# EINX Training Guide

## Overview

EINX training is a real PyTorch training loop with:

- AdamW optimizer (separate weight decay for biases vs 2D weights)
- Cosine / linear / constant LR schedule with warmup
- Gradient accumulation
- Mixed precision (fp16 / bf16 on CUDA; fp32 default)
- Gradient clipping
- Periodic evaluation on the validation set
- Periodic checkpoint saving (with keep-last-N rotation)
- Resumable from any checkpoint
- Reproducible via seed

## Quick start

```bash
# Generate synthetic corpus, train tokenizer, train model (200 steps)
python -m einx.cli train --prepare-synthetic 500 --steps 200
```

## Custom training

### 1. Prepare your dataset

Create JSONL files with a `"text"` field:

```bash
# data/processed/train.jsonl
{"text": "your training text here"}
{"text": "another training example"}
```

Split into train/val/test:

```python
from einx.data.dataset import load_jsonl, write_jsonl, train_val_test_split

records = load_jsonl("data/raw/corpus.jsonl")
train, val, test = train_val_test_split(records, val_ratio=0.1, test_ratio=0.05)
write_jsonl(train, "data/processed/train.jsonl")
write_jsonl(val, "data/processed/val.jsonl")
write_jsonl(test, "data/processed/test.jsonl")
```

### 2. Train the tokenizer

```bash
python -m einx.cli tokenizer train \
    --corpus data/processed/train.jsonl \
    --output data/tokenized/einx-bpe.json \
    --vocab-size 4096
```

### 3. Train the model

```bash
python -m einx.cli train \
    --model-config configs/model/einx-experimental.yaml \
    --training-config configs/training/experimental.yaml \
    --tokenizer data/tokenized/einx-bpe.json \
    --dataset data/processed/train.jsonl \
    --val-dataset data/processed/val.jsonl \
    --steps 1000
```

## Resuming from a checkpoint

```bash
python -m einx.cli train \
    --resume checkpoints/einx-experimental-run-1/latest.pt \
    --steps 2000
```

The trainer restores:
- Model weights
- Optimizer state (AdamW moments)
- Scheduler state (LR position)
- Step counter
- RNG state (for reproducibility)

A failed run never destroys the previous checkpoint — `latest.pt` is only
overwritten after the new one is fully written.

## Configuration

### Model config (`configs/model/einx-experimental.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 4096 | Tokenizer vocabulary size |
| `hidden_dim` | 128 | d_model |
| `n_layers` | 4 | Transformer blocks |
| `n_heads` | 4 | Attention heads |
| `head_dim` | 32 | Per-head dimension (must = hidden_dim / n_heads) |
| `max_context_length` | 256 | Maximum sequence length |
| `ffn_dim` | 512 | Feed-forward intermediate size |
| `dropout` | 0.1 | Dropout probability |
| `positional_encoding` | rope | rope \| learned |
| `norm_type` | rms | rms \| layer |
| `precision` | fp32 | fp32 \| fp16 \| bf16 |
| `tie_word_embeddings` | true | Share input/output embeddings |

### Training config (`configs/training/experimental.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Micro-batch size |
| `grad_accum_steps` | 4 | Effective batch = batch_size × grad_accum_steps |
| `learning_rate` | 3e-4 | Peak LR (after warmup) |
| `weight_decay` | 0.1 | L2 decay (2D weights only) |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `warmup_steps` | 50 | Linear warmup |
| `lr_schedule` | cosine | cosine \| linear \| constant |
| `min_lr_ratio` | 0.1 | Final LR = lr × min_lr_ratio |
| `max_steps` | 500 | Total training steps |
| `precision` | fp32 | fp32 \| fp16 \| bf16 |
| `save_every_steps` | 250 | Checkpoint frequency |
| `eval_every_steps` | 100 | Validation frequency |
| `seed` | 42 | RNG seed |

## Hardware

EINX auto-detects the best device:
- CUDA (NVIDIA GPU) — fastest, supports fp16/bf16
- MPS (Apple Silicon) — faster than CPU
- CPU — slowest but always works

Force a device with `--device cpu` or `EINX_DEVICE=cpu`.

## Monitoring

Training logs to stderr:

```
2026-09-08 06:02:04 INFO einx.training.trainer: training device: cpu
2026-09-08 06:02:09 INFO einx.training.trainer: step 10/200  loss=8.4021  lr=6.00e-06
2026-09-08 06:02:12 INFO einx.training.trainer: step 20/200  loss=8.3889  lr=1.50e-05
...
```

## Tips

- For the experimental model, 200-500 steps is enough to verify the loop works.
- For real pretraining, you'd want 10K-100K steps on a much larger corpus.
- Mixed precision (fp16/bf16) roughly doubles throughput on CUDA.
- Gradient accumulation lets you simulate large batches on small GPUs.
- Always checkpoint before stopping a run — `latest.pt` is your resume point.
