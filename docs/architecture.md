# EINX Architecture

## Design principles

1. **Correctness > reproducibility > maintainability > performance > complexity.**
2. Every major subsystem has a clear interface.
3. No giant files. No unnecessary dependencies. No dead code. No fake implementations.
4. If a feature cannot yet be implemented properly, document it clearly and create a clean extension point.

## Stack

- **PyTorch 2.0+** — the only deep-learning framework. Mature, well-supported, no abstraction tax.
- **NumPy** — numerical utilities.
- **Pydantic** — request/response validation in the API.
- **FastAPI + Uvicorn** — HTTP server.
- **PyYAML** — config files.
- **pytest** — testing.
- **httpx** — API client for tests.

No HuggingFace `tokenizers` (Rust dep), no `transformers`, no `datasets` —
EINX is built from scratch so every component is inspectable.

## Module map

### `einx/config/`
Typed dataclasses for every configuration: `EINXModelConfig`, `TrainingConfig`, `EvalConfig`.
- Every parameter is configurable — no hardcoded magic numbers.
- YAML roundtrip (`from_yaml` / `save_yaml`).
- Validation at construction time — bad configs fail fast.

### `einx/tokenizer/`
Pure-Python byte-level BPE.
- 256 base byte tokens + 4 special tokens (`<pad>`, `<bos>`, `<eos>`, `<unk>`) + learned merges.
- Word-boundary marker (Ġ, GPT-2 convention) so the tokenizer learns word-initial vs word-internal tokens.
- Deterministic given the same corpus + vocab_size.
- JSON-serializable — vocab + merges + special tokens in one file.

### `einx/data/`
Dataset pipeline.
- `load_jsonl` / `write_jsonl` — I/O with malformed-record skipping.
- `TextDataset` — in-memory text records.
- `TokenisedDataset` — pre-tokenised, returns `(input_ids, target_ids)` tensors.
- `train_val_test_split` — seeded, reproducible, no overlap.
- `generate_synthetic_corpus` — for first-run experiments + tests.

### `einx/model/`
The transformer.
- `EINXTransformer` — the full model (embeddings + blocks + LM head).
- `RotaryPositionEmbedding` — RoPE (Su et al., 2021).
- `MultiHeadAttention` — causal, fused QKV, no bias (LLaMA convention).
- `FeedForward` — SwiGLU (gated, 2/3 effective intermediate size).
- `RMSNorm` / `LayerNorm` — pick via `norm_type` config.
- Weight tying (input embeddings ↔ LM head) — configurable.

### `einx/training/`
The training loop.
- `EINXTrainer` — full loop with grad accumulation, mixed precision, checkpoints, resume.
- `build_optimizer` — AdamW / Adam / SGD with separate weight decay for biases.
- `build_scheduler` — cosine / linear / constant with warmup.
- `TrainingState` — checkpointed state (step, epoch, optimizer, scheduler, RNG).

### `einx/inference/`
Generation.
- `EINXGenerator` — high-level interface (load checkpoint, generate, stream).
- `GenerationConfig` — temperature, top-k, top-p, repetition penalty, stop sequences, seed.
- `GenerationResult` — text + token IDs + timing + throughput.

### `einx/evaluation/`
Metrics.
- `EINXEvaluator` — runs all configured metrics on a dataset.
- `EvalResult` — auditable record (model version, checkpoint, dataset, config, date, metrics).

### `einx/api/`
HTTP API.
- `build_app(generator)` — FastAPI app with `/health`, `/model`, `/generate`, `/chat`.
- Pydantic models for request/response validation.
- Runs without a model loaded — `/health` works, other endpoints return 503.

### `einx/utils/`
- `detect_device` — CUDA > MPS > CPU auto-detection.
- `set_seed` — Python + NumPy + PyTorch RNG.
- `get_logger` — configured stderr logger.

## Data flow

```
raw JSONL → load_jsonl → TextDataset → TokenisedDataset → DataLoader
                                                              ↓
                                                    EINXTransformer.forward
                                                              ↓
                                                    loss + backward
                                                              ↓
                                                    optimizer.step + scheduler.step
                                                              ↓
                                                    checkpoint (every N steps)
                                                              ↓
                                                    EINXGenerator.from_checkpoint
                                                              ↓
                                                    generate / stream / API
```

## Component replacement

Every component has a clean interface so you can swap implementations:

- **Tokenizer** — implement `encode(text) -> List[int]` + `decode(ids) -> str` + `save`/`load`.
- **Model** — implement `forward(input_ids, targets=None) -> (logits, loss)` + `generate(...)`.
- **Optimizer** — add to `build_optimizer` factory.
- **Scheduler** — add to `build_scheduler` factory.
- **API** — FastAPI app factory; add endpoints in `build_app`.
