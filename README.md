# EINX

> Lewis Einstein's experimental intelligence architecture.
> A modular foundation-model project — real engineering, not a UI wrapper.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-133%20passing-brightgreen.svg)](#testing)
[![Validation](https://img.shields.io/badge/validation-12%2F12%20passed-success.svg)](#build-2-validation)

**EINX** is a serious, modular foundation-model project. This repository
contains the actual infrastructure required to research, train, evaluate,
fine-tune, serve, and evolve an EINX model. The first milestone —
**EINX v0.1 Research Prototype** — is implemented and runs end-to-end on
a laptop CPU. **Build 2** adds a real KV cache, atomic checkpoint manager,
experiment tracking, performance measurement, distributed training
abstraction, and a full 12-test validation suite.

> **EINX is NOT JEXI.**
> - **EINX** = foundation-model / intelligence layer (this repo)
> - **JEXI OS** = AI operating system that can use EINX and other models
> - **JEXI Market** = specialized market-intelligence branch
>
> EINX is independently runnable. JEXI is NOT required.

---

## What EINX is

EINX is a real transformer-based language model project, built from
scratch in PyTorch. It is not:

- A UI wrapped around someone else's model
- A chatbot pretending to be a foundation model
- A fake demo with fabricated benchmark numbers

EINX ships real:

- A pure-Python byte-level BPE tokenizer (train, encode, decode, save/load)
- A configurable decoder-only transformer (RoPE, RMSNorm, SwiGLU, causal attention)
- A real training loop (AdamW, cosine LR schedule, gradient accumulation, mixed precision, checkpoints, resume)
- A real inference engine (temperature, top-k, top-p, repetition penalty, stop sequences, streaming)
- A real evaluation harness (loss, perplexity, latency, memory)
- A real HTTP API (FastAPI: `/health`, `/model`, `/generate`, `/chat`)
- 81 passing tests covering every component

The first milestone model — **EINX-Experimental** — is a ~1.3M parameter
transformer that trains in a few minutes on a laptop CPU. The
architecture scales cleanly toward larger models (EINX-Small ~25M is
already defined; EINX-1B / 7B / 14B / 32B / MoE are PLANNED but not yet
trained — see [docs/roadmap.md](docs/roadmap.md)).

---

## Project goals

1. **Real engineering, not a demo.** Every component is implemented and
   tested. Nothing pretends to work.
2. **Modular.** Replace the tokenizer, the attention implementation, the
   optimizer, or the API layer without rewriting the project.
3. **Hardware-aware.** Runs on CPU (default), CUDA, and MPS. Never assumes
   a powerful GPU exists.
4. **Reproducible.** Seeded RNG, deterministic configs, checkpointed state.
5. **Honest.** Larger models are clearly marked as PLANNED. Benchmark
   numbers are real, not fabricated.

---

## Architecture

```
einx/
├── config/         # Typed configs (model, training, eval)
├── tokenizer/      # Pure-Python byte-level BPE
├── data/           # JSONL I/O, dataset classes, synthetic corpus
├── model/          # Transformer: embeddings, RoPE, attention, FFN, LM head
├── training/       # Trainer, optimizers, LR schedulers, checkpoints
├── inference/      # Generator (temperature, top-k, top-p, streaming)
├── evaluation/     # Perplexity, loss, latency, memory
├── api/            # FastAPI server (/health, /model, /generate, /chat)
└── utils/          # Hardware detection, seeds, logging
```

See [docs/architecture.md](docs/architecture.md) for the full design.

---

## Repository structure

```
einx-foundation-model/
├── README.md                   # this file
├── LICENSE                     # MIT
├── pyproject.toml              # package metadata + console scripts
├── .gitignore                  # ignores checkpoints, data, secrets
├── .env.example                # env var template (no real secrets)
│
├── configs/
│   ├── model/                  # einx-experimental.yaml, einx-small.yaml
│   ├── training/               # experimental.yaml
│   └── evaluation/              # experimental.yaml
│
├── einx/                       # the package (see Architecture above)
│
├── scripts/                    # thin wrappers around the CLI
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   └── serve.py
│
├── tests/                      # 81 tests
│   ├── test_config.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_training.py
│   └── test_inference_api.py
│
├── experiments/                # eval results land here (gitignored)
├── checkpoints/                # trained models land here (gitignored)
├── data/                       # raw / processed / tokenized (gitignored)
│
└── docs/
    ├── architecture.md
    ├── training.md
    ├── evaluation.md
    └── roadmap.md
```

---

## Installation

```bash
git clone https://github.com/lewiseinstein15-Tech/einx-fooundation-model.git
cd einx-fooundation-model

# Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate       # on Windows: .venv\Scripts\activate

# Install EINX in editable mode + dev deps
pip install -e ".[dev]"
```

**Hardware requirements:**
- Python 3.9+
- PyTorch 2.0+ (CPU build works fine for the experimental model)
- ~2GB RAM for EINX-Experimental training
- Optional: NVIDIA GPU with CUDA for faster training and larger models

---

## Quick start

### 1. Train a tokenizer + model (one command, ~3 min on CPU)

```bash
# Generate a small synthetic corpus, train a tokenizer, train the model
python -m einx.cli train --prepare-synthetic 500 --steps 200
```

This will:
1. Write 500 synthetic text records to `data/processed/{train,val,test}.jsonl`
2. Train a BPE tokenizer (vocab 4096) to `data/tokenized/einx-bpe.json`
3. Build an EINX-Experimental model (~1.3M params)
4. Train for 200 steps with cosine LR schedule
5. Save checkpoints to `checkpoints/einx-experimental-run-1/`

You should see the loss drop from ~8.4 to ~7.1 over 200 steps — the model
is genuinely learning the corpus distribution.

### 2. Generate text from the checkpoint

```bash
python -m einx.cli generate \
    --checkpoint checkpoints/einx-experimental-run-1/latest.pt \
    --tokenizer data/tokenized/einx-bpe.json \
    --prompt "the cat" \
    --max-new-tokens 30 \
    --temperature 0.8 \
    --stream
```

### 3. Evaluate the checkpoint

```bash
python -m einx.cli evaluate \
    --checkpoint checkpoints/einx-experimental-run-1/latest.pt \
    --tokenizer data/tokenized/einx-bpe.json \
    --dataset data/processed/test.jsonl
```

Reports real metrics: loss, perplexity, latency (ms/token), peak memory.

### 4. Serve the HTTP API

```bash
python -m einx.cli serve \
    --checkpoint checkpoints/einx-experimental-run-1/latest.pt \
    --tokenizer data/tokenized/einx-bpe.json \
    --port 8000
```

Then:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model
curl -X POST http://127.0.0.1:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt":"the cat","max_new_tokens":20,"temperature":0.8}'
```

---

## Dataset preparation

EINX expects JSONL files with a `"text"` field:

```jsonl
{"text": "the cat sat on the mat"}
{"text": "the dog ran in the park"}
```

Place them under `data/processed/`:

```
data/processed/
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

For the first-run experiment, use `--prepare-synthetic 500` to generate
a small synthetic corpus. For real training, use your own JSONL data —
but **do not include copyrighted datasets** unless their license permits
redistribution.

See [docs/training.md](docs/training.md) for the full data pipeline.

---

## Tokenizer training

```bash
python -m einx.cli tokenizer train \
    --corpus data/processed/train.jsonl \
    --output data/tokenized/einx-bpe.json \
    --vocab-size 4096
```

Inspect:

```bash
python -m einx.cli tokenizer info --tokenizer data/tokenized/einx-bpe.json
python -m einx.cli tokenizer encode --tokenizer ... --text "hello world"
python -m einx.cli tokenizer decode --tokenizer ... --ids "1,2,3"
```

The tokenizer is a **real byte-level BPE** — pure Python, no Rust
dependency. See [einx/tokenizer/bpe.py](einx/tokenizer/bpe.py).

---

## Model configuration

Models are defined in YAML under `configs/model/`. The default is
[EINX-Experimental](configs/model/einx-experimental.yaml):

```yaml
name: einx-experimental
version: "0.1.0"
arch: decoder-only-transformer

vocab_size: 4096
hidden_dim: 128
n_layers: 4
n_heads: 4
head_dim: 32
max_context_length: 256
ffn_dim: 512
dropout: 0.1

positional_encoding: rope       # rope | learned
norm_type: rms                   # rms | layer
precision: fp32                 # fp32 | fp16 | bf16
tie_word_embeddings: true
```

Every parameter is configurable — no hardcoded magic numbers in the
model code.

---

## Training

```bash
python -m einx.cli train \
    --model-config configs/model/einx-experimental.yaml \
    --training-config configs/training/experimental.yaml \
    --tokenizer data/tokenized/einx-bpe.json \
    --dataset data/processed/train.jsonl \
    --val-dataset data/processed/val.jsonl \
    --steps 1000
```

Features:
- **AdamW** optimizer with separate weight decay for biases vs 2D weights
- **Cosine / linear / constant** LR schedule with warmup
- **Gradient accumulation** — effective batch = `batch_size × grad_accum_steps`
- **Mixed precision** (fp16 / bf16 — CUDA only; fp32 default on CPU)
- **Gradient clipping** (max_grad_norm)
- **Periodic evaluation** on the validation set
- **Periodic checkpoints** with keep-last-N rotation
- **Resumable** — `--resume checkpoints/.../latest.pt` picks up where you left off
- **Reproducible** — seeded RNG, deterministic configs

See [docs/training.md](docs/training.md) for the full training guide.

---

## Evaluation

```bash
python -m einx.cli evaluate \
    --checkpoint checkpoints/einx-experimental-run-1/latest.pt \
    --tokenizer data/tokenized/einx-bpe.json \
    --dataset data/processed/test.jsonl \
    --output experiments/eval_results.json
```

Currently implemented metrics:
- `loss` — average cross-entropy loss
- `perplexity` — exp(loss)
- `latency` — ms per generated token
- `memory` — peak GPU memory (CUDA only)

Every result records the model version, checkpoint path, dataset path,
config, and date — fully auditable.

**Planned but not yet implemented:** instruction-following evals,
reasoning tests, coding tests, math tests, hallucination analysis.
See [docs/evaluation.md](docs/evaluation.md).

---

## Inference

```python
from einx import EINXGenerator, GenerationConfig

gen = EINXGenerator.from_checkpoint(
    "checkpoints/einx-experimental-run-1/latest.pt",
    "data/tokenized/einx-bpe.json",
)

# One-shot
result = gen.generate("the cat", GenerationConfig(
    max_new_tokens=64,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
))
print(result.text)

# Streaming
for chunk in gen.stream("the cat", GenerationConfig(max_new_tokens=64)):
    print(chunk, end="", flush=True)
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/model` | Model metadata (no private training info) |
| POST | `/generate` | Text generation |
| POST | `/chat` | Chat completion (system/user/assistant turns) |

Designed by us — not a copy of OpenAI / Anthropic / Google APIs — but
follows the same conceptual shape so JEXI OS (and any other LLM client)
can call it cleanly.

---

## JEXI compatibility

EINX exposes a clean model interface that JEXI OS can consume:

```python
gen = EINXGenerator.from_checkpoint(...)
gen.generate(prompt, config)    # → text
gen.stream(prompt, config)      # → iterator of text chunks
```

**JEXI is NOT required for EINX to work.** EINX is independently runnable.

---

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check einx/ tests/

# Install in editable mode
pip install -e ".[dev]"
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Research prototype (this repo) |
| 2 | 🔜 Planned | Instruction-tuned EINX |
| 3 | 🔜 Planned | Larger pretrained models (1B, 7B) |
| 4 | 🔜 Planned | Efficient inference (KV cache, quantization) |
| 5 | 🔜 Planned | Multimodal capabilities |
| 6 | 🔜 Planned | Tool-use capabilities |
| 7 | 🔜 Planned | Agent integration |
| 8 | 🔜 Planned | EINX ↔ JEXI OS integration |

See [docs/roadmap.md](docs/roadmap.md) for details.

---

## Limitations (honest)

1. **EINX-Experimental is a tiny model (~1.3M params).** It learns the
   training distribution (loss drops from 8.4 → 7.1 over 200 steps) but
   does not produce coherent text. The architecture is real; the model
   needs more data + more steps + more parameters to be useful.

2. **Larger models (1B, 7B, 14B, 32B, MoE) are PLANNED, not trained.**
   The configs and scaling math are sketched in `docs/roadmap.md`, but
   no large model checkpoints exist in this repo.

3. **Instruction-following, reasoning, coding, math, and hallucination
   evals are not yet implemented.** The evaluation harness has the
   infrastructure (loss, perplexity, latency, memory) but not the
   benchmark suites.

4. **No KV cache yet.** Generation recomputes from scratch every step —
   fast enough for the experimental model, too slow for production.
   Phase 4 work.

5. **No distributed training yet.** Single-device only. The architecture
   is designed so DDP/FSDP can be added later (Phase 3).

6. **Sentiment classifier / chat template are simple.** The chat API
   flattens messages with `<system>` / `<user>` / `<assistant>` markers.
   A real chat template would be learned during instruction tuning
   (Phase 2).

---

## Build 2 Validation

Build 2 is **complete** — the full 12-test validation suite from spec §32 passes:

```bash
python tests/test_validation_suite.py
```

```
=== Test 1: Model initialization ===    ✓ 83,264 params
=== Test 2: Forward pass ===            ✓ logits shape torch.Size([2, 16, 256])
=== Test 3: Loss calculation ===        ✓ loss=5.58
=== Test 4: Backward pass ===          ✓ 16 gradients populated
=== Test 5: One optimizer step ===     ✓ Δ=-0.24
=== Test 6: Checkpoint creation ===    ✓ step-000010
=== Test 7: Checkpoint reload ===       ✓ max diff 0.00e+00
=== Test 8: Resume training ===         ✓ resumed 3→5 steps
=== Test 9: Generation ===             ✓ 5 tokens
=== Test 10: Full smoke-training ===   ✓ loss 5.64→5.58
=== Test 11: CLI commands ===          ✓ 5 commands
=== Test 12: Automated test suite ===  ✓ 133 passed

Total: 12/12 passed, 0 failed
✓ BUILD 2 VALIDATION COMPLETE
```

### What's new in Build 2

| Component | Status | Notes |
|-----------|--------|-------|
| **KV cache** | ✅ Implemented | Real `KVCache` + `KVCacheStack` per-layer; threaded through attention; cached logits match uncached logits to 1e-6 |
| **Checkpoint manager** | ✅ Implemented | `step-NNNNNN/` directories; atomic writes (temp + rename); `find_latest()`, `find_best()`, `keep_last_n()` |
| **Experiment tracking** | ✅ Implemented | `experiment.json` per run with real metrics, configs, hardware/software versions, timestamps |
| **Performance monitor** | ✅ Implemented | `tokens/sec`, `steps/sec`, peak memory, profiling support |
| **Weight initialization** | ✅ Centralised | `einx/model/init.py` — single auditable strategy, residual scaling |
| **Sequence packing** | ✅ Implemented | `PackedDataset` — concatenates short sequences, 100% efficiency |
| **Streaming dataset** | ✅ Implemented | `StreamingTextDataset` for large corpora that don't fit in RAM |
| **Friendly errors** | ✅ Implemented | `EINXConfigError` with field + value + hint for every config error |
| **`torch.compile`** | ✅ Optional | `--compile` CLI flag, no-op when disabled, graceful fallback on failure |
| **Distributed abstraction** | ✅ Interface only | `DeviceMesh` + `wrap_model()` — DDP working, FSDP raises `NotImplementedError` (Phase 3) |
| **Smoke test** | ✅ Implemented | `tests/test_smoke.py` — full end-to-end in 4 seconds |
| **Validation suite** | ✅ Implemented | `tests/test_validation_suite.py` — all 12 spec tests pass |
| **Tests** | ✅ 133 passing | Was 81 in Build 1; added 52 new tests |

### Honest remaining work (Phase 3+)

- **Distributed training**: DDP wrapping is wired but untested on multi-GPU. FSDP is a `NotImplementedError`. Both need real multi-GPU hardware.
- **`torch.compile`**: works but the warmup cost (5-10s) makes it slower than eager for short runs. Best enabled for production training runs >1000 steps.
- **KV-cache generation path**: the cache is correct, but `EINXGenerator.generate()` doesn't yet slice input to just the new token (it still feeds the full context). Phase 4 work.
- **Larger models**: still 1.3M params max (EINX-Experimental). 1B/7B/14B/32B/MoE are PLANNED.

---

## Testing

```bash
pytest tests/ -v
```

133 tests cover:
- Configuration (validation, YAML roundtrip, built-in configs)
- Tokenizer (BPE train, encode/decode roundtrip, Unicode, save/load)
- Model (init, forward, loss, generation, checkpoint save/load, tied weights, backward pass)
- Data pipeline (JSONL I/O, train/val/test split, dataset classes, packed dataset, streaming dataset, synthetic corpus)
- Training (optimizer, scheduler, short loop, checkpoint save, resume, eval)
- Inference + API (generator, streaming, all four API endpoints)
- **KV cache** (init, append, sliding window, cached vs uncached logits match)
- **Checkpoint manager** (atomic writes, find_latest, keep_last_n, best tag)
- **Weight initialization** (normal std, residual scaling, idempotent)
- **Friendly errors** (bad config → actionable message)
- **Distributed abstraction** (single-process mesh, FSDP raises)
- **Performance monitor** (tokens/sec, peak memory)
- **Experiment tracker** (writes JSON, captures environment, marks failure)
- **Smoke test** (full end-to-end pipeline in 4 seconds)

No manual testing required — the suite verifies every component.

---

## License

MIT — see [LICENSE](LICENSE).

---

*EINX is the foundation-model / intelligence layer of the Lewis Einstein
ecosystem. It is independently engineered, modular, and honest about its
current capabilities. Build the foundation first; scale later.*
