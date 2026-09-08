# EINX Roadmap

## Current state (v0.1 — Research Prototype)

✅ Implemented and tested:

- Pure-Python byte-level BPE tokenizer (train, encode, decode, save/load)
- Configurable decoder-only transformer (RoPE, RMSNorm, SwiGLU, causal attention)
- Real training loop (AdamW, cosine LR, grad accumulation, mixed precision, checkpoints, resume)
- Real inference engine (temperature, top-k, top-p, repetition penalty, streaming)
- Real evaluation harness (loss, perplexity, latency, memory)
- Real HTTP API (FastAPI: /health, /model, /generate, /chat)
- 81 passing tests
- End-to-end verified: train → checkpoint → generate → evaluate → serve

Model sizes **defined but not trained**:

- `einx-experimental` (~1.3M params) — architecture + configs ready, training verified end-to-end
- `einx-small` (~25M params) — architecture + configs ready, not trained

## Phase 1 — Research Prototype (DONE)

The v0.1 milestone is complete. EINX-Experimental trains, generates,
evaluates, and serves. The architecture is real; the model just needs
more data + steps to be useful.

## Phase 2 — Instruction-Tuned EINX (PLANNED)

- SFT (supervised fine-tuning) on instruction datasets
- Chat template (system / user / assistant turns, learned during tuning)
- DPO / RLHF for alignment (optional)
- Instruction-following evals

Estimated effort: 2-4 weeks of focused work.

## Phase 3 — Larger Pretrained Models (PLANNED)

- EINX-1B (~1B params)
- EINX-7B (~7B params)
- EINX-14B (~14B params)
- EINX-32B (~32B params)
- EINX-MoE (Mixture-of-Experts)

These are **configurations only** — no trained checkpoints exist.
Scaling requires:
- Distributed training (DDP / FSDP)
- Significant compute (1B = ~1 GPU-day; 7B = ~8 GPU-days; 32B = ~64+ GPU-days)
- Curated pretraining datasets

## Phase 4 — Efficient Inference (PLANNED)

- KV cache (avoid recomputing on every generation step)
- Speculative decoding
- INT8 / INT4 quantization (when the framework supports it)
- Batching for higher throughput

## Phase 5 — Multimodal Capabilities (PLANNED)

- Vision encoder (image -> embedding)
- Audio encoder (speech -> embedding)
- Cross-modal attention
- Multimodal generation (text + image)

## Phase 6 — Tool-Use Capabilities (PLANNED)

- Function calling (model emits structured tool-call requests)
- Tool routing (EINX -> external tool -> result -> EINX)
- Multi-step reasoning over tool calls

## Phase 7 — Agent Integration (PLANNED)

- Long-horizon planning
- Memory (short-term + long-term)
- Self-reflection
- Multi-agent collaboration

## Phase 8 — EINX ↔ JEXI OS Integration (PLANNED)

EINX exposes a clean model interface:

```python
gen = EINXGenerator.from_checkpoint(...)
gen.generate(prompt, config)        # → text
gen.stream(prompt, config)          # → iterator of chunks
gen.model_info()                    # → metadata
```

JEXI OS can use EINX as one of its intelligence providers. JEXI is NOT
required for EINX to work — EINX is independently runnable.

## What's NOT on the roadmap

- Pretending EINX is AGI
- Fabricating benchmark numbers
- Claiming capabilities that aren't implemented
- Wrapping another model and calling it EINX

The objective is a real, independently engineered foundation model —
not a marketing demo.
