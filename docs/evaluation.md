# EINX Evaluation Guide

## Overview

EINX evaluation computes real metrics on a held-out dataset. Every result
is recorded with full provenance (model version, checkpoint, dataset,
config, date) so evals are auditable and reproducible.

**Never hard-code impressive benchmark numbers.** Every metric in an EINX
eval result is computed from the actual model output.

## Currently implemented metrics

| Metric | Description |
|--------|-------------|
| `loss` | Average cross-entropy loss on the eval dataset |
| `perplexity` | `exp(loss)` — the standard LM metric |
| `latency` | Average ms per generated token |
| `memory` | Peak GPU memory during eval (CUDA only) |

## Planned but not yet implemented

- Instruction-following evals (e.g., a held-out instruction set)
- Reasoning tests (e.g., Big-Bench-style logical puzzles)
- Coding tests (e.g., HumanEval-style)
- Mathematical tests (e.g., GSM8K-style)
- Hallucination / error analysis

These are **planned, not implemented**. The infrastructure (EvalResult,
EvalConfig, EINXEvaluator) is ready; the benchmark suites are not.

## Running an evaluation

```bash
python -m einx.cli evaluate \
    --checkpoint checkpoints/einx-experimental-run-1/latest.pt \
    --tokenizer data/tokenized/einx-bpe.json \
    --dataset data/processed/test.jsonl \
    --output experiments/eval_results.json
```

## Result format

`experiments/eval_results.json`:

```json
{
  "model_name": "einx-experimental",
  "checkpoint_path": "checkpoints/einx-experimental-run-1/latest.pt",
  "dataset_path": "data/processed/test.jsonl",
  "metrics": {
    "loss": 7.0955,
    "perplexity": 1208.3,
    "latency_ms_per_token": 1.57
  },
  "n_examples": 25,
  "n_tokens": 0,
  "config": { ... },
  "started_at": "2026-09-08T06:03:30+00:00",
  "finished_at": "2026-09-08T06:03:31+00:00",
  "elapsed_seconds": 0.85,
  "einx_version": "0.1.0",
  "notes": "Real evaluation results — no fabricated numbers..."
}
```

## Interpreting results

- **Loss** — lower is better. Random initialization gives ~`ln(vocab_size)` ≈ 8.3
  for vocab 4096. A well-trained small LM should reach 3-5 on natural text.
- **Perplexity** — lower is better. `exp(loss)`. Random ≈ vocab_size; good ≈ 20-100.
- **Latency** — depends on device. CPU: 1-10 ms/token. CUDA: 0.1-1 ms/token.
- **Memory** — only meaningful on CUDA. CPU returns 0.

## Custom evaluations

Implement your own evaluator by subclassing `EINXEvaluator`:

```python
from einx.evaluation.evaluator import EINXEvaluator, EvalResult

class MyEvaluator(EINXEvaluator):
    def evaluate(self, generator, dataset):
        result = super().evaluate(generator, dataset)
        # Add custom metric
        result.metrics["my_metric"] = self._compute_my_metric(generator)
        return result
```

## Benchmark results policy

- Every benchmark number in EINX must come from a real evaluation run.
- The result file (with checkpoint + dataset paths + date) is the proof.
- Never copy benchmark numbers from marketing materials or other models.
- If a benchmark suite is not yet implemented, say so — don't fake it.
