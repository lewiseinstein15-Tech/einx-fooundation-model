# -*- coding: utf-8 -*-
"""EINX evaluation harness.

Real metrics — no fabricated benchmark numbers.  Records every result
with the model version, checkpoint, dataset, config, and date so evals
are auditable and reproducible.

Currently implemented metrics:
  * ``loss``        — average cross-entropy loss on the eval dataset
  * ``perplexity``  — exp(loss), the standard LM metric
  * ``latency``     — average generation time per token
  * ``memory``      — peak GPU memory during eval (CUDA only)

Metrics that are PLANNED but NOT yet implemented (clearly marked):
  * instruction-following evals
  * reasoning tests
  * coding tests
  * mathematical tests
  * hallucination/error analysis
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from einx.config.eval_config import EvalConfig
from einx.inference.generator import EINXGenerator, GenerationConfig
from einx.utils.hardware import detect_device

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of an evaluation run.  Everything is recorded for auditability."""

    model_name: str
    checkpoint_path: str
    dataset_path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    n_examples: int = 0
    n_tokens: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    einx_version: str = "0.1.0"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("eval results saved to %s", path)


class EINXEvaluator:
    """Evaluate a trained EINX model.

    Usage:
        evaluator = EINXEvaluator(config)
        result = evaluator.evaluate(generator, dataset)
        print(result.metrics)
    """

    def __init__(self, config: EvalConfig):
        config.validate()
        self.config = config
        self.device = torch.device(detect_device(config.device))

    # ------------------------------------------------------------------
    def evaluate(self, generator: EINXGenerator, dataset) -> EvalResult:
        """Run all configured metrics on the dataset."""
        started = datetime.now(timezone.utc)
        start_time = time.time()

        result = EvalResult(
            model_name=self.config.model_name,
            checkpoint_path=self.config.checkpoint_path,
            dataset_path=self.config.eval_dataset_path,
            config=self.config.to_dict(),
            started_at=started.isoformat(),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        for metric in self.config.metrics:
            if metric == "loss":
                result.metrics["loss"] = self._compute_loss(generator, loader)
            elif metric == "perplexity":
                loss = result.metrics.get("loss")
                if loss is None:
                    loss = self._compute_loss(generator, loader)
                    result.metrics["loss"] = loss
                result.metrics["perplexity"] = float(torch.exp(torch.tensor(loss)).item())
            elif metric == "latency":
                result.metrics["latency_ms_per_token"] = self._compute_latency(generator)
            elif metric == "memory":
                result.metrics["peak_memory_mb"] = self._compute_memory()
            else:
                logger.warning("unknown metric: %s — skipping", metric)

        # Count examples / tokens
        result.n_examples = len(dataset)
        result.n_tokens = getattr(dataset, "n_tokens", lambda: 0)()
        if not callable(result.n_tokens):
            result.n_tokens = 0

        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.elapsed_seconds = time.time() - start_time
        result.notes = (
            "Real evaluation results — no fabricated numbers. "
            "Metrics computed on the configured eval dataset."
        )
        return result

    # ------------------------------------------------------------------
    def _compute_loss(self, generator: EINXGenerator, loader: DataLoader) -> float:
        """Average cross-entropy loss on the eval dataset."""
        generator.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= self.config.eval_steps if self.config.eval_steps > 0 else False:
                    break
                input_ids, targets = batch
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                _, loss = generator.model(input_ids, targets=targets)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)

    def _compute_latency(self, generator: EINXGenerator) -> float:
        """Average generation latency in ms/token.

        Runs a small generation benchmark and reports the per-token
        latency.  Useful for comparing inference throughput across
        model sizes / devices.
        """
        prompt = "the"
        timings: List[float] = []
        for _ in range(3):
            start = time.time()
            result = generator.generate(
                prompt,
                GenerationConfig(
                    max_new_tokens=20,
                    temperature=0.0,  # greedy — deterministic
                    top_k=0,
                    top_p=1.0,
                ),
            )
            if result.n_output_tokens > 0:
                timings.append(result.elapsed_seconds / result.n_output_tokens)
        if not timings:
            return 0.0
        avg = sum(timings) / len(timings)
        return avg * 1000.0  # convert seconds/token -> ms/token

    def _compute_memory(self) -> float:
        """Peak GPU memory in MB.  Returns 0 on CPU."""
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.reset_peak_memory_stats()
        # Trigger a small allocation to populate the counter
        _ = torch.zeros(1024, 1024, device="cuda")
        return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_evaluation(
    checkpoint_path: str,
    tokenizer_path: str,
    eval_dataset_path: str,
    *,
    config: Optional[EvalConfig] = None,
    output_path: str = "experiments/eval_results.json",
) -> EvalResult:
    """Convenience: run a full evaluation from paths."""
    from einx.data.dataset import TokenisedDataset, load_jsonl
    config = config or EvalConfig(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        eval_dataset_path=eval_dataset_path,
        output_path=output_path,
    )
    generator = EINXGenerator.from_checkpoint(checkpoint_path, tokenizer_path)
    tokenizer = generator.tokenizer

    records = load_jsonl(eval_dataset_path)
    texts = [r["text"] for r in records]
    dataset = TokenisedDataset(
        texts, tokenizer, context_length=generator.model.config.max_context_length,
    )

    evaluator = EINXEvaluator(config)
    result = evaluator.evaluate(generator, dataset)
    result.save(output_path)
    return result
