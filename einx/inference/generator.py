# -*- coding: utf-8 -*-
"""EINX text generator.

Loads a model + tokenizer and produces text.  Used by the CLI, the
HTTP API, and the eval harness.

The generation logic itself lives in :meth:`EINXTransformer.generate`
— this module wraps it with a friendly interface + repetition penalty
+ stop-sequence handling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

from einx.config.model_config import EINXModelConfig
from einx.model.transformer import EINXTransformer
from einx.tokenizer.bpe import BPETokenizer
from einx.utils.hardware import detect_device


@dataclass
class GenerationConfig:
    """Configuration for a single generation call."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0     # 1.0 = no penalty
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.top_p is not None and not (0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


@dataclass
class GenerationResult:
    """Result of a generation call.  Includes timing + token count."""

    text: str
    token_ids: List[int]
    n_input_tokens: int
    n_output_tokens: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.n_output_tokens / self.elapsed_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "token_ids": self.token_ids,
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }


class EINXGenerator:
    """High-level text generation interface.

    Usage:
        gen = EINXGenerator.from_checkpoint("checkpoints/run/latest.pt", "tok.json")
        result = gen.generate("Once upon a time", max_new_tokens=64)
        print(result.text)

    Or stream tokens one at a time:
        for token_text in gen.stream("Once upon a time", max_new_tokens=64):
            print(token_text, end="", flush=True)
    """

    def __init__(
        self,
        model: EINXTransformer,
        tokenizer: BPETokenizer,
        *,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or detect_device("auto"))
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        *,
        device: Optional[str] = None,
    ) -> "EINXGenerator":
        """Load a model + tokenizer from disk."""
        model = EINXTransformer.load(str(checkpoint_path), map_location="cpu")
        tokenizer = BPETokenizer.load(str(tokenizer_path))
        return cls(model, tokenizer, device=device)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from a prompt.  Returns the full result."""
        config = config or GenerationConfig()
        config.validate()

        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        if not input_ids:
            input_ids = [self.tokenizer.special.bos_id]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        n_input = len(input_ids)

        # Apply repetition penalty to the prompt's logits at the start
        # (simple version: just track and penalise tokens we've seen)
        seen_ids: set = set(input_ids)

        start = time.time()
        output_ids: List[int] = []

        for _ in range(config.max_new_tokens):
            # Crop to context length
            context = input_tensor[:, -self.model.config.max_context_length:]
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :]

            # Temperature
            if config.temperature > 0:
                next_logits = next_logits / config.temperature

            # Repetition penalty (apply before top-k/top-p)
            if config.repetition_penalty > 1.0 and seen_ids:
                for tid in seen_ids:
                    next_logits[0, tid] /= config.repetition_penalty

            # Top-k
            if config.top_k is not None and config.top_k > 0:
                top_k = min(config.top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits >= threshold,
                    next_logits,
                    torch.full_like(next_logits, float("-inf")),
                )

            # Top-p (nucleus)
            if config.top_p is not None and 0 < config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = torch.zeros_like(next_logits, dtype=torch.bool)
                for b in range(next_logits.size(0)):
                    indices_to_remove[b].scatter_(
                        -1, sorted_indices[b], sorted_indices_to_remove[b]
                    )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample
            if config.temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_id = next_token.item()
            output_ids.append(next_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            seen_ids.add(next_id)

            # Check stop sequences (decoded)
            decoded_so_far = self.tokenizer.decode(output_ids)
            if any(seq in decoded_so_far for seq in config.stop_sequences):
                break

            # Check EOS
            if next_id == self.model.config.eos_token_id:
                break

        elapsed = time.time() - start
        # Decode only the generated portion
        text = self.tokenizer.decode(output_ids)
        return GenerationResult(
            text=text,
            token_ids=output_ids,
            n_input_tokens=n_input,
            n_output_tokens=len(output_ids),
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream decoded tokens one at a time.

        Yields the decoded text for each generated token as soon as it
        is produced.  The first yield may be an empty string if the
        token decodes to a word-boundary marker that doesn't print —
        callers should accumulate the chunks and the final text will be
        coherent.
        """
        config = config or GenerationConfig()
        config.validate()

        if config.seed is not None:
            torch.manual_seed(config.seed)

        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        if not input_ids:
            input_ids = [self.tokenizer.special.bos_id]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        seen_ids: set = set(input_ids)

        for _ in range(config.max_new_tokens):
            context = input_tensor[:, -self.model.config.max_context_length:]
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :]

            if config.temperature > 0:
                next_logits = next_logits / config.temperature
            if config.repetition_penalty > 1.0 and seen_ids:
                for tid in seen_ids:
                    next_logits[0, tid] /= config.repetition_penalty
            if config.top_k is not None and config.top_k > 0:
                top_k = min(config.top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits >= threshold,
                    next_logits,
                    torch.full_like(next_logits, float("-inf")),
                )

            if config.temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_id = next_token.item()
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            seen_ids.add(next_id)

            # Yield the decoded form of THIS new token — this is the
            # simplest streaming contract: one yield per generated
            # token.  Callers concatenate to get the full text.
            chunk = self.tokenizer.decode([next_id])
            yield chunk

            if next_id == self.model.config.eos_token_id:
                break

            # Check stop sequences against the accumulated output
            full_so_far = self.tokenizer.decode(input_tensor[0].tolist())
            if any(seq in full_so_far for seq in config.stop_sequences):
                break

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------
    def model_info(self) -> Dict[str, Any]:
        """Return model metadata for the /model API endpoint."""
        cfg = self.model.config
        return {
            "name": cfg.name,
            "version": cfg.version,
            "arch": cfg.arch,
            "vocab_size": cfg.vocab_size,
            "hidden_dim": cfg.hidden_dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "max_context_length": cfg.max_context_length,
            "n_params": self.model.n_params,
            "device": str(self.device),
        }
