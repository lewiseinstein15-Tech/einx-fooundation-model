# -*- coding: utf-8 -*-
"""Dataset loading + tokenisation for EINX training.

Two dataset classes:

* :class:`TextDataset` — holds raw text records, lazy-loaded from JSONL.
* :class:`TokenisedDataset` — pre-tokenised, returns PyTorch tensors
  ready to feed to the model.

Plus :func:`train_val_test_split` for reproducible partitioning and
:func:`load_jsonl` / :func:`write_jsonl` for I/O.

NEVER silently trains on corrupted data — every record is validated
and bad records are logged + dropped (not silently kept).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_jsonl(
    path: str | Path,
    *,
    text_field: str = "text",
    skip_invalid: bool = True,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of records.

    Each line must be a valid JSON object containing ``text_field``
    (default ``"text"``).  Lines that fail validation are skipped with
    a warning (unless ``skip_invalid=False`` — then they raise).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    records: List[Dict[str, Any]] = []
    n_invalid = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("record is not a JSON object")
                if text_field not in rec:
                    raise ValueError(f"missing required field: {text_field!r}")
                if not isinstance(rec[text_field], str):
                    raise ValueError(f"field {text_field!r} must be a string")
                if not rec[text_field].strip():
                    raise ValueError(f"empty text in field {text_field!r}")
                records.append(rec)
            except (json.JSONDecodeError, ValueError) as exc:
                if skip_invalid:
                    n_invalid += 1
                    continue
                raise ValueError(f"invalid record at line {line_no}: {exc}") from exc
            if max_records is not None and len(records) >= max_records:
                break

    if n_invalid > 0:
        logger.warning(
            "skipped %d invalid record(s) while loading %s "
            "(set skip_invalid=False to raise instead of skip)",
            n_invalid, path,
        )
    logger.info("loaded %d records from %s", len(records), path)
    return records


def write_jsonl(records: List[Dict[str, Any]], path: str | Path) -> None:
    """Write records to a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("wrote %d records to %s", len(records), path)


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------


def train_val_test_split(
    records: List[Dict[str, Any]],
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into train / val / test partitions.

    Uses a seeded shuffle so the split is reproducible.  No record
    appears in more than one partition.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if not 0 <= test_ratio < 1:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            f"not enough records ({n}) for the requested split ratios"
        )

    train = shuffled[:n_train]
    val = shuffled[n_train: n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """In-memory text dataset.

    Each item is a string (the ``text`` field of the record).  Use
    :class:`TokenisedDataset` to convert this into token IDs ready
    for model input.
    """

    def __init__(self, records: List[Dict[str, Any]], text_field: str = "text"):
        self.records = records
        self.text_field = text_field

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> str:
        return self.records[idx][self.text_field]

    def iter_texts(self) -> Iterator[str]:
        for rec in self.records:
            yield rec[self.text_field]

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> "TextDataset":
        records = load_jsonl(path, **kwargs)
        return cls(records)

    def to_jsonl(self, path: str | Path) -> None:
        write_jsonl(self.records, path)


class TokenisedDataset(Dataset):
    """Pre-tokenised dataset for language modeling.

    Tokenises every text record with the provided tokenizer, concatenates
    them into one long stream of token IDs, then chunks that stream into
    fixed-length sequences of ``context_length + 1`` (the +1 is so the
    target is the input shifted by one position).

    Each item is a tuple ``(input_ids, target_ids)`` where both are
    LongTensors of shape (context_length,).
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        *,
        context_length: int = 256,
        add_eos: bool = True,
    ):
        self.context_length = context_length
        self.tokenizer = tokenizer

        # Tokenise everything once — keep it in memory.  For large
        # corpora you'd want to write these to disk (Phase 3 work).
        all_ids: List[int] = []
        for text in texts:
            ids = tokenizer.encode(text, add_eos=add_eos)
            all_ids.extend(ids)
            # Add a separator between documents so the model doesn't
            # learn spurious cross-document patterns.
            all_ids.append(tokenizer.special.eos_id)

        # Chunk into context_length + 1 sequences
        chunk_size = context_length + 1
        n_chunks = (len(all_ids) - 1) // chunk_size
        # Truncate to whole chunks (drop the partial tail)
        all_ids = all_ids[: n_chunks * chunk_size]
        self.chunks = [
            torch.tensor(all_ids[i: i + chunk_size], dtype=torch.long)
            for i in range(0, len(all_ids), chunk_size)
        ]
        if not self.chunks:
            logger.warning("TokenisedDataset has 0 chunks (corpus too small for context_length=%d)",
                           context_length)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        # input = chunk[:-1], target = chunk[1:]
        return chunk[:-1], chunk[1:]

    def n_tokens(self) -> int:
        return sum(c.size(0) for c in self.chunks)
