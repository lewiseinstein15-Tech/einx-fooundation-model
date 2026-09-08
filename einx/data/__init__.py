# -*- coding: utf-8 -*-
"""EINX data pipeline.

Real dataset loading, validation, and tokenisation.  No magic —
just JSONL in, tokenised tensors out.

Expected dataset format (JSONL, one record per line):

    {"text": "..."}
    {"text": "...", "source": "wiki", "id": 42}

The pipeline:
    raw jsonl  →  validated records  →  tokenised tensors  →  train/val/test split
"""

from einx.data.dataset import (
    TextDataset,
    TokenisedDataset,
    PackedDataset,
    StreamingTextDataset,
    load_jsonl,
    write_jsonl,
    train_val_test_split,
)
from einx.data.synthetic import generate_synthetic_corpus

__all__ = [
    "TextDataset",
    "TokenisedDataset",
    "PackedDataset",
    "StreamingTextDataset",
    "load_jsonl",
    "write_jsonl",
    "train_val_test_split",
    "generate_synthetic_corpus",
]
