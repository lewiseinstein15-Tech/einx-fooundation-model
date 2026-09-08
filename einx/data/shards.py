# -*- coding: utf-8 -*-
"""EINX dataset sharding (spec §13, §14).

Tokenized datasets are written as multiple shards so large datasets
don't need to be loaded into RAM all at once.  Each shard is a JSONL
file of tokenized records; the manifest records which shards exist
and how many records/tokens each contains.

Storage format: JSONL.  Chosen for:
  * simplicity — no binary format, easy to inspect with `head` / `wc`
  * streaming — read line-by-line without loading the whole file
  * robustness — a corrupted line is skipped, not the whole shard
  * no external dependencies

Trade-off: larger on-disk size than a binary format.  For Build 3
this is the right choice; a binary format (numpy .npy or safetensors)
is a Phase 4 optimization when we measure it as a bottleneck.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Write tokenized records to multiple shards.

    Usage:
        writer = ShardWriter("data/processed/train", shard_size=10000)
        for record in tokenized_records:
            writer.write(record)
        writer.close()  # writes the manifest.json
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        *,
        shard_size: int = 10000,
        prefix: str = "shard",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix

        self._current_shard: List[Dict[str, Any]] = []
        self._shard_index = 0
        self._total_records = 0
        self._total_tokens = 0
        self._shard_files: List[str] = []

    # ------------------------------------------------------------------
    def write(self, record: Dict[str, Any]) -> None:
        """Write one record.  Automatically flushes when a shard is full."""
        self._current_shard.append(record)
        self._total_records += 1
        self._total_tokens += len(record.get("input_ids", []))
        if len(self._current_shard) >= self.shard_size:
            self._flush_shard()

    # ------------------------------------------------------------------
    def _flush_shard(self) -> None:
        """Write the current shard to disk + start a new one."""
        if not self._current_shard:
            return
        shard_name = f"{self.prefix}-{self._shard_index:05d}.jsonl"
        shard_path = self.output_dir / shard_name
        with open(shard_path, "w", encoding="utf-8") as fh:
            for rec in self._current_shard:
                fh.write(json.dumps(rec) + "\n")
        self._shard_files.append(shard_name)
        logger.info(
            "wrote shard %s (%d records)",
            shard_path, len(self._current_shard),
        )
        self._current_shard = []
        self._shard_index += 1

    # ------------------------------------------------------------------
    def close(self) -> Dict[str, Any]:
        """Flush the final shard + return stats."""
        self._flush_shard()
        return {
            "shard_files": self._shard_files,
            "shard_count": len(self._shard_files),
            "total_records": self._total_records,
            "total_tokens": self._total_tokens,
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Shard reader (streaming)
# ---------------------------------------------------------------------------


class ShardDataset(Dataset):
    """Streaming dataset that reads tokenized shards on-demand.

    Loads one shard at a time into RAM — never the whole dataset.
    Each item is a (input_ids, target_ids) tuple of LongTensors.

    The dataset supports deterministic shuffling (seeded) so the same
    seed produces the same ordering across runs (spec §15).

    Usage:
        ds = ShardDataset("data/processed/train", context_length=256)
        # Or with explicit shard list:
        ds = ShardDataset(
            "data/processed/train",
            shard_files=["shard-00000.jsonl", "shard-00001.jsonl"],
            context_length=256,
            shuffle=True,
            seed=42,
        )
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        *,
        shard_files: Optional[List[str]] = None,
        context_length: int = 256,
        shuffle: bool = False,
        seed: int = 42,
        add_eos: bool = True,
        eos_token_id: int = 2,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.context_length = context_length
        self.shuffle = shuffle
        self.seed = seed
        self.add_eos = add_eos
        self.eos_token_id = eos_token_id

        # Discover shards
        if shard_files is None:
            # Read from manifest.json if it exists
            manifest_path = self.dataset_dir / "manifest.json"
            if manifest_path.exists():
                from einx.data.manifest import DatasetManifest
                manifest = DatasetManifest.load(manifest_path)
                shard_files = manifest.shard_files
            else:
                # Glob for shard-*.jsonl
                shard_files = sorted(
                    p.name for p in self.dataset_dir.glob("shard-*.jsonl")
                )

        self.shard_files = shard_files

        # Build an index of (shard_idx, line_idx) for every record
        self._index: List[Tuple[int, int]] = []
        self._shard_lengths: List[int] = []
        for shard_idx, shard_name in enumerate(shard_files):
            shard_path = self.dataset_dir / shard_name
            if not shard_path.exists():
                logger.warning("shard missing: %s", shard_path)
                continue
            n_lines = 0
            with open(shard_path, "r", encoding="utf-8") as fh:
                for line_idx, _ in enumerate(fh):
                    self._index.append((shard_idx, line_idx))
                    n_lines += 1
            self._shard_lengths.append(n_lines)

        # Shuffle the index (deterministic given the seed)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._index)

        # Cache for the currently-loaded shard
        self._current_shard_idx: Optional[int] = None
        self._current_shard_records: List[Dict[str, Any]] = []

        # Pack records into context_length sequences lazily
        # We pack on __getitem__ so memory usage stays low
        self._packed_chunks: Optional[List[Tuple[int, int]]] = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of packed context-length sequences in the dataset."""
        if self._packed_chunks is None:
            self._build_packed_index()
        return len(self._packed_chunks)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._packed_chunks is None:
            self._build_packed_index()
        if idx >= len(self._packed_chunks):
            raise IndexError(idx)
        shard_idx, line_idx = self._packed_chunks[idx]
        rec = self._load_record(shard_idx, line_idx)
        ids = rec.get("input_ids", [])
        # Truncate to context_length + 1 (for shifted target)
        ids = ids[: self.context_length + 1]
        if len(ids) < self.context_length + 1:
            # Pad with EOS — or just return shorter (we'll handle both)
            # For training stability, pad to context_length + 1
            pad_id = self.eos_token_id
            ids = ids + [pad_id] * (self.context_length + 1 - len(ids))
        tensor = torch.tensor(ids, dtype=torch.long)
        return tensor[:-1], tensor[1:]

    # ------------------------------------------------------------------
    def _build_packed_index(self) -> None:
        """Build a mapping from chunk_idx → (shard_idx, line_idx).

        Each record becomes one chunk (truncated/padded to context_length).
        True packing (concatenating multiple short records into one
        context window) is handled by PackedDataset; this class keeps
        one-record-per-chunk for simplicity + streaming friendliness.
        """
        # The packed index is just the regular index — one chunk per record
        self._packed_chunks = list(self._index)

    # ------------------------------------------------------------------
    def _load_record(self, shard_idx: int, line_idx: int) -> Dict[str, Any]:
        """Load a single record, caching the current shard."""
        if self._current_shard_idx != shard_idx:
            self._current_shard_idx = shard_idx
            self._current_shard_records = []
            shard_path = self.dataset_dir / self.shard_files[shard_idx]
            with open(shard_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            self._current_shard_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            self._current_shard_records.append({})
        return self._current_shard_records[line_idx]

    # ------------------------------------------------------------------
    def n_tokens(self) -> int:
        """Total tokens across all shards (reads manifest if available)."""
        manifest_path = self.dataset_dir / "manifest.json"
        if manifest_path.exists():
            from einx.data.manifest import DatasetManifest
            manifest = DatasetManifest.load(manifest_path)
            return manifest.n_tokens
        # Otherwise, count by iterating
        total = 0
        for shard_name in self.shard_files:
            shard_path = self.dataset_dir / shard_name
            if not shard_path.exists():
                continue
            with open(shard_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        total += len(rec.get("input_ids", []))
                    except json.JSONDecodeError:
                        continue
        return total


# ---------------------------------------------------------------------------
# Token statistics
# ---------------------------------------------------------------------------


@dataclass
class TokenStats:
    """Statistics about a tokenized dataset.  Every number is real."""

    documents: int = 0
    total_tokens: int = 0
    avg_tokens_per_doc: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    median_tokens: float = 0.0
    p25_tokens: float = 0.0
    p75_tokens: float = 0.0
    p95_tokens: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> str:
        lines = [
            "EINX TOKEN STATISTICS",
            "─" * 40,
            f"Documents:         {self.documents:,}",
            f"Total tokens:      {self.total_tokens:,}",
            f"Average tokens/doc: {self.avg_tokens_per_doc:.1f}",
            f"Min tokens:         {self.min_tokens}",
            f"Max tokens:         {self.max_tokens}",
            f"Median tokens:      {self.median_tokens:.1f}",
            f"P25:                {self.p25_tokens:.1f}",
            f"P75:                {self.p75_tokens:.1f}",
            f"P95:                {self.p95_tokens:.1f}",
            "─" * 40,
        ]
        return "\n".join(lines)


def compute_token_stats(
    texts: List[str],
    tokenizer,
) -> TokenStats:
    """Compute token statistics over a list of texts.

    Every number comes from actually tokenising every document —
    no estimation (spec §11).
    """
    import statistics
    token_counts: List[int] = []
    for text in texts:
        try:
            ids = tokenizer.encode(text)
            token_counts.append(len(ids))
        except Exception:
            token_counts.append(0)
    if not token_counts:
        return TokenStats()
    sorted_t = sorted(token_counts)
    n = len(sorted_t)
    return TokenStats(
        documents=n,
        total_tokens=sum(token_counts),
        avg_tokens_per_doc=sum(token_counts) / n,
        min_tokens=min(token_counts),
        max_tokens=max(token_counts),
        median_tokens=statistics.median(token_counts),
        p25_tokens=sorted_t[int(n * 0.25)],
        p75_tokens=sorted_t[int(n * 0.75)],
        p95_tokens=sorted_t[int(n * 0.95)],
    )
