# -*- coding: utf-8 -*-
"""EINX document deduplicator (spec §6).

Document-level deduplication using deterministic hashing.  Supports:

  * exact matching (raw text)
  * whitespace-normalized matching (default)
  * lowercase + whitespace-normalized matching (more aggressive)

Per spec §6: "Do not claim semantic deduplication if only exact/
normalized hashing is implemented."  This module does NOT claim
semantic deduplication — only exact / normalized exact.

A :class:`NearDuplicateDetector` abstraction is provided in
``einx/data/cleaner.py`` as a clean extension point for future
MinHash / SimHash-based near-duplicate detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from einx.data.dataset import load_jsonl, write_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deduplication report
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationReport:
    """Result of deduplicating a dataset."""

    records_in: int = 0
    records_out: int = 0
    duplicates_removed: int = 0
    unique_hashes: int = 0
    normalization: str = "whitespace"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> str:
        lines = [
            "EINX DATA DEDUPLICATION",
            "─" * 40,
            f"Records in:           {self.records_in:,}",
            f"Records out:          {self.records_out:,}",
            f"Duplicates removed:   {self.duplicates_removed:,}",
            f"Unique documents:     {self.unique_hashes:,}",
            f"Normalization:        {self.normalization}",
            "─" * 40,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DeduplicationConfig:
    """Configuration for document deduplication."""

    text_field: str = "text"
    normalization: str = "whitespace"  # "none" | "whitespace" | "lowercase"

    def validate(self) -> None:
        if self.normalization not in ("none", "whitespace", "lowercase"):
            raise ValueError(
                f"normalization must be 'none' | 'whitespace' | 'lowercase', "
                f"got {self.normalization!r}"
            )


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class DatasetDeduplicator:
    """Deduplicate a dataset by document hash.

    Usage:
        dedup = DatasetDeduplicator(DeduplicationConfig(text_field="text"))
        report = dedup.dedupe_file("clean.jsonl", "deduped.jsonl")
        print(report.summary())
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    def dedupe_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> DeduplicationReport:
        """Deduplicate a JSONL file → write unique records to output_path."""
        report = DeduplicationReport(normalization=self.config.normalization)
        in_records = load_jsonl(input_path, text_field=self.config.text_field, skip_invalid=True)
        report.records_in = len(in_records)

        seen_hashes: Set[str] = set()
        unique_records: List[Dict[str, Any]] = []
        for rec in in_records:
            text = rec.get(self.config.text_field, "")
            normalized = self._normalize(text)
            h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                report.duplicates_removed += 1
                continue
            seen_hashes.add(h)
            unique_records.append(rec)

        report.records_out = len(unique_records)
        report.unique_hashes = len(seen_hashes)
        write_jsonl(unique_records, output_path)
        logger.info(
            "deduplicated %s → %s (records %d → %d, removed %d duplicates)",
            input_path, output_path,
            report.records_in, report.records_out, report.duplicates_removed,
        )
        return report

    # ------------------------------------------------------------------
    def dedupe_records(
        self,
        records: List[Dict[str, Any]],
    ) -> tuple:
        """Deduplicate in-memory records.  Returns (unique_records, report)."""
        report = DeduplicationReport(
            records_in=len(records),
            normalization=self.config.normalization,
        )
        seen_hashes: Set[str] = set()
        unique_records: List[Dict[str, Any]] = []
        for rec in records:
            text = rec.get(self.config.text_field, "")
            normalized = self._normalize(text)
            h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                report.duplicates_removed += 1
                continue
            seen_hashes.add(h)
            unique_records.append(rec)
        report.records_out = len(unique_records)
        report.unique_hashes = len(seen_hashes)
        return unique_records, report

    # ------------------------------------------------------------------
    def find_cross_dataset_duplicates(
        self,
        records_a: List[Dict[str, Any]],
        records_b: List[Dict[str, Any]],
    ) -> int:
        """Count records in B that also appear in A.

        Used for data-leakage detection (spec §17) — checking that the
        validation set doesn't contain documents from the training set.
        """
        hashes_a: Set[str] = set()
        for rec in records_a:
            text = rec.get(self.config.text_field, "")
            normalized = self._normalize(text)
            hashes_a.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
        n_overlap = 0
        for rec in records_b:
            text = rec.get(self.config.text_field, "")
            normalized = self._normalize(text)
            h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if h in hashes_a:
                n_overlap += 1
        return n_overlap

    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        if self.config.normalization == "none":
            return text
        if self.config.normalization == "whitespace":
            return " ".join(text.split())
        if self.config.normalization == "lowercase":
            return " ".join(text.lower().split())
        return text
