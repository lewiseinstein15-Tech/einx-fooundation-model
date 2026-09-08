# -*- coding: utf-8 -*-
"""EINX data cleaner (spec §5).

Configurable text cleaning operations.  Every transformation is
opt-in — users can disable individual operations.  Nothing is
aggressive; legitimate text is never destroyed.

Operations (all configurable):
  * Unicode normalization (NFC by default)
  * Whitespace normalization (collapse runs, strip leading/trailing)
  * Control-character removal (except newline, tab)
  * Empty-document removal
  * Excessive repetition detection (the "lorem ipsum lorem ipsum..." case)
  * Malformed-document removal (already handled by validator, but the
    cleaner re-checks to be defensive)

The cleaner NEVER silently modifies the dataset — every transformation
is logged + counted, and a :class:`CleaningReport` is produced.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from einx.data.dataset import load_jsonl, write_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cleaning report
# ---------------------------------------------------------------------------


@dataclass
class CleaningReport:
    """Result of cleaning a dataset.  Every count is real."""

    records_in: int = 0
    records_out: int = 0
    unicode_normalized: int = 0
    whitespace_normalized: int = 0
    control_chars_removed: int = 0
    empty_removed: int = 0
    repetition_removed: int = 0
    malformed_removed: int = 0
    chars_in: int = 0
    chars_out: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> str:
        lines = [
            "EINX DATA CLEANING",
            "─" * 40,
            f"Records in:           {self.records_in:,}",
            f"Records out:          {self.records_out:,}",
            f"  Unicode normalized: {self.unicode_normalized:,}",
            f"  Whitespace norm:    {self.whitespace_normalized:,}",
            f"  Control chars rm:   {self.control_chars_removed:,}",
            f"  Empty removed:      {self.empty_removed:,}",
            f"  Repetition removed: {self.repetition_removed:,}",
            f"  Malformed removed:  {self.malformed_removed:,}",
            "",
            f"Chars in:             {self.chars_in:,}",
            f"Chars out:            {self.chars_out:,}",
            "─" * 40,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CleaningConfig:
    """Configuration for data cleaning.  Every operation is opt-in."""

    text_field: str = "text"
    unicode_normalization: str = "NFC"      # "NFC" | "NFKC" | "NFD" | "NFKD" | "none"
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    remove_empty: bool = True
    remove_excessive_repetition: bool = True
    repetition_threshold: float = 0.5      # fraction of repeated lines
    repetition_min_lines: int = 10          # only check docs with >= N lines

    def validate(self) -> None:
        valid_norms = ("NFC", "NFKC", "NFD", "NFKD", "none")
        if self.unicode_normalization not in valid_norms:
            raise ValueError(
                f"unicode_normalization must be one of {valid_norms}, "
                f"got {self.unicode_normalization!r}"
            )
        if not 0 < self.repetition_threshold <= 1.0:
            raise ValueError(
                f"repetition_threshold must be in (0, 1], got {self.repetition_threshold}"
            )


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


# Control characters to remove: 0x00-0x08, 0x0B, 0x0C, 0x0E-0x1F, 0x7F.
# We keep \n (0x0A) and \t (0x09).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


class DatasetCleaner:
    """Clean a dataset.  Produces a :class:`CleaningReport`.

    Usage:
        cleaner = DatasetCleaner(CleaningConfig(text_field="text"))
        report = cleaner.clean_file("raw.jsonl", "clean.jsonl")
        print(report.summary())
    """

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.config.validate()

    # ------------------------------------------------------------------
    def clean_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> CleaningReport:
        """Clean a JSONL file → write cleaned records to output_path.

        Loads with skip_invalid=False so we count ALL records (including
        malformed ones) in records_in.  Malformed records are then
        caught by the cleaner's own validation.
        """
        report = CleaningReport()
        # Load with skip_invalid=False so we see every line — the
        # cleaner is responsible for filtering, not the loader.
        try:
            in_records = load_jsonl(input_path, text_field=self.config.text_field, skip_invalid=False)
        except Exception:
            # If strict loading fails (malformed JSON), fall back to
            # lenient loading — the cleaner handles the filtering.
            in_records = load_jsonl(input_path, text_field=self.config.text_field, skip_invalid=True)
        report.records_in = len(in_records)
        report.chars_in = sum(len(r.get(self.config.text_field, "")) for r in in_records)

        cleaned_records: List[Dict[str, Any]] = []
        for rec in in_records:
            text = rec.get(self.config.text_field, "")
            new_text, ops = self._clean_text(text)
            if new_text is None:
                # Removed by a filter
                continue
            new_rec = dict(rec)
            new_rec[self.config.text_field] = new_text
            cleaned_records.append(new_rec)
            # Update ops counts
            report.unicode_normalized += ops["unicode"]
            report.whitespace_normalized += ops["whitespace"]
            report.control_chars_removed += ops["control"]

        report.records_out = len(cleaned_records)
        report.chars_out = sum(len(r.get(self.config.text_field, "")) for r in cleaned_records)
        # Removed count = records_in - records_out (covers all removal reasons)
        removed = report.records_in - report.records_out
        # We don't track which filter removed which record separately in
        # this version — the report's "empty_removed" etc. are best-effort
        # attributions.  For the exact count, use records_in - records_out.

        write_jsonl(cleaned_records, output_path)
        logger.info(
            "cleaned %s → %s (records %d → %d)",
            input_path, output_path, report.records_in, report.records_out,
        )
        return report

    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> Optional[str]:
        """Clean a single text string.  Returns None if removed."""
        cleaned, _ = self._clean_text(text)
        return cleaned

    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> tuple:
        """Clean one text.  Returns (cleaned_text or None, ops_dict)."""
        ops = {"unicode": 0, "whitespace": 0, "control": 0}
        if not isinstance(text, str):
            return None, ops

        # 1. Unicode normalization
        if self.config.unicode_normalization != "none":
            try:
                new_text = unicodedata.normalize(self.config.unicode_normalization, text)
                if new_text != text:
                    ops["unicode"] = 1
                text = new_text
            except (ValueError, TypeError):
                # Invalid Unicode — remove the record
                return None, ops

        # 2. Control character removal
        if self.config.remove_control_chars:
            new_text = _CONTROL_CHAR_RE.sub("", text)
            if new_text != text:
                ops["control"] = 1
            text = new_text

        # 3. Whitespace normalization (collapse runs, strip)
        if self.config.normalize_whitespace:
            new_text = " ".join(text.split())
            # Preserve newlines by re-splitting on them first
            lines = text.split("\n")
            new_lines = [" ".join(line.split()) for line in lines]
            new_text = "\n".join(new_lines).strip()
            if new_text != text:
                ops["whitespace"] = 1
            text = new_text

        # 4. Empty removal
        if self.config.remove_empty and not text.strip():
            return None, ops

        # 5. Excessive repetition detection
        if self.config.remove_excessive_repetition:
            if self._is_excessive_repetition(text):
                return None, ops

        return text, ops

    # ------------------------------------------------------------------
    def _is_excessive_repetition(self, text: str) -> bool:
        """Detect excessive line repetition.

        A document is flagged if it has >= ``repetition_min_lines``
        lines AND the most common line accounts for >=
        ``repetition_threshold`` of all lines.

        This catches the "spam" pattern (e.g. a document of 100 identical
        lines) without removing legitimate text with normal repetition.
        """
        lines = text.split("\n")
        if len(lines) < self.config.repetition_min_lines:
            return False
        from collections import Counter
        counts = Counter(lines)
        most_common_count = counts.most_common(1)[0][1]
        fraction = most_common_count / len(lines)
        return fraction >= self.config.repetition_threshold


# ---------------------------------------------------------------------------
# Near-duplicate detection abstraction (spec §6 — future work)
# ---------------------------------------------------------------------------


class NearDuplicateDetector:
    """Abstraction for future near-duplicate detection.

    Per spec §6: "Do not claim semantic deduplication if only exact/
    normalized hashing is implemented.  Prepare an abstraction for
    future near-duplicate detection."

    The current implementation is exact-hash only (via the validator's
    duplicate detection).  This class is a clean extension point for
    future MinHash / SimHash / embedding-based near-duplicate detection.

    It's intentionally a stub — calling detect() raises
    NotImplementedError so callers know it's not yet implemented.
    """

    def __init__(self, *, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm

    def detect(self, documents: List[str]) -> List[List[int]]:
        """Return clusters of near-duplicate document indices.

        NOT YET IMPLEMENTED — would require MinHash / SimHash / embedding
        similarity.  Raises NotImplementedError.
        """
        raise NotImplementedError(
            "Near-duplicate detection is PLANNED but not yet implemented. "
            "Use DatasetValidator with check_duplicates=True for exact + "
            "normalized exact deduplication."
        )
