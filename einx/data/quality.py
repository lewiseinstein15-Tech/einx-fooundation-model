# -*- coding: utf-8 -*-
"""EINX data quality analyzer (spec §7).

Computes real statistics about a dataset:
  * document count
  * character count
  * token count (optional — requires a tokenizer)
  * average document length
  * length distribution (percentiles)
  * empty documents
  * duplicate rate (via hash)
  * malformed-record rate

Produces machine-readable JSON reports.  Every number comes from
actually scanning the dataset — no estimation.

Per spec §7: "Do NOT claim language detection unless actually
implemented."  Language detection is NOT implemented.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from einx.data.dataset import load_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    """Statistics about a dataset.  Every number is real."""

    # Counts
    documents: int = 0
    characters: int = 0
    tokens: int = 0  # 0 if no tokenizer provided

    # Length stats (in characters)
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    median_length: float = 0.0
    p25_length: float = 0.0
    p75_length: float = 0.0
    p95_length: float = 0.0
    p99_length: float = 0.0

    # Quality metrics
    empty_documents: int = 0
    duplicates: int = 0
    duplicate_rate: float = 0.0
    malformed_records: int = 0
    malformed_rate: float = 0.0

    # Token stats (if tokenizer provided)
    avg_tokens_per_doc: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0

    # Source info
    files_scanned: int = 0
    tokenizer_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def summary(self) -> str:
        lines = [
            "EINX DATA QUALITY REPORT",
            "─" * 40,
            f"Files scanned:    {self.files_scanned}",
            f"Documents:        {self.documents:,}",
            f"Characters:       {self.characters:,}",
        ]
        if self.tokens > 0:
            lines.append(f"Tokens:           {self.tokens:,}")
        lines.extend([
            "",
            "Length distribution (chars):",
            f"  Average:  {self.avg_length:.1f}",
            f"  Min:      {self.min_length}",
            f"  Max:      {self.max_length}",
            f"  Median:   {self.median_length:.1f}",
            f"  P25:      {self.p25_length:.1f}",
            f"  P75:      {self.p75_length:.1f}",
            f"  P95:      {self.p95_length:.1f}",
            f"  P99:      {self.p99_length:.1f}",
        ])
        if self.tokens > 0:
            lines.extend([
                "",
                "Token stats:",
                f"  Average tokens/doc: {self.avg_tokens_per_doc:.1f}",
                f"  Min tokens:         {self.min_tokens}",
                f"  Max tokens:         {self.max_tokens}",
            ])
        lines.extend([
            "",
            "Quality metrics:",
            f"  Empty documents:    {self.empty_documents}",
            f"  Duplicates:         {self.duplicates}",
            f"  Duplicate rate:     {self.duplicate_rate:.2%}",
            f"  Malformed records:  {self.malformed_records}",
            f"  Malformed rate:     {self.malformed_rate:.2%}",
            "─" * 40,
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class QualityAnalyzer:
    """Analyze a dataset's quality.  Produces a :class:`QualityReport`.

    Usage:
        analyzer = QualityAnalyzer(text_field="text")
        report = analyzer.analyze(["data/processed/train.jsonl"])
        print(report.summary())

        # With tokenizer (adds token stats):
        from einx.tokenizer.bpe import BPETokenizer
        tok = BPETokenizer.load("tokenizer.json")
        report = analyzer.analyze(["data/processed/train.jsonl"], tokenizer=tok)
    """

    def __init__(self, text_field: str = "text"):
        self.text_field = text_field

    # ------------------------------------------------------------------
    def analyze(
        self,
        paths: Union[str, Path, List[Union[str, Path]]],
        *,
        tokenizer=None,
    ) -> QualityReport:
        """Analyze one or more dataset files."""
        if isinstance(paths, (str, Path)):
            paths = [paths]

        report = QualityReport()
        report.files_scanned = len(paths)
        if tokenizer is not None:
            report.tokenizer_version = getattr(tokenizer, "VERSION", "unknown")

        lengths: List[int] = []
        token_counts: List[int] = []
        seen_hashes: set = set()
        n_records_total = 0
        n_malformed = 0

        for path in paths:
            path = Path(path)
            if not path.exists():
                logger.warning("file not found: %s", path)
                continue

            with open(path, "r", encoding="utf-8") as fh:
                for line_no, raw_line in enumerate(fh, start=1):
                    line = raw_line.strip()
                    if not line:
                        report.empty_documents += 1
                        continue
                    n_records_total += 1
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            n_malformed += 1
                            continue
                        text = rec.get(self.text_field)
                        if not isinstance(text, str):
                            n_malformed += 1
                            continue
                        if not text.strip():
                            report.empty_documents += 1
                            continue
                    except json.JSONDecodeError:
                        n_malformed += 1
                        continue

                    # Length stats
                    n_chars = len(text)
                    lengths.append(n_chars)
                    report.characters += n_chars
                    report.documents += 1

                    # Token stats (if tokenizer)
                    if tokenizer is not None:
                        try:
                            ids = tokenizer.encode(text)
                            token_counts.append(len(ids))
                            report.tokens += len(ids)
                        except Exception:
                            pass

                    # Duplicate detection
                    h = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
                    if h in seen_hashes:
                        report.duplicates += 1
                    seen_hashes.add(h)

        # Compute statistics
        if lengths:
            report.avg_length = sum(lengths) / len(lengths)
            report.min_length = min(lengths)
            report.max_length = max(lengths)
            report.median_length = statistics.median(lengths)
            sorted_l = sorted(lengths)
            n = len(sorted_l)
            report.p25_length = sorted_l[int(n * 0.25)]
            report.p75_length = sorted_l[int(n * 0.75)]
            report.p95_length = sorted_l[int(n * 0.95)]
            report.p99_length = sorted_l[int(n * 0.99)] if n >= 100 else sorted_l[-1]

        if token_counts:
            report.avg_tokens_per_doc = sum(token_counts) / len(token_counts)
            report.min_tokens = min(token_counts)
            report.max_tokens = max(token_counts)

        report.malformed_records = n_malformed
        report.malformed_rate = (n_malformed / n_records_total) if n_records_total else 0.0
        report.duplicate_rate = (report.duplicates / report.documents) if report.documents else 0.0

        return report
