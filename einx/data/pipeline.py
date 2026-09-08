# -*- coding: utf-8 -*-
"""EINX unified data pipeline (spec §2, §10, §26).

Orchestrates the full pipeline:
    RAW DATA → validate → clean → dedupe → split → tokenize → pack → shard → manifest

Each stage has a clear interface and can be run independently or as
part of the full pipeline via :func:`build_dataset`.

Per spec §2: "Each stage must have a clear interface."
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from einx.data.cleaner import CleaningConfig, DatasetCleaner
from einx.data.deduplicator import DeduplicationConfig, DatasetDeduplicator
from einx.data.manifest import create_manifest, DatasetManifest, hash_files
from einx.data.quality import QualityAnalyzer
from einx.data.shards import ShardWriter, TokenStats, compute_token_stats
from einx.data.validator import ValidationConfig, DatasetValidator, ValidationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the full data pipeline."""

    name: str = "einx-dataset"
    version: str = "0.1.0"

    # Input
    input_paths: List[str] = field(default_factory=list)
    text_field: str = "text"

    # Validation
    min_chars: int = 1
    max_chars: int = 1_000_000

    # Cleaning
    unicode_normalization: str = "NFC"
    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    remove_empty: bool = True
    remove_excessive_repetition: bool = True

    # Deduplication
    dedup_normalization: str = "whitespace"

    # Splitting
    validation_ratio: float = 0.05
    test_ratio: float = 0.0
    seed: int = 42

    # Tokenization
    tokenizer_path: str = ""

    # Packing + sharding
    context_length: int = 256
    shard_size: int = 10000

    # Output
    output_dir: str = "data/processed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def processing_config_dict(self) -> Dict[str, Any]:
        """The processing config used for the manifest hash.

        Includes every parameter that affects the output — if any of
        these change, the dataset identity changes (spec §8).
        """
        return {
            "text_field": self.text_field,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "unicode_normalization": self.unicode_normalization,
            "normalize_whitespace": self.normalize_whitespace,
            "remove_control_chars": self.remove_control_chars,
            "remove_empty": self.remove_empty,
            "remove_excessive_repetition": self.remove_excessive_repetition,
            "dedup_normalization": self.dedup_normalization,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "context_length": self.context_length,
            "shard_size": self.shard_size,
        }


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Result of running the full data pipeline."""

    manifest: Optional[DatasetManifest] = None
    validation_report: Optional[ValidationReport] = None
    n_input_records: int = 0
    n_valid_records: int = 0
    n_clean_records: int = 0
    n_unique_records: int = 0
    n_train_records: int = 0
    n_val_records: int = 0
    n_tokens: int = 0
    n_shards: int = 0
    output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_input_records": self.n_input_records,
            "n_valid_records": self.n_valid_records,
            "n_clean_records": self.n_clean_records,
            "n_unique_records": self.n_unique_records,
            "n_train_records": self.n_train_records,
            "n_val_records": self.n_val_records,
            "n_tokens": self.n_tokens,
            "n_shards": self.n_shards,
            "output_dir": self.output_dir,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class DataPipeline:
    """The full data pipeline.  Orchestrates all stages.

    Usage:
        pipeline = DataPipeline(PipelineConfig(
            input_paths=["data/raw/train.jsonl"],
            tokenizer_path="tokenizer.json",
            output_dir="data/processed",
        ))
        result = pipeline.run()
        print(f"Built dataset with {result.n_train_records} train records, "
              f"{result.n_tokens} tokens")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        """Run the full pipeline.  Returns a :class:`PipelineResult`."""
        result = PipelineResult(output_dir=self.config.output_dir)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Stage 1: VALIDATION ----------------------------------
        logger.info("Stage 1: VALIDATION")
        validator = DatasetValidator(ValidationConfig(
            text_field=self.config.text_field,
            min_chars=self.config.min_chars,
            max_chars=self.config.max_chars,
            check_duplicates=False,  # dedupe is a separate stage
            duplicate_normalization=self.config.dedup_normalization,
        ))
        report = validator.validate(self.config.input_paths)
        result.validation_report = report
        result.n_input_records = report.records_scanned
        result.n_valid_records = report.valid
        print(report.summary())

        # ---- Stage 2: CLEANING ------------------------------------
        logger.info("Stage 2: CLEANING")
        cleaner = DatasetCleaner(CleaningConfig(
            text_field=self.config.text_field,
            unicode_normalization=self.config.unicode_normalization,
            normalize_whitespace=self.config.normalize_whitespace,
            remove_control_chars=self.config.remove_control_chars,
            remove_empty=self.config.remove_empty,
            remove_excessive_repetition=self.config.remove_excessive_repetition,
        ))
        clean_path = output_dir / "clean.jsonl"
        # Concatenate all input files through the cleaner
        all_clean_records = []
        for input_path in self.config.input_paths:
            tmp_out = output_dir / f"clean_{Path(input_path).stem}.jsonl"
            cleaner.clean_file(input_path, tmp_out)
            from einx.data.dataset import load_jsonl
            all_clean_records.extend(load_jsonl(tmp_out, text_field=self.config.text_field))
        result.n_clean_records = len(all_clean_records)

        # ---- Stage 3: DEDUPLICATION ------------------------------
        logger.info("Stage 3: DEDUPLICATION")
        dedup = DatasetDeduplicator(DeduplicationConfig(
            text_field=self.config.text_field,
            normalization=self.config.dedup_normalization,
        ))
        unique_records, dedup_report = dedup.dedupe_records(all_clean_records)
        result.n_unique_records = len(unique_records)
        print(dedup_report.summary())

        # ---- Stage 4: TRAIN/VAL SPLIT -----------------------------
        logger.info("Stage 4: TRAIN/VAL SPLIT")
        from einx.data.dataset import train_val_test_split, write_jsonl
        train_records, val_records, test_records = train_val_test_split(
            unique_records,
            val_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
        )
        result.n_train_records = len(train_records)
        result.n_val_records = len(val_records)

        # ---- Stage 5: TOKENIZATION --------------------------------
        logger.info("Stage 5: TOKENIZATION")
        from einx.tokenizer.bpe import BPETokenizer
        tokenizer = BPETokenizer.load(self.config.tokenizer_path)

        # ---- Stage 6: PACKING + SHARDING --------------------------
        logger.info("Stage 6: PACKING + SHARDING")
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        train_stats = self._tokenize_and_shard(
            train_records, tokenizer, train_dir,
            context_length=self.config.context_length,
            shard_size=self.config.shard_size,
        )
        val_stats = self._tokenize_and_shard(
            val_records, tokenizer, val_dir,
            context_length=self.config.context_length,
            shard_size=self.config.shard_size,
        )
        result.n_tokens = train_stats["total_tokens"] + val_stats["total_tokens"]
        result.n_shards = train_stats["shard_count"] + val_stats["shard_count"]

        # ---- Stage 7: MANIFEST ------------------------------------
        logger.info("Stage 7: MANIFEST")
        processing_config = self.config.processing_config_dict()
        manifest = create_manifest(
            name=self.config.name,
            version=self.config.version,
            source_files=self.config.input_paths,
            processing_config=processing_config,
            tokenizer_path=self.config.tokenizer_path,
            tokenizer_version=tokenizer.VERSION,
            n_records=result.n_unique_records,
            n_tokens=result.n_tokens,
            n_train_records=result.n_train_records,
            n_val_records=result.n_val_records,
            shard_files=train_stats["shard_files"] + val_stats["shard_files"],
            seed=self.config.seed,
            context_length=self.config.context_length,
        )
        manifest.save(output_dir / "manifest.json")
        result.manifest = manifest

        print(f"\nPipeline complete:")
        print(f"  Input records:    {result.n_input_records:,}")
        print(f"  Valid records:    {result.n_valid_records:,}")
        print(f"  Clean records:    {result.n_clean_records:,}")
        print(f"  Unique records:   {result.n_unique_records:,}")
        print(f"  Train records:    {result.n_train_records:,}")
        print(f"  Val records:      {result.n_val_records:,}")
        print(f"  Total tokens:    {result.n_tokens:,}")
        print(f"  Shards:           {result.n_shards}")
        print(f"  Identity hash:   {manifest.identity_hash[:16]}")

        return result

    # ------------------------------------------------------------------
    def _tokenize_and_shard(
        self,
        records: List[Dict[str, Any]],
        tokenizer,
        output_dir: Path,
        *,
        context_length: int,
        shard_size: int,
    ) -> Dict[str, Any]:
        """Tokenize + pack + shard a list of records."""
        writer = ShardWriter(output_dir, shard_size=shard_size)
        for rec in records:
            text = rec.get(self.config.text_field, "")
            ids = tokenizer.encode(text, add_eos=True)
            # Truncate to context_length + 1 (so target = input[1:])
            ids = ids[: context_length + 1]
            writer.write({"input_ids": ids, "text_preview": text[:50]})
        return writer.close()


# ---------------------------------------------------------------------------
# Training-data preflight validator (spec §19)
# ---------------------------------------------------------------------------


def preflight_check(
    *,
    dataset_dir: Union[str, Path],
    tokenizer_path: Union[str, Path],
    model_config,
    training_config,
) -> None:
    """Validate dataset + tokenizer + model + config compatibility BEFORE training.

    Per spec §19:
        validate dataset → validate tokenizer → validate model context length →
        validate dataset/tokenizer compatibility → validate configuration →
        start training

    Raises a clear error on any mismatch.
    """
    from einx.utils.errors import EINXConfigError

    dataset_dir = Path(dataset_dir)
    tokenizer_path = Path(tokenizer_path)

    # 1. Dataset exists + has manifest
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise EINXConfigError(
            f"Dataset manifest not found at {manifest_path}. "
            "Run `einx data build` first to process the dataset.",
            field="dataset_dir",
            value=str(dataset_dir),
            hint="Run: einx data build --input <raw> --output <processed> --tokenizer <tok>",
        )

    # 2. Tokenizer exists
    if not tokenizer_path.exists():
        raise EINXConfigError(
            f"Tokenizer not found at {tokenizer_path}",
            field="tokenizer_path",
            value=str(tokenizer_path),
            hint="Run: einx tokenizer train --corpus <data> --output <path>",
        )

    # 3. Model context length is sane
    if model_config.max_context_length <= 0:
        raise EINXConfigError(
            "Model max_context_length must be positive",
            field="max_context_length",
            value=model_config.max_context_length,
        )

    # 4. Dataset context length matches model context length
    manifest = DatasetManifest.load(manifest_path)
    if manifest.context_length > 0 and manifest.context_length != model_config.max_context_length:
        raise EINXConfigError(
            "Dataset context length doesn't match model context length",
            field="context_length",
            value=f"dataset={manifest.context_length}, model={model_config.max_context_length}",
            hint="Either re-process the dataset with the right context_length, "
                 "or change the model config to match.",
        )

    # 5. Tokenizer version matches manifest
    from einx.tokenizer.bpe import BPETokenizer
    tokenizer = BPETokenizer.load(str(tokenizer_path))
    if tokenizer.VERSION != manifest.tokenizer_version:
        raise EINXConfigError(
            "Tokenizer version mismatch — the dataset was tokenized with a "
            "different tokenizer version.",
            field="tokenizer_version",
            value=f"tokenizer={tokenizer.VERSION}, manifest={manifest.tokenizer_version}",
            hint="Either re-tokenize the dataset with this tokenizer, "
                 "or use the tokenizer referenced in the manifest.",
        )

    # 6. Vocab size matches
    if tokenizer.vocab_size() != model_config.vocab_size:
        raise EINXConfigError(
            "Tokenizer vocab size doesn't match model vocab size",
            field="vocab_size",
            value=f"tokenizer={tokenizer.vocab_size()}, model={model_config.vocab_size}",
            hint="Either retrain the tokenizer with the right vocab size, "
                 "or change the model config's vocab_size to match.",
        )

    logger.info("preflight check passed: dataset + tokenizer + model are compatible")
