# -*- coding: utf-8 -*-
"""EINX data pipeline.

Real dataset loading, validation, cleaning, deduplication, tokenisation,
packing, sharding, and manifest management.  No magic — every number
in every report comes from actually scanning the data, never fabricated.

Pipeline (spec §2):
    RAW DATA → validate → clean → dedupe → split → tokenize → pack →
    shard → manifest → training dataloader → EINX
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
from einx.data.validator import (
    DatasetValidator,
    ValidationConfig,
    ValidationReport,
    validate_dataset,
)
from einx.data.cleaner import (
    DatasetCleaner,
    CleaningConfig,
    CleaningReport,
    NearDuplicateDetector,
)
from einx.data.deduplicator import (
    DatasetDeduplicator,
    DeduplicationConfig,
    DeduplicationReport,
)
from einx.data.quality import (
    QualityAnalyzer,
    QualityReport,
)
from einx.data.manifest import (
    DatasetManifest,
    create_manifest,
    hash_files,
    hash_config,
)
from einx.data.shards import (
    ShardWriter,
    ShardDataset,
    TokenStats,
    compute_token_stats,
)
from einx.data.pipeline import (
    DataPipeline,
    PipelineConfig,
    PipelineResult,
    preflight_check,
)

__all__ = [
    # Original Build 2 classes
    "TextDataset",
    "TokenisedDataset",
    "PackedDataset",
    "StreamingTextDataset",
    "load_jsonl",
    "write_jsonl",
    "train_val_test_split",
    "generate_synthetic_corpus",
    # Build 3 — validation
    "DatasetValidator",
    "ValidationConfig",
    "ValidationReport",
    "validate_dataset",
    # Build 3 — cleaning
    "DatasetCleaner",
    "CleaningConfig",
    "CleaningReport",
    "NearDuplicateDetector",
    # Build 3 — deduplication
    "DatasetDeduplicator",
    "DeduplicationConfig",
    "DeduplicationReport",
    # Build 3 — quality
    "QualityAnalyzer",
    "QualityReport",
    # Build 3 — manifest
    "DatasetManifest",
    "create_manifest",
    "hash_files",
    "hash_config",
    # Build 3 — sharding
    "ShardWriter",
    "ShardDataset",
    "TokenStats",
    "compute_token_stats",
    # Build 3 — pipeline
    "DataPipeline",
    "PipelineConfig",
    "PipelineResult",
    "preflight_check",
]
