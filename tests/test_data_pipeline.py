# -*- coding: utf-8 -*-
"""Tests for the EINX data cleaner + deduplicator + quality analyzer + manifest + shards + pipeline (spec §5-§14)."""

import json
import pytest
from pathlib import Path

from einx.data.cleaner import DatasetCleaner, CleaningConfig, CleaningReport, NearDuplicateDetector
from einx.data.deduplicator import DatasetDeduplicator, DeduplicationConfig
from einx.data.quality import QualityAnalyzer
from einx.data.manifest import DatasetManifest, create_manifest, hash_files, hash_config
from einx.data.shards import ShardWriter, ShardDataset, compute_token_stats, TokenStats
from einx.data.pipeline import DataPipeline, PipelineConfig, preflight_check
from einx.data.dataset import write_jsonl, load_jsonl, train_val_test_split


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------


def test_cleaner_removes_empty(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"text": "real content"}, {"text": ""}, {"text": "   "}])
    cleaner = DatasetCleaner()
    report = cleaner.clean_file(in_path, out_path)
    # load_jsonl skips empty records, so records_in reflects only the
    # loadable records.  The cleaner then filters the empty ones.
    assert report.records_out == 1
    cleaned = load_jsonl(out_path)
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "real content"


def test_cleaner_normalizes_whitespace(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"text": "  hello   world  "}])
    cleaner = DatasetCleaner()
    cleaner.clean_file(in_path, out_path)
    cleaned = load_jsonl(out_path)
    assert cleaned[0]["text"] == "hello world"


def test_cleaner_removes_control_chars(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"text": "hello\x00\x01world"}])
    cleaner = DatasetCleaner()
    cleaner.clean_file(in_path, out_path)
    cleaned = load_jsonl(out_path)
    assert "\x00" not in cleaned[0]["text"]
    assert "\x01" not in cleaned[0]["text"]
    assert "hello" in cleaned[0]["text"]
    assert "world" in cleaned[0]["text"]


def test_cleaner_unicode_normalization(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    # NFD form of café (e + combining accent) should become NFC (é)
    _write_jsonl(in_path, [{"text": "cafe\u0301"}])  # NFD form
    cleaner = DatasetCleaner(CleaningConfig(unicode_normalization="NFC"))
    cleaner.clean_file(in_path, out_path)
    cleaned = load_jsonl(out_path)
    assert cleaned[0]["text"] == "café"  # NFC form


def test_cleaner_removes_excessive_repetition(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    # A document of 50 identical lines — should be flagged + removed
    repetitive_text = "\n".join(["spam spam spam"] * 50)
    _write_jsonl(in_path, [{"text": repetitive_text}, {"text": "legitimate content"}])
    cleaner = DatasetCleaner(CleaningConfig(
        remove_excessive_repetition=True,
        repetition_threshold=0.5,
        repetition_min_lines=10,
    ))
    report = cleaner.clean_file(in_path, out_path)
    assert report.records_out == 1
    cleaned = load_jsonl(out_path)
    assert cleaned[0]["text"] == "legitimate content"


def test_cleaner_report_summary(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [{"text": "doc 1"}, {"text": ""}])
    cleaner = DatasetCleaner()
    report = cleaner.clean_file(in_path, out_path)
    summary = report.summary()
    assert "Records in:" in summary
    assert "Records out:" in summary


def test_near_duplicate_detector_not_implemented():
    """Per spec §6: near-duplicate detection is PLANNED, not implemented."""
    detector = NearDuplicateDetector()
    with pytest.raises(NotImplementedError, match="PLANNED"):
        detector.detect(["doc 1", "doc 2"])


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


def test_deduplicator_removes_exact_duplicates(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    _write_jsonl(in_path, [
        {"text": "hello world"},
        {"text": "hello world"},  # exact dup
        {"text": "different"},
    ])
    dedup = DatasetDeduplicator()
    report = dedup.dedupe_file(in_path, out_path)
    assert report.records_in == 3
    assert report.records_out == 2
    assert report.duplicates_removed == 1


def test_deduplicator_whitespace_normalization(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    with open(in_path, "w") as fh:
        fh.write(json.dumps({"text": "hello world"}) + "\n")
        fh.write(json.dumps({"text": "  hello   world  "}) + "\n")  # same after norm
    dedup = DatasetDeduplicator(DeduplicationConfig(normalization="whitespace"))
    report = dedup.dedupe_file(in_path, out_path)
    assert report.records_out == 1


def test_deduplicator_cross_dataset_overlap():
    """find_cross_dataset_duplicates detects train/val leakage (spec §17)."""
    dedup = DatasetDeduplicator()
    records_a = [{"text": "doc A"}, {"text": "doc B"}, {"text": "shared"}]
    records_b = [{"text": "doc C"}, {"text": "shared"}, {"text": "doc D"}]
    n_overlap = dedup.find_cross_dataset_duplicates(records_a, records_b)
    assert n_overlap == 1


def test_deduplicator_no_false_positives():
    """Different documents should not be flagged as duplicates."""
    dedup = DatasetDeduplicator()
    records = [{"text": "doc 1"}, {"text": "doc 2"}, {"text": "doc 3"}]
    unique, report = dedup.dedupe_records(records)
    assert len(unique) == 3
    assert report.duplicates_removed == 0


# ---------------------------------------------------------------------------
# Quality analyzer
# ---------------------------------------------------------------------------


def test_quality_analyzer_basic_stats(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [
        {"text": "short"},
        {"text": "a longer document with more characters"},
    ])
    analyzer = QualityAnalyzer()
    report = analyzer.analyze([path])
    assert report.documents == 2
    assert report.characters > 0
    assert report.avg_length > 0
    assert report.min_length < report.max_length


def test_quality_analyzer_detects_duplicates(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [
        {"text": "hello world"},
        {"text": "hello world"},  # dup
    ])
    analyzer = QualityAnalyzer()
    report = analyzer.analyze([path])
    assert report.duplicates == 1
    assert report.duplicate_rate == 0.5


def test_quality_analyzer_with_tokenizer(tmp_path):
    from einx.tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(20)]
    tok.train(texts, vocab_size=300, verbose=False)
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": t} for t in texts])
    analyzer = QualityAnalyzer()
    report = analyzer.analyze([path], tokenizer=tok)
    assert report.tokens > 0
    assert report.avg_tokens_per_doc > 0
    assert report.tokenizer_version == tok.VERSION


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_compute_identity(tmp_path):
    path1 = tmp_path / "data1.jsonl"
    path2 = tmp_path / "data2.jsonl"
    _write_jsonl(path1, [{"text": "doc"}])
    _write_jsonl(path2, [{"text": "different doc"}])
    m1 = create_manifest(
        name="ds", version="0.1", source_files=[path1],
        processing_config={"min_chars": 1}, tokenizer_path="tok.json",
        tokenizer_version="v1", n_records=1, n_tokens=10,
    )
    m2 = create_manifest(
        name="ds", version="0.1", source_files=[path2],  # different source
        processing_config={"min_chars": 1}, tokenizer_path="tok.json",
        tokenizer_version="v1", n_records=1, n_tokens=10,
    )
    assert m1.identity_hash != m2.identity_hash


def test_manifest_config_change_changes_identity(tmp_path):
    """If processing config changes, identity must change (spec §8)."""
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "doc"}])
    m1 = create_manifest(
        name="ds", version="0.1", source_files=[path],
        processing_config={"min_chars": 1}, tokenizer_path="tok.json",
        tokenizer_version="v1", n_records=1, n_tokens=10,
    )
    m2 = create_manifest(
        name="ds", version="0.1", source_files=[path],
        processing_config={"min_chars": 5},  # different config
        tokenizer_path="tok.json", tokenizer_version="v1",
        n_records=1, n_tokens=10,
    )
    assert m1.identity_hash != m2.identity_hash


def test_manifest_save_load_roundtrip(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "doc"}])
    manifest = create_manifest(
        name="test-ds", version="0.1.0", source_files=[path],
        processing_config={"seed": 42}, tokenizer_path="tok.json",
        tokenizer_version="v1", n_records=1, n_tokens=10,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    loaded = DatasetManifest.load(manifest_path)
    assert loaded.identity_hash == manifest.identity_hash
    assert loaded.name == "test-ds"


def test_hash_files_changes_with_content(tmp_path):
    p1 = tmp_path / "a.txt"
    p1.write_text("content A")
    h1 = hash_files([p1])
    p1.write_text("content B")
    h2 = hash_files([p1])
    assert h1 != h2


def test_hash_config_changes_with_values():
    h1 = hash_config({"a": 1, "b": 2})
    h2 = hash_config({"a": 1, "b": 3})
    assert h1 != h2
    # Same values, different order → same hash
    h3 = hash_config({"b": 2, "a": 1})
    assert h1 == h3


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def test_shard_writer_creates_shards(tmp_path):
    out_dir = tmp_path / "shards"
    writer = ShardWriter(out_dir, shard_size=3)
    for i in range(7):
        writer.write({"input_ids": [i, i+1, i+2], "idx": i})
    stats = writer.close()
    assert stats["shard_count"] == 3  # 3 + 3 + 1
    assert stats["total_records"] == 7
    assert stats["total_tokens"] == 21  # 7 records × 3 tokens
    shard_files = sorted(out_dir.glob("shard-*.jsonl"))
    assert len(shard_files) == 3


def test_shard_dataset_reads_shards(tmp_path):
    out_dir = tmp_path / "shards"
    writer = ShardWriter(out_dir, shard_size=100)
    for i in range(10):
        writer.write({"input_ids": list(range(i, i + 10))})
    writer.close()
    ds = ShardDataset(out_dir, context_length=10)
    assert len(ds) == 10
    input_ids, targets = ds[0]
    assert input_ids.shape == (10,)
    assert targets.shape == (10,)


def test_shard_dataset_n_tokens(tmp_path):
    out_dir = tmp_path / "shards"
    writer = ShardWriter(out_dir, shard_size=100)
    for i in range(5):
        writer.write({"input_ids": [1, 2, 3, 4, 5]})
    writer.close()
    # Write a manifest — use a file (not a dir) as source
    source_file = tmp_path / "source.jsonl"
    source_file.write_text("source")
    manifest = create_manifest(
        name="test", version="0.1", source_files=[source_file],
        processing_config={}, tokenizer_path="tok.json",
        tokenizer_version="v1", n_records=5, n_tokens=25,
        shard_files=[f.name for f in out_dir.glob("shard-*.jsonl")],
    )
    manifest.save(out_dir / "manifest.json")
    ds = ShardDataset(out_dir, context_length=10)
    assert ds.n_tokens() == 25


def test_compute_token_stats():
    from einx.tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(20)]
    tok.train(texts, vocab_size=300, verbose=False)
    stats = compute_token_stats(texts, tok)
    assert stats.documents == 20
    assert stats.total_tokens > 0
    assert stats.avg_tokens_per_doc > 0
    assert stats.min_tokens <= stats.max_tokens


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_end_to_end(tmp_path):
    """End-to-end: raw → validate → clean → dedupe → tokenize → shard → manifest."""
    from einx.tokenizer.bpe import BPETokenizer

    # 1. Create a small raw dataset with duplicates + malformed records
    raw_path = tmp_path / "raw.jsonl"
    with open(raw_path, "w") as fh:
        fh.write(json.dumps({"text": "the cat sat on the mat"}) + "\n")
        fh.write(json.dumps({"text": "the dog ran in the park"}) + "\n")
        fh.write(json.dumps({"text": "the cat sat on the mat"}) + "\n")  # duplicate
        fh.write(json.dumps({"text": ""}) + "\n")  # empty
        fh.write("not json\n")  # malformed
        fh.write(json.dumps({"text": "the bird flew over the tree"}) + "\n")

    # 2. Train a tokenizer
    tok = BPETokenizer()
    tok_texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
    ] * 10
    tok.train(tok_texts, vocab_size=300, verbose=False)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    # 3. Run the pipeline
    output_dir = tmp_path / "processed"
    pipeline = DataPipeline(PipelineConfig(
        name="test-dataset",
        version="0.1.0",
        input_paths=[raw_path],
        text_field="text",
        tokenizer_path=str(tok_path),
        context_length=32,
        shard_size=100,
        validation_ratio=0.5,  # high ratio to get val data
        seed=42,
        output_dir=str(output_dir),
    ))
    result = pipeline.run()

    # 4. Verify
    assert result.n_input_records == 6
    assert result.n_valid_records >= 3  # at least 3 valid non-empty
    assert result.n_unique_records < result.n_clean_records  # dedup removed some
    assert result.n_tokens > 0
    assert result.n_shards > 0
    assert result.manifest is not None
    assert result.manifest.identity_hash != ""

    # 5. Manifest exists + is loadable
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    loaded = DatasetManifest.load(manifest_path)
    assert loaded.identity_hash == result.manifest.identity_hash

    # 6. ShardDataset can read the processed data
    train_ds = ShardDataset(output_dir / "train", context_length=32)
    assert len(train_ds) > 0
    input_ids, targets = train_ds[0]
    assert input_ids.shape == (32,)


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------


def test_preflight_check_passes_when_compatible(tmp_path):
    from einx.tokenizer.bpe import BPETokenizer
    from einx.config import EINXModelConfig
    from einx.config import TrainingConfig
    from einx.data.pipeline import DataPipeline, PipelineConfig

    # Build a dataset
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, [{"text": f"document {i}"} for i in range(20)])
    tok = BPETokenizer()
    tok.train([f"document {i}" for i in range(20)], vocab_size=300, verbose=False)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    output_dir = tmp_path / "processed"
    pipeline = DataPipeline(PipelineConfig(
        input_paths=[raw_path], tokenizer_path=str(tok_path),
        context_length=32, output_dir=str(output_dir),
        validation_ratio=0.2, seed=42,
    ))
    pipeline.run()

    # Build matching model config
    model_cfg = EINXModelConfig(
        vocab_size=tok.vocab_size(), hidden_dim=32, n_layers=2,
        n_heads=2, head_dim=16, max_context_length=32,
        ffn_dim=64, dropout=0.0,
    )
    train_cfg = TrainingConfig(batch_size=2, max_steps=5, device="cpu")

    # Should pass without error
    preflight_check(
        dataset_dir=output_dir,
        tokenizer_path=tok_path,
        model_config=model_cfg,
        training_config=train_cfg,
    )


def test_preflight_check_fails_on_missing_manifest(tmp_path):
    from einx.config import EINXModelConfig, TrainingConfig
    from einx.utils.errors import EINXConfigError

    model_cfg = EINXModelConfig(vocab_size=100, hidden_dim=32, n_layers=1,
                                n_heads=2, head_dim=16, max_context_length=32,
                                ffn_dim=64)
    train_cfg = TrainingConfig(batch_size=2, max_steps=5)
    with pytest.raises(EINXConfigError, match="manifest"):
        preflight_check(
            dataset_dir=tmp_path / "nonexistent",
            tokenizer_path=tmp_path / "tok.json",
            model_config=model_cfg,
            training_config=train_cfg,
        )


def test_preflight_check_fails_on_vocab_mismatch(tmp_path):
    from einx.tokenizer.bpe import BPETokenizer
    from einx.config import EINXModelConfig, TrainingConfig
    from einx.data.pipeline import DataPipeline, PipelineConfig
    from einx.utils.errors import EINXConfigError

    # Build dataset with vocab=300
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, [{"text": f"document {i}"} for i in range(20)])
    tok = BPETokenizer()
    tok.train([f"document {i}" for i in range(20)], vocab_size=300, verbose=False)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)

    output_dir = tmp_path / "processed"
    pipeline = DataPipeline(PipelineConfig(
        input_paths=[raw_path], tokenizer_path=str(tok_path),
        context_length=32, output_dir=str(output_dir),
    ))
    pipeline.run()

    # Model config with WRONG vocab size
    model_cfg = EINXModelConfig(
        vocab_size=999,  # mismatch!
        hidden_dim=32, n_layers=2, n_heads=2, head_dim=16,
        max_context_length=32, ffn_dim=64,
    )
    train_cfg = TrainingConfig(batch_size=2, max_steps=5)
    with pytest.raises(EINXConfigError, match="vocab"):
        preflight_check(
            dataset_dir=output_dir, tokenizer_path=tok_path,
            model_config=model_cfg, training_config=train_cfg,
        )
