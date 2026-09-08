# -*- coding: utf-8 -*-
"""Tests for the EINX data validator (spec §4)."""

import json
import pytest
from pathlib import Path

from einx.data.validator import (
    DatasetValidator,
    ValidationConfig,
    ValidationReport,
    validate_dataset,
)


def _write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_validation_report_defaults():
    r = ValidationReport()
    assert r.records_scanned == 0
    assert r.valid == 0
    assert r.final_usable == 0


def test_validation_report_summary_contains_real_numbers(tmp_path):
    """The summary must contain numbers that come from the actual data."""
    path = tmp_path / "data.jsonl"
    records = [
        {"text": "hello world"},
        {"text": "second document"},
        {"text": "hello world"},  # duplicate
        {"text": ""},  # empty
        {"no_text": "missing"},  # missing field
        "not json at all",  # malformed
    ]
    _write_jsonl(path, [r if isinstance(r, dict) else {"_raw": r} for r in records])
    # Overwrite with a mix
    with open(path, "w") as fh:
        fh.write(json.dumps({"text": "hello world"}) + "\n")
        fh.write(json.dumps({"text": "second document"}) + "\n")
        fh.write(json.dumps({"text": "hello world"}) + "\n")  # dup
        fh.write(json.dumps({"text": ""}) + "\n")  # empty
        fh.write(json.dumps({"no_text": "missing"}) + "\n")  # missing
        fh.write("not json\n")  # malformed

    validator = DatasetValidator(ValidationConfig(text_field="text", check_duplicates=True))
    report = validator.validate([path])
    summary = report.summary()

    assert "Files scanned:    1" in summary
    assert "Records scanned:  6" in summary
    assert report.records_scanned == 6
    assert report.valid == 2  # "hello world" (first occurrence) + "second document"
    assert report.duplicates == 1  # the duplicate "hello world"
    assert report.empty == 1
    assert report.missing_text_field == 1
    assert report.malformed_json == 1
    assert report.final_usable == report.valid - report.duplicates


def test_validate_dataset_helper(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "doc 1"}, {"text": "doc 2"}])
    report = validate_dataset([path])
    assert report.valid == 2
    assert report.duplicates == 0


def test_validator_handles_missing_file(tmp_path):
    validator = DatasetValidator()
    report = validator.validate([tmp_path / "nonexistent.jsonl"])
    assert report.files_scanned == 1
    assert report.records_scanned == 0


def test_validator_min_chars_filter(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [
        {"text": "ab"},  # too short (min_chars=5)
        {"text": "hello world"},  # OK
    ])
    validator = DatasetValidator(ValidationConfig(min_chars=5))
    report = validator.validate([path])
    assert report.valid == 1
    assert report.too_short == 1


def test_validator_max_chars_filter(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [
        {"text": "short"},  # OK
        {"text": "x" * 200},  # too long (max_chars=100)
    ])
    validator = DatasetValidator(ValidationConfig(max_chars=100))
    report = validator.validate([path])
    assert report.valid == 1
    assert report.too_long == 1


def test_validator_duplicate_normalization(tmp_path):
    """Whitespace-normalized dedup catches 'hello world' == 'hello   world'."""
    path = tmp_path / "data.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"text": "hello world"}) + "\n")
        fh.write(json.dumps({"text": "hello   world"}) + "\n")  # same after whitespace norm
    validator = DatasetValidator(ValidationConfig(
        duplicate_normalization="whitespace",
    ))
    report = validator.validate([path])
    assert report.valid == 1
    assert report.duplicates == 1


def test_validator_lowercase_normalization(tmp_path):
    path = tmp_path / "data.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"text": "Hello World"}) + "\n")
        fh.write(json.dumps({"text": "hello world"}) + "\n")  # same after lowercase
    validator = DatasetValidator(ValidationConfig(
        duplicate_normalization="lowercase",
    ))
    report = validator.validate([path])
    assert report.valid == 1
    assert report.duplicates == 1


def test_validator_config_validation():
    with pytest.raises(ValueError):
        ValidationConfig(min_chars=-1).validate()
    with pytest.raises(ValueError):
        ValidationConfig(duplicate_normalization="invalid").validate()


def test_validator_report_to_json(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "doc 1"}])
    report = validate_dataset([path])
    j = report.to_json()
    parsed = json.loads(j)
    assert parsed["valid"] == 1


def test_validator_writes_report_to_file(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"text": "doc 1"}, {"text": "doc 2"}])
    report_path = tmp_path / "report.json"
    validate_dataset([path], report_path=report_path)
    assert report_path.exists()
    with open(report_path) as fh:
        data = json.load(fh)
    assert data["valid"] == 2
