# -*- coding: utf-8 -*-
"""Tests for the data pipeline."""

import json
import pytest
from einx.data.dataset import (
    TextDataset,
    TokenisedDataset,
    load_jsonl,
    write_jsonl,
    train_val_test_split,
)
from einx.data.synthetic import generate_synthetic_corpus


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl([{"text": "hello"}, {"text": "world"}], path)
    records = load_jsonl(path)
    assert len(records) == 2
    assert records[0]["text"] == "hello"


def test_load_jsonl_skips_invalid(tmp_path):
    path = tmp_path / "data.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"text": "valid"}) + "\n")
        fh.write("not json\n")
        fh.write(json.dumps({"no_text": "missing"}) + "\n")
        fh.write(json.dumps({"text": ""}) + "\n")  # empty text
        fh.write(json.dumps({"text": "valid2"}) + "\n")
    records = load_jsonl(path)
    assert len(records) == 2  # only valid records kept
    assert records[0]["text"] == "valid"
    assert records[1]["text"] == "valid2"


def test_load_jsonl_raises_when_skip_invalid_false(tmp_path):
    path = tmp_path / "data.jsonl"
    with open(path, "w") as fh:
        fh.write("not json\n")
    with pytest.raises(ValueError, match="invalid record"):
        load_jsonl(path, skip_invalid=False)


def test_load_jsonl_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nonexistent.jsonl")


def test_train_val_test_split_sizes():
    records = [{"text": str(i)} for i in range(100)]
    train, val, test = train_val_test_split(records, val_ratio=0.1, test_ratio=0.1)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_train_val_test_split_no_overlap():
    records = [{"text": str(i)} for i in range(100)]
    train, val, test = train_val_test_split(records, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_ids = {r["text"] for r in train}
    val_ids = {r["text"] for r in val}
    test_ids = {r["text"] for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_train_val_test_split_reproducible():
    records = [{"text": str(i)} for i in range(100)]
    train1, val1, test1 = train_val_test_split(records, seed=42)
    train2, val2, test2 = train_val_test_split(records, seed=42)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_train_val_test_split_invalid_ratios():
    records = [{"text": "x"}]
    with pytest.raises(ValueError):
        train_val_test_split(records, val_ratio=0.5, test_ratio=0.5)


def test_text_dataset_from_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl([{"text": "a"}, {"text": "b"}], path)
    ds = TextDataset.from_jsonl(path)
    assert len(ds) == 2
    assert ds[0] == "a"


def test_tokenised_dataset():
    import torch
    from einx.tokenizer.bpe import BPETokenizer
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(20)]
    tok.train(texts, vocab_size=400, verbose=False)
    ds = TokenisedDataset(texts, tok, context_length=32)
    assert len(ds) > 0
    input_ids, targets = ds[0]
    assert input_ids.shape == (32,)
    assert targets.shape == (32,)
    # Target should be input shifted by 1
    assert torch.equal(input_ids[1:], targets[:-1])


def test_synthetic_corpus_generation():
    records = generate_synthetic_corpus(n_records=10, seed=0)
    assert len(records) == 10
    assert all(isinstance(r, str) for r in records)
    assert all(r.strip() for r in records)  # non-empty
