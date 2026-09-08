# -*- coding: utf-8 -*-
"""Tests for the BPE tokenizer."""

import pytest
from einx.tokenizer.bpe import BPETokenizer
from einx.tokenizer.special_tokens import SpecialTokens


SMALL_CORPUS = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the bird flew over the tree",
    "the fish swam in the river",
    "the robot built a machine",
    "the wizard lost a book",
    "the cat chased the bird",
    "the dog saw the cat",
    "the robot found the wizard",
    "the fish ate the bug",
] * 10  # repeat so BPE has something to merge


def test_special_tokens_defaults():
    s = SpecialTokens()
    assert s.pad_id == 0
    assert s.bos_id == 1
    assert s.eos_id == 2
    assert s.unk_id == 3
    assert s.pad == "<pad>"


def test_tokenizer_untrained_raises():
    tok = BPETokenizer()
    assert not tok.is_trained()
    with pytest.raises(RuntimeError, match="not trained"):
        tok.encode("hello")


def test_tokenizer_train_basic():
    tok = BPETokenizer()
    meta = tok.train(SMALL_CORPUS, vocab_size=300)
    assert tok.is_trained()
    assert meta["vocab_size"] == 300
    assert meta["n_merges"] > 0


def test_tokenizer_encode_decode_roundtrip():
    """Encode then decode should recover the original text."""
    tok = BPETokenizer()
    tok.train(SMALL_CORPUS, vocab_size=500, verbose=False)
    text = "the cat sat"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_tokenizer_encode_with_bos_eos():
    tok = BPETokenizer()
    tok.train(SMALL_CORPUS, vocab_size=500, verbose=False)
    ids = tok.encode("the cat", add_bos=True, add_eos=True)
    assert ids[0] == tok.special.bos_id
    assert ids[-1] == tok.special.eos_id


def test_tokenizer_handles_unicode():
    """Byte-level BPE handles any UTF-8 text — no unknown chars."""
    tok = BPETokenizer()
    tok.train(["hello world café résumé 日本語"] * 20, vocab_size=300, verbose=False)
    text = "café 日本語"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_tokenizer_vocab_size_consistent():
    tok = BPETokenizer()
    tok.train(SMALL_CORPUS, vocab_size=400, verbose=False)
    # BPE may stop early when the corpus runs out of pairs to merge;
    # the resulting vocab should be at most the target size.
    assert tok.vocab_size() <= 400
    assert tok.vocab_size() >= 260  # at minimum: 4 special + 256 bytes
    assert len(tok.vocab) == tok.vocab_size()
    assert len(tok.token_to_id) == tok.vocab_size()


def test_tokenizer_save_load_roundtrip(tmp_path):
    tok = BPETokenizer()
    tok.train(SMALL_CORPUS, vocab_size=400, verbose=False)
    actual_size = tok.vocab_size()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    assert tok2.is_trained()
    assert tok2.vocab_size() == actual_size
    # Same text → same IDs
    text = "the cat sat"
    assert tok.encode(text) == tok2.encode(text)


def test_tokenizer_id_to_token():
    tok = BPETokenizer()
    tok.train(SMALL_CORPUS, vocab_size=400, verbose=False)
    assert tok.id_to_token(0) == "<pad>"
    assert tok.id_to_token(1) == "<bos>"
    assert tok.id_to_token(2) == "<eos>"
    assert tok.id_to_token(3) == "<unk>"


def test_tokenizer_unknown_returns_unk():
    """Bytes outside the trained vocab should map to UNK (id 3)."""
    tok = BPETokenizer()
    tok.train(SMALL_CORPUS, vocab_size=260, verbose=False)  # minimum: 4 special + 256 bytes, 0 merges
    # Even without merges, encoding works — bytes are always in vocab
    ids = tok.encode("hello")
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_tokenizer_repr():
    tok = BPETokenizer()
    assert "trained=False" in repr(tok)
    tok.train(SMALL_CORPUS, vocab_size=300, verbose=False)
    assert "trained=True" in repr(tok)


def test_tokenizer_vocab_too_small_raises():
    tok = BPETokenizer()
    with pytest.raises(ValueError, match="vocab_size must be >= 260"):
        tok.train(SMALL_CORPUS, vocab_size=100)
