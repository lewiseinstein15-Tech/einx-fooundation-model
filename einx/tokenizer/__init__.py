# -*- coding: utf-8 -*-
"""EINX tokenizer subsystem.

A real, pure-Python Byte-Pair Encoding (BPE) tokenizer.  No Rust
dependency, fully inspectable, deterministic given the same training
corpus + seed.

Capabilities:
  * train a BPE tokenizer from a text corpus
  * encode text -> token IDs
  * decode token IDs -> text
  * save / load as JSON (vocab + merges + special tokens)
  * inspect vocabulary
  * special tokens: <pad>, <bos>, <eos>, <unk>

The implementation is byte-level (operates on UTF-8 bytes) so the
tokenizer handles any Unicode text without an "unknown character"
problem — every byte is in the base vocabulary.
"""

from einx.tokenizer.bpe import BPETokenizer
from einx.tokenizer.special_tokens import SpecialTokens

__all__ = ["BPETokenizer", "SpecialTokens"]
