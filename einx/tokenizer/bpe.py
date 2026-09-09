# -*- coding: utf-8 -*-
"""Byte-level BPE tokenizer for EINX.

Implementation notes
---------------------
This is a real BPE tokenizer — not a wrapper around HuggingFace
``tokenizers``.  It is intentionally pure Python so:

  * there are no Rust / Cargo build dependencies
  * every step is inspectable and testable
  * training is deterministic given the same corpus + seed
  * the tokenizer serialises to plain JSON (vocab + merges)

Algorithm:
  1. Encode every line of the training corpus as UTF-8 bytes.
  2. Each byte is an initial "symbol" in the vocabulary (256 base symbols).
  3. Prepend a word-boundary marker (Ġ, following GPT-2 convention) to
     the first byte of each whitespace-separated word so the tokenizer
     can distinguish "the" at start-of-word vs mid-word.
  4. Iteratively merge the most frequent adjacent pair, ``vocab_size -
     256 - len(special)`` times.
  5. Encode = apply the learned merges greedily; decode = reverse.

The vocabulary contains (in this exact order):
  [0..3]      special tokens (<pad>, <bos>, <eos>, <unk>)
  [4..259]    the 256 base byte symbols
  [260..]     learned BPE merges
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from einx.tokenizer.special_tokens import SpecialTokens

logger = logging.getLogger(__name__)

# Word-boundary marker (single character; will be the first byte of each
# whitespace-delimited word so merges can learn word-initial vs word-internal
# tokens).  We use the same Ġ (U+0120) convention as GPT-2/Roberta so the
# emitted token strings look familiar to anyone who has debugged a GPT-2
# tokenizer.
WORD_BOUNDARY = "Ġ"

# Base byte vocab (0..255) is mapped to a Unicode character so the
# vocabulary entries are printable / debuggable.  This is the same
# byte-to-unicode mapping used by GPT-2's tokenizer.
def _bytes_to_unicode() -> Dict[int, str]:
    """Reversible map from byte (0..255) to a printable unicode char.

    Same scheme as GPT-2: printable ASCII + latin-1 pass through, the
    rest get mapped to U+0100..U+0241 so they're all printable.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}


class BPETokenizer:
    """Byte-level BPE tokenizer.

    Construct with ``BPETokenizer()`` then either:

    * ``train(corpus, vocab_size)`` to learn a new tokenizer, or
    * ``load(path)`` to load a previously trained one.

    The tokenizer is JSON-serialisable — ``save(path)`` writes vocab,
    merges, special tokens, and version metadata to a single file.
    """

    VERSION = "einx-bpe-0.1"

    def __init__(self, special: Optional[SpecialTokens] = None):
        self.special = special or SpecialTokens()
        self.vocab: List[str] = []              # id -> token string
        self.token_to_id: Dict[str, int] = {}    # token string -> id
        self.merges: List[Tuple[str, str]] = []  # ordered list of merges
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        # When trained, set; otherwise None
        self._trained: bool = False
        # Initialise with the special tokens only — vocab is otherwise empty
        # until ``train`` or ``load`` is called.
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.vocab = [tok for tok, _ in self.special.all]
        self.token_to_id = {tok: idx for tok, idx in self.special.all}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        corpus: Iterable[str],
        vocab_size: int = 4096,
        *,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train a byte-level BPE on ``corpus``.

        ``corpus`` is any iterable of strings (lines, documents, etc.).
        ``vocab_size`` is the target vocabulary size — must be >= 260
        (special tokens + 256 base bytes).

        Returns metadata about the training run.
        """
        if vocab_size < 260:
            raise ValueError(
                f"vocab_size must be >= 260 (4 special + 256 base bytes), got {vocab_size}"
            )

        # Reset state — train from scratch.
        self._init_special_tokens()
        self.merges = []
        self.merge_ranks = {}

        # 1. Add the 256 base byte tokens.
        for b in range(256):
            tok = BYTE_TO_UNICODE[b]
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.vocab)
                self.vocab.append(tok)

        # 2. Pre-tokenise: split on whitespace, prepend word-boundary
        # marker to the first byte of each word.  Count word frequencies
        # so we don't waste merges on duplicate words.
        word_freqs: Counter = Counter()
        for line in corpus:
            for word in line.split():
                # Encode word as UTF-8 bytes, then map each byte to its
                # printable unicode char.  Prepend the word-boundary marker
                # to the first character.
                encoded = line.encode("utf-8")  # noqa: F841 — kept for clarity
                # Use the actual word, not the line, for word freq
                word_bytes = word.encode("utf-8")
                chars = [BYTE_TO_UNICODE[b] for b in word_bytes]
                if not chars:
                    continue
                # Insert WORD_BOUNDARY as a SEPARATE token (not concatenated).
                # Same fix as in encode() — prevents "+" etc. from becoming
                # a single unknown "Ġ+" token.
                chars = [WORD_BOUNDARY] + chars
                word_key = " ".join(chars)
                word_freqs[word_key] += 1

        # 3. Convert each unique word into a tuple of symbols (initially
        # one symbol per character) and count adjacent pairs across the
        # whole corpus (weighted by word frequency).
        # State: word_key -> list of symbols
        words: Dict[str, List[str]] = {
            wk: wk.split(" ") for wk in word_freqs
        }

        target_merges = vocab_size - len(self.vocab)
        if target_merges <= 0:
            self._trained = True
            return {"vocab_size": len(self.vocab), "n_merges": 0}

        for step in range(target_merges):
            # Count adjacent pairs (weighted by word frequency)
            pair_counts: Counter = Counter()
            for wk, symbols in words.items():
                freq = word_freqs[wk]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break  # nothing left to merge

            # Pick the most frequent pair (ties broken by lex order for determinism)
            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], kv[0])
            )

            # Apply the merge across all words
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            self.token_to_id[new_token] = len(self.vocab)
            self.vocab.append(new_token)

            for wk, symbols in words.items():
                if len(symbols) < 2:
                    continue
                new_symbols: List[str] = []
                i = 0
                while i < len(symbols):
                    if (
                        i < len(symbols) - 1
                        and (symbols[i], symbols[i + 1]) == best_pair
                    ):
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                words[wk] = new_symbols

            if verbose and (step < 5 or (step + 1) % 100 == 0):
                logger.info(
                    "BPE step %d/%d: merged %r (count=%d), vocab=%d",
                    step + 1, target_merges, best_pair, best_count, len(self.vocab),
                )

        self._trained = True
        return {
            "vocab_size": len(self.vocab),
            "n_merges": len(self.merges),
            "n_unique_words": len(word_freqs),
            "version": self.VERSION,
        }

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def _bpe(self, tokens: List[str]) -> List[str]:
        """Apply BPE merges to a list of initial byte-tokens.

        Greedy: at each step, find the lowest-rank adjacent pair and
        merge it, until no merges apply.
        """
        if len(tokens) < 2:
            return tokens
        while True:
            # Find the pair with the lowest merge rank
            best_rank: Optional[int] = None
            best_idx: int = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            # Merge the chosen pair
            tokens = (
                tokens[:best_idx]
                + [tokens[best_idx] + tokens[best_idx + 1]]
                + tokens[best_idx + 2:]
            )
        return tokens

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to a list of token IDs.

        ``add_bos`` / ``add_eos`` prepend / append the corresponding
        special-token IDs.
        """
        if not self._trained:
            raise RuntimeError("tokenizer is not trained — call train() or load() first")

        ids: List[int] = []
        if add_bos:
            ids.append(self.special.bos_id)
        for word in text.split():
            word_bytes = word.encode("utf-8")
            chars = [BYTE_TO_UNICODE[b] for b in word_bytes]
            if not chars:
                continue
            # Insert WORD_BOUNDARY as a SEPARATE token (not concatenated).
            # This fixes the bug where "+" became "Ġ+" (a single unknown
            # token) instead of ["Ġ", "+"] (two known byte tokens).
            chars = [WORD_BOUNDARY] + chars
            # Apply BPE
            merged = self._bpe(chars)
            for tok in merged:
                tid = self.token_to_id.get(tok)
                if tid is None:
                    # Fallback: if the merged token isn't in the vocab,
                    # decompose it into individual byte characters and
                    # look each one up.  Every byte character IS in the
                    # vocab (they're the 256 base tokens), so this never
                    # returns UNK for a real character.
                    for ch in tok:
                        tid = self.token_to_id.get(ch, self.special.unk_id)
                        ids.append(tid)
                else:
                    ids.append(tid)
        if add_eos:
            ids.append(self.special.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Special tokens are skipped (their string forms like ``<bos>``
        are not part of the original text).
        """
        if not self._trained:
            raise RuntimeError("tokenizer is not trained")
        chars: List[str] = []
        for tid in ids:
            if tid < 4:  # special token — skip
                continue
            if tid >= len(self.vocab):
                continue
            chars.append(self.vocab[tid])
        # Join, then convert word-boundary markers back to spaces
        text = "".join(chars)
        text = text.replace(WORD_BOUNDARY, " ")
        # The first token of a sequence may have a leading space introduced
        # by the word-boundary marker on the first word.  Strip it so the
        # roundtrip encode(decode(x)) == x.
        if text.startswith(" "):
            text = text[1:]
        # Convert printable-unicode back to real bytes
        out_bytes = bytearray()
        for ch in text:
            if ch in UNICODE_TO_BYTE:
                out_bytes.append(UNICODE_TO_BYTE[ch])
            else:
                # Shouldn't happen if vocab is consistent, but degrade gracefully
                out_bytes.extend(ch.encode("utf-8"))
        return out_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    def id_to_token(self, token_id: int) -> Optional[str]:
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return None

    def token_to_id_method(self, token: str) -> Optional[int]:
        return self.token_to_id.get(token)

    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "vocab": self.vocab,
            "merges": [[a, b] for a, b in self.merges],
            "special_tokens": {
                "pad": self.special.pad,
                "bos": self.special.bos,
                "eos": self.special.eos,
                "unk": self.special.unk,
            },
            "special_ids": {
                "pad": self.special.pad_id,
                "bos": self.special.bos_id,
                "eos": self.special.eos_id,
                "unk": self.special.unk_id,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save the tokenizer to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        logger.info("tokenizer saved to %s (vocab=%d)", path, len(self.vocab))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        special = SpecialTokens(
            pad=data["special_tokens"].get("pad", "<pad>"),
            bos=data["special_tokens"].get("bos", "<bos>"),
            eos=data["special_tokens"].get("eos", "<eos>"),
            unk=data["special_tokens"].get("unk", "<unk>"),
        )
        tok = cls(special=special)
        tok.vocab = list(data["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_ranks = {m: i for i, m in enumerate(tok.merges)}
        tok._trained = True
        logger.info("tokenizer loaded from %s (vocab=%d, merges=%d)",
                    path, len(tok.vocab), len(tok.merges))
        return tok

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def encode_batch(self, texts: List[str], *, add_eos: bool = True) -> List[List[int]]:
        return [self.encode(t, add_eos=add_eos) for t in texts]

    def __len__(self) -> int:
        return self.vocab_size()

    def __repr__(self) -> str:
        return f"<BPETokenizer trained={self._trained} vocab_size={self.vocab_size()}>"
