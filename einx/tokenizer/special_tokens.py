# -*- coding: utf-8 -*-
"""Special tokens for EINX tokenizers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    """Reserved special tokens.  IDs are fixed at the start of the vocab."""

    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"

    # IDs — always 0, 1, 2, 3 in that order so the model config can
    # hardcode them and the tokenizer never has to be loaded to know
    # which ID is the EOS token.
    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def all(self) -> list:
        return [
            (self.pad, self.pad_id),
            (self.bos, self.bos_id),
            (self.eos, self.eos_id),
            (self.unk, self.unk_id),
        ]

    def is_special(self, token: str) -> bool:
        return token in (self.pad, self.bos, self.eos, self.unk)
