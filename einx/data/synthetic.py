# -*- coding: utf-8 -*-
"""Synthetic corpus generator — for first-run experiments + tests.

Generates a tiny but real text corpus so the EINX v0.1 milestone can
be exercised end-to-end without downloading a copyrighted dataset.

The generated corpus is a mix of:
  * short templated sentences (subject-verb-object)
  * number sequences (so the model can learn simple counting)
  * repeated phrases (so BPE has something to merge)

It's deliberately NOT a substitute for real pretraining data — it
exists so the training loop, checkpoint save/load, and generation
can all be verified to work.
"""

from __future__ import annotations

import random
from typing import List

# Deliberately tiny vocabulary so the synthetic model trains fast.
SUBJECTS = ["the cat", "the dog", "the bird", "the fish", "the robot", "the wizard"]
VERBS = ["saw", "ate", "chased", "found", "built", "lost"]
OBJECTS = ["a star", "a tree", "a book", "a machine", "a number", "a stone"]
ADJECTIVES = ["bright", "small", "ancient", "fast", "calm", "shining"]


def _sentence(rng: random.Random) -> str:
    parts = [
        rng.choice(SUBJECTS),
        rng.choice(VERBS),
        rng.choice(ADJECTIVES),
        rng.choice(OBJECTS) + ".",
    ]
    return " ".join(parts)


def _counting(rng: random.Random) -> str:
    """A simple counting sequence — model can learn the pattern."""
    start = rng.randint(1, 10)
    n = rng.randint(3, 8)
    nums = list(range(start, start + n))
    return "count: " + ", ".join(str(n) for n in nums) + "."


def _repeated_phrase(rng: random.Random) -> str:
    phrase = rng.choice([
        "the model learns",
        "data is power",
        "small steps forward",
        "the foundation is strong",
        "begin with the basics",
    ])
    return " ".join([phrase] * rng.randint(2, 4))


def generate_synthetic_corpus(
    n_records: int = 500,
    *,
    seed: int = 42,
    mix: tuple = (0.5, 0.25, 0.25),
) -> List[str]:
    """Generate ``n_records`` synthetic text records.

    ``mix`` controls the proportion of (sentences, counting, repeated
    phrases) in the corpus.  Defaults: 50% sentences, 25% counting,
    25% repeated.
    """
    rng = random.Random(seed)
    records: List[str] = []
    for _ in range(n_records):
        r = rng.random()
        if r < mix[0]:
            records.append(_sentence(rng))
        elif r < mix[0] + mix[1]:
            records.append(_counting(rng))
        else:
            records.append(_repeated_phrase(rng))
    return records


def write_synthetic_corpus(
    path: str,
    n_records: int = 500,
    *,
    seed: int = 42,
) -> int:
    """Write a synthetic corpus to a JSONL file.  Returns the record count."""
    import json
    from pathlib import Path

    records = generate_synthetic_corpus(n_records, seed=seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for text in records:
            fh.write(json.dumps({"text": text, "source": "synthetic"}) + "\n")
    return len(records)
