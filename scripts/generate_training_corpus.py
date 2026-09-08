# -*- coding: utf-8 -*-
"""Generate a substantial training corpus for EINX pretraining.

Instead of just templated sentences, this creates varied text with:
- Narrative paragraphs (stories with structure)
- Question-answer pairs
- Definitions
- Number sequences
- Repeated phrases (for BPE merge learning)
- Varied sentence lengths
- A controlled vocabulary the model can actually learn

Target: 10,000+ documents, ~500K+ characters of varied text.
"""
import random
import json
from pathlib import Path

SUBJECTS = ["the cat", "the dog", "the bird", "the fish", "the robot", "the wizard",
            "the knight", "the merchant", "the scholar", "the farmer", "the sailor",
            "the healer", "the baker", "the king", "the queen", "the child"]
VERBS = ["saw", "found", "built", "chased", "lost", "discovered", "created",
         "destroyed", "explored", "guarded", "studied", "protected", "followed",
         "befriended", "defeated", "remembered"]
OBJECTS = ["a star", "a river", "a castle", "a forest", "a mountain", "a book",
           "a sword", "a garden", "a ship", "a tower", "a map", "a crystal",
           "a bridge", "a candle", "a storm", "a path"]
ADJECTIVES = ["ancient", "bright", "mysterious", "golden", "silent", "hidden",
              "distant", "forgotten", "sacred", "endless", "broken", "shining",
              "dark", "gentle", "fierce", "wise"]
PLACES = ["in the valley", "on the hill", "by the sea", "in the city",
          "across the desert", "through the forest", "under the mountain",
          "near the river", "beyond the wall", "in the garden"]
TIMES = ["at dawn", "at noon", "at dusk", "at midnight", "in the morning",
         "in the evening", "before winter", "after the storm", "during the feast"]

def _story(rng):
    """Generate a short narrative paragraph."""
    n_sentences = rng.randint(3, 6)
    sentences = []
    for _ in range(n_sentences):
        s = rng.choice(SUBJECTS)
        v = rng.choice(VERBS)
        o = rng.choice(OBJECTS)
        a = rng.choice(ADJECTIVES)
        p = rng.choice(PLACES)
        t = rng.choice(TIMES)
        # Randomly pick a sentence pattern
        pattern = rng.randint(0, 4)
        if pattern == 0:
            sentences.append(f"{t}, {s} {v} {a} {o} {p}.")
        elif pattern == 1:
            sentences.append(f"{s.capitalize()} {v} {o} {p} {t}.")
        elif pattern == 2:
            sentences.append(f"the {a} {o.split()[-1]} was {v} by {s} {p}.")
        elif pattern == 3:
            sentences.append(f"{s.capitalize()} remembered {a} {o} from long ago {p}.")
        else:
            sentences.append(f"{t} {s} decided to {v[:-1]} the {a} {o.split()[-1]} {p}.")
    return " ".join(sentences)

def _definition(rng):
    """Generate a definition-style document."""
    subject = rng.choice(["the cat", "the dog", "the bird", "the robot",
                          "the wizard", "the knight", "the river", "the mountain"])
    adj = rng.choice(ADJECTIVES)
    verb = rng.choice(VERBS)
    obj = rng.choice(OBJECTS)
    definitions = [
        f"{subject.capitalize()} is known for being {adj}.",
        f"the {adj} {subject} can {verb} {obj}.",
        f"in stories, {subject} represents {adj} qualities.",
        f"a {subject} that {verb}s {obj} is considered {adj}.",
    ]
    return " ".join(rng.choices(definitions, k=rng.randint(2, 4)))

def _counting(rng):
    """Generate a counting sequence (model can learn number patterns)."""
    start = rng.randint(1, 20)
    step = rng.choice([1, 2, 5, 10])
    n = rng.randint(5, 12)
    nums = list(range(start, start + step * n, step))
    return "count: " + ", ".join(str(n) for n in nums) + "."

def _qa(rng):
    """Generate a question-answer pair."""
    s = rng.choice(SUBJECTS)
    v = rng.choice(VERBS)
    o = rng.choice(OBJECTS)
    a = rng.choice(ADJECTIVES)
    q = f"what did {s} {v}?"
    ans = f"{s} {v} {a} {o}."
    return f"{q} {ans}"

def _dialogue(rng):
    """Generate a simple dialogue."""
    s1 = rng.choice(SUBJECTS)
    s2 = rng.choice([x for x in SUBJECTS if x != s1])
    o = rng.choice(OBJECTS)
    a = rng.choice(ADJECTIVES)
    v = rng.choice(VERBS)
    lines = [
        f'"{s2.capitalize()}, did you see {a} {o}?" asked {s1}.',
        f'"yes, i {v} it {rng.choice(PLACES)}," replied {s2}.',
        f'"we should {v[:-1]} it again," said {s1}.',
    ]
    return " ".join(rng.choices(lines, k=rng.randint(2, 3)))

def generate_training_corpus(n_records=10000, seed=42):
    """Generate a diverse training corpus."""
    rng = random.Random(seed)
    records = []
    mix = [("story", 0.35), ("definition", 0.15), ("counting", 0.1),
           ("qa", 0.15), ("dialogue", 0.15), ("story", 0.1)]
    for _ in range(n_records):
        r = rng.random()
        cumulative = 0
        for kind, weight in mix:
            cumulative += weight
            if r < cumulative:
                if kind == "story":
                    records.append({"text": _story(rng)})
                elif kind == "definition":
                    records.append({"text": _definition(rng)})
                elif kind == "counting":
                    records.append({"text": _counting(rng)})
                elif kind == "qa":
                    records.append({"text": _qa(rng)})
                elif kind == "dialogue":
                    records.append({"text": _dialogue(rng)})
                break
    return records

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    records = generate_training_corpus(n)
    output = Path("data/raw/training_corpus.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    total_chars = sum(len(r["text"]) for r in records)
    print(f"Generated {len(records)} records, {total_chars:,} characters")
    print(f"Saved to {output}")
