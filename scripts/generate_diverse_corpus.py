# -*- coding: utf-8 -*-
"""Generate a much larger, more diverse training corpus for EINX.

Build 2 corpus: 10K templated docs, 1.8M chars
This corpus:    50K diverse docs, 10M+ chars with:
  - Multi-paragraph stories with character arcs
  - Factual descriptions (animals, places, objects)
  - Conversations with turn-taking
  - Arithmetic and logic
  - Poetry/rhyme
  - Instruction-style text
  - Varied sentence structure + vocabulary
"""
import random
import json
from pathlib import Path

# Expanded vocabulary
SUBJECTS = ["the cat", "the dog", "the bird", "the fish", "the robot", "the wizard",
            "the knight", "the merchant", "the scholar", "the farmer", "the sailor",
            "the healer", "the baker", "the king", "the queen", "the child",
            "the dragon", "the ghost", "the giant", "the fairy", "the pirate",
            "the inventor", "the artist", "the musician", "the explorer", "the warrior",
            "the priest", "the thief", "the judge", "the hunter"]
VERBS = ["saw", "found", "built", "chased", "lost", "discovered", "created",
         "destroyed", "explored", "guarded", "studied", "protected", "followed",
         "befriended", "defeated", "remembered", "forgot", "loved", "hated",
         "rebuilt", "opened", "closed", "broke", "fixed", "hid", "revealed",
         "stole", "returned", "borrowed", "shared", "kept"]
OBJECTS = ["a star", "a river", "a castle", "a forest", "a mountain", "a book",
           "a sword", "a garden", "a ship", "a tower", "a map", "a crystal",
           "a bridge", "a candle", "a storm", "a path", "a door", "a key",
           "a mirror", "a crown", "a shield", "a lantern", "a bell", "a feather",
           "a stone", "a flame", "a shadow", "a song", "a dream", "a promise"]
ADJECTIVES = ["ancient", "bright", "mysterious", "golden", "silent", "hidden",
              "distant", "forgotten", "sacred", "endless", "broken", "shining",
              "dark", "gentle", "fierce", "wise", "cold", "warm", "heavy",
              "light", "fragile", "strong", "old", "new", "deep", "shallow",
              "wild", "calm", "restless", "sacred"]
PLACES = ["in the valley", "on the hill", "by the sea", "in the city",
          "across the desert", "through the forest", "under the mountain",
          "near the river", "beyond the wall", "in the garden", "on the cliff",
          "in the cave", "by the lake", "on the bridge", "in the temple",
          "at the market", "in the library", "on the ship", "in the tower",
          "beneath the stars"]
TIMES = ["at dawn", "at noon", "at dusk", "at midnight", "in the morning",
         "in the evening", "before winter", "after the storm", "during the feast",
         "as the sun set", "when the moon rose", "before the harvest",
         "after the battle", "during the long night", "at the turn of the year"]
EMOTIONS = ["with joy", "with sorrow", "with fear", "with hope", "with anger",
            "with wonder", "with relief", "with dread", "with gratitude", "with pride",
            "with surprise", "with determination"]
CONJUNCTIONS = ["and", "but", "so", "yet", "for", "then", "because", "although"]
NUMBERS_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight",
                 "nine", "ten", "eleven", "twelve", "twenty", "fifty", "hundred"]
COLORS = ["red", "blue", "green", "gold", "silver", "black", "white", "purple",
          "orange", "yellow", "crimson", "azure", "emerald", "amber"]
ANIMALS = ["cat", "dog", "bird", "fish", "horse", "wolf", "bear", "eagle",
           "snake", "rabbit", "deer", "fox", "owl", "lion", "tiger", "dragon"]
SEASONS = ["spring", "summer", "autumn", "winter"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _story(rng):
    """Multi-paragraph narrative with arc."""
    n_paras = rng.randint(1, 3)
    paras = []
    for _ in range(n_paras):
        n_sents = rng.randint(3, 6)
        sents = []
        for _ in range(n_sents):
            s = rng.choice(SUBJECTS)
            v = rng.choice(VERBS)
            o = rng.choice(OBJECTS)
            a = rng.choice(ADJECTIVES)
            p = rng.choice(PLACES)
            t = rng.choice(TIMES)
            e = rng.choice(EMOTIONS)
            conj = rng.choice(CONJUNCTIONS)
            pattern = rng.randint(0, 7)
            if pattern == 0:
                sents.append(f"{t}, {s} {v} {a} {o} {p}.")
            elif pattern == 1:
                sents.append(f"{s.capitalize()} {v} {o} {p} {t} {e}.")
            elif pattern == 2:
                sents.append(f"the {a} {o.split()[-1]} was {v} by {s} {p}, {conj} it was never forgotten.")
            elif pattern == 3:
                sents.append(f"{s.capitalize()} remembered {a} {o} from long ago {p} {e}.")
            elif pattern == 4:
                sents.append(f"{t} {s} decided to {v[:-1]} the {a} {o.split()[-1]} {p}.")
            elif pattern == 5:
                sents.append(f"no {s} had ever {v} {a} {o} before, {conj} the task seemed impossible.")
            elif pattern == 6:
                sents.append(f"with great effort, {s} {v} {o} {p} {e}, {conj} the {a} result surprised everyone.")
            else:
                sents.append(f"{conj} {s} knew that {v} {o} would change everything {p}.")
        paras.append(" ".join(sents))
    return "\n".join(paras)


def _description(rng):
    """Factual description of an animal/place/object."""
    animal = rng.choice(ANIMALS)
    adj = rng.choice(ADJECTIVES)
    color = rng.choice(COLORS)
    place = rng.choice(PLACES)
    size = rng.choice(["small", "large", "tiny", "massive", "medium"])
    facts = [
        f"the {animal} is a {size} creature known for its {adj} nature.",
        f"it has {color} fur and lives {place}.",
        f"the {color} {animal} can be found {place} during {rng.choice(SEASONS)}.",
        f"a {size} {animal} weighs about {rng.randint(1, 50)} pounds.",
        f"the {animal} has been a symbol of {adj} power for centuries.",
        f"in {rng.choice(SEASONS)}, the {animal} migrates {place}.",
    ]
    n = rng.randint(2, 4)
    return " ".join(rng.sample(facts, min(n, len(facts))))


def _conversation(rng):
    """Multi-turn dialogue."""
    s1 = rng.choice(SUBJECTS)
    s2 = rng.choice([x for x in SUBJECTS if x != s1])
    o = rng.choice(OBJECTS)
    a = rng.choice(ADJECTIVES)
    v = rng.choice(VERBS)
    p = rng.choice(PLACES)
    turns = [
        f'"{s2.capitalize()}, did you see {a} {o}?" asked {s1}.',
        f'"yes, i {v} it {p}," replied {s2}.',
        f'"we should {v[:-1]} it again," said {s1}.',
        f'"perhaps {rng.choice(TIMES)}," answered {s2}.',
        f'"the {a} {o.split()[-1]} belongs to no one," warned {s1}.',
        f'"then we must share it," agreed {s2}.',
    ]
    n = rng.randint(2, 4)
    return " ".join(rng.sample(turns, min(n, len(turns))))


def _arithmetic(rng):
    """Simple arithmetic."""
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)
    op = rng.choice(["+", "-", "x"])
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    else:
        result = a * b
    return f"{a} {op} {b} = {result}."


def _counting(rng):
    """Counting sequences with different patterns."""
    start = rng.randint(1, 30)
    step = rng.choice([1, 2, 3, 5, 10])
    n = rng.randint(5, 12)
    nums = list(range(start, start + step * n, step))
    return "count: " + ", ".join(str(n) for n in nums) + "."


def _poetry(rng):
    """Simple rhyming verse."""
    adj1 = rng.choice(ADJECTIVES)
    adj2 = rng.choice(ADJECTIVES)
    o1 = rng.choice(OBJECTS).split()[-1]
    o2 = rng.choice(OBJECTS).split()[-1]
    s1 = rng.choice(SUBJECTS)
    s2 = rng.choice(SUBJECTS)
    lines = [
        f"the {adj1} {o1} calls to {s1},",
        f"the {adj2} {o2} answers {s2}.",
        f"in {rng.choice(SEASONS)} they shall meet,",
        f"and the world will be complete.",
    ]
    return "\n".join(lines)


def _definition(rng):
    """Dictionary-style definitions."""
    word = rng.choice(ANIMALS + [o.split()[-1] for o in OBJECTS] + [v for v in VERBS])
    adj = rng.choice(ADJECTIVES)
    definitions = [
        f"a {word} is defined as something that is {adj}.",
        f"the word {word} means {adj} and powerful.",
        f"in the old tongue, {word} translates to '{adj} one'.",
        f"scholars describe {word} as both {adj} and essential.",
    ]
    return " ".join(rng.choices(definitions, k=rng.randint(1, 3)))


def _instruction(rng):
    """Step-by-step instructions."""
    task = rng.choice([
        "build a tower", "find a river", "train a bird",
        "plant a garden", "light a fire", "read a map",
        "cross a bridge", "open a door", "climb a mountain",
        "sail a ship"
    ])
    n_steps = rng.randint(3, 5)
    steps = []
    for i in range(1, n_steps + 1):
        verb = rng.choice(VERBS)
        obj = rng.choice(OBJECTS)
        adj = rng.choice(ADJECTIVES)
        steps.append(f"step {i}: {verb} the {adj} {obj.split()[-1]}.")
    return f"how to {task}: " + " ".join(steps)


def _calendar(rng):
    """Days/seasons/numbers — helps model learn temporal patterns."""
    day = rng.choice(DAYS)
    season = rng.choice(SEASONS)
    n = rng.choice(NUMBERS_WORDS)
    return f"on {day}, in {season}, after {n} days, the journey began."


def generate_diverse_corpus(n_records=50000, seed=42):
    """Generate a diverse training corpus."""
    rng = random.Random(seed)
    records = []
    mix = [
        ("story", 0.20),
        ("description", 0.15),
        ("conversation", 0.10),
        ("arithmetic", 0.10),
        ("counting", 0.05),
        ("poetry", 0.10),
        ("definition", 0.10),
        ("instruction", 0.10),
        ("calendar", 0.05),
        ("story", 0.05),
    ]
    for _ in range(n_records):
        r = rng.random()
        cumulative = 0
        for kind, weight in mix:
            cumulative += weight
            if r < cumulative:
                if kind == "story":
                    records.append({"text": _story(rng)})
                elif kind == "description":
                    records.append({"text": _description(rng)})
                elif kind == "conversation":
                    records.append({"text": _conversation(rng)})
                elif kind == "arithmetic":
                    records.append({"text": _arithmetic(rng)})
                elif kind == "counting":
                    records.append({"text": _counting(rng)})
                elif kind == "poetry":
                    records.append({"text": _poetry(rng)})
                elif kind == "definition":
                    records.append({"text": _definition(rng)})
                elif kind == "instruction":
                    records.append({"text": _instruction(rng)})
                elif kind == "calendar":
                    records.append({"text": _calendar(rng)})
                break
    return records


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    records = generate_diverse_corpus(n)
    output = Path("data/raw/training_corpus_v2.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    total_chars = sum(len(r["text"]) for r in records)
    avg_len = total_chars / len(records)
    print(f"Generated {len(records):,} records")
    print(f"Total characters: {total_chars:,}")
    print(f"Average doc length: {avg_len:.1f} chars")
    print(f"Saved to {output}")
