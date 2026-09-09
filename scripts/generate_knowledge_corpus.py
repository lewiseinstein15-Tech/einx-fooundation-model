# -*- coding: utf-8 -*-
"""Generate a REAL KNOWLEDGE corpus for EINX.

Previous corpus: templated stories (the cat saw the mat) — the model
learned grammar but NOT knowledge.

This corpus: REAL facts, REAL reasoning, REAL math, REAL logic.
The model will be tested on whether it actually KNOWS things.

Categories:
  1. Math facts (2+2=4, multiplication tables, etc.)
  2. World facts (capital of France is Paris, water boils at 100C)
  3. Logic puzzles (if A>B and B>C then A>C)
  4. Definitions (what is a mammal, what is gravity)
  5. Sequences (what comes next: 2,4,6,8,?)
  6. Cause and effect (rain makes things wet)
  7. Comparisons (a whale is bigger than a mouse)
  8. Q&A pairs with real answers
  9. Reasoning chains (step-by-step problem solving)
  10. Commonsense (fire is hot, ice is cold)
"""
import json
import random
from pathlib import Path

def _math_facts(rng):
    """Real arithmetic facts."""
    a = rng.randint(1, 20)
    b = rng.randint(1, 20)
    op = rng.choice(["+", "-", "x"])
    if op == "+":
        result = a + b
        return f"{a} + {b} = {result}."
    elif op == "-":
        a, b = max(a, b), min(a, b)  # avoid negatives
        return f"{a} - {b} = {a - b}."
    else:
        a, b = rng.randint(1, 12), rng.randint(1, 12)
        return f"{a} x {b} = {a * b}."

def _world_facts(rng):
    """Real world knowledge."""
    facts = [
        "the capital of france is paris.",
        "the capital of japan is tokyo.",
        "the capital of england is london.",
        "the capital of germany is berlin.",
        "the capital of italy is rome.",
        "the capital of china is beijing.",
        "the capital of russia is moscow.",
        "the capital of india is new delhi.",
        "the capital of brazil is brasilia.",
        "the capital of egypt is cairo.",
        "the capital of kenya is nairobi.",
        "the capital of canada is ottawa.",
        "the capital of australia is canberra.",
        "the capital of spain is madrid.",
        "the capital of greece is athens.",
        "water boils at 100 degrees celsius.",
        "water freezes at 0 degrees celsius.",
        "the earth orbits the sun.",
        "the moon orbits the earth.",
        "the sun is a star.",
        "gravity pulls objects toward the earth.",
        "sound travels at 343 meters per second in air.",
        "light travels at 299792458 meters per second.",
        "a year has 365 days.",
        "a week has 7 days.",
        "a day has 24 hours.",
        "an hour has 60 minutes.",
        "a minute has 60 seconds.",
        "there are 12 months in a year.",
        "january is the first month.",
        "december is the last month.",
        "humans have 206 bones in their body.",
        "the human heart has 4 chambers.",
        "blood is red because of hemoglobin.",
        "plants make food through photosynthesis.",
        "photosynthesis converts sunlight into energy.",
        "the largest planet in the solar system is jupiter.",
        "the smallest planet in the solar system is mercury.",
        "mars is called the red planet.",
        "venus is the hottest planet.",
        "the pacific ocean is the largest ocean.",
        "mount everest is the tallest mountain.",
        "the nile is the longest river in africa.",
        "the amazon is the longest river in south america.",
        "a triangle has 3 sides.",
        "a square has 4 equal sides.",
        "a circle has no corners.",
        "a pentagon has 5 sides.",
        "a hexagon has 6 sides.",
        "an octagon has 8 sides.",
    ]
    return rng.choice(facts)

def _logic_chains(rng):
    """Real logical reasoning."""
    chains = [
        "if a is bigger than b, and b is bigger than c, then a is bigger than c.",
        "if all cats are animals, and tom is a cat, then tom is an animal.",
        "if all birds can fly, and a penguin is a bird, then a penguin can fly. this is false because penguins cannot fly.",
        "if it is raining, the ground gets wet. the ground is wet, so it may have rained.",
        "if x is greater than 5, and 5 is greater than 3, then x is greater than 3.",
        "if all squares have 4 sides, and this shape has 3 sides, then it is not a square.",
        "if a number is even, it is divisible by 2. 8 is even, so 8 is divisible by 2.",
        "if a number is odd, it is not divisible by 2. 7 is odd, so 7 is not divisible by 2.",
        "if today is monday, tomorrow is tuesday.",
        "if today is friday, tomorrow is saturday.",
        "if today is sunday, yesterday was saturday.",
        "if all mammals have hair, and a whale is a mammal, then whales have hair.",
        "if a shape has 3 sides, it is a triangle.",
        "if a shape has 4 equal sides, it is a square.",
        "if a + b = 10 and a = 3, then b = 7.",
        "if a + b = 20 and b = 8, then a = 12.",
        "if 2x = 10, then x = 5.",
        "if 3x = 15, then x = 5.",
        "if x + 5 = 12, then x = 7.",
        "if all fish live in water, and a salmon is a fish, then salmon live in water.",
        "if fire is hot, and ice is cold, then fire and ice have different temperatures.",
        "if the sun rises in the east, and sets in the west, then the sun moves from east to west.",
    ]
    return rng.choice(chains)

def _definitions(rng):
    """Real definitions."""
    defs = [
        "a mammal is an animal that has hair and feeds its young with milk.",
        "a reptile is an animal with scales that lays eggs.",
        "a bird is an animal with feathers and a beak.",
        "a fish is an animal that lives in water and has gills.",
        "an insect is an animal with 6 legs and 3 body parts.",
        "gravity is the force that pulls objects toward each other.",
        "friction is the force that opposes motion when two surfaces touch.",
        "energy is the ability to do work.",
        "a molecule is two or more atoms joined together.",
        "an atom is the smallest unit of matter.",
        "a cell is the basic unit of life.",
        "dna is the molecule that carries genetic information.",
        "a planet is a large object that orbits a star.",
        "a star is a ball of hot gas that produces light.",
        "a galaxy is a group of stars, gas, and dust.",
        "a continent is a large landmass on earth.",
        "an island is land surrounded by water.",
        "a peninsula is land surrounded by water on 3 sides.",
        "a mountain is a tall natural elevation.",
        "a valley is a low area between hills or mountains.",
        "a river is a large stream of water that flows to the sea.",
        "a lake is a large body of water surrounded by land.",
        "an ocean is the largest body of salt water.",
        "a desert is a dry area with very little rain.",
        "a forest is an area with many trees.",
        "temperature measures how hot or cold something is.",
        "mass is the amount of matter in an object.",
        "volume is the amount of space something takes up.",
        "density is mass divided by volume.",
        "speed is distance divided by time.",
    ]
    return rng.choice(defs)

def _sequences(rng):
    """Pattern completion — what comes next."""
    start = rng.randint(1, 10)
    step = rng.choice([1, 2, 3, 5])
    n = rng.randint(4, 8)
    nums = list(range(start, start + step * n, step))
    text = ", ".join(str(n) for n in nums) + ", " + str(nums[-1] + step) + "."
    return f"what comes next: {text}"

def _cause_effect(rng):
    """Real cause and effect."""
    pairs = [
        "if you heat ice, it melts into water.",
        "if you heat water, it boils and becomes steam.",
        "if you cool water, it freezes into ice.",
        "if you drop something, gravity pulls it down.",
        "if you plant a seed and water it, it grows.",
        "if you do not sleep, you become tired.",
        "if you eat too much, you feel full.",
        "if you exercise, your muscles get stronger.",
        "if you study, you learn.",
        "if you do not drink water, you become thirsty.",
        "if the sun shines, things become warm.",
        "if wind blows, leaves move.",
        "if you cut an onion, your eyes may water.",
        "if you mix red and blue paint, you get purple.",
        "if you mix yellow and blue paint, you get green.",
        "if you mix red and yellow paint, you get orange.",
        "if you mix all colors of paint, you get brown.",
        "if you mix all colors of light, you get white.",
        "if you leave bread out, it becomes stale.",
        "if you leave metal in water, it may rust.",
    ]
    return rng.choice(pairs)

def _comparisons(rng):
    """Real comparisons."""
    comps = [
        "a whale is bigger than a mouse.",
        "an elephant is heavier than a cat.",
        "a cheetah is faster than a turtle.",
        "a skyscraper is taller than a house.",
        "the ocean is deeper than a pool.",
        "the sun is hotter than fire.",
        "a mountain is taller than a hill.",
        "a galaxy is bigger than a solar system.",
        "an atom is smaller than a cell.",
        "a year is longer than a month.",
        "a century is longer than a decade.",
        "a kilometer is longer than a meter.",
        "a ton is heavier than a kilogram.",
        "a liter is more than a milliliter.",
        "iron is heavier than wood.",
        "gold is more valuable than copper.",
        "diamond is harder than glass.",
        "steel is stronger than plastic.",
        "a tree is taller than grass.",
        "a river is wider than a stream.",
    ]
    return rng.choice(comps)

def _qa_pairs(rng):
    """Real question-answer pairs with actual knowledge."""
    qa = [
        "question: what is the capital of france? answer: paris.",
        "question: what is the capital of japan? answer: tokyo.",
        "question: what is 2 + 2? answer: 4.",
        "question: what is 5 x 5? answer: 25.",
        "question: what is 10 - 3? answer: 7.",
        "question: how many days are in a week? answer: 7.",
        "question: how many months are in a year? answer: 12.",
        "question: what color is the sky on a clear day? answer: blue.",
        "question: what color is grass? answer: green.",
        "question: what color is a banana? answer: yellow.",
        "question: what color is blood? answer: red.",
        "question: what animal says meow? answer: a cat.",
        "question: what animal says woof? answer: a dog.",
        "question: what animal says moo? answer: a cow.",
        "question: what is the largest ocean? answer: the pacific ocean.",
        "question: what is the tallest mountain? answer: mount everest.",
        "question: how many legs does a spider have? answer: 8.",
        "question: how many legs does an insect have? answer: 6.",
        "question: how many legs does a dog have? answer: 4.",
        "question: what planet do we live on? answer: earth.",
        "question: what is the closest star to earth? answer: the sun.",
        "question: how many planets are in the solar system? answer: 8.",
        "question: what gas do humans breathe in? answer: oxygen.",
        "question: what gas do humans breathe out? answer: carbon dioxide.",
        "question: what is water made of? answer: hydrogen and oxygen.",
        "question: what is the freezing point of water? answer: 0 degrees celsius.",
        "question: what is the boiling point of water? answer: 100 degrees celsius.",
        "question: how many sides does a triangle have? answer: 3.",
        "question: how many sides does a square have? answer: 4.",
        "question: how many sides does a hexagon have? answer: 6.",
        "question: what is 7 + 8? answer: 15.",
        "question: what is 9 x 9? answer: 81.",
        "question: what is 100 divided by 10? answer: 10.",
        "question: what is half of 10? answer: 5.",
        "question: what comes after monday? answer: tuesday.",
        "question: what comes after friday? answer: saturday.",
        "question: what comes after december? answer: january.",
        "question: what is the opposite of hot? answer: cold.",
        "question: what is the opposite of up? answer: down.",
        "question: what is the opposite of big? answer: small.",
        "question: what is the opposite of fast? answer: slow.",
        "question: what is the opposite of light? answer: dark.",
        "question: what is the opposite of good? answer: bad.",
        "question: what is the opposite of old? answer: new.",
        "question: what is the opposite of open? answer: closed.",
        "question: what is the opposite of day? answer: night.",
        "question: what is the opposite of wet? answer: dry.",
    ]
    return rng.choice(qa)

def _commonsense(rng):
    """Real commonsense knowledge."""
    cs = [
        "fire is hot.",
        "ice is cold.",
        "the sky is blue during the day.",
        "the sky is dark at night.",
        "you should drink water when you are thirsty.",
        "you should eat food when you are hungry.",
        "you should sleep when you are tired.",
        "you should wear warm clothes in winter.",
        "you should wear light clothes in summer.",
        "you should look both ways before crossing the street.",
        "you should wash your hands before eating.",
        "you should brush your teeth to keep them healthy.",
        "rain makes the ground wet.",
        "sun makes things warm.",
        "wind can blow things away.",
        "snow is cold and white.",
        "a knife is sharp.",
        "a pillow is soft.",
        "a rock is hard.",
        "a feather is light.",
        "you need air to breathe.",
        "you need water to live.",
        "you need food to live.",
        "you need sleep to be healthy.",
        "plants need sunlight to grow.",
        "plants need water to grow.",
        "fish live in water.",
        "birds live in trees.",
        "humans walk on two legs.",
        "dogs walk on four legs.",
    ]
    return rng.choice(cs)

def _reasoning_steps(rng):
    """Step-by-step problem solving."""
    problems = [
        "problem: if you have 5 apples and eat 2, how many are left? solution: 5 - 2 = 3. answer: 3 apples.",
        "problem: if you have 10 dollars and buy a toy for 3 dollars, how much money is left? solution: 10 - 3 = 7. answer: 7 dollars.",
        "problem: if a train travels 60 km per hour for 2 hours, how far does it go? solution: 60 x 2 = 120. answer: 120 km.",
        "problem: if 3 people share 12 cookies equally, how many does each get? solution: 12 / 3 = 4. answer: 4 cookies.",
        "problem: if water freezes at 0 degrees and it is -5 degrees outside, will water freeze? solution: -5 is less than 0. answer: yes.",
        "problem: if you have 2 red balls and 3 blue balls, how many balls do you have? solution: 2 + 3 = 5. answer: 5 balls.",
        "problem: if a shop opens at 9 am and closes at 5 pm, how many hours is it open? solution: 5 - 9 is negative, so 12 - 9 = 3, then 3 + 5 = 8. answer: 8 hours.",
        "problem: if today is wednesday, what day was yesterday? solution: the day before wednesday is tuesday. answer: tuesday.",
        "problem: if today is thursday, what day is tomorrow? solution: the day after thursday is friday. answer: friday.",
        "problem: if you have 20 dollars and each pencil costs 2 dollars, how many pencils can you buy? solution: 20 / 2 = 10. answer: 10 pencils.",
        "problem: if a rectangle has length 4 and width 3, what is the area? solution: area = length x width = 4 x 3 = 12. answer: 12.",
        "problem: if a rectangle has length 6 and width 2, what is the perimeter? solution: perimeter = 2 x (length + width) = 2 x 8 = 16. answer: 16.",
        "problem: if you flip a coin, what is the probability of heads? solution: 1 side out of 2 sides. answer: 1/2 or 50 percent.",
        "problem: if you roll a die, what is the probability of getting a 6? solution: 1 side out of 6 sides. answer: 1/6.",
        "problem: if a is 5 and b is 3, what is a + b? solution: 5 + 3 = 8. answer: 8.",
    ]
    return rng.choice(problems)

def generate_knowledge_corpus(n_records=30000, seed=42):
    """Generate a corpus of REAL knowledge."""
    rng = random.Random(seed)
    records = []
    mix = [
        ("math", 0.15),
        ("world_facts", 0.15),
        ("logic", 0.10),
        ("definitions", 0.10),
        ("sequences", 0.08),
        ("cause_effect", 0.08),
        ("comparisons", 0.08),
        ("qa", 0.15),
        ("commonsense", 0.06),
        ("reasoning", 0.05),
    ]
    for _ in range(n_records):
        r = rng.random()
        cumulative = 0
        for kind, weight in mix:
            cumulative += weight
            if r < cumulative:
                if kind == "math":
                    records.append({"text": _math_facts(rng)})
                elif kind == "world_facts":
                    records.append({"text": _world_facts(rng)})
                elif kind == "logic":
                    records.append({"text": _logic_chains(rng)})
                elif kind == "definitions":
                    records.append({"text": _definitions(rng)})
                elif kind == "sequences":
                    records.append({"text": _sequences(rng)})
                elif kind == "cause_effect":
                    records.append({"text": _cause_effect(rng)})
                elif kind == "comparisons":
                    records.append({"text": _comparisons(rng)})
                elif kind == "qa":
                    records.append({"text": _qa_pairs(rng)})
                elif kind == "commonsense":
                    records.append({"text": _commonsense(rng)})
                elif kind == "reasoning":
                    records.append({"text": _reasoning_steps(rng)})
                break
    return records


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    records = generate_knowledge_corpus(n)
    output = Path("data/raw/knowledge_corpus.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    total_chars = sum(len(r["text"]) for r in records)
    print(f"Generated {len(records):,} records")
    print(f"Total characters: {total_chars:,}")
    print(f"Average length: {total_chars/len(records):.1f} chars")
    print(f"Saved to {output}")
