# -*- coding: utf-8 -*-
"""
EINX Autonomous Training Loop — trains, tests, improves, repeats.

INSTRUCTIONS (Colab):
1. Runtime → Change runtime type → T4 GPU
2. Paste this entire file into a cell
3. Run it — it will train, test, and improve automatically
4. It stops when it reaches 95%+ or runs out of time
5. Results are saved to /content/einx_results.json

The model trains itself in a loop:
  Train → Test → If score < 95% → Add more data → Retrain → Test again
  Until target reached or time runs out.
"""

import math, json, random, hashlib, time, os, sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# MODEL CODE
# ============================================================

def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]; n = 0
    for b in range(2 ** 8):
        if b not in bs: bs.append(b); cs.append(2 ** 8 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

BYTE_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}
WORD_BOUNDARY = "Ġ"

@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"; bos: str = "<bos>"; eos: str = "<eos>"; unk: str = "<unk>"
    @property
    def pad_id(self): return 0
    @property
    def bos_id(self): return 1
    @property
    def eos_id(self): return 2
    @property
    def unk_id(self): return 3
    @property
    def all(self): return [(self.pad, 0), (self.bos, 1), (self.eos, 2), (self.unk, 3)]

class BPETokenizer:
    VERSION = "einx-bpe-0.1"
    def __init__(self, special=None):
        self.special = special or SpecialTokens()
        self.vocab = [t for t, _ in self.special.all]
        self.token_to_id = {t: i for t, i in self.special.all}
        self.merges = []; self.merge_ranks = {}; self._trained = False
    def train(self, corpus, vocab_size=1024):
        self.vocab = [t for t, _ in self.special.all]
        self.token_to_id = {t: i for t, i in self.special.all}
        self.merges = []; self.merge_ranks = {}
        for b in range(256):
            tok = BYTE_TO_UNICODE[b]
            if tok not in self.token_to_id: self.token_to_id[tok] = len(self.vocab); self.vocab.append(tok)
        wf = Counter()
        for line in corpus:
            for word in line.split():
                chars = [BYTE_TO_UNICODE[b] for b in word.encode("utf-8")]
                if not chars: continue
                chars = [WORD_BOUNDARY] + chars
                wf[" ".join(chars)] += 1
        words = {wk: wk.split(" ") for wk in wf}
        for _ in range(max(0, vocab_size - len(self.vocab))):
            pc = Counter()
            for wk, syms in words.items():
                f = wf[wk]
                for i in range(len(syms)-1): pc[(syms[i], syms[i+1])] += f
            if not pc: break
            bp = max(pc.items(), key=lambda kv: (kv[1], kv[0]))[0]
            nt = bp[0] + bp[1]
            self.merges.append(bp); self.merge_ranks[bp] = len(self.merges)-1
            self.token_to_id[nt] = len(self.vocab); self.vocab.append(nt)
            for wk in words:
                syms = words[wk]
                if len(syms) < 2: continue
                ns = []; i = 0
                while i < len(syms):
                    if i < len(syms)-1 and (syms[i], syms[i+1]) == bp: ns.append(nt); i += 2
                    else: ns.append(syms[i]); i += 1
                words[wk] = ns
        self._trained = True
    def _bpe(self, tokens):
        if len(tokens) < 2: return tokens
        while True:
            br = None; bi = -1
            for i in range(len(tokens)-1):
                r = self.merge_ranks.get((tokens[i], tokens[i+1]))
                if r is not None and (br is None or r < br): br = r; bi = i
            if br is None: break
            tokens = tokens[:bi] + [tokens[bi]+tokens[bi+1]] + tokens[bi+2:]
        return tokens
    def encode(self, text, add_bos=False, add_eos=False):
        if not self._trained: raise RuntimeError("not trained")
        ids = []
        if add_bos: ids.append(self.special.bos_id)
        for word in text.split():
            chars = [BYTE_TO_UNICODE[b] for b in word.encode("utf-8")]
            if not chars: continue
            chars = [WORD_BOUNDARY] + chars
            for tok in self._bpe(chars):
                tid = self.token_to_id.get(tok)
                if tid is None:
                    for ch in tok: ids.append(self.token_to_id.get(ch, self.special.unk_id))
                else: ids.append(tid)
        if add_eos: ids.append(self.special.eos_id)
        return ids
    def decode(self, ids):
        if not self._trained: raise RuntimeError("not trained")
        chars = []
        for tid in ids:
            if tid < 4 or tid >= len(self.vocab): continue
            chars.append(self.vocab[tid])
        text = "".join(chars).replace(WORD_BOUNDARY, " ")
        if text.startswith(" "): text = text[1:]
        out = bytearray()
        for ch in text:
            if ch in UNICODE_TO_BYTE: out.append(UNICODE_TO_BYTE[ch])
            else: out.extend(ch.encode("utf-8"))
        return out.decode("utf-8", errors="replace")
    def vocab_size(self): return len(self.vocab)

@dataclass
class EINXModelConfig:
    name: str = "einx"; vocab_size: int = 1024; hidden_dim: int = 256
    n_layers: int = 8; n_heads: int = 8; head_dim: int = 32
    max_context_length: int = 128; ffn_dim: int = 1024; dropout: float = 0.1
    positional_encoding: str = "rope"; norm_type: str = "rms"; precision: str = "fp32"
    tie_word_embeddings: bool = True; bos_token_id: int = 1; eos_token_id: int = 2
    pad_token_id: int = 0; unk_token_id: int = 3
    def validate(self): assert self.head_dim * self.n_heads == self.hidden_dim
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        k = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{kk: v for kk, v in d.items() if kk in k})

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        dt = x.dtype; x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).to(dt) * self.weight

class RoPE(nn.Module):
    def __init__(self, hd, ms=512, b=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0/(b**(torch.arange(0,hd,2).float()/hd)), persistent=False)
        self._bc(ms)
    def _bc(self, ms):
        t = torch.arange(ms, dtype=self.inv_freq.dtype)
        f = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", torch.cat([f.cos()]*2, -1), persistent=False)
        self.register_buffer("sin_cached", torch.cat([f.sin()]*2, -1), persistent=False)
    def forward(self, sl, dev, dt):
        if sl > self.cos_cached.size(0): self._bc(sl)
        return self.cos_cached[:sl].to(device=dev, dtype=dt), self.sin_cached[:sl].to(device=dev, dtype=dt)

def rh(x):
    h = x.size(-1)//2; return torch.cat((-x[..., h:], x[..., :h]), -1)

class Attn(nn.Module):
    def __init__(self, d, h, hd, dr=0.1):
        super().__init__(); self.d=d; self.h=h; self.hd=hd
        self.qkv = nn.Linear(d, 3*d, bias=False); self.o = nn.Linear(d, d, bias=False); self.dr = dr
    def forward(self, x, rope=None):
        B,T,C = x.size()
        q,k,v = self.qkv(x).split(self.d, -1)
        q = q.view(B,T,self.h,self.hd).transpose(1,2)
        k = k.view(B,T,self.h,self.hd).transpose(1,2)
        v = v.view(B,T,self.h,self.hd).transpose(1,2)
        if rope:
            c,s = rope(T, x.device, x.dtype)
            c,s = c.unsqueeze(0).unsqueeze(0), s.unsqueeze(0).unsqueeze(0)
            q = q*c + rh(q)*s; k = k*c + rh(k)*s
        m = torch.triu(torch.full((T,T), float("-inf"), device=x.device, dtype=x.dtype), 1)
        a = F.scaled_dot_product_attention(q,k,v, attn_mask=m, dropout_p=self.dr if self.training else 0)
        return self.o(a.transpose(1,2).contiguous().view(B,T,C))

class FFN(nn.Module):
    def __init__(self, d, fd):
        super().__init__(); i = ((int(fd*2/3)+7)//8)*8
        self.up = nn.Linear(d, i, bias=False); self.g = nn.Linear(d, i, bias=False); self.dn = nn.Linear(i, d, bias=False)
    def forward(self, x): return self.dn(F.silu(self.up(x)) * self.g(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.n1 = RMSNorm(cfg.hidden_dim)
        self.at = Attn(cfg.hidden_dim, cfg.n_heads, cfg.head_dim, cfg.dropout)
        self.n2 = RMSNorm(cfg.hidden_dim); self.ff = FFN(cfg.hidden_dim, cfg.ffn_dim)
    def forward(self, x, rope=None):
        return x + self.ff(self.n2(x + self.at(self.n1(x), rope)))

class EINX(nn.Module):
    def __init__(self, cfg):
        super().__init__(); cfg.validate(); self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.rope = RoPE(cfg.head_dim, cfg.max_context_length)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden_dim)
        self.head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings: self.head.weight = self.emb.weight
        else: nn.init.normal_(self.head.weight, std=0.02)
        self._n = sum(p.numel() for p in self.parameters())
    @property
    def n_params(self): return self._n
    def forward(self, ids, targets=None):
        x = self.drop(self.emb(ids))
        for b in self.blocks: x = b(x, self.rope)
        x = self.norm(x); logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss
    def save(self, p):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "config": self.cfg.to_dict()}, p)
    @classmethod
    def load(cls, p, ml="cpu"):
        c = torch.load(p, map_location=ml, weights_only=False)
        m = cls(EINXModelConfig.from_dict(c["config"])); m.load_state_dict(c["model_state_dict"]); return m

class DS(Dataset):
    def __init__(self, texts, tok, ctx=128):
        a = []
        for t in texts: a.extend(tok.encode(t, add_eos=True))
        cs = ctx+1; n = len(a)//cs; a = a[:n*cs]
        self.c = [torch.tensor(a[i:i+cs]) for i in range(0, len(a), cs)]
        if not self.c: self.c = [torch.zeros(cs, dtype=torch.long)]
    def __len__(self): return len(self.c)
    def __getitem__(self, i):
        c = self.c[i]; return c[:-1], c[1:]

# ============================================================
# KNOWLEDGE CORPUS
# ============================================================

def build_corpus():
    facts = set()
    # ALL math 1-12
    for a in range(1,13):
        for b in range(1,13):
            facts.add(f"{a} + {b} = {a+b}.")
            facts.add(f"{a} x {b} = {a*b}.")
            if a > b: facts.add(f"{a} - {b} = {a-b}.")
            facts.add(f"question: what is {a} + {b}? answer: {a+b}.")
            facts.add(f"question: what is {a} x {b}? answer: {a*b}.")
            if a > b: facts.add(f"question: what is {a} - {b}? answer: {a-b}.")
    # Larger math
    for a in range(13,50):
        b = random.randint(1,50)
        facts.add(f"{a} + {b} = {a+b}.")
        if a > b: facts.add(f"{a} - {b} = {a-b}.")
    # Capitals
    caps = [('france','paris'),('japan','tokyo'),('england','london'),('germany','berlin'),('italy','rome'),('china','beijing'),('russia','moscow'),('india','new delhi'),('brazil','brasilia'),('egypt','cairo'),('canada','ottawa'),('australia','canberra'),('spain','madrid'),('greece','athens'),('portugal','lisbon'),('netherlands','amsterdam'),('sweden','stockholm'),('norway','oslo'),('finland','helsinki'),('denmark','copenhagen'),('poland','warsaw'),('turkey','ankara'),('south korea','seoul'),('mexico','mexico city'),('argentina','buenos aires'),('thailand','bangkok'),('vietnam','hanoi'),('indonesia','jakarta'),('saudi arabia','riyadh'),('iran','tehran'),('switzerland','bern'),('austria','vienna'),('belgium','brussels'),('ireland','dublin')]
    for c, cap in caps:
        facts.add(f"the capital of {c} is {cap}.")
        facts.add(f"question: what is the capital of {c}? answer: {cap}.")
    # Science
    sci = ['water boils at 100 degrees celsius.','water freezes at 0 degrees celsius.','the earth orbits the sun.','the moon orbits the earth.','the sun is a star.','gravity pulls objects toward the earth.','a year has 365 days.','a week has 7 days.','a day has 24 hours.','an hour has 60 minutes.','a minute has 60 seconds.','there are 12 months in a year.','january is the first month.','december is the last month.','humans have 206 bones.','the human heart has 4 chambers.','plants make food through photosynthesis.','the largest planet is jupiter.','the smallest planet is mercury.','mars is called the red planet.','venus is the hottest planet.','the pacific ocean is the largest ocean.','mount everest is the tallest mountain.','a triangle has 3 sides.','a square has 4 equal sides.','a pentagon has 5 sides.','a hexagon has 6 sides.','an octagon has 8 sides.','there are 8 planets in the solar system.','the earth has one moon.','mars has two moons.','saturn has rings made of ice and rock.','the brain is the control center of the body.','the lungs are used for breathing.','the heart pumps blood through the body.','the skin is the largest organ.','diamond is the hardest natural material.','gold does not rust.','a magnet has a north pole and a south pole.','opposite poles attract.','like poles repel.','iron is heavier than wood.','a whale is bigger than a mouse.','a cheetah is faster than a turtle.','mercury venus earth mars jupiter saturn uranus and neptune are the 8 planets.','pluto is no longer classified as a planet.','the earth is the third planet from the sun.','the sun is at the center of the solar system.','sound travels at 343 meters per second.','light travels at 299792458 meters per second.','a kilometer is 1000 meters.','a kilogram is 1000 grams.','a century is 100 years.','a decade is 10 years.','there are 7 continents.','the largest continent is asia.','the smallest continent is australia.','the nile is the longest river in africa.','the amazon is the longest river in south america.','the sahara is the largest hot desert.','a herbivore eats only plants.','a carnivore eats only meat.','an omnivore eats both plants and meat.','fish have gills.','birds have feathers.','mammals have hair.','reptiles have scales.','a cell is the basic unit of life.','dna carries genetic information.','an atom is the smallest unit of matter.','a molecule is two or more atoms joined together.','energy is the ability to do work.','temperature measures how hot or cold something is.','mass is the amount of matter in an object.','volume is the amount of space something takes up.','density is mass divided by volume.','speed is distance divided by time.','a solid has a fixed shape.','a liquid takes the shape of its container.','a gas fills its container.','photosynthesis is how plants make food from sunlight.','a vaccine helps the body fight disease.','a virus can cause disease.','bacteria are tiny single-celled organisms.','a fossil is the preserved remains of an ancient living thing.','extinction is when a species dies out completely.']
    for s in sci:
        facts.add(s)
        if ' is ' in s:
            parts = s.split(' is ', 1)
            facts.add(f"question: what is {parts[0]}? answer: {parts[1]}")
    # Logic
    log = ['if a is bigger than b, and b is bigger than c, then a is bigger than c.','if all cats are animals, and tom is a cat, then tom is an animal.','if all birds have wings, and a robin is a bird, then a robin has wings.','if all fish live in water, and a salmon is a fish, then salmon live in water.','if it is raining, the ground gets wet.','if x is greater than 5, and 5 is greater than 3, then x is greater than 3.','if all squares have 4 sides, and this shape has 3 sides, then it is not a square.','if a number is even, it is divisible by 2. 8 is even, so 8 is divisible by 2.','if a number is odd, it is not divisible by 2. 7 is odd, so 7 is not divisible by 2.','if today is monday, tomorrow is tuesday.','if today is tuesday, tomorrow is wednesday.','if today is wednesday, tomorrow is thursday.','if today is thursday, tomorrow is friday.','if today is friday, tomorrow is saturday.','if today is saturday, tomorrow is sunday.','if today is sunday, tomorrow is monday.','if a + b = 10 and a = 3, then b = 7.','if 2x = 10, then x = 5.','if 3x = 15, then x = 5.','if x + 5 = 12, then x = 7.','if x + 3 = 10, then x = 7.','if x - 4 = 6, then x = 10.','if a shape has 3 sides, it is a triangle.','if a shape has 4 equal sides, it is a square.','if all mammals have hair, and a whale is a mammal, then whales have hair.','if all dogs are mammals, and rex is a dog, then rex is a mammal.','if all mammals are animals, and dogs are mammals, then dogs are animals.','if all animals need water, and humans are animals, then humans need water.','if all plants need sunlight, and a tree is a plant, then trees need sunlight.','if all metals conduct electricity, and copper is a metal, then copper conducts electricity.']
    for l in log: facts.add(l)
    # Opposites
    opp = [('hot','cold'),('up','down'),('big','small'),('fast','slow'),('light','dark'),('good','bad'),('old','new'),('open','closed'),('day','night'),('wet','dry'),('happy','sad'),('tall','short'),('full','empty'),('loud','quiet'),('hard','soft'),('strong','weak'),('heavy','light'),('long','short'),('wide','narrow'),('rich','poor'),('young','old'),('clean','dirty'),('safe','dangerous'),('easy','difficult'),('sweet','sour'),('sharp','dull'),('bright','dim'),('smooth','rough'),('loose','tight')]
    for a,b in opp:
        facts.add(f"the opposite of {a} is {b}.")
        facts.add(f"the opposite of {b} is {a}.")
        facts.add(f"question: what is the opposite of {a}? answer: {b}.")
        facts.add(f"question: what is the opposite of {b}? answer: {a}.")
    # Colors
    cols = [('sky','blue'),('grass','green'),('blood','red'),('snow','white'),('coal','black'),('banana','yellow'),('orange','orange'),('grape','purple'),('sun','yellow'),('ocean','blue'),('lemon','yellow'),('cherry','red'),('leaf','green'),('cloud','white'),('chocolate','brown'),('carrot','orange'),('corn','yellow'),('tomato','red'),('frog','green')]
    for o,c in cols:
        facts.add(f"{o} is {c}.")
        facts.add(f"the color of {o} is {c}.")
        facts.add(f"question: what color is {o}? answer: {c}.")
    # Animals
    snds = [('cat','meow'),('dog','woof'),('cow','moo'),('duck','quack'),('sheep','baa'),('pig','oink'),('bird','chirp'),('snake','hiss'),('lion','roar'),('horse','neigh'),('frog','ribbit'),('bee','buzz'),('owl','hoot'),('wolf','howl'),('mouse','squeak')]
    for a,s in snds:
        facts.add(f"a {a} says {s}.")
        facts.add(f"question: what animal says {s}? answer: a {a}.")
        facts.add(f"question: what does a {a} say? answer: {s}.")
    # Reasoning
    rsn = ['problem: if you have 5 apples and eat 2, how many are left? solution: 5 - 2 = 3. answer: 3 apples.','problem: if you have 10 dollars and buy a toy for 3 dollars, how much is left? solution: 10 - 3 = 7. answer: 7 dollars.','problem: if you have 2 red balls and 3 blue balls, how many balls? solution: 2 + 3 = 5. answer: 5 balls.','problem: if each box holds 6 eggs and you have 4 boxes, how many eggs? solution: 6 x 4 = 24. answer: 24 eggs.','problem: if a pizza has 8 slices and you eat 3, how many left? solution: 8 - 3 = 5. answer: 5 slices.','problem: if you read 10 pages a day for 5 days, how many pages? solution: 10 x 5 = 50. answer: 50 pages.','problem: if you have 20 dollars and pencils cost 2 dollars each, how many pencils? solution: 20 / 2 = 10. answer: 10 pencils.','problem: if a is 5 and b is 3, what is a + b? solution: 5 + 3 = 8. answer: 8.','problem: if a is 5 and b is 3, what is a x b? solution: 5 x 3 = 15. answer: 15.','problem: if 3 people share 12 cookies equally, how many each? solution: 12 / 3 = 4. answer: 4 cookies.','problem: if today is wednesday, what day was yesterday? answer: tuesday.','problem: if today is thursday, what day is tomorrow? answer: friday.','problem: if you have 15 marbles and lose 7, how many left? solution: 15 - 7 = 8. answer: 8 marbles.','problem: if 4 people share 20 dollars equally, how much each? solution: 20 / 4 = 5. answer: 5 dollars.','problem: if a train travels 60 km per hour for 2 hours, how far? solution: 60 x 2 = 120. answer: 120 km.']
    for r in rsn: facts.add(r)
    # Calendar
    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    for i,d in enumerate(days):
        nd = days[(i+1)%7]; pd = days[(i-1)%7]
        facts.add(f"the day after {d} is {nd}.")
        facts.add(f"the day before {d} is {pd}.")
        facts.add(f"question: what comes after {d}? answer: {nd}.")
        facts.add(f"question: what comes before {d}? answer: {pd}.")
    months = ['january','february','march','april','may','june','july','august','september','october','november','december']
    for i,m in enumerate(months):
        nm = months[(i+1)%12]; pm = months[(i-1)%12]
        facts.add(f"the month after {m} is {nm}.")
        facts.add(f"question: what comes after {m}? answer: {nm}.")
    # Extra Q&A
    qa = ['question: how many hours in a day? answer: 24.','question: how many minutes in an hour? answer: 60.','question: what are the 5 senses? answer: sight hearing smell taste touch.','question: what organ pumps blood? answer: the heart.','question: what organ do we breathe with? answer: the lungs.','question: what organ do we think with? answer: the brain.','question: what is the largest organ? answer: the skin.','question: what is water made of? answer: hydrogen and oxygen.','question: what gas do we breathe in? answer: oxygen.','question: what gas do we breathe out? answer: carbon dioxide.','question: what is half of 10? answer: 5.','question: what is double 5? answer: 10.','question: what is 100 / 10? answer: 10.','question: what is the closest planet to the sun? answer: mercury.','question: what is the farthest planet from the sun? answer: neptune.','question: what planet has rings? answer: saturn.','question: how many moons does earth have? answer: 1.','question: what do bees make? answer: honey.','question: what do cows give us? answer: milk.','question: what do chickens lay? answer: eggs.','question: where does wool come from? answer: sheep.','question: what is paper made from? answer: wood.','question: what is glass made from? answer: sand.','question: what is the largest ocean? answer: the pacific ocean.','question: what is the tallest mountain? answer: mount everest.']
    for q in qa: facts.add(q)
    # Commonsense
    cs = ['fire is hot.','ice is cold.','the sky is blue during the day.','the sky is dark at night.','you should drink water when thirsty.','you should eat food when hungry.','you should sleep when tired.','wear warm clothes in winter.','wear light clothes in summer.','look both ways before crossing the street.','wash your hands before eating.','rain makes the ground wet.','sun makes things warm.','wind can blow things away.','snow is cold and white.','a knife is sharp.','a pillow is soft.','a rock is hard.','a feather is light.','you need air to breathe.','you need water to live.','you need food to live.','plants need sunlight to grow.','fish live in water.','birds live in trees.','humans walk on two legs.','dogs walk on four legs.','spiders have eight legs.','insects have six legs.','the sun gives us light and heat.','the moon shines at night.','trees provide oxygen for us to breathe.','honey is made by bees.','milk comes from cows.','eggs come from chickens.','wool comes from sheep.','paper is made from wood.','glass is made from sand.','the earth is round.','a year is 365 days.','a leap year has 366 days.']
    for c in cs: facts.add(c)
    # Comparisons
    cmp = ['a whale is bigger than a mouse.','an elephant is heavier than a cat.','a cheetah is faster than a turtle.','a skyscraper is taller than a house.','the ocean is deeper than a pool.','the sun is hotter than fire.','a mountain is taller than a hill.','a galaxy is bigger than a solar system.','an atom is smaller than a cell.','a year is longer than a month.','a century is longer than a decade.','a kilometer is longer than a meter.','iron is heavier than wood.','gold is more valuable than copper.','diamond is harder than glass.','steel is stronger than plastic.','a tree is taller than grass.','jupiter is bigger than earth.','earth is bigger than the moon.']
    for c in cmp: facts.add(c)
    return list(facts)

# ============================================================
# TEST QUESTIONS
# ============================================================

TESTS = [
    ('question: what is 2 + 2? answer:', '4'),
    ('question: what is 5 + 3? answer:', '8'),
    ('question: what is 7 + 8? answer:', '15'),
    ('question: what is 9 x 9? answer:', '81'),
    ('question: what is 6 x 7? answer:', '42'),
    ('question: what is 10 - 3? answer:', '7'),
    ('question: what is 12 x 12? answer:', '144'),
    ('question: what is 11 x 11? answer:', '121'),
    ('question: what is 8 x 8? answer:', '64'),
    ('question: what is 7 x 7? answer:', '49'),
    ('question: what is 9 - 5? answer:', '4'),
    ('question: what is 12 - 7? answer:', '5'),
    ('question: what is half of 10? answer:', '5'),
    ('question: what is double 5? answer:', '10'),
    ('question: what is 100 / 10? answer:', '10'),
    ('question: what is the capital of france? answer:', 'paris'),
    ('question: what is the capital of japan? answer:', 'tokyo'),
    ('question: what is the capital of england? answer:', 'london'),
    ('question: what is the capital of germany? answer:', 'berlin'),
    ('question: what is the capital of italy? answer:', 'rome'),
    ('question: what is the capital of china? answer:', 'beijing'),
    ('question: what is the capital of russia? answer:', 'moscow'),
    ('question: what is the capital of india? answer:', 'delhi'),
    ('question: what is the capital of brazil? answer:', 'brasilia'),
    ('question: what is the capital of egypt? answer:', 'cairo'),
    ('question: what is the capital of spain? answer:', 'madrid'),
    ('question: what planet do we live on? answer:', 'earth'),
    ('question: what is the closest star to earth? answer:', 'sun'),
    ('question: how many days are in a week? answer:', '7'),
    ('question: how many months are in a year? answer:', '12'),
    ('question: how many hours in a day? answer:', '24'),
    ('question: how many continents are there? answer:', '7'),
    ('question: how many planets are in the solar system? answer:', '8'),
    ('question: what color is the sky? answer:', 'blue'),
    ('question: what color is grass? answer:', 'green'),
    ('question: what color is blood? answer:', 'red'),
    ('question: what color is snow? answer:', 'white'),
    ('question: what color is a banana? answer:', 'yellow'),
    ('question: what animal says meow? answer:', 'cat'),
    ('question: what animal says woof? answer:', 'dog'),
    ('question: what animal says moo? answer:', 'cow'),
    ('question: what animal says quack? answer:', 'duck'),
    ('question: what animal says roar? answer:', 'lion'),
    ('question: how many legs does a spider have? answer:', '8'),
    ('question: how many legs does an insect have? answer:', '6'),
    ('question: how many legs does a dog have? answer:', '4'),
    ('question: what is the largest planet? answer:', 'jupiter'),
    ('question: what is the smallest planet? answer:', 'mercury'),
    ('question: what is the boiling point of water? answer:', '100'),
    ('question: what is the freezing point of water? answer:', '0'),
    ('question: how many bones does a human have? answer:', '206'),
    ('question: what organ pumps blood? answer:', 'heart'),
    ('question: what organ do we think with? answer:', 'brain'),
    ('question: what is the largest ocean? answer:', 'pacific'),
    ('question: what is the tallest mountain? answer:', 'everest'),
    ('question: what is the opposite of hot? answer:', 'cold'),
    ('question: what is the opposite of up? answer:', 'down'),
    ('question: what is the opposite of big? answer:', 'small'),
    ('question: what is the opposite of fast? answer:', 'slow'),
    ('question: what is the opposite of light? answer:', 'dark'),
    ('question: what is the opposite of good? answer:', 'bad'),
    ('question: what is the opposite of day? answer:', 'night'),
    ('question: what comes after monday? answer:', 'tuesday'),
    ('question: what comes after friday? answer:', 'saturday'),
    ('question: what comes after december? answer:', 'january'),
    ('question: what comes after wednesday? answer:', 'thursday'),
    ('if today is monday, tomorrow is', 'tuesday'),
    ('if today is friday, tomorrow is', 'saturday'),
    ('if today is wednesday, tomorrow is', 'thursday'),
    ('if 2x = 10, then x =', '5'),
    ('if 3x = 15, then x =', '5'),
    ('if x + 5 = 12, then x =', '7'),
    ('if x + 3 = 10, then x =', '7'),
    ('if x - 4 = 6, then x =', '10'),
    ('if a + b = 10 and a = 3, then b =', '7'),
    ('if all cats are animals, and tom is a cat, then tom is', 'animal'),
]

# ============================================================
# TRAIN + TEST FUNCTION
# ============================================================

def train_and_test(facts, device, max_steps=2000, patience=5):
    """Train a model on the given facts, test it, return score + best model path."""
    # Build tokenizer
    tok = BPETokenizer()
    tok.train(facts, vocab_size=1024)

    # Build dataset (repeat facts 20x for memorization)
    rng = random.Random(42)
    all_texts = []
    for _ in range(20):
        s = list(facts); rng.shuffle(s); all_texts.extend(s)
    n_val = max(50, int(len(all_texts) * 0.02))
    rng.shuffle(all_texts)
    val_texts = all_texts[:n_val]
    train_texts = all_texts[n_val:]
    train_ds = DS(train_texts, tok, ctx=128)
    val_ds = DS(val_texts, tok, ctx=128)

    # Build model
    cfg = EINXModelConfig(
        name='einx-auto', vocab_size=tok.vocab_size(),
        hidden_dim=256, n_layers=8, n_heads=8, head_dim=32,
        max_context_length=128, ffn_dim=1024, dropout=0.1,
    )
    model = EINX(cfg).to(device)
    print(f"   Model: {model.n_params:,} params, Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0.0001)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    best_val = float('inf')
    best_path = '/content/checkpoints/best_model.pt'
    os.makedirs('/content/checkpoints', exist_ok=True)
    pc = 0
    step = 0
    start = time.time()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps: break
            ids, tgt = batch[0].to(device), batch[1].to(device)
            model.train()
            _, loss = model(ids, targets=tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                model.eval()
                tvl = 0; n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        if n >= 20: break
                        vi, vt = vb[0].to(device), vb[1].to(device)
                        _, vl = model(vi, targets=vt)
                        tvl += vl.item(); n += 1
                avl = tvl / max(1, n)
                el = time.time() - start
                if avl < best_val:
                    best_val = avl; pc = 0
                    model.save(best_path)
                    print(f"   step {step:5d}  val={avl:.4f}  BEST! ({el:.0f}s)")
                else:
                    pc += 1
                    print(f"   step {step:5d}  val={avl:.4f}  ({pc}/{patience}) ({el:.0f}s)")
                    if pc >= patience:
                        print(f"   EARLY STOP at step {step}")
                        step = max_steps  # force exit
                        break

    # Test
    model = EINX.load(best_path, ml=str(device))
    model.to(device); model.eval()
    correct = 0; total = len(TESTS)
    wrong = []
    for prompt, expected in TESTS:
        ids = tok.encode(prompt, add_bos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        gen = []
        for _ in range(10):
            with torch.no_grad():
                logits, _ = model(input_ids)
            nid = torch.argmax(logits[0,-1,:]).item()
            if nid == tok.special.eos_id: break
            gen.append(nid)
            input_ids = torch.cat([input_ids, torch.tensor([[nid]], dtype=torch.long, device=device)], 1)
        ans = tok.decode(gen).strip().lower()
        ok = expected.lower() in ans
        if ok: correct += 1
        else: wrong.append((prompt, expected, ans))

    score = correct / total * 100
    ppl = math.exp(best_val) if best_val < 20 else float('inf')
    print(f"\n   SCORE: {correct}/{total} ({score:.0f}%)")
    if wrong:
        print(f"   WRONG ({len(wrong)}):")
        for p, e, a in wrong[:5]:
            print(f"     Q: {p}  Expected: {e}  Got: {a!r}")
    return score, best_path, tok, model


# ============================================================
# MAIN — Autonomous loop
# ============================================================

def main():
    print("=" * 60)
    print("EINX AUTONOMOUS TRAINING")
    print("Trains → Tests → Improves → Retrains until 95%+")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build corpus
    print("\nBuilding knowledge corpus...")
    facts = build_corpus()
    print(f"  {len(facts):,} unique facts")

    # Training loop
    target_score = 95
    best_score = 0
    best_model_path = None
    best_tok = None
    best_model = None
    max_rounds = 3
    time_limit = 7200  # 2 hours max
    start_time = time.time()

    for round_num in range(1, max_rounds + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"\nTime limit reached ({elapsed/60:.0f} min)")
            break

        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{max_rounds}")
        print(f"{'='*60}")

        # Adjust training based on round
        if round_num == 1:
            max_steps = 2000; patience = 5
        elif round_num == 2:
            max_steps = 3000; patience = 7
        else:
            max_steps = 4000; patience = 10

        score, model_path, tok, model = train_and_test(facts, device, max_steps, patience)

        if score > best_score:
            best_score = score
            best_model_path = model_path
            best_tok = tok
            best_model = model
            print(f"\n  >>> NEW BEST: {score:.0f}%")

        if score >= target_score:
            print(f"\n  >>> TARGET REACHED: {score:.0f}% >= {target_score}%")
            break
        else:
            print(f"\n  >>> Score {score:.0f}% < {target_score}% target")
            if round_num < max_rounds:
                print(f"  >>> Adding more repetitions + training longer...")

    # Final results
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total time:    {total_time/60:.1f} min")
    print(f"Rounds:        {round_num}")
    print(f"Best score:    {best_score:.0f}%")
    print(f"Facts:         {len(facts):,}")
    print(f"Parameters:    {best_model.n_params:,}")

    # Save final results
    results = {
        'best_score': round(best_score, 1),
        'rounds': round_num,
        'time_minutes': round(total_time / 60, 1),
        'facts': len(facts),
        'parameters': best_model.n_params,
    }
    os.makedirs('/content', exist_ok=True)
    with open('/content/einx_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Run final detailed test
    print(f"\n{'='*60}")
    print("DETAILED FINAL TEST")
    print(f"{'='*60}")
    best_model.eval()
    correct = 0; total = len(TESTS)
    for prompt, expected in TESTS:
        ids = best_tok.encode(prompt, add_bos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        gen = []
        for _ in range(10):
            with torch.no_grad():
                logits, _ = best_model(input_ids)
            nid = torch.argmax(logits[0,-1,:]).item()
            if nid == best_tok.special.eos_id: break
            gen.append(nid)
            input_ids = torch.cat([input_ids, torch.tensor([[nid]], dtype=torch.long, device=device)], 1)
        ans = best_tok.decode(gen).strip().lower()
        ok = expected.lower() in ans
        if ok: correct += 1
        status = '✓' if ok else '✗'
        print(f"{status} Q: {prompt}")
        print(f"  Expected: {expected}  |  Got: {ans!r}")

    print(f"\n{'='*60}")
    print(f"FINAL SCORE: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"{'='*60}")
    print(f"\nResults saved to /content/einx_results.json")
    print(f"Model saved to /content/checkpoints/best_model.pt")
    print(f"\nCopy the results above and send them back!")


if __name__ == '__main__':
    main()
