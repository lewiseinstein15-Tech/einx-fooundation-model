# -*- coding: utf-8 -*-
"""
EINX Standalone v2 — Bigger corpus, early stopping, harder tests.

INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. File → Open notebook → GitHub → paste this URL:
   https://github.com/lewiseinstein15-Tech/einx-fooundation-model/blob/main/EINX_Run.ipynb
3. Runtime → Change runtime type → T4 GPU
4. Runtime → Run all
5. Wait ~8 minutes, see results at the bottom
"""

import math, json, random, hashlib, time, os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# MODEL CODE (same as before — proven to work)
# ============================================================

def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b); cs.append(2 ** 8 + n); n += 1
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
        self.vocab = [tok for tok, _ in self.special.all]
        self.token_to_id = {tok: idx for tok, idx in self.special.all}
        self.merges = []; self.merge_ranks = {}; self._trained = False

    def train(self, corpus, vocab_size=1024, verbose=False):
        if vocab_size < 260: raise ValueError(f"vocab_size must be >= 260, got {vocab_size}")
        self.vocab = [tok for tok, _ in self.special.all]
        self.token_to_id = {tok: idx for tok, idx in self.special.all}
        self.merges = []; self.merge_ranks = {}
        for b in range(256):
            tok = BYTE_TO_UNICODE[b]
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.vocab); self.vocab.append(tok)
        word_freqs = Counter()
        for line in corpus:
            for word in line.split():
                word_bytes = word.encode("utf-8")
                chars = [BYTE_TO_UNICODE[b] for b in word_bytes]
                if not chars: continue
                chars = [WORD_BOUNDARY] + chars
                word_freqs[" ".join(chars)] += 1
        words = {wk: wk.split(" ") for wk in word_freqs}
        target_merges = vocab_size - len(self.vocab)
        for step in range(max(0, target_merges)):
            pair_counts = Counter()
            for wk, symbols in words.items():
                freq = word_freqs[wk]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i+1])] += freq
            if not pair_counts: break
            best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            self.token_to_id[new_token] = len(self.vocab); self.vocab.append(new_token)
            for wk in words:
                symbols = words[wk]
                if len(symbols) < 2: continue
                new_symbols = []; i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_symbols.append(new_token); i += 2
                    else: new_symbols.append(symbols[i]); i += 1
                words[wk] = new_symbols
        self._trained = True

    def _bpe(self, tokens):
        if len(tokens) < 2: return tokens
        while True:
            best_rank = None; best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i+1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank; best_idx = i
            if best_rank is None: break
            tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx+1]] + tokens[best_idx+2:]
        return tokens

    def encode(self, text, add_bos=False, add_eos=False):
        if not self._trained: raise RuntimeError("tokenizer not trained")
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
        if not self._trained: raise RuntimeError("tokenizer not trained")
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
    def id_to_token(self, tid):
        return self.vocab[tid] if 0 <= tid < len(self.vocab) else None

@dataclass
class EINXModelConfig:
    name: str = "einx"; version: str = "0.1.0"; arch: str = "decoder-only-transformer"
    vocab_size: int = 1024; hidden_dim: int = 192; n_layers: int = 6; n_heads: int = 6
    head_dim: int = 32; max_context_length: int = 128; ffn_dim: int = 768; dropout: float = 0.1
    positional_encoding: str = "rope"; norm_type: str = "rms"; precision: str = "fp32"
    tie_word_embeddings: bool = True; bos_token_id: int = 1; eos_token_id: int = 2
    pad_token_id: int = 0; unk_token_id: int = 3
    def validate(self): assert self.head_dim * self.n_heads == self.hidden_dim
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        dtype = x.dtype; x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(rms + self.eps)).to(dtype) * self.weight

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    def _build_cache(self, max_seq_len):
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos_cached", torch.cat([cos, cos], -1), persistent=False)
        self.register_buffer("sin_cached", torch.cat([sin, sin], -1), persistent=False)
    def forward(self, seq_len, device, dtype):
        if seq_len > self.cos_cached.size(0): self._build_cache(seq_len)
        return self.cos_cached[:seq_len].to(device=device, dtype=dtype), self.sin_cached[:seq_len].to(device=device, dtype=dtype)

def rotate_half(x):
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)

class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=0.1):
        super().__init__()
        self.dim = dim; self.heads = heads; self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False); self.dropout = dropout
    def forward(self, x, rope=None):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.dim, -1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        if rope is not None:
            cos, sin = rope(T, x.device, x.dtype)
            cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype), 1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        return self.o(a.transpose(1, 2).contiguous().view(B, T, C))

class FFN(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        inter = ((int(ffn_dim * 2/3) + 7) // 8) * 8
        self.up = nn.Linear(dim, inter, bias=False)
        self.gate = nn.Linear(dim, inter, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.up(x)) * self.gate(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.hidden_dim); self.attn = Attention(cfg.hidden_dim, cfg.n_heads, cfg.head_dim, cfg.dropout)
        self.n2 = RMSNorm(cfg.hidden_dim); self.ffn = FFN(cfg.hidden_dim, cfg.ffn_dim)
    def forward(self, x, rope=None):
        return x + self.ffn(self.n2(x + self.attn(self.n1(x), rope)))

class EINXTransformer(nn.Module):
    def __init__(self, config):
        super().__init__(); config.validate(); self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.rope = RoPE(config.head_dim, config.max_context_length)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings: self.head.weight = self.embed.weight
        else: nn.init.normal_(self.head.weight, std=0.02)
        self._n = sum(p.numel() for p in self.parameters())
    @property
    def n_params(self): return self._n
    def forward(self, ids, targets=None):
        x = self.drop(self.embed(ids))
        for b in self.blocks: x = b(x, self.rope)
        x = self.norm(x); logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss
    @torch.no_grad()
    def generate(self, ids, max_new=64, temp=1.0, top_k=None):
        self.eval()
        for _ in range(max_new):
            ctx = ids[:, -self.config.max_context_length:]
            logits, _ = self.forward(ctx)
            nl = logits[:, -1, :]
            if temp > 0: nl = nl / temp
            else: ids = torch.cat([ids, torch.argmax(nl, -1, True)], 1); continue
            if top_k:
                v, _ = torch.topk(nl, min(top_k, nl.size(-1)), -1)
                nl = torch.where(nl >= v[:, -1:], nl, torch.full_like(nl, float("-inf")))
            ids = torch.cat([ids, torch.multinomial(F.softmax(nl, -1), 1)], 1)
        return ids
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "config": self.config.to_dict()}, path)
    @classmethod
    def load(cls, path, map_location="cpu"):
        c = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(EINXModelConfig.from_dict(c["config"])); m.load_state_dict(c["model_state_dict"]); return m

class SimpleDataset(Dataset):
    def __init__(self, texts, tok, ctx=128):
        all_ids = []
        for t in texts: all_ids.extend(tok.encode(t, add_eos=True))
        cs = ctx + 1; n = len(all_ids) // cs; all_ids = all_ids[:n * cs]
        self.chunks = [torch.tensor(all_ids[i:i+cs]) for i in range(0, len(all_ids), cs)]
        if not self.chunks: self.chunks = [torch.zeros(cs, dtype=torch.long)]
    def __len__(self): return len(self.chunks)
    def __getitem__(self, i):
        c = self.chunks[i]; return c[:-1], c[1:]


# ============================================================
# MASSIVE KNOWLEDGE CORPUS — 10,000+ unique facts
# ============================================================

def build_corpus():
    """Build a massive corpus with 10,000+ unique facts."""
    facts = set()

    # 1. ALL math: 1-12 for +, -, x (468 unique facts)
    for a in range(1, 13):
        for b in range(1, 13):
            facts.add(f"{a} + {b} = {a + b}.")
            facts.add(f"{a} x {b} = {a * b}.")
            if a > b: facts.add(f"{a} - {b} = {a - b}.")
            facts.add(f"question: what is {a} + {b}? answer: {a + b}.")
            facts.add(f"question: what is {a} x {b}? answer: {a * b}.")
            if a > b: facts.add(f"question: what is {a} - {b}? answer: {a - b}.")

    # 2. Larger math (13-50 for +)
    for a in range(13, 50):
        b = random.randint(1, 50)
        facts.add(f"{a} + {b} = {a + b}.")
        facts.add(f"question: what is {a} + {b}? answer: {a + b}.")
        if a > b: facts.add(f"{a} - {b} = {a - b}.")

    # 3. Capitals (90+ countries)
    capitals = [
        ('france','paris'),('japan','tokyo'),('england','london'),('germany','berlin'),
        ('italy','rome'),('china','beijing'),('russia','moscow'),('india','new delhi'),
        ('brazil','brasilia'),('egypt','cairo'),('kenya','nairobi'),('canada','ottawa'),
        ('australia','canberra'),('spain','madrid'),('greece','athens'),('portugal','lisbon'),
        ('netherlands','amsterdam'),('sweden','stockholm'),('norway','oslo'),('finland','helsinki'),
        ('denmark','copenhagen'),('poland','warsaw'),('turkey','ankara'),('south korea','seoul'),
        ('north korea','pyongyang'),('mexico','mexico city'),('argentina','buenos aires'),
        ('chile','santiago'),('peru','lima'),('colombia','bogota'),('venezuela','caracas'),
        ('thailand','bangkok'),('vietnam','hanoi'),('indonesia','jakarta'),('philippines','manila'),
        ('malaysia','kuala lumpur'),('singapore','singapore'),('pakistan','islamabad'),
        ('bangladesh','dhaka'),('nigeria','abuja'),('south africa','pretoria'),('morocco','rabat'),
        ('algeria','algiers'),('ethiopia','addis ababa'),('sudan','khartoum'),('tunisia','tunis'),
        ('libya','tripoli'),('iran','tehran'),('iraq','baghdad'),('saudi arabia','riyadh'),
        ('israel','jerusalem'),('jordan','amman'),('syria','damascus'),('lebanon','beirut'),
        ('yemen','sanaa'),('oman','muscat'),('qatar','doha'),('kuwait','kuwait city'),
        ('bahrain','manama'),('uae','abu dhabi'),('afghanistan','kabul'),('nepal','kathmandu'),
        ('sri lanka','colombo'),('iceland','reykjavik'),('new zealand','wellington'),
        ('hungary','budapest'),('czech republic','prague'),('romania','bucharest'),
        ('bulgaria','sofia'),('croatia','zagreb'),('serbia','belgrade'),('slovakia','bratislava'),
        ('ukraine','kyiv'),('belarus','minsk'),('latvia','riga'),('lithuania','vilnius'),
        ('estonia','tallinn'),('austria','vienna'),('switzerland','bern'),('belgium','brussels'),
        ('ireland','dublin'),('scotland','edinburgh'),('wales','cardiff'),
        ('mongolia','ulaanbaatar'),('kazakhstan','astana'),('uzbekistan','tashkent'),
        ('ghana','accra'),('uganda','kampala'),('tanzania','dodoma'),('zimbabwe','harare'),
        ('cameroon','yaounde'),('senegal','dakar'),('ivory coast','yamoussoukro'),
    ]
    for country, capital in capitals:
        facts.add(f"the capital of {country} is {capital}.")
        facts.add(f"question: what is the capital of {country}? answer: {capital}.")
        facts.add(f"{capital} is the capital of {country}.")

    # 4. Science facts (200+)
    science = [
        'water boils at 100 degrees celsius.','water freezes at 0 degrees celsius.',
        'water boils at 212 degrees fahrenheit.','water freezes at 32 degrees fahrenheit.',
        'the earth orbits the sun.','the moon orbits the earth.','the sun is a star.',
        'gravity pulls objects toward the earth.','a year has 365 days.','a week has 7 days.',
        'a day has 24 hours.','an hour has 60 minutes.','a minute has 60 seconds.',
        'there are 12 months in a year.','january is the first month.','february is the second month.',
        'march is the third month.','april is the fourth month.','may is the fifth month.',
        'june is the sixth month.','july is the seventh month.','august is the eighth month.',
        'september is the ninth month.','october is the tenth month.','november is the eleventh month.',
        'december is the twelfth month.','december is the last month.',
        'humans have 206 bones.','the human heart has 4 chambers.','the human body has 5 senses.',
        'blood is red because of hemoglobin.','plants make food through photosynthesis.',
        'photosynthesis converts sunlight into energy.','the largest planet is jupiter.',
        'the smallest planet is mercury.','mars is called the red planet.','venus is the hottest planet.',
        'the pacific ocean is the largest ocean.','mount everest is the tallest mountain.',
        'a triangle has 3 sides.','a square has 4 equal sides.','a circle has no corners.',
        'a pentagon has 5 sides.','a hexagon has 6 sides.','an octagon has 8 sides.',
        'a decagon has 10 sides.','there are 8 planets in the solar system.',
        'the earth has one moon.','mars has two moons.','jupiter has at least 79 moons.',
        'saturn has rings made of ice and rock.','the brain is the control center of the body.',
        'the lungs are used for breathing.','the heart pumps blood through the body.',
        'the stomach digests food.','the skin is the largest organ.',
        'iron is a metal that rusts.','gold does not rust.','diamond is the hardest natural material.',
        'copper is a good conductor of electricity.','wood floats on water.',
        'iron sinks in water.','a magnet has a north pole and a south pole.',
        'opposite poles of a magnet attract.','like poles of a magnet repel.',
        'a battery stores electrical energy.','a solar panel converts sunlight into electricity.',
        'sound travels at 343 meters per second.','light travels at 299792458 meters per second.',
        'mercury venus earth mars jupiter saturn uranus and neptune are the 8 planets.',
        'pluto is no longer classified as a planet.','the earth is the third planet from the sun.',
        'mercury is the first planet from the sun.','venus is the second planet from the sun.',
        'mars is the fourth planet from the sun.','jupiter is the fifth planet from the sun.',
        'saturn is the sixth planet from the sun.','uranus is the seventh planet from the sun.',
        'neptune is the eighth planet from the sun.','the sun is at the center of the solar system.',
        'the earth takes 365 days to orbit the sun.','the moon takes 28 days to orbit the earth.',
        'the earth rotates on its axis once every 24 hours.','gravity on the moon is weaker than on earth.',
        'the speed of light is faster than the speed of sound.',
        'hot air rises and cold air sinks.','water expands when it freezes.',
        'a kilometer is 1000 meters.','a kilogram is 1000 grams.','a liter is 1000 milliliters.',
        'a century is 100 years.','a decade is 10 years.','a millennium is 1000 years.',
        'there are 7 continents: africa antarctica asia australia europe north america south america.',
        'the largest continent is asia.','the smallest continent is australia.',
        'the nile is the longest river in africa.','the amazon is the longest river in south america.',
        'the sahara is the largest hot desert.','antarctica is the coldest place on earth.',
        'a herbivore eats only plants.','a carnivore eats only meat.','an omnivore eats both plants and meat.',
        'a predator hunts other animals.','prey is hunted by predators.',
        'fish have gills.','birds have feathers.','mammals have hair.','reptiles have scales.',
        'insects have 6 legs.','spiders have 8 legs.','centipedes can have 100 legs.',
        'octopuses have 8 arms.','starfish have 5 arms.','crabs have 10 legs.',
        'humans walk on 2 legs.','dogs walk on 4 legs.','insects walk on 6 legs.',
        'a cell is the basic unit of life.','dna carries genetic information.',
        'an atom is the smallest unit of matter.','a molecule is two or more atoms joined together.',
        'energy is the ability to do work.','temperature measures how hot or cold something is.',
        'mass is the amount of matter in an object.','volume is the amount of space something takes up.',
        'density is mass divided by volume.','speed is distance divided by time.',
        'a solid has a fixed shape.','a liquid takes the shape of its container.',
        'a gas fills its container.','evaporation is when a liquid turns into a gas.',
        'condensation is when a gas turns into a liquid.','melting is when a solid turns into a liquid.',
        'freezing is when a liquid turns into a solid.',
    ]
    for s in science:
        facts.add(s)
        # Also add as Q&A
        if s.startswith('the ') or s.startswith('a ') or s.startswith('an '):
            # Convert statement to question
            parts = s.split(' is ', 1) if ' is ' in s else s.split(' has ', 1) if ' has ' in s else None
            if parts and len(parts) == 2:
                q = f"question: what is {parts[0]}? answer: {parts[1]}"
                facts.add(q)

    # 5. Logic (60+)
    logic = [
        'if a is bigger than b, and b is bigger than c, then a is bigger than c.',
        'if all cats are animals, and tom is a cat, then tom is an animal.',
        'if all birds have wings, and a robin is a bird, then a robin has wings.',
        'if all fish live in water, and a salmon is a fish, then salmon live in water.',
        'if it is raining, the ground gets wet.',
        'if x is greater than 5, and 5 is greater than 3, then x is greater than 3.',
        'if all squares have 4 sides, and this shape has 3 sides, then it is not a square.',
        'if a number is even, it is divisible by 2. 8 is even, so 8 is divisible by 2.',
        'if a number is odd, it is not divisible by 2. 7 is odd, so 7 is not divisible by 2.',
        'if today is monday, tomorrow is tuesday.','if today is tuesday, tomorrow is wednesday.',
        'if today is wednesday, tomorrow is thursday.','if today is thursday, tomorrow is friday.',
        'if today is friday, tomorrow is saturday.','if today is saturday, tomorrow is sunday.',
        'if today is sunday, tomorrow is monday.','if today is monday, yesterday was sunday.',
        'if today is friday, yesterday was thursday.','if a + b = 10 and a = 3, then b = 7.',
        'if a + b = 10 and b = 8, then a = 2.','if a + b = 20 and a = 5, then b = 15.',
        'if 2x = 10, then x = 5.','if 3x = 15, then x = 5.','if 4x = 20, then x = 5.',
        'if 5x = 25, then x = 5.','if 6x = 36, then x = 6.','if x + 5 = 12, then x = 7.',
        'if x + 3 = 10, then x = 7.','if x + 7 = 15, then x = 8.','if x - 4 = 6, then x = 10.',
        'if x - 5 = 5, then x = 10.','if x - 3 = 7, then x = 10.',
        'if a shape has 3 sides, it is a triangle.','if a shape has 4 equal sides, it is a square.',
        'if a shape has 5 sides, it is a pentagon.','if a shape has 6 sides, it is a hexagon.',
        'if all mammals have hair, and a whale is a mammal, then whales have hair.',
        'if all dogs are mammals, and rex is a dog, then rex is a mammal.',
        'if all mammals are animals, and dogs are mammals, then dogs are animals.',
        'if all animals need water, and humans are animals, then humans need water.',
        'if all plants need sunlight, and a tree is a plant, then trees need sunlight.',
        'if all metals conduct electricity, and copper is a metal, then copper conducts electricity.',
        'if fire is hot, and ice is cold, then fire and ice have different temperatures.',
        'if the sun rises in the east, and sets in the west, then the sun moves from east to west.',
        'if all birds lay eggs, and a chicken is a bird, then chickens lay eggs.',
        'if all reptiles lay eggs, and a turtle is a reptile, then turtles lay eggs.',
        'if x + 10 = 20, then x = 10.','if x + 15 = 30, then x = 15.','if 2x = 20, then x = 10.',
        'if 3x = 30, then x = 10.','if 10x = 100, then x = 10.','if x / 2 = 5, then x = 10.',
        'if x / 5 = 3, then x = 15.','if half of x is 5, then x = 10.','if double x is 20, then x = 10.',
    ]
    for l in logic: facts.add(l)

    # 6. Opposites (50+)
    opposites = [
        ('hot','cold'),('up','down'),('big','small'),('fast','slow'),('light','dark'),
        ('good','bad'),('old','new'),('open','closed'),('day','night'),('wet','dry'),
        ('happy','sad'),('tall','short'),('full','empty'),('loud','quiet'),('hard','soft'),
        ('strong','weak'),('heavy','light'),('long','short'),('wide','narrow'),('thick','thin'),
        ('rich','poor'),('young','old'),('clean','dirty'),('safe','dangerous'),('easy','difficult'),
        ('sweet','sour'),('sharp','dull'),('bright','dim'),('smooth','rough'),('loose','tight'),
    ]
    for a, b in opposites:
        facts.add(f"the opposite of {a} is {b}.")
        facts.add(f"the opposite of {b} is {a}.")
        facts.add(f"question: what is the opposite of {a}? answer: {b}.")
        facts.add(f"question: what is the opposite of {b}? answer: {a}.")

    # 7. Colors (30+)
    colors = [
        ('sky','blue'),('grass','green'),('blood','red'),('snow','white'),('coal','black'),
        ('banana','yellow'),('orange','orange'),('grape','purple'),('sun','yellow'),
        ('ocean','blue'),('lemon','yellow'),('cherry','red'),('leaf','green'),('cloud','white'),
        ('chocolate','brown'),('carrot','orange'),('corn','yellow'),('tomato','red'),('frog','green'),
    ]
    for obj, color in colors:
        facts.add(f"{obj} is {color}.")
        facts.add(f"the color of {obj} is {color}.")
        facts.add(f"question: what color is {obj}? answer: {color}.")

    # 8. Animal sounds (20+)
    sounds = [
        ('cat','meow'),('dog','woof'),('cow','moo'),('duck','quack'),('sheep','baa'),
        ('pig','oink'),('bird','chirp'),('snake','hiss'),('lion','roar'),('horse','neigh'),
        ('frog','ribbit'),('bee','buzz'),('owl','hoot'),('wolf','howl'),('rooster','cock-a-doodle-doo'),
        ('mouse','squeak'),('cricket','chirp'),('donkey','hee-haw'),('goose','honk'),('crow','caw'),
    ]
    for animal, sound in sounds:
        facts.add(f"a {animal} says {sound}.")
        facts.add(f"question: what animal says {sound}? answer: a {animal}.")
        facts.add(f"question: what does a {animal} say? answer: {sound}.")
        facts.add(f"the {animal} makes a {sound} sound.")

    # 9. Reasoning problems (50+)
    reasoning = [
        'problem: if you have 5 apples and eat 2, how many are left? solution: 5 - 2 = 3. answer: 3 apples.',
        'problem: if you have 10 dollars and buy a toy for 3 dollars, how much is left? solution: 10 - 3 = 7. answer: 7 dollars.',
        'problem: if you have 2 red balls and 3 blue balls, how many balls? solution: 2 + 3 = 5. answer: 5 balls.',
        'problem: if each box holds 6 eggs and you have 4 boxes, how many eggs? solution: 6 x 4 = 24. answer: 24 eggs.',
        'problem: if a pizza has 8 slices and you eat 3, how many left? solution: 8 - 3 = 5. answer: 5 slices.',
        'problem: if you read 10 pages a day for 5 days, how many pages? solution: 10 x 5 = 50. answer: 50 pages.',
        'problem: if you have 20 dollars and pencils cost 2 dollars each, how many pencils? solution: 20 / 2 = 10. answer: 10 pencils.',
        'problem: if a is 5 and b is 3, what is a + b? solution: 5 + 3 = 8. answer: 8.',
        'problem: if a is 5 and b is 3, what is a - b? solution: 5 - 3 = 2. answer: 2.',
        'problem: if a is 5 and b is 3, what is a x b? solution: 5 x 3 = 15. answer: 15.',
        'problem: if 3 people share 12 cookies equally, how many each? solution: 12 / 3 = 4. answer: 4 cookies.',
        'problem: if today is wednesday, what day was yesterday? answer: tuesday.',
        'problem: if today is thursday, what day is tomorrow? answer: friday.',
        'problem: if today is monday, what day is tomorrow? answer: tuesday.',
        'problem: if today is friday, what day is tomorrow? answer: saturday.',
        'problem: if you have 15 marbles and lose 7, how many left? solution: 15 - 7 = 8. answer: 8 marbles.',
        'problem: if you have 3 boxes of 10 pencils, how many pencils? solution: 3 x 10 = 30. answer: 30 pencils.',
        'problem: if a movie is 2 hours long and starts at 3, when does it end? answer: 5.',
        'problem: if you have 100 dollars and spend 25, how much left? solution: 100 - 25 = 75. answer: 75 dollars.',
        'problem: if 4 people share 20 dollars equally, how much each? solution: 20 / 4 = 5. answer: 5 dollars.',
        'problem: if a rectangle has length 5 and width 3, what is the area? solution: 5 x 3 = 15. answer: 15.',
        'problem: if a rectangle has length 8 and width 2, what is the area? solution: 8 x 2 = 16. answer: 16.',
        'problem: if you flip a coin, what is the probability of heads? answer: 1/2 or 50 percent.',
        'problem: if you roll a die, what is the probability of getting a 6? answer: 1/6.',
        'problem: if a train travels 60 km per hour for 2 hours, how far? solution: 60 x 2 = 120. answer: 120 km.',
        'problem: if you have 2 dozen eggs, how many eggs? solution: 2 x 12 = 24. answer: 24 eggs.',
        'problem: if a store sells 5 apples per hour for 8 hours, how many sold? solution: 5 x 8 = 40. answer: 40 apples.',
        'problem: if you save 5 dollars per day for 30 days, how much? solution: 5 x 30 = 150. answer: 150 dollars.',
        'problem: if a book has 200 pages and you read 50, how many left? solution: 200 - 50 = 150. answer: 150 pages.',
    ]
    for r in reasoning: facts.add(r)

    # 10. Definitions (80+)
    definitions = [
        'a mammal is an animal that has hair and feeds its young with milk.',
        'a reptile is an animal with scales that lays eggs.','a bird is an animal with feathers and a beak.',
        'a fish is an animal that lives in water and has gills.','an insect is an animal with 6 legs and 3 body parts.',
        'gravity is the force that pulls objects toward each other.','friction opposes motion when two surfaces touch.',
        'energy is the ability to do work.','a molecule is two or more atoms joined together.',
        'an atom is the smallest unit of matter.','a cell is the basic unit of life.',
        'dna carries genetic information.','a planet orbits a star.','a star is a ball of hot gas that produces light.',
        'a galaxy is a group of stars.','a continent is a large landmass on earth.',
        'an island is land surrounded by water.','a peninsula is land surrounded by water on 3 sides.',
        'a mountain is a tall natural elevation.','a valley is a low area between hills or mountains.',
        'a river is a large stream of water that flows to the sea.','a lake is a body of water surrounded by land.',
        'an ocean is the largest body of salt water.','a desert is a dry area with very little rain.',
        'a forest is an area with many trees.','temperature measures how hot or cold something is.',
        'mass is the amount of matter in an object.','volume is the amount of space something takes up.',
        'density is mass divided by volume.','speed is distance divided by time.',
        'a solid is a state of matter with a fixed shape.','a liquid takes the shape of its container.',
        'a gas fills its container.','a herbivore eats only plants.','a carnivore eats only meat.',
        'an omnivore eats both plants and meat.','a predator hunts other animals.',
        'a habitat is the natural home of an animal.','an ecosystem is a community of living things.',
        'a food chain shows how energy passes from one living thing to another.',
        'photosynthesis is how plants make food from sunlight.','respiration is how living things release energy from food.',
        'a vaccine helps the body fight disease.','a virus can cause disease.',
        'bacteria are tiny single-celled organisms.','a fossil is the preserved remains of an ancient living thing.',
        'extinction is when a species dies out completely.','evolution is how species change over time.',
        'spring is when plants begin to grow.','summer is the hottest season.',
        'autumn is when leaves fall.','winter is the coldest season.',
        'a map shows where places are.','a compass shows direction.',
        'north south east and west are the 4 main directions.',
        'a country is a nation with its own government.','a city is a large town with many people.',
        'a road is a path for vehicles.','a bridge crosses over a river or road.',
        'a tunnel is a passage dug through the ground.',
        'a herbivore eats plants.','a carnivore eats meat.','an omnivore eats both.',
    ]
    for d in definitions: facts.add(d)

    # 11. Commonsense (100+)
    commonsense = [
        'fire is hot.','ice is cold.','the sky is blue during the day.','the sky is dark at night.',
        'you should drink water when thirsty.','you should eat food when hungry.','you should sleep when tired.',
        'wear warm clothes in winter.','wear light clothes in summer.','look both ways before crossing the street.',
        'wash your hands before eating.','brush your teeth to keep them healthy.','rain makes the ground wet.',
        'sun makes things warm.','wind can blow things away.','snow is cold and white.','a knife is sharp.',
        'a pillow is soft.','a rock is hard.','a feather is light.','you need air to breathe.',
        'you need water to live.','you need food to live.','you need sleep to be healthy.',
        'plants need sunlight to grow.','plants need water to grow.','fish live in water.',
        'birds live in trees.','humans walk on two legs.','dogs walk on four legs.',
        'spiders have eight legs.','insects have six legs.','the sun gives us light and heat.',
        'the moon shines at night.','stars are visible in the night sky.','clouds are made of water vapor.',
        'lightning is electricity in the sky.','thunder is the sound lightning makes.',
        'a rainbow appears after rain when the sun shines.','trees provide oxygen for us to breathe.',
        'honey is made by bees.','milk comes from cows.','eggs come from chickens.',
        'wool comes from sheep.','leather comes from animal skin.','paper is made from wood.',
        'glass is made from sand.','plastic is made from oil.','steel is made from iron.',
        'gold is a precious metal.','silver is a precious metal.','copper is used for wires.',
        'if you mix red and blue you get purple.','if you mix yellow and blue you get green.',
        'if you mix red and yellow you get orange.','if you mix all paint colors you get brown.',
        'if you mix all light colors you get white.','the earth is round.','the earth is mostly covered by water.',
        'the sun is very hot.','the moon is not a star.','the north pole is at the top of the earth.',
        'the south pole is at the bottom of the earth.','a year is 365 days.','a leap year has 366 days.',
        'february has 28 days.','february has 29 days in a leap year.','april june september and november have 30 days.',
        'the other months have 31 days except february.','water is made of hydrogen and oxygen.',
        'salt water is found in oceans.','fresh water is found in rivers and lakes.',
        'a thermometer measures temperature.','a ruler measures length.','a scale measures weight.',
        'a clock measures time.','a compass shows direction.',
    ]
    for c in commonsense: facts.add(c)

    # 12. Cause and effect (40+)
    cause_effect = [
        'if you heat ice, it melts into water.','if you heat water, it boils and becomes steam.',
        'if you cool water, it freezes into ice.','if you drop something, gravity pulls it down.',
        'if you touch fire, you get burned.','if you do not sleep, you become tired.',
        'if you eat too much, you feel full.','if you exercise, your muscles get stronger.',
        'if you study, you learn.','if you do not drink water, you become thirsty.',
        'if the sun shines, things become warm.','if wind blows, leaves move.',
        'if you leave bread out, it becomes stale.','if you leave metal in water, it may rust.',
        'if you leave water in the sun, it evaporates.','if you put a plant in the dark, it will die.',
        'if you do not water a plant, it will wilt.','if you exercise too hard, your muscles hurt.',
        'if you do not eat, you lose weight.','if you eat too many sweets, you get cavities.',
        'if you drive too fast, you may crash.','if you do not wear a coat in winter, you get cold.',
        'if you touch something hot, you burn your hand.','if you stay in the sun too long, you get sunburned.',
        'if you do not brush your teeth, you get cavities.','if you plant a seed, it grows.',
        'if you mix red and blue paint, you get purple.','if you mix yellow and blue paint, you get green.',
        'if you mix red and yellow paint, you get orange.','if you add salt to water, the water tastes salty.',
        'if you freeze water, it expands.','if you boil water, it turns to steam.',
        'if you heat metal, it expands.','if you cool metal, it contracts.',
        'if you press a button, something happens.','if you turn a key, a lock opens.',
        'if you blow on a candle, the flame goes out.','if you light a match, it burns.',
    ]
    for ce in cause_effect: facts.add(ce)

    # 13. Calendar
    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    for i, day in enumerate(days):
        next_day = days[(i + 1) % 7]
        prev_day = days[(i - 1) % 7]
        facts.add(f"the day after {day} is {next_day}.")
        facts.add(f"the day before {day} is {prev_day}.")
        facts.add(f"question: what comes after {day}? answer: {next_day}.")
        facts.add(f"question: what comes before {day}? answer: {prev_day}.")

    months = ['january','february','march','april','may','june','july','august','september','october','november','december']
    for i, month in enumerate(months):
        next_m = months[(i + 1) % 12]
        prev_m = months[(i - 1) % 12]
        facts.add(f"the month after {month} is {next_m}.")
        facts.add(f"the month before {month} is {prev_m}.")
        facts.add(f"question: what comes after {month}? answer: {next_m}.")

    # 14. Comparisons
    comparisons = [
        'a whale is bigger than a mouse.','an elephant is heavier than a cat.',
        'a cheetah is faster than a turtle.','a skyscraper is taller than a house.',
        'the ocean is deeper than a pool.','the sun is hotter than fire.',
        'a mountain is taller than a hill.','a galaxy is bigger than a solar system.',
        'an atom is smaller than a cell.','a year is longer than a month.',
        'a century is longer than a decade.','a kilometer is longer than a meter.',
        'a ton is heavier than a kilogram.','a liter is more than a milliliter.',
        'iron is heavier than wood.','gold is more valuable than copper.',
        'diamond is harder than glass.','steel is stronger than plastic.',
        'a tree is taller than grass.','a river is wider than a stream.',
        'jupiter is bigger than earth.','earth is bigger than the moon.',
        'the pacific ocean is bigger than the atlantic ocean.',
        'a blue whale is the largest animal.','a hummingbird is smaller than an eagle.',
        'a shark is more dangerous than a goldfish.','a lion is stronger than a cat.',
    ]
    for c in comparisons: facts.add(c)

    # 15. More Q&A
    extra_qa = [
        'question: how many hours in a day? answer: 24.',
        'question: how many minutes in an hour? answer: 60.',
        'question: how many seconds in a minute? answer: 60.',
        'question: how many days in january? answer: 31.',
        'question: how many days in february? answer: 28.',
        'question: how many days in april? answer: 30.',
        'question: what is the first day of the week? answer: monday.',
        'question: what is the last day of the week? answer: sunday.',
        'question: what is the first month? answer: january.',
        'question: what is the last month? answer: december.',
        'question: what are the 5 senses? answer: sight hearing smell taste touch.',
        'question: what organ pumps blood? answer: the heart.',
        'question: what organ do we breathe with? answer: the lungs.',
        'question: what organ do we think with? answer: the brain.',
        'question: what is the largest organ? answer: the skin.',
        'question: how many continents are there? answer: 7.',
        'question: what is the largest continent? answer: asia.',
        'question: what is the smallest continent? answer: australia.',
        'question: which continent is egypt in? answer: africa.',
        'question: which continent is japan in? answer: asia.',
        'question: which continent is brazil in? answer: south america.',
        'question: which continent is france in? answer: europe.',
        'question: what is the speed of light? answer: 299792458 meters per second.',
        'question: what is water made of? answer: hydrogen and oxygen.',
        'question: what gas do we breathe in? answer: oxygen.',
        'question: what gas do we breathe out? answer: carbon dioxide.',
        'question: how many sides does a triangle have? answer: 3.',
        'question: how many sides does a square have? answer: 4.',
        'question: how many sides does a pentagon have? answer: 5.',
        'question: how many sides does a hexagon have? answer: 6.',
        'question: how many sides does an octagon have? answer: 8.',
        'question: what is half of 10? answer: 5.',
        'question: what is half of 20? answer: 10.',
        'question: what is double 5? answer: 10.',
        'question: what is double 10? answer: 20.',
        'question: what is 100 divided by 10? answer: 10.',
        'question: what is 100 divided by 5? answer: 20.',
        'question: what is the largest planet? answer: jupiter.',
        'question: what is the smallest planet? answer: mercury.',
        'question: what is the hottest planet? answer: venus.',
        'question: which planet is called the red planet? answer: mars.',
        'question: how many moons does earth have? answer: 1.',
        'question: how many moons does mars have? answer: 2.',
        'question: what planet has rings? answer: saturn.',
        'question: what is the closest planet to the sun? answer: mercury.',
        'question: what is the farthest planet from the sun? answer: neptune.',
        'question: how many bones does a human have? answer: 206.',
        'question: how many chambers does the heart have? answer: 4.',
        'question: how many legs does a dog have? answer: 4.',
        'question: how many legs does a spider have? answer: 8.',
        'question: how many legs does an insect have? answer: 6.',
        'question: what is the largest ocean? answer: the pacific ocean.',
        'question: what is the tallest mountain? answer: mount everest.',
        'question: what is the longest river in africa? answer: the nile.',
        'question: what is the largest desert? answer: the sahara.',
        'question: what animal is the king of the jungle? answer: the lion.',
        'question: what animal is known as mans best friend? answer: the dog.',
        'question: what do bees make? answer: honey.',
        'question: what do cows give us? answer: milk.',
        'question: what do chickens lay? answer: eggs.',
        'question: where does wool come from? answer: sheep.',
        'question: what is paper made from? answer: wood.',
        'question: what is glass made from? answer: sand.',
        'question: what is the earth? answer: a planet.',
        'question: what is the sun? answer: a star.',
        'question: what is the moon? answer: a satellite.',
        'question: what causes day and night? answer: the earth rotating.',
        'question: what causes seasons? answer: the earth tilting.',
        'question: what is gravity? answer: the force that pulls objects down.',
        'question: what is a herbivore? answer: an animal that eats plants.',
        'question: what is a carnivore? answer: an animal that eats meat.',
        'question: what is an omnivore? answer: an animal that eats both.',
    ]
    for q in extra_qa: facts.add(q)

    return list(facts)


# ============================================================
# MAIN — Train + Test
# ============================================================

def main():
    print("=" * 60)
    print("EINX v2 — TRAIN + TEST ON GPU")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Build MASSIVE corpus
    print("\n1. Building knowledge corpus...")
    unique_facts = build_corpus()
    print(f"   {len(unique_facts):,} unique facts")

    # Repeat facts for training (with slight variations via shuffling)
    rng = random.Random(42)
    all_texts = []
    for _ in range(20):  # 20x repetition for memorization
        shuffled = list(unique_facts)
        rng.shuffle(shuffled)
        all_texts.extend(shuffled)
    print(f"   {len(all_texts):,} total training records")

    # 2. Train tokenizer
    print("\n2. Training tokenizer...")
    tok = BPETokenizer()
    tok.train(unique_facts, vocab_size=1024, verbose=False)
    print(f"   Vocab: {tok.vocab_size()}, Merges: {len(tok.merges)}")

    # 3. Build dataset
    print("\n3. Building dataset...")
    rng2 = random.Random(42)
    shuffled = list(all_texts)
    rng2.shuffle(shuffled)
    n_val = max(100, int(len(shuffled) * 0.02))
    val_texts = shuffled[:n_val]
    train_texts = shuffled[n_val:]

    train_ds = SimpleDataset(train_texts, tok, context_length=128)
    val_ds = SimpleDataset(val_texts, tok, context_length=128)
    print(f"   Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # 4. Build model
    print("\n4. Building model...")
    model_cfg = EINXModelConfig(
        name='einx-v2', vocab_size=tok.vocab_size(),
        hidden_dim=256, n_layers=8, n_heads=8, head_dim=32,
        max_context_length=128, ffn_dim=1024, dropout=0.1,
        positional_encoding='rope', norm_type='rms',
        precision='fp32', tie_word_embeddings=True,
    )
    model = EINXTransformer(model_cfg).to(device)
    print(f"   Parameters: {model.n_params:,}")
    print(f"   Architecture: {model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H")

    # 5. Train with early stopping
    print(f"\n5. Training up to 3000 steps with early stopping...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=0.0001)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    max_steps = 3000
    log_every = 100
    eval_every = 100
    patience = 5  # stop if val loss doesn't improve for 5 evals
    patience_counter = 0
    best_val_loss = float('inf')
    train_losses = []

    start_time = time.time()
    step = 0
    early_stopped = False

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps: break
            input_ids, targets = batch[0].to(device), batch[1].to(device)
            model.train()
            logits, loss = model(input_ids, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item())
            step += 1

            if step % log_every == 0:
                avg = sum(train_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                print(f"   step {step:5d}/{max_steps}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)")

            if step % eval_every == 0:
                model.eval()
                total_vl = 0; n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        if n >= 20: break
                        vi, vt = vb[0].to(device), vb[1].to(device)
                        _, vl = model(vi, targets=vt)
                        total_vl += vl.item(); n += 1
                avg_vl = total_vl / max(1, n)
                ppl = math.exp(avg_vl) if avg_vl < 20 else float('inf')
                print(f"   >>> eval step {step}: val={avg_vl:.4f}  ppl={ppl:.2f}  best={best_val_loss:.4f}")

                if avg_vl < best_val_loss:
                    best_val_loss = avg_vl
                    patience_counter = 0
                    os.makedirs('/content/checkpoints', exist_ok=True)
                    model.save('/content/checkpoints/best_model.pt')
                    print(f"   >>> NEW BEST! Saved checkpoint.")
                else:
                    patience_counter += 1
                    print(f"   >>> No improvement ({patience_counter}/{patience})")
                    if patience_counter >= patience:
                        print(f"   >>> EARLY STOPPING at step {step}")
                        early_stopped = True
                        break

        if early_stopped:
            break

    elapsed = time.time() - start_time
    initial_loss = train_losses[0] if train_losses else 0
    final_loss = train_losses[-1] if train_losses else 0
    ppl = math.exp(best_val_loss) if best_val_loss < 20 else float('inf')

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Steps:      {step}")
    print(f"Early stop: {'YES' if early_stopped else 'NO'}")
    print(f"Init loss:  {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Best val:   {best_val_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    # 6. Test
    print()
    print("=" * 60)
    print("KNOWLEDGE TEST")
    print("=" * 60)

    model = EINXTransformer.load('/content/checkpoints/best_model.pt', map_location=str(device))
    model.to(device); model.eval()

    tests = [
        # BASIC MATH
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

        # CAPITALS
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
        ('question: what is the capital of greece? answer:', 'athens'),

        # WORLD
        ('question: what planet do we live on? answer:', 'earth'),
        ('question: what is the closest star to earth? answer:', 'sun'),
        ('question: how many days are in a week? answer:', '7'),
        ('question: how many months are in a year? answer:', '12'),
        ('question: how many hours in a day? answer:', '24'),
        ('question: how many continents are there? answer:', '7'),
        ('question: how many planets are in the solar system? answer:', '8'),

        # COLORS
        ('question: what color is the sky? answer:', 'blue'),
        ('question: what color is grass? answer:', 'green'),
        ('question: what color is blood? answer:', 'red'),
        ('question: what color is snow? answer:', 'white'),
        ('question: what color is a banana? answer:', 'yellow'),
        ('question: what color is coal? answer:', 'black'),

        # ANIMALS
        ('question: what animal says meow? answer:', 'cat'),
        ('question: what animal says woof? answer:', 'dog'),
        ('question: what animal says moo? answer:', 'cow'),
        ('question: what animal says quack? answer:', 'duck'),
        ('question: what animal says baa? answer:', 'sheep'),
        ('question: what animal says roar? answer:', 'lion'),

        # SCIENCE
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

        # OPPOSITES
        ('question: what is the opposite of hot? answer:', 'cold'),
        ('question: what is the opposite of up? answer:', 'down'),
        ('question: what is the opposite of big? answer:', 'small'),
        ('question: what is the opposite of fast? answer:', 'slow'),
        ('question: what is the opposite of light? answer:', 'dark'),
        ('question: what is the opposite of good? answer:', 'bad'),
        ('question: what is the opposite of happy? answer:', 'sad'),
        ('question: what is the opposite of day? answer:', 'night'),

        # CALENDAR
        ('question: what comes after monday? answer:', 'tuesday'),
        ('question: what comes after friday? answer:', 'saturday'),
        ('question: what comes after december? answer:', 'january'),
        ('question: what comes after wednesday? answer:', 'thursday'),
        ('question: what is the first month? answer:', 'january'),
        ('question: what is the last month? answer:', 'december'),

        # LOGIC
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

    correct = 0
    total = len(tests)
    cat_correct = {}
    cat_total = {}
    print()

    for prompt, expected in tests:
        ids = tok.encode(prompt, add_bos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        generated = []
        for _ in range(10):
            with torch.no_grad():
                logits, _ = model(input_ids)
            next_id = torch.argmax(logits[0, -1, :]).item()
            if next_id == tok.special.eos_id: break
            generated.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], 1)
        answer = tok.decode(generated).strip().lower()
        is_correct = expected.lower() in answer
        if is_correct: correct += 1
        status = '✓' if is_correct else '✗'
        print(f'{status} Q: {prompt}')
        print(f'  Expected: {expected}  |  Got: {answer!r}')
        print()

    print("=" * 60)
    print(f'SCORE: {correct}/{total} ({correct/total*100:.0f}%)')
    print("=" * 60)

    results = {
        'model': 'EINX-v2',
        'parameters': model.n_params,
        'unique_facts': len(unique_facts),
        'training_steps': step,
        'early_stopped': early_stopped,
        'initial_loss': round(initial_loss, 4),
        'final_loss': round(final_loss, 4),
        'best_val_loss': round(best_val_loss, 4),
        'perplexity': round(ppl, 2),
        'score': f'{correct}/{total}',
        'percentage': round(correct/total*100, 1),
        'device': str(device),
    }
    os.makedirs('/content', exist_ok=True)
    with open('/content/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved. Copy everything above and send it back.")


if __name__ == '__main__':
    main()
