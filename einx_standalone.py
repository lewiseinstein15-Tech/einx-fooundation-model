# -*- coding: utf-8 -*-
"""
EINX Standalone — Train + Test on Free GPU (Google Colab)

INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. Click "File" → "Upload notebook"
3. Upload this file (einx_standalone.ipynb) — OR just create a new notebook and paste this code
4. Runtime → Change runtime type → T4 GPU
5. Runtime → Run all
6. Wait ~5 minutes, see results at the bottom

This file is 100% self-contained. No repo cloning, no imports from einx.
Everything is right here.
"""

# ============================================================
# PART 1: All EINX model code (inlined — no imports needed)
# ============================================================

import math, json, random, hashlib, time, os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- Tokenizer ---

def _bytes_to_unicode():
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
WORD_BOUNDARY = "Ġ"

@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"
    @property
    def pad_id(self): return 0
    @property
    def bos_id(self): return 1
    @property
    def eos_id(self): return 2
    @property
    def unk_id(self): return 3
    @property
    def all(self):
        return [(self.pad, 0), (self.bos, 1), (self.eos, 2), (self.unk, 3)]

class BPETokenizer:
    VERSION = "einx-bpe-0.1"

    def __init__(self, special=None):
        self.special = special or SpecialTokens()
        self.vocab = [tok for tok, _ in self.special.all]
        self.token_to_id = {tok: idx for tok, idx in self.special.all}
        self.merges = []
        self.merge_ranks = {}
        self._trained = False

    def train(self, corpus, vocab_size=1024, verbose=False):
        if vocab_size < 260:
            raise ValueError(f"vocab_size must be >= 260, got {vocab_size}")
        self.vocab = [tok for tok, _ in self.special.all]
        self.token_to_id = {tok: idx for tok, idx in self.special.all}
        self.merges = []
        self.merge_ranks = {}
        for b in range(256):
            tok = BYTE_TO_UNICODE[b]
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.vocab)
                self.vocab.append(tok)
        word_freqs = Counter()
        for line in corpus:
            for word in line.split():
                word_bytes = word.encode("utf-8")
                chars = [BYTE_TO_UNICODE[b] for b in word_bytes]
                if not chars: continue
                chars = [WORD_BOUNDARY] + chars
                word_key = " ".join(chars)
                word_freqs[word_key] += 1
        words = {wk: wk.split(" ") for wk in word_freqs}
        target_merges = vocab_size - len(self.vocab)
        if target_merges <= 0:
            self._trained = True
            return {"vocab_size": len(self.vocab), "n_merges": 0}
        for step in range(target_merges):
            pair_counts = Counter()
            for wk, symbols in words.items():
                freq = word_freqs[wk]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i+1])] += freq
            if not pair_counts: break
            best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            self.token_to_id[new_token] = len(self.vocab)
            self.vocab.append(new_token)
            for wk, symbols in words.items():
                if len(symbols) < 2: continue
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == best_pair:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                words[wk] = new_symbols
        self._trained = True
        return {"vocab_size": len(self.vocab), "n_merges": len(self.merges)}

    def _bpe(self, tokens):
        if len(tokens) < 2: return tokens
        while True:
            best_rank = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None: break
            tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx+1]] + tokens[best_idx+2:]
        return tokens

    def encode(self, text, add_bos=False, add_eos=False):
        if not self._trained:
            raise RuntimeError("tokenizer not trained")
        ids = []
        if add_bos: ids.append(self.special.bos_id)
        for word in text.split():
            word_bytes = word.encode("utf-8")
            chars = [BYTE_TO_UNICODE[b] for b in word_bytes]
            if not chars: continue
            chars = [WORD_BOUNDARY] + chars
            merged = self._bpe(chars)
            for tok in merged:
                tid = self.token_to_id.get(tok)
                if tid is None:
                    for ch in tok:
                        tid = self.token_to_id.get(ch, self.special.unk_id)
                        ids.append(tid)
                else:
                    ids.append(tid)
        if add_eos: ids.append(self.special.eos_id)
        return ids

    def decode(self, ids):
        if not self._trained:
            raise RuntimeError("tokenizer not trained")
        chars = []
        for tid in ids:
            if tid < 4: continue
            if tid >= len(self.vocab): continue
            chars.append(self.vocab[tid])
        text = "".join(chars)
        text = text.replace(WORD_BOUNDARY, " ")
        if text.startswith(" "):
            text = text[1:]
        out_bytes = bytearray()
        for ch in text:
            if ch in UNICODE_TO_BYTE:
                out_bytes.append(UNICODE_TO_BYTE[ch])
            else:
                out_bytes.extend(ch.encode("utf-8"))
        return out_bytes.decode("utf-8", errors="replace")

    def vocab_size(self): return len(self.vocab)
    def id_to_token(self, tid):
        if 0 <= tid < len(self.vocab): return self.vocab[tid]
        return None
    def is_trained(self): return self._trained

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "version": self.VERSION,
                "vocab": self.vocab,
                "merges": [[a,b] for a,b in self.merges],
                "special_tokens": {"pad":"<pad>","bos":"<bos>","eos":"<eos>","unk":"<unk>"},
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = list(data["vocab"])
        tok.token_to_id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_ranks = {m: i for i, m in enumerate(tok.merges)}
        tok._trained = True
        return tok

# --- Model config ---

@dataclass
class EINXModelConfig:
    name: str = "einx"
    version: str = "0.1.0"
    arch: str = "decoder-only-transformer"
    vocab_size: int = 1024
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 32
    max_context_length: int = 128
    ffn_dim: int = 512
    dropout: float = 0.1
    positional_encoding: str = "rope"
    norm_type: str = "rms"
    precision: str = "fp32"
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3

    def validate(self):
        assert self.head_dim * self.n_heads == self.hidden_dim
        assert self.vocab_size > 0
        assert self.n_layers > 0

    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

# --- Model layers ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return (x.to(dtype)) * self.weight

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)
    def _build_cache(self, max_seq_len):
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos_cached", torch.cat([cos, cos], dim=-1), persistent=False)
        self.register_buffer("sin_cached", torch.cat([sin, sin], dim=-1), persistent=False)
    def forward(self, seq_len, device, dtype):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (self.cos_cached[:seq_len].to(device=device, dtype=dtype),
                self.sin_cached[:seq_len].to(device=device, dtype=dtype))

def rotate_half(x):
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, head_dim, dropout=0.1, bias=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = dropout
    def forward(self, x, rope=None, mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if rope is not None:
            cos, sin = rope(T, device=x.device, dtype=x.dtype)
            q, k = apply_rotary(q, k, cos, sin)
        if mask is None:
            mask = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(attn)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, bias=False):
        super().__init__()
        intermediate = int(ffn_dim * 2 / 3)
        intermediate = ((intermediate + 7) // 8) * 8
        self.w_up = nn.Linear(hidden_dim, intermediate, bias=bias)
        self.w_gate = nn.Linear(hidden_dim, intermediate, bias=bias)
        self.w_down = nn.Linear(intermediate, hidden_dim, bias=bias)
    def forward(self, x):
        return self.w_down(F.silu(self.w_up(x)) * self.w_gate(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_dim)
        self.attn = MultiHeadAttention(cfg.hidden_dim, cfg.n_heads, cfg.head_dim, cfg.dropout)
        self.norm2 = RMSNorm(cfg.hidden_dim)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.ffn_dim)
    def forward(self, x, rope=None):
        h = self.norm1(x)
        h = self.attn(h, rope=rope)
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        return x + h

# --- Full model ---

class EINXTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.rope = RotaryPositionEmbedding(head_dim=config.head_dim, max_seq_len=config.max_context_length)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_final = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        self._n_params = sum(p.numel() for p in self.parameters())

    @property
    def n_params(self): return self._n_params

    def forward(self, input_ids, targets=None):
        B, T = input_ids.size()
        x = self.token_embedding(input_ids)
        x = self.emb_dropout(x)
        for block in self.blocks:
            x = block(x, rope=self.rope)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=self.config.pad_token_id)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_context_length:]
            logits, _ = self.forward(context)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                continue
            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(next_logits >= threshold, next_logits,
                                          torch.full_like(next_logits, float("-inf")))
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"model_state_dict": self.state_dict(), "config": self.config.to_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = EINXModelConfig.from_dict(ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model

# --- Simple dataset ---

class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, context_length=128):
        self.context_length = context_length
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_eos=True)
            all_ids.extend(ids)
        chunk_size = context_length + 1
        n_chunks = len(all_ids) // chunk_size
        all_ids = all_ids[:n_chunks * chunk_size]
        self.chunks = [torch.tensor(all_ids[i:i+chunk_size], dtype=torch.long)
                       for i in range(0, len(all_ids), chunk_size)]
        if not self.chunks:
            self.chunks = [torch.zeros(chunk_size, dtype=torch.long)]
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]


# ============================================================
# PART 2: Knowledge corpus generator
# ============================================================

CAPITALS = [('france','paris'),('japan','tokyo'),('england','london'),('germany','berlin'),
    ('italy','rome'),('china','beijing'),('russia','moscow'),('india','new delhi'),
    ('brazil','brasilia'),('egypt','cairo'),('canada','ottawa'),('australia','canberra'),
    ('spain','madrid'),('greece','athens'),('portugal','lisbon'),('netherlands','amsterdam'),
    ('sweden','stockholm'),('norway','oslo'),('finland','helsinki'),('denmark','copenhagen'),
    ('poland','warsaw'),('turkey','ankara'),('south korea','seoul'),('mexico','mexico city'),
    ('argentina','buenos aires'),('thailand','bangkok'),('vietnam','hanoi'),('indonesia','jakarta'),
    ('saudi arabia','riyadh'),('iran','tehran'),('switzerland','bern'),('austria','vienna'),
    ('belgium','brussels'),('ireland','dublin')]

SCIENCE = [
    'water boils at 100 degrees celsius.','water freezes at 0 degrees celsius.',
    'the earth orbits the sun.','the moon orbits the earth.','the sun is a star.',
    'gravity pulls objects toward the earth.','a year has 365 days.','a week has 7 days.',
    'a day has 24 hours.','an hour has 60 minutes.','a minute has 60 seconds.',
    'there are 12 months in a year.','january is the first month.','december is the last month.',
    'humans have 206 bones.','the human heart has 4 chambers.','plants make food through photosynthesis.',
    'the largest planet is jupiter.','the smallest planet is mercury.','mars is called the red planet.',
    'venus is the hottest planet.','the pacific ocean is the largest ocean.',
    'mount everest is the tallest mountain.','a triangle has 3 sides.','a square has 4 equal sides.',
    'a pentagon has 5 sides.','a hexagon has 6 sides.','an octagon has 8 sides.',
    'there are 8 planets in the solar system.','the earth has one moon.','mars has two moons.',
    'saturn has rings made of ice and rock.','the human body has 5 senses.',
    'the brain is the control center of the body.','the lungs are used for breathing.',
    'the heart pumps blood through the body.','the skin is the largest organ.',
    'diamond is the hardest natural material.','gold does not rust.',
    'a magnet has a north pole and a south pole.','opposite poles attract.','like poles repel.',
    'iron is heavier than wood.','a whale is bigger than a mouse.','a cheetah is faster than a turtle.']

LOGIC = [
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
    'if today is sunday, tomorrow is monday.','if a + b = 10 and a = 3, then b = 7.',
    'if 2x = 10, then x = 5.','if 3x = 15, then x = 5.','if x + 5 = 12, then x = 7.',
    'if x + 3 = 10, then x = 7.','if x - 4 = 6, then x = 10.',
    'if a shape has 3 sides, it is a triangle.','if a shape has 4 equal sides, it is a square.',
    'if all mammals have hair, and a whale is a mammal, then whales have hair.']

DEFS = [
    'a mammal is an animal that has hair and feeds its young with milk.',
    'a reptile is an animal with scales that lays eggs.','a bird is an animal with feathers and a beak.',
    'a fish is an animal that lives in water and has gills.','an insect is an animal with 6 legs and 3 body parts.',
    'gravity is the force that pulls objects toward each other.','energy is the ability to do work.',
    'a molecule is two or more atoms joined together.','an atom is the smallest unit of matter.',
    'a cell is the basic unit of life.','dna carries genetic information.',
    'a planet orbits a star.','a star is a ball of hot gas that produces light.',
    'temperature measures how hot or cold something is.','mass is the amount of matter in an object.',
    'a solid has a fixed shape.','a liquid takes the shape of its container.','a gas fills its container.',
    'a herbivore eats only plants.','a carnivore eats only meat.','an omnivore eats both plants and meat.']

COMMON = [
    'fire is hot.','ice is cold.','the sky is blue during the day.','the sky is dark at night.',
    'you should drink water when thirsty.','you should eat food when hungry.','you should sleep when tired.',
    'wear warm clothes in winter.','wear light clothes in summer.','look both ways before crossing the street.',
    'wash your hands before eating.','rain makes the ground wet.','sun makes things warm.',
    'wind can blow things away.','snow is cold and white.','a knife is sharp.','a pillow is soft.',
    'a rock is hard.','a feather is light.','you need air to breathe.','you need water to live.',
    'you need food to live.','plants need sunlight to grow.','fish live in water.','birds live in trees.',
    'humans walk on two legs.','dogs walk on four legs.','spiders have eight legs.','insects have six legs.',
    'the sun gives us light and heat.','the moon shines at night.']

CAUSE = [
    'if you heat ice, it melts into water.','if you heat water, it boils and becomes steam.',
    'if you cool water, it freezes into ice.','if you drop something, gravity pulls it down.',
    'if you mix red and blue paint, you get purple.','if you mix yellow and blue paint, you get green.',
    'if you mix red and yellow paint, you get orange.','if you touch something hot, you will burn your hand.',
    'if you do not eat, you will lose weight.','if you exercise, your muscles get stronger.',
    'if you study, you learn.','if you do not drink water, you become thirsty.',
    'if the sun shines, things become warm.','if wind blows, leaves move.',
    'if you leave metal in water, it may rust.','if you plant a seed and water it, it grows.']

REASON = [
    'problem: if you have 5 apples and eat 2, how many are left? solution: 5 - 2 = 3. answer: 3 apples.',
    'problem: if you have 10 dollars and buy a toy for 3 dollars, how much is left? solution: 10 - 3 = 7. answer: 7 dollars.',
    'problem: if you have 2 red balls and 3 blue balls, how many balls? solution: 2 + 3 = 5. answer: 5 balls.',
    'problem: if each box holds 6 eggs and you have 4 boxes, how many eggs? solution: 6 x 4 = 24. answer: 24 eggs.',
    'problem: if a pizza has 8 slices and you eat 3, how many left? solution: 8 - 3 = 5. answer: 5 slices.',
    'problem: if you read 10 pages a day for 5 days, how many pages? solution: 10 x 5 = 50. answer: 50 pages.',
    'problem: if you have 20 dollars and pencils cost 2 dollars each, how many pencils? solution: 20 / 2 = 10. answer: 10 pencils.',
    'problem: if a is 5 and b is 3, what is a + b? solution: 5 + 3 = 8. answer: 8.',
    'problem: if a is 5 and b is 3, what is a x b? solution: 5 x 3 = 15. answer: 15.',
    'problem: if 3 people share 12 cookies equally, how many each? solution: 12 / 3 = 4. answer: 4 cookies.',
    'problem: if today is wednesday, what day was yesterday? answer: tuesday.',
    'problem: if today is thursday, what day is tomorrow? answer: friday.']

QA = [
    'question: how many days are in a week? answer: 7.',
    'question: how many months are in a year? answer: 12.',
    'question: what color is the sky? answer: blue.',
    'question: what color is grass? answer: green.',
    'question: what color is blood? answer: red.',
    'question: what color is snow? answer: white.',
    'question: what animal says meow? answer: a cat.',
    'question: what animal says woof? answer: a dog.',
    'question: what animal says moo? answer: a cow.',
    'question: what animal says quack? answer: a duck.',
    'question: what planet do we live on? answer: earth.',
    'question: what is the closest star to earth? answer: the sun.',
    'question: how many planets are in the solar system? answer: 8.',
    'question: what is the largest planet? answer: jupiter.',
    'question: what is the smallest planet? answer: mercury.',
    'question: what is the boiling point of water? answer: 100 degrees celsius.',
    'question: what is the freezing point of water? answer: 0 degrees celsius.',
    'question: how many legs does a spider have? answer: 8.',
    'question: how many legs does an insect have? answer: 6.',
    'question: how many legs does a dog have? answer: 4.',
    'question: what is the opposite of hot? answer: cold.',
    'question: what is the opposite of up? answer: down.',
    'question: what is the opposite of big? answer: small.',
    'question: what is the opposite of fast? answer: slow.',
    'question: what is the opposite of light? answer: dark.',
    'question: what is the opposite of good? answer: bad.',
    'question: what is the opposite of day? answer: night.',
    'question: what is the opposite of wet? answer: dry.',
    'question: what comes after monday? answer: tuesday.',
    'question: what comes after friday? answer: saturday.',
    'question: what comes after december? answer: january.',
    'question: how many bones does a human have? answer: 206.',
    'question: what is the largest ocean? answer: the pacific ocean.',
    'question: what is the tallest mountain? answer: mount everest.']


def gen_corpus(n=30000, seed=42):
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        r = rng.random()
        if r < 0.15:
            a, b = rng.randint(1,12), rng.randint(1,12)
            op = rng.choice(['+','-','x'])
            if op == '+': records.append({'text': f'{a} + {b} = {a+b}.'})
            elif op == '-': a,b=max(a,b),min(a,b); records.append({'text': f'{a} - {b} = {a-b}.'})
            else: records.append({'text': f'{a} x {b} = {a*b}.'})
        elif r < 0.30:
            if rng.random() < 0.5: c,cap=rng.choice(CAPITALS); records.append({'text': f'the capital of {c} is {cap}.'})
            else: records.append({'text': rng.choice(SCIENCE)})
        elif r < 0.40: records.append({'text': rng.choice(LOGIC)})
        elif r < 0.50: records.append({'text': rng.choice(DEFS)})
        elif r < 0.60: records.append({'text': rng.choice(COMMON)})
        elif r < 0.70: records.append({'text': rng.choice(CAUSE)})
        elif r < 0.78: records.append({'text': rng.choice(REASON)})
        else:
            if rng.random() < 0.4: c,cap=rng.choice(CAPITALS); records.append({'text': f'question: what is the capital of {c}? answer: {cap}.'})
            elif rng.random() < 0.6: a,b=rng.randint(1,12),rng.randint(1,12); records.append({'text': f'question: what is {a} + {b}? answer: {a+b}.'})
            else: records.append({'text': rng.choice(QA)})
    return records


# ============================================================
# PART 3: Train + Test
# ============================================================

def main():
    print("=" * 60)
    print("EINX KNOWLEDGE MODEL — TRAIN + TEST ON GPU")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Generate corpus
    print("\n1. Generating knowledge corpus...")
    records = gen_corpus(30000, seed=42)
    texts = [r['text'] for r in records]
    unique = set(texts)
    print(f"   {len(records):,} records ({len(unique):,} unique facts)")

    # 2. Train tokenizer
    print("\n2. Training tokenizer...")
    tok = BPETokenizer()
    tok.train(texts, vocab_size=1024, verbose=False)
    print(f"   Vocab: {tok.vocab_size()}, Merges: {len(tok.merges)}")

    # 3. Build dataset (keep duplicates for memorization)
    print("\n3. Building dataset...")
    rng = random.Random(42)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.05))
    val_recs = shuffled[:n_val]
    train_recs = shuffled[n_val:]
    train_texts = [r['text'] for r in train_recs]
    val_texts = [r['text'] for r in val_recs]

    train_ds = SimpleDataset(train_texts, tok, context_length=128)
    val_ds = SimpleDataset(val_texts, tok, context_length=128)
    print(f"   Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # 4. Build model
    print("\n4. Building model...")
    model_cfg = EINXModelConfig(
        name='einx-knowledge-gpu', vocab_size=tok.vocab_size(),
        hidden_dim=192, n_layers=6, n_heads=6, head_dim=32,
        max_context_length=128, ffn_dim=768, dropout=0.1,
        positional_encoding='rope', norm_type='rms',
        precision='fp32', tie_word_embeddings=True,
    )
    model = EINXTransformer(model_cfg).to(device)
    print(f"   Parameters: {model.n_params:,}")
    print(f"   Architecture: {model_cfg.n_layers}L {model_cfg.hidden_dim}D {model_cfg.n_heads}H")

    # 5. Train
    print(f"\n5. Training 2000 steps on {device}...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=0.0001)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)

    max_steps = 2000
    log_every = 100
    eval_every = 200
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    start_time = time.time()
    step = 0
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
                avg_loss = sum(train_losses[-log_every:]) / log_every
                elapsed = time.time() - start_time
                print(f"   step {step:5d}/{max_steps}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)")

            if step % eval_every == 0:
                model.eval()
                total_val_loss = 0
                n_val_batches = 0
                with torch.no_grad():
                    for vb in val_loader:
                        if n_val_batches >= 20: break
                        vi, vt = vb[0].to(device), vb[1].to(device)
                        _, vl = model(vi, targets=vt)
                        total_val_loss += vl.item()
                        n_val_batches += 1
                avg_val = total_val_loss / max(1, n_val_batches)
                val_losses.append((step, avg_val))
                ppl = math.exp(avg_val) if avg_val < 20 else float('inf')
                print(f"   >>> eval step {step}: val_loss={avg_val:.4f}  perplexity={ppl:.2f}")
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    # Save best model
                    os.makedirs('/content/checkpoints', exist_ok=True)
                    model.save('/content/checkpoints/best_model.pt')

    elapsed = time.time() - start_time
    initial_loss = train_losses[0] if train_losses else 0
    final_loss = train_losses[-1] if train_losses else 0
    final_val = val_losses[-1][1] if val_losses else 0
    ppl = math.exp(final_val) if final_val < 20 else float('inf')

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Steps:      {step}")
    print(f"Init loss:  {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Val loss:   {final_val:.4f}")
    print(f"Best val:   {best_val_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")

    # 6. Test with real questions
    print()
    print("=" * 60)
    print("KNOWLEDGE TEST")
    print("=" * 60)

    # Load best model
    model = EINXTransformer.load('/content/checkpoints/best_model.pt', map_location=str(device))
    model.to(device)
    model.eval()

    tests = [
        ('question: what is 2 + 2? answer:', '4'),
        ('question: what is 5 + 3? answer:', '8'),
        ('question: what is 7 + 8? answer:', '15'),
        ('question: what is 9 x 9? answer:', '81'),
        ('question: what is 6 x 7? answer:', '42'),
        ('question: what is 10 - 3? answer:', '7'),
        ('question: what is the capital of france? answer:', 'paris'),
        ('question: what is the capital of japan? answer:', 'tokyo'),
        ('question: what is the capital of england? answer:', 'london'),
        ('question: what is the capital of germany? answer:', 'berlin'),
        ('question: what is the capital of italy? answer:', 'rome'),
        ('question: what is the capital of china? answer:', 'beijing'),
        ('question: what planet do we live on? answer:', 'earth'),
        ('question: what is the closest star to earth? answer:', 'sun'),
        ('question: how many days are in a week? answer:', '7'),
        ('question: how many months are in a year? answer:', '12'),
        ('question: what color is the sky? answer:', 'blue'),
        ('question: what color is grass? answer:', 'green'),
        ('question: what color is blood? answer:', 'red'),
        ('question: what animal says meow? answer:', 'cat'),
        ('question: what animal says woof? answer:', 'dog'),
        ('question: what animal says moo? answer:', 'cow'),
        ('question: how many legs does a spider have? answer:', '8'),
        ('question: how many legs does an insect have? answer:', '6'),
        ('question: how many planets are in the solar system? answer:', '8'),
        ('question: what is the largest planet? answer:', 'jupiter'),
        ('question: what is the opposite of hot? answer:', 'cold'),
        ('question: what is the opposite of up? answer:', 'down'),
        ('question: what is the opposite of big? answer:', 'small'),
        ('question: what comes after monday? answer:', 'tuesday'),
        ('question: what comes after friday? answer:', 'saturday'),
        ('if today is monday, tomorrow is', 'tuesday'),
        ('if today is friday, tomorrow is', 'saturday'),
        ('if 2x = 10, then x =', '5'),
        ('if x + 5 = 12, then x =', '7'),
    ]

    correct = 0
    total = len(tests)
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
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
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

    # Save results
    results = {
        'model': 'EINX-Knowledge-GPU',
        'parameters': model.n_params,
        'training_steps': step,
        'initial_loss': round(initial_loss, 4),
        'final_loss': round(final_loss, 4),
        'val_loss': round(final_val, 4),
        'perplexity': round(ppl, 2),
        'score': f'{correct}/{total}',
        'percentage': round(correct/total*100, 1),
        'device': str(device),
    }
    os.makedirs('/content', exist_ok=True)
    with open('/content/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /content/results.json")
    print("\nDone! Copy the results above and send them back.")


if __name__ == '__main__':
    main()
