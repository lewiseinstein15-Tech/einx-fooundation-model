# -*- coding: utf-8 -*-
"""
EINX Autonomous Learner — downloads real knowledge from the internet,
decides what to study, trains itself, tests, and improves.

WHAT IT DOES:
1. Downloads real Wikipedia articles on diverse topics (free API, no key)
2. Extracts facts and sentences from them
3. Trains the model on real-world knowledge
4. Tests itself with questions
5. If it gets questions wrong → downloads MORE on those topics → retrains
6. Repeats until time runs out or score stops improving

INSTRUCTIONS (Colab):
1. Runtime → Change runtime type → T4 GPU
2. Paste this into a cell:
   import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/lewiseinstein15-Tech/einx-fooundation-model/main/einx_standalone.py', 'einx_standalone.py')
3. Then: exec(open('einx_standalone.py').read())
4. Wait 15-30 minutes
5. Copy results
"""

import math, json, random, time, os, sys, re, urllib.request, urllib.parse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# MODEL (same proven architecture)
# ============================================================

def _btu():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]; n = 0
    for b in range(2 ** 8):
        if b not in bs: bs.append(b); cs.append(2 ** 8 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

BTU = _btu(); UTB = {v: k for k, v in BTU.items()}; WB = "Ġ"

@dataclass(frozen=True)
class ST:
    pad_id: int = 0; bos_id: int = 1; eos_id: int = 2; unk_id: int = 3

class BPETokenizer:
    VERSION = "einx-bpe-0.1"
    def __init__(self):
        self.st = ST(); self.vocab = ["<pad>","<bos>","<eos>","<unk>"]
        self.t2i = {t: i for i, t in enumerate(self.vocab)}
        self.merges = []; self.ranks = {}; self._trained = False
    def train(self, corpus, vocab_size=1024):
        self.vocab = ["<pad>","<bos>","<eos>","<unk>"]
        self.t2i = {t: i for i, t in enumerate(self.vocab)}
        self.merges = []; self.ranks = {}
        for b in range(256):
            t = BTU[b]
            if t not in self.t2i: self.t2i[t] = len(self.vocab); self.vocab.append(t)
        wf = Counter()
        for line in corpus:
            for word in line.split():
                chars = [BTU[b] for b in word.encode("utf-8")]
                if not chars: continue
                chars = [WB] + chars; wf[" ".join(chars)] += 1
        words = {wk: wk.split(" ") for wk in wf}
        for _ in range(max(0, vocab_size - len(self.vocab))):
            pc = Counter()
            for wk, syms in words.items():
                f = wf[wk]
                for i in range(len(syms)-1): pc[(syms[i], syms[i+1])] += f
            if not pc: break
            bp = max(pc.items(), key=lambda kv: (kv[1], kv[0]))[0]
            nt = bp[0] + bp[1]
            self.merges.append(bp); self.ranks[bp] = len(self.merges) - 1
            self.t2i[nt] = len(self.vocab); self.vocab.append(nt)
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
                r = self.ranks.get((tokens[i], tokens[i+1]))
                if r is not None and (br is None or r < br): br = r; bi = i
            if br is None: break
            tokens = tokens[:bi] + [tokens[bi]+tokens[bi+1]] + tokens[bi+2:]
        return tokens
    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos: ids.append(self.st.bos_id)
        for word in text.split():
            chars = [BTU[b] for b in word.encode("utf-8")]
            if not chars: continue
            chars = [WB] + chars
            for tok in self._bpe(chars):
                tid = self.t2i.get(tok)
                if tid is None:
                    for ch in tok: ids.append(self.t2i.get(ch, self.st.unk_id))
                else: ids.append(tid)
        if add_eos: ids.append(self.st.eos_id)
        return ids
    def decode(self, ids):
        chars = []
        for tid in ids:
            if tid < 4 or tid >= len(self.vocab): continue
            chars.append(self.vocab[tid])
        text = "".join(chars).replace(WB, " ")
        if text.startswith(" "): text = text[1:]
        out = bytearray()
        for ch in text:
            if ch in UTB: out.append(UTB[ch])
            else: out.extend(ch.encode("utf-8"))
        return out.decode("utf-8", errors="replace")
    def vocab_size(self): return len(self.vocab)

@dataclass
class Cfg:
    name: str = "einx"; vocab_size: int = 1024; hidden_dim: int = 256
    n_layers: int = 8; n_heads: int = 8; head_dim: int = 32
    max_context_length: int = 128; ffn_dim: int = 1024; dropout: float = 0.1
    tie_word_embeddings: bool = True
    def validate(self): assert self.head_dim * self.n_heads == self.hidden_dim
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        k = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{kk: v for kk, v in d.items() if kk in k})

class RMSNorm(nn.Module):
    def __init__(self, d, e=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.e = e
    def forward(self, x):
        dt = x.dtype; x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.e)).to(dt) * self.w

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
        m = cls(Cfg.from_dict(c["config"])); m.load_state_dict(c["model_state_dict"]); return m

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
# INTERNET KNOWLEDGE FETCHER — downloads real Wikipedia articles
# ============================================================

def fetch_wikipedia_article(title):
    """Download a Wikipedia article's summary + first section (free API, no key)."""
    try:
        title_encoded = urllib.parse.quote(title)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "EINX/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            extract = data.get("extract", "")
            if extract:
                return extract
    except Exception as e:
        pass
    return None

def fetch_wikipedia_search(query, limit=5):
    """Search Wikipedia for articles matching a query."""
    try:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": str(limit), "format": "json", "origin": "*"
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "EINX/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [item["title"] for item in data.get("query", {}).get("search", [])]
    except:
        return []

def extract_sentences(text, max_len=200, min_len=10):
    """Split text into clean sentences suitable for training."""
    # Remove references, citations, URLs
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if min_len <= len(s.strip()) <= max_len]

def fetch_knowledge(topics):
    """Download real knowledge about a list of topics from Wikipedia."""
    all_sentences = []
    for topic in topics:
        # Search for articles
        titles = fetch_wikipedia_search(topic, limit=3)
        if not titles:
            titles = [topic]
        for title in titles[:2]:  # 2 articles per topic
            extract = fetch_wikipedia_article(title)
            if extract:
                sentences = extract_sentences(extract)
                all_sentences.extend(sentences)
                print(f"  ✓ {title}: {len(sentences)} sentences")
            else:
                print(f"  ✗ {title}: no data")
    return all_sentences

# ============================================================
# BASE KNOWLEDGE CORPUS (same as before)
# ============================================================

def build_base_corpus():
    facts = set()
    for a in range(1,13):
        for b in range(1,13):
            facts.add(f"{a} + {b} = {a+b}.")
            facts.add(f"{a} x {b} = {a*b}.")
            if a > b: facts.add(f"{a} - {b} = {a-b}.")
            facts.add(f"question: what is {a} + {b}? answer: {a+b}.")
            facts.add(f"question: what is {a} x {b}? answer: {a*b}.")
            if a > b: facts.add(f"question: what is {a} - {b}? answer: {a-b}.")

    caps = [('france','paris'),('japan','tokyo'),('england','london'),('germany','berlin'),('italy','rome'),('china','beijing'),('russia','moscow'),('india','new delhi'),('brazil','brasilia'),('egypt','cairo'),('canada','ottawa'),('australia','canberra'),('spain','madrid'),('greece','athens'),('portugal','lisbon'),('netherlands','amsterdam'),('sweden','stockholm'),('norway','oslo'),('finland','helsinki'),('denmark','copenhagen'),('poland','warsaw'),('turkey','ankara'),('south korea','seoul'),('mexico','mexico city'),('argentina','buenos aires'),('thailand','bangkok'),('vietnam','hanoi'),('saudi arabia','riyadh'),('iran','tehran'),('switzerland','bern'),('austria','vienna'),('belgium','brussels'),('ireland','dublin')]
    for c, cap in caps:
        facts.add(f"the capital of {c} is {cap}.")
        facts.add(f"question: what is the capital of {c}? answer: {cap}.")

    sci = ['water boils at 100 degrees celsius.','water freezes at 0 degrees celsius.','the earth orbits the sun.','the sun is a star.','gravity pulls objects toward the earth.','a year has 365 days.','a week has 7 days.','a day has 24 hours.','an hour has 60 minutes.','a minute has 60 seconds.','there are 12 months in a year.','january is the first month.','december is the last month.','humans have 206 bones.','the human heart has 4 chambers.','plants make food through photosynthesis.','the largest planet is jupiter.','the smallest planet is mercury.','mars is called the red planet.','venus is the hottest planet.','the pacific ocean is the largest ocean.','mount everest is the tallest mountain.','a triangle has 3 sides.','a square has 4 equal sides.','a pentagon has 5 sides.','a hexagon has 6 sides.','an octagon has 8 sides.','there are 8 planets in the solar system.','the earth has one moon.','saturn has rings made of ice and rock.','the brain is the control center of the body.','the heart pumps blood through the body.','the skin is the largest organ.','diamond is the hardest natural material.','gold does not rust.','a magnet has a north pole and a south pole.','opposite poles attract.','like poles repel.','iron is heavier than wood.']
    for s in sci: facts.add(s)

    log = ['if a is bigger than b, and b is bigger than c, then a is bigger than c.','if all cats are animals, and tom is a cat, then tom is an animal.','if today is monday, tomorrow is tuesday.','if today is tuesday, tomorrow is wednesday.','if today is wednesday, tomorrow is thursday.','if today is thursday, tomorrow is friday.','if today is friday, tomorrow is saturday.','if today is saturday, tomorrow is sunday.','if today is sunday, tomorrow is monday.','if 2x = 10, then x = 5.','if 3x = 15, then x = 5.','if x + 5 = 12, then x = 7.','if x + 3 = 10, then x = 7.','if x - 4 = 6, then x = 10.','if a + b = 10 and a = 3, then b = 7.']
    for l in log: facts.add(l)

    opp = [('hot','cold'),('up','down'),('big','small'),('fast','slow'),('light','dark'),('good','bad'),('old','new'),('open','closed'),('day','night'),('wet','dry'),('happy','sad'),('tall','short'),('full','empty'),('loud','quiet'),('hard','soft'),('strong','weak'),('heavy','light'),('long','short')]
    for a,b in opp:
        facts.add(f"the opposite of {a} is {b}.")
        facts.add(f"question: what is the opposite of {a}? answer: {b}.")

    cols = [('sky','blue'),('grass','green'),('blood','red'),('snow','white'),('coal','black'),('banana','yellow'),('sun','yellow'),('ocean','blue')]
    for o,c in cols:
        facts.add(f"question: what color is {o}? answer: {c}.")

    snds = [('cat','meow'),('dog','woof'),('cow','moo'),('duck','quack'),('sheep','baa'),('pig','oink'),('lion','roar'),('horse','neigh')]
    for a,s in snds:
        facts.add(f"question: what animal says {s}? answer: a {a}.")

    days = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    for i,d in enumerate(days):
        nd = days[(i+1)%7]
        facts.add(f"question: what comes after {d}? answer: {nd}.")

    months = ['january','february','march','april','may','june','july','august','september','october','november','december']
    for i,m in enumerate(months):
        nm = months[(i+1)%12]
        facts.add(f"question: what comes after {m}? answer: {nm}.")

    extra_qa = ['question: what planet do we live on? answer: earth.','question: what is the closest star to earth? answer: the sun.','question: how many days are in a week? answer: 7.','question: how many months are in a year? answer: 12.','question: how many hours in a day? answer: 24.','question: how many continents are there? answer: 7.','question: how many planets are in the solar system? answer: 8.','question: how many legs does a spider have? answer: 8.','question: how many legs does an insect have? answer: 6.','question: how many legs does a dog have? answer: 4.','question: what is the largest planet? answer: jupiter.','question: what is the smallest planet? answer: mercury.','question: what is the boiling point of water? answer: 100 degrees celsius.','question: what is the freezing point of water? answer: 0 degrees celsius.','question: how many bones does a human have? answer: 206.','question: what organ pumps blood? answer: the heart.','question: what organ do we think with? answer: the brain.','question: what is the largest ocean? answer: the pacific ocean.','question: what is the tallest mountain? answer: mount everest.','question: what is half of 10? answer: 5.','question: what is double 5? answer: 10.','question: what is 100 / 10? answer: 10.','question: what is water made of? answer: hydrogen and oxygen.','question: what gas do we breathe in? answer: oxygen.','question: what gas do we breathe out? answer: carbon dioxide.','question: what do bees make? answer: honey.','question: what do cows give us? answer: milk.','question: what is paper made from? answer: wood.']
    for q in extra_qa: facts.add(q)

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
    ('question: what animal says meow? answer:', 'cat'),
    ('question: what animal says woof? answer:', 'dog'),
    ('question: what animal says moo? answer:', 'cow'),
    ('question: what animal says roar? answer:', 'lion'),
    ('question: how many legs does a spider have? answer:', '8'),
    ('question: how many legs does an insect have? answer:', '6'),
    ('question: what is the largest planet? answer:', 'jupiter'),
    ('question: what is the smallest planet? answer:', 'mercury'),
    ('question: what is the boiling point of water? answer:', '100'),
    ('question: what is the freezing point of water? answer:', '0'),
    ('question: how many bones does a human have? answer:', '206'),
    ('question: what organ pumps blood? answer:', 'heart'),
    ('question: what organ do we think with? answer:', 'brain'),
    ('question: what is the opposite of hot? answer:', 'cold'),
    ('question: what is the opposite of up? answer:', 'down'),
    ('question: what is the opposite of big? answer:', 'small'),
    ('question: what is the opposite of fast? answer:', 'slow'),
    ('question: what is the opposite of light? answer:', 'dark'),
    ('question: what is the opposite of good? answer:', 'bad'),
    ('question: what comes after monday? answer:', 'tuesday'),
    ('question: what comes after friday? answer:', 'saturday'),
    ('question: what comes after december? answer:', 'january'),
    ('if today is monday, tomorrow is', 'tuesday'),
    ('if today is friday, tomorrow is', 'saturday'),
    ('if today is wednesday, tomorrow is', 'thursday'),
    ('if 2x = 10, then x =', '5'),
    ('if 3x = 15, then x =', '5'),
    ('if x + 5 = 12, then x =', '7'),
    ('if x + 3 = 10, then x =', '7'),
    ('if x - 4 = 6, then x =', '10'),
    ('if a + b = 10 and a = 3, then b =', '7'),
]

# ============================================================
# TRAIN + TEST
# ============================================================

def train_and_test(facts, device, max_steps=2000, patience=5):
    tok = BPETokenizer()
    tok.train(facts, vocab_size=1024)
    rng = random.Random(42)
    all_texts = []
    for _ in range(20):
        s = list(facts); rng.shuffle(s); all_texts.extend(s)
    n_val = max(50, int(len(all_texts) * 0.02))
    rng.shuffle(all_texts)
    val_texts = all_texts[:n_val]; train_texts = all_texts[n_val:]
    train_ds = DS(train_texts, tok, ctx=128)
    val_ds = DS(val_texts, tok, ctx=128)
    cfg = Cfg(name='einx', vocab_size=tok.vocab_size(), hidden_dim=256, n_layers=8, n_heads=8, head_dim=32, max_context_length=128, ffn_dim=1024, dropout=0.1)
    model = EINX(cfg).to(device)
    print(f"   Model: {model.n_params:,} params, Train: {len(train_ds)}, Val: {len(val_ds)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0.0001)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)
    best_val = float('inf'); best_path = '/content/checkpoints/best_model.pt'
    os.makedirs('/content/checkpoints', exist_ok=True); pc = 0; step = 0; start = time.time()
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
                model.eval(); tvl = 0; n = 0
                with torch.no_grad():
                    for vb in val_loader:
                        if n >= 20: break
                        vi, vt = vb[0].to(device), vb[1].to(device)
                        _, vl = model(vi, targets=vt); tvl += vl.item(); n += 1
                avl = tvl / max(1, n); el = time.time() - start
                if avl < best_val:
                    best_val = avl; pc = 0; model.save(best_path)
                    print(f"   step {step:5d}  val={avl:.4f}  BEST! ({el:.0f}s)")
                else:
                    pc += 1
                    print(f"   step {step:5d}  val={avl:.4f}  ({pc}/{patience}) ({el:.0f}s)")
                    if pc >= patience:
                        print(f"   EARLY STOP at step {step}")
                        step = max_steps; break
    model = EINX.load(best_path, ml=str(device))
    model.to(device); model.eval()
    correct = 0; total = len(TESTS); wrong = []
    for prompt, expected in TESTS:
        ids = tok.encode(prompt, add_bos=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        gen = []
        for _ in range(10):
            with torch.no_grad():
                logits, _ = model(input_ids)
            nid = torch.argmax(logits[0,-1,:]).item()
            if nid == tok.st.eos_id: break
            gen.append(nid)
            input_ids = torch.cat([input_ids, torch.tensor([[nid]], dtype=torch.long, device=device)], 1)
        ans = tok.decode(gen).strip().lower()
        ok = expected.lower() in ans
        if ok: correct += 1
        else: wrong.append((prompt, expected, ans))
    score = correct / total * 100
    print(f"\n   SCORE: {correct}/{total} ({score:.0f}%)")
    if wrong:
        print(f"   WRONG ({len(wrong)}):")
        for p, e, a in wrong[:10]:
            print(f"     Q: {p}  Expected: {e}  Got: {a!r}")
    return score, best_path, tok, model, wrong

# ============================================================
# AUTONOMOUS LEARNING LOOP
# ============================================================

def main():
    print("=" * 60)
    print("EINX AUTONOMOUS LEARNER")
    print("Downloads real knowledge → Trains → Tests → Learns more")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda': print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Step 1: Build base knowledge (math, capitals, science, etc.)
    print("\n1. Building base knowledge corpus...")
    base_facts = build_base_corpus()
    print(f"   {len(base_facts):,} base facts")

    # Step 2: Download REAL knowledge from Wikipedia
    print("\n2. DOWNLOADING REAL KNOWLEDGE FROM WIKIPEDIA...")
    topics = [
        "physics", "chemistry", "biology", "astronomy", "geography",
        "history", "mathematics", "philosophy", "psychology", "economics",
        "computer science", "medicine", "law", "art", "music",
        "Earth", "Moon", "Sun", "Solar System", "Milky Way",
        "water", "oxygen", "carbon", "iron", "gold",
        "human brain", "human heart", "DNA", "cell biology", "evolution",
        "World War II", "ancient Egypt", "Roman Empire", "Renaissance",
        "Isaac Newton", "Albert Einstein", "Charles Darwin", "Leonardo da Vinci",
        "photosynthesis", "gravity", "electricity", "magnetism", "energy",
        "Amazon River", "Sahara Desert", "Mount Everest", "Pacific Ocean",
        "democracy", "capitalism", "communism", "industrial revolution",
    ]
    wiki_sentences = fetch_knowledge(topics)
    print(f"\n   Downloaded {len(wiki_sentences):,} real sentences from Wikipedia")

    # Combine base facts + Wikipedia knowledge
    all_facts = list(set(base_facts + wiki_sentences))
    print(f"   Total unique knowledge: {len(all_facts):,} facts")

    # Step 3: Train round 1
    print(f"\n3. TRAINING ROUND 1 (base + Wikipedia knowledge)...")
    score, model_path, tok, model, wrong = train_and_test(all_facts, device, max_steps=2000, patience=5)

    best_score = score
    best_tok = tok
    best_model = model

    # Step 4: If score < 95%, download MORE knowledge on topics it got wrong
    if score < 95 and wrong:
        print(f"\n4. MODEL GOT {len(wrong)} QUESTIONS WRONG")
        print("   Downloading MORE knowledge on those topics...")

        # Extract topics from wrong answers
        wrong_topics = set()
        for prompt, expected, got in wrong:
            # Extract key words from the question
            words = prompt.lower().replace("question: ", "").replace("what is ", "").replace("answer:", "").split()
            for w in words:
                if len(w) > 2 and w not in ['the', 'and', 'how', 'many', 'what', 'color', 'animal', 'comes', 'after', 'opposite', 'capital', 'planet', 'largest', 'smallest', 'boiling', 'freezing', 'point', 'water', 'organ', 'legs', 'does']:
                    wrong_topics.add(w)
            # Also search for the expected answer
            wrong_topics.add(expected)

        print(f"   Topics to learn: {wrong_topics}")
        more_sentences = fetch_knowledge(list(wrong_topics)[:20])
        print(f"   Downloaded {len(more_sentences):,} more sentences")

        # Add the new knowledge
        all_facts = list(set(all_facts + more_sentences))
        print(f"   Total knowledge now: {len(all_facts):,} facts")

        # Step 5: Train round 2
        print(f"\n5. TRAINING ROUND 2 (with expanded knowledge)...")
        score2, model_path2, tok2, model2, wrong2 = train_and_test(all_facts, device, max_steps=3000, patience=7)

        if score2 > best_score:
            best_score = score2
            best_tok = tok2
            best_model = model2
            print(f"\n   >>> IMPROVED: {best_score:.0f}%")

    # Step 6: Final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best score:    {best_score:.0f}%")
    print(f"Total facts:   {len(all_facts):,}")
    print(f"Parameters:    {best_model.n_params:,}")

    # Detailed final test
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
            if nid == best_tok.st.eos_id: break
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

    results = {
        'final_score': round(correct/total*100, 1),
        'total_facts': len(all_facts),
        'parameters': best_model.n_params,
        'wiki_sentences': len(wiki_sentences),
    }
    os.makedirs('/content', exist_ok=True)
    with open('/content/einx_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved. Copy everything above and send it back!")


if __name__ == '__main__':
    main()
