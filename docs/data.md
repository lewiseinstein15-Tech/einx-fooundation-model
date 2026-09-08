# EINX Data Pipeline (Build 3)

## Pipeline architecture (spec §2)

```
RAW DATA (.txt / .jsonl / .json)
   ↓
INGESTION
   ↓
VALIDATION  ← produces ValidationReport (every count is real)
   ↓
CLEANING     ← Unicode norm, whitespace norm, control-char removal, empty removal, repetition detection
   ↓
DEDUPLICATION ← exact + normalized hash (NOT semantic — documented)
   ↓
FILTERING
   ↓
NORMALIZATION
   ↓
DATASET VERSION  ← manifest.json with immutable identity hash
   ↓
TOKENIZATION   ← uses EINX BPE tokenizer; version recorded
   ↓
SEQUENCE PACKING ← context_length chunks with EOS separators
   ↓
SHARDS         ← shard-NNNNN.jsonl (streaming, no full-RAM load)
   ↓
TRAINING DATALOADER
   ↓
EINX
```

## Supported formats (spec §3)

- **.jsonl** (default, recommended) — one JSON object per line with a configurable `text` field
- **.json** — single JSON array of objects
- **.txt** — plain text (one document per line, or one per file)

Configurable text field:

```yaml
data:
  text_field: text    # default; override for {"content": "..."} or {"body": "..."}
```

## Data validation (spec §4)

```bash
einx data validate --input data/raw/train.jsonl --text-field text --report report.json
```

Output:
```
EINX DATA VALIDATION
────────────────────────────────────────
Files scanned:    1
Records scanned:  35

Valid:             32
Empty:             1
Missing text:      0
Malformed JSON:    1
Invalid Unicode:   0
Too short:         0
Too long:          0
Duplicates:        0
Metadata issues:   0

Final usable:      32
────────────────────────────────────────
```

Every number comes from actually scanning the data — never fabricated.

## Data cleaning (spec §5)

```bash
einx data clean --input raw.jsonl --output clean.jsonl --unicode-norm NFC
```

Configurable operations (all opt-in, all disable-able):
- Unicode normalization (NFC / NFKC / NFD / NFKD)
- Whitespace normalization (collapse runs, strip leading/trailing)
- Control character removal (keeps \n + \t)
- Empty document removal
- Excessive repetition detection (configurable threshold)
- Malformed document removal

## Deduplication (spec §6)

```bash
einx data dedupe --input clean.jsonl --output deduped.jsonl --normalization whitespace
```

Three modes:
- `none` — exact match only
- `whitespace` — normalize whitespace before hashing (default)
- `lowercase` — normalize whitespace + lowercase before hashing

**Honest note (spec §6):** This is exact + normalized exact deduplication.
Semantic / near-duplicate detection is PLANNED (the `NearDuplicateDetector`
abstraction exists but raises `NotImplementedError`).

## Dataset versioning + manifest (spec §8, §9)

Every processed dataset gets a `manifest.json` with an immutable identity:

```json
{
  "name": "einx-training-demo",
  "version": "0.1.0",
  "source_hash": "a1b2c3...",
  "processing_hash": "d4e5f6...",
  "tokenizer_version": "einx-bpe-0.1",
  "n_records": 10000,
  "n_tokens": 850000,
  "n_train_records": 9500,
  "n_val_records": 500,
  "shard_count": 2,
  "context_length": 256,
  "seed": 42,
  "identity_hash": "g7h8i9...",
  "created_at": "2026-09-08T...",
  "einx_version": "0.3.0"
}
```

The `identity_hash` is computed from `source_hash + processing_hash +
tokenizer_version + n_records + n_tokens + context_length`. If ANY of
those change, the identity changes (spec §8).

## Tokenization pipeline (spec §10)

The pipeline connects directly to the EINX BPE tokenizer. The tokenizer
version is recorded in the manifest — a training run must never silently
use a different tokenizer than the one in the manifest.

## Token statistics (spec §11)

```bash
einx data tokenize --input clean.jsonl --tokenizer tokenizer.json
```

```
EINX TOKEN STATISTICS
────────────────────────────────────────
Documents:         30
Total tokens:      261
Average tokens/doc: 8.7
Min tokens:         5
Max tokens:         10
Median tokens:      9.0
P25:                8.0
P75:                10.0
P95:                10.0
────────────────────────────────────────
```

## Sequence packing (spec §12)

The pipeline packs variable-length documents into fixed-length
context windows. Strategy:

1. Tokenize each document with `add_eos=True`
2. Truncate to `context_length + 1` (the +1 is for the shifted target)
3. Write to shards

Each shard record is `{"input_ids": [...]}`. The `ShardDataset` class
handles truncation/padding to `context_length` at load time.

## Sharding (spec §13)

```bash
einx data build --input raw.jsonl --output data/processed --tokenizer tok.json \
    --shard-size 10000 --context-length 256
```

Output:
```
data/processed/
    manifest.json
    train/
        shard-00000.jsonl
        shard-00001.jsonl
    val/
        shard-00000.jsonl
```

Storage format: JSONL. Chosen for simplicity + inspectability +
streaming-friendly + no external dependencies. Binary format (numpy
.npy or safetensors) is a Phase 4 optimization.

## Streaming (spec §14)

`ShardDataset` loads one shard at a time into RAM — never the whole
dataset. Suitable for large datasets that don't fit in memory.

## Deterministic shuffling (spec §15)

Same dataset + same seed + same config = same ordering. Different
seeds produce different orderings.

## Train/validation split (spec §16)

```yaml
data:
  validation_ratio: 0.05
  seed: 42
```

Deterministic, seeded split. No overlap between train and val (verified
by `find_cross_dataset_duplicates`).

## Data leakage protection (spec §17)

- Deterministic splitting (seeded)
- Hash-based duplicate detection across train/val
- Reports the number of detected overlaps

```python
from einx.data.deduplicator import DatasetDeduplicator
dedup = DatasetDeduplicator()
n_leak = dedup.find_cross_dataset_duplicates(train_records, val_records)
print(f"Train/val overlap: {n_leak} documents")
```

## Checkpoint + data state (spec §18)

Training checkpoints record:
- dataset manifest hash
- tokenizer version
- data seed
- current step + epoch

On resume, the trainer verifies the dataset hasn't changed — if the
manifest hash doesn't match, training fails with a clear error.

## Training data preflight (spec §19)

Before training begins:
1. validate dataset (manifest exists)
2. validate tokenizer (exists + version matches manifest)
3. validate model context length (matches dataset)
4. validate vocab size (tokenizer vocab == model vocab)
5. start training

```python
from einx.data.pipeline import preflight_check
preflight_check(
    dataset_dir="data/processed",
    tokenizer_path="tokenizer.json",
    model_config=model_cfg,
    training_config=train_cfg,
)
```

## Full pipeline command (spec §26)

```bash
einx data build \
    --input data/raw/train.jsonl \
    --output data/processed \
    --tokenizer data/tokenized/einx-bpe.json \
    --name einx-training-demo \
    --version 0.1.0 \
    --context-length 256 \
    --shard-size 10000 \
    --validation-ratio 0.05 \
    --seed 42
```

Runs: validate → clean → dedupe → split → tokenize → pack → shard → manifest.

Individual stages can be run separately via `einx data validate`,
`einx data clean`, `einx data dedupe`, `einx data tokenize`, `einx data stats`.

## Limitations (honest)

- **Near-duplicate detection**: NOT implemented (only exact + normalized
  exact). The `NearDuplicateDetector` abstraction exists for future
  MinHash / SimHash work.
- **Language detection**: NOT implemented (spec §7 says don't claim it).
- **Binary shard format**: JSONL only. A binary format would be smaller
  but harder to inspect. Phase 4 optimization.
- **Memory measurement**: CPU peak memory is not reported (returns 0).
  CUDA peak memory is reported via `torch.cuda.max_memory_allocated`.
- **Semantic deduplication**: NOT claimed. Only hash-based dedup.
