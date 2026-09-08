# EINX Hardware Requirements + Test Matrix (Build 2.1)

EINX is **hardware-adaptive**: the same codebase runs on a CPU laptop,
single-GPU workstation, multi-GPU server, or cloud GPU instance — no
rewrite needed.  Hardware is auto-detected at startup; nothing is
hardcoded.

## Hardware modes (spec §1)

| Mode | Hardware | Behaviour |
|------|----------|-----------|
| **A — CPU** | no CUDA | Everything runs on CPU. fp32 only. |
| **B — single GPU** | 1 CUDA GPU | Auto-uses the GPU. Mixed precision (fp16/bf16) where supported. |
| **C — multi GPU** | ≥2 CUDA GPUs | DDP available. FSDP available if all GPUs are BF16-capable (Ampere+). |

The current environment reports:

```
CPU: AVAILABLE
CUDA: NOT AVAILABLE
DDP GPU runtime: NOT TESTED (no CUDA GPUs)
FSDP GPU runtime: NOT TESTED (no CUDA GPUs)
CPU distributed infrastructure: TESTABLE (via Gloo backend)
```

This is the **honest** state — EINX does NOT fake CUDA availability.

## Automatic hardware detection

```bash
einx hardware
```

Output on this CPU-only environment:

```
EINX Runtime
────────────────────────────────────────
Device: CPU
CUDA: unavailable
GPUs: 0
Precision: FP32
Distributed: disabled
Reason: no CUDA devices detected
────────────────────────────────────────
```

On a 2-GPU BF16-capable machine it would print:

```
EINX Runtime
────────────────────────────────────────
Device: CUDA
CUDA: available
GPUs: 2
  GPU 0: NVIDIA A100-SXM4-40GB (40000 MB, cc 8.0)
  GPU 1: NVIDIA A100-SXM4-40GB (40000 MB, cc 8.0)
Precision: BF16
Distributed: FSDP
World size: 2
Backend: nccl
────────────────────────────────────────
```

## Device selection (spec §3)

One centralized device-selection system.  Every `torch.device()` call
goes through `einx.utils.hardware.detect_device()` — never scattered.

```bash
einx train --device auto      # auto-detect (default)
einx train --device cpu       # force CPU
einx train --device cuda      # force CUDA (falls back to CPU if unavailable)
einx train --device cuda:1    # force specific GPU
einx train --device mps       # force Apple Silicon GPU
```

## Precision selection (spec §4)

| Environment | Recommended precision |
|-------------|----------------------|
| CPU | FP32 (no hardware FP16/BF16) |
| CUDA, BF16-capable GPU (cc ≥ 8.0) | BF16 |
| CUDA, FP16-only GPU (cc 5.3–7.x) | FP16 |
| MPS (Apple Silicon) | FP16 |

```bash
einx train --precision auto     # auto-detect (default)
einx train --precision fp32      # force FP32
einx train --precision bf16      # force BF16 (CUDA only — errors on CPU)
```

Asking for FP16/BF16 on CPU raises a friendly error (spec §22 — never fake):

```
EINXConfigError: precision 'bf16' requested but device is CPU —
  CPU does not support hardware FP16/BF16.
  hint: Set precision='fp32' for CPU, or device='cuda' for GPU mixed precision.
```

## Distributed training detection (spec §5)

| Environment | Distributed capability |
|-------------|----------------------|
| No CUDA | unavailable (single-process CPU) |
| 1 CUDA GPU | unavailable (single-process GPU) |
| ≥2 CUDA GPUs (any) | DDP available |
| ≥2 CUDA GPUs, all BF16-capable | FSDP available |

## DDP (spec §6)

Real DDP support via `torch.distributed`.  Features:

- One process per GPU (launched via `torchrun` or `einx distributed-train`)
- DistributedSampler for data sharding
- Rank-aware logging (only rank 0 logs — prevents spam)
- Rank-aware checkpointing (only rank 0 writes — prevents duplicates)
- Clean process group shutdown
- `set_epoch()` called on the sampler each epoch for proper reshuffling

```bash
einx distributed-train --strategy ddp --training-args "--steps 1000"
```

On a 4-GPU machine this auto-launches `torchrun --nproc_per_node=4`.

## FSDP (spec §7)

FSDP is **implemented** (real code, importable + unit-testable without CUDA)
but its **runtime** requires a compatible CUDA environment:

- ≥2 CUDA GPUs
- All GPUs support BF16 (compute capability ≥ 8.0, Ampere+)

On CPU it raises:

```
RuntimeError: FSDP requires a CUDA environment.
FSDP implementation is present, but FSDP runtime validation
requires compatible accelerator hardware (BF16-capable GPUs).
Detected: no CUDA devices.
```

This is the honest contract (spec §22): **FSDP implementation present.
FSDP runtime validation requires compatible accelerator hardware.**

```bash
einx distributed-train --strategy fsdp --training-args "--steps 1000"
```

## Test matrix (spec §19)

| Environment | Tests | Status on this env |
|-------------|-------|-------------------|
| CPU | Full core functionality | ✓ TESTED (133 unit + 11 hardware + 11 distributed CPU) |
| CPU + Gloo | Distributed infrastructure tests | ✓ TESTED (init, rank, sampler, cleanup, checkpoint ownership) |
| 1 CUDA GPU | Single-GPU training | ✗ SKIPPED (no CUDA) |
| ≥2 CUDA GPUs | DDP | ✗ SKIPPED (no CUDA) |
| Compatible multi-GPU | FSDP | ✗ SKIPPED (no BF16-capable GPUs) |

The skipped tests are marked `pytest.skip` with an explicit reason —
NOT passed, NOT failed, SKIPPED.  When a real CUDA machine becomes
available, they will automatically run.

## Checkpoint cross-device compatibility (spec §15)

Checkpoints created on CPU can be loaded on GPU, and vice versa.

```python
# Save on CPU
model.save("ckpt.pt")  # CPU tensors

# Load on GPU
model = EINXTransformer.load("ckpt.pt", map_location="cuda:0")

# Save on GPU
model.save("ckpt.pt")  # CUDA tensors

# Load on CPU
model = EINXTransformer.load("ckpt.pt", map_location="cpu")
```

The `map_location` parameter is passed through to `torch.load`, so
tensors are moved to the target device during load.  No assumption is
made about the checkpoint's original device.

## Configuration (spec §16)

Runtime config in YAML (see `configs/runtime/default.yaml`):

```yaml
runtime:
  device: auto          # auto | cpu | cuda | cuda:0 | mps
  precision: auto       # auto | fp32 | fp16 | bf16
  distributed: auto     # auto | none | ddp | fsdp
  backend: auto         # auto | nccl | gloo | mpi
  compile: false        # torch.compile toggle
```

Default is `auto` for everything — EINX decides based on actual hardware.

## CI (spec §20)

The GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **CPU tests** — run unconditionally on every push/PR.  Will never
   fail because a runner has no GPU.
2. **CPU distributed infrastructure tests** — also unconditional, using
   the Gloo backend.
3. **GPU tests** — skipped by default (GitHub-hosted runners have no
   GPU).  Self-hosted GPU runners can opt in by setting the
   `EINX_GPU_RUNNER` variable.

The CI never fails because of missing hardware — only because of real
bugs.

## Hardware-independent development (spec §17)

A developer can clone EINX onto:

- CPU laptop → `einx train --device cpu`
- single GPU workstation → `einx train --device auto` (uses the GPU)
- multi-GPU workstation → `einx distributed-train --strategy ddp`
- cloud GPU server → `einx distributed-train --strategy fsdp`

Same repository, same code, same tests.  No machine-specific assumptions.

## Performance reporting (spec §18)

Every completed training run reports:

| Metric | Available on CPU | Available on CUDA |
|--------|------------------|------------------|
| device | ✓ | ✓ |
| precision | ✓ | ✓ |
| batch size | ✓ | ✓ |
| gradient accumulation | ✓ | ✓ |
| tokens/sec | ✓ | ✓ |
| steps/sec | ✓ | ✓ |
| peak memory | N/A (returns 0) | ✓ |

When a metric is unavailable, EINX reports `N/A` — never estimates.

## Honest status (spec §22)

| Component | Implementation | Runtime tested |
|-----------|---------------|---------------|
| Hardware detection | ✓ present | ✓ CPU |
| Automatic device selection | ✓ present | ✓ CPU |
| Precision selection | ✓ present | ✓ CPU (fp32) |
| CPU training | ✓ present | ✓ CPU |
| Single GPU support | ✓ present | ✗ not tested (no CUDA) |
| DDP infrastructure | ✓ present | ✓ CPU (Gloo) |
| DDP multi-GPU runtime | ✓ present | ✗ not tested (no CUDA) |
| FSDP infrastructure | ✓ present | ✗ not tested (no compatible GPU) |
| CPU distributed tests | ✓ present | ✓ passed |
| Conditional GPU tests | ✓ present | ✓ correctly skipped |

When a real CUDA machine becomes available, EINX uses it without rewrite.
