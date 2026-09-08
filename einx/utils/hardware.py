# -*- coding: utf-8 -*-
"""Hardware detection for EINX (Build 2.1 — hardware-adaptive).

Detects the best available compute device (CUDA > MPS > CPU), recommends
a precision, and reports distributed-training capability.  Everything
is auto-detected from the actual runtime — never hardcoded.

Three execution modes (spec §1):

  * MODE A — CPU:           no CUDA → everything runs on CPU
  * MODE B — single GPU:    exactly 1 CUDA GPU → auto-use it
  * MODE C — multi GPU:     >=2 CUDA GPUs → DDP available, FSDP possible

Public API:
    detect_device(preference)         → "cpu" | "cuda" | "mps"
    recommend_precision(device)        → "fp32" | "fp16" | "bf16"
    detect_distributed_capability()    → DistributedCapability
    get_hardware_report(preference)   → HardwareReport
    print_hardware_report(report)      → stdout banner

Never claims CUDA exists when it doesn't.  Never recommends BF16 on a
GPU that doesn't support it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    """Information about one CUDA GPU."""

    index: int
    name: str
    total_memory_mb: float
    major: int = 0           # CUDA compute capability major
    minor: int = 0           # CUDA compute capability minor

    @property
    def supports_bf16(self) -> bool:
        """BF16 requires CUDA compute capability >= 8.0 (Ampere+)."""
        return self.major >= 8

    @property
    def supports_fp16(self) -> bool:
        """FP16 requires CUDA compute capability >= 5.3 (Maxwell+)."""
        return (self.major, self.minor) >= (5, 3)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "total_memory_mb": round(self.total_memory_mb, 1),
            "compute_capability": f"{self.major}.{self.minor}",
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
        }


@dataclass
class DistributedCapability:
    """Whether + how EINX can run distributed training in this env."""

    available: bool                  # True if distributed GPU training is possible
    strategy: str = "none"           # "none" | "ddp" | "fsdp"
    world_size: int = 1               # number of usable GPUs
    backend: str = "gloo"             # "nccl" (CUDA) | "gloo" (CPU)
    reason: str = ""                  # why distributed is/isn't available

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "strategy": self.strategy,
            "world_size": self.world_size,
            "backend": self.backend,
            "reason": self.reason,
        }


@dataclass
class HardwareReport:
    """Full hardware report — the single source of truth for the runtime.

    Built by :func:`get_hardware_report` from actual hardware detection.
    Every other module reads from this object instead of calling
    ``torch.cuda.is_available()`` directly — keeps the detection logic
    in one place.
    """

    cpu_available: bool = True
    cuda_available: bool = False
    cuda_device_count: int = 0
    mps_available: bool = False
    gpus: List[GPUInfo] = field(default_factory=list)
    selected_device: str = "cpu"           # "cpu" | "cuda" | "mps"
    selected_device_name: str = "CPU"
    recommended_precision: str = "fp32"    # "fp32" | "fp16" | "bf16"
    distributed: DistributedCapability = field(default_factory=DistributedCapability)
    pytorch_version: str = ""
    cuda_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cpu_available": self.cpu_available,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "mps_available": self.mps_available,
            "gpus": [g.to_dict() for g in self.gpus],
            "selected_device": self.selected_device,
            "selected_device_name": self.selected_device_name,
            "recommended_precision": self.recommended_precision,
            "distributed": self.distributed.to_dict(),
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
        }


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------


def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_mps_available() -> bool:
    try:
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _cuda_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _list_gpus() -> List[GPUInfo]:
    """Enumerate every CUDA GPU with its real name + memory + compute capability.

    Returns an empty list when CUDA is unavailable — never raises.
    """
    if not is_cuda_available():
        return []
    try:
        import torch
        gpus: List[GPUInfo] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory_mb=props.total_memory / (1024 * 1024),
                major=props.major,
                minor=props.minor,
            ))
        return gpus
    except Exception as exc:
        logger.warning("GPU enumeration failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Device + precision selection
# ---------------------------------------------------------------------------


def detect_device(preference: str = "auto") -> str:
    """Pick the best device given the environment + user preference.

    ``preference``:
      * "auto"   — best available (cuda > mps > cpu)
      * "cpu"    — forced CPU (always works)
      * "cuda"   — forced CUDA (falls back to CPU with a warning if unavailable)
      * "cuda:0" — specific GPU index
      * "mps"    — forced Apple Silicon GPU
    """
    if preference == "auto":
        if is_cuda_available():
            return "cuda"
        if is_mps_available():
            return "mps"
        return "cpu"
    if preference.startswith("cuda"):
        if not is_cuda_available():
            logger.warning("CUDA requested but not available — falling back to CPU")
            return "cpu"
        # Handle "cuda:0" / "cuda:1" — validate the index exists
        if ":" in preference:
            try:
                idx = int(preference.split(":")[1])
                if idx >= _cuda_device_count():
                    logger.warning(
                        "cuda:%d requested but only %d GPUs available — falling back to cuda:0",
                        idx, _cuda_device_count(),
                    )
                    return "cuda"
            except (ValueError, IndexError):
                pass
        return preference
    if preference == "mps" and not is_mps_available():
        logger.warning("MPS requested but not available — falling back to CPU")
        return "cpu"
    return preference


def recommend_precision(device: str = "auto") -> str:
    """Recommend a precision for the given device.

    Rules (spec §4):
      * CPU                          → fp32  (no hardware FP16/BF16 support)
      * CUDA with BF16-capable GPU   → bf16  (Ampere+ — compute capability >= 8.0)
      * CUDA with FP16-only GPU      → fp16  (Maxwell through Turing)
      * MPS                          → fp16  (Apple Silicon supports FP16)
      * unsupported environment     → fp32  (safe fallback)
    """
    if device == "auto":
        device = detect_device("auto")
    if device == "cpu":
        return "fp32"
    if device == "mps":
        return "fp16"
    if device.startswith("cuda"):
        gpus = _list_gpus()
        if not gpus:
            return "fp32"  # CUDA requested but no GPUs — safe fallback
        # Use GPU 0's capability as the reference
        gpu0 = gpus[0]
        if gpu0.supports_bf16:
            return "bf16"
        if gpu0.supports_fp16:
            return "fp16"
        return "fp32"
    return "fp32"


def detect_distributed_capability(
    *,
    force_strategy: str = "auto",
    backend: str = "auto",
) -> DistributedCapability:
    """Determine whether distributed GPU training is possible.

    Rules (spec §5):
      * CUDA unavailable         → single-process CPU, distributed=none
      * CUDA available + 1 GPU   → single-process GPU, distributed=none
      * CUDA available + >=2 GPU → DDP available, world_size = GPU count
      * FSDP requires BF16-capable GPUs (compute capability >= 8.0)
    """
    if not is_cuda_available():
        return DistributedCapability(
            available=False,
            strategy="none",
            world_size=1,
            backend="gloo",
            reason="no CUDA devices detected",
        )
    n_gpus = _cuda_device_count()
    if n_gpus < 2:
        return DistributedCapability(
            available=False,
            strategy="none",
            world_size=1,
            backend="nccl",
            reason=f"only {n_gpus} CUDA GPU(s) — need >=2 for distributed training",
        )
    # Multi-GPU — DDP is available
    gpus = _list_gpus()
    # FSDP requires BF16-capable GPUs (Ampere+) for stable sharded training
    fsdp_capable = all(g.supports_bf16 for g in gpus)
    strategy = force_strategy
    if strategy == "auto":
        strategy = "fsdp" if fsdp_capable else "ddp"
    elif strategy == "fsdp" and not fsdp_capable:
        logger.warning(
            "FSDP requested but not all GPUs support BF16 (compute capability >= 8.0) "
            "— falling back to DDP"
        )
        strategy = "ddp"
    chosen_backend = backend
    if chosen_backend == "auto":
        chosen_backend = "nccl" if is_cuda_available() else "gloo"
    return DistributedCapability(
        available=True,
        strategy=strategy,
        world_size=n_gpus,
        backend=chosen_backend,
        reason=f"{n_gpus} CUDA GPUs available",
    )


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def get_hardware_report(preference: str = "auto") -> HardwareReport:
    """Build the full hardware report from actual detection.

    This is the single entry point — every other module should read
    from this report instead of calling torch.cuda directly.
    """
    import torch

    gpus = _list_gpus()
    selected_device = detect_device(preference)
    selected_name = "CPU"
    if selected_device == "cuda":
        selected_name = gpus[0].name if gpus else "CUDA GPU"
    elif selected_device == "mps":
        selected_name = "Apple Silicon GPU (MPS)"

    precision = recommend_precision(selected_device)
    distributed = detect_distributed_capability()

    return HardwareReport(
        cpu_available=True,
        cuda_available=is_cuda_available(),
        cuda_device_count=len(gpus),
        mps_available=is_mps_available(),
        gpus=gpus,
        selected_device=selected_device,
        selected_device_name=selected_name,
        recommended_precision=precision,
        distributed=distributed,
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda if is_cuda_available() else None,
    )


# ---------------------------------------------------------------------------
# Reporting (stdout banner)
# ---------------------------------------------------------------------------


def format_hardware_report(report: HardwareReport) -> str:
    """Format the report as a printable banner (spec §14)."""
    lines = [
        "EINX Runtime",
        "─" * 40,
        f"Device: {report.selected_device.upper()}",
        f"CUDA: {'available' if report.cuda_available else 'unavailable'}",
        f"GPUs: {report.cuda_device_count}",
    ]
    if report.gpus:
        for g in report.gpus:
            lines.append(f"  GPU {g.index}: {g.name} ({g.total_memory_mb:.0f} MB, cc {g.major}.{g.minor})")
    lines.append(f"Precision: {report.recommended_precision.upper()}")
    if report.distributed.available:
        lines.append(f"Distributed: {report.distributed.strategy.upper()}")
        lines.append(f"World size: {report.distributed.world_size}")
        lines.append(f"Backend: {report.distributed.backend}")
    else:
        lines.append("Distributed: disabled")
        lines.append(f"Reason: {report.distributed.reason}")
    lines.append("─" * 40)
    return "\n".join(lines)


def print_hardware_report(preference: str = "auto") -> HardwareReport:
    """Build + print the hardware report.  Returns the report for reuse."""
    report = get_hardware_report(preference)
    print(format_hardware_report(report))
    return report


# ---------------------------------------------------------------------------
# Backwards-compat shim — Build 2 code still imports DeviceInfo + get_device_info
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    """Legacy compatibility — prefer HardwareReport (Build 2.1+)."""

    device: str
    name: str
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "name": self.name,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "mps_available": self.mps_available,
        }


def get_device_info(preference: str = "auto") -> DeviceInfo:
    """Legacy compat — delegates to :func:`get_hardware_report`."""
    report = get_hardware_report(preference)
    return DeviceInfo(
        device=report.selected_device,
        name=report.selected_device_name,
        cuda_available=report.cuda_available,
        cuda_device_count=report.cuda_device_count,
        mps_available=report.mps_available,
    )
