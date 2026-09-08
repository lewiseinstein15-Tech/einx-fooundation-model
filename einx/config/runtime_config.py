# -*- coding: utf-8 -*-
"""Runtime configuration for EINX (Build 2.1 — spec §16).

A typed dataclass that captures every hardware/distributed/precision
decision in one place.  Default is ``auto`` for everything — EINX
decides based on actual hardware detection.

Example YAML (configs/runtime/default.yaml):

    runtime:
      device: auto          # auto | cpu | cuda | cuda:0 | mps
      precision: auto       # auto | fp32 | fp16 | bf16
      distributed: auto     # auto | none | ddp | fsdp
      backend: auto         # auto | nccl | gloo
      compile: false        # torch.compile toggle
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RuntimeConfig:
    """Hardware + distributed runtime configuration.

    Everything defaults to ``auto`` — meaning EINX detects the best
    option from the actual hardware.  Users can override any field
    explicitly.
    """

    device: str = "auto"             # auto | cpu | cuda | cuda:0 | mps
    precision: str = "auto"          # auto | fp32 | fp16 | bf16
    distributed: str = "auto"        # auto | none | ddp | fsdp
    backend: str = "auto"             # auto | nccl | gloo | mpi
    compile: bool = False             # torch.compile toggle

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda", "mps") and not self.device.startswith("cuda:"):
            raise ValueError(
                f"runtime.device must be 'auto' | 'cpu' | 'cuda' | 'cuda:N' | 'mps', "
                f"got {self.device!r}"
            )
        if self.precision not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError(
                f"runtime.precision must be 'auto' | 'fp32' | 'fp16' | 'bf16', "
                f"got {self.precision!r}"
            )
        if self.distributed not in ("auto", "none", "ddp", "fsdp"):
            raise ValueError(
                f"runtime.distributed must be 'auto' | 'none' | 'ddp' | 'fsdp', "
                f"got {self.distributed!r}"
            )
        if self.backend not in ("auto", "nccl", "gloo", "mpi"):
            raise ValueError(
                f"runtime.backend must be 'auto' | 'nccl' | 'gloo' | 'mpi', "
                f"got {self.backend!r}"
            )

    # ------------------------------------------------------------------
    # Resolve ``auto`` against actual hardware
    # ------------------------------------------------------------------
    def resolve(self) -> "ResolvedRuntime":
        """Resolve every ``auto`` field against actual hardware detection.

        Returns a :class:`ResolvedRuntime` with concrete values — no
        ``auto`` left.  Safe to call repeatedly; cheap.
        """
        from einx.utils.hardware import (
            detect_device,
            recommend_precision,
            detect_distributed_capability,
            get_hardware_report,
        )
        self.validate()
        report = get_hardware_report(self.device)

        # Resolve device
        device = detect_device(self.device)
        # Resolve precision
        if self.precision == "auto":
            precision = recommend_precision(device)
        else:
            precision = self.precision
            # Sanity-check: don't allow fp16/bf16 on CPU
            if device == "cpu" and precision in ("fp16", "bf16"):
                from einx.utils.errors import EINXConfigError
                raise EINXConfigError(
                    f"precision {precision!r} requested but device is CPU — "
                    "CPU does not support hardware FP16/BF16. "
                    "Use precision='fp32' or device='cuda'.",
                    field="precision",
                    value=precision,
                    hint="Set precision='fp32' for CPU, or device='cuda' for GPU mixed precision.",
                )
        # Resolve distributed
        if self.distributed == "auto":
            cap = detect_distributed_capability()
            distributed = cap.strategy if cap.available else "none"
        else:
            distributed = self.distributed
            # Sanity-check: don't allow ddp/fsdp on CPU unless explicitly using gloo
            cap = detect_distributed_capability()
            if distributed in ("ddp", "fsdp") and not cap.available:
                # Allow CPU + gloo for infrastructure testing
                if self.backend != "gloo":
                    from einx.utils.errors import EINXConfigError
                    raise EINXConfigError(
                        f"distributed={distributed!r} requested but no multi-GPU CUDA hardware. "
                        "Distributed GPU training requires >=2 CUDA GPUs.",
                        field="distributed",
                        value=distributed,
                        hint="Use distributed='none' for single-device, or set backend='gloo' for CPU distributed testing.",
                    )
        # Resolve backend
        if self.backend == "auto":
            backend = "nccl" if device.startswith("cuda") else "gloo"
        else:
            backend = self.backend

        return ResolvedRuntime(
            device=device,
            precision=precision,
            distributed=distributed,
            backend=backend,
            compile=self.compile,
            hardware_report=report,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump({"runtime": self.to_dict()}, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RuntimeConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        # Accept either a top-level 'runtime' key or the fields directly
        if "runtime" in data:
            data = data["runtime"]
        return cls.from_dict(data)


@dataclass
class ResolvedRuntime:
    """A RuntimeConfig with every ``auto`` resolved against actual hardware.

    Built by :meth:`RuntimeConfig.resolve`.  Carries the full
    :class:`HardwareReport` so callers don't need to re-detect.
    """

    device: str                # "cpu" | "cuda" | "cuda:0" | "mps"
    precision: str             # "fp32" | "fp16" | "bf16"
    distributed: str           # "none" | "ddp" | "fsdp"
    backend: str                # "nccl" | "gloo"
    compile: bool
    hardware_report: Any       # HardwareReport

    @property
    def is_distributed(self) -> bool:
        return self.distributed in ("ddp", "fsdp")

    @property
    def use_amp(self) -> bool:
        """Whether to enable autocast (mixed precision)."""
        return self.precision in ("fp16", "bf16") and self.device.startswith("cuda")

    @property
    def amp_dtype(self) -> Any:
        if not self.use_amp:
            return None
        import torch
        return torch.float16 if self.precision == "fp16" else torch.bfloat16

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "precision": self.precision,
            "distributed": self.distributed,
            "backend": self.backend,
            "compile": self.compile,
            "is_distributed": self.is_distributed,
            "use_amp": self.use_amp,
        }


def get_default_runtime_config() -> RuntimeConfig:
    """The default runtime — everything ``auto``."""
    return RuntimeConfig()
