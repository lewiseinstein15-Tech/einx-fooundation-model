# -*- coding: utf-8 -*-
"""Tests for the Build 2.1 hardware-adaptive layer (spec §2, §3, §4, §5)."""

import pytest
import torch

from einx.utils.hardware import (
    detect_device,
    recommend_precision,
    detect_distributed_capability,
    get_hardware_report,
    format_hardware_report,
    print_hardware_report,
    HardwareReport,
    GPUInfo,
    DistributedCapability,
    is_cuda_available,
    is_mps_available,
)
from einx.config.runtime_config import RuntimeConfig, ResolvedRuntime, get_default_runtime_config


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def test_detect_device_auto_on_cpu_returns_cpu():
    """On this CPU-only env, 'auto' must resolve to 'cpu'."""
    if is_cuda_available() or is_mps_available():
        pytest.skip("this test only runs on CPU-only environments")
    assert detect_device("auto") == "cpu"


def test_detect_device_forced_cpu():
    assert detect_device("cpu") == "cpu"


def test_detect_device_cuda_falls_back_to_cpu_when_unavailable():
    """Forcing CUDA on a CPU-only env should warn + fall back to CPU."""
    if is_cuda_available():
        pytest.skip("CUDA is available — can't test the fallback path")
    assert detect_device("cuda") == "cpu"


def test_detect_device_cuda_index_validation():
    """cuda:99 with only 0 GPUs falls back to cuda:0 (which then falls to CPU)."""
    if is_cuda_available():
        pytest.skip("CUDA is available — can't test invalid index fallback")
    # On CPU, "cuda:99" → "cpu" (cuda unavailable)
    assert detect_device("cuda:99") == "cpu"


# ---------------------------------------------------------------------------
# Precision recommendation
# ---------------------------------------------------------------------------


def test_recommend_precision_cpu_always_fp32():
    """CPU must always recommend fp32 — no hardware FP16/BF16 support."""
    assert recommend_precision("cpu") == "fp32"


def test_recommend_precision_auto_on_cpu_is_fp32():
    if is_cuda_available() or is_mps_available():
        pytest.skip("this test only runs on CPU-only environments")
    assert recommend_precision("auto") == "fp32"


def test_recommend_precision_mps_is_fp16():
    """MPS (Apple Silicon) supports FP16."""
    # We can't force MPS availability in tests, but we can verify the
    # function logic: if device=="mps", precision should be "fp16".
    if not is_mps_available():
        # On this env, we just verify the function doesn't crash for "mps"
        # — it may return fp16 or fall back to fp32 depending on availability.
        result = recommend_precision("mps")
        assert result in ("fp16", "fp32")
    else:
        assert recommend_precision("mps") == "fp16"


# ---------------------------------------------------------------------------
# Distributed capability detection
# ---------------------------------------------------------------------------


def test_detect_distributed_capability_cpu_unavailable():
    """On a CPU-only env, distributed GPU training is unavailable."""
    if is_cuda_available():
        pytest.skip("CUDA is available — can't test the CPU-only path")
    cap = detect_distributed_capability()
    assert cap.available is False
    assert cap.strategy == "none"
    assert cap.world_size == 1
    assert "no cuda" in cap.reason.lower()


def test_detect_distributed_capability_single_gpu_unavailable():
    """On a single-GPU env, distributed training is unavailable (need >=2)."""
    if not is_cuda_available():
        pytest.skip("CUDA unavailable — covered by test_detect_distributed_capability_cpu_unavailable")
    if torch.cuda.device_count() >= 2:
        pytest.skip("multi-GPU env — distributed IS available, can't test single-GPU path")
    cap = detect_distributed_capability()
    assert cap.available is False
    assert "need >=2" in cap.reason.lower() or "only 1" in cap.reason.lower()


# ---------------------------------------------------------------------------
# Hardware report
# ---------------------------------------------------------------------------


def test_get_hardware_report_returns_full_report():
    report = get_hardware_report("auto")
    assert isinstance(report, HardwareReport)
    assert report.cpu_available is True
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.cuda_device_count, int)
    assert isinstance(report.selected_device, str)
    assert report.selected_device in ("cpu", "cuda", "mps")
    assert report.recommended_precision in ("fp32", "fp16", "bf16")
    assert isinstance(report.distributed, DistributedCapability)


def test_hardware_report_to_dict():
    report = get_hardware_report()
    d = report.to_dict()
    assert "cpu_available" in d
    assert "cuda_available" in d
    assert "selected_device" in d
    assert "recommended_precision" in d
    assert "distributed" in d
    assert "gpus" in d


def test_format_hardware_report_contains_required_sections():
    """The banner (spec §14) must contain device, CUDA, precision, distributed."""
    report = get_hardware_report()
    text = format_hardware_report(report)
    assert "EINX Runtime" in text
    assert "Device:" in text
    assert "CUDA:" in text
    assert "Precision:" in text
    assert "Distributed:" in text
    if not report.distributed.available:
        assert "Reason:" in text


def test_print_hardware_report_returns_report(capsys):
    report = print_hardware_report("cpu")
    captured = capsys.readouterr()
    assert "EINX Runtime" in captured.out
    assert report.selected_device in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


def test_runtime_config_defaults_to_auto():
    cfg = RuntimeConfig()
    assert cfg.device == "auto"
    assert cfg.precision == "auto"
    assert cfg.distributed == "auto"
    assert cfg.backend == "auto"
    assert cfg.compile is False


def test_runtime_config_validation_rejects_bad_device():
    cfg = RuntimeConfig(device="tpu")
    with pytest.raises(ValueError, match="runtime.device"):
        cfg.validate()


def test_runtime_config_validation_rejects_bad_precision():
    cfg = RuntimeConfig(precision="int8")
    with pytest.raises(ValueError, match="runtime.precision"):
        cfg.validate()


def test_runtime_config_validation_rejects_bad_distributed():
    cfg = RuntimeConfig(distributed="sharded")
    with pytest.raises(ValueError, match="runtime.distributed"):
        cfg.validate()


def test_runtime_config_resolve_on_cpu():
    """On CPU, resolving 'auto' should produce fp32, no distributed."""
    if is_cuda_available():
        pytest.skip("CUDA available — can't test CPU resolve path")
    cfg = RuntimeConfig()
    resolved = cfg.resolve()
    assert isinstance(resolved, ResolvedRuntime)
    assert resolved.device == "cpu"
    assert resolved.precision == "fp32"
    assert resolved.distributed == "none"
    assert resolved.is_distributed is False
    assert resolved.use_amp is False


def test_runtime_config_resolve_rejects_fp16_on_cpu():
    """Asking for fp16 on CPU must raise a friendly error (spec §4)."""
    if is_cuda_available():
        pytest.skip("CUDA available — fp16 on CPU error doesn't apply")
    cfg = RuntimeConfig(device="cpu", precision="fp16")
    from einx.utils.errors import EINXConfigError
    with pytest.raises(EINXConfigError, match="CPU does not support"):
        cfg.resolve()


def test_runtime_config_resolve_rejects_ddp_without_gpus():
    """Asking for DDP without multi-GPU hardware must raise (spec §5)."""
    if is_cuda_available() and torch.cuda.device_count() >= 2:
        pytest.skip("multi-GPU env — DDP is actually available")
    cfg = RuntimeConfig(distributed="ddp")
    from einx.utils.errors import EINXConfigError
    with pytest.raises(EINXConfigError, match="multi-GPU"):
        cfg.resolve()


def test_runtime_config_yaml_roundtrip(tmp_path):
    cfg = RuntimeConfig(device="cpu", precision="fp32", compile=True)
    path = tmp_path / "runtime.yaml"
    with open(path, "w") as fh:
        fh.write(cfg.to_yaml())
    cfg2 = RuntimeConfig.from_yaml(path)
    assert cfg2.device == "cpu"
    assert cfg2.precision == "fp32"
    assert cfg2.compile is True


def test_get_default_runtime_config():
    cfg = get_default_runtime_config()
    assert cfg.device == "auto"
    assert cfg.precision == "auto"


# ---------------------------------------------------------------------------
# GPUInfo
# ---------------------------------------------------------------------------


def test_gpu_info_supports_bf16_ampere():
    """Compute capability 8.0+ (Ampere) supports BF16."""
    gpu = GPUInfo(index=0, name="A100", total_memory_mb=40000, major=8, minor=0)
    assert gpu.supports_bf16 is True
    assert gpu.supports_fp16 is True


def test_gpu_info_no_bf16_on_turing():
    """Compute capability 7.x (Turing) supports FP16 but NOT BF16."""
    gpu = GPUInfo(index=0, name="RTX 2080", total_memory_mb=8000, major=7, minor=5)
    assert gpu.supports_bf16 is False
    assert gpu.supports_fp16 is True


def test_gpu_info_to_dict():
    gpu = GPUInfo(index=0, name="A100", total_memory_mb=40000, major=8, minor=0)
    d = gpu.to_dict()
    assert d["name"] == "A100"
    assert d["compute_capability"] == "8.0"
    assert d["supports_bf16"] is True
