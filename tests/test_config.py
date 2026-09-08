# -*- coding: utf-8 -*-
"""Tests for the EINX configuration system."""

import pytest
from einx.config import EINXModelConfig, TrainingConfig, EvalConfig, get_model_config, get_training_config


def test_model_config_defaults():
    cfg = EINXModelConfig()
    assert cfg.name == "einx-experimental"
    assert cfg.arch == "decoder-only-transformer"
    assert cfg.vocab_size > 0
    assert cfg.n_layers > 0
    assert cfg.hidden_dim > 0


def test_model_config_validation_head_dim():
    """head_dim * n_heads must equal hidden_dim."""
    cfg = EINXModelConfig(hidden_dim=128, n_heads=4, head_dim=33)
    with pytest.raises(ValueError, match="hidden_dim"):
        cfg.validate()


def test_model_config_validation_pos_encoding():
    cfg = EINXModelConfig(positional_encoding="invalid")
    with pytest.raises(ValueError, match="positional_encoding"):
        cfg.validate()


def test_model_config_validation_norm_type():
    cfg = EINXModelConfig(norm_type="invalid")
    with pytest.raises(ValueError, match="norm_type"):
        cfg.validate()


def test_model_config_validation_precision():
    cfg = EINXModelConfig(precision="int8")
    with pytest.raises(ValueError, match="precision"):
        cfg.validate()


def test_model_config_to_from_dict_roundtrip():
    cfg = EINXModelConfig()
    d = cfg.to_dict()
    cfg2 = EINXModelConfig.from_dict(d)
    assert cfg2.to_dict() == d


def test_model_config_yaml_roundtrip(tmp_path):
    cfg = EINXModelConfig(name="test-model")
    path = tmp_path / "model.yaml"
    cfg.save_yaml(path)
    cfg2 = EINXModelConfig.from_yaml(path)
    assert cfg2.name == "test-model"
    assert cfg2.vocab_size == cfg.vocab_size


def test_get_model_config_experimental():
    cfg = get_model_config("einx-experimental")
    assert cfg.name == "einx-experimental"
    assert cfg.n_layers == 4
    assert cfg.hidden_dim == 128


def test_get_model_config_small():
    cfg = get_model_config("einx-small")
    assert cfg.name == "einx-small"
    assert cfg.hidden_dim == 256


def test_get_model_config_unknown():
    with pytest.raises(KeyError, match="unknown model name"):
        get_model_config("einx-1b")


def test_model_config_approx_param_count():
    cfg = EINXModelConfig()
    n = cfg.approx_param_count()
    assert n > 0
    # Should be on the order of 1M for the experimental config
    assert 100_000 < n < 5_000_000


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.batch_size > 0
    assert cfg.learning_rate > 0
    assert cfg.max_steps > 0


def test_training_config_validation():
    cfg = TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        cfg.validate()


def test_training_config_yaml_roundtrip(tmp_path):
    cfg = TrainingConfig(run_name="test-run", learning_rate=1e-4)
    path = tmp_path / "training.yaml"
    cfg.save_yaml(path)
    cfg2 = TrainingConfig.from_yaml(path)
    assert cfg2.run_name == "test-run"
    assert cfg2.learning_rate == 1e-4


def test_get_training_config():
    cfg = get_training_config("experimental")
    assert cfg.model_name == "einx-experimental"


def test_eval_config_validation():
    cfg = EvalConfig(checkpoint_path="")
    with pytest.raises(ValueError, match="checkpoint_path"):
        cfg.validate()


def test_eval_config_defaults():
    cfg = EvalConfig(checkpoint_path="dummy.pt")
    assert "perplexity" in cfg.metrics
    assert "loss" in cfg.metrics
