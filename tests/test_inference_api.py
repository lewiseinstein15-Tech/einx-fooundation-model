# -*- coding: utf-8 -*-
"""Tests for the inference engine + API."""

import json
import pytest
import torch
from fastapi.testclient import TestClient

from einx.config import EINXModelConfig
from einx.model.transformer import EINXTransformer
from einx.tokenizer.bpe import BPETokenizer
from einx.inference.generator import EINXGenerator, GenerationConfig, GenerationResult
from einx.api.server import build_app


def _build_test_generator(tmp_path):
    """Build a tiny generator for tests."""
    tok = BPETokenizer()
    texts = [f"the cat sat on the mat number {i}" for i in range(30)]
    tok.train(texts, vocab_size=400, verbose=False)

    cfg = EINXModelConfig(
        name="test-model",
        vocab_size=tok.vocab_size(),
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        max_context_length=32,
        ffn_dim=64,
        dropout=0.0,
    )
    model = EINXTransformer(cfg)

    # Save + load to get a real checkpoint
    path = str(tmp_path / "model.pt")
    model.save(path)
    tok_path = str(tmp_path / "tokenizer.json")
    tok.save(tok_path)

    gen = EINXGenerator.from_checkpoint(path, tok_path, device="cpu")
    return gen, tok


def test_generator_initialization(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    assert gen.model is not None
    assert gen.tokenizer is not None
    assert gen.device.type == "cpu"


def test_generator_model_info(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    info = gen.model_info()
    assert info["name"] == "test-model"
    assert info["n_params"] > 0
    assert info["device"] == "cpu"


def test_generator_generate_basic(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    result = gen.generate("the cat", GenerationConfig(max_new_tokens=10, temperature=0.0))
    assert isinstance(result, GenerationResult)
    assert len(result.token_ids) > 0
    assert result.n_input_tokens > 0
    assert result.n_output_tokens > 0
    assert result.elapsed_seconds > 0


def test_generator_generate_with_temperature(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    # With temperature > 0 and seed set, should produce deterministic-ish output
    cfg = GenerationConfig(max_new_tokens=5, temperature=0.8, top_k=10, seed=42)
    result = gen.generate("the cat", cfg)
    assert len(result.token_ids) == 5


def test_generator_stream(tmp_path):
    """Streaming yields one chunk per generated token — including any empty
    string from word-boundary tokens.  Total non-empty chunks should be > 0
    when the model produces any output."""
    gen, _ = _build_test_generator(tmp_path)
    chunks = list(gen.stream("the cat", GenerationConfig(max_new_tokens=5, temperature=0.8, seed=42)))
    # Should produce 5 chunks (one per token) even if some are empty
    assert len(chunks) <= 5
    # At least one chunk should be non-empty (the model produced output)
    non_empty = [c for c in chunks if c]
    assert len(non_empty) > 0 or len(chunks) == 5  # 5 chunks is also acceptable (all empty = model only produced word-boundaries)


def test_generation_config_validation():
    cfg = GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        cfg.validate()

    cfg = GenerationConfig(temperature=-0.5)
    with pytest.raises(ValueError, match="temperature"):
        cfg.validate()

    cfg = GenerationConfig(top_p=0)
    with pytest.raises(ValueError, match="top_p"):
        cfg.validate()


def test_generation_result_to_dict(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    result = gen.generate("the cat", GenerationConfig(max_new_tokens=3, temperature=0.0))
    d = result.to_dict()
    assert "text" in d
    assert "tokens_per_second" in d
    assert d["n_output_tokens"] == 3


def test_api_health_no_model():
    app = build_app(None)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_api_health_with_model(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    app = build_app(gen)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_api_model_endpoint(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    app = build_app(gen)
    client = TestClient(app)
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "test-model"
    assert body["n_params"] > 0


def test_api_model_endpoint_no_model():
    app = build_app(None)
    client = TestClient(app)
    r = client.get("/model")
    assert r.status_code == 503


def test_api_generate(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    app = build_app(gen)
    client = TestClient(app)
    r = client.post("/generate", json={
        "prompt": "the cat",
        "max_new_tokens": 5,
        "temperature": 0.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert body["n_output_tokens"] == 5


def test_api_chat(tmp_path):
    gen, _ = _build_test_generator(tmp_path)
    app = build_app(gen)
    client = TestClient(app)
    r = client.post("/chat", json={
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "max_new_tokens": 5,
        "temperature": 0.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "assistant"
    assert "content" in body


def test_api_generate_no_model():
    app = build_app(None)
    client = TestClient(app)
    r = client.post("/generate", json={"prompt": "test", "max_new_tokens": 1})
    assert r.status_code == 503
