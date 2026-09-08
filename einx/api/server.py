# -*- coding: utf-8 -*-
"""FastAPI server for EINX.

Endpoints:
  GET  /health      — service health check
  GET  /model       — model metadata (no private training info)
  POST /generate    — text generation
  POST /chat        — chat-style completion (system + user + assistant turns)

The API is designed by us — not a copy of OpenAI / Anthropic / Google
APIs — but follows the same conceptual shape so JEXI OS (and any other
LLM client) can call it cleanly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from einx.inference.generator import EINXGenerator, GenerationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The text prompt to generate from")
    max_new_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=1000)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    text: str
    n_input_tokens: int
    n_output_tokens: int
    elapsed_seconds: float
    tokens_per_second: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="system | user | assistant")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=1000)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    role: str = "assistant"
    content: str
    n_input_tokens: int
    n_output_tokens: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(generator: Optional[EINXGenerator] = None) -> FastAPI:
    """Build the FastAPI app.  If ``generator`` is None, endpoints return
    a clear 'no model loaded' error rather than crashing — so /health
    works even before a checkpoint is loaded."""
    app = FastAPI(
        title="EINX API",
        description="Lewis Einstein's experimental intelligence architecture — HTTP API",
        version="0.1.0",
    )
    app.state.generator = generator

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "EINX API",
            "version": "0.1.0",
            "model_loaded": app.state.generator is not None,
        }

    @app.get("/model")
    def model_info() -> Dict[str, Any]:
        gen = app.state.generator
        if gen is None:
            raise HTTPException(status_code=503, detail="no model loaded")
        return gen.model_info()

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        gen = app.state.generator
        if gen is None:
            raise HTTPException(status_code=503, detail="no model loaded")
        config = GenerationConfig(
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop_sequences=req.stop or [],
            seed=req.seed,
        )
        result = gen.generate(req.prompt, config)
        return GenerateResponse(
            text=result.text,
            n_input_tokens=result.n_input_tokens,
            n_output_tokens=result.n_output_tokens,
            elapsed_seconds=result.elapsed_seconds,
            tokens_per_second=result.tokens_per_second,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        gen = app.state.generator
        if gen is None:
            raise HTTPException(status_code=503, detail="no model loaded")
        # Flatten messages into a prompt.  Simple format — JEXI OS
        # and other clients can call this without learning a special
        # chat template.
        parts: List[str] = []
        for msg in req.messages:
            if msg.role == "system":
                parts.append(f"<system> {msg.content}")
            elif msg.role == "user":
                parts.append(f"<user> {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<assistant> {msg.content}")
        parts.append("<assistant>")
        prompt = "\n".join(parts)
        config = GenerationConfig(
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            stop_sequences=["<user>", "<system>"],
        )
        result = gen.generate(prompt, config)
        return ChatResponse(
            role="assistant",
            content=result.text.strip(),
            n_input_tokens=result.n_input_tokens,
            n_output_tokens=result.n_output_tokens,
            elapsed_seconds=result.elapsed_seconds,
        )

    return app


def create_app(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> FastAPI:
    """Build the app, loading a model + tokenizer if paths are provided."""
    if checkpoint_path and tokenizer_path:
        generator = EINXGenerator.from_checkpoint(checkpoint_path, tokenizer_path)
        return build_app(generator)
    return build_app(None)
