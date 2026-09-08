# -*- coding: utf-8 -*-
"""EINX inference engine.

Loads a trained checkpoint and generates text.  Implements temperature,
top-k, top-p, repetition penalty, stop sequences.  Supports streaming
generation (yields tokens one at a time) for responsive UX.
"""

from einx.inference.generator import EINXGenerator, GenerationConfig, GenerationResult

__all__ = ["EINXGenerator", "GenerationConfig", "GenerationResult"]
