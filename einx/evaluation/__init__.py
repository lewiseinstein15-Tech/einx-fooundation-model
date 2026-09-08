# -*- coding: utf-8 -*-
"""EINX evaluation framework.

Computes real metrics — never fabricates benchmark numbers.  Every
result records the model version, checkpoint path, dataset version,
and date so evals are reproducible.
"""

from einx.evaluation.evaluator import EINXEvaluator, EvalResult, run_evaluation

__all__ = ["EINXEvaluator", "EvalResult", "run_evaluation"]
