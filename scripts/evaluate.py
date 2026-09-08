#!/usr/bin/env python3
"""Evaluate an EINX checkpoint. See `python -m einx.cli evaluate --help`."""
import sys
from einx.cli import evaluate_main
sys.exit(evaluate_main())
