#!/usr/bin/env python3
"""Generate text from an EINX checkpoint. See `python -m einx.cli generate --help`."""
import sys
from einx.cli import generate_main
sys.exit(generate_main())
