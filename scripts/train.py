#!/usr/bin/env python3
"""Train an EINX model. See `python -m einx.cli train --help`."""
import sys
from einx.cli import train_main
sys.exit(train_main())
