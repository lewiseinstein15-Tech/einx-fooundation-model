#!/usr/bin/env python3
"""Start the EINX HTTP API server. See `python -m einx.cli serve --help`."""
import sys
from einx.cli import serve_main
sys.exit(serve_main())
