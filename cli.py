#!/usr/bin/env python3
"""Backward-compatible entry point: `python cli.py ...` == `pseudoraw ...`.
The real CLI lives in pseudoraw/cli.py (installed as the `pseudoraw` command)."""
import sys

from pseudoraw.cli import main

if __name__ == "__main__":
    sys.exit(main())
