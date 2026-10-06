#!/usr/bin/env python3
"""Backward-compatible entry point: `python cli.py ...` == `openraw ...`.
The real CLI lives in openraw/cli.py (installed as the `openraw` command)."""
import sys

from openraw.cli import main

if __name__ == "__main__":
    sys.exit(main())
