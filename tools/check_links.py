#!/usr/bin/env python3
"""Check that every relative link and image in the repo's Markdown points at
a file that exists (external http(s) links are not fetched).

    python tools/check_links.py        # exit code 1 if anything is broken
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s]+)")                  # [text](target) and ![alt](target)
HTML = re.compile(r"""(?:src|srcset|href)="([^"]+)\"""")  # <img src="..."> etc.


def markdown_files() -> list[Path]:
    try:  # tracked files only, when in a git checkout
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.split()]
    except (OSError, subprocess.CalledProcessError):
        files = list(ROOT.rglob("*.md"))
    return [f for f in files if "third_party" not in f.parts]


def broken_links(md: Path) -> list[str]:
    bad = []
    text = md.read_text(encoding="utf-8")
    for target in LINK.findall(text) + HTML.findall(text):
        target = target.split("#", 1)[0]
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):  # anchors, http:, mailto:
            continue
        if not (md.parent / target).exists():
            bad.append(target)
    return bad


def main() -> int:
    failures = 0
    files = markdown_files()
    for md in files:
        for target in broken_links(md):
            print(f"{md.relative_to(ROOT)}: broken link -> {target}")
            failures += 1
    print(f"checked {len(files)} Markdown files: {failures} broken link(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
