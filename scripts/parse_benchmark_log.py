#!/usr/bin/env python3
"""
Parse benchmark stdout logs for progress and final summary blocks.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional


PROGRESS_RE = re.compile(r"Progress:\s+(\d+)/(\d+)\s+\(([^)]+)\)")


def _find_last_progress(lines: List[str]) -> Optional[str]:
    for line in reversed(lines):
        if PROGRESS_RE.search(line):
            return line.strip()
    return None


def _find_last_dataset_block(lines: List[str]) -> Optional[List[str]]:
    idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Dataset:"):
            idx = i
            break
    if idx is None:
        return None
    block: List[str] = []
    for line in lines[idx:]:
        if not line.strip() and block:
            break
        block.append(line.rstrip())
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse benchmark log for progress and summary.")
    parser.add_argument("log_path", help="Path to benchmark log file.")
    args = parser.parse_args()

    path = Path(args.log_path)
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        print("Log is empty.")
        return

    dataset_block = _find_last_dataset_block(lines)
    if dataset_block:
        print("\n".join(dataset_block))
        return

    progress_line = _find_last_progress(lines)
    if progress_line:
        print(progress_line)
        return

    print("No progress or dataset summary found yet.")


if __name__ == "__main__":
    main()
