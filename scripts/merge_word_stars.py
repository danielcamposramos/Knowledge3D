#!/usr/bin/env python3
"""
Merge multiple word star JSONL files into a single deduplicated file.

Deduplication key: (lang, lemma)
If fields collide, later files override earlier ones for matching keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge word star JSONL files.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files.")
    p.add_argument("--output", required=True, help="Output merged JSONL file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merged: Dict[Tuple[str, str], dict] = {}

    for path_str in args.inputs:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                star = json.loads(line)
                key = (star.get("lang", ""), star.get("lemma", ""))
                merged[key] = star  # later wins

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for star in merged.values():
            out_f.write(json.dumps(star, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
