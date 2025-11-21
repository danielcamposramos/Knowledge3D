#!/usr/bin/env python3
"""
Compress word stars' meaning_program into a compact PD-style payload.

This is a lightweight stand-in for PD02/PD04: we apply zlib + base64 to the
meaning_program string and emit `meaning_pd` while keeping morph_rpn and the
core metadata. Input is the JSONL produced by ingest_ud_word_stars.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack word stars meaning_program into PD-compressed field.")
    p.add_argument("--input", required=True, help="Input JSONL of word stars.")
    p.add_argument("--output", required=True, help="Output JSONL with meaning_pd.")
    return p.parse_args()


def pd_pack(text: str) -> str:
    """Zlib + base64 encode a UTF-8 text payload (deterministic, no headers)."""
    compressed = zlib.compress(text.encode("utf-8"), level=9)
    return base64.b64encode(compressed).decode("ascii")


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            star = json.loads(line)
            mp = star.get("meaning_program", "")
            star["meaning_pd"] = pd_pack(mp)
            fout.write(json.dumps(star, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
