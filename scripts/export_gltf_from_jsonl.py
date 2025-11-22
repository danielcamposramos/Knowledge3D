#!/usr/bin/env python3
"""
Export minimal glTF files (extras.k3d only) from JSONL star files.

Usage:
  python scripts/export_gltf_from_jsonl.py --input <stars.jsonl> --output <out.gltf>

This is a stopgap until full ctypes/GLB pipeline is in place; geometry is omitted
(procedural-first), nodes carry extras.k3d.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export minimal glTF from JSONL stars (extras.k3d only).")
    ap.add_argument("--input", type=Path, required=True, help="Input JSONL of stars.")
    ap.add_argument("--output", type=Path, required=True, help="Output glTF path.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = []
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stars.append(json.loads(line))

    gltf = {
        "asset": {"version": "2.0", "generator": "k3d-export-jsonl"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(stars)))}],
        "nodes": [],
    }
    for idx, star in enumerate(stars):
        gltf["nodes"].append(
            {
                "name": star.get("id")
                or star.get("letter_concept")
                or star.get("meaning_id")
                or star.get("symbol_concept")
                or star.get("phrase_id")
                or f"star_{idx}",
                "extras": {"k3d": star},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gltf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(stars)} stars to {args.output}")


if __name__ == "__main__":
    main()
