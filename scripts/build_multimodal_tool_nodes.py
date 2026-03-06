#!/usr/bin/env python3
"""Build unified Tool-galaxy payloads for always-on procedural means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge3d.knowledgeverse.tool_galaxy import build_tool_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified Tool-galaxy payload JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../Knowledge3D.local/fundamental_augmentation/tool_nodes_phase0.jsonl"),
        help="Output JSONL file for Tool galaxy payloads.",
    )
    args = parser.parse_args()

    rows = build_tool_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[tool-nodes] rows={len(rows)}")
    print(f"[tool-nodes] output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
