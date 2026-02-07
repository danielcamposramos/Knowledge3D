#!/usr/bin/env python3
"""Merge multiple drawing enrichment JSONL files with confidence-aware dedup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _confidence(entry: dict) -> float:
    try:
        return float(entry.get("metadata", {}).get("confidence", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge drawing enrichment JSONL files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL paths")
    parser.add_argument("--output", required=True, help="Merged JSONL output")
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    for in_path_str in args.inputs:
        in_path = Path(in_path_str)
        if not in_path.exists():
            continue
        for line in in_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                continue
            prev = merged.get(row_id)
            if prev is None or _confidence(row) > _confidence(prev):
                merged[row_id] = row

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(merged.values(), key=_confidence, reverse=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")

    print(f"merged {len(rows)} entries -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

