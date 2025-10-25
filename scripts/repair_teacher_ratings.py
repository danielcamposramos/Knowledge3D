#!/usr/bin/env python3
"""
Reprocess teacher evaluation JSONL to fix rating labels and numeric scores.

Usage:
    python scripts/repair_teacher_ratings.py \
        --input /path/to/teacher_evaluations.jsonl \
        --output /path/to/teacher_evaluations_fixed.jsonl

Use --in-place to overwrite the input file safely (writes to temp file first).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from knowledge3d.training.rlwhf.teacher_eval_ollama import extract_rating


def process_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            evaluation = record.get("teacher_evaluation")
            if evaluation:
                response = evaluation.get("teacher_response", "")
                label, score = extract_rating(response)
                evaluation["rating"] = label
                evaluation["rating_score"] = score
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to existing teacher evaluations JSONL")
    parser.add_argument("--output", type=Path, help="Destination path for repaired file")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file in-place")
    args = parser.parse_args()

    input_path = args.input
    if args.in_place and args.output:
        parser.error("Specify either --output or --in-place, not both.")

    if args.in_place:
        tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
        process_file(input_path, tmp_path)
        tmp_path.replace(input_path)
        print(f"Repaired ratings written in-place to {input_path}")
    else:
        if not args.output:
            parser.error("Provide --output when not using --in-place.")
        process_file(input_path, args.output)
        print(f"Repaired ratings written to {args.output}")


if __name__ == "__main__":
    main()
