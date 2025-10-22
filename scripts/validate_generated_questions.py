#!/usr/bin/env python3
"""
Validate generated RLWHF questions for quality and diversity.

Computes dataset statistics (difficulty mix, source spread, lengths,
duplicates) to ensure the question pool is balanced before student /
teacher processing.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


def _load_questions(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    questions: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def _describe_lengths(values: Sequence[int]) -> str:
    if not values:
        return "  (no data)"
    return (
        f"  Min: {min(values)} words\n"
        f"  Max: {max(values)} words\n"
        f"  Avg: {statistics.mean(values):.1f} words\n"
        f"  Std: {statistics.pstdev(values):.1f} words"
    )


def validate_questions(path: Path) -> None:
    questions = _load_questions(path)
    total = len(questions)

    print("📊 Question Dataset Statistics")
    print("=" * 50)
    print(f"Total questions: {total}")

    if not questions:
        return

    difficulties = Counter(q.get("difficulty", "unknown") for q in questions)
    print("\nDifficulty distribution:")
    for diff, count in difficulties.most_common():
        pct = (count / total) * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")

    sources = Counter(q.get("pdf_name", "unknown") for q in questions)
    print("\nTop 10 sources:")
    for source, count in sources.most_common(10):
        print(f"  {source}: {count}")

    q_lengths = [len(str(q.get("question", "")).split()) for q in questions]
    a_lengths = [len(str(q.get("answer", "")).split()) for q in questions]

    print("\nQuestion length:")
    print(_describe_lengths(q_lengths))

    print("\nAnswer length:")
    print(_describe_lengths(a_lengths))

    question_texts = [str(q.get("question", "")).strip().lower() for q in questions]
    duplicates = total - len(set(question_texts))
    dup_pct = (duplicates / total) * 100 if total else 0.0
    print(f"\nDuplicate questions: {duplicates} ({dup_pct:.1f}%)")

    print("\n✅ Validation complete!")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to generated questions JSONL")
    args = parser.parse_args()

    validate_questions(args.input)


if __name__ == "__main__":
    main()
