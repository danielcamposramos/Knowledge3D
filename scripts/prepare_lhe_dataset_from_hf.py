#!/usr/bin/env python3
"""Prepare Last Humanity Exam dataset from Hugging Face into K3D benchmark format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _extract_question_text(row: dict[str, Any]) -> str:
    for key in ("question_text", "question", "prompt", "query"):
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _extract_options(row: dict[str, Any]) -> list[str]:
    raw = row.get("options")
    if isinstance(raw, list):
        return [str(item) for item in raw]

    for key in ("choices", "choice", "answers"):
        value = row.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, dict):
            # Common HF format: {"label": ["A","B"], "text": ["...","..."]}
            if isinstance(value.get("text"), list):
                return [str(item) for item in value["text"]]
            # Fallback: deterministic key order to stable option list.
            try:
                return [str(value[k]) for k in sorted(value.keys())]
            except Exception:
                pass
    return []


def _extract_answer(row: dict[str, Any], options: list[str]) -> str | None:
    for key in ("correct_answer", "answer", "answer_text", "gold"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    for key in ("answer_idx", "label", "target"):
        value = row.get(key)
        if isinstance(value, int) and 0 <= value < len(options):
            return str(options[value]).strip()

    for key in ("answer_key",):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            letter = value.strip().upper()
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(options):
                return str(options[idx]).strip()
            return letter
    return None


def _extract_domain(row: dict[str, Any]) -> str:
    for key in ("domain", "subject", "category", "topic"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "multi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cais/hle", help="HF dataset id")
    parser.add_argument("--split", default="test", help="HF split name")
    parser.add_argument(
        "--output-dir",
        default="/K3D/Knowledge3D.local/datasets/last_humanity_exam",
        help="Directory to write questions.json for K3D benchmark loader",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to convert (0 = all)")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows: list[dict[str, Any]] = []
    dropped = 0
    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            dropped += 1
            continue

        question_text = _extract_question_text(row)
        options = _extract_options(row)
        answer = _extract_answer(row, options)
        if not question_text or answer is None:
            dropped += 1
            continue

        rows.append(
            {
                "id": str(row.get("id") or f"hle_{idx}"),
                "domain": _extract_domain(row),
                "question_text": question_text,
                "options": [str(item) for item in options] if options else [],
                "correct_answer": str(answer),
                "question_type": "multiple_choice" if options else "open_ended",
                "answer_type": str(row.get("answer_type", "")).strip() or None,
                "category": str(row.get("category", "")).strip() or None,
                "subject": str(row.get("raw_subject", "")).strip() or None,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "questions.json"
    output_path.write_text(json.dumps({"questions": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Dataset source: {args.dataset} [{args.split}]")
    print(f"Rows converted: {len(rows)}")
    print(f"Rows dropped:   {dropped}")
    print(f"Output:         {output_path}")


if __name__ == "__main__":
    main()
