#!/usr/bin/env python3
"""Download and normalize the Last Humanity Exam corpus for local benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


CHOICE_SPLIT_RE = re.compile(r"\n?\s*Answer Choices:\s*", re.IGNORECASE)
CHOICE_LINE_RE = re.compile(r"^\s*([A-Z])\.\s*(.*)$")


def parse_answer_choices(question: str) -> tuple[str, list[str], dict[str, str]]:
    text = str(question or "").strip()
    if not text:
        return "", [], {}
    parts = CHOICE_SPLIT_RE.split(text, maxsplit=1)
    if len(parts) == 1:
        return text, [], {}

    stem = parts[0].strip()
    raw_choices = parts[1]
    options: list[str] = []
    choice_map: dict[str, str] = {}
    current_label: str | None = None
    current_chunks: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_chunks
        if current_label is None:
            return
        option_text = " ".join(chunk for chunk in current_chunks if chunk).strip()
        if option_text:
            choice_map[current_label] = option_text
            options.append(option_text)
        current_label = None
        current_chunks = []

    for line in raw_choices.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = CHOICE_LINE_RE.match(stripped)
        if match:
            flush()
            current_label = match.group(1).strip()
            current_chunks = [match.group(2).strip()]
            continue
        if current_label is not None:
            current_chunks.append(stripped)
    flush()
    if not options:
        return text, [], {}
    return stem, options, choice_map


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    raw_question = str(record.get("question") or "").strip()
    answer = str(record.get("answer") or "").strip()
    answer_type = str(record.get("answer_type") or "").strip()
    question_text, options, choice_map = parse_answer_choices(raw_question)
    has_image = bool(record.get("image")) or bool(record.get("image_preview"))

    question_type = "open_ended"
    correct_answer = answer
    if answer_type == "multipleChoice" and options:
        question_type = "multiple_choice"
        correct_answer = choice_map.get(answer, answer)

    return {
        "id": str(record.get("id") or ""),
        "domain": str(record.get("raw_subject") or record.get("category") or "multi"),
        "category": str(record.get("category") or ""),
        "question_text": question_text or raw_question,
        "options": options,
        "correct_answer": correct_answer,
        "question_type": question_type,
        "answer_type": answer_type,
        "has_image": has_image,
        "author_name": str(record.get("author_name") or ""),
        "source_dataset": "cais/hle",
        "source_split": "test",
        "source_canary": str(record.get("canary") or ""),
    }


def build_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    answer_types = Counter(str(row.get("answer_type") or "") for row in records)
    question_types = Counter(str(row.get("question_type") or "") for row in records)
    domains = Counter(str(row.get("domain") or "multi") for row in records)
    image_count = sum(1 for row in records if row.get("has_image"))
    return {
        "source_dataset": "cais/hle",
        "source_split": "test",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(records),
        "image_questions": image_count,
        "text_only_questions": len(records) - image_count,
        "answer_type_counts": dict(answer_types),
        "question_type_counts": dict(question_types),
        "top_domains": domains.most_common(20),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/K3D/K3D_llama_cpp/datasets/last_humanity_exam",
        help="Directory to store the normalized HLE corpus.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("cais/hle", split="test")
    records = [normalize_record(dict(row)) for row in dataset]
    manifest = build_manifest(records)

    payload = {
        "source_dataset": "cais/hle",
        "source_split": "test",
        "questions": records,
    }
    dataset_path = output_dir / "last_humanity_exam.json"
    manifest_path = output_dir / "manifest.json"
    dataset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"dataset_path": str(dataset_path), "manifest_path": str(manifest_path), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
