"""Build a pedagogy-focused corpus from the teaching library."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
KEYWORDS = [
    "teach", "learning", "instruction", "strategy", "student", "assessment",
    "pedagogy", "practice", "lesson", "feedback", "classroom", "motivation",
    "curriculum", "approach", "skill"
]
UNWANTED = (
    "copyright",
    "publisher",
    "press",
    "isbn",
    "©",
    "all rights reserved",
)


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for chunk in SENTENCE_PATTERN.split(text):
        sentence = chunk.strip()
        if len(sentence) < 50 or len(sentence) > 450:
            continue
        lowered = sentence.lower()
        if any(token in lowered for token in UNWANTED):
            continue
        sentences.append(sentence)
    return sentences


def extract_text(obj: object) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        if "content" in obj:
            yield from extract_text(obj["content"])
        else:
            for value in obj.values():
                yield from extract_text(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_text(item)


def build_corpus(source_dir: Path, output: Path, limit: int = 2500) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as fh:
        for json_path in sorted(source_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for block in extract_text(data):
                for sentence in split_sentences(block):
                    if not any(keyword in sentence.lower() for keyword in KEYWORDS):
                        continue
                    question = f"How would you teach this concept? {sentence}"
                    record: Dict[str, str] = {
                        "question": question,
                        "answer": sentence,
                        "source_file": json_path.name,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    if total >= limit:
                        break
                if total >= limit:
                    break
            if total >= limit:
                break
    print(f"📚 Built teaching corpus with {total} records → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teaching corpus")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=2500)
    args = parser.parse_args()
    build_corpus(Path(args.source), Path(args.output), limit=args.limit)


if __name__ == "__main__":
    main()
