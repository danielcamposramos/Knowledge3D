"""Build a self-reflection corpus from curated self-reflection libraries."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
KEY_PATTERN = re.compile(r"\b(honesty|integrity|reflection|memory|trust|lie|truth|self|awareness|reason|instruction)\b", re.IGNORECASE)


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for chunk in SENTENCE_PATTERN.split(text):
        sentence = chunk.strip()
        if len(sentence) < 40 or len(sentence) > 420:
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


def build_corpus(source_dir: Path, output: Path, limit: int = 2000) -> None:
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
                    if not KEY_PATTERN.search(sentence):
                        continue
                    question = f"Reflect on this idea: {sentence}"
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
    print(f"🪞 Built self-reflection corpus with {total} records → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-reflection corpus")
    parser.add_argument("--source", required=True, help="Directory with self-reflection JSON files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    build_corpus(Path(args.source), Path(args.output), limit=args.limit)


if __name__ == "__main__":
    main()
