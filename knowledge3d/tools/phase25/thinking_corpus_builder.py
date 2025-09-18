"""Generate an Algorithmic Thinking corpus from curated 'How to think' JSON assets."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ]{4,}")


def split_sentences(text: str) -> Iterable[str]:
    buffer = []
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        sentence = chunk.strip()
        if len(sentence) < 40 or len(sentence) > 320:
            continue
        if sentence.count(" ") < 3:
            continue
        buffer.append(sentence)
    return buffer


def select_keyword(sentence: str) -> Optional[str]:
    candidates = WORD_PATTERN.findall(sentence)
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def create_cloze(sentence: str, keyword: str) -> str:
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub("_____", sentence, count=1)


def extract_entries_from_json(obj: object) -> Iterable[Tuple[str, Optional[str]]]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj, None
    elif isinstance(obj, dict):
        page = str(obj.get("page")) if obj.get("page") is not None else None
        content = obj.get("content")
        if content is not None:
            for item in extract_entries_from_json(content):
                yield item
        else:
            for value in obj.values():
                for item in extract_entries_from_json(value):
                    yield item
    elif isinstance(obj, list):
        for item in obj:
            for entry in extract_entries_from_json(item):
                yield entry


def build_corpus(source_dir: Path, output: Path, limit: int = 500) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as fh:
        for json_path in sorted(source_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for text_block, _ in extract_entries_from_json(data):
                for sentence in split_sentences(text_block):
                    keyword = select_keyword(sentence)
                    if not keyword:
                        continue
                    question = create_cloze(sentence, keyword)
                    record: Dict[str, object] = {
                        "question": f"Fill in the missing concept: {question}",
                        "answer": keyword,
                        "source_file": json_path.name,
                        "sentence": sentence,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    if total >= limit:
                        break
                if total >= limit:
                    break
            if total >= limit:
                break
    print(f"📘 Built thinking corpus with {total} records → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Algorithmic Thinking corpus")
    parser.add_argument("--source", required=True, help="Directory containing How-to-think JSON files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    build_corpus(Path(args.source), Path(args.output), limit=args.limit)


if __name__ == "__main__":
    main()
