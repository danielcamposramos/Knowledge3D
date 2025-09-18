"""Build a temporal understanding corpus from the Architect's time libraries."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PARA_SPLIT = re.compile(r"\n{2,}")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
DATE_PATTERN = re.compile(r"\b(\d{3,4})\b")
KEYWORDS = [
    "epoch", "era", "timeline", "chronology", "calendar", "duration",
    "frequency", "sequence", "revolution", "time", "moment", "period",
    "cycle", "phase", "temporal", "chronicle"
]
UNWANTED = (
    "copyright",
    "publisher",
    "press",
    "isbn",
    "all rights reserved",
    "©",
)


def split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    for chunk in SENTENCE_PATTERN.split(text):
        sentence = chunk.strip()
        if len(sentence) < 40 or len(sentence) > 480:
            continue
        if sentence.count(" ") < 3:
            continue
        lowered = sentence.lower()
        if any(token in lowered for token in UNWANTED):
            continue
        sentences.append(sentence)
    return sentences


def select_fact(sentence: str) -> Optional[Dict[str, str]]:
    years = DATE_PATTERN.findall(sentence)
    if years:
        year = years[0]
        question = f"What significant event happened in {year}?"
        return {"question": question, "answer": sentence}
    for keyword in KEYWORDS:
        if keyword.lower() in sentence.lower():
            question = f"Explain the concept of '{keyword}' as described."
            return {"question": question, "answer": sentence}
    return None


def extract_text_blocks(obj: object) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        if "content" in obj:
            yield from extract_text_blocks(obj["content"])
        else:
            for value in obj.values():
                yield from extract_text_blocks(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from extract_text_blocks(item)


def build_corpus(source_dir: Path, output: Path, limit: int = 2000) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as fh:
        for json_path in sorted(source_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for block in extract_text_blocks(data):
                paragraphs = [blk.strip() for blk in PARA_SPLIT.split(block) if blk.strip()]
                for para in paragraphs:
                    for sentence in split_sentences(para):
                        fact = select_fact(sentence)
                        if not fact:
                            continue
                        record: Dict[str, str] = {
                            "question": fact["question"],
                            "answer": fact["answer"],
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
            if total >= limit:
                break
    print(f"⏳ Built time corpus with {total} records → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal understanding corpus")
    parser.add_argument("--source", required=True, help="Directory with time JSON files")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()
    build_corpus(Path(args.source), Path(args.output), limit=args.limit)


if __name__ == "__main__":
    main()
