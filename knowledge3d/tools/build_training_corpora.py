"""Generate time and math training corpora from curated JSON sources."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_MAX_LEN = 600


def _iter_text_fragments(obj) -> Iterator[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        text = obj.strip()
        if text:
            yield text
        return
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_text_fragments(value)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_text_fragments(item)
        return


def _chunk_text(text: str, max_len: int = DEFAULT_MAX_LEN) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: List[str] = []
    buffer: List[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        candidate_len = current_len + len(sentence) + (1 if buffer else 0)
        if candidate_len > max_len and buffer:
            chunks.append(" ".join(buffer).strip())
            buffer = [sentence]
            current_len = len(sentence)
        else:
            buffer.append(sentence)
            current_len = candidate_len
    if buffer:
        chunks.append(" ".join(buffer).strip())
    return chunks


def _build_entries(source_path: Path, topic: str, max_len: int) -> List[Dict[str, str]]:
    try:
        data = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    fragments = list(_iter_text_fragments(data))
    entries: List[Dict[str, str]] = []
    base_name = source_path.stem.replace("_", " ")
    for idx, fragment in enumerate(fragments):
        for chunk in _chunk_text(fragment, max_len=max_len):
            question = (
                f"Summarize the excerpt from '{base_name}' about {topic}."
            )
            answer = chunk
            entries.append({
                "question": question,
                "answer": answer,
                "source_file": source_path.name,
            })
    return entries


def build_corpus(
    sources: Sequence[Tuple[str, str]],
    output: Path,
    max_len: int,
    limit: Optional[int] = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, str]] = []
    for directory, topic in sources:
        for path in sorted(Path(directory).glob("*.json")):
            entries.extend(_build_entries(path, topic, max_len=max_len))

    if not entries:
        raise RuntimeError(f"No entries generated for {output}")

    if limit is not None and limit > 0:
        entries = entries[:limit]

    with output.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated training corpora")
    parser.add_argument("--time-out", default="viewer/public/galaxy/working/time_corpus.jsonl")
    parser.add_argument("--math-out", default="viewer/public/galaxy/working/math_corpus.jsonl")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="Maximum characters per answer chunk")
    parser.add_argument("--time-limit", type=int, default=1200, help="Maximum entries in time corpus")
    parser.add_argument("--math-limit", type=int, default=3000, help="Maximum entries in math corpus")
    parser.add_argument(
        "--time-sources",
        nargs="*",
        default=[
            "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Understand Time/JSON",
        ],
        help="Directories containing time JSON sources",
    )
    parser.add_argument(
        "--math-sources",
        nargs="*",
        default=[
            "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/BasicMath/JSON",
            "/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Advanced Maths/Financial Math/JSON",
        ],
        help="Directories containing math JSON sources",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI utility
    args = parse_args()
    time_sources = [(path, "time") for path in args.time_sources]
    math_sources = [(path, "mathematics") for path in args.math_sources]
    build_corpus(time_sources, Path(args.time_out), args.max_len, args.time_limit)
    build_corpus(math_sources, Path(args.math_out), args.max_len, args.math_limit)


if __name__ == "__main__":  # pragma: no cover
    main()
