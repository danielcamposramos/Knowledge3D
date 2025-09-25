"""Ingest Wikipedia dumps into Knowledge3D corpus JSONL files."""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from urllib.parse import quote
import xml.etree.ElementTree as ET

DEFAULT_MAX_CHARS = 1400
DEFAULT_OVERLAP = 200
LICENSE = "CC BY-SA 3.0"
NAMESPACE = "{http://www.mediawiki.org/xml/export-0.10/}"


@dataclass
class WikiChunk:
    title: str
    section: str
    chunk_index: int
    total_chunks: int
    text: str
    language: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia dump into Knowledge3D corpus")
    parser.add_argument("--input", required=True, type=Path, help="Path to Wikipedia XML (bz2 or xml)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("viewer/public/galaxy/working"),
        help="Output directory for generated corpus files",
    )
    parser.add_argument("--language", default="en", help="ISO language code for the dump")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per chunk before splitting",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Character overlap between consecutive chunks",
    )
    parser.add_argument(
        "--limit-articles",
        type=int,
        default=None,
        help="Optional cap on number of articles for pilot runs",
    )
    return parser.parse_args()


def iter_wikipedia_pages(path: Path) -> Iterator[Tuple[str, str]]:
    if path.suffix == ".bz2":
        handle = bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    else:
        handle = path.open("r", encoding="utf-8", errors="ignore")

    buffer = ""
    chunk_size = 1024 * 1024  # 1 MB
    parse_errors = 0
    max_logged_errors = 5
    try:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            while True:
                start = buffer.find("<page>")
                if start == -1:
                    break
                end = buffer.find("</page>", start)
                if end == -1:
                    # keep buffer for next read
                    buffer = buffer[start:]
                    break
                page_xml = buffer[start : end + len("</page>")]
                buffer = buffer[end + len("</page>") :]
                try:
                    elem = ET.fromstring(page_xml)
                except ET.ParseError as exc:
                    if parse_errors < max_logged_errors:
                        print(
                            f"⚠️  Failed to parse page chunk: {exc}",
                            file=sys.stderr,
                        )
                    parse_errors += 1
                    continue
                title = elem.findtext('.//{*}title') or ""
                text = elem.findtext('.//{*}revision/{*}text') or ""
                if title and text:
                    yield title, text
        # process remaining buffer
        while True:
            start = buffer.find("<page>")
            end = buffer.find("</page>", start)
            if start == -1 or end == -1:
                break
            page_xml = buffer[start : end + len("</page>")]
            buffer = buffer[end + len("</page>") :]
            try:
                elem = ET.fromstring(page_xml)
            except ET.ParseError as exc:
                if parse_errors < max_logged_errors:
                    print(
                        f"⚠️  Failed to parse trailing page chunk: {exc}",
                        file=sys.stderr,
                    )
                parse_errors += 1
                continue
            title = elem.findtext('.//{*}title') or ""
            text = elem.findtext('.//{*}revision/{*}text') or ""
            if title and text:
                yield title, text
    finally:
        handle.close()


def clean_markup(text: str) -> str:
    # remove basic wiki markup (links, templates) using heuristics
    text = re.sub(r"\{\{[^{}]+\}\}", "", text)
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^[\]]+)\]\]", r"\1", text)
    text = re.sub(r"==+\s*(.+?)\s*==+", r"\n\1\n", text)
    text = re.sub(r"'''+([^']+)'''+", r"\1", text)
    text = re.sub(r"''([^']+)''", r"\1", text)
    return text.strip()


def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for para in paragraphs:
        para_len = len(para)
        if length + para_len + 2 > max_chars and current:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if overlap > 0 and chunks:
                tail = chunk[-overlap:]
                current = [tail + "\n\n" + para]
                length = len(current[0])
            else:
                current = [para]
                length = para_len
        else:
            current.append(para)
            length += para_len + 2
    if current:
        chunk = "\n\n".join(current).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def build_chunks(
    articles: Iterator[Tuple[str, str]],
    language: str,
    max_chars: int,
    overlap: int,
    limit: Optional[int] = None,
) -> Iterator[WikiChunk]:
    for idx, (title, raw_text) in enumerate(articles):
        if limit is not None and idx >= limit:
            break
        cleaned = clean_markup(raw_text)
        if not cleaned:
            continue
        if cleaned.lstrip().lower().startswith("#redirect"):
            continue
        text_chunks = chunk_text(cleaned, max_chars=max_chars, overlap=overlap)
        if not text_chunks:
            continue
        url = f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        total = len(text_chunks)
        for chunk_index, chunk in enumerate(text_chunks):
            yield WikiChunk(
                title=title,
                section="main",
                chunk_index=chunk_index,
                total_chunks=total,
                text=chunk,
                language=language,
                url=url,
            )


def write_outputs(chunks: Iterable[WikiChunk], output_dir: Path, language: str) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "wikipedia_corpus.jsonl"
    manifest_path = output_dir / "wikipedia_manifest.json"

    article_count = 0
    chunk_count = 0
    current_article = None

    with corpus_path.open("w", encoding="utf-8") as corpus_file:
        for chunk in chunks:
            chunk_count += 1
            if current_article != chunk.title:
                article_count += 1
                current_article = chunk.title
            question = (
                f"Summarize the excerpt from Wikipedia article '{chunk.title}' "
                f"(chunk {chunk.chunk_index + 1} of {chunk.total_chunks})."
            )
            payload = {
                "question": question,
                "answer": chunk.text,
                "source": {
                    "title": chunk.title,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "language": chunk.language,
                    "url": chunk.url,
                    "license": LICENSE,
                },
            }
            corpus_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = {
        "language": language,
        "articles": article_count,
        "chunks": chunk_count,
        "source": "Wikimedia dumps",
        "license": LICENSE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return article_count, chunk_count


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        return 1

    print(f"📥 Ingesting Wikipedia dump: {args.input}")
    articles = iter_wikipedia_pages(args.input)
    chunks = build_chunks(
        articles,
        language=args.language,
        max_chars=args.max_chars,
        overlap=args.overlap,
        limit=args.limit_articles,
    )
    articles_written, chunks_written = write_outputs(chunks, args.output_dir, args.language)
    print(
        f"✅ Wikipedia ingest complete — {articles_written} articles, {chunks_written} chunks → "
        f"{args.output_dir / 'wikipedia_corpus.jsonl'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
