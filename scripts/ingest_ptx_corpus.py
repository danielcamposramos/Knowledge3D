#!/usr/bin/env python3
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import pypdf
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION = "k3d_ptx"
DIM = 384
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_NAME = "fast-all-minilm-l6-v2"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
API_KEY = os.getenv("QDRANT_API_KEY", "@20Cooool58")
CORPUS = Path("/mnt/arquivos/0 ChatGPTs/DataBase/Programming Languages/PTX/")
CHUNK = 1800
BATCH = 64
SEC_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+[A-Z]")
NS = uuid.NAMESPACE_URL

VERSION_MAP = {
    "ptx_isa_9.0": "ptx_isa_9.0",
    "PTX ISA - ptx_isa_9.0": "ptx_isa_9.0_annotated",
    "ptx_isa_8.7": "ptx_isa_8.7",
    "ptx_isa_8.5": "ptx_isa_8.5",
    "Inline_PTX_Assembly": "inline_ptx_asm",
    "CUDA_C_Programming_Guide": "cuda_c_guide",
}


def load_pages(pdf: Path) -> list[str]:
    js = pdf.with_suffix(pdf.suffix + ".json")
    if js.exists():
        data = json.loads(js.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(p, dict) and "text" in p for p in data):
            return [p["text"] for p in data if p["text"].strip()]
    return [p.extract_text() or "" for p in pypdf.PdfReader(str(pdf)).pages]


def _split_long_text(text: str) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(normalized) <= CHUNK:
        return [normalized] if normalized else []

    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > CHUNK:
        split_at = remaining.rfind(" ", 0, CHUNK + 1)
        if split_at <= 0:
            split_at = CHUNK
        piece = remaining[:split_at].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def chunk_sections(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_section = "preamble"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_section, current_lines[:]))
        current_lines = []

    for line in lines:
        if SEC_RE.match(line):
            flush()
            current_section = " ".join(line.split())
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()

    chunked: list[dict[str, Any]] = []
    for section_name, section_lines in sections:
        body = "\n".join(section_lines).strip()
        pieces = _split_long_text(body)
        for idx, piece in enumerate(pieces):
            rendered_section = section_name if idx == 0 else f"{section_name}.cont"
            chunked.append({
                "section": rendered_section,
                "chunk_idx": idx,
                "text": piece,
            })
    return chunked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    embedder = TextEmbedding(model_name=MODEL)
    client = QdrantClient(url=QDRANT_URL, api_key=API_KEY, timeout=300.0)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            COLLECTION,
            vectors_config={
                VECTOR_NAME: VectorParams(size=DIM, distance=Distance.COSINE),
            },
            timeout=300,
        )

    for pdf in sorted(CORPUS.glob("*.pdf")):
        t0 = time.time()
        stem = pdf.stem
        version = VERSION_MAP.get(stem, stem.lower().replace(" ", "_"))
        source = pdf.name
        text = "\n".join(load_pages(pdf))
        chunks = chunk_sections(text)

        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            vecs = [vec.tolist() for vec in embedder.embed([c["text"] for c in batch])]
            points = [
                PointStruct(
                    id=str(uuid.uuid5(NS, f"{source}:{chunk['section']}:{chunk['chunk_idx']}")),
                    vector={VECTOR_NAME: vec},
                    payload={
                        "source": source,
                        "section": chunk["section"],
                        "chunk_idx": chunk["chunk_idx"],
                        "text": chunk["text"],
                        "document": chunk["text"],
                        "version": version,
                    },
                )
                for chunk, vec in zip(batch, vecs)
            ]
            client.upsert(collection_name=COLLECTION, points=points, wait=True)

        logging.info("%-40s chunks=%d elapsed=%.1fs", pdf.name, len(chunks), time.time() - t0)


if __name__ == "__main__":
    main()
