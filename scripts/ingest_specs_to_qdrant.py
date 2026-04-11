#!/usr/bin/env python3
"""
Ingest K3D specification files into Qdrant for semantic search by all agents.

Usage:
    pip install qdrant-client fastembed
    python scripts/ingest_specs_to_qdrant.py

Connects to Qdrant at localhost:6333, recreates collection k3d_specifications
with the named vector expected by MCP FastEmbedProvider, chunks all
docs/vocabulary/*.md files by headers, embeds with FastEmbed
(sentence-transformers/all-MiniLM-L6-v2, 384 dims, CPU-only), and upserts.

Re-running updates existing points (deterministic IDs based on content hash).
"""

import hashlib
import os
import re
from pathlib import Path

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


DOCS_DIR = "docs/vocabulary"
COLLECTION = "k3d_specifications"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_NAME = "fast-all-minilm-l6-v2"
VECTOR_SIZE = 384
TARGET_CHUNK_CHARS = 2000
OVERLAP_CHARS = 200


def split_by_headers(content: str, file_path: str):
    """Split markdown by ## and ### headers, preserving section hierarchy."""
    lines = content.split("\n")
    sections = []
    h2_title = ""
    current_title = ""
    current_level = 0
    current_lines = []
    current_line_start = 1

    def flush():
        if current_lines or current_title:
            text = "\n".join(current_lines).strip()
            if text:
                path = [h2_title] if h2_title and current_level == 3 else []
                if current_title:
                    path.append(current_title)
                sections.append({
                    "header": current_title or "(preamble)",
                    "content": text,
                    "section_path": path,
                    "level": current_level or 1,
                    "line_start": current_line_start,
                })

    for i, line in enumerate(lines):
        if line.startswith("## ") and not line.startswith("### "):
            flush()
            h2_title = line.lstrip("# ").strip()
            current_title = h2_title
            current_level = 2
            current_lines = []
            current_line_start = i + 1
        elif line.startswith("### "):
            flush()
            current_title = line.lstrip("# ").strip()
            current_level = 3
            current_lines = []
            current_line_start = i + 1
        else:
            current_lines.append(line)

    flush()
    return sections


def chunk_section(section: dict):
    """Split large sections into smaller chunks, preserving code blocks."""
    content = section["content"]
    if len(content) <= TARGET_CHUNK_CHARS:
        return [section]

    # Find code block boundaries to avoid splitting them
    code_blocks = [(m.start(), m.end()) for m in re.finditer(r"```.*?```", content, re.DOTALL)]

    def in_code_block(pos):
        for start, end in code_blocks:
            if start <= pos < end:
                return end
        return None

    chunks = []
    start = 0
    while start < len(content):
        end = start + TARGET_CHUNK_CHARS
        if end >= len(content):
            end = len(content)
        else:
            # Check if we're cutting inside a code block
            block_end = in_code_block(end)
            if block_end is not None:
                end = min(block_end, len(content))
            else:
                # Try to break at a paragraph boundary
                newline_pos = content.rfind("\n\n", start + TARGET_CHUNK_CHARS // 2, end + 200)
                if newline_pos > start:
                    end = newline_pos

        chunk_text = content[start:end].strip()
        if chunk_text:
            chunks.append({
                **section,
                "content": chunk_text,
                "chunk_index": len(chunks),
            })

        next_start = end - OVERLAP_CHARS
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def classify_content(title: str, content: str) -> str:
    t = title.lower()
    if any(kw in t for kw in ("requirement", "must", "shall", "constraint", "invariant")):
        return "requirement"
    if "example" in t:
        return "example"
    if "```" in content:
        return "code"
    return "definition"


def make_id(file_path: str, header: str, chunk_index: int = 0) -> str:
    """Deterministic UUID from file + header + chunk index."""
    import uuid
    raw = f"{file_path}::{header}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def main():
    api_key = os.getenv("QDRANT_API_KEY", "@20Cooool58")
    client = QdrantClient(url="http://localhost:6333", api_key=api_key)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"Deleting existing collection '{COLLECTION}' to repair named-vector schema...")
        client.delete_collection(collection_name=COLLECTION)

    print(f"Creating collection '{COLLECTION}' ({VECTOR_SIZE} dims, cosine, vector={VECTOR_NAME})...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        },
    )
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="spec_name",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="content_type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print("  Collection and indexes created.")

    # Load embedding model (downloads ~30MB on first run)
    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)

    # Process all spec files
    md_files = sorted(Path(DOCS_DIR).glob("*.md"))
    print(f"\nFound {len(md_files)} specification files in {DOCS_DIR}/\n")

    all_points = []
    for md_path in md_files:
        spec_name = md_path.stem.lower().replace("_specification", "").replace("_", " ")
        rel_path = str(md_path)
        content = md_path.read_text(encoding="utf-8")

        sections = split_by_headers(content, rel_path)
        chunks = []
        for sec in sections:
            chunks.extend(chunk_section(sec))

        # Embed all chunks for this file
        texts_to_embed = [
            f"{c['header']}\n\n{c['content']}" for c in chunks
        ]
        embeddings = list(embedder.embed(texts_to_embed))

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_idx = chunk.get("chunk_index", 0)
            point_id = make_id(rel_path, chunk["header"], chunk_idx)
            chunk_text = f"{chunk['header']}\n\n{chunk['content']}"

            all_points.append(models.PointStruct(
                id=point_id,
                vector={VECTOR_NAME: embedding.tolist()},
                payload={
                    "text": chunk_text,
                    "document": chunk_text,
                    "spec_name": spec_name,
                    "file_path": rel_path,
                    "section_path": chunk["section_path"],
                    "section_level": chunk["level"],
                    "content_type": classify_content(chunk["header"], chunk["content"]),
                    "line_start": chunk["line_start"],
                },
            ))

        print(f"  {md_path.name:55s} → {len(chunks):3d} chunks")

    # Batch upsert (Qdrant handles up to 100 points per call efficiently)
    print(f"\nUpserting {len(all_points)} points to Qdrant...")
    batch_size = 64
    for i in range(0, len(all_points), batch_size):
        batch = all_points[i : i + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)

    # Report
    info = client.get_collection(COLLECTION)
    print(f"\nDone. Collection '{COLLECTION}': {info.points_count} points indexed.")
    print(f"Vector name: {VECTOR_NAME}")
    print(f"Agents can now search via MCP: qdrant-find('What is Galaxy Universe?')")


if __name__ == "__main__":
    main()
