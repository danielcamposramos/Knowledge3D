#!/usr/bin/env python3
"""Seed the canonical K3D registry into Qdrant.

This is ingestion-path infrastructure only. It defines canonical IDs for
characters, drawing primitives, grammar templates, symlink kinds, and meaning
classes so proceduralizers stop inventing duplicate identifiers.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.canonical_lookup import (
    CANONICAL_COLLECTION,
    CANONICAL_VECTOR_NAME,
    canonical_char_star_id,
    canonical_drawing_primitive_id,
    canonical_entry_id,
    canonical_grammar_template_id,
)
from knowledge3d.ingestion.qdrant_credentials import resolve_qdrant_api_key


VECTOR_SIZE = 384
DOCS_DIR = Path("docs/vocabulary")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _iter_entries() -> Iterable[dict[str, object]]:
    for char in "abcdefghijklmnopqrstuvwxyz0123456789":
        yield {
            "kind": "star_id",
            "key": f"char::{char}",
            "star_id": canonical_char_star_id(char),
            "document": f"canonical character id for {char}",
        }
    for primitive in ("line", "arc", "quad", "cubic", "circle", "rect", "tri"):
        yield {
            "kind": "drawing_primitive",
            "key": primitive,
            "star_id": canonical_drawing_primitive_id(primitive),
            "document": f"canonical drawing primitive {primitive}",
        }
    for language, template in (
        *[(lang, "copula") for lang in ("en", "pt", "es", "fr", "de", "it", "ja", "zh", "ru")],
        *[(lang, "periphrastic_explanation") for lang in ("en", "pt", "es", "fr", "de", "it", "ja", "zh", "ru")],
    ):
        yield {
            "kind": "grammar_template",
            "key": f"{language}:{template}",
            "star_id": canonical_grammar_template_id(language, template),
            "document": f"canonical grammar template {language} {template}",
        }
    for meaning_class in (
        # Phase 7.0 ontological classes
        "concept", "relation", "action", "property", "meta", "form",
        # Phase 7.A.1 galaxy meaning classes
        "drawing", "glyph", "word", "number", "grammar",
        "reality", "math", "audio", "object_3d", "tool", "game_2d",
    ):
        yield {
            "kind": "meaning_class",
            "key": meaning_class,
            "star_id": meaning_class,
            "document": f"canonical meaning class {meaning_class}",
        }
    for symlink_kind in (
        # Phase 7.0 symlink kinds
        "taxonomy_refs", "meta_refs", "grammar_refs", "component_refs",
        "composite_of", "mathematical_role",
        # Phase 7.A.1 additional field paths from CANONICAL_REGISTRY_SPECIFICATION.md §7
        "reality_refs", "visual_refs", "audio_refs", "char_refs",
        "surface_forms_word_ref", "glyph_refs", "rpn_refs",
    ):
        yield {
            "kind": "symlink_kind",
            "key": symlink_kind,
            "star_id": symlink_kind,
            "document": f"canonical symlink kind {symlink_kind}",
        }


def main() -> None:
    client = QdrantClient(url="http://localhost:6333", api_key=resolve_qdrant_api_key(), timeout=60.0)

    existing = {collection.name for collection in client.get_collections().collections}
    if CANONICAL_COLLECTION in existing:
        client.delete_collection(CANONICAL_COLLECTION)

    client.create_collection(
        collection_name=CANONICAL_COLLECTION,
        vectors_config={
            CANONICAL_VECTOR_NAME: models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        },
    )
    client.create_payload_index(CANONICAL_COLLECTION, "kind", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(CANONICAL_COLLECTION, "key", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(CANONICAL_COLLECTION, "context_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(CANONICAL_COLLECTION, "ethical_trit", models.PayloadSchemaType.KEYWORD)

    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    points: list[models.PointStruct] = []
    for entry in _iter_entries():
        document = str(entry["document"])
        vector = next(embedder.embed([document]))
        points.append(
            models.PointStruct(
                id=canonical_entry_id(str(entry["kind"]), str(entry["key"])),
                vector={CANONICAL_VECTOR_NAME: vector.tolist()},
                payload={
                    "kind": str(entry["kind"]),
                    "key": str(entry["key"]),
                    "star_id": str(entry["star_id"]),
                    "text": document,
                    "document": document,
                    "context_id": int(entry.get("context_id", 0) or 0),
                    "ethical_trit": max(-1, min(1, int(entry.get("ethical_trit", 0) or 0))),
                },
            )
        )

    for start in range(0, len(points), 64):
        client.upsert(CANONICAL_COLLECTION, points[start : start + 64])

    info = client.get_collection(CANONICAL_COLLECTION)
    print(f"{CANONICAL_COLLECTION}: {info.points_count} canonical entries")


if __name__ == "__main__":
    main()
