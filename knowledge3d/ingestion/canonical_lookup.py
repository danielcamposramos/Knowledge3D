"""Strict canonical ID and Qdrant lookup helpers for ingestion.

Pure ID helpers in this module are safe for non-hot-path callers. The Qdrant
lookup surface remains ingestion-path only and must never enter PTX or the
sovereign runtime tick.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from knowledge3d.ingestion.qdrant_credentials import resolve_qdrant_api_key


CANONICAL_COLLECTION = "k3d_canonical"
CANONICAL_VECTOR_NAME = "fast-all-minilm-l6-v2"
def canonical_slug(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return "unknown"
    parts: list[str] = []
    pending_sep = False
    for char in raw:
        if char.isascii() and char.isalnum():
            if pending_sep and parts and parts[-1] != "_":
                parts.append("_")
            parts.append(char)
            pending_sep = False
            continue
        if char.isspace() or char in {"-", "_", "/", ".", ":"}:
            pending_sep = True
            continue
        if char.isalnum():
            if pending_sep and parts and parts[-1] != "_":
                parts.append("_")
            parts.append(f"u{ord(char):04x}")
            pending_sep = False
            continue
        pending_sep = True
    slug = "".join(parts).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def canonical_word_star_id(language: str, lemma: str) -> str:
    return f"word_{str(language or '').strip().lower()}_{canonical_slug(lemma)}"


def canonical_char_star_id(char: str) -> str:
    glyph = str(char or "")
    if len(glyph) != 1:
        raise ValueError(f"canonical_char_requires_single_codepoint:{glyph!r}")
    if glyph.isascii() and glyph.isalnum():
        return f"char_{glyph.lower()}"
    return f"char_u{ord(glyph):04x}"


def canonical_grammar_template_id(language: str, template_name: str) -> str:
    return f"grammar_template_{str(language or '').strip().lower()}_{canonical_slug(template_name)}"


def canonical_drawing_primitive_id(primitive_name: str) -> str:
    return f"drawing_primitive_{canonical_slug(primitive_name)}"


def canonical_entry_id(kind: str, key: str) -> str:
    payload = f"{str(kind).strip().lower()}::{str(key).strip().lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def canonical_context_id(region: str = "", era: str = "") -> int:
    """Deterministic u32 context id for era/region-scoped microtheories.

    `0` is reserved for universal context, so empty region+era maps to 0.
    Non-empty inputs use the same uuid5 namespace pattern as canonical entries
    and truncate the UUID to a stable non-zero u32.
    """

    region_key = canonical_slug(region)
    era_key = canonical_slug(era)
    if region_key == "unknown" and era_key == "unknown":
        return 0
    payload = f"context::{region_key}::{era_key}"
    value = uuid.uuid5(uuid.NAMESPACE_URL, payload).int & 0xFFFFFFFF
    return int(value or 1)


class CanonicalLookup:
    """Authoritative canonical registry over Qdrant.

    No deterministic fallback is provided here. If Qdrant or the collection is
    unavailable, callers fail and fix.
    """

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = CANONICAL_COLLECTION,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from qdrant_client import QdrantClient  # pylint: disable=import-outside-toplevel

            resolved_api_key = api_key or resolve_qdrant_api_key()
            client = QdrantClient(url=url, api_key=resolved_api_key, timeout=60.0)
        self.client = client
        self.collection_name = str(collection_name)

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        names = {str(collection.name) for collection in collections}
        if self.collection_name not in names:
            raise RuntimeError(f"canonical_collection_missing:{self.collection_name}")

    def _scroll_exact(self, *, kind: str, key: str) -> dict[str, Any] | None:
        from qdrant_client import models  # pylint: disable=import-outside-toplevel

        self.ensure_collection()
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="kind", match=models.MatchValue(value=str(kind))),
                    models.FieldCondition(key="key", match=models.MatchValue(value=str(key))),
                ]
            ),
        )
        if not points:
            return None
        return dict(points[0].payload or {})

    def find_star_id(self, *, kind: str, key: str) -> str:
        payload = self._scroll_exact(kind=kind, key=key)
        if payload is None:
            raise KeyError(f"canonical_lookup_miss:{kind}:{key}")
        star_id = str(payload.get("star_id", "")).strip()
        if not star_id:
            raise RuntimeError(f"canonical_lookup_missing_star_id:{kind}:{key}")
        return star_id

    def exists(self, *, kind: str, key: str) -> bool:
        return self._scroll_exact(kind=kind, key=key) is not None

    def star_id_exists(self, star_id: str) -> bool:
        from qdrant_client import models  # pylint: disable=import-outside-toplevel

        target = str(star_id or "").strip()
        if not target:
            return False
        self.ensure_collection()
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="star_id", match=models.MatchValue(value=target))]
            ),
        )
        return bool(points)

    def register(
        self,
        *,
        kind: str,
        key: str,
        star_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from fastembed import TextEmbedding  # pylint: disable=import-outside-toplevel
        from qdrant_client import models  # pylint: disable=import-outside-toplevel

        self.ensure_collection()
        text = f"{kind}\n{key}\n{star_id}"
        embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector = next(embedder.embed([text]))
        point = models.PointStruct(
            id=canonical_entry_id(kind, key),
            vector={CANONICAL_VECTOR_NAME: vector.tolist()},
            payload={
                "kind": str(kind),
                "key": str(key),
                "star_id": str(star_id),
                "text": text,
                "document": text,
                "context_id": int((metadata or {}).get("context_id", 0) or 0),
                "ethical_trit": max(-1, min(1, int((metadata or {}).get("ethical_trit", 0) or 0))),
                "metadata": dict(metadata or {}),
            },
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return str(star_id)


__all__ = [
    "CANONICAL_COLLECTION",
    "CANONICAL_VECTOR_NAME",
    "CanonicalLookup",
    "canonical_context_id",
    "canonical_char_star_id",
    "canonical_drawing_primitive_id",
    "canonical_entry_id",
    "canonical_grammar_template_id",
    "canonical_slug",
    "canonical_word_star_id",
]
