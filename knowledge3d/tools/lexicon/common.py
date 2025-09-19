from __future__ import annotations

"""Shared helpers for lexicon star builders."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

DEFAULT_DIM = 512


def slugify(text: str) -> str:
    """Convert text to a filesystem-friendly slug."""
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    cleaned = []
    for ch in text.lower():
        if ch in allowed:
            cleaned.append(ch)
        elif ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("-")
    slug = "".join(cleaned)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "item"


def hashed_embedding(*parts: str, dim: int = DEFAULT_DIM) -> List[float]:
    """Create a deterministic pseudo-embedding from textual parts."""
    text = "\u241f".join(part.strip() for part in parts if part)
    if not text:
        text = "lexicon"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: List[float] = []
    seed = digest
    while len(values) < dim:
        for byte in seed:
            values.append(((byte / 255.0) * 2.0) - 1.0)
            if len(values) >= dim:
                break
        seed = hashlib.sha256(seed).digest()
    return values


def make_star_id(language: str, source: str, lemma: str, sense_ref: str) -> str:
    lemma_slug = slugify(lemma)
    sense_slug = slugify(sense_ref.replace(":", "-"))
    return f"star_lex_{language}_{source}_{lemma_slug}_{sense_slug}"


def build_star(
    *,
    language: str,
    source: str,
    lemma: str,
    pos: Optional[str],
    sense_ref: str,
    definition: Optional[str],
    embedding_parts: Sequence[str],
    relations: Mapping[str, Sequence[str]],
    extra: MutableMapping[str, object],
    zone: str = "Zone 2 (Study)",
    tags: Optional[Iterable[str]] = None,
    modalities: Optional[Sequence[str]] = None,
) -> MutableMapping[str, object]:
    """Assemble a Galaxy star dictionary for lexicon content."""
    star_id = make_star_id(language, source, lemma, sense_ref)
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    tag_set = ["lexicon", f"lang:{language}", f"source:{source}"]
    if pos:
        tag_set.append(f"pos:{pos}")
    if tags:
        for tag in tags:
            if tag not in tag_set:
                tag_set.append(tag)

    definition_text = definition or ""
    embedding = hashed_embedding(*embedding_parts, definition_text)

    star: MutableMapping[str, object] = {
        "type": "star",
        "id": star_id,
        "name": f"{lemma} — {language.upper()} lexicon",
        "created_at": created_at,
        "honesty_score": 1.0,
        "embedding": embedding,
        "modality_fusion": list(modalities) if modalities else ["text"],
        "zone_placement": zone,
        "tags": tag_set,
        "relations": {
            key: list(sorted(set(value)))
            for key, value in relations.items()
            if value
        },
    }
    star.update(extra)
    return star


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
