"""Proceduralize augmented content into symlink-like meaning references."""

from __future__ import annotations

from typing import Callable

from knowledge3d.tools.augmentation_providers import AugmentationResult


def tokenize_words(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in str(text or ""):
        if char.isalnum() or char in {"_", "-", "'"}:
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def proceduralize_text(text: str, galaxy_lookup: Callable[[str], str | None]) -> list[dict[str, str | bool]]:
    symlinks: list[dict[str, str | bool]] = []
    for token in tokenize_words(text):
        normalized = token.lower()
        ref = galaxy_lookup(normalized)
        symlinks.append(
            {
                "token": token,
                "normalized": normalized,
                "star_id": ref or f"candidate::{normalized}",
                "missing": ref is None,
            }
        )
    return symlinks


def proceduralize_content(
    augmented: AugmentationResult,
    *,
    galaxy_lookup: Callable[[str], str | None],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    records.append(
        {
            "source": "summary",
            "text": augmented.summary,
            "links": proceduralize_text(augmented.summary, galaxy_lookup),
        }
    )
    for index, entity in enumerate(augmented.entities):
        name = str(entity.get("name", "")).strip()
        content = str(entity.get("content", "")).strip()
        records.append(
            {
                "source": f"entity:{index}",
                "entity_type": str(entity.get("type", "")).strip(),
                "text": " ".join(part for part in (name, content) if part),
                "links": proceduralize_text(" ".join(part for part in (name, content) if part), galaxy_lookup),
            }
        )
    return records


__all__ = ["proceduralize_content", "proceduralize_text", "tokenize_words"]
