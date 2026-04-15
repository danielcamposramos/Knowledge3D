"""Batch 8 canonical-id normalisation for HS math catalogues."""

from __future__ import annotations

from knowledge3d.ingestion.canonical_lookup import canonical_slug


CATEGORIES: frozenset[str] = frozenset({"formula", "identity", "theorem", "rule", "concept", "method"})


class MathCanonicalIdError(ValueError):
    """Raised when a HS-math canonical id cannot be normalised safely."""


def normalise_canonical_id(raw: str) -> tuple[str, str]:
    cleaned = str(raw or "").strip().strip("`")
    if not cleaned:
        raise MathCanonicalIdError("empty_canonical_id")
    token = cleaned.replace("::", "_")
    token = "_".join(part for part in token.split("_") if part)
    if "_" not in token:
        raise MathCanonicalIdError(f"missing_category_prefix:{cleaned}")
    category, leaf = token.split("_", 1)
    category = canonical_slug(category)
    if category not in CATEGORIES:
        raise MathCanonicalIdError(f"unknown_category:{cleaned}")
    slug = canonical_slug(leaf)
    if not slug:
        raise MathCanonicalIdError(f"empty_slug:{cleaned}")
    if any(ord(char) > 127 for char in slug):
        raise MathCanonicalIdError(f"non_ascii_slug:{cleaned}")
    return category, f"{category}_{slug}"


__all__ = ["CATEGORIES", "MathCanonicalIdError", "normalise_canonical_id"]
