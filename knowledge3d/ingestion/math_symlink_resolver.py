"""Batch 8 HS-math symlink resolver."""

from __future__ import annotations

from pathlib import Path

from knowledge3d.ingestion.canonical_lookup import CanonicalLookup, canonical_char_star_id
from knowledge3d.ingestion.math_semantic_aliases import CONCEPT_MEANING_STARS, CONSTANT_ALIASES, LETTER_ALIASES, SYMBOL_ALIASES


class MathSymlinkResolveError(LookupError):
    """Raised when a normalised HS-math symlink ref cannot be resolved."""


def _load_allowlist(path: Path | None) -> frozenset[str]:
    if path is None or not Path(path).exists():
        return frozenset()
    rows: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        cleaned = line.split("#", 1)[0].strip()
        if cleaned:
            rows.append(cleaned)
    return frozenset(rows)


class MathSymlinkResolver:
    def __init__(self, canonical_lookup: CanonicalLookup, allowlist_path: Path | None = None):
        self.canonical_lookup = canonical_lookup
        self.allowlist = _load_allowlist(allowlist_path)
        self._cache: dict[str, str | None] = {}

    def resolve(self, ref: str) -> str | None:
        token = str(ref or "").strip()
        if token in self._cache:
            return self._cache[token]
        if token in self.allowlist:
            self._cache[token] = None
            return None
        resolved = self._resolve_uncached(token)
        self._cache[token] = resolved
        return resolved

    def _resolve_uncached(self, ref: str) -> str:
        if ref.startswith("letter::"):
            tail = ref.split("::", 1)[1]
            if len(tail) == 1 and tail.isascii() and tail.isalpha():
                return canonical_char_star_id(tail)
            try:
                return LETTER_ALIASES[tail]
            except KeyError as exc:
                raise MathSymlinkResolveError(f"invalid_letter_ref:{ref}") from exc
        if ref.startswith("constant::"):
            tail = ref.split("::", 1)[1]
            try:
                return CONSTANT_ALIASES[tail]
            except KeyError as exc:
                raise MathSymlinkResolveError(f"unknown_constant_alias:{ref}") from exc
        if ref.startswith("symbol::"):
            tail = ref.split("::", 1)[1]
            try:
                return SYMBOL_ALIASES[tail]
            except KeyError as exc:
                raise MathSymlinkResolveError(f"unknown_symbol_alias:{ref}") from exc
        if ref.startswith("concept::"):
            tail = ref.split("::", 1)[1]
            key = f"concept_{tail}"
            if key in CONCEPT_MEANING_STARS.values():
                return key
            if not self.canonical_lookup.exists(kind="meaning_star", key=key):
                raise MathSymlinkResolveError(f"missing_concept_ref:{ref}")
            return self.canonical_lookup.find_star_id(kind="meaning_star", key=key)
        raise MathSymlinkResolveError(f"unsupported_namespace:{ref}")


__all__ = ["MathSymlinkResolveError", "MathSymlinkResolver"]
