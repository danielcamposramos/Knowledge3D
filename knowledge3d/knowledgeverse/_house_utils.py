"""Shared helpers for House template construction."""

from __future__ import annotations

from knowledge3d.ingestion.canonical_lookup import canonical_char_star_id, canonical_word_star_id

from .meaning_star import SurfaceForm


def char_refs(text: str, language: str) -> list[str]:
    del language
    refs: list[str] = []
    for char in text:
        if char.isspace():
            continue
        refs.append(canonical_char_star_id(char))
    return refs


def surface_forms(en: str, pt: str, ja: str) -> dict[str, SurfaceForm]:
    return {
        "en": SurfaceForm(word_ref=canonical_word_star_id("en", en), char_refs=char_refs(en, "en")),
        "pt": SurfaceForm(word_ref=canonical_word_star_id("pt", pt), char_refs=char_refs(pt, "pt")),
        "ja": SurfaceForm(word_ref=canonical_word_star_id("ja", ja), char_refs=char_refs(ja, "ja")),
    }


__all__ = ["char_refs", "surface_forms"]
