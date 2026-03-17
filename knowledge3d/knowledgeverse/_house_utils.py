"""Shared helpers for House template construction."""

from __future__ import annotations

from .meaning_star import SurfaceForm


def char_refs(text: str, language: str) -> list[str]:
    refs: list[str] = []
    for char in text:
        if char.isspace():
            continue
        if char.isascii() and char.isalnum():
            refs.append(f"char_{char.lower()}")
        else:
            refs.append(f"char_{language}_u{ord(char):04x}")
    return refs


def surface_forms(en: str, pt: str, ja: str) -> dict[str, SurfaceForm]:
    ja_word_ref = f"ja_{ord(ja[0]):04x}" if ja else "ja_unknown"
    return {
        "en": SurfaceForm(word_ref=en.lower().replace(" ", "_"), char_refs=char_refs(en, "en")),
        "pt": SurfaceForm(word_ref=pt.lower().replace(" ", "_"), char_refs=char_refs(pt, "pt")),
        "ja": SurfaceForm(word_ref=ja_word_ref, char_refs=char_refs(ja, "ja")),
    }


__all__ = ["char_refs", "surface_forms"]
