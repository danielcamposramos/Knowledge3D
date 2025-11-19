#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Cyrillic glyph harvester.

Scans common system font directories, checks which fonts contain Cyrillic
characters, and emits a lightweight dataset compatible with the procedural
drawing specialist training scripts. This is intentionally minimal: it records
metadata (character, font, languages) and generates a deterministic placeholder
RPN program so the downstream pipeline can ingest/verify Cyrillic coverage
before the full font-outline extractor is wired up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from PIL import ImageFont

from knowledge3d.cranium.specialists.character_languages import (
    get_character_languages,
)


FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    str(Path.home() / ".fonts"),
    str(Path.home() / ".local/share/fonts"),
]

OUTPUT_PATH = Path("/K3D/Knowledge3D.local/datasets/atomic/fonts_cyrillic_simple.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CYRILLIC_CHARS = [
    # Uppercase А-Я
    *(chr(code) for code in range(0x0410, 0x042F + 1)),
    # Lowercase а-я
    *(chr(code) for code in range(0x0430, 0x044F + 1)),
    # Extended samples
    'Ё', 'ё', 'Є', 'є', 'Ї', 'ї', 'Ґ', 'ґ',
]


def list_fonts(limit: int = 200) -> List[Path]:
    """Return a list of TTF/OTF fonts from standard directories."""
    fonts: List[Path] = []
    for font_dir in FONT_DIRS:
        path = Path(font_dir)
        if not path.exists():
            continue
        fonts.extend(path.rglob("*.ttf"))
        fonts.extend(path.rglob("*.otf"))
        if len(fonts) >= limit:
            break
    return fonts[:limit]


def font_supports_char(font_path: Path, char: str) -> bool:
    """Return True if the font supports the given character."""
    try:
        font = ImageFont.truetype(str(font_path), size=48, encoding="unic")
        bbox = font.getbbox(char)
        return bbox is not None
    except Exception:
        return False


def generate_placeholder_rpn(char: str, font_index: int, glyph_index: int) -> str:
    """Generate a deterministic placeholder RPN program for the character."""
    seed = (ord(char) + font_index + glyph_index) % 1000 / 1000.0
    move_x = round(0.2 + 0.6 * seed, 3)
    move_y = round(0.1 + 0.7 * ((seed * 37) % 1.0), 3)
    line_x = round(0.8 - move_x, 3)
    line_y = round(0.9 - move_y, 3)
    return f"{move_x} {move_y} MOVE {line_x} {line_y} LINE STROKE"


def harvest() -> List[Dict]:
    fonts = list_fonts(limit=150)
    if not fonts:
        raise RuntimeError("No fonts found; please install Cyrillic-capable fonts.")

    entries: List[Dict] = []
    for font_idx, font_path in enumerate(fonts):
        supported_chars = []
        for char in CYRILLIC_CHARS:
            if font_supports_char(font_path, char):
                supported_chars.append(char)
        if not supported_chars:
            continue

        for glyph_idx, char in enumerate(supported_chars):
            rpn = generate_placeholder_rpn(char, font_idx, glyph_idx)
            entry = {
                'char': char,
                'rpn': rpn,
                'font': font_path.stem,
                'font_path': str(font_path),
                'type': 'cyrillic_glyph',
                'category': 'cyrillic',
                'languages': get_character_languages(char),
            }
            entries.append(entry)

    if not entries:
        raise RuntimeError("No Cyrillic glyphs detected in scanned fonts.")

    with OUTPUT_PATH.open('w', encoding='utf-8') as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"[Cyrillic Harvester] Fonts scanned: {len(fonts)}")
    print(f"[Cyrillic Harvester] Glyph entries written: {len(entries)}")
    print(f"[Cyrillic Harvester] Output: {OUTPUT_PATH}")
    return entries


if __name__ == "__main__":
    harvest()
