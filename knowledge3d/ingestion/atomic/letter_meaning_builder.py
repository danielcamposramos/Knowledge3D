"""
Meaning-first builder for letter galaxies (per script).

Purpose:
- Group all glyph variants (uppercase/lower/italic/bold/etc.) of the same letter meaning
  into a single star per script.
- Preserve procedural programs as primary data (visual_rpn variants, optional audio,
  meaning_rpn). Embeddings remain secondary/regenerable (not computed here).
- Output a JSONL of letter-meaning stars ready for GLB/ProceduralGalaxy upsert.

Inputs:
- fonts_*_procedural.jsonl (per script) produced by font harvesters. Each line should
  include: character, visual_rpn, font_metadata, optional languages.

Outputs:
- JSONL where each line is one letter-meaning star:
  {
    "letter_concept": "LETTER_A_LATIN",
    "script": "Latin",
    "languages": [...],
    "glyph_variants": [
        {"visual_rpn": "...", "codepoint": "U+0041", "case": "uppercase", "font_metadata": {...}},
        ...
    ],
    "procedural_programs": {
        "visual_rpn": "...canonical...",
        "math_rpn": "ALPHABET_LATIN POSITION_1 LETTER_NAME A",
        "audio_rpn": null
    },
    "usage_rules": {...}  # compositional defaults stubbed; refine downstream
  }

Notes:
- Math symbols/operators are NOT handled here; they belong to the math symbol galaxy.
- Case variants stay inside the same letter star; do not split by Unicode codepoint.
- Diacritic forms are grouped by base letter per script (A/À/Á → LETTER_A_LATIN).
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _detect_script(ch: str) -> str:
    """Heuristic script detection by Unicode block."""
    code = ord(ch)
    if 0x0041 <= code <= 0x024F:  # Latin ranges (basic + extended)
        return "Latin"
    if 0x0400 <= code <= 0x04FF:
        return "Cyrillic"
    if 0x0600 <= code <= 0x06FF:
        return "Arabic"
    if 0x4E00 <= code <= 0x9FFF:
        return "CJK"
    if 0x2800 <= code <= 0x28FF:
        return "Braille"
    return "Unknown"


def _strip_diacritics(ch: str) -> str:
    """Normalize to remove diacritics for base-letter grouping."""
    decomposed = unicodedata.normalize("NFD", ch)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped or ch


def _case_from_char(ch: str) -> str:
    if ch.isupper():
        return "uppercase"
    if ch.islower():
        return "lowercase"
    return "neutral"


def _alphabet_position(base_letter: str, script: str) -> int:
    """Best-effort position; returns -1 if unknown."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if script != "Latin":
        return -1
    try:
        return alphabet.index(base_letter.upper()) + 1
    except ValueError:
        return -1


def _iter_fonts(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_letter_stars(font_jsonl: Path) -> List[Dict]:
    stars: Dict[str, Dict] = {}

    for item in _iter_fonts(font_jsonl):
        ch = item.get("character", "")
        if not ch:
            continue
        script = _detect_script(ch)
        base_letter = _strip_diacritics(ch.upper())
        letter_concept = f"LETTER_{base_letter}_{script.upper()}"
        case = _case_from_char(ch)
        codepoint = f"U+{ord(ch):04X}"
        glyph_entry = {
            "visual_rpn": item.get("visual_rpn", ""),
            "codepoint": codepoint,
            "case": case,
            "font_metadata": item.get("font_metadata", {}),
        }

        if letter_concept not in stars:
            pos = _alphabet_position(base_letter, script)
            stars[letter_concept] = {
                "letter_concept": letter_concept,
                "script": script,
                "alphabet_position": pos,
                "languages": set(item.get("languages", [])),
                "glyph_variants": [],
            }

        star = stars[letter_concept]
        star["glyph_variants"].append(glyph_entry)
        star["languages"].update(item.get("languages", []))

    results = []
    for letter_concept, data in stars.items():
        base_letter = letter_concept.replace("LETTER_", "").split("_", 1)[0]
        script = data.get("script", "Unknown")
        pos = data.get("alphabet_position", -1)
        # canonical = first glyph
        canonical_vrpn = data["glyph_variants"][0]["visual_rpn"] if data["glyph_variants"] else ""
        usage_rules = {
            "word_construction": {
                "sentence_start": "uppercase",
                "proper_noun": "uppercase",
                "default": "lowercase",
            },
            "case_transitions": {
                "upper_to_lower": {"kerning_adjust": -0.05},
                "lower_to_upper": {"kerning_adjust": 0.0},
            },
        }
        results.append(
            {
                "letter_concept": letter_concept,
                "script": script,
                "alphabet_position": pos,
                "languages": sorted(data["languages"]),
                "glyph_variants": data["glyph_variants"],
                "procedural_programs": {
                    "visual_rpn": canonical_vrpn,
                    "math_rpn": f"ALPHABET_{script.upper()} POSITION_{pos} LETTER_NAME {base_letter}",
                    "audio_rpn": None,  # attach later from audio stars
                    "meaning_rpn": f"LETTER {base_letter} {script}",
                },
                "usage_rules": usage_rules,
            }
        )
    return results


def write_jsonl(stars: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build letter-meaning stars (per script) from procedural font JSONL.")
    ap.add_argument("--fonts-jsonl", type=Path, required=True, help="Input fonts_*_procedural.jsonl (one script).")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL of letter-meaning stars.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_letter_stars(args.fonts_jsonl)
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} letter-meaning stars -> {args.output}")


if __name__ == "__main__":
    main()
