"""
Word Meaning Galaxy builder (sense-disambiguated, compositional, hierarchical).

Purpose:
- Build word-meaning stars from word_stars JSONL (UD/lexicons) with meaning_id/sense.
- Keep procedural programs primary (meaning_rpn, morph_rpn, phonetic_rpn, syntactic hints).
- Attach references hierarchically:
    * morpheme_refs (if available) OR syllable_refs (if available) as primary sublexical links
    * letter_refs only for leftover characters not covered by morphemes/syllables
  This reduces crossing edges: words → sublexical units; sublexical → letters.
- Output JSONL ready for ProceduralGalaxy/GLB upsert.

Inputs:
- word_stars_all.jsonl with fields: lang, lemma, sense/meaning_id, morph_rpn,
  meaning_program, phonetic (optional), dependencies (optional).

Outputs:
- JSONL where each line is one word-meaning star:
  {
    "meaning_id": "WORD_en_apple_default",
    "lemma": "apple",
    "lang": "en",
    "morpheme_refs": [...],  # optional
    "syllable_refs": [...],  # optional
    "letter_refs": [...],    # only for uncovered positions
    "procedural_programs": {...},
    "embeddings": {"matryoshka": null, "regenerable": true}
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .segmenter import syllabify, morph_segment


def _detect_script(ch: str) -> str:
    code = ord(ch)
    if 0x0041 <= code <= 0x024F:
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


def _letter_concept(ch: str) -> str:
    script = _detect_script(ch)
    base = ch.upper()
    return f"LETTER_{base}_{script.upper()}"


def _case_for_position(index: int, word: str, lang: str, is_proper: bool) -> str:
    if index == 0 and (is_proper or word[:1].isupper()):
        return "uppercase"
    return "lowercase"


def _iter_word_stars(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_word_stars(words_jsonl: Path) -> List[Dict]:
    stars: List[Dict] = []
    for star in _iter_word_stars(words_jsonl):
        lang = star.get("lang", "unknown")
        lemma = star.get("lemma", "")
        sense = star.get("sense", "default")
        meaning_id = star.get("meaning_id") or f"WORD_{lang}_{lemma}_{sense}"
        forms = star.get("forms", [])
        # Choose canonical word form for letter refs (prefer lowercase lemma)
        spelling = lemma or (forms[0] if forms else "")
        is_proper = spelling[:1].isupper()
        # Hierarchical linking: prefer morpheme_refs/syllable_refs; leftover letters as letter_refs.
        morpheme_refs = star.get("morpheme_refs") or morph_segment(spelling, lang)
        syllable_refs = star.get("syllable_refs") or syllabify(spelling, lang)
        covered = set()
        for ref in morpheme_refs + syllable_refs:
            for pos in range(ref.get("start", 0), ref.get("end", 0)):
                covered.add(pos)

        letter_refs = []
        for idx, ch in enumerate(spelling):
            if idx in covered:
                continue  # already covered by morpheme/syllable
            lc = _letter_concept(ch)
            letter_refs.append(
                {
                    "letter_concept": lc,
                    "case": _case_for_position(idx, spelling, lang, is_proper),
                    "position": idx,
                }
            )
        procedural_programs = {
            "meaning_rpn": star.get("meaning_program"),
            "morphological_rpn": star.get("morph_rpn"),
            "phonetic_rpn": star.get("phonetic") or star.get("phonetic_rpn"),
            "syntactic_hints": star.get("dependencies") or star.get("dep_roles"),
        }
        stars.append(
            {
                "meaning_id": meaning_id,
                "lemma": lemma,
                "lang": lang,
                "sense": sense,
                "morpheme_refs": morpheme_refs,
                "syllable_refs": syllable_refs,
                "letter_refs": letter_refs,
                "procedural_programs": procedural_programs,
                "embeddings": {"matryoshka": None, "regenerable": True},
                "forms": forms,
                "sources": star.get("sources", []),
            }
        )
    return stars


def write_jsonl(stars: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build word-meaning stars (sense-disambiguated, compositional).")
    ap.add_argument("--word-stars", type=Path, required=True, help="Input word_stars JSONL (with sense/meaning_id).")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL of word-meaning stars.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_word_stars(args.word_stars)
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} word-meaning stars -> {args.output}")


if __name__ == "__main__":
    main()
