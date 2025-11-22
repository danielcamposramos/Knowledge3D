"""
Morpheme builder (sublexical, meaning-first).

Purpose:
- Build morpheme stars (prefix/suffix/root/inflection) with letter_refs composition,
  morph_rpn (role), meaning_rpn (brief semantics), and optional phonetic_rpn.
- Identity by meaning/function: MORFEMA_prefix_re_pt, MORFEMA_sufixo_mente_pt, etc.

Inputs:
- morpheme JSONL lines like:
  {"lang": "pt", "form": "re", "role": "prefix", "meaning": "again", "morph_rpn": "..."}

Outputs:
- JSONL stars:
  {
    "morpheme_id": "MORFEMA_prefix_re_pt",
    "lang": "pt",
    "form": "re",
    "role": "prefix",
    "letter_refs": [...],
    "procedural_programs": {"morph_rpn": "...", "meaning_rpn": "...", "phonetic_rpn": null},
    "embeddings": {"matryoshka": null, "regenerable": True}
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_morphemes(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


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
    return f"LETTER_{ch.upper()}_{script.upper()}"


def build_morpheme_stars(jsonl_path: Path) -> List[Dict]:
    stars: List[Dict] = []
    for item in _iter_morphemes(jsonl_path):
        lang = item.get("lang", "unknown")
        form = item.get("form", "")
        role = item.get("role", "unk")
        if not form:
            continue
        letter_refs = []
        for idx, ch in enumerate(form):
            letter_refs.append(
                {"letter_concept": _letter_concept(ch), "case": "lowercase", "position": idx}
            )
        morph_rpn = item.get("morph_rpn") or f"MORPHEME {role}"
        meaning_rpn = item.get("meaning_rpn") or item.get("meaning") or f"{role.upper()}_{form}"
        mid = f"MORPHEME_{role}_{form}_{lang}"
        stars.append(
            {
                "morpheme_id": mid,
                "lang": lang,
                "form": form,
                "role": role,
                "letter_refs": letter_refs,
                "procedural_programs": {
                    "morph_rpn": morph_rpn,
                    "meaning_rpn": meaning_rpn,
                    "phonetic_rpn": item.get("phonetic_rpn"),
                },
                "embeddings": {"matryoshka": None, "regenerable": True},
            }
        )
    return stars


def write_jsonl(stars: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build morpheme stars (sublexical, meaning-first).")
    ap.add_argument("--morphemes-jsonl", type=Path, required=True, help="Input morpheme JSONL.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL for morpheme stars.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_morpheme_stars(args.morphemes_jsonl)
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} morpheme stars -> {args.output}")


if __name__ == "__main__":
    main()
