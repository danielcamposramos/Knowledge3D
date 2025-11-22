"""
Syllable builder (sublexical, procedural-first, meaning-first per pattern/language).

Purpose:
- Build syllable stars with letter_refs (composition), optional phonetic_rpn,
  and a simple meaning_rpn describing syllable structure (CV, CVC, etc.).
- Identity by (language, syllable string, pattern). Case handled via letters.

Inputs:
- syllable JSONL (one per line): {"lang": "pt", "syllable": "ca", "pattern": "CV"}
- or a wordlist-derived generator can emit these; this builder just shapes stars.

Outputs:
- JSONL stars:
  {
    "syllable_id": "SYL_pt_ca_CV",
    "lang": "pt",
    "syllable": "ca",
    "pattern": "CV",
    "letter_refs": [...],
    "procedural_programs": {"meaning_rpn": "SYLLABLE CV", "phonetic_rpn": null},
    "embeddings": {"matryoshka": null, "regenerable": True}
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_syllables(path: Path) -> Iterable[Dict]:
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


def build_syllable_stars(jsonl_path: Path) -> List[Dict]:
    stars: List[Dict] = []
    for item in _iter_syllables(jsonl_path):
        lang = item.get("lang", "unknown")
        syl = item.get("syllable", "")
        pattern = item.get("pattern", "UNK")
        if not syl:
            continue
        letter_refs = []
        for idx, ch in enumerate(syl):
            letter_refs.append(
                {"letter_concept": _letter_concept(ch), "case": "lowercase", "position": idx}
            )
        sid = f"SYL_{lang}_{syl}_{pattern}"
        stars.append(
            {
                "syllable_id": sid,
                "lang": lang,
                "syllable": syl,
                "pattern": pattern,
                "letter_refs": letter_refs,
                "procedural_programs": {
                    "meaning_rpn": f"SYLLABLE {pattern}",
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
    ap = argparse.ArgumentParser(description="Build syllable stars (sublexical).")
    ap.add_argument("--syllables-jsonl", type=Path, required=True, help="Input syllable JSONL.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL for syllable stars.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_syllable_stars(args.syllables_jsonl)
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} syllable stars -> {args.output}")


if __name__ == "__main__":
    main()
