#!/usr/bin/env python3
"""
Generate syllable and morpheme JSONL from a word list using heuristic segmenters.

Inputs:
  --words <word_stars_all.jsonl>  (expects fields: lang, lemma)

Outputs:
  --syllables-out <path>          (default: /K3D/Knowledge3D.local/datasets/syllables_auto.jsonl)
  --morphemes-out <path>          (default: /K3D/Knowledge3D.local/datasets/morphemes_auto.jsonl)

Notes:
- Uses heuristic segmenters (pt/es/en) from knowledge3d.ingestion.atomic.segmenter.
- Deduplicates by (lang, syllable) and (lang, morpheme_id).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set, Tuple

from knowledge3d.ingestion.atomic.segmenter import syllabify, morph_segment


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Auto-segment words into syllables and morphemes (heuristic).")
    ap.add_argument("--words", type=Path, required=True, help="word_stars_all.jsonl (fields: lang, lemma).")
    ap.add_argument("--syllables-out", type=Path, default=Path("/K3D/Knowledge3D.local/datasets/syllables_auto.jsonl"))
    ap.add_argument("--morphemes-out", type=Path, default=Path("/K3D/Knowledge3D.local/datasets/morphemes_auto.jsonl"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    syl_seen: Set[Tuple[str, str]] = set()
    mor_seen: Set[str] = set()
    syllables_out = []
    morphemes_out = []

    with args.words.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            star = json.loads(line)
            lang = star.get("lang", "unknown")
            lemma = star.get("lemma", "")
            if not lemma:
                continue
            # syllables
            for syl in syllabify(lemma, lang):
                key = (lang, syl["syllable"])
                if key in syl_seen:
                    continue
                syl_seen.add(key)
                syllables_out.append({"lang": lang, "syllable": syl["syllable"], "pattern": syl["pattern"]})
            # morphemes
            for mor in morph_segment(lemma, lang):
                mid = mor["morpheme_id"]
                if mid in mor_seen:
                    continue
                mor_seen.add(mid)
                morphemes_out.append({"lang": lang, "form": lemma[mor["start"]:mor["end"]], "role": mid.split("_")[1]})

    args.syllables_out.parent.mkdir(parents=True, exist_ok=True)
    with args.syllables_out.open("w", encoding="utf-8") as f:
        for syl in syllables_out:
            f.write(json.dumps(syl, ensure_ascii=False) + "\n")

    args.morphemes_out.parent.mkdir(parents=True, exist_ok=True)
    with args.morphemes_out.open("w", encoding="utf-8") as f:
        for mor in morphemes_out:
            f.write(json.dumps(mor, ensure_ascii=False) + "\n")

    print(f"Syllables written: {len(syllables_out)} -> {args.syllables_out}")
    print(f"Morphemes written: {len(morphemes_out)} -> {args.morphemes_out}")


if __name__ == "__main__":
    main()
