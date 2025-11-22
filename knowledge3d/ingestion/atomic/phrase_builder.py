"""
Phrase Meaning builder (idioms / multiword expressions).

Purpose:
- Build phrase-meaning stars (idioms, sayings, multiword expressions) with word_refs,
  meaning_rpn (idiomatic meaning), usage metadata. Identity by meaning_id/sense.
- Intended for both curated idioms and user-defined phrases (user galaxy).

Inputs (JSONL):
  {
    "lang": "pt",
    "phrase": "cair a ficha",
    "meaning": "comprehend/realize",
    "usage": "idiom",
    "register": "informal",
    "sense": "default"
  }

Outputs (JSONL):
  {
    "phrase_id": "PHRASE_pt_cair_a_ficha_default",
    "lang": "pt",
    "phrase": "cair a ficha",
    "word_refs": [{"word": "cair"}, {"word": "a"}, {"word": "ficha"}],
    "procedural_programs": {"meaning_rpn": "...", "usage": "..."},
    "embeddings": {"matryoshka": null, "regenerable": true}
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_phrases(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _slug(text: str) -> str:
    return text.replace(" ", "_").replace("\t", "_")


def _phrase_id(lang: str, phrase: str, sense: str) -> str:
    # Keep it stable with a hash to avoid extremely long IDs
    h = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:8]
    return f"PHRASE_{lang}_{_slug(phrase)}_{sense}_{h}"


def build_phrase_stars(jsonl_path: Path) -> List[Dict]:
    stars: List[Dict] = []
    for item in _iter_phrases(jsonl_path):
        lang = item.get("lang", "unknown")
        phrase = item.get("phrase", "")
        if not phrase:
            continue
        sense = item.get("sense", "default")
        pid = item.get("phrase_id") or _phrase_id(lang, phrase, sense)
        tokens = phrase.strip().split()
        word_refs = [{"word": tok} for tok in tokens]
        stars.append(
            {
                "phrase_id": pid,
                "lang": lang,
                "phrase": phrase,
                "sense": sense,
                "word_refs": word_refs,
                "procedural_programs": {
                    "meaning_rpn": item.get("meaning") or item.get("meaning_rpn"),
                    "usage": item.get("usage", "idiom"),
                    "register": item.get("register", "informal"),
                },
                "embeddings": {"matryoshka": None, "regenerable": True},
                "sources": item.get("sources", []),
            }
        )
    return stars


def write_jsonl(stars: List[Dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build phrase-meaning stars (idioms/multiword expressions).")
    ap.add_argument("--phrases-jsonl", type=Path, required=True, help="Input phrases JSONL.")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL for phrase stars.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    stars = build_phrase_stars(args.phrases_jsonl)
    write_jsonl(stars, args.output)
    print(f"Wrote {len(stars)} phrase stars -> {args.output}")


if __name__ == "__main__":
    main()
