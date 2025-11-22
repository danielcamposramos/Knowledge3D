#!/usr/bin/env python3
"""
Driver script to build meaning-first, procedural-first galaxies (JSONL artifacts)
from existing datasets. Upsert bridge currently writes JSONL snapshots; replace
with ProceduralGalaxy/House GLB wiring when available.

Outputs (defaults under /K3D/Knowledge3D.local/galaxy_pending/):
- letter_meanings.jsonl
- word_meanings.jsonl
- math_symbols.jsonl
- math_grammar.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from knowledge3d.ingestion.atomic.letter_meaning_builder import build_letter_stars, write_jsonl as write_letters
from knowledge3d.ingestion.atomic.word_meaning_builder import build_word_stars, write_jsonl as write_words
from knowledge3d.ingestion.atomic.math_symbol_builder import build_math_symbols, write_jsonl as write_math_symbols
from knowledge3d.ingestion.atomic.math_grammar_builder import build_math_grammar, write_jsonl as write_math_grammar
from knowledge3d.ingestion.atomic.syllable_builder import build_syllable_stars, write_jsonl as write_syllables
from knowledge3d.ingestion.atomic.morpheme_builder import build_morpheme_stars, write_jsonl as write_morphemes
from knowledge3d.ingestion.atomic.phrase_builder import build_phrase_stars, write_jsonl as write_phrases
from knowledge3d.ingestion.upsert_bridge import GalaxyUpsertBridge


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build meaning-first, procedural-first galaxy JSONLs.")
    ap.add_argument("--fonts-jsonl", type=Path, help="fonts_*_procedural.jsonl (per script) → letter meanings.")
    ap.add_argument("--word-stars", type=Path, help="word_stars_all.jsonl → word meanings.")
    ap.add_argument("--syllables-jsonl", type=Path, help="syllables JSONL → syllable stars.")
    ap.add_argument("--morphemes-jsonl", type=Path, help="morphemes JSONL → morpheme stars.")
    ap.add_argument("--math-symbols-jsonl", type=Path, help="math_symbols_procedural.jsonl → math symbols.")
    ap.add_argument("--phrases-jsonl", type=Path, help="phrases/idioms JSONL → phrase meanings.")
    ap.add_argument("--output-dir", type=Path, default=Path("/K3D/Knowledge3D.local/galaxy_pending"), help="Output directory for JSONLs.")
    ap.add_argument("--export-gltf", action="store_true", help="Also emit minimal .gltf snapshots with extras.k3d.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    bridge = GalaxyUpsertBridge(output_dir=args.output_dir)

    if args.fonts_jsonl:
        letters = build_letter_stars(args.fonts_jsonl)
        path = bridge.upsert_letter_meanings(letters)
        print(f"Letter meanings: {len(letters)} -> {path}")
        if args.export_gltf:
            gpath = bridge.export_gltf("letter_meanings", letters)
            print(f"Letter meanings glTF: {gpath}")

    if args.word_stars:
        words = build_word_stars(args.word_stars)
        path = bridge.upsert_word_meanings(words)
        print(f"Word meanings: {len(words)} -> {path}")
        if args.export_gltf:
            gpath = bridge.export_gltf("word_meanings", words)
            print(f"Word meanings glTF: {gpath}")

    if args.syllables_jsonl:
        syllables = build_syllable_stars(args.syllables_jsonl)
        spath = bridge._write_jsonl("syllables", syllables)
        print(f"Syllables: {len(syllables)} -> {spath}")

    if args.morphemes_jsonl:
        morphemes = build_morpheme_stars(args.morphemes_jsonl)
        mpath = bridge._write_jsonl("morphemes", morphemes)
        print(f"Morphemes: {len(morphemes)} -> {mpath}")

    if args.math_symbols_jsonl:
        math_syms = build_math_symbols(args.math_symbols_jsonl)
        path = bridge.upsert_math_symbols(math_syms)
        print(f"Math symbols: {len(math_syms)} -> {path}")
        if args.export_gltf:
            gpath = bridge.export_gltf("math_symbols", math_syms)
            print(f"Math symbols glTF: {gpath}")

    if args.phrases_jsonl:
        phrases = build_phrase_stars(args.phrases_jsonl)
        ppath = bridge.upsert_phrase_meanings(phrases)
        print(f"Phrases: {len(phrases)} -> {ppath}")
        if args.export_gltf:
            gpath = bridge.export_gltf("phrase_meanings", phrases)
            print(f"Phrases glTF: {gpath}")

    grammar = build_math_grammar()
    gpath = bridge._write_jsonl("math_grammar", grammar)
    print(f"Math grammar: {len(grammar)} -> {gpath}")
    if args.export_gltf:
        g2 = bridge.export_gltf("math_grammar", grammar)
        print(f"Math grammar glTF: {g2}")


if __name__ == "__main__":
    main()
