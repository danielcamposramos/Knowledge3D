#!/usr/bin/env python3
"""
Ingest Universal Dependencies v2.14 treebanks into word-level stars.

Each star captures lemma-level aggregates per language:
- surface forms observed
- UPOS/XPOS frequencies
- FEATS bundle frequencies
- dependency relation frequencies
- source treebanks that contributed the evidence
- procedural morph_rpn (simple deterministic RPN string)
- meaning_program: compact JSON string (procedural payload for PD/galaxy ingest)

Input:
    --ud-root /K3D/K3D_llama_cpp/datasets/ud/ud-treebanks-v2.14
Output:
    --out /K3D/Knowledge3D.local/datasets/word_stars_ud.jsonl

No external deps beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest UD treebanks into word stars.")
    parser.add_argument(
        "--ud-root",
        required=True,
        help="Path to ud-treebanks-v2.14 directory",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path for word stars",
    )
    return parser.parse_args()


def iter_conllu(path: Path) -> Iterable[Dict[str, str]]:
    """Yield token dicts from a CoNLL-U file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) != 10:
                continue  # skip malformed
            tid = cols[0]
            if "-" in tid or "." in tid:  # skip multiword and empty nodes
                continue
            yield {
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "xpos": cols[4],
                "feats": cols[5],
                "head": cols[6],
                "deprel": cols[7],
            }


def lang_from_filename(fname: str) -> str:
    """
    Derive lang code from UD filename prefix (e.g., en_ewt-ud-train.conllu -> en).
    """
    base = os.path.basename(fname)
    if "_" in base:
        return base.split("_", 1)[0]
    return base.split("-", 1)[0]


def build_morph_rpn(lang: str, lemma: str, upos_counter: Counter, feats_counter: Counter) -> str:
    """
    Deterministic, compact RPN describing lemma morphology.
    Example: LANG en LEMMA dog POS NOUN 10 POS ADJ 2 FEAT Number=Sing 7 FEAT Number=Plur 5 END
    """
    parts = ["LANG", lang, "LEMMA", lemma]
    for pos, cnt in upos_counter.most_common():
        parts.extend(["POS", pos, str(cnt)])
    for feat, cnt in feats_counter.most_common():
        parts.extend(["FEAT", feat, str(cnt)])
    parts.append("END")
    return " ".join(parts)


def meaning_program(word_star: Dict) -> str:
    """
    Encode meaning payload as compact JSON string (procedural-friendly).
    """
    payload = {
        "lemma": word_star["lemma"],
        "lang": word_star["lang"],
        "forms": sorted(word_star["forms"]),
        "upos": word_star["upos"],
        "xpos": word_star["xpos"],
        "feats": word_star["feats"],
        "dep_roles": word_star["dep_roles"],
        "sources": sorted(word_star["sources"]),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def ingest_ud(ud_root: Path, out_path: Path) -> None:
    aggregates: Dict[Tuple[str, str], Dict] = {}

    conllu_files = list(ud_root.rglob("*.conllu"))
    for conllu_file in conllu_files:
        lang = lang_from_filename(conllu_file.name)
        treebank = conllu_file.parent.name
        for tok in iter_conllu(conllu_file):
            lemma = tok["lemma"]
            key = (lang, lemma)
            if key not in aggregates:
                aggregates[key] = {
                    "lang": lang,
                    "lemma": lemma,
                    "forms": set(),
                    "upos": Counter(),
                    "xpos": Counter(),
                    "feats": Counter(),
                    "dep_roles": Counter(),
                    "sources": set(),
                }
            agg = aggregates[key]
            agg["forms"].add(tok["form"])
            if tok["upos"]:
                agg["upos"][tok["upos"]] += 1
            if tok["xpos"]:
                agg["xpos"][tok["xpos"]] += 1
            if tok["feats"] and tok["feats"] != "_":
                for feat in tok["feats"].split("|"):
                    agg["feats"][feat] += 1
            if tok["deprel"]:
                agg["dep_roles"][tok["deprel"]] += 1
            agg["sources"].add(treebank)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for (lang, lemma), data in aggregates.items():
            morph_rpn = build_morph_rpn(lang, lemma, data["upos"], data["feats"])
            star = {
                "lang": lang,
                "lemma": lemma,
                "forms": sorted(data["forms"]),
                "upos": dict(data["upos"]),
                "xpos": dict(data["xpos"]),
                "feats": dict(data["feats"]),
                "dep_roles": dict(data["dep_roles"]),
                "sources": sorted(data["sources"]),
                "morph_rpn": morph_rpn,
                "meaning_program": meaning_program(data),
            }
            out_f.write(json.dumps(star, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ud_root = Path(args.ud_root)
    out_path = Path(args.out)
    if not ud_root.exists():
        raise FileNotFoundError(f"UD root not found: {ud_root}")
    ingest_ud(ud_root, out_path)


if __name__ == "__main__":
    main()
