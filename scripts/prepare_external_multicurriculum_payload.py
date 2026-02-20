#!/usr/bin/env python3
"""Prepare external dataset payloads for single-world Knowledgeverse ingestion.

This script builds JSONL payloads that can be generated in parallel and ingested
later through one persistent Knowledgeverse instance.

Each JSONL line format:
{"galaxy": "Word|Audio|3DObjects", "entry": {...}}
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable

OBJECT3D_KEYWORD_TEMPLATES: dict[str, str] = {
    "cube": "SIZE GENERATE_CUBE_VERTICES GENERATE_CUBE_FACES",
    "sphere": "RADIUS STACKS SLICES GENERATE_UV_SPHERE",
    "cylinder": "RADIUS HEIGHT SEGMENTS GENERATE_CYLINDER_MESH",
    "cone": "RADIUS HEIGHT SEGMENTS GENERATE_CONE_MESH",
    "pyramid": "BASE HEIGHT GENERATE_PYRAMID_MESH",
    "prism": "PROFILE HEIGHT GENERATE_PRISM_MESH",
    "mesh": "VERTICES FACES BUILD_MESH",
    "rotate": "ANGLE AXIS MAT4_ROT_AXIS_ANGLE",
    "rotation": "ANGLE AXIS MAT4_ROT_AXIS_ANGLE",
    "translate": "TX TY TZ MAT4_TRANSLATE",
    "translation": "TX TY TZ MAT4_TRANSLATE",
    "scale": "SX SY SZ MAT4_SCALE",
    "transform": "MAT4_A MAT4_B MAT4_MUL",
    "camera": "FOV ASPECT Z_NEAR Z_FAR MAT4_PERSPECTIVE",
    "projection": "VIEW PROJ MAT4_MUL",
    "ray": "RAY_O RAY_D TRACE_SCENE",
    "collision": "SHAPE_A SHAPE_B COLLISION_TEST",
    "voxel": "GRID VOXELIZE",
}

WORDNET_INDEX_FILES = (
    "WordNet-3.0/dict/index.noun",
    "WordNet-3.0/dict/index.verb",
    "WordNet-3.0/dict/index.adj",
    "WordNet-3.0/dict/index.adv",
)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _iter_wordnet_lemmas(wordnet_tar: Path, max_items: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not wordnet_tar.exists():
        return out

    with tarfile.open(wordnet_tar, "r:gz") as tf:
        for member_name in WORDNET_INDEX_FILES:
            try:
                fileobj = tf.extractfile(member_name)
            except KeyError:
                fileobj = None
            if fileobj is None:
                continue
            for raw in fileobj:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                lemma, pos = parts[0], parts[1]
                if pos not in {"n", "v", "a", "r", "s"}:
                    continue
                if lemma.isdigit():
                    continue
                cleaned = lemma.replace("_", " ").strip()
                if not cleaned:
                    continue
                out.append((cleaned, pos))
                if len(out) >= max_items:
                    return out
    return out


def _iter_dbnary_words(dbnary_dir: Path, langs: list[str], max_items: int) -> list[tuple[str, str]]:
    # Match lines like: dbnary:writtenForm      "ажвар"@abq .
    pattern = re.compile(r'dbnary:writtenForm\s+"(.+?)"@([a-z\-]+)')
    out: list[tuple[str, str]] = []

    for lang in langs:
        file_path = dbnary_dir / f"{lang}_dbnary_ontolex.ttl.bz2"
        if not file_path.exists():
            continue
        with bz2.open(file_path, "rt", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                m = pattern.search(line)
                if not m:
                    continue
                token = m.group(1).strip()
                tag = m.group(2).strip()
                if not token:
                    continue
                out.append((token, tag))
                if len(out) >= max_items:
                    return out
    return out


def _build_lexicon_payload(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    wordnet_terms = _iter_wordnet_lemmas(args.wordnet_tar, max_items=args.max_wordnet)
    for token, pos in wordnet_terms:
        token_norm = token.lower()
        entry_id = f"word_wordnet_{_sha(token_norm + pos)}"
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        rows.append(
            {
                "galaxy": "Word",
                "entry": {
                    "id": entry_id,
                    "name": token_norm,
                    "domain": "word",
                    "category": "lexicon_wordnet",
                    "rpn_program": f"WORD {token_norm} TOKEN",
                    "metadata": {
                        "source": "wordnet_3.0",
                        "pos": pos,
                        "symlink": "character_galaxy",
                        "cross_modal": ["character", "grammar"],
                        "confidence": 0.86,
                    },
                },
            }
        )

    dbnary_terms = _iter_dbnary_words(args.dbnary_dir, args.dbnary_langs, max_items=args.max_dbnary)
    for token, lang in dbnary_terms:
        token_norm = token.strip().lower()
        entry_id = f"word_dbnary_{_sha(token_norm + lang)}"
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)
        rows.append(
            {
                "galaxy": "Word",
                "entry": {
                    "id": entry_id,
                    "name": token_norm,
                    "domain": "word",
                    "category": "lexicon_dbnary",
                    "rpn_program": f"WORD {token_norm} TOKEN",
                    "metadata": {
                        "source": "dbnary",
                        "lang": lang,
                        "symlink": "character_galaxy",
                        "cross_modal": ["character", "grammar"],
                        "confidence": 0.8,
                    },
                },
            }
        )

    return rows


def _iter_audio_manifest_rows(audio_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest in audio_root.rglob("manifest.csv"):
        try:
            with manifest.open("r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if isinstance(row, dict):
                        rows.append({k: str(v) for k, v in row.items()})
        except Exception:
            continue
    return rows


def _build_audio_payload(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    manifest_rows = _iter_audio_manifest_rows(args.audio_root)
    for item in manifest_rows:
        path = item.get("path", "").strip()
        text = item.get("text", "").strip()
        lang = item.get("lang", "unknown").strip() or "unknown"
        if not path or not text:
            continue

        entry_id = f"audio_phoneme_{_sha(path + text + lang)}"
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)

        char_ref = f"char_u{ord(text[0]):04x}" if len(text) == 1 else None
        rows.append(
            {
                "galaxy": "Audio",
                "entry": {
                    "id": entry_id,
                    "name": f"{text} ({lang})",
                    "domain": "audio",
                    "category": "phoneme_sample",
                    "rpn_program": "FREQ AMP DUR ENVELOPE ADSR SYNTH",
                    "metadata": {
                        "source": "phoneme_external_manifest",
                        "audio_path": path,
                        "token": text,
                        "lang": lang,
                        "symlink": "character_galaxy" if char_ref else "audio_galaxy",
                        "char_ref": char_ref,
                        "cross_modal": ["character", "drawing", "word"],
                        "confidence": 0.82,
                    },
                },
            }
        )

        if len(rows) >= args.max_audio:
            break
    return rows


def _iter_geometry_sentences(text_root: Path) -> list[str]:
    out: list[str] = []
    for path in text_root.rglob("*.txt"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for sentence in re.split(r"[\n.!?;]+", content):
            s = sentence.strip()
            if len(s) >= 24:
                out.append(s)
    return out


def _build_geometry_payload(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    text_root = args.geometry_root / "text"
    sentences = _iter_geometry_sentences(text_root)

    for sentence in sentences:
        low = sentence.lower()
        matched = [kw for kw in OBJECT3D_KEYWORD_TEMPLATES if kw in low]
        if not matched:
            continue
        kw = matched[0]
        entry_id = f"obj3d_ext_{kw}_{_sha(sentence)}"
        if entry_id in seen_ids:
            continue
        seen_ids.add(entry_id)

        rows.append(
            {
                "galaxy": "3DObjects",
                "entry": {
                    "id": entry_id,
                    "name": f"External Geometry Pattern ({kw})",
                    "domain": "3d_objects",
                    "category": "external_geometry",
                    "rpn_program": OBJECT3D_KEYWORD_TEMPLATES[kw],
                    "metadata": {
                        "source": "galaxy_geometry_text",
                        "keyword": kw,
                        "source_sentence": sentence[:300],
                        "cross_modal": ["drawing", "math", "reality"],
                        "confidence": 0.78,
                        "procedural": True,
                    },
                },
            }
        )
        if len(rows) >= args.max_geometry:
            break

    if not rows and args.use_fallback_templates:
        # Dataset can have empty 3d/core_solids. Provide safe procedural fallback set.
        fallback = [
            ("cube", "SIZE GENERATE_CUBE_VERTICES GENERATE_CUBE_FACES"),
            ("sphere", "RADIUS STACKS SLICES GENERATE_UV_SPHERE"),
            ("rotate", "ANGLE AXIS MAT4_ROT_AXIS_ANGLE"),
            ("translate", "TX TY TZ MAT4_TRANSLATE"),
            ("collision", "SHAPE_A SHAPE_B COLLISION_TEST"),
        ]
        for kw, rpn in fallback:
            entry_id = f"obj3d_fallback_{kw}"
            rows.append(
                {
                    "galaxy": "3DObjects",
                    "entry": {
                        "id": entry_id,
                        "name": f"Fallback 3D Template ({kw})",
                        "domain": "3d_objects",
                        "category": "external_geometry_fallback",
                        "rpn_program": rpn,
                        "metadata": {
                            "source": "fallback_templates",
                            "keyword": kw,
                            "cross_modal": ["drawing", "math", "reality"],
                            "confidence": 0.72,
                            "procedural": True,
                        },
                    },
                }
            )

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external payloads for lexicon/audio/3D ingestion")
    parser.add_argument("--modality", choices=["lexicon", "audio", "geometry3d"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL payload path")

    parser.add_argument("--wordnet-tar", type=Path, default=Path("/K3D/K3D_llama_cpp/datasets/wordnet/WordNet-3.0.tar.gz"))
    parser.add_argument("--dbnary-dir", type=Path, default=Path("/K3D/K3D_llama_cpp/datasets/dbnary"))
    parser.add_argument("--dbnary-langs", nargs="+", default=["en", "pt", "es", "zh"])
    parser.add_argument("--max-wordnet", type=int, default=6000)
    parser.add_argument("--max-dbnary", type=int, default=4000)

    parser.add_argument("--audio-root", type=Path, default=Path("/K3D/K3D_llama_cpp/datasets/audio"))
    parser.add_argument("--max-audio", type=int, default=5000)

    parser.add_argument("--geometry-root", type=Path, default=Path("/K3D/K3D_llama_cpp/datasets/galaxy_geometry"))
    parser.add_argument("--max-geometry", type=int, default=2000)
    parser.add_argument("--use-fallback-templates", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.modality == "lexicon":
        rows = _build_lexicon_payload(args)
    elif args.modality == "audio":
        rows = _build_audio_payload(args)
    else:
        rows = _build_geometry_payload(args)

    count = _write_jsonl(args.output, rows)
    print(f"[payload] modality={args.modality} rows={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
