#!/usr/bin/env python3
"""Week 22 multi-curriculum ingestion orchestrator (single persistent world).

This script intentionally runs all ingestion stages inside one Knowledgeverse
instance so the same world stays active through the entire pass.

Dependency order:
1. Character (form primitives; drawing-symlinked)
2. Word (references Character entries)
3. Math (symbolic/domain expansion)
4. Reality and 3DObjects (procedural systems expansion)
5. Optional SleepTime consolidation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.objects_3d_galaxy import default_3d_objects_entries
from knowledge3d.knowledgeverse.reality_galaxy import default_reality_entries

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,31}")
MATH_TOKEN_RE = re.compile(r"(\\[A-Za-z]+|[A-Za-z]{1,24}|[0-9]+(?:\\.[0-9]+)?|[=+*/^()<>-])")

MATH_OP_RPN = {
    "+": "A B ADD",
    "-": "A B SUB",
    "*": "A B MUL",
    "/": "A B DIV",
    "^": "A B POW",
    "=": "A B EQ",
    "<": "A B LT",
    ">": "A B GT",
}

REALITY_KEYWORD_TEMPLATES: dict[str, str] = {
    "force": "M A MUL",
    "acceleration": "DV DT DIV",
    "velocity": "DX DT DIV",
    "momentum": "M V MUL",
    "energy": "0.5 M MUL V V MUL MUL",
    "kinetic": "0.5 M MUL V V MUL MUL",
    "potential": "M G MUL H MUL",
    "gravity": "G CONST",
    "friction": "MU N MUL",
    "torque": "R F MUL",
    "power": "WORK TIME DIV",
    "work": "F D MUL",
    "pressure": "F A DIV",
    "density": "M VOLUME DIV",
    "temperature": "Q C MUL DIV",
    "entropy": "Q_REV T DIV",
    "electric": "K Q MUL R R MUL DIV",
    "magnetic": "Q V MUL B MUL",
    "wave": "FREQ LAMBDA MUL",
    "frequency": "1 PERIOD DIV",
    "amplitude": "SIGNAL AMP_SCALE MUL",
    "chemical": "REACTION_RATE C1 C2 MUL",
    "molecule": "ATOM_GRAPH BUILD",
    "cell": "STATE TRANSITION APPLY",
    "dna": "BASES ENCODE",
}

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


def _galaxy_counts(kv: Knowledgeverse) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in kv.DEFAULT_GALAXIES:
        counts[name] = len(kv.galaxy_manager.get_galaxy(name).entries)
    return counts


def _iter_printable_codepoints(max_codepoint: int) -> list[int]:
    cps: set[int] = set()
    for cp in range(32, 127):
        cps.add(cp)
    for cp in range(160, 256):
        cps.add(cp)
    for cp in range(32, max(32, max_codepoint) + 1):
        try:
            ch = chr(cp)
        except ValueError:
            continue
        if ch.isprintable():
            cps.add(cp)
    return sorted(cps)


def _glyph_rpn_for_codepoint(cp: int) -> str:
    x = ((cp % 13) * 0.03) + 0.2
    y = ((cp % 7) * 0.02) + 0.2
    w = 0.2 + ((cp % 5) * 0.04)
    h = 0.3 + ((cp % 3) * 0.06)
    return (
        f"{x:.3f} {y:.3f} MOVE {x + w:.3f} {y:.3f} LINE "
        f"{x + w:.3f} {y + h:.3f} LINE {x:.3f} {y + h:.3f} LINE CLOSE STROKE"
    )


def _populate_character(kv: Knowledgeverse, *, max_codepoint: int) -> int:
    galaxy = kv.galaxy_manager.get_galaxy("Character")
    existing_ids = {str(entry.get("id", "")) for entry in galaxy.entries}
    added = 0
    for cp in _iter_printable_codepoints(max_codepoint=max_codepoint):
        char = chr(cp)
        entry_id = f"char_u{cp:04x}"
        if entry_id in existing_ids:
            continue
        kv.galaxy_manager.add_entry(
            "Character",
            {
                "id": entry_id,
                "name": char,
                "domain": "character",
                "category": "glyph",
                "rpn_program": _glyph_rpn_for_codepoint(cp),
                "metadata": {
                    "codepoint": cp,
                    "char": char,
                    "source": "scripts/week22_multicurriculum_ingestion.py",
                    "symlink": "drawing_galaxy",
                    "form_to_meaning": True,
                    "form_premise": "drawing_procedural_glyph",
                    "positive_form_ref": entry_id,
                    "negative_form_ref": f"{entry_id}::negative",
                    "negative_form_strategy": "canvas_minus_positive",
                    "form_polarity_support": ["positive", "negative"],
                    "confidence": 0.95,
                },
            },
        )
        existing_ids.add(entry_id)
        added += 1
    return added


def _iter_word_sources(dataset_root: Path, include_global: bool) -> list[Path]:
    out: list[Path] = []
    phase = dataset_root / "knowledge_prep_phase1b"
    if phase.exists():
        for path in phase.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"}:
                out.append(path)
    if include_global:
        glob = dataset_root / "global_benchmarks"
        if glob.exists():
            for path in glob.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"}:
                    out.append(path)
    return out


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _populate_word(
    kv: Knowledgeverse,
    *,
    dataset_root: Path,
    include_global_benchmarks: bool,
    max_words: int,
) -> tuple[int, int]:
    word_galaxy = kv.galaxy_manager.get_galaxy("Word")
    character_galaxy = kv.galaxy_manager.get_galaxy("Character")
    existing_word_ids = {str(entry.get("id", "")) for entry in word_galaxy.entries}
    existing_char_ids = {str(entry.get("id", "")) for entry in character_galaxy.entries}

    counter: Counter[str] = Counter()
    sources = _iter_word_sources(dataset_root, include_global_benchmarks)
    for file_path in sources:
        text = _read_text(file_path)
        if not text:
            continue
        for match in WORD_RE.finditer(text):
            token = match.group(0).lower()
            if len(token) >= 2:
                counter[token] += 1

    added = 0
    for token, freq in counter.most_common():
        if added >= max_words:
            break
        safe = re.sub(r"[^a-z0-9_]+", "_", token).strip("_")
        if not safe:
            continue
        word_id = f"word_{safe}"
        if word_id in existing_word_ids:
            continue
        char_refs = [f"char_u{ord(ch):04x}" for ch in token]
        if any(ref not in existing_char_ids for ref in char_refs):
            continue
        kv.galaxy_manager.add_entry(
            "Word",
            {
                "id": word_id,
                "name": token,
                "domain": "word",
                "category": "lexeme",
                "rpn_program": f"WORD {token} TOKEN",
                "metadata": {
                    "frequency": int(freq),
                    "char_refs": char_refs,
                    "symlink": "character_galaxy",
                    "form_to_meaning": True,
                    "source": "scripts/week22_multicurriculum_ingestion.py",
                    "confidence": 0.9,
                },
            },
        )
        existing_word_ids.add(word_id)
        added += 1
    return added, len(sources)


def _iter_math_sources(dataset_root: Path) -> list[Path]:
    out: list[Path] = []
    phase = dataset_root / "knowledge_prep_phase1b"
    if phase.exists():
        for path in phase.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".jsonl", ".txt", ".csv"}:
                continue
            low = str(path).lower()
            if any(key in low for key in ("math", "geometry", "theorem", "competition", "algebra", "calculus", "mechanics")):
                out.append(path)
    global_b = dataset_root / "global_benchmarks"
    for sub in ("mmlu", "theoremqa", "gsm8k", "drop", "alphageometry", "math"):
        root = global_b / sub
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"}:
                out.append(path)
    return out


def _iter_reality_sources(dataset_root: Path) -> list[Path]:
    out: list[Path] = []
    phase = dataset_root / "knowledge_prep_phase1b"
    keys = (
        "classical_mechanics",
        "physics",
        "chemistry",
        "biology",
        "thermo",
        "optics",
        "acoustic",
        "electromag",
        "reality",
    )
    if phase.exists():
        for path in phase.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".jsonl", ".txt", ".csv"}:
                continue
            low = str(path).lower()
            if any(key in low for key in keys):
                out.append(path)
    return out


def _iter_3d_sources(dataset_root: Path) -> list[Path]:
    out: list[Path] = []
    phase = dataset_root / "knowledge_prep_phase1b"
    keys = (
        "geometry",
        "drawing",
        "graphics",
        "spatial",
        "arc_agi_training",
    )
    if phase.exists():
        for path in phase.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".jsonl", ".txt", ".csv"}:
                continue
            low = str(path).lower()
            if any(key in low for key in keys):
                out.append(path)
    return out


def _stable_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _extract_sentences(text: str) -> list[str]:
    raw = re.split(r"[.\n!?;]+", text)
    out = []
    for line in raw:
        line = line.strip()
        if len(line) >= 20:
            out.append(line[:400])
    return out


def _build_math_entry(token: str, freq: int, fallback_index: int) -> dict[str, Any]:
    if token in MATH_OP_RPN:
        category = "operator"
        rpn_program = MATH_OP_RPN[token]
    elif token.startswith("\\"):
        category = "latex_command"
        rpn_program = f"TOKEN {token} LATEX_OP"
    elif token.replace(".", "", 1).isdigit():
        category = "number_literal"
        rpn_program = f"{token} PUSH"
    elif len(token) == 1 and token.isalpha():
        category = "symbol_variable"
        rpn_program = f"VAR {token} LOAD"
    else:
        category = "math_token"
        rpn_program = f"TOKEN {token} LOOKUP"

    safe = re.sub(r"[^a-zA-Z0-9]+", "_", token).strip("_")
    entry_id = f"math_tok_{safe or 'sym'}_{fallback_index}"
    return {
        "id": entry_id,
        "name": token,
        "domain": "math",
        "category": category,
        "rpn_program": rpn_program,
        "metadata": {
            "frequency": int(freq),
            "source": "scripts/week22_multicurriculum_ingestion.py",
            "confidence": 0.85,
            "token": token,
        },
    }


def _populate_math(kv: Knowledgeverse, *, dataset_root: Path, max_new_entries: int) -> tuple[int, int]:
    math_galaxy = kv.galaxy_manager.get_galaxy("Math")
    existing_ids = {str(entry.get("id", "")) for entry in math_galaxy.entries}
    token_counter: Counter[str] = Counter()
    sources = _iter_math_sources(dataset_root)
    for file_path in sources:
        text = _read_text(file_path)
        if not text:
            continue
        for m in MATH_TOKEN_RE.finditer(text):
            token = m.group(1)
            if token:
                token_counter[token] += 1

    added = 0
    collision_index = 0
    for token, freq in token_counter.most_common():
        if added >= max_new_entries:
            break
        entry = _build_math_entry(token, freq, collision_index)
        while str(entry["id"]) in existing_ids:
            collision_index += 1
            entry = _build_math_entry(token, freq, collision_index)
        kv.galaxy_manager.add_entry("Math", entry)
        existing_ids.add(str(entry["id"]))
        added += 1
        collision_index += 1

    return added, len(sources)


def _populate_reality_from_corpus(
    kv: Knowledgeverse,
    *,
    dataset_root: Path,
    max_new_entries: int,
) -> tuple[int, int]:
    galaxy = kv.galaxy_manager.get_galaxy("Reality")
    existing_ids = {str(entry.get("id", "")) for entry in galaxy.entries}
    added = 0
    sources = _iter_reality_sources(dataset_root)
    for file_path in sources:
        if added >= max_new_entries:
            break
        text = _read_text(file_path)
        if not text:
            continue
        for sentence in _extract_sentences(text):
            if added >= max_new_entries:
                break
            low = sentence.lower()
            matched = [kw for kw in REALITY_KEYWORD_TEMPLATES if kw in low]
            if not matched:
                continue
            keyword = matched[0]
            digest = _stable_digest(f"{file_path}:{sentence}:{keyword}")
            entry_id = f"reality_corpus_{keyword}_{digest}"
            if entry_id in existing_ids:
                continue
            kv.galaxy_manager.add_entry(
                "Reality",
                {
                    "id": entry_id,
                    "name": f"Corpus Reality Pattern ({keyword})",
                    "domain": "reality",
                    "category": "corpus_procedural",
                    "rpn_program": REALITY_KEYWORD_TEMPLATES[keyword],
                    "metadata": {
                        "keyword": keyword,
                        "source_file": str(file_path),
                        "source_sentence": sentence,
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "cross_modal": ["math", "drawing", "3d_objects"],
                        "confidence": 0.78,
                        "procedural": True,
                        "generative": True,
                    },
                },
            )
            existing_ids.add(entry_id)
            added += 1
    return added, len(sources)


def _populate_3dobjects_from_corpus(
    kv: Knowledgeverse,
    *,
    dataset_root: Path,
    max_new_entries: int,
) -> tuple[int, int]:
    galaxy = kv.galaxy_manager.get_galaxy("3DObjects")
    existing_ids = {str(entry.get("id", "")) for entry in galaxy.entries}
    added = 0
    sources = _iter_3d_sources(dataset_root)
    for file_path in sources:
        if added >= max_new_entries:
            break
        text = _read_text(file_path)
        if not text:
            continue
        for sentence in _extract_sentences(text):
            if added >= max_new_entries:
                break
            low = sentence.lower()
            matched = [kw for kw in OBJECT3D_KEYWORD_TEMPLATES if kw in low]
            if not matched:
                continue
            keyword = matched[0]
            digest = _stable_digest(f"{file_path}:{sentence}:{keyword}")
            entry_id = f"obj3d_corpus_{keyword}_{digest}"
            if entry_id in existing_ids:
                continue
            kv.galaxy_manager.add_entry(
                "3DObjects",
                {
                    "id": entry_id,
                    "name": f"Corpus 3D Pattern ({keyword})",
                    "domain": "3d_objects",
                    "category": "corpus_spatial",
                    "rpn_program": OBJECT3D_KEYWORD_TEMPLATES[keyword],
                    "metadata": {
                        "keyword": keyword,
                        "source_file": str(file_path),
                        "source_sentence": sentence,
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "cross_modal": ["drawing", "math", "reality"],
                        "confidence": 0.76,
                        "procedural": True,
                        "generative": True,
                    },
                },
            )
            existing_ids.add(entry_id)
            added += 1
    return added, len(sources)


def _populate_reality_parametric_sweep(kv: Knowledgeverse, *, max_new_entries: int) -> int:
    galaxy = kv.galaxy_manager.get_galaxy("Reality")
    existing_ids = {str(entry.get("id", "")) for entry in galaxy.entries}
    added = 0

    for mass in range(1, 61):
        for accel in range(-15, 16):
            if added >= max_new_entries:
                return added
            entry_id = f"reality_sweep_newton_m{mass}_a{accel:+d}"
            if entry_id in existing_ids:
                continue
            kv.galaxy_manager.add_entry(
                "Reality",
                {
                    "id": entry_id,
                    "name": f"Newton Sweep m={mass} a={accel}",
                    "domain": "reality",
                    "category": "parametric_sweep_dynamics",
                    "rpn_program": f"{mass} {accel} MUL",
                    "metadata": {
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "cross_modal": ["math", "3d_objects"],
                        "procedural": True,
                        "generative": True,
                        "confidence": 0.74,
                    },
                },
            )
            existing_ids.add(entry_id)
            added += 1

    for charge in range(1, 41):
        for distance in range(1, 21):
            if added >= max_new_entries:
                return added
            entry_id = f"reality_sweep_coulomb_q{charge}_r{distance}"
            if entry_id in existing_ids:
                continue
            kv.galaxy_manager.add_entry(
                "Reality",
                {
                    "id": entry_id,
                    "name": f"Coulomb Sweep q={charge} r={distance}",
                    "domain": "reality",
                    "category": "parametric_sweep_electromagnetism",
                    "rpn_program": f"K {charge} MUL {distance} {distance} MUL DIV",
                    "metadata": {
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "cross_modal": ["math", "drawing"],
                        "procedural": True,
                        "generative": True,
                        "confidence": 0.73,
                    },
                },
            )
            existing_ids.add(entry_id)
            added += 1

    return added


def _populate_3d_parametric_sweep(kv: Knowledgeverse, *, max_new_entries: int) -> int:
    galaxy = kv.galaxy_manager.get_galaxy("3DObjects")
    existing_ids = {str(entry.get("id", "")) for entry in galaxy.entries}
    added = 0

    for radius in range(1, 26):
        for stacks in range(6, 42, 2):
            if added >= max_new_entries:
                return added
            slices = stacks * 2
            entry_id = f"obj3d_sweep_sphere_r{radius}_s{stacks}_c{slices}"
            if entry_id in existing_ids:
                continue
            kv.galaxy_manager.add_entry(
                "3DObjects",
                {
                    "id": entry_id,
                    "name": f"Sphere Sweep r={radius} stacks={stacks} slices={slices}",
                    "domain": "3d_objects",
                    "category": "parametric_sweep_mesh",
                    "rpn_program": f"{radius} {stacks} {slices} GENERATE_UV_SPHERE",
                    "metadata": {
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "cross_modal": ["drawing", "math", "reality"],
                        "procedural": True,
                        "generative": True,
                        "confidence": 0.76,
                    },
                },
            )
            existing_ids.add(entry_id)
            added += 1

    for tx in range(-5, 6):
        for ty in range(-5, 6):
            if added >= max_new_entries:
                return added
            entry_id = f"obj3d_sweep_translate_tx{tx:+d}_ty{ty:+d}"
            if entry_id in existing_ids:
                continue
            kv.galaxy_manager.add_entry(
                "3DObjects",
                {
                    "id": entry_id,
                    "name": f"Translate Sweep tx={tx} ty={ty}",
                    "domain": "3d_objects",
                    "category": "parametric_sweep_transform",
                    "rpn_program": f"{tx} {ty} 0 MAT4_TRANSLATE",
                    "metadata": {
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "cross_modal": ["drawing", "math", "reality"],
                        "procedural": True,
                        "generative": True,
                        "confidence": 0.74,
                    },
                },
            )
            existing_ids.add(entry_id)
            added += 1
    return added


def _append_missing_entries(kv: Knowledgeverse, galaxy_name: str, rows: list[dict[str, Any]]) -> int:
    galaxy = kv.galaxy_manager.get_galaxy(galaxy_name)
    existing_ids = {str(entry.get("id", "")) for entry in galaxy.entries}
    added = 0
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id or row_id in existing_ids:
            continue
        kv.galaxy_manager.add_entry(galaxy_name, row)
        existing_ids.add(row_id)
        added += 1
    return added


def _populate_grammar_cross_modal(kv: Knowledgeverse, *, max_new_entries: int) -> int:
    grammar = kv.galaxy_manager.get_galaxy("Grammar")
    existing_ids = {str(entry.get("id", "")) for entry in grammar.entries}
    added = 0

    sources: list[tuple[str, list[dict[str, Any]]]] = []
    for galaxy_name in ("Word", "Math", "Reality", "3DObjects"):
        galaxy = kv.galaxy_manager.get_galaxy(galaxy_name)
        # Prefer recent entries for bridge creation.
        sources.append((galaxy_name, list(galaxy.entries[-200:])))

    for galaxy_name, entries in sources:
        for entry in entries:
            if added >= max_new_entries:
                return added
            target_id = str(entry.get("id", "")).strip()
            if not target_id:
                continue
            rule_id = f"grammar_bridge_{galaxy_name.lower()}_{target_id}"
            if rule_id in existing_ids:
                continue
            name = str(entry.get("name", target_id))
            kv.galaxy_manager.add_entry(
                "Grammar",
                {
                    "id": rule_id,
                    "name": f"Bridge {galaxy_name}:{name}",
                    "domain": "grammar",
                    "category": "cross_modal_bridge",
                    "rpn_program": f"QUERY {galaxy_name.upper()} {target_id} COMPOSE_RULE",
                    "metadata": {
                        "source": "scripts/week22_multicurriculum_ingestion.py",
                        "bridge_target_galaxy": galaxy_name,
                        "bridge_target_id": target_id,
                        "symlink": f"{galaxy_name.lower()}_galaxy",
                        "cross_modal": True,
                        "confidence": 0.8,
                    },
                },
            )
            existing_ids.add(rule_id)
            added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local/galaxies_enriched",
        help="Knowledgeverse world root (contains galaxies/).",
    )
    parser.add_argument(
        "--dataset-root",
        default="/K3D/Knowledge3D.local/datasets",
        help="Dataset root for corpus ingestion.",
    )
    parser.add_argument(
        "--character-max-codepoint",
        type=int,
        default=2303,
        help="Upper printable codepoint bound for Character expansion.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=20000,
        help="Maximum new Word entries to add.",
    )
    parser.add_argument(
        "--max-math-entries",
        type=int,
        default=8000,
        help="Maximum new Math entries to add.",
    )
    parser.add_argument(
        "--max-reality-corpus-entries",
        type=int,
        default=1200,
        help="Maximum new Reality entries mined from corpus text.",
    )
    parser.add_argument(
        "--max-3d-corpus-entries",
        type=int,
        default=600,
        help="Maximum new 3DObjects entries mined from corpus text.",
    )
    parser.add_argument(
        "--max-grammar-bridge-entries",
        type=int,
        default=1200,
        help="Maximum cross-modal bridge rules to add to Grammar.",
    )
    parser.add_argument(
        "--max-reality-sweep-entries",
        type=int,
        default=1200,
        help="Maximum additional Reality parametric sweep entries.",
    )
    parser.add_argument(
        "--max-3d-sweep-entries",
        type=int,
        default=800,
        help="Maximum additional 3DObjects parametric sweep entries.",
    )
    parser.add_argument(
        "--include-global-benchmarks",
        action="store_true",
        help="Use global benchmark datasets as additional text/math corpora.",
    )
    parser.add_argument(
        "--run-sleeptime",
        action="store_true",
        help="Run Knowledgeverse SleepTime execute() after ingestion stages.",
    )
    parser.add_argument(
        "--min-galaxies-touched",
        type=int,
        default=5,
        help="Soft coverage gate for number of galaxies with count deltas > 0.",
    )
    parser.add_argument(
        "--output",
        default="/K3D/Knowledge3D.local/results/week22_multicurriculum/week22_ingestion_report.json",
        help="Path to JSON report.",
    )
    args = parser.parse_args()

    storage_root = Path(args.storage_root)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kv = Knowledgeverse(storage_root=storage_root)
    before = _galaxy_counts(kv)

    char_added = _populate_character(kv, max_codepoint=args.character_max_codepoint)
    word_added, word_sources = _populate_word(
        kv,
        dataset_root=dataset_root,
        include_global_benchmarks=bool(args.include_global_benchmarks),
        max_words=int(args.max_words),
    )
    math_added, math_sources = _populate_math(
        kv,
        dataset_root=dataset_root,
        max_new_entries=int(args.max_math_entries),
    )

    reality_added = _append_missing_entries(kv, "Reality", default_reality_entries())
    objects_added = _append_missing_entries(kv, "3DObjects", default_3d_objects_entries())
    reality_corpus_added, reality_sources = _populate_reality_from_corpus(
        kv,
        dataset_root=dataset_root,
        max_new_entries=int(args.max_reality_corpus_entries),
    )
    objects_corpus_added, objects_sources = _populate_3dobjects_from_corpus(
        kv,
        dataset_root=dataset_root,
        max_new_entries=int(args.max_3d_corpus_entries),
    )
    reality_sweep_added = _populate_reality_parametric_sweep(
        kv,
        max_new_entries=int(args.max_reality_sweep_entries),
    )
    objects_sweep_added = _populate_3d_parametric_sweep(
        kv,
        max_new_entries=int(args.max_3d_sweep_entries),
    )
    grammar_bridge_added = _populate_grammar_cross_modal(
        kv,
        max_new_entries=int(args.max_grammar_bridge_entries),
    )

    sleeptime_result: dict[str, Any] | None = None
    if args.run_sleeptime:
        sleeptime_result = kv.sleeptime.execute()

    after = _galaxy_counts(kv)
    deltas = {name: int(after.get(name, 0) - before.get(name, 0)) for name in kv.DEFAULT_GALAXIES}
    touched = sorted([name for name, delta in deltas.items() if delta > 0])
    coverage_passed = len(touched) >= max(0, int(args.min_galaxies_touched))

    report = {
        "storage_root": str(storage_root),
        "dataset_root": str(dataset_root),
        "single_instance_id": int(id(kv)),
        "before_counts": before,
        "after_counts": after,
        "delta_counts": deltas,
        "stages": {
            "character_added": int(char_added),
            "word_added": int(word_added),
            "word_sources_scanned": int(word_sources),
            "math_added": int(math_added),
            "math_sources_scanned": int(math_sources),
            "reality_added": int(reality_added),
            "objects3d_added": int(objects_added),
            "reality_corpus_added": int(reality_corpus_added),
            "reality_sources_scanned": int(reality_sources),
            "reality_sweep_added": int(reality_sweep_added),
            "objects3d_corpus_added": int(objects_corpus_added),
            "objects3d_sources_scanned": int(objects_sources),
            "objects3d_sweep_added": int(objects_sweep_added),
            "grammar_bridge_added": int(grammar_bridge_added),
            "sleeptime_ran": bool(args.run_sleeptime),
            "sleeptime_result": sleeptime_result,
        },
        "coverage": {
            "touched_galaxies": touched,
            "min_required": int(args.min_galaxies_touched),
            "passed": bool(coverage_passed),
            "soft_gate_only": True,
        },
        "notes": [
            "Character uses drawing-symlink metadata (form-to-meaning premise).",
            "Word entries are only created if all character refs exist.",
            "Reality/3DObjects append missing deterministic defaults idempotently.",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
