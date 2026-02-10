#!/usr/bin/env python3
"""Expand Math Galaxy from local math/science corpora.

This script mines math-like tokens/patterns from local datasets and writes
K3D-standard entries into Math galaxy (same unified world).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

TOKEN_RE = re.compile(r"(\\[A-Za-z]+|[A-Za-z]{1,24}|[0-9]+(?:\\.[0-9]+)?|[=+*/^()<>-])")

OP_RPN = {
    "+": "A B ADD",
    "-": "A B SUB",
    "*": "A B MUL",
    "/": "A B DIV",
    "^": "A B POW",
    "=": "A B EQ",
    "<": "A B LT",
    ">": "A B GT",
}


def _iter_math_files(dataset_root: Path) -> Iterable[Path]:
    phase = dataset_root / "knowledge_prep_phase1b"
    for path in phase.rglob("*"):
        if path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"} and path.is_file():
            low = str(path).lower()
            if any(key in low for key in ("math", "geometry", "theorem", "competition", "algebra", "calculus", "mechanics")):
                yield path

    gb = dataset_root / "global_benchmarks"
    for sub in ("mmlu", "theoremqa", "gsm8k", "drop", "alphageometry"):
        root = gb / sub
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"} and path.is_file():
                yield path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _iter_tokens(text: str) -> Iterable[str]:
    for m in TOKEN_RE.finditer(text):
        tok = m.group(1)
        if tok.strip():
            yield tok


def _build_entry(token: str, freq: int) -> dict[str, object]:
    if token in OP_RPN:
        category = "operator"
        rpn_program = OP_RPN[token]
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
    entry_id = f"math_tok_{safe or 'sym'}"
    return {
        "id": entry_id,
        "name": token,
        "domain": "math",
        "category": category,
        "rpn_program": rpn_program,
        "metadata": {
            "frequency": int(freq),
            "source": "scripts/expand_math_galaxy.py",
            "confidence": 0.85,
            "token": token,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local/galaxies_enriched",
        help="Knowledgeverse storage root (contains galaxies/).",
    )
    parser.add_argument(
        "--dataset-root",
        default="../Knowledge3D.local/datasets",
        help="Dataset root to scan.",
    )
    parser.add_argument(
        "--max-new-entries",
        type=int,
        default=8000,
        help="Maximum number of new Math entries to add.",
    )
    args = parser.parse_args()

    kv = Knowledgeverse(storage_root=Path(args.storage_root))
    kv.ensure_default_galaxies_loaded()
    math_galaxy = kv.galaxy_manager.get_galaxy("Math")
    existing_ids = {str(entry.get("id", "")) for entry in math_galaxy.entries}

    token_counter: Counter[str] = Counter()
    sources_scanned = 0
    for file_path in _iter_math_files(Path(args.dataset_root)):
        text = _read_text(file_path)
        if not text:
            continue
        token_counter.update(_iter_tokens(text))
        sources_scanned += 1

    added = 0
    for token, freq in token_counter.most_common():
        if added >= args.max_new_entries:
            break
        entry = _build_entry(token, freq)
        entry_id = str(entry["id"])
        if entry_id in existing_ids:
            # Resolve collision deterministically
            entry_id = f"{entry_id}_{added}"
            entry["id"] = entry_id
        kv.galaxy_manager.add_entry("Math", entry)
        existing_ids.add(entry_id)
        added += 1

    payload = {
        "storage_root": str(args.storage_root),
        "dataset_root": str(args.dataset_root),
        "sources_scanned": sources_scanned,
        "added": added,
        "total": len(kv.galaxy_manager.get_galaxy("Math").entries),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
