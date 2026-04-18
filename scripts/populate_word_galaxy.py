#!/usr/bin/env python3
"""Populate Word Galaxy using local corpora with Character symlinks.

Sources:
- knowledge_prep_phase1b markdown + enrichment JSON
- optional global benchmark JSON/CSV/TXT under datasets
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

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{1,31}")


def _iter_source_files(dataset_root: Path, include_global: bool) -> Iterable[Path]:
    phase1b = dataset_root / "knowledge_prep_phase1b"
    if phase1b.exists():
        for path in phase1b.rglob("*"):
            if path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"} and path.is_file():
                yield path
    if include_global:
        glob = dataset_root / "global_benchmarks"
        if glob.exists():
            for sub in ("mmlu", "gsm8k", "gpqa", "truthfulqa", "theoremqa", "drop"):
                root = glob / sub
                if not root.exists():
                    continue
                for path in root.rglob("*"):
                    if path.suffix.lower() in {".md", ".json", ".jsonl", ".txt", ".csv"} and path.is_file():
                        yield path


def _iter_words(text: str) -> Iterable[str]:
    for match in WORD_RE.finditer(text):
        token = match.group(0).lower()
        if len(token) < 2:
            continue
        yield token


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _char_refs_for_word(word: str) -> list[str]:
    refs: list[str] = []
    for ch in word:
        refs.append(f"char_u{ord(ch):04x}")
    return refs


def _safe_word_id(word: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", word.lower()).strip("_")
    return f"word_{safe}" if safe else "word_unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local/galaxies_enriched",
        help="Knowledgeverse storage root (contains galaxies/).",
    )
    parser.add_argument(
        "--dataset-root",
        default="/K3D/Knowledge3D.local/datasets",
        help="Dataset root to scan for corpora.",
    )
    parser.add_argument(
        "--include-global-benchmarks",
        action="store_true",
        help="Include global benchmark text payloads as corpus sources.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=20000,
        help="Maximum number of new Word entries to add.",
    )
    args = parser.parse_args()

    storage_root = Path(args.storage_root)
    dataset_root = Path(args.dataset_root)

    kv = Knowledgeverse(storage_root=storage_root)
    kv.ensure_default_galaxies_loaded()
    word_galaxy = kv.galaxy_manager.get_galaxy("Word")
    character_galaxy = kv.galaxy_manager.get_galaxy("Character")

    existing_word_ids = {str(entry.get("id", "")) for entry in word_galaxy.entries}
    existing_char_ids = {str(entry.get("id", "")) for entry in character_galaxy.entries}

    counter: Counter[str] = Counter()
    sources_scanned = 0
    for file_path in _iter_source_files(dataset_root, args.include_global_benchmarks):
        text = _read_text(file_path)
        if not text:
            continue
        counter.update(_iter_words(text))
        sources_scanned += 1

    added = 0
    for word, freq in counter.most_common():
        if added >= args.max_words:
            break
        word_id = _safe_word_id(word)
        if word_id in existing_word_ids:
            continue

        char_refs = _char_refs_for_word(word)
        # Skip words whose character refs are not present yet.
        if any(ref not in existing_char_ids for ref in char_refs):
            continue

        entry = {
            "id": word_id,
            "name": word,
            "domain": "word",
            "category": "lexeme",
            "rpn_program": f"WORD {word} TOKEN",
            "metadata": {
                "frequency": int(freq),
                "char_refs": char_refs,
                "symlink": "character_galaxy",
                "form_to_meaning": True,
                "form_refs": char_refs,
                "source": "scripts/populate_word_galaxy.py",
                "confidence": 0.9,
            },
        }
        kv.galaxy_manager.add_entry("Word", entry)
        added += 1
        existing_word_ids.add(word_id)

    word_path = kv.galaxy_manager._galaxy_path("Word")  # pylint: disable=protected-access
    on_disk_lines = 0
    if word_path.exists():
        with word_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    on_disk_lines += 1

    payload = {
        "storage_root": str(storage_root),
        "dataset_root": str(dataset_root),
        "sources_scanned": sources_scanned,
        "added": added,
        "total": len(kv.galaxy_manager.get_galaxy("Word").entries),
        "word_path": str(word_path),
        "on_disk_lines": on_disk_lines,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
