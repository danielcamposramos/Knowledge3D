#!/usr/bin/env python3
"""Populate Character Galaxy with procedural glyph entries.

This script is intentionally lightweight and deterministic:
- Uses ASCII + Latin-1 ranges by default.
- Writes Character entries in K3D Galaxy JSONL format.
- Avoids duplicate IDs by checking existing entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def _iter_codepoints(include_latin1: bool) -> Iterable[int]:
    # Printable ASCII
    for cp in range(32, 127):
        yield cp
    if include_latin1:
        # Latin-1 supplement (printable subset)
        for cp in range(160, 256):
            yield cp


def _iter_printable_range(start_cp: int, end_cp: int) -> Iterable[int]:
    for cp in range(max(32, start_cp), max(32, end_cp) + 1):
        try:
            ch = chr(cp)
        except ValueError:
            continue
        if ch.isprintable():
            yield cp


def _glyph_rpn_for_codepoint(cp: int) -> str:
    # Deterministic tiny procedural program; keeps hot path sovereignty.
    x = ((cp % 13) * 0.03) + 0.2
    y = ((cp % 7) * 0.02) + 0.2
    w = 0.2 + ((cp % 5) * 0.04)
    h = 0.3 + ((cp % 3) * 0.06)
    return f"{x:.3f} {y:.3f} MOVE {x + w:.3f} {y:.3f} LINE {x + w:.3f} {y + h:.3f} LINE {x:.3f} {y + h:.3f} LINE CLOSE STROKE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local/galaxies_enriched",
        help="Knowledgeverse storage root (the directory that contains galaxies/).",
    )
    parser.add_argument(
        "--include-latin1",
        action="store_true",
        help="Also populate Latin-1 supplement printable codepoints (160-255).",
    )
    parser.add_argument(
        "--max-codepoint",
        type=int,
        default=None,
        help="Optional upper codepoint bound for printable auto-seeding (e.g. 2303).",
    )
    args = parser.parse_args()

    kv = Knowledgeverse(storage_root=Path(args.storage_root))
    kv.ensure_default_galaxies_loaded()
    character = kv.galaxy_manager.get_galaxy("Character")
    existing_ids = {str(entry.get("id", "")) for entry in character.entries}

    added = 0
    codepoints: set[int] = set(_iter_codepoints(include_latin1=args.include_latin1))
    if args.max_codepoint is not None:
        codepoints.update(_iter_printable_range(32, args.max_codepoint))

    for cp in sorted(codepoints):
        char = chr(cp)
        entry_id = f"char_u{cp:04x}"
        if entry_id in existing_ids:
            continue
        entry = {
            "id": entry_id,
            "name": char,
            "domain": "character",
            "category": "glyph",
            "rpn_program": _glyph_rpn_for_codepoint(cp),
            "metadata": {
                "codepoint": cp,
                "char": char,
                "script": "latin",
                "source": "scripts/populate_character_galaxy.py",
                "symlink": "drawing_galaxy",
                "form_to_meaning": True,
                "form_premise": "drawing_procedural_glyph",
                "positive_form_ref": entry_id,
                "negative_form_ref": f"{entry_id}::negative",
                "negative_form_strategy": "canvas_minus_positive",
                "form_polarity_support": ["positive", "negative"],
                "confidence": 0.95,
            },
        }
        kv.galaxy_manager.add_entry("Character", entry)
        added += 1
        existing_ids.add(entry_id)

    payload = {
        "storage_root": str(args.storage_root),
        "added": added,
        "total": len(kv.galaxy_manager.get_galaxy("Character").entries),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
