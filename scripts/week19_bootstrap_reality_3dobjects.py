#!/usr/bin/env python3
"""Week 19 bootstrap for Reality + 3DObjects galaxies.

Appends deterministic procedural entries while preserving existing galaxy data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge3d.knowledgeverse.objects_3d_galaxy import bootstrap_3d_objects_galaxy
from knowledge3d.knowledgeverse.reality_galaxy import bootstrap_reality_galaxy


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="/K3D/Knowledge3D.local",
        help="Knowledgeverse storage root containing galaxies/*.jsonl",
    )
    parser.add_argument(
        "--print-counts",
        action="store_true",
        help="Print post-bootstrap counts for key galaxies.",
    )
    args = parser.parse_args()

    storage_root = Path(args.storage_root)
    reality_summary = bootstrap_reality_galaxy(storage_root=storage_root)
    objects3d_summary = bootstrap_3d_objects_galaxy(storage_root=storage_root)

    report = {
        "storage_root": str(storage_root),
        "summary": {
            "Reality": reality_summary,
            "3DObjects": objects3d_summary,
        },
    }
    if args.print_counts:
        galaxies = storage_root / "galaxies"
        report["counts"] = {
            "Drawing": _jsonl_count(galaxies / "Drawing.jsonl"),
            "Grammar": _jsonl_count(galaxies / "Grammar.jsonl"),
            "Math": _jsonl_count(galaxies / "Math.jsonl"),
            "Reality": _jsonl_count(galaxies / "Reality.jsonl"),
            "3DObjects": _jsonl_count(galaxies / "3DObjects.jsonl"),
            "Audio": _jsonl_count(galaxies / "Audio.jsonl"),
        }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

