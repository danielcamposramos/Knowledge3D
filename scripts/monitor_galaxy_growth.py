#!/usr/bin/env python3
"""Monitor Galaxy growth from autonomous generation metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def monitor_galaxy_growth(storage_root: Path) -> dict[str, dict[str, Any]]:
    """Inspect generated-entry growth across core galaxies."""
    kv = Knowledgeverse(storage_root=storage_root)
    report: dict[str, dict[str, Any]] = {}

    for galaxy_name in ("Drawing", "Grammar", "Math", "Reality", "3DObjects", "Audio"):
        galaxy = kv.galaxy_manager.get_galaxy(galaxy_name)
        generated_entries = [
            entry
            for entry in galaxy.entries
            if isinstance(entry, dict)
            and isinstance(entry.get("metadata", {}), dict)
            and entry.get("metadata", {}).get("generated") is True
        ]
        source_galaxies: dict[str, int] = {}
        for entry in generated_entries:
            source = str(entry.get("metadata", {}).get("source_galaxy", "unknown"))
            source_galaxies[source] = source_galaxies.get(source, 0) + 1

        total = len(galaxy.entries)
        generated_total = len(generated_entries)
        report[galaxy_name] = {
            "total_entries": total,
            "generated_entries": generated_total,
            "generation_rate": (generated_total / total) if total else 0.0,
            "source_galaxies": source_galaxies,
        }

    return report


def _print_report(report: dict[str, dict[str, Any]]) -> None:
    print("\n=== GALAXY UNIVERSE GROWTH REPORT ===")
    for galaxy_name, stats in report.items():
        print(f"\n{galaxy_name} Galaxy:")
        print(f"  Total entries: {stats['total_entries']}")
        print(
            "  Generated entries: "
            f"{stats['generated_entries']} ({stats['generation_rate']:.1%})"
        )
        if stats["source_galaxies"]:
            print(f"  Sources: {stats['source_galaxies']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-root",
        default="../Knowledge3D.local/galaxies_enriched",
        help="Knowledgeverse storage root to inspect",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON report",
    )
    args = parser.parse_args()

    report = monitor_galaxy_growth(Path(args.storage_root))
    _print_report(report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
