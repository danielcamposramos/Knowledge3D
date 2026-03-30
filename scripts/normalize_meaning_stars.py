from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.galaxy_manager import normalize_disk_entry


DEFAULT_MEANING_STARS_PATH = Path("/K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl")


def normalize_meaning_stars_file(path: Path) -> dict[str, int | str]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"meaning-layer file not found: {target}")

    temp_path = target.with_suffix(target.suffix + ".tmp")
    total = 0
    normalized = 0
    skipped = 0
    written = 0

    with target.open("r", encoding="utf-8") as source, temp_path.open("w", encoding="utf-8") as handle:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(entry, dict):
                skipped += 1
                continue
            normalized_entry = normalize_disk_entry("meaning_layer_stars", entry)
            if normalized_entry != entry:
                normalized += 1
            handle.write(json.dumps(normalized_entry, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1

    temp_path.replace(target)
    return {
        "path": str(target),
        "total": total,
        "written": written,
        "normalized": normalized,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize meaning_layer_stars.jsonl to canonical galaxy schema.")
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_MEANING_STARS_PATH,
        help="Path to meaning_layer_stars.jsonl",
    )
    args = parser.parse_args()
    summary = normalize_meaning_stars_file(args.path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
