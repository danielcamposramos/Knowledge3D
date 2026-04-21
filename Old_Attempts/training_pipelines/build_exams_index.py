"""
Build exams_index.json combining ARC training tasks and any existing entries.

ARC tasks are referenced directly from the local datasets server:
  /exams/arc-src/data/training/<file>.json

Usage:
  python3 -m knowledge3d.tools.build_exams_index --max-arc 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build exams_index.json from local datasets")
    ap.add_argument("--max-arc", type=int, default=200)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root = repo.parent / f"{repo.name}.local" / "datasets" / "exams"
    idx_path = root.parent / "exams_index.json"

    entries: List[dict] = []
    if idx_path.exists():
        try:
            entries = json.loads(idx_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []

    # Remove previous ARC entries
    entries = [e for e in entries if e.get("source") != "ARC-AGI"]

    # List ARC training tasks
    arc_dir = root / "arc-src" / "data" / "training"
    if arc_dir.exists():
        files = sorted([p for p in arc_dir.iterdir() if p.suffix == ".json"])[: int(args.max_arc)]
        for p in files:
            entries.append({
                "id": p.stem,
                "source": "ARC-AGI",
                "title": p.stem,
                "url": f"/exams/arc-src/data/training/{p.name}",
                "kind": "arc",
            })
    else:
        print(f"ARC dir not found: {arc_dir}")

    idx_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote index with {len(entries)} items -> {idx_path}")


if __name__ == "__main__":
    main()

