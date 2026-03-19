#!/usr/bin/env python3
"""Run H19: parse the full OMW dataset into multilingual meaning stars."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.ingestion.universal_knowledge.multilingual_meanings import (  # noqa: E402
    build_meaning_layer_stars,
    meaning_layer_stats,
)
from knowledge3d.tools.content_to_stars import write_stars_jsonl  # noqa: E402


def main() -> int:
    stars = build_meaning_layer_stars(min_languages=2)
    stats = meaning_layer_stats(stars)
    output_path = Path("/K3D/Knowledge3D.local/galaxies/meaning_layer_stars.jsonl")
    write_stars_jsonl(stars, output_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    print(f"\nWrote {len(stars)} stars to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
