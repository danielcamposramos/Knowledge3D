#!/usr/bin/env python3
"""
Build procedural audio stars from one or more audio seed JSONL files.

Each seed contains procedural parameters (harmonics, envelope, noise).
This script aggregates them into a unified JSONL of "audio stars" with
explicit language-qualified kind labels (e.g., phoneme-en, spoken_name-en).

Usage:
    PYTHONPATH=. python3 scripts/build_audio_stars.py \
        --seeds /K3D/Knowledge3D.local/datasets/procedural_audio_seeds.jsonl \
        --seeds /K3D/Knowledge3D.local/datasets/procedural_audio_seeds_es.jsonl \
        --seeds /K3D/Knowledge3D.local/datasets/procedural_audio_seeds_zh.jsonl \
        --output /K3D/Knowledge3D.local/datasets/audio_stars.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def load_seeds(paths: List[Path]) -> List[Dict]:
    records: List[Dict] = []
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    return records


def build_star(seed: Dict) -> Dict:
    lang = seed.get("language", "unknown")
    phoneme = seed.get("phoneme", "")
    kind = seed.get("kind", "phoneme")
    if lang and not kind.endswith(f"-{lang}"):
        kind = f"{kind}-{lang}"
    star_id = f"{lang}:{kind}:{phoneme}"
    return {
        "id": star_id,
        "language": lang,
        "kind": kind,
        "phoneme": phoneme,
        "sample_rate": seed.get("sample_rate"),
        "duration_sec": seed.get("duration_sec"),
        "harmonics": seed.get("harmonics", []),
        "envelope": seed.get("envelope", []),
        "noise_level": seed.get("noise_level", 0.0),
        "source": seed.get("source", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=Path, nargs="+", required=True, help="Seed JSONL files to merge")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/audio_stars.jsonl"),
        help="Output JSONL for audio stars",
    )
    args = ap.parse_args()

    seeds = load_seeds(args.seeds)
    stars = [build_star(s) for s in seeds]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for star in stars:
            f.write(json.dumps(star) + "\n")

    print(f"Wrote {len(stars)} audio stars to {args.output}")


if __name__ == "__main__":
    main()
