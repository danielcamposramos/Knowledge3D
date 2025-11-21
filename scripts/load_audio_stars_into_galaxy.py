#!/usr/bin/env python3
"""
Load procedural audio stars into the character Galaxy.

This is a lightweight, non-training loader that attaches language-qualified
audio programs (phoneme/name) to existing character entries. It expects
audio stars produced by scripts/build_audio_stars.py:
    {
      "id": "en:phoneme-en:A",
      "language": "en",
      "kind": "phoneme-en",
      "phoneme": "A",
      "harmonics": [...],
      "envelope": [...],
      "noise_level": 0.0,
      "source": "...",
      ...
    }

Outline:
  - Read audio_stars JSONL
  - For each star, build a minimal audio descriptor and upsert into Galaxy
  - Mapping strategy: use "phoneme" field as key; attach under character audio
    bucket keyed by kind (phoneme-<lang>, spoken_name-<lang>, etc.)

This operates at the metadata layer; actual procedural audio is already
encoded as harmonics/envelope. No CPU audio calculation is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Note: No direct Galaxy upsert bridge exists in sovereign_bridges.py.
# This loader prepares upsert payloads; replace the TODO hook below with the
# project’s actual Galaxy/House attachment API when available.


def load_audio_stars(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def build_audio_entry(star: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": star.get("kind", "phoneme"),
        "language": star.get("language", "unknown"),
        "phoneme": star.get("phoneme", ""),
        "sample_rate": star.get("sample_rate"),
        "duration_sec": star.get("duration_sec"),
        "harmonics": star.get("harmonics", []),
        "envelope": star.get("envelope", []),
        "noise_level": star.get("noise_level", 0.0),
        "source": star.get("source", ""),
        "timestamp": star.get("timestamp", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audio-stars",
        type=Path,
        required=True,
        help="Path to audio_stars JSONL built by build_audio_stars.py",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/audio_stars_attached.jsonl"),
        help="Optional output JSONL of upsert payloads (diagnostic).",
    )
    args = ap.parse_args()

    collected = 0
    out_entries = []
    for star in load_audio_stars(args.audio_stars):
        entry = build_audio_entry(star)
        # TODO: Replace this with the project’s Galaxy/House upsert call.
        out_entries.append(entry)
        collected += 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for e in out_entries:
                f.write(json.dumps(e) + "\n")

    print(f"Prepared {collected} audio entries. Diagnostic dump: {args.output}")


if __name__ == "__main__":
    main()
