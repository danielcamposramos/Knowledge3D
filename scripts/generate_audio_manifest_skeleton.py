#!/usr/bin/env python3
"""
Generate a skeleton manifest for existing audio files.

This scans known audio roots and writes a CSV with columns:
    path,text,phoneme,lang

`text` and `phoneme` are left empty for manual/automatic filling later
because current datasets (e.g., minds14, multilingual librispeech) do
not embed per-letter labels in filenames. This manifest can then be
edited or augmented by downstream alignment tools and fed into
`scripts/proceduralize_audio.py --manifest ...`.

Default roots scanned:
    /K3D/K3D_llama_cpp/datasets/audio
    /K3D/Knowledge3D.local/datasets/audio   (if exists)

Usage:
    PYTHONPATH=. python3 scripts/generate_audio_manifest_skeleton.py \
        --output /K3D/Knowledge3D.local/datasets/audio_manifest_skeleton.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

SUPPORTED_EXTS = {".wav", ".flac", ".mp3"}

DEFAULT_ROOTS = [
    Path("/K3D/K3D_llama_cpp/datasets/audio"),
    Path("/K3D/Knowledge3D.local/datasets/audio"),
]


def list_audio_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in SUPPORTED_EXTS:
            files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def infer_lang(path: Path) -> str:
    """
    Infer language from parent folder name (best-effort).
    Expect layouts like .../audio/<lang>/.../file.wav
    """
    parts = path.parts
    if "audio" in parts:
        idx = parts.index("audio")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/datasets/audio_manifest_skeleton.csv"),
        help="Where to write the manifest CSV",
    )
    ap.add_argument(
        "--roots",
        type=Path,
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Roots to scan for audio files",
    )
    args = ap.parse_args()

    files = list_audio_files(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "text", "phoneme", "lang"])
        writer.writeheader()
        for p in files:
            writer.writerow(
                {
                    "path": str(p),
                    "text": "",
                    "phoneme": "",
                    "lang": infer_lang(p),
                }
            )

    print(f"Wrote skeleton manifest with {len(files)} rows to {args.output}")


if __name__ == "__main__":
    main()
