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

    # Single source dir with explicit lang/type (e.g., kana batch)
    PYTHONPATH=. python3 scripts/generate_audio_manifest_skeleton.py \
        --source_dir /K3D/K3D_llama_cpp/datasets/audio/phoneme_external/ja_kana \
        --lang ja --type phoneme \
        --output /K3D/K3D_llama_cpp/datasets/audio/phoneme_external/ja_kana_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence

# Include ogg/oga for Commons + Lingua Libre pulls.
SUPPORTED_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus"}

DEFAULT_ROOTS = [
    Path("/K3D/K3D_llama_cpp/datasets/audio"),
    Path("/K3D/Knowledge3D.local/datasets/audio"),
]


def list_audio_files(roots: Iterable[Path], exts: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    normalized = {e.lower() for e in exts}
    for root in roots:
        if not root.exists():
            continue
        for ext in normalized:
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
    ap.add_argument(
        "--source_dir",
        type=Path,
        help="Shortcut for a single source directory (overrides --roots when provided)",
    )
    ap.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Optional language override to set the lang column for all rows",
    )
    ap.add_argument(
        "--type",
        type=str,
        default=None,
        help="Optional type column value for all rows (e.g., phoneme)",
    )
    ap.add_argument(
        "--exts",
        nargs="+",
        default=sorted(SUPPORTED_EXTS),
        help="File extensions to include (defaults to common audio types)",
    )
    args = ap.parse_args()

    roots = [args.source_dir] if args.source_dir else args.roots
    files = list_audio_files(roots, args.exts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "text", "phoneme", "lang"]
    if args.type:
        fieldnames.append("type")
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in files:
            row = {
                "path": str(p),
                "text": "",
                "phoneme": "",
                "lang": args.lang or infer_lang(p),
            }
            if args.type:
                row["type"] = args.type
            writer.writerow(row)

    print(f"Wrote skeleton manifest with {len(files)} rows to {args.output}")


if __name__ == "__main__":
    main()
