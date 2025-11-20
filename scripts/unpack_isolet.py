#!/usr/bin/env python3
"""
Unpack ISOLET dataset into per-letter WAVs and a manifest.

ISOLET provides feature vectors, not audio PCM. This script reads the
ASCII data files, treats each feature vector as already-extracted MFCC-like
features, and cannot reconstruct PCM audio. As a placeholder, it emits a
manifest with letter labels and references to the data rows, which downstream
could map to synthetic audio generation (e.g., via espeak phoneme names).

If you have access to true ISOLET audio, replace this pipeline accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

ISOLET_PATH = Path("/K3D/K3D_llama_cpp/datasets/audio/phoneme_external")


def parse_isolet(data_files):
    rows = []
    for df in data_files:
        with df.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 1:
                    continue
                label = parts[-1].strip()
                # label is 1-26 for A-Z; map to letter
                try:
                    idx = int(float(label))
                    letter = chr(ord("A") + idx - 1)
                except Exception:
                    letter = "?"
                rows.append({"path": str(df), "text": letter, "phoneme": letter.lower(), "lang": "en"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=ISOLET_PATH / "isolet_manifest.csv",
        help="Where to write the manifest",
    )
    args = ap.parse_args()

    data_files = [ISOLET_PATH / "isolet1+2+3+4.data", ISOLET_PATH / "isolet5.data"]
    rows = parse_isolet(data_files)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "text", "phoneme", "lang"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote ISOLET manifest with {len(rows)} entries to {args.output}")


if __name__ == "__main__":
    main()
