#!/usr/bin/env python3
"""
Aggregate procedural compression stats for trained characters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from knowledge3d.cranium import AdaptiveDimensionCompressor


def _parse_key(path: Path) -> str | None:
    stem = path.stem
    if not stem.startswith("char_"):
        return None
    return stem


def _load_canonical_embedding(checkpoint_dir: Path, key: str) -> np.ndarray | None:
    embedding_path = checkpoint_dir / f"{key}_embeddings.npz"
    if not embedding_path.exists():
        return None
    data = np.load(embedding_path)
    embeddings = data["embeddings"]
    if embeddings.size == 0:
        return None
    return embeddings.mean(axis=0).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / denom)


def _write_summary(md_path: Path, rows: List[str], context: Dict[str, float | int]) -> None:
    lines = [
        "# Character Training Compression Summary",
        "",
        f"**Procedural Galaxy**: `{context['galaxy_root']}`",
        f"**Checkpoints scanned**: {context['processed']}",
        "",
        "## Aggregate Metrics",
        f"- Stored programs: {context['programs']}",
        f"- Average compression (vs. 2048D): {context['avg_compression']:.2f}:1",
        f"- Average fidelity: {context['avg_fidelity']:.6f}",
        f"- PD04 usage: {context['pd04_ratio'] * 100:.2f}%",
        "",
        "## Per-character Samples",
        "",
        "| Key | Compression | Fidelity | Codec | Bytes |",
        "|-----|-------------|----------|-------|-------|",
    ]
    lines.extend(rows)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise character procedural compression stats.")
    parser.add_argument("--galaxy-root", type=Path, default=Path("/K3D/Knowledge3D.local/procedural_galaxy"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars"))
    parser.add_argument("--output-md", type=Path, default=Path("validation_results/character_training_summary.md"))
    parser.add_argument("--output-json", type=Path, default=Path("validation_results/character_training_summary.json"))
    args = parser.parse_args()

    compressor = AdaptiveDimensionCompressor()

    rows: List[str] = []
    ratios: List[float] = []
    fidelities: List[float] = []
    codecs: Dict[str, int] = {"PD04": 0, "PD02": 0, "PD01": 0}
    programs = 0

    for program_path in sorted(args.galaxy_root.glob("char_*_*.ppr")):
        key = _parse_key(program_path)
        if key is None:
            continue
        canonical = _load_canonical_embedding(args.checkpoint_dir, key)
        if canonical is None:
            continue
        payload = program_path.read_bytes()
        reconstructed = compressor.decompress(payload)
        cosine = _cosine(canonical, reconstructed)
        compression = (2048 * 4) / max(1, len(payload))
        ratios.append(compression)
        fidelities.append(cosine)
        programs += 1
        magic = payload[:4]
        if magic == b"PD04":
            codecs["PD04"] += 1
        elif magic == b"PD02":
            codecs["PD02"] += 1
        elif magic == b"PD01":
            codecs["PD01"] += 1
        rows.append(f"| {key} | {compression:.2f}:1 | {cosine:.6f} | {magic.decode('ascii', 'ignore')} | {len(payload)} |")

    avg_compression = float(np.mean(ratios)) if ratios else 0.0
    avg_fidelity = float(np.mean(fidelities)) if fidelities else 0.0
    total_codecs = sum(codecs.values()) or 1
    pd04_ratio = codecs["PD04"] / total_codecs

    context = {
        "galaxy_root": str(args.galaxy_root),
        "processed": len(list(args.galaxy_root.glob("*.ppr"))),
        "programs": programs,
        "avg_compression": avg_compression,
        "avg_fidelity": avg_fidelity,
        "pd04_ratio": pd04_ratio,
    }

    _write_summary(args.output_md, rows[:50], context)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "galaxy_root": str(args.galaxy_root),
                "programs": programs,
                "average_compression": avg_compression,
                "average_fidelity": avg_fidelity,
                "codec_counts": codecs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Processed {programs} procedural programs.")
    print(f"Average compression: {avg_compression:.2f}:1 | fidelity: {avg_fidelity:.6f}")


if __name__ == "__main__":
    main()
