#!/usr/bin/env python3
"""
Galaxy Character Embedding Validation

Validates the consolidated embeddings produced by atomic character training.
Reports coverage, distribution, and basic statistics to ensure the Galaxy
memory is ready for downstream composition tasks.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


def main() -> None:
    checkpoint_dir = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/atomic_chars")
    galaxy_path = checkpoint_dir / "galaxy_character_embeddings.npz"

    if not galaxy_path.exists():
        print(f"Galaxy embeddings not found: {galaxy_path}")
        raise SystemExit(1)

    data = np.load(galaxy_path)
    embeddings = data["embeddings"]
    char_ids = data["char_ids"]
    low_embeddings = data["embeddings_low"] if "embeddings_low" in data.files else None
    dim_high = int(data.get("embed_dim_high", embeddings.shape[1]))
    dim_low = int(
        data.get("embed_dim_low", low_embeddings.shape[1] if low_embeddings is not None else 0)
    )

    print("=" * 80)
    print("GALAXY CHARACTER EMBEDDINGS VALIDATION")
    print("=" * 80)
    print()

    print(f"Total embeddings: {len(embeddings)}")
    if embeddings.ndim != 2:
        print(f"⚠️  Embeddings have unexpected shape: {embeddings.shape}")
    else:
        print(f"Embedding dimension (high): {dim_high}")
        if low_embeddings is not None:
            print(f"Embedding dimension (low): {dim_low}")
    print(f"Unique characters: {len(np.unique(char_ids))}")
    print()

    print("Character coverage:")
    for char_id in sorted(np.unique(char_ids)):
        count = int((char_ids == char_id).sum())
        print(f"  {char_id:3d} ('{chr(char_id)}'): {count:3d} embeddings")
    print()

    print("Embedding statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")

    zero_embeddings = int((np.linalg.norm(embeddings, axis=1) < 1e-6).sum())
    if zero_embeddings > 0:
        print(f"  ⚠️  Zero embeddings: {zero_embeddings}/{len(embeddings)}")
    else:
        print("  ✓ No zero embeddings detected")

    if low_embeddings is not None:
        low_zero = int((np.linalg.norm(low_embeddings, axis=1) < 1e-6).sum())
        if low_zero > 0:
            print(f"  ⚠️  Zero low-d embeddings: {low_zero}/{len(low_embeddings)}")
        else:
            print("  ✓ No zero low-d embeddings detected")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
