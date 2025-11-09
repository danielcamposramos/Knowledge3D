#!/usr/bin/env python3
"""
Phase H adaptive compression pipeline demo.

Flows:
  RPN embedding → Matryoshka projection → Adaptive compressor → Procedural Galaxy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from knowledge3d.cranium import (
    AdaptiveDimensionCompressor,
    PhaseHProceduralIntegration,
    MatryoshkaTRM,
    ProceduralGalaxy,
)


def _load_base_vector(path: Path | None, dims: int) -> np.ndarray:
    if path is None:
        return np.random.randn(dims).astype(np.float32)
    array = np.load(path)
    if array.ndim != 1 or array.size < dims:
        raise ValueError(f"Embedding file must contain at least {dims} elements.")
    return array.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase H → Adaptive compression pipeline demo.")
    parser.add_argument("--cache-dir", type=Path, default=Path("validation_cache"), help="Directory containing dictionaries/prototype tables.")
    parser.add_argument("--galaxy-root", type=Path, default=Path("validation_cache/demo_galaxy"), help="Procedural Galaxy output directory.")
    parser.add_argument("--embedding-file", type=Path, default=None, help="Optional .npy file storing a 2048D vector.")
    parser.add_argument("--quality", choices=["ultrafast", "fast", "balanced", "maximum"], default="fast")
    parser.add_argument("--store-key", default="phase_h_demo", help="Key for Procedural Galaxy storage.")
    args = parser.parse_args()

    compressor = AdaptiveDimensionCompressor(cache_dir=args.cache_dir)
    galaxy = ProceduralGalaxy(root=args.galaxy_root)
    matryoshka = MatryoshkaTRM(max_dims=2048, min_dims=64)
    integration = PhaseHProceduralIntegration(
        compressor=compressor,
        matryoshka_model=matryoshka,
        procedural_galaxy=galaxy,
    )

    base_vector = _load_base_vector(args.embedding_file, dims=2048)
    program, metadata = integration.compress_matryoshka_vector(
        base_vector,
        quality=args.quality,
        store_key=args.store_key,
    )

    print("Phase H Adaptive Compression Demo")
    print("=" * 60)
    print(f"Quality level      : {args.quality}")
    print(f"Target dimension   : {metadata['target_dim']}D")
    print(f"Compressed bytes   : {len(program)}")
    print(f"Actual compression : {metadata['actual_compression']:.1f}:1")
    print(f"Fidelity (cosine)  : {metadata['actual_fidelity']:.6f}")
    print(f"Galaxy store key   : {args.store_key}")
    print(f"Galaxy root        : {args.galaxy_root}")


if __name__ == "__main__":
    main()
