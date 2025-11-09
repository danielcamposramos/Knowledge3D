#!/usr/bin/env python3
"""
Demonstration of adaptive procedural compression.
"""

from __future__ import annotations

import numpy as np

from knowledge3d.cranium import AdaptiveDimensionCompressor


def main() -> None:
    try:
        compressor = AdaptiveDimensionCompressor()
    except FileNotFoundError as exc:
        print("Adaptive compressor not initialised:", exc)
        print("Run scripts/train_dictionary.py to generate dictionaries.")
        return

    embedding = np.random.randn(2048).astype(np.float32)
    print("Adaptive Procedural Compression Demo")
    print("=" * 60)
    print(f"Original embedding size: {embedding.nbytes} bytes\n")

    for quality in ["ultrafast", "fast", "balanced", "maximum"]:
        program, metadata = compressor.compress(embedding, quality=quality, return_metadata=True)
        stats = compressor.get_compression_stats(quality)
        print(f"{quality.upper():>10}:")
        print(f"  Target dimension   : {metadata['target_dim']}D")
        print(f"  Compressed size    : {len(program)} bytes")
        print(f"  Actual compression : {metadata['actual_compression']:.1f}:1")
        print(f"  Expected fidelity  : {stats['expected_fidelity']:.5f}")
        reconstructed = compressor.decompress(program, metadata['target_dim'])
        print(f"  Reconstructed shape: {reconstructed.shape}\n")


if __name__ == "__main__":
    main()
