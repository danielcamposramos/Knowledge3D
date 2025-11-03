#!/usr/bin/env python3
"""
Quick validation for the sovereign spatial mean pooling kernel.

Generates random feature maps, compares the GPU kernel output against NumPy's
mean reduction, and reports the maximum absolute deviation.
"""

from __future__ import annotations

import argparse
import ctypes
import numpy as np

from knowledge3d.cranium.bridges.spatial_pool_bridge import SpatialMeanPooler
from knowledge3d.cranium.sovereign import loader


def validate_once(pooler: SpatialMeanPooler, H: int, W: int, C: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(H, W, C)).astype(np.float32)

    d_features = loader.gpu_malloc(features.nbytes)
    loader.memcpy_htod(d_features, features.ctypes.data_as(ctypes.c_void_p), features.nbytes)

    gpu_mean = pooler.mean_pool_host(d_features, H, W, C)
    loader.gpu_free(d_features)

    cpu_mean = features.mean(axis=(0, 1)).astype(np.float32)
    max_diff = float(np.max(np.abs(gpu_mean - cpu_mean)))

    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate spatial mean pooling kernel against NumPy.")
    parser.add_argument("--trials", type=int, default=3, help="Number of random trials to run")
    parser.add_argument("--height", type=int, default=8, help="Feature map height")
    parser.add_argument("--width", type=int, default=8, help="Feature map width")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    args = parser.parse_args()

    print("Spatial mean pooling validation")
    print(f"  Trials:    {args.trials}")
    print(f"  Shape:     {args.height}x{args.width} with {args.channels} channels")

    worst = 0.0
    pooler = SpatialMeanPooler()
    for trial in range(args.trials):
        diff = validate_once(pooler, args.height, args.width, args.channels, seed=trial)
        worst = max(worst, diff)
        print(f"  Trial {trial + 1}: max |GPU-CPU| = {diff:.6f}")

    print(f"\nWorst-case deviation: {worst:.6f}")
    if worst < 1e-5:
        print("✓ Spatial pooling kernel matches NumPy mean within tolerance.")
    else:
        print("⚠️  Deviation exceeds expected tolerance.")


if __name__ == "__main__":
    main()
