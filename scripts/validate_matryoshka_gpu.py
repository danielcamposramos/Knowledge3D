#!/usr/bin/env python3
"""
Validate Matryoshka GPU projection against NumPy slicing/matvec.
"""

from __future__ import annotations

import argparse
import ctypes
import numpy as np

from knowledge3d.cranium.bridges.matryoshka_bridge import MatryoshkaProjectionBridge
from knowledge3d.cranium.sovereign import loader


def validate_once(bridge: MatryoshkaProjectionBridge, base_matrix: np.ndarray, target_dim: int) -> float:
    max_dim = base_matrix.shape[0]
    assert base_matrix.shape == (max_dim, max_dim)

    # Create input vector
    rng = np.random.default_rng(seed=target_dim)
    vector = rng.normal(size=target_dim).astype(np.float32)

    # Upload weights (if not already) and vector
    d_weights = loader.gpu_malloc(base_matrix.nbytes)
    loader.memcpy_htod(d_weights, base_matrix.ctypes.data_as(ctypes.c_void_p), base_matrix.nbytes)

    d_vector = loader.gpu_malloc(target_dim * 4)
    loader.memcpy_htod(d_vector, vector.ctypes.data_as(ctypes.c_void_p), vector.nbytes)

    d_output = loader.gpu_malloc(target_dim * 4)

    try:
        bridge.project_device(d_weights, d_vector, d_output, target_dim, max_dim)
        gpu_out = np.empty(target_dim, dtype=np.float32)
        loader.memcpy_dtoh(gpu_out.ctypes.data_as(ctypes.c_void_p), d_output, gpu_out.nbytes)
    finally:
        loader.gpu_free(d_weights)
        loader.gpu_free(d_vector)
        loader.gpu_free(d_output)

    # Reference using NumPy
    sub_matrix = base_matrix[:target_dim, :target_dim]
    ref = sub_matrix @ vector
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(ref, -10.0, 10.0, out=ref)

    return float(np.max(np.abs(gpu_out - ref)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Matryoshka GPU projection.")
    parser.add_argument("--max-dim", type=int, default=2048, help="Maximum Matryoshka capacity")
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256, 512, 1024], help="Target dimensions to validate")
    args = parser.parse_args()

    bridge = MatryoshkaProjectionBridge()

    rng = np.random.default_rng(seed=0)
    base_matrix = rng.normal(size=(args.max_dim, args.max_dim)).astype(np.float32)

    print("=" * 80)
    print("MATRYOSHKA GPU PROJECTION VALIDATION")
    print("=" * 80)
    worst = 0.0
    for dim in args.dims:
        diff = validate_once(bridge, base_matrix, dim)
        worst = max(worst, diff)
        print(f"  Dim {dim:4d}: max |GPU-CPU| = {diff:.6f}")

    print()
    print(f"Worst deviation: {worst:.6f}")
    if worst < 1e-4:
        print("✓ Matryoshka GPU projection matches NumPy within tolerance.")
    else:
        print("⚠️  Deviation exceeds tolerance!")


if __name__ == "__main__":
    main()
