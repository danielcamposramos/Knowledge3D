#!/usr/bin/env python3
"""Test if CuPy can use nvcc instead of NVRTC."""

def main() -> int:
    import os
    from pathlib import Path

    # Set environment to use nvcc
    os.environ["CUPY_CACHE_DIR"] = str(Path(__file__).parent / ".cupy_cache")
    os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"

    print("Testing CuPy with different backends...")

    import cupy as cp

    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")

    # Try the failing linalg.norm operation
    print("\nTesting cp.linalg.norm (the operation that fails)...")
    try:
        vec = cp.random.rand(10, 3).astype(cp.float32)
        norms = cp.linalg.norm(vec, axis=1)
        print(f"✓ SUCCESS: linalg.norm worked! Result shape: {norms.shape}")
        return 0
    except Exception as e:
        error_msg = str(e)
        if "hypotf" in error_msg or "atan2" in error_msg:
            print("✗ FAILED: GCC 15 incompatibility")
            print(f"   Error: {error_msg[:200]}...")
        else:
            print(f"✗ FAILED: {error_msg[:200]}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
