#!/usr/bin/env python3
"""Test NVRTC with explicit include paths for GCC 11."""

def main() -> int:
    import os
    from pathlib import Path

    # Force CuPy to use GCC 11 system headers
    # NVRTC uses -I flags, so we need to prepend GCC 11 paths
    gcc11_paths = [
        "/usr/lib/gcc/x86_64-linux-gnu/11/include",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
    ]

    # CuPy uses CUPY_CUDA_COMPILE_OPTS for NVRTC flags
    include_flags = " ".join([f"-I{p}" for p in gcc11_paths if Path(p).exists()])
    os.environ["CUPY_CUDA_COMPILE_OPTS"] = include_flags

    print(f"CUPY_CUDA_COMPILE_OPTS: {include_flags}\n")

    import cupy as cp

    print(f"CuPy version: {cp.__version__}")

    # Test the failing operation
    print("\nTesting cp.linalg.norm with explicit GCC 11 includes...")
    try:
        vec = cp.random.rand(10, 3).astype(cp.float32)
        norms = cp.linalg.norm(vec, axis=1)
        print("✓ SUCCESS! linalg.norm worked with GCC 11 headers")
        print(f"  Result shape: {norms.shape}")
        print(f"  First 3 norms: {norms[:3].get()}")
        return 0
    except Exception as e:
        error_msg = str(e)
        if "hypotf" in error_msg:
            print("✗ STILL FAILED: GCC headers not picked up by NVRTC")
        else:
            print(f"✗ FAILED: {error_msg[:150]}...")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
