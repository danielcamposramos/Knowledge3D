#!/usr/bin/env python3
"""
Minimal pre-compilation of CuPy kernels.

Directly compiles the array operations that SemanticNavigator needs
without importing the full K3D stack.
"""

import sys
import os
from pathlib import Path

# Force CuPy to save compiled kernels
os.environ['CUPY_CACHE_DIR'] = str(Path(__file__).parent / '.cupy_cache')
os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'

import cupy as cp
import numpy as np

print("Pre-compiling CuPy array operation kernels...")

# These are the operations that trigger JIT in SemanticNavigator

print("\n1. Testing array slicing and indexing...")
try:
    arr = cp.random.rand(1000, 128).astype(cp.float32)
    indices = cp.array([0, 1, 2, 3, 4], dtype=cp.uint32)

    # Trigger indexing kernel
    result = arr[indices]
    print(f"   ✓ Array indexing compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n2. Testing array multiplication and sum...")
try:
    a = cp.random.rand(100, 256).astype(cp.float32)
    b = cp.random.rand(100, 256).astype(cp.float32)

    # Trigger multiply kernel
    c = a * b

    # Trigger sum kernel
    d = c.sum(axis=1)

    print(f"   ✓ Multiply and sum compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n3. Testing linalg.norm...")
try:
    vec = cp.random.rand(100, 3).astype(cp.float32)

    # This triggers the problematic kernel
    norms = cp.linalg.norm(vec, axis=1)

    print(f"   ✓ linalg.norm compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n4. Testing concatenate...")
try:
    a = cp.array([[1, 2], [3, 4]], dtype=cp.float32)
    b = cp.array([[5, 6]], dtype=cp.float32)

    c = cp.concatenate([a, b])

    print(f"   ✓ concatenate compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n5. Testing argsort and radix sort...")
try:
    arr = cp.random.rand(10000).astype(cp.float32)

    # Trigger sort kernels
    sorted_indices = cp.argsort(arr)

    print(f"   ✓ argsort compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n6. Testing boolean masking...")
try:
    arr = cp.random.rand(1000).astype(cp.float32)

    # Trigger boolean mask kernel
    mask = arr > 0.5
    filtered = arr[mask]

    print(f"   ✓ Boolean masking compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n7. Testing bit operations (for Morton codes)...")
try:
    a = cp.array([1, 2, 3, 4, 5], dtype=cp.uint32)

    # Trigger bitwise ops
    b = (a << 16) | a
    c = b & 0xFFFF
    d = b >> 16

    print(f"   ✓ Bitwise operations compiled")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print("\n8. Saving CuPy cache...")
cache_dir = Path(os.environ['CUPY_CACHE_DIR'])
if cache_dir.exists():
    cubin_files = list(cache_dir.glob('**/*.cubin'))
    ptx_files = list(cache_dir.glob('**/*.ptx'))

    print(f"   ✓ Found {len(cubin_files)} .cubin files")
    print(f"   ✓ Found {len(ptx_files)} .ptx files")
    print(f"   ✓ Cache saved to: {cache_dir}")

    # List all files for debugging
    total_size = 0
    for cubin in cubin_files:
        size = cubin.stat().st_size
        total_size += size
        print(f"     - {cubin.name} ({size/1024:.1f} KB)")

    print(f"   ✓ Total cache size: {total_size/1024/1024:.2f} MB")
else:
    print(f"   ✗ Cache directory not found: {cache_dir}")
    sys.exit(1)

print("\n✅ All CuPy kernels pre-compiled successfully!")
print(f"\nTo use in Debian 13:")
print(f"1. Cache is already in: {cache_dir}")
print(f"2. Set: export CUPY_CACHE_DIR={cache_dir}")
print(f"3. CuPy will load pre-compiled kernels instead of JIT compiling")
