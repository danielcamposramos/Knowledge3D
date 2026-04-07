#!/usr/bin/env python3
"""
Pre-compile all CuPy kernels that SemanticNavigator needs.

This script runs in Docker (Ubuntu 22.04 + GCC 11) and generates
.cubin files that can be loaded in Debian 13 without NVRTC.

Usage:
    docker run --gpus all -v $(pwd):/workspace k3d-compile \
        python precompile_cupy_kernels.py
"""

import sys
import os
from pathlib import Path

# Force CuPy to save compiled kernels
os.environ['CUPY_CACHE_DIR'] = str(Path(__file__).parent / '.cupy_cache')
os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'

import cupy as cp
import numpy as np

print("Pre-compiling CuPy kernels for SemanticNavigator...")

# Import all the modules that trigger CuPy compilation
from knowledge3d.spatial.morton_octree import MortonOctree
from knowledge3d.spatial.led_pathfinder import DependencyKernel, LEDPathfinder

print("\n1. Testing Morton Octree compilation...")
try:
    # Create test data
    positions = np.random.rand(100, 3).astype(np.float32) * 100.0
    positions_gpu = cp.asarray(positions)

    # This will trigger Morton octree PTX load (already pre-compiled)
    octree = MortonOctree()
    octree.build_from_gpu_positions(positions_gpu)

    # Test query (triggers CuPy array operations)
    center = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    results = octree.query_radius_gpu(center, 10.0, refine_euclidean=True)

    print(f"   ✓ Morton octree compiled successfully")
    print(f"   ✓ Found {len(results)} neighbors")
except Exception as e:
    print(f"   ✗ Morton octree failed: {e}")
    sys.exit(1)

print("\n2. Testing LED-A* Pathfinder compilation...")
try:
    # Create test graph
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [0, 2], [1, 3], [2, 4]
    ], dtype=np.uint32)

    embeddings = np.random.randn(5, 128).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    positions = np.random.randn(5, 3).astype(np.float32)

    # Build kernel (triggers CuPy array ops)
    kernel = DependencyKernel(num_vertices=5)
    kernel.build_from_edges(edges, embeddings, positions, similarity_threshold=0.5)

    print(f"   ✓ LED-A* kernel compiled successfully")
    print(f"   ✓ Kernel size: {kernel.num_edges} edges")
except Exception as e:
    print(f"   ✗ LED-A* failed: {e}")
    sys.exit(1)

print("\n3. Testing LED-A* pathfinding...")
try:
    pathfinder = LEDPathfinder()
    pathfinder.build_kernel_from_octree(edges, embeddings, positions, threshold=0.5)

    path, cost = pathfinder.find_path(0, 4, alpha=0.5, beta=0.5)

    print(f"   ✓ LED-A* pathfinding compiled successfully")
    print(f"   ✓ Found path: {path} (cost: {cost:.3f})")
except Exception as e:
    print(f"   ✗ LED-A* pathfinding failed: {e}")
    sys.exit(1)

print("\n4. Saving CuPy cache...")
cache_dir = Path(os.environ['CUPY_CACHE_DIR'])
if cache_dir.exists():
    cubin_files = list(cache_dir.glob('**/*.cubin'))
    ptx_files = list(cache_dir.glob('**/*.ptx'))

    print(f"   ✓ Found {len(cubin_files)} .cubin files")
    print(f"   ✓ Found {len(ptx_files)} .ptx files")
    print(f"   ✓ Cache saved to: {cache_dir}")

    # List all files for debugging
    for cubin in cubin_files:
        size = cubin.stat().st_size / 1024
        print(f"     - {cubin.name} ({size:.1f} KB)")
else:
    print(f"   ✗ Cache directory not found: {cache_dir}")
    sys.exit(1)

print("\n✅ All kernels pre-compiled successfully!")
print(f"\nTo use in Debian 13:")
print(f"1. Copy .cupy_cache/ to your Debian 13 environment")
print(f"2. Set CUPY_CACHE_DIR={cache_dir}")
print(f"3. CuPy will load pre-compiled kernels instead of JIT compiling")
