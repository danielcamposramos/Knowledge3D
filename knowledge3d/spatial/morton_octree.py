"""
Morton Octree - GPU-Native Spatial Indexing for K3D House Memory

Replaces CPU-bound k-NN with GPU-native Morton code indexing for <50ms queries.

Design:
- Positions → Morton codes (Z-order curve, bit interleaving)
- GPU radix sort (CuPy/Thrust)
- Binary search on sorted Morton codes
- Optional Euclidean refinement

Author: Claude (K3D Core Team)
Date: 2025-10-04
License: Apache-2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from knowledge3d.cranium.ptx.ptx_loader import load_cu_kernel

_logger = logging.getLogger(__name__)


class MortonOctree:
    """
    GPU-accelerated spatial index using Morton codes (Z-order curve).

    Workflow:
    1. Build octree once from GPU positions (compute Morton codes, sort)
    2. Query via binary search on sorted Morton codes
    3. Optional Euclidean refinement for exact radius

    Performance target: <50ms for 10K node queries.
    """

    def __init__(self, kernel_path: Optional[Path] = None):
        """
        Initialize octree with PTX/CUDA kernel.

        Args:
            kernel_path: Path to morton_octree.cu (auto-detected if None)
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("MortonOctree requires CuPy. Install: pip install cupy-cuda12x")

        # Auto-detect kernel path
        if kernel_path is None:
            kernel_path = Path(__file__).parent.parent / "cranium/ptx/morton_octree.cu"

        if not kernel_path.exists():
            raise FileNotFoundError(f"Morton octree kernel not found: {kernel_path}")

        # Load CUDA module
        self.module = load_cu_kernel(str(kernel_path))

        # Extract kernels
        self.compute_morton_kernel = self.module.get_function("compute_morton_codes")
        self.query_kernel = self.module.get_function("octree_query_morton")
        self.refine_kernel = self.module.get_function("refine_query_euclidean")

        # State (populated during build)
        self.morton_codes: Optional[cp.ndarray] = None
        self.node_ids: Optional[cp.ndarray] = None
        self.positions_gpu: Optional[cp.ndarray] = None
        self.node_count: int = 0
        self.bbox_min: Optional[cp.ndarray] = None
        self.bbox_size: float = 0.0

        _logger.info(f"MortonOctree initialized with kernel: {kernel_path}")

    def build_from_gpu_positions(self, positions_gpu: cp.ndarray) -> MortonOctree:
        """
        Build octree from GPU-resident positions.

        Args:
            positions_gpu: CuPy array of shape (N, 3) with node positions

        Returns:
            Self (for chaining)
        """
        if positions_gpu.ndim != 2 or positions_gpu.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) positions, got {positions_gpu.shape}")

        self.node_count = len(positions_gpu)
        self.positions_gpu = positions_gpu

        # Compute bounding box
        self.bbox_min = positions_gpu.min(axis=0)
        bbox_max = positions_gpu.max(axis=0)
        self.bbox_size = float((bbox_max - self.bbox_min).max())

        _logger.info(f"Building octree for {self.node_count} nodes, bbox_size={self.bbox_size:.2f}")

        # Allocate Morton code buffer
        self.morton_codes = cp.zeros(self.node_count, dtype=cp.uint32)
        self.node_ids = cp.arange(self.node_count, dtype=cp.uint32)

        # Launch Morton code computation kernel
        threads_per_block = 256
        blocks = (self.node_count + threads_per_block - 1) // threads_per_block

        self.compute_morton_kernel(
            (blocks,), (threads_per_block,),
            (
                positions_gpu,
                cp.uint32(self.node_count),
                self.morton_codes,
                cp.float32(self.bbox_min[0]),
                cp.float32(self.bbox_min[1]),
                cp.float32(self.bbox_min[2]),
                cp.float32(self.bbox_size)
            )
        )

        cp.cuda.Device().synchronize()

        # Sort by Morton code (CuPy uses Thrust radix sort internally)
        sorted_indices = cp.argsort(self.morton_codes)
        self.morton_codes = self.morton_codes[sorted_indices]
        self.node_ids = self.node_ids[sorted_indices]

        _logger.info(f"Octree built successfully. Morton codes range: {self.morton_codes.min()}-{self.morton_codes.max()}")

        return self

    def query_radius_gpu(
        self,
        center: np.ndarray | cp.ndarray,
        radius: float,
        refine_euclidean: bool = True,
        max_results: int = 10000
    ) -> cp.ndarray:
        """
        Query nodes within radius of center (GPU operation).

        Args:
            center: Query center (x, y, z) as numpy or cupy array
            radius: Query radius (Euclidean distance)
            refine_euclidean: If True, post-filter by exact Euclidean distance
            max_results: Maximum results to return (buffer size)

        Returns:
            CuPy array of node IDs within radius
        """
        if self.morton_codes is None:
            raise RuntimeError("Octree not built. Call build_from_gpu_positions() first.")

        # Convert center to CuPy if needed
        if isinstance(center, np.ndarray):
            center = cp.asarray(center)

        # Normalize center to [0, 1]
        center_normalized = (center - self.bbox_min) / self.bbox_size
        center_normalized = cp.clip(center_normalized, 0.0, 1.0)

        # Quantize to 10-bit integers
        center_quantized = (center_normalized * 1023).astype(cp.uint32)

        # Compute query Morton code (Python-side for simplicity; could be PTX)
        query_morton = self._morton_encode_3d(
            int(center_quantized[0]),
            int(center_quantized[1]),
            int(center_quantized[2])
        )

        # Compute Morton radius (conservative approximation)
        # Morton space != Euclidean space, so we over-estimate to avoid false negatives
        radius_normalized = radius / self.bbox_size
        radius_morton = int(radius_normalized * 1023 * 1.732)  # sqrt(3) factor for diagonal

        # Allocate result buffers
        result_buffer = cp.zeros(max_results, dtype=cp.uint32)
        result_count = cp.zeros(1, dtype=cp.uint32)

        # Launch query kernel (single-threaded binary search)
        self.query_kernel(
            (1,), (1,),
            (
                self.morton_codes,
                self.node_ids,
                cp.uint32(self.node_count),
                cp.uint32(query_morton),
                cp.uint32(radius_morton),
                result_buffer,
                result_count,
                cp.uint32(max_results)
            )
        )

        cp.cuda.Device().synchronize()

        # Get candidates
        count = int(result_count[0])
        candidates = result_buffer[:count]

        _logger.debug(f"Morton query returned {count} candidates for radius={radius:.2f}")

        # Optional Euclidean refinement
        if refine_euclidean and count > 0:
            refined_buffer = cp.zeros(count, dtype=cp.uint32)
            refined_count = cp.zeros(1, dtype=cp.uint32)

            threads = 256
            blocks = (count + threads - 1) // threads

            self.refine_kernel(
                (blocks,), (threads,),
                (
                    self.positions_gpu,
                    candidates,
                    cp.uint32(count),
                    cp.float32(center[0]),
                    cp.float32(center[1]),
                    cp.float32(center[2]),
                    cp.float32(radius),
                    refined_buffer,
                    refined_count,
                    cp.uint32(count)
                )
            )

            cp.cuda.Device().synchronize()

            refined_count_val = int(refined_count[0])
            result = refined_buffer[:refined_count_val]

            _logger.debug(f"Euclidean refinement reduced {count} → {refined_count_val} results")
        else:
            result = candidates

        return result

    def _morton_encode_3d(self, x: int, y: int, z: int) -> int:
        """
        Python implementation of Morton encoding (for query center).

        Interleaves 10-bit x, y, z into 30-bit Morton code.

        Args:
            x, y, z: 10-bit integers (0-1023)

        Returns:
            30-bit Morton code
        """
        def part1by2(n: int) -> int:
            """Spread bits with 2 zeros between each."""
            n &= 0x000003ff  # Keep only 10 bits
            n = (n ^ (n << 16)) & 0xff0000ff
            n = (n ^ (n <<  8)) & 0x0300f00f
            n = (n ^ (n <<  4)) & 0x030c30c3
            n = (n ^ (n <<  2)) & 0x09249249
            return n

        mx = part1by2(x)
        my = part1by2(y)
        mz = part1by2(z)

        # Interleave: X at bit 2, Y at bit 1, Z at bit 0
        return (mx << 2) | (my << 1) | mz

    def get_stats(self) -> dict:
        """Return octree statistics for monitoring."""
        if self.morton_codes is None:
            return {"status": "not_built"}

        return {
            "status": "built",
            "node_count": self.node_count,
            "bbox_size": self.bbox_size,
            "morton_min": int(self.morton_codes.min()),
            "morton_max": int(self.morton_codes.max()),
            "memory_mb": (
                self.morton_codes.nbytes +
                self.node_ids.nbytes +
                self.positions_gpu.nbytes
            ) / (1024 * 1024)
        }


# Example usage
if __name__ == "__main__":
    # Demo: build octree from random positions
    import time

    N = 10000
    positions = np.random.rand(N, 3).astype(np.float32) * 100.0
    positions_gpu = cp.asarray(positions)

    # Build octree
    octree = MortonOctree()
    start = time.perf_counter()
    octree.build_from_gpu_positions(positions_gpu)
    build_time = (time.perf_counter() - start) * 1000

    print(f"Octree built in {build_time:.2f}ms")
    print(f"Stats: {octree.get_stats()}")

    # Query
    center = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    radius = 10.0

    start = time.perf_counter()
    results = octree.query_radius_gpu(center, radius, refine_euclidean=True)
    query_time = (time.perf_counter() - start) * 1000

    print(f"Query returned {len(results)} nodes in {query_time:.2f}ms")

    # Verify correctness (brute force)
    dists = np.linalg.norm(positions - center, axis=1)
    expected = np.where(dists <= radius)[0]

    print(f"Expected {len(expected)} nodes (brute force)")
    print(f"Match: {set(results.get()) == set(expected)}")
