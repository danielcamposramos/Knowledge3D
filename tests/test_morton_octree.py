"""
Test suite for Morton Octree spatial indexing.

Validates correctness against brute-force and performance targets (<50ms).

Author: Claude (K3D Core Team)
Date: 2025-10-04
"""

import pytest
import numpy as np
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from knowledge3d.spatial.morton_octree import MortonOctree


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
class TestMortonOctree:
    """Test Morton octree correctness and performance."""

    def test_build_octree(self):
        """Octree builds without errors."""
        N = 1000
        positions = np.random.rand(N, 3).astype(np.float32) * 100.0
        positions_gpu = cp.asarray(positions)

        octree = MortonOctree()
        octree.build_from_gpu_positions(positions_gpu)

        stats = octree.get_stats()
        assert stats["status"] == "built"
        assert stats["node_count"] == N
        assert stats["morton_min"] >= 0
        assert stats["morton_max"] <= (1 << 30)  # 30-bit morton codes

    def test_query_correctness_brute_force(self):
        """Morton query matches brute-force ground truth."""
        N = 5000
        positions = np.random.rand(N, 3).astype(np.float32) * 100.0
        positions_gpu = cp.asarray(positions)

        octree = MortonOctree()
        octree.build_from_gpu_positions(positions_gpu)

        # Test 100 random queries
        for _ in range(100):
            center = np.random.rand(3).astype(np.float32) * 100.0
            radius = np.random.uniform(2.0, 10.0)

            # GPU octree query
            octree_results = octree.query_radius_gpu(
                center, radius, refine_euclidean=True
            ).get()

            # Brute force (Python)
            dists = np.linalg.norm(positions - center, axis=1)
            brute_force_results = np.where(dists <= radius)[0]

            # Check correctness
            assert set(octree_results) == set(brute_force_results), (
                f"Query mismatch: octree={len(octree_results)}, "
                f"brute_force={len(brute_force_results)}"
            )

    def test_query_performance_10k_nodes(self):
        """Query completes <50ms on 10K nodes."""
        N = 10000
        positions = np.random.rand(N, 3).astype(np.float32) * 100.0
        positions_gpu = cp.asarray(positions)

        octree = MortonOctree()
        build_start = time.perf_counter()
        octree.build_from_gpu_positions(positions_gpu)
        build_time = (time.perf_counter() - build_start) * 1000

        print(f"Build time: {build_time:.2f}ms")

        # Warm-up
        center = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        _ = octree.query_radius_gpu(center, 10.0)

        # Measure query latency
        latencies = []
        for _ in range(100):
            center = np.random.rand(3).astype(np.float32) * 100.0
            radius = 10.0

            start = time.perf_counter()
            _ = octree.query_radius_gpu(center, radius, refine_euclidean=True)
            cp.cuda.Device().synchronize()
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"Query latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms")

        # Assert <50ms average
        assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms exceeds 50ms target"
        assert p95_latency < 100.0, f"P95 latency {p95_latency:.2f}ms exceeds 100ms"

    def test_morton_encoding_correctness(self):
        """Morton encoding preserves spatial locality."""
        octree = MortonOctree()

        # Test known cases
        assert octree._morton_encode_3d(0, 0, 0) == 0
        assert octree._morton_encode_3d(1, 0, 0) == 0b100  # X at bit 2
        assert octree._morton_encode_3d(0, 1, 0) == 0b010  # Y at bit 1
        assert octree._morton_encode_3d(0, 0, 1) == 0b001  # Z at bit 0
        assert octree._morton_encode_3d(1, 1, 1) == 0b111

        # Spatial locality: nearby points have nearby Morton codes
        positions = [
            (100, 100, 100),  # Center
            (101, 100, 100),  # X+1
            (100, 101, 100),  # Y+1
            (100, 100, 101),  # Z+1
            (200, 200, 200),  # Far away
        ]

        morton_codes = [octree._morton_encode_3d(x, y, z) for x, y, z in positions]

        # Nearby points have closer Morton codes than far points
        center_code = morton_codes[0]
        nearby_diff = max(abs(morton_codes[i] - center_code) for i in range(1, 4))
        far_diff = abs(morton_codes[4] - center_code)

        assert far_diff > nearby_diff, "Spatial locality violated"

    def test_empty_query_results(self):
        """Query with no results returns empty array."""
        N = 1000
        positions = np.random.rand(N, 3).astype(np.float32) * 100.0
        positions_gpu = cp.asarray(positions)

        octree = MortonOctree()
        octree.build_from_gpu_positions(positions_gpu)

        # Query far outside bounds
        center = np.array([1000.0, 1000.0, 1000.0], dtype=np.float32)
        radius = 1.0

        results = octree.query_radius_gpu(center, radius)
        assert len(results) == 0

    def test_large_radius_returns_all_nodes(self):
        """Very large radius returns all nodes."""
        N = 500
        positions = np.random.rand(N, 3).astype(np.float32) * 100.0
        positions_gpu = cp.asarray(positions)

        octree = MortonOctree()
        octree.build_from_gpu_positions(positions_gpu)

        # Query with huge radius
        center = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        radius = 1000.0

        results = octree.query_radius_gpu(center, radius)
        assert len(results) == N


@pytest.mark.benchmark
@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_octree_vs_bruteforce_speed():
    """Benchmark: Octree vs brute-force speedup."""
    N = 50000
    positions = np.random.rand(N, 3).astype(np.float32) * 1000.0
    positions_gpu = cp.asarray(positions)

    octree = MortonOctree()
    octree.build_from_gpu_positions(positions_gpu)

    center = np.array([500.0, 500.0, 500.0], dtype=np.float32)
    radius = 50.0

    # Octree query
    start = time.perf_counter()
    octree_results = octree.query_radius_gpu(center, radius)
    cp.cuda.Device().synchronize()
    octree_time = (time.perf_counter() - start) * 1000

    # Brute force (CPU)
    start = time.perf_counter()
    dists = np.linalg.norm(positions - center, axis=1)
    brute_results = np.where(dists <= radius)[0]
    brute_time = (time.perf_counter() - start) * 1000

    speedup = brute_time / octree_time

    print(f"Octree: {octree_time:.2f}ms")
    print(f"Brute force: {brute_time:.2f}ms")
    print(f"Speedup: {speedup:.1f}x")

    assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"
