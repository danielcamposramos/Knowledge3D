"""Sovereign Morton octree tests."""
from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.spatial.morton_octree import MortonOctreeSovereign


@pytest.fixture
def octree():
    try:
        instance = MortonOctreeSovereign()
    except RuntimeError as exc:
        pytest.skip(f"Sovereign loader unavailable: {exc}")
    yield instance


class TestMortonOctreeEncoding:
    def test_encode_shape_and_range(self, octree: MortonOctreeSovereign):
        points = np.random.rand(128, 3).astype(np.float32) * 50.0
        codes = octree.encode(points)

        assert codes.shape == (128,)
        assert all(isinstance(int(code), int) for code in codes)
        assert codes.min(initial=0) >= 0
        assert codes.max(initial=0) < (1 << 30)

    def test_encode_consistency(self, octree: MortonOctreeSovereign):
        points = np.array([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            [0.9, 0.1, 0.2],
        ], dtype=np.float32)

        codes = octree.encode(points)
        assert codes[0] == codes[1]
        assert codes[2] != codes[0]


class TestMortonOctreeSorting:
    def test_sort_matches_numpy(self, octree: MortonOctreeSovereign):
        codes = np.random.randint(0, 1 << 20, size=64, dtype=np.uint32)
        sorted_codes = octree.sort(codes)
        assert np.array_equal(sorted_codes, np.sort(codes))

    def test_sort_returns_indices(self, octree: MortonOctreeSovereign):
        codes = np.array([10, 3, 7, 1], dtype=np.uint32)
        sorted_codes, order = octree.sort(codes, return_indices=True)

        assert np.array_equal(sorted_codes, np.array([1, 3, 7, 10], dtype=np.uint32))
        assert np.array_equal(order, np.array([3, 1, 2, 0], dtype=np.uint32))


class TestMortonOctreeBuild:
    def test_build_tree_populates_stats(self, octree: MortonOctreeSovereign):
        points = np.random.rand(32, 3).astype(np.float32)
        result = octree.build_tree(points)

        stats = result["stats"]
        assert stats["status"] == "built"
        assert stats["node_count"] == 32
        assert stats["morton_min"] <= stats["morton_max"]
        assert result["codes"].shape == (32,)
        assert result["indices"].shape == (32,)
