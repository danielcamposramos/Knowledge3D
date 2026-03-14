"""Sovereign LED pathfinder smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.spatial.led_pathfinder import LEDPathfinder


@pytest.fixture
def pathfinder():
    try:
        instance = LEDPathfinder()
    except RuntimeError as exc:
        pytest.skip(f"Sovereign loader unavailable: {exc}")
    yield instance


class TestLEDPathfinderSovereign:
    def test_straight_path(self, pathfinder: LEDPathfinder):
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        goal = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        path = pathfinder.find_path(start, goal, np.empty((0, 3), dtype=np.float32))

        assert path.shape[0] >= 2
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[-1], goal)

    def test_obstacle_detour(self, pathfinder: LEDPathfinder):
        start = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        goal = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        obstacles = np.array([
            [5.0, 0.0, 0.0],
            [5.0, 0.5, 0.0],
            [5.0, -0.5, 0.0],
        ], dtype=np.float32)

        path = pathfinder.find_path(start, goal, obstacles)
        assert path.shape[0] >= 3
        assert not np.allclose(path[1], (start + goal) * 0.5)

    def test_distance_kernel(self, pathfinder: LEDPathfinder):
        points = np.random.rand(16, 3).astype(np.float32)
        reference = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        distances = pathfinder.compute_distances(points, reference)

        assert distances.shape == (16,)
        np.testing.assert_allclose(
            distances,
            np.linalg.norm(points - reference, axis=1),
            atol=1e-4,
        )

    def test_priority_queue(self, pathfinder: LEDPathfinder):
        costs = np.array([4.0, 1.0, 3.0, 2.0], dtype=np.float32)
        nodes = np.array([10, 11, 12, 13], dtype=np.int32)
        min_node, idx = pathfinder.rpn_priority_queue_pop(costs, nodes)

        assert min_node == 11
        assert idx == 1

    def test_csr_navigation(self, pathfinder: LEDPathfinder):
        row_offsets = np.array([0, 2, 3, 4], dtype=np.uint32)
        col_indices = np.array([1, 2, 2, 1], dtype=np.uint32)
        packed_costs = np.array(
            [
                (10 << 16) | 1,
                (100 << 16) | 8,
                (5 << 16) | 1,
                (5 << 16) | 1,
            ],
            dtype=np.uint32,
        )

        path = pathfinder.navigate_csr(
            row_offsets,
            col_indices,
            packed_costs,
            start=0,
            goal=2,
        )

        assert path.tolist() == [0, 1, 2]
