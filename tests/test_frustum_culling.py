"""
Sovereign frustum culling tests – validates the PTX wrapper against a CPU
reference implementation and basic performance expectations.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from knowledge3d.spatial.frustum import (
    FrustumCuller,
    create_perspective_matrix,
    create_view_matrix,
    matmul_4x4,
    matvec_4,
)


# ---------------------------------------------------------------------------
# CPU reference to validate GPU behaviour
# ---------------------------------------------------------------------------
def cpu_frustum_reference(
    positions: np.ndarray,
    view_proj,
    view,
    margin_xy: float = 0.11,
    margin_z: float = 1.0,
) -> np.ndarray:
    n = len(positions)
    visible = np.ones(n, dtype=bool)

    for i in range(n):
        vec = [float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2]), 1.0]

        # View-space depth check (camera looks down -Z)
        vz = sum(float(a) * float(b) for a, b in zip(view[2], vec))
        if vz >= 0.0:
            visible[i] = False
            continue

        clip = matvec_4(view_proj, vec)
        if clip[3] <= 0.0:
            visible[i] = False
            continue

        ndc = [float(component) / float(clip[3]) for component in clip[:3]]
        if ndc[0] < -margin_xy or ndc[0] > margin_xy:
            visible[i] = False
            continue
        if ndc[1] < -margin_xy or ndc[1] > margin_xy:
            visible[i] = False
            continue
        if ndc[2] < -margin_z or ndc[2] > margin_z:
            visible[i] = False

    return visible


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def frustum_culler():
    try:
        culler = FrustumCuller(enable_profiling=True)
    except RuntimeError as exc:  # pragma: no cover - GPU unavailable
        pytest.skip(f"Sovereign loader unavailable: {exc}")
    yield culler
    culler.close()


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
class TestFrustumCullingCorrectness:
    def test_single_warp_correctness(self, frustum_culler: FrustumCuller):
        np.random.seed(42)
        positions = np.random.uniform(-10, 10, (32, 3)).astype(np.float32)

        view = create_view_matrix(
            eye=np.array([0, 0, 20], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32),
        )
        proj = create_perspective_matrix(60.0, 1.0, 0.1, 100.0)
        view_proj = matmul_4x4(proj, view)

        gpu_visible = frustum_culler.cull_nodes(positions, view_proj=view_proj, view=view)
        cpu_visible = cpu_frustum_reference(positions, view_proj, view)

        expected = np.where(cpu_visible)[0]
        np.testing.assert_array_equal(
            np.sort(gpu_visible),
            expected,
        )

    def test_multiple_warps_correctness(self, frustum_culler: FrustumCuller):
        np.random.seed(123)
        positions = np.random.uniform(-20, 20, (128, 3)).astype(np.float32)

        view = create_view_matrix(
            eye=np.array([10, 10, 30], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32),
        )
        proj = create_perspective_matrix(60.0, 16 / 9, 0.1, 200.0)
        view_proj = matmul_4x4(proj, view)

        gpu_visible = frustum_culler.cull_nodes(positions, view_proj=view_proj, view=view)
        cpu_visible = cpu_frustum_reference(positions, view_proj, view)
        expected = np.where(cpu_visible)[0]

        assert len(gpu_visible) == len(expected)
        np.testing.assert_array_equal(np.sort(gpu_visible), expected)

    def test_edge_cases(self, frustum_culler: FrustumCuller):
        view = create_view_matrix(
            eye=np.array([0, 0, 10], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32),
        )
        proj = create_perspective_matrix(90.0, 1.0, 0.1, 100.0)
        view_proj = matmul_4x4(proj, view)

        # Wide FOV, positions around origin → expect most visible
        positions = np.random.uniform(-1, 1, (64, 3)).astype(np.float32)
        visible = frustum_culler.cull_nodes(positions, view_proj=view_proj, view=view)
        assert len(visible) > 50

        # All nodes behind camera → expect zero
        positions_behind = np.random.uniform(-5, 5, (64, 3)).astype(np.float32)
        positions_behind[:, 2] = np.random.uniform(15, 50, 64)
        visible_behind = frustum_culler.cull_nodes(positions_behind, view_proj=view_proj, view=view)
        assert len(visible_behind) == 0

    def test_empty_input(self, frustum_culler: FrustumCuller):
        positions = np.zeros((100, 3), dtype=np.float32)
        candidates = np.array([], dtype=np.uint32)
        view_proj = np.eye(4, dtype=np.float32)
        view = np.eye(4, dtype=np.float32)

        visible = frustum_culler.cull_nodes(positions, candidates, view_proj, view)
        assert visible.size == 0


# ---------------------------------------------------------------------------
# Performance-focused tests (upper bounds are conservative to keep CI stable)
# ---------------------------------------------------------------------------
class TestFrustumCullingPerformance:
    def _time_cull(
        self,
        culler: FrustumCuller,
        positions: np.ndarray,
        view_proj: np.ndarray,
        view: np.ndarray,
        runs: int,
    ) -> float:
        # Warmup once
        culler.cull_nodes(positions, view_proj=view_proj, view=view)
        culler.reset_statistics()

        start = time.perf_counter()
        for _ in range(runs):
            culler.cull_nodes(positions, view_proj=view_proj, view=view)
        end = time.perf_counter()
        return (end - start) * 1000.0 / runs  # ms per run

    @pytest.mark.benchmark
    def test_performance_1k_nodes(self, frustum_culler: FrustumCuller):
        np.random.seed(456)
        positions = np.random.uniform(-50, 50, (1000, 3)).astype(np.float32)

        view = create_view_matrix(
            eye=np.array([0, 0, 100], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32),
        )
        proj = create_perspective_matrix(60.0, 1.0, 1.0, 500.0)
        view_proj = matmul_4x4(proj, view)

        avg_ms = self._time_cull(frustum_culler, positions, view_proj, view, runs=50)
        assert avg_ms < 1.5, f"Expected <1.5ms per 1K nodes, measured {avg_ms:.3f}ms"

    @pytest.mark.benchmark
    def test_performance_28k_nodes(self, frustum_culler: FrustumCuller):
        np.random.seed(789)
        positions = np.random.uniform(-100, 100, (28_000, 3)).astype(np.float32)

        view = create_view_matrix(
            eye=np.array([50, 50, 150], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32),
        )
        proj = create_perspective_matrix(60.0, 16 / 9, 1.0, 1000.0)
        view_proj = matmul_4x4(proj, view)

        avg_ms = self._time_cull(frustum_culler, positions, view_proj, view, runs=25)
        assert avg_ms < 6.0, f"Expected <6.0ms per 28K nodes, measured {avg_ms:.3f}ms"

    @pytest.mark.benchmark
    def test_reduction_rate(self, frustum_culler: FrustumCuller):
        np.random.seed(999)
        n = 10_000
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0, np.pi, n)
        r = np.random.uniform(10, 100, n)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        positions = np.stack([x, y, z], axis=-1).astype(np.float32)

        view = create_view_matrix(
            eye=np.array([0, 0, 200], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32),
        )
        proj = create_perspective_matrix(45.0, 1.0, 10.0, 500.0)
        view_proj = matmul_4x4(proj, view)

        frustum_culler.reset_statistics()
        visible = frustum_culler.cull_nodes(positions, view_proj=view_proj, view=view)
        stats = frustum_culler.get_statistics()
        reduction = stats["avg_reduction"]

        assert reduction > 0.75, f"Expected >75% reduction, observed {reduction*100:.1f}%"
        assert visible.size < positions.shape[0] * 0.25
