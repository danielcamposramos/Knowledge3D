"""
Test suite for frustum culling system.

Tests cover:
1. Correctness: SIMD output matches CPU ground truth
2. Performance: <0.02ms on 28k nodes
3. Reduction: >80% candidate reduction
4. Integration: Chain from Morton to LED-A*

Author: The Swarm (Codex's test harness spec)
Branch: phase4-frustum-simd-v1
"""

import pytest
import numpy as np
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - skip on environments without CuPy
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False

pytestmark = []  # type: ignore[var-annotated]
if not _CUPY_AVAILABLE:
    pytestmark.append(pytest.mark.skip(reason="CuPy not available"))
else:  # pragma: no branch - best effort device detection
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytestmark.append(pytest.mark.skip(reason="CUDA device not available"))
    except Exception:
        pytestmark.append(pytest.mark.skip(reason="CUDA device not available"))

# Import frustum culling
from knowledge3d.spatial.frustum import (
    FrustumCuller,
    create_perspective_matrix,
    create_view_matrix
)


class TestFrustumCullingCorrectness:
    """Test frustum culling correctness against CPU ground truth."""

    def cpu_frustum_test(
        self,
        positions: np.ndarray,
        view_proj: np.ndarray,
        view: np.ndarray,
        margin_xy: float = 0.11,
        margin_z: float = 1.0,
    ) -> np.ndarray:
        """
        CPU reference implementation matching the GPU kernel logic.

        Args:
            positions: (N, 3) node positions
            view_proj: 4x4 view-projection matrix
            view: 4x4 view matrix (for depth test)
            margin: NDC margin (default 5%)

        Returns:
            Boolean array of visible nodes
        """
        n = len(positions)
        visible = np.ones(n, dtype=bool)

        for i in range(n):
            vec = np.append(positions[i], 1.0).astype(np.float32)

            # View-space depth: camera looks down -Z → vz must be < 0
            vz = float(view[2].dot(vec))
            if vz >= 0.0:
                visible[i] = False
                continue

            clip = view_proj @ vec

            # Degenerate projection, cull
            if clip[3] <= 0.0:
                visible[i] = False
                continue

            ndc = clip[:3] / clip[3]

            if ndc[0] < -margin_xy or ndc[0] > margin_xy:
                visible[i] = False
                continue
            if ndc[1] < -margin_xy or ndc[1] > margin_xy:
                visible[i] = False
                continue
            if ndc[2] < -margin_z or ndc[2] > margin_z:
                visible[i] = False

        return visible

    def test_single_warp_correctness(self):
        """Test SIMD kernel on single warp (32 nodes)."""
        # Generate random positions in [-10, 10]^3
        np.random.seed(42)
        positions = np.random.uniform(-10, 10, (32, 3)).astype(np.float32)

        # Create view matrix (camera looking at origin from [0, 0, 20])
        view = create_view_matrix(
            eye=np.array([0, 0, 20], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )

        # Create projection matrix (60° FOV, near=0.1, far=100)
        proj = create_perspective_matrix(
            fov_degrees=60.0,
            aspect_ratio=1.0,
            near=0.1,
            far=100.0
        )

        view_proj = proj @ view

        # Upload to GPU
        positions_gpu = cp.asarray(positions)

        # Run GPU frustum culling
        culler = FrustumCuller(enable_profiling=False)
        visible_indices_gpu = culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)
        visible_indices = cp.asnumpy(visible_indices_gpu)

        # Run CPU reference
        visible_cpu = self.cpu_frustum_test(positions, view_proj, view)
        expected_indices = np.where(visible_cpu)[0]

        # Compare
        np.testing.assert_array_equal(
            np.sort(visible_indices),
            np.sort(expected_indices),
            err_msg="GPU and CPU frustum results differ"
        )

    def test_multiple_warps_correctness(self):
        """Test SIMD kernel on multiple warps (128 nodes)."""
        np.random.seed(123)
        positions = np.random.uniform(-20, 20, (128, 3)).astype(np.float32)

        # Create view-projection
        view = create_view_matrix(
            eye=np.array([10, 10, 30], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )

        proj = create_perspective_matrix(60.0, 16/9, 0.1, 200.0)
        view_proj = proj @ view

        # GPU culling
        positions_gpu = cp.asarray(positions)
        culler = FrustumCuller()
        visible_gpu = culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        # CPU reference
        visible_cpu = self.cpu_frustum_test(positions, view_proj, view)
        expected = np.where(visible_cpu)[0]

        # Verify
        assert len(visible_gpu) == len(expected), \
            f"GPU found {len(visible_gpu)} visible, CPU found {len(expected)}"

        np.testing.assert_array_equal(
            np.sort(cp.asnumpy(visible_gpu)),
            np.sort(expected)
        )

    def test_edge_cases(self):
        """Test edge cases: all visible, all culled, partial."""
        culler = FrustumCuller()

        # Case 1: All nodes inside frustum (clustered at origin)
        positions = np.random.uniform(-1, 1, (64, 3)).astype(np.float32)
        view = create_view_matrix(
            eye=np.array([0, 0, 10], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )
        proj = create_perspective_matrix(90.0, 1.0, 0.1, 100.0)  # Wide FOV
        view_proj = proj @ view

        positions_gpu = cp.asarray(positions)
        visible = culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        # Should see most/all nodes (wide FOV, clustered)
        assert len(visible) > 50, "Wide FOV should see most nodes"

        # Case 2: All nodes behind camera (should be culled)
        positions_behind = np.random.uniform(-5, 5, (64, 3)).astype(np.float32)
        positions_behind[:, 2] = np.random.uniform(15, 50, 64)  # All behind camera (positive Z)

        positions_behind_gpu = cp.asarray(positions_behind)
        visible_behind = culler.cull_nodes(positions_behind_gpu, view_proj=view_proj, view=view)

        # Should cull all (behind camera)
        assert len(visible_behind) == 0, "Nodes behind camera should be culled"

    def test_empty_input(self):
        """Test with zero candidates."""
        culler = FrustumCuller()
        positions_gpu = cp.zeros((100, 3), dtype=np.float32)
        candidates = cp.array([], dtype=cp.uint32)

        view_proj = np.eye(4, dtype=np.float32)
        view = np.eye(4, dtype=np.float32)
        visible = culler.cull_nodes(positions_gpu, candidates, view_proj, view)

        assert len(visible) == 0, "Empty input should return empty output"


class TestFrustumCullingPerformance:
    """Test frustum culling performance targets."""

    @pytest.mark.benchmark
    def test_performance_1k_nodes(self):
        """Test culling performance on 1k nodes."""
        np.random.seed(456)
        positions = np.random.uniform(-50, 50, (1000, 3)).astype(np.float32)
        positions_gpu = cp.asarray(positions)

        view = create_view_matrix(
            eye=np.array([0, 0, 100], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )
        proj = create_perspective_matrix(60.0, 1.0, 1.0, 500.0)
        view_proj = proj @ view

        # Warmup
        culler = FrustumCuller(enable_profiling=True)
        for _ in range(5):
            culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        # Timed runs
        culler.reset_statistics()
        n_runs = 100

        for _ in range(n_runs):
            culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        stats = culler.get_statistics()
        avg_time_ms = stats['avg_time_ms']

        print(f"\n1K nodes: {avg_time_ms:.4f}ms average")
        assert avg_time_ms < 0.015, f"1K nodes should cull in <0.015ms, got {avg_time_ms:.4f}ms"

    @pytest.mark.benchmark
    def test_performance_28k_nodes(self):
        """Test culling performance on 28k nodes (MVP target)."""
        np.random.seed(789)
        positions = np.random.uniform(-100, 100, (28000, 3)).astype(np.float32)
        positions_gpu = cp.asarray(positions)

        view = create_view_matrix(
            eye=np.array([50, 50, 150], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )
        proj = create_perspective_matrix(60.0, 16/9, 1.0, 1000.0)
        view_proj = proj @ view

        # Warmup
        culler = FrustumCuller(enable_profiling=True)
        for _ in range(3):
            culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        # Timed runs
        culler.reset_statistics()
        n_runs = 50

        for _ in range(n_runs):
            culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        stats = culler.get_statistics()
        avg_time_ms = stats['avg_time_ms']

        print(f"\n28K nodes: {avg_time_ms:.4f}ms average (target: <0.020ms)")
        print(f"  Reduction: {stats['avg_reduction']*100:.1f}% (target: >80%)")
        print(f"  Input: {stats['avg_input_size']}, Output: {stats['avg_output_size']}")

        # MVP performance target: <0.020ms (Kimi's SIMD target: 0.018ms)
        assert avg_time_ms < 0.020, \
            f"28K nodes should cull in <0.020ms, got {avg_time_ms:.4f}ms"

    @pytest.mark.benchmark
    def test_reduction_rate(self):
        """Test that frustum achieves >80% candidate reduction."""
        np.random.seed(999)

        # Generate positions uniformly in sphere around origin
        n = 10000
        theta = np.random.uniform(0, 2*np.pi, n)
        phi = np.random.uniform(0, np.pi, n)
        r = np.random.uniform(10, 100, n)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        positions = np.stack([x, y, z], axis=-1).astype(np.float32)
        positions_gpu = cp.asarray(positions)

        # Camera looking at partial sphere (should cull ~80-90%)
        view = create_view_matrix(
            eye=np.array([0, 0, 200], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )
        proj = create_perspective_matrix(45.0, 1.0, 10.0, 500.0)  # Narrow FOV
        view_proj = proj @ view

        culler = FrustumCuller(enable_profiling=True)
        visible = culler.cull_nodes(positions_gpu, view_proj=view_proj, view=view)

        stats = culler.get_statistics()
        reduction = stats['avg_reduction']

        print(f"\nReduction rate: {reduction*100:.1f}% (target: >80%)")
        assert reduction > 0.80, \
            f"Frustum should reduce by >80%, got {reduction*100:.1f}%"


class TestFrustumCullingIntegration:
    """Test frustum integration with Morton octree and navigator."""

    def test_chain_from_morton(self):
        """Test frustum chaining from Morton octree candidates."""
        # This test requires morton_octree.py integration
        # Placeholder for now - will implement after semantic_navigator integration

        # Expected flow:
        #   1. Morton octree query -> 30% of nodes
        #   2. Frustum cull -> 5% of nodes (80% reduction from 30%)
        #   3. LED-A* pathfinding on 5%

        pytest.skip("Requires semantic_navigator integration")

    def test_view_projection_from_avatar(self):
        """Test view-projection extraction from avatar/fused head."""
        # Placeholder - will implement after fused_head integration
        pytest.skip("Requires fused_head integration")

    def test_end_to_end_query_latency(self):
        """Test full query chain: Morton -> Frustum -> LED-A* in <100ms."""
        # Placeholder - will implement after full integration
        pytest.skip("Requires full semantic_navigator integration")


class TestMatrixUtilities:
    """Test matrix creation utilities."""

    def test_perspective_matrix(self):
        """Test perspective projection matrix creation."""
        proj = create_perspective_matrix(60.0, 16/9, 0.1, 100.0)

        assert proj.shape == (4, 4)
        assert proj.dtype == np.float32

        # Check basic properties
        assert proj[3, 2] == -1.0  # Perspective projection marker
        assert proj[2, 3] < 0  # Near-far encoding

    def test_view_matrix(self):
        """Test view matrix creation."""
        eye = np.array([10, 5, 20], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)

        view = create_view_matrix(eye, target, up)

        assert view.shape == (4, 4)
        assert view.dtype == np.float32

        # Transform origin to check view transform
        origin_world = np.array([0, 0, 0, 1], dtype=np.float32)
        origin_view = view @ origin_world

        # Origin should be at -camera_distance in view space
        camera_dist = np.linalg.norm(eye - target)
        assert abs(origin_view[2] + camera_dist) < 0.01

    def test_frustum_plane_extraction(self):
        """Test frustum plane extraction from view-projection matrix."""
        view = create_view_matrix(
            eye=np.array([0, 0, 10], dtype=np.float32),
            target=np.array([0, 0, 0], dtype=np.float32),
            up=np.array([0, 1, 0], dtype=np.float32)
        )
        proj = create_perspective_matrix(60.0, 1.0, 1.0, 100.0)
        view_proj = proj @ view

        culler = FrustumCuller()
        planes = culler.extract_frustum_planes_from_matrix(view_proj)

        assert planes.shape == (6, 4)
        assert planes.dtype == np.float32

        # Check planes are normalized
        for i in range(6):
            normal_length = np.sqrt(planes[i, 0]**2 + planes[i, 1]**2 + planes[i, 2]**2)
            assert abs(normal_length - 1.0) < 0.01, f"Plane {i} not normalized"


if __name__ == "__main__":
    # Run basic correctness tests
    print("Running frustum culling tests...")

    correctness = TestFrustumCullingCorrectness()
    print("\n=== Correctness Tests ===")
    correctness.test_single_warp_correctness()
    print("✓ Single warp correctness")

    correctness.test_multiple_warps_correctness()
    print("✓ Multiple warps correctness")

    correctness.test_edge_cases()
    print("✓ Edge cases")

    correctness.test_empty_input()
    print("✓ Empty input")

    # Run performance tests
    performance = TestFrustumCullingPerformance()
    print("\n=== Performance Tests ===")

    performance.test_performance_1k_nodes()
    print("✓ 1K nodes performance")

    performance.test_performance_28k_nodes()
    print("✓ 28K nodes performance (MVP target)")

    performance.test_reduction_rate()
    print("✓ Reduction rate >80%")

    # Matrix utilities
    matrix_tests = TestMatrixUtilities()
    print("\n=== Matrix Utilities ===")

    matrix_tests.test_perspective_matrix()
    print("✓ Perspective matrix")

    matrix_tests.test_view_matrix()
    print("✓ View matrix")

    matrix_tests.test_frustum_plane_extraction()
    print("✓ Frustum plane extraction")

    print("\n✅ All tests passed!")
