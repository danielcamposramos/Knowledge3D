from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_contour_to_lathe_mesh_emits_vertices_faces_and_normals():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge

    grid = np.zeros((16, 16), dtype=np.int32)
    grid[3:13, 5:10] = 1

    bridge = ProceduralGeometryBridge()
    plan = bridge.contour_to_lathe_mesh(grid, color=1, pad=1, segments=12)

    assert plan.vertices.ndim == 2 and plan.vertices.shape[1] == 3
    assert plan.indices.ndim == 2 and plan.indices.shape[1] == 3
    assert plan.normals.shape == plan.vertices.shape
    assert plan.metadata["mesh_kind"] == "lathe"
    assert int(plan.metadata["segments"]) == 12
    assert plan.profile.nonzero_count > 0
    assert plan.vertices.shape[0] > 0
    assert plan.indices.shape[0] > 0
    assert "row_delta_trits" in plan.metadata
    assert set(plan.metadata["row_delta_trits"]).issubset({-1, 0, 1})
    assert "profile_rows_used" in plan.metadata
    assert len(plan.metadata["profile_rows_used"]) <= int(plan.metadata["profile_rows_total"])
    assert int(plan.metadata["math_core_plan"]["preferred_tier"]) == 2
    assert int(plan.metadata["math_core_plan"]["batch_size"]) >= 1


@pytest.mark.cuda
def test_ternary_trend_reduction_collapses_neutral_rows():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 6:12] = 1

    bridge = ProceduralGeometryBridge()
    plan = bridge.contour_to_lathe_mesh(grid, color=1, pad=1, segments=12)

    used_rows = plan.metadata["profile_rows_used"]
    total_rows = int(plan.metadata["profile_rows_total"])
    assert len(used_rows) < total_rows
    assert any(int(v) == 0 for v in plan.metadata["row_delta_trits"])


@pytest.mark.cuda
def test_geometry_bridge_warmup_is_idempotent():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge

    bridge = ProceduralGeometryBridge()
    first = bridge.warmup_runtime()
    second = bridge.warmup_runtime()

    assert first["status"] == "ready"
    assert float(first["total_warmup_ms"]) > 0.0
    assert int(first["warm_extrude_vertex_count"]) > 0
    assert int(first["warm_extrude_triangle_count"]) > 0
    assert int(first["warm_sweep_vertex_count"]) > 0
    assert int(first["warm_sweep_triangle_count"]) > 0
    assert first == second


@pytest.mark.cuda
def test_contour_to_extrude_mesh_emits_vertices_faces_and_normals():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge

    grid = np.zeros((18, 18), dtype=np.int32)
    grid[4:15, 6:10] = 1
    grid[8:15, 10] = 1

    bridge = ProceduralGeometryBridge()
    plan = bridge.contour_to_extrude_mesh(grid, color=1, pad=1, depth_scale=0.4, width_scale=1.2)

    assert plan.vertices.ndim == 2 and plan.vertices.shape[1] == 3
    assert plan.indices.ndim == 2 and plan.indices.shape[1] == 3
    assert plan.normals.shape == plan.vertices.shape
    assert plan.metadata["mesh_kind"] == "extrude"
    assert float(plan.metadata["depth_scale"]) == 0.4
    assert "width_delta_trits" in plan.metadata
    assert "left_delta_trits" in plan.metadata
    assert "right_delta_trits" in plan.metadata
    assert set(plan.metadata["width_delta_trits"]).issubset({-1, 0, 1})
    assert set(plan.metadata["left_delta_trits"]).issubset({-1, 0, 1})
    assert set(plan.metadata["right_delta_trits"]).issubset({-1, 0, 1})
    assert len(plan.metadata["profile_rows_used"]) <= int(plan.metadata["profile_rows_total"])
    assert int(plan.metadata["math_core_plan"]["preferred_tier"]) == 2
    assert plan.vertices.shape[0] > 0
    assert plan.indices.shape[0] > 0


@pytest.mark.cuda
def test_contour_to_sweep_mesh_emits_vertices_faces_and_normals():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 7:12] = 1
    grid[8:16, 12] = 1

    bridge = ProceduralGeometryBridge()
    plan = bridge.contour_to_sweep_mesh(grid, color=1, pad=1, depth_scale=0.4, width_scale=1.1)

    assert plan.vertices.ndim == 2 and plan.vertices.shape[1] == 3
    assert plan.indices.ndim == 2 and plan.indices.shape[1] == 3
    assert plan.normals.shape == plan.vertices.shape
    assert plan.metadata["mesh_kind"] == "sweep"
    assert "width_delta_trits" in plan.metadata
    assert "center_delta_trits" in plan.metadata
    assert set(plan.metadata["width_delta_trits"]).issubset({-1, 0, 1})
    assert set(plan.metadata["center_delta_trits"]).issubset({-1, 0, 1})
    assert len(plan.metadata["profile_rows_used"]) <= int(plan.metadata["profile_rows_total"])
    assert int(plan.metadata["math_core_plan"]["preferred_tier"]) == 2
    assert plan.vertices.shape[0] > 0
    assert plan.indices.shape[0] > 0
