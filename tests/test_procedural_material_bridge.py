from __future__ import annotations

import numpy as np
import pytest



def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_material_selection_prefers_positive_candidate():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )

    bridge = ProceduralMaterialBridge()
    target = SurfaceMaterialCandidate(
        material_id="target",
        name="Target",
        palette=((0.1, 0.2, 0.8, 1.0), (0.5, 0.6, 0.95, 1.0), (0.9, 0.95, 1.0, 1.0)),
    )
    positive = SurfaceMaterialCandidate(
        material_id="positive",
        name="Positive",
        palette=((0.12, 0.18, 0.78, 1.0), (0.52, 0.62, 0.94, 1.0), (0.92, 0.97, 1.0, 1.0)),
    )
    negative = SurfaceMaterialCandidate(
        material_id="negative",
        name="Negative",
        palette=((0.8, 0.2, 0.1, 1.0), (0.92, 0.55, 0.25, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    selection = bridge.select_material(
        target_material=target,
        candidates=(positive, negative),
        negative_materials=(negative,),
    )

    assert selection.selected.material_id == "positive"
    assert selection.score_table[0]["score"] > selection.score_table[1]["score"]
    assert selection.selected_stops
    assert int(selection.math_core_plan["preferred_tier"]) == 2
    assert int(selection.math_core_plan["batch_size"]) >= 1


@pytest.mark.cuda
def test_project_material_emits_vertex_rgba_and_weights():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )

    grid = np.zeros((18, 18), dtype=np.int32)
    grid[4:14, 6:11] = 1

    geometry = ProceduralGeometryBridge()
    mesh = geometry.contour_to_lathe_mesh(grid, color=1, pad=1, segments=12)
    material = SurfaceMaterialCandidate(
        material_id="cool_glass",
        name="Cool Glass",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
        projection_strategy="triplanar",
        tiling=1.5,
    )

    bridge = ProceduralMaterialBridge()
    plan = bridge.project_material(mesh, material, preview_size=32)

    assert plan.material_preview.shape == (32, 32, 4)
    assert plan.normal_hint.shape == (32, 32)
    assert plan.vertex_rgba.shape == (mesh.vertices.shape[0], 4)
    assert plan.projection_weights.shape == (mesh.vertices.shape[0], 3)
    assert np.allclose(np.sum(plan.projection_weights, axis=1), 1.0, atol=1e-4)
    assert float(np.mean(plan.vertex_rgba[:, 2])) > float(np.mean(plan.vertex_rgba[:, 0]))
    assert plan.metadata["material_id"] == "cool_glass"
    assert int(plan.metadata["math_core_plan"]["preferred_tier"]) == 2


@pytest.mark.cuda
def test_contour_to_textured_lathe_mesh_uses_selected_material():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 7:12] = 1

    target = SurfaceMaterialCandidate(
        material_id="target",
        name="Target",
        palette=((0.05, 0.1, 0.45, 1.0), (0.3, 0.5, 0.8, 1.0), (0.85, 0.95, 1.0, 1.0)),
    )
    preferred = SurfaceMaterialCandidate(
        material_id="preferred",
        name="Preferred",
        base_stop=(0.0, 0.05, 0.1, 0.45, 1.0),
        position_layers=((0, 0, 0), (0, 1, 0)),
        color_layers=(
            ((1, 1, 0, 0), (1, 1, 0, 0), (0, 1, 1, 0)),
            ((0, 0, 1, 0), (0, 1, 1, 0), (1, 1, 0, 0)),
        ),
        tiling=1.25,
    )
    rejected = SurfaceMaterialCandidate(
        material_id="rejected",
        name="Rejected",
        palette=((0.8, 0.15, 0.08, 1.0), (0.95, 0.55, 0.2, 1.0), (1.0, 0.9, 0.65, 1.0)),
    )

    bridge = ProceduralMaterialBridge()
    plan = bridge.contour_to_textured_lathe_mesh(
        grid,
        color=1,
        pad=1,
        segments=12,
        target_material=target,
        candidates=(preferred, rejected),
        negative_materials=(rejected,),
        preview_size=32,
    )

    assert plan.selected_material.material_id == "preferred"
    assert plan.mesh.indices.shape[0] > 0
    assert plan.material_preview.shape == (32, 32, 4)
    assert plan.metadata["material_score_table"]
    assert plan.metadata["selected_material_stops"]
    assert int(plan.metadata["selection_math_core_plan"]["preferred_tier"]) == 2


@pytest.mark.cuda
def test_contour_to_textured_extrude_mesh_uses_selected_material():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 7:12] = 1

    target = SurfaceMaterialCandidate(
        material_id="target_extrude",
        name="Target Extrude",
        palette=((0.05, 0.1, 0.45, 1.0), (0.3, 0.5, 0.8, 1.0), (0.85, 0.95, 1.0, 1.0)),
    )
    preferred = SurfaceMaterialCandidate(
        material_id="preferred_extrude",
        name="Preferred Extrude",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    rejected = SurfaceMaterialCandidate(
        material_id="rejected_extrude",
        name="Rejected Extrude",
        palette=((0.8, 0.15, 0.08, 1.0), (0.95, 0.55, 0.2, 1.0), (1.0, 0.9, 0.65, 1.0)),
    )

    bridge = ProceduralMaterialBridge()
    plan = bridge.contour_to_textured_extrude_mesh(
        grid,
        color=1,
        pad=1,
        depth_scale=0.4,
        target_material=target,
        candidates=(preferred, rejected),
        negative_materials=(rejected,),
        preview_size=32,
    )

    assert plan.selected_material.material_id == "preferred_extrude"
    assert plan.mesh.metadata["mesh_kind"] == "extrude"
    assert plan.vertex_rgba.shape == (plan.mesh.vertices.shape[0], 4)
    assert int(plan.metadata["selection_math_core_plan"]["preferred_tier"]) == 2


@pytest.mark.cuda
def test_contour_to_textured_sweep_mesh_uses_selected_material():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 7:12] = 1

    target = SurfaceMaterialCandidate(
        material_id="target_sweep",
        name="Target Sweep",
        palette=((0.05, 0.1, 0.45, 1.0), (0.3, 0.5, 0.8, 1.0), (0.85, 0.95, 1.0, 1.0)),
    )
    preferred = SurfaceMaterialCandidate(
        material_id="preferred_sweep",
        name="Preferred Sweep",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    rejected = SurfaceMaterialCandidate(
        material_id="rejected_sweep",
        name="Rejected Sweep",
        palette=((0.8, 0.15, 0.08, 1.0), (0.95, 0.55, 0.2, 1.0), (1.0, 0.9, 0.65, 1.0)),
    )

    bridge = ProceduralMaterialBridge()
    plan = bridge.contour_to_textured_sweep_mesh(
        grid,
        color=1,
        pad=1,
        depth_scale=0.4,
        target_material=target,
        candidates=(preferred, rejected),
        negative_materials=(rejected,),
        preview_size=32,
    )

    assert plan.selected_material.material_id == "preferred_sweep"
    assert plan.mesh.metadata["mesh_kind"] == "sweep"
    assert plan.vertex_rgba.shape == (plan.mesh.vertices.shape[0], 4)
    assert int(plan.metadata["selection_math_core_plan"]["preferred_tier"]) == 2


@pytest.mark.cuda
def test_project_material_planar_xy_uses_tier1_projection_plan():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_geometry_bridge import ProceduralGeometryBridge
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )

    grid = np.zeros((18, 18), dtype=np.int32)
    grid[4:14, 6:11] = 1

    geometry = ProceduralGeometryBridge()
    mesh = geometry.contour_to_extrude_mesh(grid, color=1, pad=1, depth_scale=0.4, width_scale=1.2)
    material = SurfaceMaterialCandidate(
        material_id="flat_paint",
        name="Flat Paint",
        palette=((0.2, 0.3, 0.7, 1.0), (0.7, 0.85, 1.0, 1.0)),
        projection_strategy="planar_xy",
        tiling=1.0,
    )

    bridge = ProceduralMaterialBridge()
    plan = bridge.project_material(mesh, material, preview_size=32)

    assert plan.vertex_rgba.shape == (mesh.vertices.shape[0], 4)
    assert int(plan.metadata["math_core_plan"]["preferred_tier"]) == 1
    assert plan.metadata["projection_strategy"] == "planar_xy"


@pytest.mark.cuda
def test_material_bridge_reuses_global_signal_bridge_cache():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import ProceduralMaterialBridge

    first = ProceduralMaterialBridge()
    second = ProceduralMaterialBridge()

    bridge_a = first._signal_bridge_for(frame_size=256, threshold=0.2)
    bridge_b = second._signal_bridge_for(frame_size=256, threshold=0.2)
    bridge_c = second._signal_bridge_for(frame_size=512, threshold=0.2)

    assert bridge_a is bridge_b
    assert bridge_a is not bridge_c


@pytest.mark.cuda
def test_material_bridge_warmup_is_idempotent():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import ProceduralMaterialBridge

    bridge = ProceduralMaterialBridge()
    first = bridge.warmup_runtime()
    second = bridge.warmup_runtime()

    assert first["status"] == "ready"
    assert float(first["total_warmup_ms"]) > 0.0
    assert first == second


@pytest.mark.cuda
def test_signal_to_textured_surface_composes_signal_geometry_and_material():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.ternary import TernaryVector

    samples = TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)])
    cool = SurfaceMaterialCandidate(
        material_id="cool",
        name="Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
        projection_strategy="triplanar",
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
        projection_strategy="triplanar",
    )

    bridge = ProceduralMaterialBridge()
    plan = bridge.signal_to_textured_surface(
        "signal_material",
        samples,
        candidates=(cool, warm),
        negative_materials=(warm,),
        frame_size=256,
        threshold=0.15,
        displacement_gain=0.5,
        preview_size=32,
    )

    assert plan.mesh.vertices.shape[0] > 0
    assert plan.mesh.indices.shape[0] > 0
    assert plan.vertex_rgba.shape == (plan.mesh.vertices.shape[0], 4)
    assert plan.metadata["signal_projection_summary"]["frame_count"] == 4
    assert int(plan.metadata["signal_math_core_plan"]["preferred_tier"]) == 2
    assert int(plan.metadata["signal_surface_math_core_plan"]["preferred_tier"]) == 3
    assert int(plan.metadata["selection_math_core_plan"]["preferred_tier"]) == 2
    assert int(plan.metadata["math_core_plan"]["preferred_tier"]) == 2
