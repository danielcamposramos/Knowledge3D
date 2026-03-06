from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_surface_material_to_temporal_preview_emits_frames_and_coherence():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 7:12] = 1

    target = SurfaceMaterialCandidate(
        material_id="timeline_target",
        name="Timeline Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="timeline_warm",
        name="Timeline Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    material_bridge = ProceduralMaterialBridge()
    surface_plan = material_bridge.contour_to_textured_lathe_mesh(
        grid,
        color=1,
        pad=1,
        segments=12,
        target_material=target,
        candidates=(target, warm),
        negative_materials=(warm,),
        preview_size=32,
    )

    temporal_bridge = ProceduralTemporalBridge()
    preview = temporal_bridge.surface_material_to_temporal_preview(
        surface_plan,
        frame_count=4,
        time_span=0.3,
        feature_grid=8,
        encode_frames=True,
    )

    assert preview.frames.shape == (4, 32, 32, 3)
    assert preview.frame_features.shape[0] == 4
    assert preview.temporal_deltas.shape == preview.frame_features.shape
    assert preview.coherence_scores.shape[0] == preview.frame_features.shape[1]
    assert preview.metadata["frame_count"] == 4
    assert preview.metadata["codec_enabled"] is True
    assert len(preview.metadata["encoded_frames"]) == 4
    assert float(preview.metadata["overall_coherence"]) >= 0.0


@pytest.mark.cuda
def test_surface_material_to_timeline_preset_emits_named_ui_and_world_metadata():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    grid = np.zeros((24, 24), dtype=np.int32)
    grid[5:19, 8:14] = 1

    cool = SurfaceMaterialCandidate(
        material_id="timeline_cool",
        name="Timeline Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="timeline_warm",
        name="Timeline Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    material_bridge = ProceduralMaterialBridge()
    surface_plan = material_bridge.contour_to_textured_lathe_mesh(
        grid,
        color=1,
        pad=1,
        segments=12,
        target_material=cool,
        candidates=(cool, warm),
        negative_materials=(warm,),
        preview_size=32,
    )

    temporal_bridge = ProceduralTemporalBridge()
    ui_plan = temporal_bridge.surface_material_to_timeline_preset(
        surface_plan,
        timeline_preset="ui_focus",
        encode_frames=True,
    )
    world_plan = temporal_bridge.surface_material_to_timeline_preset(
        surface_plan,
        timeline_preset="world_orbit",
        encode_frames=True,
    )

    assert ui_plan.metadata["timeline_preset"] == "ui_focus"
    assert ui_plan.metadata["timeline_domain"] == "ui"
    assert ui_plan.metadata["timeline_effect"] == "ui_focus"
    assert ui_plan.metadata["frame_count"] == 8
    assert world_plan.metadata["timeline_preset"] == "world_orbit"
    assert world_plan.metadata["timeline_domain"] == "world"
    assert world_plan.metadata["timeline_curve"] == "orbit"
    assert world_plan.metadata["frame_count"] == 12
    assert ui_plan.frames.shape[0] != world_plan.frames.shape[0]


@pytest.mark.cuda
def test_surface_material_to_timeline_preset_reuses_cached_plan_without_timeline_id():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    grid = np.zeros((24, 24), dtype=np.int32)
    grid[5:19, 8:14] = 1

    cool = SurfaceMaterialCandidate(
        material_id="timeline_cache",
        name="Timeline Cache",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )

    material_bridge = ProceduralMaterialBridge()
    surface_plan = material_bridge.contour_to_textured_lathe_mesh(
        grid,
        color=1,
        pad=1,
        segments=12,
        target_material=cool,
        candidates=(cool,),
        preview_size=32,
    )

    temporal_bridge = ProceduralTemporalBridge()
    first = temporal_bridge.surface_material_to_timeline_preset(
        surface_plan,
        timeline_preset="ui_idle",
        encode_frames=True,
    )
    second = temporal_bridge.surface_material_to_timeline_preset(
        surface_plan,
        timeline_preset="ui_idle",
        encode_frames=True,
    )

    assert first is second


@pytest.mark.cuda
def test_surface_material_to_timeline_preset_bypasses_cache_with_timeline_id():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    grid = np.zeros((24, 24), dtype=np.int32)
    grid[5:19, 8:14] = 1

    cool = SurfaceMaterialCandidate(
        material_id="timeline_cache_id",
        name="Timeline Cache Id",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )

    material_bridge = ProceduralMaterialBridge()
    surface_plan = material_bridge.contour_to_textured_lathe_mesh(
        grid,
        color=1,
        pad=1,
        segments=12,
        target_material=cool,
        candidates=(cool,),
        preview_size=32,
    )

    temporal_bridge = ProceduralTemporalBridge()
    first = temporal_bridge.surface_material_to_timeline_preset(
        surface_plan,
        timeline_preset="ui_idle",
        encode_frames=True,
        timeline_id="timeline_a",
    )
    second = temporal_bridge.surface_material_to_timeline_preset(
        surface_plan,
        timeline_preset="ui_idle",
        encode_frames=True,
        timeline_id="timeline_b",
    )

    assert first is not second
    assert first.metadata["encoded_frames"][0]["frame_id"].startswith("timeline_a")
    assert second.metadata["encoded_frames"][0]["frame_id"].startswith("timeline_b")


@pytest.mark.cuda
def test_surface_materials_to_scene_timeline_composes_multiple_layers():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    grid_a = np.zeros((24, 24), dtype=np.int32)
    grid_a[5:19, 8:13] = 1
    grid_b = np.zeros((24, 24), dtype=np.int32)
    grid_b[6:18, 10:16] = 1

    cool = SurfaceMaterialCandidate(
        material_id="scene_cool",
        name="Scene Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="scene_warm",
        name="Scene Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    material_bridge = ProceduralMaterialBridge()
    lathe_plan = material_bridge.contour_to_textured_lathe_mesh(
        grid_a,
        color=1,
        pad=1,
        segments=12,
        target_material=cool,
        candidates=(cool, warm),
        negative_materials=(warm,),
        preview_size=32,
    )
    sweep_plan = material_bridge.contour_to_textured_sweep_mesh(
        grid_b,
        color=1,
        pad=1,
        depth_scale=0.5,
        width_scale=1.0,
        height_scale=1.0,
        target_material=warm,
        candidates=(cool, warm),
        negative_materials=(cool,),
        preview_size=32,
    )

    temporal_bridge = ProceduralTemporalBridge()
    scene = temporal_bridge.surface_materials_to_scene_timeline(
        (lathe_plan, sweep_plan),
        timeline_preset="ui_idle",
        frame_count=4,
        scene_layout="horizontal_strip",
        encode_frames=True,
        scene_id="ui_scene_test",
    )

    assert scene.frames.shape == (4, 32, 64, 4)
    assert scene.metadata["frame_count"] == 4
    assert scene.metadata["layer_count"] == 2
    assert scene.metadata["scene_layout"] == "horizontal_strip"
    assert scene.metadata["scene_domain"] == "ui"
    assert scene.metadata["codec_enabled"] is True
    assert len(scene.metadata["encoded_frames"]) == 4
    assert all(layer.preview_plan.metadata["codec_enabled"] is False for layer in scene.layers)


@pytest.mark.cuda
def test_replay_journal_to_scene_timeline_builds_golden_orbit_playback(tmp_path):
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    journal_path = tmp_path / "replay_actions.jsonl"
    journal_path.write_text(
        "\n".join(
            [
                '{"timestamp": 1, "action_type": "NAV_MOVE", "raw_confidence": 0.6, "final_confidence": 0.72, "curiosity": 0.1}',
                '{"timestamp": 2, "action_type": "DIALOGUE", "raw_confidence": 0.7, "final_confidence": 0.82, "curiosity": 0.2}',
                '{"timestamp": 3, "action_type": "WRITE_MEM", "raw_confidence": 0.8, "final_confidence": 0.9, "curiosity": 0.3}',
            ]
        ),
        encoding="utf-8",
    )

    temporal_bridge = ProceduralTemporalBridge()
    scene = temporal_bridge.replay_journal_to_scene_timeline(
        journal_path=journal_path,
        frame_count=5,
        scene_layout="golden_orbit",
        scene_id="replay_scene_test",
    )

    assert scene.frames.shape[0] == 5
    assert scene.frames.shape[3] == 4
    assert scene.metadata["scene_layout"] == "golden_orbit"
    assert scene.metadata["replay_count"] == 3
    assert scene.metadata["replay_source"] == "journal"
    assert scene.metadata["replay_action_types"] == ["NAV_MOVE", "DIALOGUE", "WRITE_MEM"]
    assert scene.metadata["layer_count"] == 3


@pytest.mark.cuda
def test_execution_events_to_house_room_scene_reuses_cached_plan_without_scene_id():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    execution_events = [
        {"tool_id": "tool_fusion_surface_material_ui_animation_v1", "outcome": 1, "quality_signal": 0.92, "ternary_quality": 1, "timestamp_us": 1},
        {"tool_id": "tool_fusion_surface_material_world_animation_v1", "outcome": 1, "quality_signal": 0.86, "ternary_quality": 1, "timestamp_us": 2},
        {"tool_id": "tool_fusion_signal_surface_material_world_animation_v1", "outcome": 0, "quality_signal": 0.58, "ternary_quality": 0, "timestamp_us": 3, "curiosity": 0.8},
        {"tool_id": "tool_house_replay_scene_v1", "outcome": -1, "quality_signal": 0.22, "ternary_quality": -1, "timestamp_us": 4},
    ]

    temporal_bridge = ProceduralTemporalBridge()
    first = temporal_bridge.execution_events_to_house_room_scene(
        execution_events=execution_events,
        room_preset="house_library",
        encode_frames=True,
    )
    second = temporal_bridge.execution_events_to_house_room_scene(
        execution_events=execution_events,
        room_preset="house_library",
        encode_frames=True,
    )

    assert first is second
    assert first.metadata["house_room_preset"] == "house_library"
    assert first.metadata["event_count"] >= 1


@pytest.mark.cuda
def test_execution_events_to_house_tour_scene_bypasses_cache_with_scene_id():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    execution_events = [
        {"tool_id": "tool_fusion_surface_material_ui_animation_v1", "outcome": 1, "quality_signal": 0.92, "ternary_quality": 1, "timestamp_us": 1},
        {"tool_id": "tool_fusion_surface_material_world_animation_v1", "outcome": 1, "quality_signal": 0.81, "ternary_quality": 1, "timestamp_us": 2, "curiosity": 0.4},
        {"tool_id": "tool_fusion_signal_surface_material_world_animation_v1", "outcome": 0, "quality_signal": 0.55, "ternary_quality": 0, "timestamp_us": 3, "curiosity": 0.9},
        {"tool_id": "tool_house_replay_scene_v1", "outcome": -1, "quality_signal": 0.18, "ternary_quality": -1, "timestamp_us": 4},
    ]

    temporal_bridge = ProceduralTemporalBridge()
    first = temporal_bridge.execution_events_to_house_tour_scene(
        execution_events=execution_events,
        encode_frames=True,
        scene_id="tour_scene_a",
    )
    second = temporal_bridge.execution_events_to_house_tour_scene(
        execution_events=execution_events,
        encode_frames=True,
        scene_id="tour_scene_b",
    )

    assert first is not second
    assert first.metadata["house_room_preset"] == "house_tour"
    assert second.metadata["house_room_preset"] == "house_tour"
    assert first.metadata["encoded_frames"][0]["frame_id"].startswith("tour_scene_a")
    assert second.metadata["encoded_frames"][0]["frame_id"].startswith("tour_scene_b")
