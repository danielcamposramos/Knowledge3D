from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_scene_composition_applies_ternary_quality_weighting():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_material_bridge import (
        ProceduralMaterialBridge,
        SurfaceMaterialCandidate,
    )
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import (
        ProceduralTemporalBridge,
        TemporalSceneLayer,
    )

    grid = np.zeros((20, 20), dtype=np.int32)
    grid[4:16, 7:12] = 1
    cool = SurfaceMaterialCandidate(
        material_id="scene_quality_cool",
        name="Scene Quality Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="scene_quality_warm",
        name="Scene Quality Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    material_bridge = ProceduralMaterialBridge()
    surface = material_bridge.contour_to_textured_lathe_mesh(
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
    preview = temporal_bridge.surface_material_to_timeline_preset(
        surface,
        timeline_preset="ui_idle",
        frame_count=4,
        encode_frames=False,
    )

    low = TemporalSceneLayer(
        layer_id="low_quality",
        preview_plan=preview,
        opacity=1.0,
        z_index=10,
        metadata={"quality_signal": 0.20, "ternary_quality": -1},
    )
    high = TemporalSceneLayer(
        layer_id="high_quality",
        preview_plan=preview,
        opacity=1.0,
        z_index=0,
        metadata={"quality_signal": 0.86, "ternary_quality": 1},
    )
    scene = temporal_bridge.compose_scene_timeline([high, low], feature_grid=8, encode_frames=False)

    assert scene.layers[0].layer_id == "low_quality"
    assert scene.layers[-1].layer_id == "high_quality"
    assert scene.layers[0].opacity < scene.layers[-1].opacity
    assert scene.metadata["quality_weighting"] == "ternary_contrastive"
    assert scene.metadata["layer_ternary_quality"] == [-1, 1]


@pytest.mark.cuda
def test_replay_scene_detects_grammar_and_house_room_preset():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    temporal_bridge = ProceduralTemporalBridge()
    entries = [
        {"timestamp": 1, "action_type": "NAV_MOVE", "quality_signal": 0.82, "ternary_quality": 1, "final_confidence": 0.82},
        {"timestamp": 2, "action_type": "DIALOGUE", "quality_signal": 0.78, "ternary_quality": 1, "final_confidence": 0.78},
        {"timestamp": 3, "action_type": "NAV_MOVE", "quality_signal": 0.81, "ternary_quality": 1, "final_confidence": 0.81},
        {"timestamp": 4, "action_type": "DIALOGUE", "quality_signal": 0.77, "ternary_quality": 1, "final_confidence": 0.77},
        {"timestamp": 5, "action_type": "NAV_MOVE", "quality_signal": 0.84, "ternary_quality": 1, "final_confidence": 0.84},
        {"timestamp": 6, "action_type": "DIALOGUE", "quality_signal": 0.79, "ternary_quality": 1, "final_confidence": 0.79},
    ]
    scene = temporal_bridge.replay_journal_to_scene_timeline(
        replay_entries=entries,
        frame_count=4,
        scene_layout="golden_orbit",
        encode_frames=False,
    )

    assert scene.metadata["house_room_preset"] == "house_library"
    assert scene.metadata["scene_grammars"]
    assert scene.metadata["scene_grammars"][0]["sequence"] == ["NAV_MOVE", "DIALOGUE"]
    assert scene.metadata["scene_grammars"][0]["count"] >= 3


@pytest.mark.cuda
def test_execution_events_drive_library_garden_museum_and_tour_presets():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_temporal_bridge import ProceduralTemporalBridge

    bridge = ProceduralTemporalBridge()
    execution_events = [
        {
            "tool_id": "tool_house_library_scene_v1",
            "query_context": "what I know about settled geometry",
            "quality_signal": 0.91,
            "ternary_quality": 1,
            "outcome": 1,
            "timestamp_us": 10,
            "chain_tool_ids": ["tool_geom_profile_lathe_mesh_v1", "tool_fusion_surface_material_projection_v1"],
        },
        {
            "tool_id": "tool_house_garden_scene_v1",
            "query_context": "what I am learning about growing multimodal scenes",
            "quality_signal": 0.55,
            "ternary_quality": 0,
            "outcome": 0,
            "timestamp_us": 20,
            "chain_tool_ids": ["tool_signal_audio_spectrogram_v1", "tool_signal_spectrogram_surface_v1"],
        },
        {
            "tool_id": "tool_house_museum_scene_v1",
            "query_context": "my history of failed material attempts",
            "quality_signal": 0.18,
            "ternary_quality": -1,
            "outcome": -1,
            "timestamp_us": 30,
            "chain_tool_ids": ["tool_fusion_surface_material_projection_v1"],
        },
    ]

    library = bridge.execution_events_to_house_room_scene(
        execution_events=execution_events,
        room_preset="house_library",
        max_events=3,
        encode_frames=False,
    )
    garden = bridge.execution_events_to_house_room_scene(
        execution_events=execution_events,
        room_preset="house_garden",
        max_events=3,
        encode_frames=False,
    )
    museum = bridge.execution_events_to_house_room_scene(
        execution_events=execution_events,
        room_preset="house_museum",
        max_events=3,
        encode_frames=False,
    )
    tour = bridge.execution_events_to_house_tour_scene(
        execution_events=execution_events,
        max_events_per_room=3,
        encode_frames=False,
    )

    assert library.metadata["house_room_preset"] == "house_library"
    assert library.metadata["scene_layout"] == "overlay"
    assert all(int((layer.metadata or {}).get("ternary_quality", 0)) == 1 for layer in library.layers)

    assert garden.metadata["house_room_preset"] == "house_garden"
    assert garden.metadata["scene_layout"] == "golden_orbit"
    assert any(float((layer.metadata or {}).get("curiosity_signal", 0.0)) > 0.3 for layer in garden.layers)

    assert museum.metadata["house_room_preset"] == "house_museum"
    assert museum.metadata["scene_layout"] == "horizontal_strip"
    assert any(bool((layer.metadata or {}).get("contrastive_lesson", False)) for layer in museum.layers)

    assert tour.metadata["house_room_preset"] == "house_tour"
    assert "house_library" in tour.metadata["tour_rooms"]
    assert "house_garden" in tour.metadata["tour_rooms"]
    assert "house_museum" in tour.metadata["tour_rooms"]
