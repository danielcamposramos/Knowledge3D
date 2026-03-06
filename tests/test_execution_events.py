from __future__ import annotations

import json

import pytest

from knowledge3d.cranium.bridges.procedural_material_bridge import SurfaceMaterialCandidate
from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.tool_execution import ToolExecutionResolver


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_execution_events_record_three_distinct_tool_routes(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_events")

    cool = SurfaceMaterialCandidate(
        material_id="cool",
        name="Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    signal_program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=kv.galaxy_manager.query(
            query_text="audio signal surface material fusion",
            specialist="audio",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="audio",
    )
    kv.trm_navigator.execute(
        signal_program,
        input_data={
            "clip_id": "signal_exec_event",
            "audio_signal": TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
            "material_candidates": (cool, warm),
            "negative_materials": (warm,),
            "frame_size": 256,
            "threshold": 0.15,
            "displacement_gain": 0.4,
            "preview_size": 32,
        },
    )

    contour_program = kv.trm_navigator.compose(
        query="contour textured mesh surface material projection",
        patterns=kv.galaxy_manager.query(
            query_text="contour textured mesh surface material projection",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1
    kv.trm_navigator.execute(
        contour_program,
        input_data={
            "drawing_contour": grid,
            "surface_material": cool,
            "material_candidates": (cool, warm),
            "color": 1,
            "preview_size": 32,
        },
    )

    replay_program = kv.trm_navigator.compose(
        query="house replay journal scene playback",
        patterns=kv.galaxy_manager.query(
            query_text="house replay journal scene playback",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )
    kv.trm_navigator.execute(
        replay_program,
        input_data={
            "replay_entries": [
                {"timestamp": 1, "action_type": "NAV_MOVE", "quality_signal": 0.82, "ternary_quality": 1, "final_confidence": 0.82},
                {"timestamp": 2, "action_type": "DIALOGUE", "quality_signal": 0.74, "ternary_quality": 1, "final_confidence": 0.74},
                {"timestamp": 3, "action_type": "WRITE_MEM", "quality_signal": 0.88, "ternary_quality": 1, "final_confidence": 0.88},
            ],
            "frame_count": 4,
            "scene_layout": "golden_orbit",
        },
    )

    event_path = tmp_path / "kv_exec_events" / "logs" / "execution_events.jsonl"
    ToolExecutionResolver.clear_caches()
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) >= 3
    tool_ids = {row["tool_id"] for row in rows}
    assert "tool_fusion_signal_surface_material_v1" in tool_ids
    assert "tool_fusion_surface_material_projection_v1" in tool_ids
    assert "tool_house_replay_scene_v1" in tool_ids
    for row in rows[-3:]:
        assert row["outcome"] in (-1, 0, 1)
        assert row["ternary_quality"] in (-1, 0, 1)
        assert 0.0 <= float(row["quality_signal"]) <= 1.0
        assert int(row["timestamp_us"]) > 0
        assert int(row["chain_depth"]) >= 1
        assert isinstance(row.get("chain_tool_ids", []), list)
        assert isinstance(row.get("chain_runtime_statuses", []), list)


@pytest.mark.cuda
def test_execution_events_reach_phase_2d_density_target(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_density")

    cool = SurfaceMaterialCandidate(
        material_id="cool",
        name="Cool",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )

    signal_program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=kv.galaxy_manager.query(
            query_text="audio signal surface material fusion",
            specialist="audio",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="audio",
    )
    contour_program = kv.trm_navigator.compose(
        query="contour textured mesh surface material projection",
        patterns=kv.galaxy_manager.query(
            query_text="contour textured mesh surface material projection",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )
    library_program = kv.trm_navigator.compose(
        query="knowledge library settled scene playback",
        patterns=kv.galaxy_manager.query(
            query_text="knowledge library settled scene playback",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )
    garden_program = kv.trm_navigator.compose(
        query="learning growing garden scene playback",
        patterns=kv.galaxy_manager.query(
            query_text="learning growing garden scene playback",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )
    museum_program = kv.trm_navigator.compose(
        query="history archive failures lessons museum scene playback",
        patterns=kv.galaxy_manager.query(
            query_text="history archive failures lessons museum scene playback",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )
    tour_program = kv.trm_navigator.compose(
        query="house tour overview all scene playback",
        patterns=kv.galaxy_manager.query(
            query_text="house tour overview all scene playback",
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        ),
        specialist="visual",
    )

    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    room_events = [
        {
            "tool_id": "tool_house_library_scene_v1",
            "query_context": "what I know",
            "quality_signal": 0.91,
            "ternary_quality": 1,
            "outcome": 1,
            "timestamp_us": 10,
            "chain_tool_ids": ["tool_geom_profile_lathe_mesh_v1", "tool_fusion_surface_material_projection_v1"],
        },
        {
            "tool_id": "tool_house_garden_scene_v1",
            "query_context": "what I am learning",
            "quality_signal": 0.56,
            "ternary_quality": 0,
            "outcome": 0,
            "timestamp_us": 20,
            "chain_tool_ids": ["tool_signal_audio_spectrogram_v1", "tool_signal_spectrogram_surface_v1"],
        },
        {
            "tool_id": "tool_house_museum_scene_v1",
            "query_context": "my history",
            "quality_signal": 0.18,
            "ternary_quality": -1,
            "outcome": -1,
            "timestamp_us": 30,
            "chain_tool_ids": ["tool_fusion_surface_material_projection_v1"],
        },
    ]

    for idx in range(8):
        kv.trm_navigator.execute(
            signal_program,
            input_data={
                "clip_id": f"signal_density_{idx}",
                "audio_signal": TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
                "material_candidates": (cool, warm),
                "negative_materials": (warm,),
                "frame_size": 256,
                "threshold": 0.15,
                "displacement_gain": 0.4,
                "preview_size": 32,
            },
        )
        kv.trm_navigator.execute(
            contour_program,
            input_data={
                "drawing_contour": grid,
                "surface_material": cool,
                "material_candidates": (cool, warm),
                "color": 1,
                "preview_size": 32,
            },
        )
        kv.trm_navigator.execute(
            library_program,
            input_data={"execution_events": room_events, "max_events": 3},
        )
        kv.trm_navigator.execute(
            garden_program,
            input_data={"execution_events": room_events, "max_events": 3},
        )
        kv.trm_navigator.execute(
            museum_program,
            input_data={"execution_events": room_events, "max_events": 3},
        )
        kv.trm_navigator.execute(
            tour_program,
            input_data={"execution_events": room_events, "max_events_per_room": 3},
        )

    event_path = tmp_path / "kv_exec_density" / "logs" / "execution_events.jsonl"
    ToolExecutionResolver.clear_caches()
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 100
    assert any(str(row.get("tool_id", "")) == "tool_house_tour_scene_v1" for row in rows)
    assert any(str(row.get("execution_mode", "")) == "tool_chain_step" for row in rows)
