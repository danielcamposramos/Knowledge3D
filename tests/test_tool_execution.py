from __future__ import annotations

import pytest

from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.tool_execution import ToolExecutionResolver
from knowledge3d.cranium.bridges.procedural_material_bridge import SurfaceMaterialCandidate
from knowledge3d.cranium.ternary import TernaryVector


class _RouteDispatchFixtures:
    def kernel_callable(self, value: int) -> dict[str, int | str]:
        return {"selected_route": "kernel", "value": int(value)}

    def bridge_callable(self, value: int) -> dict[str, int | str]:
        return {"selected_route": "bridge", "value": int(value)}


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


def test_execution_plan_blueprint_resolves_primary_signal_material_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_blueprint")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )

    execution_plan = composed.get("execution_plan")
    blueprint = kv.trm_navigator.resolve_execution_plan(execution_plan)

    assert blueprint is not None
    assert blueprint["primary_tool_id"] == "tool_fusion_signal_surface_material_v1"
    assert blueprint["primary_entrypoint"].owner_name == "ProceduralMaterialBridge"
    assert blueprint["primary_entrypoint"].callable_name == "signal_to_textured_surface"
    assert blueprint["primary_argument_schema"]["required_kwargs"][0]["name"] == "candidates"


@pytest.mark.cuda
def test_execution_plan_primary_entrypoint_is_instantiable(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_instantiate")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
    blueprint = kv.trm_navigator.resolve_execution_plan(composed.get("execution_plan"))

    bound = ToolExecutionResolver.instantiate_entrypoint(blueprint["primary_entrypoint"])
    assert callable(bound)
    assert getattr(bound, "__name__", "") == "signal_to_textured_surface"


@pytest.mark.cuda
def test_execution_plan_primary_entrypoint_is_invokable(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_invoke")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )

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
    samples = TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)])

    plan = kv.trm_navigator.invoke_execution_plan(
        composed.get("execution_plan"),
        "signal_exec",
        samples,
        candidates=(cool, warm),
        negative_materials=(warm,),
        frame_size=256,
        threshold=0.15,
        displacement_gain=0.5,
        preview_size=32,
    )

    assert plan.selected_material.material_id == "cool"
    assert plan.vertex_rgba.shape[1] == 4
    assert plan.metadata["signal_projection_summary"]["frame_count"] == 4


@pytest.mark.cuda
def test_execution_plan_primary_entrypoint_is_invokable_from_payload(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_payload")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )

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
    samples = TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)])

    payload = {
        "clip_id": "signal_exec_payload",
        "samples": samples,
        "candidates": (cool, warm),
        "negative_materials": (warm,),
        "frame_size": 256,
        "threshold": 0.15,
        "displacement_gain": 0.5,
        "preview_size": 32,
    }
    plan = kv.trm_navigator.invoke_execution_plan_from_payload(composed.get("execution_plan"), payload)

    assert plan.selected_material.material_id == "cool"
    assert plan.vertex_rgba.shape[1] == 4
    assert plan.metadata["signal_projection_summary"]["frame_count"] == 4


@pytest.mark.cuda
def test_execution_plan_rejects_missing_required_keyword_arguments(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_validate")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
    samples = TernaryVector([0] * 1024)

    with pytest.raises(ValueError, match="missing required keyword arguments: candidates"):
        kv.trm_navigator.invoke_execution_plan(
            composed.get("execution_plan"),
            "signal_exec",
            samples,
            frame_size=256,
        )


def test_execution_plan_rejects_missing_payload_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_payload_validate")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )

    with pytest.raises(ValueError, match="missing payload field for required keyword argument: candidates"):
        ToolExecutionResolver.bind_primary_arguments(
            composed.get("execution_plan"),
            {"clip_id": "signal_exec_payload", "samples": TernaryVector([0] * 1024)},
        )


def test_execution_plan_binds_semantic_payload_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_alias_bind")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    args, kwargs = ToolExecutionResolver.bind_primary_arguments(
        composed.get("execution_plan"),
        {
            "clip_id": "signal_exec_alias",
            "audio_signal": TernaryVector([0] * 1024),
            "material_candidates": (cool, warm),
            "preview_size": 32,
        },
    )

    assert args[0] == "signal_exec_alias"
    assert isinstance(args[1], TernaryVector)
    assert kwargs["candidates"][0].material_id == "cool"
    assert kwargs["preview_size"] == 32


def test_execution_plan_binds_semantic_payload_with_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_alias_defaults")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    args, kwargs = ToolExecutionResolver.bind_primary_arguments(
        composed.get("execution_plan"),
        {
            "clip_id": "signal_exec_alias_defaults",
            "audio_signal": TernaryVector([0] * 1024),
            "material_candidates": (cool, warm),
        },
    )

    assert args[0] == "signal_exec_alias_defaults"
    assert kwargs["frame_size"] == 1024
    assert kwargs["threshold"] == 0.2
    assert kwargs["displacement_gain"] == 0.25
    assert kwargs["preview_size"] == 64
    assert kwargs["negative_materials"] == []


def test_execution_plan_selects_contour_entrypoint_from_semantic_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_select")
    rows = kv.galaxy_manager.query(
        query_text="contour textured mesh surface material projection",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="contour textured mesh surface material projection",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    selected = ToolExecutionResolver.select_entrypoint_for_payload(
        composed.get("execution_plan"),
        {
            "drawing_contour": grid,
            "surface_material": target,
            "material_candidates": (target, warm),
            "color": 1,
            "preview_size": 32,
        },
    )

    assert selected["tool_id"] == "tool_fusion_surface_material_projection_v1"
    assert selected["blueprint"].callable_name == "contour_to_textured_lathe_mesh"


def test_execution_plan_selects_audio_surface_chain_from_semantic_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_chain_select")
    rows = kv.galaxy_manager.query(
        query_text="audio spectrogram surface displacement",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio spectrogram surface displacement",
        patterns=rows,
        specialist="audio",
    )
    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        composed.get("execution_plan"),
        {
            "clip_id": "signal_chain_select",
            "audio_signal": TernaryVector([0] * 1024),
        },
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_signal_spectrogram_surface_v1"
    assert selected["preset_name"] == "audio_surface_chain"


def test_execution_plan_selects_signal_timeline_chain_with_named_preset(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_timeline_preset")
    rows = kv.galaxy_manager.query(
        query_text="audio temporal video timeline animation",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio temporal video timeline animation",
        patterns=rows,
        specialist="audio",
    )
    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        composed.get("execution_plan"),
        {
            "clip_id": "signal_timeline_select",
            "audio_signal": TernaryVector([0] * 1024),
            "material_candidates": (),
            "timeline_preset": "world_orbit",
        },
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_fusion_signal_surface_material_timeline_v1"
    assert selected["preset_name"] == "signal_timeline_chain"


def test_execution_plan_prefers_ui_animation_tool_for_ui_query(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_ui_route")
    rows = kv.galaxy_manager.query(
        query_text="ui overlay panel contour animation",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="ui overlay panel contour animation",
        patterns=rows,
        specialist="visual",
    )
    blueprint = kv.trm_navigator.resolve_execution_plan(composed.get("execution_plan"))

    assert blueprint is not None
    assert blueprint["primary_tool_id"] == "tool_fusion_surface_material_ui_animation_v1"


def test_execution_plan_prefers_world_animation_tool_for_world_query(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_world_route")
    rows = kv.galaxy_manager.query(
        query_text="world ambient audio animation",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="world ambient audio animation",
        patterns=rows,
        specialist="audio",
    )
    blueprint = kv.trm_navigator.resolve_execution_plan(composed.get("execution_plan"))

    assert blueprint is not None
    assert blueprint["primary_tool_id"] == "tool_fusion_signal_surface_material_world_animation_v1"


def test_route_aware_entrypoint_selection_prefers_healthy_bridge_over_bad_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_route_dispatch")

    for idx in range(3):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_kernel_bad",
                "query_context": "route aware dispatch",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 3,
                "execution_us": 1200 + idx,
                "outcome": -1,
                "quality_signal": 0.12,
                "ternary_quality": -1,
                "timestamp_us": 10 + idx,
                "chain_depth": 1,
                "promotion_pressure": True,
                "runtime_status": "ptx_rpn_available",
                "tool_kind": "scene_fusion",
            }
        )
    for idx in range(2):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_bridge_ok",
                "query_context": "route aware dispatch",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 2,
                "execution_us": 1500 + idx,
                "outcome": 1,
                "quality_signal": 0.82,
                "ternary_quality": 1,
                "timestamp_us": 20 + idx,
                "chain_depth": 1,
                "promotion_pressure": False,
                "runtime_status": "ptx_bridge_available",
                "tool_kind": "scene_fusion",
            }
        )

    schema = {
        "positional": [{"name": "value"}],
        "required_kwargs": [],
        "optional_kwargs": [],
        "strict_kwargs": True,
    }
    execution_plan = {
        "mode": "tool_entrypoint_chain",
        "primary_tool_id": "tool_kernel_bad",
        "primary_entrypoint": "tests.test_tool_execution._RouteDispatchFixtures.kernel_callable",
        "primary_argument_schema": schema,
        "execution_chain": [
            {
                "tool_id": "tool_kernel_bad",
                "tool_kind": "scene_fusion",
                "runtime_status": "ptx_rpn_available",
                "inputs": ["value"],
                "outputs": ["selected_route"],
                "entrypoints": ["tests.test_tool_execution._RouteDispatchFixtures.kernel_callable"],
                "argument_schemas": {
                    "tests.test_tool_execution._RouteDispatchFixtures.kernel_callable": schema,
                },
                "chain_presets": {},
            },
            {
                "tool_id": "tool_bridge_ok",
                "tool_kind": "scene_fusion",
                "runtime_status": "ptx_bridge_available",
                "inputs": ["value"],
                "outputs": ["selected_route"],
                "entrypoints": ["tests.test_tool_execution._RouteDispatchFixtures.bridge_callable"],
                "argument_schemas": {
                    "tests.test_tool_execution._RouteDispatchFixtures.bridge_callable": schema,
                },
                "chain_presets": {},
            },
        ],
    }

    selected = ToolExecutionResolver.select_entrypoint_for_payload(
        execution_plan,
        {"value": 7},
        quality_tracker=kv.trm_navigator.execution_quality_tracker,
    )

    assert selected["tool_id"] == "tool_bridge_ok"
    assert selected["blueprint"].callable_name == "bridge_callable"
    assert selected["route_source"] == "bridge"

    result = kv.trm_navigator.invoke_execution_plan_from_payload(
        execution_plan,
        {"value": 7},
        query_context="route aware dispatch",
        specialist_id="VisualSpecialist",
    )
    assert result["selected_route"] == "bridge"


def test_route_aware_chain_selection_prefers_healthy_bridge_chain_over_bad_kernel(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_route_chain_dispatch")

    for idx in range(3):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_kernel_bad",
                "query_context": "route aware chain dispatch",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 3,
                "execution_us": 1300 + idx,
                "outcome": -1,
                "quality_signal": 0.1,
                "ternary_quality": -1,
                "timestamp_us": 30 + idx,
                "chain_depth": 1,
                "promotion_pressure": True,
                "runtime_status": "ptx_rpn_available",
                "tool_kind": "scene_fusion",
            }
        )
    for idx in range(2):
        kv.trm_navigator.observe_execution_event(
            {
                "tool_id": "tool_bridge_ok",
                "query_context": "route aware chain dispatch",
                "specialist_id": "VisualSpecialist",
                "math_core_tier": 2,
                "execution_us": 1400 + idx,
                "outcome": 1,
                "quality_signal": 0.84,
                "ternary_quality": 1,
                "timestamp_us": 40 + idx,
                "chain_depth": 1,
                "promotion_pressure": False,
                "runtime_status": "ptx_bridge_available",
                "tool_kind": "scene_fusion",
            }
        )

    schema = {
        "positional": [{"name": "value"}],
        "required_kwargs": [],
        "optional_kwargs": [],
        "strict_kwargs": True,
    }
    execution_plan = {
        "mode": "tool_entrypoint_chain",
        "primary_tool_id": "tool_kernel_bad",
        "primary_entrypoint": "tests.test_tool_execution._RouteDispatchFixtures.kernel_callable",
        "primary_argument_schema": schema,
        "chain_presets": {
            "kernel_chain": {
                "required_inputs": ["value"],
                "return_alias": "route_result",
                "steps": [
                    {
                        "entrypoint": "tests.test_tool_execution._RouteDispatchFixtures.kernel_callable",
                        "argument_schema": schema,
                        "store_as": ["route_result"],
                        "store_fields": {},
                    }
                ],
            }
        },
        "execution_chain": [
            {
                "tool_id": "tool_kernel_bad",
                "tool_kind": "scene_fusion",
                "runtime_status": "ptx_rpn_available",
                "inputs": ["value"],
                "outputs": ["selected_route"],
                "entrypoints": ["tests.test_tool_execution._RouteDispatchFixtures.kernel_callable"],
                "argument_schemas": {
                    "tests.test_tool_execution._RouteDispatchFixtures.kernel_callable": schema,
                },
                "chain_presets": {},
            },
            {
                "tool_id": "tool_bridge_ok",
                "tool_kind": "scene_fusion",
                "runtime_status": "ptx_bridge_available",
                "inputs": ["value"],
                "outputs": ["selected_route"],
                "entrypoints": ["tests.test_tool_execution._RouteDispatchFixtures.bridge_callable"],
                "argument_schemas": {
                    "tests.test_tool_execution._RouteDispatchFixtures.bridge_callable": schema,
                },
                "chain_presets": {
                    "bridge_chain": {
                        "required_inputs": ["value"],
                        "return_alias": "route_result",
                        "steps": [
                            {
                                "entrypoint": "tests.test_tool_execution._RouteDispatchFixtures.bridge_callable",
                                "argument_schema": schema,
                                "store_as": ["route_result"],
                                "store_fields": {},
                            }
                        ],
                    }
                },
            },
        ],
    }

    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        execution_plan,
        {"value": 9},
        quality_tracker=kv.trm_navigator.execution_quality_tracker,
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_bridge_ok"
    assert selected["preset_name"] == "bridge_chain"
    assert selected["route_source"] == "bridge"

    result = kv.trm_navigator.invoke_execution_plan_from_payload(
        execution_plan,
        {"value": 9},
        query_context="route aware chain dispatch",
        specialist_id="VisualSpecialist",
    )
    assert result["selected_route"] == "bridge"


def test_execution_plan_selects_contour_material_chain_from_semantic_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_chain_select")
    rows = kv.galaxy_manager.query(
        query_text="contour textured mesh surface material projection",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="contour textured mesh surface material projection",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        composed.get("execution_plan"),
        {
            "drawing_contour": [[0] * 16 for _ in range(16)],
            "surface_material": target,
            "material_candidates": (target, warm),
        },
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_fusion_surface_material_projection_v1"
    assert selected["preset_name"] == "contour_material_chain"


def test_execution_plan_selects_contour_extrude_material_chain_from_semantic_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_extrude_chain_select")
    rows = kv.galaxy_manager.query(
        query_text="contour extrude textured mesh surface material projection",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="contour extrude textured mesh surface material projection",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        composed.get("execution_plan"),
        {
            "drawing_contour": [[0] * 16 for _ in range(16)],
            "surface_material": target,
            "material_candidates": (target, warm),
            "geometry_mode": "extrude",
        },
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_fusion_surface_material_projection_v1"
    assert selected["preset_name"] == "contour_extrude_material_chain"


def test_execution_plan_selects_signal_surface_material_chain_from_semantic_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_material_chain_select")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        composed.get("execution_plan"),
        {
            "clip_id": "signal_chain_select",
            "audio_signal": TernaryVector([0] * 1024),
            "material_candidates": (cool, warm),
        },
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_fusion_signal_surface_material_v1"
    assert selected["preset_name"] == "signal_surface_material_chain"


def test_execution_plan_selects_signal_timeline_chain_from_semantic_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_timeline_chain_select")
    rows = kv.galaxy_manager.query(
        query_text="audio signal temporal video timeline material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio signal temporal video timeline material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    selected = ToolExecutionResolver.select_chain_preset_for_payload(
        composed.get("execution_plan"),
        {
            "clip_id": "signal_timeline_select",
            "audio_signal": TernaryVector([0] * 1024),
            "material_candidates": (cool, warm),
        },
    )

    assert selected is not None
    assert selected["tool_id"] == "tool_fusion_signal_surface_material_timeline_v1"
    assert selected["preset_name"] == "signal_timeline_chain"


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_to_tool_entrypoint_chain(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_auto")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    payload = {
        "clip_id": "signal_exec_auto",
        "samples": TernaryVector([(-1 if i % 5 == 0 else (1 if i % 2 == 0 else 0)) for i in range(1024)]),
        "candidates": (cool, warm),
        "negative_materials": (warm,),
        "frame_size": 256,
        "threshold": 0.15,
        "displacement_gain": 0.4,
        "preview_size": 32,
    }

    result = kv.trm_navigator.execute(program, input_data=payload)

    assert result.selected_material.material_id == "cool"
    assert result.vertex_rgba.shape[1] == 4


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_with_semantic_alias_payload(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_auto_alias")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    payload = {
        "clip_id": "signal_exec_auto_alias",
        "audio_signal": TernaryVector([(-1 if i % 4 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
        "material_candidates": (cool, warm),
        "negative_materials": (warm,),
        "frame_size": 256,
        "threshold": 0.15,
        "displacement_gain": 0.4,
        "preview_size": 32,
    }

    result = kv.trm_navigator.execute(program, input_data=payload)

    assert result.selected_material.material_id == "cool"
    assert result.vertex_rgba.shape[1] == 4


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_contour_semantic_payload(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_auto")
    rows = kv.galaxy_manager.query(
        query_text="contour textured mesh surface material projection",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="contour textured mesh surface material projection",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "drawing_contour": grid,
            "surface_material": target,
            "material_candidates": (target, warm),
            "negative_materials": (warm,),
            "color": 1,
            "preview_size": 32,
            "pad": 1,
            "segments": 12,
        },
    )

    assert result.selected_material.material_id == "cool_target"
    assert result.mesh.vertices.shape[1] == 3
    assert result.vertex_rgba.shape[1] == 4


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_contour_extrude_chain(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_extrude_auto")
    rows = kv.galaxy_manager.query(
        query_text="contour extrude textured mesh surface material projection",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="contour extrude textured mesh surface material projection",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "drawing_contour": grid,
            "surface_material": target,
            "material_candidates": (target, warm),
            "geometry_mode": "extrude",
            "negative_materials": (warm,),
            "color": 1,
            "preview_size": 32,
            "pad": 1,
            "depth_scale": 0.4,
        },
    )

    assert result.selected_material.material_id == "cool_target"
    assert result.mesh.metadata["mesh_kind"] == "extrude"
    assert result.vertex_rgba.shape[1] == 4


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_with_minimal_semantic_signal_payload(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_auto_signal_min")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    result = kv.trm_navigator.execute(
        program,
        input_data={
            "clip_id": "signal_exec_minimal",
            "audio_signal": TernaryVector([(-1 if i % 4 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
            "material_candidates": (cool, warm),
        },
    )

    assert result.selected_material.material_id == "cool"
    assert result.metadata["signal_frame_size"] == 1024
    assert result.metadata["signal_threshold"] == 0.2


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_signal_chain_with_generated_target_material(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_generated_target")
    rows = kv.galaxy_manager.query(
        query_text="audio signal surface material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio signal surface material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    result = kv.trm_navigator.execute(
        program,
        input_data={
            "clip_id": "signal_exec_generated_target",
            "audio_signal": TernaryVector([(-1 if i % 4 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
            "material_candidates": (cool, warm),
            "frame_size": 256,
            "threshold": 0.15,
            "displacement_gain": 0.4,
            "preview_size": 32,
        },
    )

    assert result.selected_material.material_id == "cool"
    assert result.metadata["signal_projection_summary"]["frame_count"] == 4
    assert result.metadata["signal_frame_size"] == 256
    assert result.vertex_rgba.shape[1] == 4


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_signal_timeline_chain(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_timeline")
    rows = kv.galaxy_manager.query(
        query_text="audio signal temporal video timeline material fusion",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio signal temporal video timeline material fusion",
        patterns=rows,
        specialist="audio",
    )
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
    result = kv.trm_navigator.execute(
        program,
        input_data={
            "clip_id": "signal_exec_timeline",
            "audio_signal": TernaryVector([(-1 if i % 4 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
            "material_candidates": (cool, warm),
            "frame_size": 256,
            "threshold": 0.15,
            "displacement_gain": 0.4,
            "preview_size": 32,
            "frame_count": 4,
            "time_span": 0.3,
        },
    )

    assert result.frames.shape == (4, 32, 32, 3)
    assert result.surface_plan.selected_material.material_id == "cool"
    assert result.metadata["frame_count"] == 4
    assert "overall_coherence" in result.metadata


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_contour_timeline_chain(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_timeline")
    rows = kv.galaxy_manager.query(
        query_text="contour temporal video timeline textured mesh surface material",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="contour temporal video timeline textured mesh surface material",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "drawing_contour": grid,
            "surface_material": target,
            "material_candidates": (target, warm),
            "geometry_mode": "sweep",
            "negative_materials": (warm,),
            "color": 1,
            "preview_size": 32,
            "pad": 1,
            "depth_scale": 0.4,
            "frame_count": 3,
            "time_span": 0.25,
        },
    )

    assert result.frames.shape[0] == 3
    assert result.surface_plan.mesh.metadata["mesh_kind"] == "sweep"
    assert result.surface_plan.selected_material.material_id == "cool_target"


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_signal_timeline_chain_with_named_preset(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_timeline_named")
    rows = kv.galaxy_manager.query(
        query_text="audio temporal video timeline animation",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio temporal video timeline animation",
        patterns=rows,
        specialist="audio",
    )
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
    result = kv.trm_navigator.execute(
        program,
        input_data={
            "clip_id": "signal_timeline_named",
            "audio_signal": TernaryVector([(-1 if i % 5 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
            "material_candidates": (cool, warm),
            "timeline_preset": "world_orbit",
            "preview_size": 32,
        },
    )

    assert result.frames.shape[0] == 12
    assert result.metadata["timeline_preset"] == "world_orbit"
    assert result.metadata["timeline_domain"] == "world"


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_ui_animation_tool_defaults(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_ui_defaults")
    rows = kv.galaxy_manager.query(
        query_text="ui overlay panel contour animation",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="ui overlay panel contour animation",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "drawing_contour": grid,
            "surface_material": target,
            "material_candidates": (target, warm),
            "preview_size": 32,
        },
    )

    assert result.metadata["timeline_domain"] == "ui"
    assert result.metadata["timeline_preset"] == "ui_idle"
    assert result.frames.shape[0] == 6


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_world_animation_tool_defaults(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_world_defaults")
    rows = kv.galaxy_manager.query(
        query_text="world ambient audio animation",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="world ambient audio animation",
        patterns=rows,
        specialist="audio",
    )
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

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "clip_id": "signal_world_defaults",
            "audio_signal": TernaryVector([(-1 if i % 5 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
            "material_candidates": (cool, warm),
            "preview_size": 32,
        },
    )

    assert result.metadata["timeline_domain"] == "world"
    assert result.metadata["timeline_preset"] == "world_breathe"
    assert result.frames.shape[0] == 10


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_ui_scene_tool_defaults(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    from knowledge3d.cranium.bridges.procedural_material_bridge import ProceduralMaterialBridge

    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_ui_scene_defaults")
    rows = kv.galaxy_manager.query(
        query_text="ui hud layered scene playback overlay",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="ui hud layered scene playback overlay",
        patterns=rows,
        specialist="visual",
    )
    assert program.get("tool_context", {}).get("primary_tool_id") == "tool_fusion_surface_material_ui_scene_v1"

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
    grid_a = [[0] * 24 for _ in range(24)]
    grid_b = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid_a[y][x] = 1
    for y in range(6, 18):
        for x in range(10, 16):
            grid_b[y][x] = 1
    surface_a = material_bridge.contour_to_textured_lathe_mesh(
        grid_a,
        color=1,
        pad=1,
        segments=12,
        target_material=cool,
        candidates=(cool, warm),
        negative_materials=(warm,),
        preview_size=32,
    )
    surface_b = material_bridge.contour_to_textured_sweep_mesh(
        grid_b,
        color=1,
        pad=1,
        depth_scale=0.4,
        width_scale=1.0,
        height_scale=1.0,
        target_material=warm,
        candidates=(cool, warm),
        negative_materials=(cool,),
        preview_size=32,
    )

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "textured_surfaces": (surface_a, surface_b),
        },
    )

    assert result.metadata["scene_domain"] == "ui"
    assert result.metadata["scene_layout"] == "overlay"
    assert result.metadata["layer_count"] == 2
    assert result.frames.shape == (6, 32, 32, 4)


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_replay_scene_tool(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_replay_scene")
    rows = kv.galaxy_manager.query(
        query_text="house replay journal scene playback",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="house replay journal scene playback",
        patterns=rows,
        specialist="visual",
    )
    assert program.get("tool_context", {}).get("primary_tool_id") == "tool_house_replay_scene_v1"

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "replay_entries": (
                {"timestamp": 1, "action_type": "NAV_MOVE", "raw_confidence": 0.6, "final_confidence": 0.72, "curiosity": 0.1},
                {"timestamp": 2, "action_type": "DIALOGUE", "raw_confidence": 0.7, "final_confidence": 0.82, "curiosity": 0.2},
                {"timestamp": 3, "action_type": "WRITE_MEM", "raw_confidence": 0.8, "final_confidence": 0.9, "curiosity": 0.3},
            ),
            "frame_count": 5,
        },
    )

    assert result.metadata["replay_source"] == "journal"
    assert result.metadata["scene_layout"] == "golden_orbit"
    assert result.metadata["layer_count"] == 3
    assert result.frames.shape[0] == 5
    assert result.frames.shape[3] == 4


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_library_garden_museum_and_tour_scene_tools(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_house_rooms")

    room_events = (
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
    )

    cases = [
        ("knowledge library settled scene playback", "tool_house_library_scene_v1", "house_library"),
        ("learning growing garden scene playback", "tool_house_garden_scene_v1", "house_garden"),
        ("history archive failures lessons museum scene playback", "tool_house_museum_scene_v1", "house_museum"),
        ("house tour overview all scene playback", "tool_house_tour_scene_v1", "house_tour"),
    ]

    for query_text, expected_tool_id, expected_room in cases:
        rows = kv.galaxy_manager.query(
            query_text=query_text,
            specialist="visual",
            top_k=8,
            galaxies=["Tool"],
        )
        program = kv.trm_navigator.compose(
            query=query_text,
            patterns=rows,
            specialist="visual",
        )
        assert program.get("tool_context", {}).get("primary_tool_id") == expected_tool_id

        payload = {"execution_events": room_events}
        if expected_room == "house_tour":
            payload["max_events_per_room"] = 3
        else:
            payload["max_events"] = 3
        result = kv.trm_navigator.execute(program, input_data=payload)
        assert result.metadata["house_room_preset"] == expected_room


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_contour_minimal_semantic_payload(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_contour_minimal")
    rows = kv.galaxy_manager.query(
        query_text="contour textured mesh surface material projection",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="contour textured mesh surface material projection",
        patterns=rows,
        specialist="visual",
    )
    target = SurfaceMaterialCandidate(
        material_id="cool_target",
        name="Cool Target",
        palette=((0.08, 0.16, 0.62, 1.0), (0.4, 0.62, 0.95, 1.0), (0.92, 0.98, 1.0, 1.0)),
    )
    warm = SurfaceMaterialCandidate(
        material_id="warm",
        name="Warm",
        palette=((0.7, 0.18, 0.08, 1.0), (0.9, 0.48, 0.2, 1.0), (1.0, 0.9, 0.7, 1.0)),
    )
    grid = [[0] * 24 for _ in range(24)]
    for y in range(5, 19):
        for x in range(8, 13):
            grid[y][x] = 1

    result = kv.trm_navigator.execute(
        program,
        input_data={
            "drawing_contour": grid,
            "surface_material": target,
            "material_candidates": (target, warm),
        },
    )

    assert result.selected_material.material_id == "cool_target"
    assert result.metadata["preview_size"] == 64
    assert result.mesh.vertices.shape[1] == 3


@pytest.mark.cuda
def test_trm_execute_auto_dispatches_audio_surface_chain(tmp_path, monkeypatch):
    _require_gpu()
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_exec_signal_chain")
    rows = kv.galaxy_manager.query(
        query_text="audio spectrogram surface displacement",
        specialist="audio",
        top_k=8,
        galaxies=["Tool"],
    )
    program = kv.trm_navigator.compose(
        query="audio spectrogram surface displacement",
        patterns=rows,
        specialist="audio",
    )
    result = kv.trm_navigator.execute(
        program,
        input_data={
            "clip_id": "signal_surface_chain",
            "audio_signal": TernaryVector([(-1 if i % 4 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)]),
        },
    )

    assert result.vertices.shape[1] == 3
    assert result.indices.shape[1] == 3
    assert "signal_projection_summary" in result.metadata
    assert result.metadata["signal_projection_summary"]["frame_count"] == 1
