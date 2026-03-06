from __future__ import annotations

from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.tool_galaxy import (
    bootstrap_tool_galaxy,
    build_tool_payload,
    default_tool_entries,
)


def test_default_tool_entries_unify_codec_and_fusion_tools():
    entries = default_tool_entries()
    assert len(entries) >= 37

    by_id = {str(entry.get("id")): entry for entry in entries}
    math_tier1 = by_id["tool_mathcore_tier1_scalar_worker_worker_v1"]
    math_tier2 = by_id["tool_mathcore_tier2_vector_worker_v1"]
    math_tier3 = by_id["tool_mathcore_tier3_master_v1"]
    math_allocator = by_id["tool_mathcore_spawn_cascade_v1"]
    contour = by_id["tool_fusion_contour_to_mesh_v1"]
    spectrogram = by_id["tool_signal_audio_spectrogram_v1"]
    signal_material = by_id["tool_fusion_signal_surface_material_v1"]
    codec = by_id["tool_codec_audio_mdct_v1"]
    painter = by_id["tool_paint_filter_stack_v1"]
    palette_contrastive = by_id["tool_paint_palette_contrastive_v1"]
    gradient_contrastive = by_id["tool_paint_gradient_contrastive_v1"]
    gradient_cascade = by_id["tool_paint_gradient_cascade_v1"]
    geom_bbox = by_id["tool_geom_bbox_crop_v1"]
    geom_profile = by_id["tool_geom_profile_prep_v1"]
    geom_lathe = by_id["tool_geom_profile_lathe_mesh_v1"]
    geom_extrude = by_id["tool_geom_profile_extrude_mesh_v1"]
    geom_sweep = by_id["tool_geom_profile_sweep_mesh_v1"]
    temporal_preview = by_id["tool_video_temporal_preview_v1"]
    surface_timeline = by_id["tool_fusion_surface_material_timeline_v1"]
    signal_timeline = by_id["tool_fusion_signal_surface_material_timeline_v1"]
    surface_ui_animation = by_id["tool_fusion_surface_material_ui_animation_v1"]
    surface_world_animation = by_id["tool_fusion_surface_material_world_animation_v1"]
    signal_ui_animation = by_id["tool_fusion_signal_surface_material_ui_animation_v1"]
    signal_world_animation = by_id["tool_fusion_signal_surface_material_world_animation_v1"]
    temporal_scene = by_id["tool_video_temporal_scene_v1"]
    ui_scene = by_id["tool_fusion_surface_material_ui_scene_v1"]
    world_scene = by_id["tool_fusion_surface_material_world_scene_v1"]
    replay_scene = by_id["tool_house_replay_scene_v1"]
    library_scene = by_id["tool_house_library_scene_v1"]
    garden_scene = by_id["tool_house_garden_scene_v1"]
    museum_scene = by_id["tool_house_museum_scene_v1"]
    tour_scene = by_id["tool_house_tour_scene_v1"]

    assert math_tier1["metadata"]["math_core"]["preferred_tier"] == 1
    assert math_tier1["metadata"]["math_core"]["tier_role"] == "worker_worker"
    assert math_tier2["metadata"]["math_core"]["preferred_tier"] == 2
    assert math_tier2["metadata"]["math_core"]["tier_role"] == "worker"
    assert math_tier3["metadata"]["math_core"]["preferred_tier"] == 3
    assert math_tier3["metadata"]["math_core"]["tier_role"] == "master"
    assert "MathCorePool.snapshot" in " ".join(math_allocator["metadata"]["entrypoints"])
    assert math_allocator["metadata"]["memory_residency"] == "knowledgeverse_galaxy"
    assert math_allocator["metadata"]["execution_residency"] == "gpu_ptx"

    assert contour["type"] == "tool_node"
    assert contour["metadata"]["promotion_stage"] == "recipe"
    assert "glyph_curve_transfer" in contour["drawing_refs"]
    assert "obj3d_mesh_compute_normal" in contour["object_refs"]
    assert "tool_geom_profile_lathe_mesh_v1" in contour["rpn_program"]
    assert "tool_geom_profile_extrude_mesh_v1" in contour["rpn_program"]
    assert "tool_geom_profile_sweep_mesh_v1" in contour["rpn_program"]
    assert "TERNARY_QUANT" in contour["metadata"]["codec_ops"]
    assert contour["metadata"]["math_core"]["preferred_tier"] == 3
    assert contour["metadata"]["math_core"]["cascade"] == [
        "parallel_fanout",
        "worker_reduce",
        "master_commit",
    ]

    assert set(spectrogram["metadata"]["modalities"]) == {"audio", "drawing", "signal"}
    assert "curve_to_waveform_map" in spectrogram["drawing_refs"]
    assert "MDCT" in spectrogram["rpn_program"]
    assert "BLOCKS_TO_GRID" in spectrogram["metadata"]["codec_ops"]
    assert spectrogram["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert spectrogram["metadata"]["math_core"]["preferred_tier"] == 2
    assert "ProceduralSignalBridge.audio_to_spectrogram" in " ".join(spectrogram["metadata"]["entrypoints"])
    assert "ProceduralSignalBridge.audio_to_spectrogram_configured" in " ".join(
        spectrogram["metadata"]["entrypoints"]
    )
    assert "entrypoint_argument_schemas" in spectrogram["metadata"]
    assert (
        spectrogram["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.audio_to_spectrogram"
        ]["positional"][1]["aliases"][0]
        == "audio_signal"
    )
    assert (
        spectrogram["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.audio_to_spectrogram_configured"
        ]["optional_kwargs"][0]["default"]
        == 1024
    )
    assert "verified_by" in spectrogram["metadata"]
    assert by_id["tool_signal_spectrogram_surface_v1"]["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "ProceduralSignalBridge.spectrogram_to_surface" in " ".join(
        by_id["tool_signal_spectrogram_surface_v1"]["metadata"]["entrypoints"]
    )
    assert "audio_surface_chain" in by_id["tool_signal_spectrogram_surface_v1"]["metadata"]["execution_chain_presets"]
    assert signal_material["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "tool_signal_spectrogram_surface_v1" in signal_material["tool_refs"]
    assert "tool_fusion_surface_material_projection_v1" in signal_material["tool_refs"]
    assert "ProceduralMaterialBridge.signal_to_textured_surface" in " ".join(
        signal_material["metadata"]["entrypoints"]
    )
    assert "ProceduralMaterialBridge.signal_projection_to_material_target" in " ".join(
        signal_material["metadata"]["entrypoints"]
    )
    assert (
        signal_material["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface"
        ]["required_kwargs"][0]["aliases"][0]
        == "material_candidates"
    )
    assert (
        signal_material["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface"
        ]["optional_kwargs"][2]["default"]
        == 1024
    )
    assert "signal_surface_material_chain" in signal_material["metadata"]["execution_chain_presets"]
    assert "signal_surface_material_with_target_chain" in signal_material["metadata"]["execution_chain_presets"]
    assert signal_material["metadata"]["math_core"]["preferred_tier"] == 3
    assert temporal_preview["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "ProceduralTemporalBridge.surface_material_to_temporal_preview" in " ".join(
        temporal_preview["metadata"]["entrypoints"]
    )
    assert "ProceduralTemporalBridge.surface_material_to_timeline_preset" in " ".join(
        temporal_preview["metadata"]["entrypoints"]
    )
    assert (
        temporal_preview["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_temporal_preview"
        ]["optional_kwargs"][0]["default"]
        == 4
    )
    assert (
        temporal_preview["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset"
        ]["optional_kwargs"][0]["name"]
        == "timeline_preset"
    )
    assert temporal_preview["metadata"]["timeline_presets"] == [
        "ui_idle",
        "ui_focus",
        "world_breathe",
        "world_orbit",
    ]

    assert codec["metadata"]["promotion_stage"] == "kernel"
    assert codec["metadata"]["runtime_status"] == "ptx_rpn_available"
    assert "BATCH_MDCT" in codec["rpn_program"]
    assert "TERNARY_QUANT" in codec["metadata"]["codec_ops"]
    assert "verified_by" in codec["metadata"]
    assert codec["metadata"]["math_core"]["preferred_tier"] == 2

    assert painter["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "FILTER_BLUR" in painter["rpn_program"]
    assert "entrypoints" in painter["metadata"]
    assert palette_contrastive["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "TERNARY_QUANT" in palette_contrastive["metadata"]["codec_ops"]
    assert "contrastive_palette_score" in " ".join(palette_contrastive["metadata"]["entrypoints"])
    assert gradient_contrastive["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "TERNARY_QUANT" in gradient_contrastive["metadata"]["codec_ops"]
    assert gradient_cascade["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "linear_gradient_from_ternary_cascade" in " ".join(gradient_cascade["metadata"]["entrypoints"])

    assert geom_bbox["metadata"]["runtime_status"] == "ptx_runtime_available"
    assert "crop_gpu" in " ".join(geom_bbox["metadata"]["entrypoints"])
    assert geom_profile["metadata"]["runtime_status"] == "ptx_runtime_available"
    assert "tool_geom_profile_prep_v1" in contour["tool_refs"]
    assert "prepare_profile" in " ".join(geom_profile["metadata"]["entrypoints"])
    assert geom_lathe["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "tool_geom_profile_lathe_mesh_v1" in contour["tool_refs"]
    assert "contour_to_lathe_mesh" in " ".join(geom_lathe["metadata"]["entrypoints"])
    assert "TERNARY_QUANT" in geom_lathe["metadata"]["codec_ops"]
    assert geom_extrude["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "tool_geom_profile_extrude_mesh_v1" in contour["tool_refs"]
    assert "contour_to_extrude_mesh" in " ".join(geom_extrude["metadata"]["entrypoints"])
    assert "TERNARY_QUANT" in geom_extrude["metadata"]["codec_ops"]
    assert geom_sweep["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "tool_geom_profile_sweep_mesh_v1" in contour["tool_refs"]
    assert "contour_to_sweep_mesh" in " ".join(geom_sweep["metadata"]["entrypoints"])
    assert "TERNARY_QUANT" in geom_sweep["metadata"]["codec_ops"]
    assert geom_sweep["metadata"]["math_core"]["preferred_tier"] == 2
    assert by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "tool_paint_palette_contrastive_v1" in by_id["tool_fusion_surface_material_projection_v1"]["tool_refs"]
    assert "ProceduralMaterialBridge.project_material" in " ".join(
        by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["entrypoints"]
    )
    assert "contour_material_chain" in by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["execution_chain_presets"]
    assert "contour_extrude_material_chain" in by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["execution_chain_presets"]
    assert "contour_sweep_material_chain" in by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["execution_chain_presets"]
    assert (
        by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["entrypoint_argument_schemas"][
            "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_lathe_mesh"
        ]["optional_kwargs"][0]["default"]
        == 1
    )
    assert "contour_to_textured_extrude_mesh" in " ".join(
        by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["entrypoints"]
    )
    assert "contour_to_textured_sweep_mesh" in " ".join(
        by_id["tool_fusion_surface_material_projection_v1"]["metadata"]["entrypoints"]
    )
    assert surface_timeline["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "contour_timeline_chain" in surface_timeline["metadata"]["execution_chain_presets"]
    assert "contour_extrude_timeline_chain" in surface_timeline["metadata"]["execution_chain_presets"]
    assert "contour_sweep_timeline_chain" in surface_timeline["metadata"]["execution_chain_presets"]
    assert "timeline_preset" in surface_timeline["metadata"]["inputs"]
    assert surface_timeline["metadata"]["timeline_presets"] == [
        "ui_idle",
        "ui_focus",
        "world_breathe",
        "world_orbit",
    ]
    assert signal_timeline["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "signal_timeline_chain" in signal_timeline["metadata"]["execution_chain_presets"]
    assert "timeline_preset" in signal_timeline["metadata"]["inputs"]
    assert signal_timeline["metadata"]["timeline_presets"] == [
        "ui_idle",
        "ui_focus",
        "world_breathe",
        "world_orbit",
    ]
    assert "tool_video_temporal_preview_v1" in signal_timeline["tool_refs"]
    assert surface_ui_animation["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert surface_ui_animation["metadata"]["default_timeline_preset"] == "ui_idle"
    assert surface_ui_animation["metadata"]["timeline_presets"] == ["ui_idle", "ui_focus"]
    assert "tool_video_temporal_preview_v1" in surface_ui_animation["tool_refs"]
    assert "ui_lathe_chain" in surface_ui_animation["metadata"]["execution_chain_presets"]
    assert surface_world_animation["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert surface_world_animation["metadata"]["default_timeline_preset"] == "world_breathe"
    assert surface_world_animation["metadata"]["timeline_presets"] == ["world_breathe", "world_orbit"]
    assert "world_lathe_chain" in surface_world_animation["metadata"]["execution_chain_presets"]
    assert signal_ui_animation["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert signal_ui_animation["metadata"]["default_timeline_preset"] == "ui_idle"
    assert "ui_signal_chain" in signal_ui_animation["metadata"]["execution_chain_presets"]
    assert signal_world_animation["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert signal_world_animation["metadata"]["default_timeline_preset"] == "world_breathe"
    assert "world_signal_chain" in signal_world_animation["metadata"]["execution_chain_presets"]
    assert temporal_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "compose_scene_timeline" in " ".join(temporal_scene["metadata"]["entrypoints"])
    assert "surface_materials_to_scene_timeline" in " ".join(temporal_scene["metadata"]["entrypoints"])
    assert temporal_scene["metadata"]["scene_layouts"] == ["overlay", "horizontal_strip", "vertical_strip", "golden_orbit"]
    assert ui_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert ui_scene["metadata"]["default_timeline_preset"] == "ui_idle"
    assert ui_scene["metadata"]["default_scene_layout"] == "overlay"
    assert "tool_video_temporal_scene_v1" in ui_scene["tool_refs"]
    assert world_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert world_scene["metadata"]["default_timeline_preset"] == "world_breathe"
    assert world_scene["metadata"]["default_scene_layout"] == "golden_orbit"
    assert "tool_video_temporal_scene_v1" in world_scene["tool_refs"]
    assert replay_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "replay_journal_to_scene_timeline" in " ".join(replay_scene["metadata"]["entrypoints"])
    assert replay_scene["metadata"]["default_scene_layout"] == "golden_orbit"
    assert library_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert library_scene["metadata"]["default_room_preset"] == "house_library"
    assert library_scene["metadata"]["default_scene_layout"] == "overlay"
    assert garden_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert garden_scene["metadata"]["default_room_preset"] == "house_garden"
    assert garden_scene["metadata"]["default_scene_layout"] == "golden_orbit"
    assert museum_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert museum_scene["metadata"]["default_room_preset"] == "house_museum"
    assert museum_scene["metadata"]["default_scene_layout"] == "horizontal_strip"
    assert tour_scene["metadata"]["runtime_status"] == "ptx_bridge_available"
    assert "house_library" in tour_scene["metadata"]["tour_rooms"]
    assert "house_garden" in tour_scene["metadata"]["tour_rooms"]
    assert "house_museum" in tour_scene["metadata"]["tour_rooms"]


def test_tool_galaxy_bootstrap_is_additive_and_idempotent(tmp_path):
    storage_root = tmp_path / "kv_tools"
    first = bootstrap_tool_galaxy(storage_root=storage_root)
    second = bootstrap_tool_galaxy(storage_root=storage_root)

    assert first["appended"] >= 17
    assert first["after"] == first["generated"]
    assert second["appended"] == 0
    assert second["after"] == first["after"]


def test_tool_payload_builder_targets_tool_galaxy():
    rows = build_tool_payload()
    assert rows
    assert all(row["galaxy"] == "Tool" for row in rows)
    assert all(isinstance(row.get("entry"), dict) for row in rows)


def test_tool_galaxy_entries_are_queryable_by_audio_and_visual_specialists(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    storage_root = tmp_path / "kv_query"
    bootstrap_tool_galaxy(storage_root=storage_root)

    kv = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    kv.galaxy_manager.get_galaxy("Tool")

    audio_results = kv.galaxy_manager.query(
        "spectrogram audio signal waveform",
        specialist="audio",
        top_k=5,
        galaxies=["Tool"],
    )
    visual_results = kv.galaxy_manager.query(
        "spectrogram drawable signal visual",
        specialist="visual",
        top_k=5,
        galaxies=["Tool"],
    )

    audio_ids = {str(item["entry"].get("id")) for item in audio_results}
    visual_ids = {str(item["entry"].get("id")) for item in visual_results}

    assert "tool_signal_audio_spectrogram_v1" in audio_ids
    assert "tool_signal_audio_spectrogram_v1" in visual_ids
    assert "tool_paint_gradient_backdrop_v1" in visual_ids


def test_knowledgeverse_bootstraps_tool_galaxy_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_default")
    tool = kv.galaxy_manager.get_galaxy("Tool")

    ids = {str(entry.get("id")) for entry in tool.entries}
    assert "tool_codec_audio_mdct_v1" in ids
    assert "tool_codec_video_dct8_grid_v1" in ids
    assert "tool_mathcore_tier1_scalar_worker_worker_v1" in ids
    assert "tool_mathcore_tier2_vector_worker_v1" in ids
    assert "tool_mathcore_tier3_master_v1" in ids
    assert "tool_mathcore_spawn_cascade_v1" in ids
    assert "tool_paint_filter_stack_v1" in ids
    assert "tool_paint_palette_contrastive_v1" in ids
    assert "tool_paint_gradient_contrastive_v1" in ids
    assert "tool_paint_gradient_cascade_v1" in ids
    assert "tool_signal_audio_spectrogram_v1" in ids
    assert "tool_fusion_signal_surface_material_v1" in ids
    assert "tool_fusion_contour_to_mesh_v1" in ids
    assert "tool_geom_profile_prep_v1" in ids
    assert "tool_geom_profile_lathe_mesh_v1" in ids
    assert "tool_geom_profile_extrude_mesh_v1" in ids
    assert "tool_geom_profile_sweep_mesh_v1" in ids
    assert "tool_fusion_surface_material_projection_v1" in ids
    assert "tool_video_temporal_preview_v1" in ids
    assert "tool_fusion_surface_material_timeline_v1" in ids
    assert "tool_fusion_signal_surface_material_timeline_v1" in ids
    assert "tool_fusion_surface_material_ui_animation_v1" in ids
    assert "tool_fusion_surface_material_world_animation_v1" in ids
    assert "tool_fusion_signal_surface_material_ui_animation_v1" in ids
    assert "tool_fusion_signal_surface_material_world_animation_v1" in ids
    assert "tool_video_temporal_scene_v1" in ids
    assert "tool_fusion_surface_material_ui_scene_v1" in ids
    assert "tool_fusion_surface_material_world_scene_v1" in ids
    assert "tool_house_replay_scene_v1" in ids
    assert "tool_house_library_scene_v1" in ids
    assert "tool_house_garden_scene_v1" in ids
    assert "tool_house_museum_scene_v1" in ids
    assert "tool_house_tour_scene_v1" in ids
    assert kv.foundational_bootstrap_summary["tool"]["inserted"] >= 33


def test_router_includes_tool_for_audio_math_any_and_cartographer(tmp_path):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_router", bootstrap_foundational_galaxies=False)
    audio_route = kv.specialist_router.route("spectrogram waveform audio signal", specialist="auto")
    math_route = kv.specialist_router.route("solve x + 2 = 5", specialist="math")
    cart_route = kv.specialist_router.route("mesh texture contour surface", specialist="auto")
    any_route = kv.specialist_router.route("generic query", specialist="any")

    assert audio_route["specialist"] == "audio"
    assert "Tool" in audio_route["galaxy_names"]
    assert math_route["specialist"] == "math"
    assert "Tool" in math_route["galaxy_names"]
    assert cart_route["specialist"] in {"visual", "cartographer", "physics"}
    assert "Tool" in cart_route["galaxy_names"]
    assert "Tool" in any_route["galaxy_names"]


def test_trm_compose_surfaces_tool_context(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_compose")
    rows = kv.galaxy_manager.query(
        query_text="audio spectrogram signal",
        specialist="audio",
        top_k=5,
        galaxies=["Tool"],
    )
    composed = kv.trm_navigator.compose(
        query="audio spectrogram signal",
        patterns=rows,
        specialist="audio",
    )

    tool_context = composed.get("tool_context")
    assert isinstance(tool_context, dict)
    assert "tool_signal_audio_spectrogram_v1" in tool_context["tool_ids"]
    assert "MDCT" in tool_context["codec_ops"]
    assert tool_context["tool_kinds"]
    assert "2" in tool_context["math_core_tiers"]
    assert "worker" in tool_context["math_core_roles"]
    assert "adaptive_reuse" in tool_context["math_core_spawn_policies"]
    assert "knowledgeverse_galaxy" in tool_context["memory_residencies"]
    assert tool_context["executable_tool_ids"]
    assert tool_context["entrypoints"]
    assert tool_context["primary_tool_id"]
    assert tool_context["primary_entrypoint"]
    assert tool_context["execution_chain"]
    execution_plan = composed.get("execution_plan")
    assert isinstance(execution_plan, dict)
    assert execution_plan["mode"] == "tool_entrypoint_chain"
    assert execution_plan["primary_entrypoint"]
    assert execution_plan["primary_argument_schema"]["positional"][0]["name"] == "clip_id"
    assert "audio_signal" in execution_plan["inputs"]
    assert tool_context["math_core_plan"]["preferred_tier"] in {2, 3}
    assert tool_context["math_core_plan"]["tier_role"] in {"worker", "master"}


def test_trm_compose_prefers_signal_material_fusion_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_signal_fusion")
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

    tool_context = composed.get("tool_context")
    assert isinstance(tool_context, dict)
    assert "tool_fusion_signal_surface_material_v1" in tool_context["tool_ids"]
    assert "tool_fusion_signal_surface_material_v1" in tool_context["executable_tool_ids"]
    assert "ProceduralMaterialBridge.signal_to_textured_surface" in " ".join(tool_context["entrypoints"])
    assert tool_context["primary_entrypoint"]
    execution_plan = composed.get("execution_plan")
    assert isinstance(execution_plan, dict)
    assert execution_plan["primary_tool_id"] == "tool_fusion_signal_surface_material_v1"
    assert "ProceduralMaterialBridge.signal_to_textured_surface" in execution_plan["primary_entrypoint"]


def test_codec_tools_are_queryable_as_always_on_procedural_means(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_codec_query")

    audio_results = kv.galaxy_manager.query(
        query_text="audio mdct codec ternary signal",
        specialist="audio",
        top_k=5,
        galaxies=["Tool"],
    )
    visual_results = kv.galaxy_manager.query(
        query_text="dct8 codec blocks image video grid",
        specialist="visual",
        top_k=5,
        galaxies=["Tool"],
    )

    audio_ids = {str(item["entry"].get("id")) for item in audio_results}
    visual_ids = {str(item["entry"].get("id")) for item in visual_results}

    assert "tool_codec_audio_mdct_v1" in audio_ids
    assert "tool_codec_video_dct8_grid_v1" in visual_ids


def test_geometry_prep_tools_are_queryable_for_contour_workflows(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_geom_query")

    rows = kv.galaxy_manager.query(
        query_text="contour profile bbox crop lathe extrusion",
        specialist="visual",
        top_k=8,
        galaxies=["Tool"],
    )
    ids = {str(item["entry"].get("id")) for item in rows}

    assert "tool_geom_bbox_crop_v1" in ids
    assert "tool_geom_profile_prep_v1" in ids
    assert "tool_geom_profile_lathe_mesh_v1" in ids
    assert "tool_geom_profile_extrude_mesh_v1" in ids
    assert "tool_geom_profile_sweep_mesh_v1" in ids
    assert "tool_fusion_surface_material_projection_v1" in ids


def test_math_core_tools_are_queryable_for_math_specialist(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_REQUIRE_PTX_QUERY", "false")
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mathcore_query")

    rows = kv.galaxy_manager.query(
        query_text="tiered math core rpn worker master pool",
        specialist="math",
        top_k=6,
        galaxies=["Tool"],
    )
    ids = {str(item["entry"].get("id")) for item in rows}

    assert "tool_mathcore_tier1_scalar_worker_worker_v1" in ids
    assert "tool_mathcore_tier2_vector_worker_v1" in ids
    assert "tool_mathcore_tier3_master_v1" in ids
