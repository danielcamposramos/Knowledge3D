from __future__ import annotations

import json

from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
from knowledge3d.knowledgeverse.galaxy_loader import load_all_galaxies_from_disk
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.knowledge_gap_inventory import curated_math_question_required_ids
from knowledge3d.knowledgeverse.galaxy_vram_table import (
    GalaxyVRAMTable,
    ROLE_EXECUTOR,
    ROLE_ROUTER,
    ROLE_VALIDATOR,
    compose_star_embedding,
)
from knowledge3d.knowledgeverse.gpu_task_dispatch import GPUTaskDispatch
from knowledge3d.knowledgeverse.vram_task_buffer import VRAMTaskBuffer

from tests.foundational_test_utils import build_resolved_foundational_stars


SPINE_LAYOUT = {
    "math_surface_bridge": ("router", 4, False, "MATH"),
    "math_question_router": ("router", 4, False, "MATH"),
    "math_compute_executor": ("executor", 3, False, "MATH"),
    "math_word_problem_executor": ("executor", 3, False, "MATH"),
    "math_quantity_binding_executor": ("executor", 3, False, "MATH"),
    "math_goal_trace_executor": ("executor", 3, False, "MATH"),
    "math_operation_chain_executor": ("executor", 3, False, "MATH"),
    "math_answer_materializer": ("executor", 3, False, "MATH"),
    "question_surface_bridge": ("router", 4, False, "QUESTION"),
    "math_answer_validator": ("validator", 4, True, "MATH"),
    "math_normalization_validator": ("validator", 4, True, "MATH"),
    "math_unit_magnitude_validator": ("validator", 4, True, "MATH"),
    "question_router": ("router", 4, False, "QUESTION"),
    "question_subject_grounding_executor": ("executor", 3, False, "QUESTION"),
    "knowledge_lookup_executor": ("executor", 3, False, "QUESTION"),
    "question_option_elimination_executor": ("executor", 3, False, "QUESTION"),
    "question_choice_materializer": ("executor", 3, False, "QUESTION"),
    "grammar_surface_bridge": ("router", 4, False, "GRAMMAR"),
    "question_evidence_validator": ("validator", 4, True, "QUESTION"),
    "question_choice_alignment_validator": ("validator", 4, True, "QUESTION"),
    "question_answer_validator": ("validator", 4, True, "QUESTION"),
    "grammar_router": ("router", 4, False, "GRAMMAR"),
    "grammar_parse_executor": ("executor", 3, False, "GRAMMAR"),
    "grammar_slot_binding_executor": ("executor", 3, False, "GRAMMAR"),
    "grammar_sequence_executor": ("executor", 3, False, "GRAMMAR"),
    "grammar_transform_executor": ("executor", 3, False, "GRAMMAR"),
    "grammar_answer_materializer": ("executor", 3, False, "GRAMMAR"),
    "grammar_normalization_validator": ("validator", 4, True, "GRAMMAR"),
    "grammar_answer_validator": ("validator", 4, True, "GRAMMAR"),
    "general_surface_bridge": ("router", 4, False, "GENERAL"),
    "general_router": ("router", 4, False, "GENERAL"),
    "general_lookup_executor": ("executor", 3, False, "GENERAL"),
    "general_compare_executor": ("executor", 3, False, "GENERAL"),
    "general_evidence_executor": ("executor", 3, False, "GENERAL"),
    "general_answer_materializer": ("executor", 3, False, "GENERAL"),
    "general_grounding_validator": ("validator", 4, True, "GENERAL"),
    "general_consistency_validator": ("validator", 4, True, "GENERAL"),
    "general_answer_validator": ("validator", 4, True, "GENERAL"),
    "game2d_surface_bridge": ("router", 4, False, "GAME_2D"),
    "game2d_router": ("router", 4, False, "GAME_2D"),
    "game2d_state_parse_executor": ("executor", 3, False, "GAME_2D"),
    "game2d_delta_extractor_executor": ("executor", 3, False, "GAME_2D"),
    "game2d_transform_inference_executor": ("executor", 3, False, "GAME_2D"),
    "game2d_action_materializer": ("executor", 3, False, "GAME_2D"),
    "game2d_grid_materializer": ("executor", 3, False, "GAME_2D"),
    "game2d_state_transition_validator": ("validator", 4, True, "GAME_2D"),
    "game2d_output_validator": ("validator", 4, True, "GAME_2D"),
    "chat_router": ("router", 4, False, "CHAT"),
    "chat_intent_executor": ("executor", 3, False, "CHAT"),
    "chat_grounding_executor": ("executor", 3, False, "CHAT"),
    "chat_grounding_validator": ("validator", 4, True, "CHAT"),
    "chat_response_validator": ("validator", 4, True, "CHAT"),
}

ANTI_PATTERN_LAYOUT = {
    "anti_pattern_generic_language_numeric_winner": ("anti_pattern", 4, False, "MATH"),
    "anti_pattern_unit_magnitude_mismatch": ("anti_pattern", 4, False, "MATH"),
    "anti_pattern_unchecked_unit_transfer": ("anti_pattern", 4, False, "MATH"),
    "anti_pattern_numeric_without_materialization": ("anti_pattern", 4, False, "MATH"),
    "anti_pattern_generic_language_factual_winner": ("anti_pattern", 4, False, "GENERAL"),
    "anti_pattern_missing_evidence_consistency": ("anti_pattern", 4, False, "GENERAL"),
    "anti_pattern_empty_route_dispatch": ("anti_pattern", 4, False, "GENERAL"),
    "anti_pattern_shallow_router_stop": ("anti_pattern", 4, False, "QUESTION"),
    "anti_pattern_unsupported_option_leap": ("anti_pattern", 4, False, "QUESTION"),
    "anti_pattern_option_emission_without_comparison": ("anti_pattern", 4, False, "QUESTION"),
    "anti_pattern_validator_as_answer_leakage": ("anti_pattern", 4, False, "QUESTION"),
    "anti_pattern_answer_format_mismatch": ("anti_pattern", 4, False, "GRAMMAR"),
    "anti_pattern_symbol_meaning_drift": ("anti_pattern", 4, False, "GRAMMAR"),
    "anti_pattern_wrong_family_grounding": ("anti_pattern", 4, False, "GAME_2D"),
    "anti_pattern_action_without_state_transition": ("anti_pattern", 4, False, "GAME_2D"),
    "anti_pattern_grid_without_transform_inference": ("anti_pattern", 4, False, "GAME_2D"),
    "anti_pattern_chat_ungrounded_response": ("anti_pattern", 4, False, "CHAT"),
    "anti_pattern_missing_validator_traversal": ("anti_pattern", 4, False, "CHAT"),
}


def test_foundational_builder_preserves_string_refs_and_spine_roles():
    stars = build_foundational_galaxy_table()
    stars_by_id = {str(star["id"]): star for star in stars}

    for star_id, (selection_role, layer_id, answer_eligible, route_family) in SPINE_LAYOUT.items():
        star = stars_by_id[star_id]
        assert isinstance(star.get("_ref_ids"), list)
        assert star["selection_role"] == selection_role
        assert star["layer_id"] == layer_id
        assert star["answer_eligible"] is answer_eligible
        assert star["route_family"] == route_family

    for star_id, (selection_role, layer_id, answer_eligible, route_family) in ANTI_PATTERN_LAYOUT.items():
        star = stars_by_id[star_id]
        assert isinstance(star.get("_ref_ids"), list)
        assert star["selection_role"] == selection_role
        assert star["layer_id"] == layer_id
        assert star["answer_eligible"] is answer_eligible
        assert star["route_family"] == route_family

    math_router = stars_by_id["math_question_router"]
    math_surface_bridge = stars_by_id["math_surface_bridge"]
    question_router = stars_by_id["question_router"]
    question_surface_bridge = stars_by_id["question_surface_bridge"]
    grammar_router = stars_by_id["grammar_router"]
    grammar_surface_bridge = stars_by_id["grammar_surface_bridge"]
    general_router = stars_by_id["general_router"]
    general_surface_bridge = stars_by_id["general_surface_bridge"]
    game2d_router = stars_by_id["game2d_router"]
    game2d_surface_bridge = stars_by_id["game2d_surface_bridge"]
    chat_router = stars_by_id["chat_router"]

    assert "math_answer_materializer" in math_surface_bridge["_ref_ids"]
    assert math_surface_bridge["route_policy"]["surface_bridge"] is True
    assert "math_compute_executor" in math_router["_ref_ids"]
    assert "math_word_problem_executor" in math_router["_ref_ids"]
    assert "math_answer_validator" in math_router["_ref_ids"]
    assert {"math_compute_executor", "math_word_problem_executor"}.issubset(set(math_router["executor_refs"]))
    assert {
        "math_answer_validator",
        "math_normalization_validator",
        "math_unit_magnitude_validator",
    }.issubset(set(math_router["validator_refs"]))
    assert "anti_pattern_generic_language_numeric_winner" in math_router["anti_pattern_refs"]
    assert "anti_pattern_unchecked_unit_transfer" in math_router["anti_pattern_refs"]

    assert "question_choice_materializer" in question_surface_bridge["_ref_ids"]
    assert question_surface_bridge["route_policy"]["surface_bridge"] is True
    assert "question_subject_grounding_executor" in question_router["_ref_ids"]
    assert "knowledge_lookup_executor" in question_router["_ref_ids"]
    assert "question_choice_materializer" in question_router["_ref_ids"]
    assert "question_answer_validator" in question_router["_ref_ids"]
    assert {"question_subject_grounding_executor", "question_option_elimination_executor"}.issubset(set(question_router["executor_refs"]))
    assert {"question_evidence_validator", "question_answer_validator"}.issubset(set(question_router["validator_refs"]))
    assert "anti_pattern_unsupported_option_leap" in question_router["anti_pattern_refs"]
    assert "anti_pattern_option_emission_without_comparison" in question_router["anti_pattern_refs"]
    assert "anti_pattern_validator_as_answer_leakage" in question_router["anti_pattern_refs"]

    assert "grammar_answer_materializer" in grammar_surface_bridge["_ref_ids"]
    assert grammar_surface_bridge["route_policy"]["surface_bridge"] is True
    assert {"grammar_parse_executor", "grammar_transform_executor"}.issubset(set(grammar_router["executor_refs"]))
    assert {"grammar_normalization_validator", "grammar_answer_validator"}.issubset(set(grammar_router["validator_refs"]))
    assert "anti_pattern_answer_format_mismatch" in grammar_router["anti_pattern_refs"]
    assert "anti_pattern_symbol_meaning_drift" in grammar_router["anti_pattern_refs"]
    assert "anti_pattern_validator_as_answer_leakage" in grammar_router["anti_pattern_refs"]

    assert "general_answer_materializer" in general_surface_bridge["_ref_ids"]
    assert general_surface_bridge["route_policy"]["surface_bridge"] is True
    assert {"general_lookup_executor", "general_evidence_executor"}.issubset(set(general_router["executor_refs"]))
    assert {"general_consistency_validator", "general_answer_validator"}.issubset(set(general_router["validator_refs"]))
    assert "anti_pattern_generic_language_factual_winner" in general_router["anti_pattern_refs"]
    assert "anti_pattern_missing_evidence_consistency" in general_router["anti_pattern_refs"]
    assert "anti_pattern_empty_route_dispatch" in general_router["anti_pattern_refs"]

    assert "game2d_action_materializer" in game2d_surface_bridge["_ref_ids"]
    assert game2d_surface_bridge["route_policy"]["surface_bridge"] is True
    assert {"game2d_state_parse_executor", "game2d_delta_extractor_executor"}.issubset(set(game2d_router["executor_refs"]))
    assert {"game2d_state_transition_validator", "game2d_output_validator"}.issubset(set(game2d_router["validator_refs"]))
    assert "anti_pattern_wrong_family_grounding" in game2d_router["anti_pattern_refs"]
    assert "anti_pattern_action_without_state_transition" in game2d_router["anti_pattern_refs"]
    assert "anti_pattern_grid_without_transform_inference" in game2d_router["anti_pattern_refs"]

    assert {"chat_intent_executor", "chat_grounding_executor"}.issubset(set(chat_router["executor_refs"]))
    assert {"chat_grounding_validator", "chat_response_validator"}.issubset(set(chat_router["validator_refs"]))
    assert "anti_pattern_chat_ungrounded_response" in chat_router["anti_pattern_refs"]
    assert "anti_pattern_missing_validator_traversal" in chat_router["anti_pattern_refs"]


def test_load_all_galaxies_from_disk_resolves_foundational_and_disk_role_refs(tmp_path):
    galaxy_dir = tmp_path / "galaxies"
    galaxy_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "disk_question_bridge",
        "embedding": [0.03125 * (idx + 1) for idx in range(32)],
        "selection_role": "router",
        "layer_id": 2,
        "answer_eligible": False,
        "component_refs": ["question_router", "knowledge_lookup_executor", "question_answer_validator"],
        "executor_refs": ["knowledge_lookup_executor"],
        "validator_refs": ["question_answer_validator"],
        "route_policy": {
            "requires_executor": True,
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    }
    (galaxy_dir / "question_bridge.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    stars = load_all_galaxies_from_disk(galaxy_dir)
    stars_by_id = {str(star.get("id") or star.get("_id")): star for star in stars}

    math_router = stars_by_id["math_question_router"]
    question_router = stars_by_id["question_router"]
    disk_bridge = stars_by_id["disk_question_bridge"]

    assert math_router["executor_refs"]
    assert math_router["validator_refs"]
    assert all(isinstance(value, int) for value in math_router["component_refs"])
    assert all(isinstance(value, int) for value in math_router["executor_refs"])
    assert all(isinstance(value, int) for value in math_router["validator_refs"])

    assert question_router["executor_refs"]
    assert question_router["validator_refs"]
    assert all(isinstance(value, int) for value in question_router["component_refs"])

    assert disk_bridge["executor_refs"]
    assert disk_bridge["validator_refs"]
    assert all(isinstance(value, int) for value in disk_bridge["component_refs"])
    assert all(isinstance(value, int) for value in disk_bridge["executor_refs"])
    assert all(isinstance(value, int) for value in disk_bridge["validator_refs"])


def test_default_knowledge_materialization_seeds_foundational_route_spines(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_default_spines",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    monkeypatch.setattr(kv, "save_consolidated_state", lambda: {"status": "skipped"})

    source_summary = kv._default_knowledge_source_summary()
    summary = kv._materialize_default_knowledge(force=True)
    persisted = kv._persistable_galaxy_entries()

    reality_ids = {str(entry.get("id") or "") for entry in persisted.get("Reality", [])}
    grammar_ids = {str(entry.get("id") or "") for entry in persisted.get("Grammar", [])}
    math_ids = {str(entry.get("id") or "") for entry in persisted.get("Math", [])}
    all_ids = reality_ids | grammar_ids | math_ids

    assert source_summary["foundational_spine_rows"] > 0
    assert source_summary["signature"]
    assert summary["foundational_spine_rows"] == source_summary["foundational_spine_rows"]
    assert "question_router" in reality_ids
    assert "knowledge_lookup_executor" in reality_ids
    assert "question_evidence_validator" in reality_ids
    assert "question_answer_validator" in reality_ids
    assert "general_router" in reality_ids
    assert "grammar_router" in grammar_ids
    assert "chat_router" in grammar_ids
    assert "math_question_router" in math_ids
    assert "math_answer_validator" in math_ids
    assert "game2d_router" in all_ids
    assert "game2d_output_validator" in all_ids
    assert "form_marker_fraction" in all_ids
    assert "meaning_quantity_initial" in all_ids
    assert "domain_packet_chess_core" in all_ids


def test_curated_math_question_wave_presence_and_contract():
    stars = build_foundational_galaxy_table()
    stars_by_id = {str(star["id"]): dict(star) for star in stars}

    missing = sorted(set(curated_math_question_required_ids()) - set(stars_by_id))
    assert not missing

    assert stars_by_id["question_open_text_materializer"]["selection_role"] == "executor"
    assert stars_by_id["question_open_text_materializer"]["route_family"] == "QUESTION"
    assert stars_by_id["question_open_text_materializer"]["route_policy"]["materialize_answer"] is True
    assert stars_by_id["general_open_text_materializer"]["selection_role"] == "executor"
    assert stars_by_id["general_open_text_materializer"]["route_policy"]["materialize_answer"] is True
    assert stars_by_id["meta_route_math_operation_family_before_emit"]["selection_role"] == "router"
    assert stars_by_id["anti_pattern_wrong_math_template_family"]["selection_role"] == "anti_pattern"

    for anchor_id in (
        "form_marker_fraction",
        "meaning_relation_definition_of",
        "domain_packet_chess_core",
    ):
        star = stars_by_id[anchor_id]
        assert star["selection_role"] == "unknown"
        assert star["layer_id"] == 0
        assert star["answer_eligible"] is False
        assert star["sovereign_route_exempt"] is True

    assert "question_open_text_materializer" in stars_by_id["question_router"]["_ref_ids"]
    assert "question_definition_lookup_executor" in stars_by_id["question_router"]["executor_refs"]
    assert "question_evidence_compare_executor" in stars_by_id["question_definition_lookup_executor"]["executor_refs"]
    assert "question_choice_materializer" in stars_by_id["question_definition_lookup_executor"]["executor_refs"]
    assert "question_choice_alignment_validator" in stars_by_id["question_definition_lookup_executor"]["validator_refs"]
    assert "question_choice_materializer" in stars_by_id["meta_route_definition_lookup_before_answer_emit"]["executor_refs"]
    assert "question_evidence_compare_executor" in stars_by_id["meta_route_definition_lookup_before_answer_emit"]["executor_refs"]
    assert "general_open_text_materializer" in stars_by_id["general_router"]["_ref_ids"]
    assert "math_multiplicative_comparison_executor" in stars_by_id["math_question_router"]["executor_refs"]


def test_sovereign_star_build_places_router_spine_on_meta_plane(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_spine_plane",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    runtime = kv._get_sovereign_hot_path()
    stars_by_id = {str(star["id"]): dict(star) for star in build_foundational_galaxy_table()}

    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(stars_by_id[str(row.get("id") or "")]),
    )

    try:
        star_ids = (
            "math_surface_bridge",
            "math_question_router",
            "math_compute_executor",
            "math_quantity_binding_executor",
            "math_goal_trace_executor",
            "math_operation_chain_executor",
            "math_word_problem_executor",
            "math_answer_materializer",
            "math_normalization_validator",
            "math_unit_magnitude_validator",
            "math_answer_validator",
            "question_surface_bridge",
            "question_router",
            "question_subject_grounding_executor",
            "knowledge_lookup_executor",
            "question_option_elimination_executor",
            "question_choice_materializer",
            "question_evidence_validator",
            "question_choice_alignment_validator",
            "question_answer_validator",
            "grammar_surface_bridge",
            "grammar_router",
            "grammar_parse_executor",
            "grammar_slot_binding_executor",
            "grammar_sequence_executor",
            "grammar_transform_executor",
            "grammar_answer_materializer",
            "grammar_normalization_validator",
            "grammar_answer_validator",
            "general_surface_bridge",
            "general_router",
            "general_lookup_executor",
            "general_compare_executor",
            "general_evidence_executor",
            "general_answer_materializer",
            "general_grounding_validator",
            "general_consistency_validator",
            "general_answer_validator",
            "game2d_surface_bridge",
            "game2d_router",
            "game2d_state_parse_executor",
            "game2d_delta_extractor_executor",
            "game2d_transform_inference_executor",
            "game2d_action_materializer",
            "game2d_grid_materializer",
            "game2d_state_transition_validator",
            "game2d_output_validator",
            "chat_router",
            "chat_intent_executor",
            "chat_grounding_executor",
            "chat_grounding_validator",
            "chat_response_validator",
        )
        built = runtime._build_stars_from_catalog(
            [
                {
                    "galaxy": str(stars_by_id[star_id].get("route_family") or "GENERAL"),
                    "id": star_id,
                    "domain_hash": 11.0 if str(stars_by_id[star_id].get("route_family") or "") == "MATH" else 13.0,
                    "subject_hash": 7.0 if "router" in star_id else 5.0,
                    "embedding16": list(stars_by_id[star_id]["embedding"][:16]),
                    "confidence": 0.9 if "router" in star_id else 0.85,
                }
                for star_id in star_ids
            ]
        )
        built_by_id = {str(star["id"]): star for star in built}

        assert built_by_id["math_question_router"]["layer_id"] == 4
        assert built_by_id["question_router"]["layer_id"] == 4
        assert built_by_id["grammar_router"]["layer_id"] == 4
        assert built_by_id["general_router"]["layer_id"] == 4
        assert built_by_id["game2d_router"]["layer_id"] == 4
        assert built_by_id["chat_router"]["layer_id"] == 4
        assert built_by_id["math_question_router"]["semantic_position"][2] == 1.0
        assert built_by_id["question_router"]["semantic_position"][2] == 1.0
        assert built_by_id["grammar_router"]["semantic_position"][2] == 1.0
        assert built_by_id["general_router"]["semantic_position"][2] == 1.0
        assert built_by_id["game2d_router"]["semantic_position"][2] == 1.0
        assert built_by_id["chat_router"]["semantic_position"][2] == 1.0
    finally:
        runtime.close()
        kv._sovereign_hot_path = None


def _dispatch_spine_task(
    *,
    stars: list[dict[str, object]],
    router_id: str,
    task_type: str,
    option_count: int,
) -> dict[str, object]:
    id_to_index = {str(star["id"]): index for index, star in enumerate(stars)}
    router_index = id_to_index[router_id]
    query_embedding = compose_star_embedding(stars, router_index)
    task = {
        "type": task_type,
        "query_embedding": query_embedding,
        "option_embeddings": [[1.0 if dim == option_index else 0.0 for dim in range(32)] for option_index in range(option_count)],
        "subject": f"{task_type.lower()}_subject",
        "domain_hint": f"{task_type.lower()}_domain",
        "thinking_budget": 10,
        "ternary_signal": 0,
    }
    task_buffer = VRAMTaskBuffer(max_tasks=1)
    galaxy_table = GalaxyVRAMTable(max_stars=len(stars) + 8)
    dispatcher = GPUTaskDispatch()
    try:
        galaxy_table.load_stars(stars)
        task_buffer.bulk_load([task])
        dispatcher.launch(task_buffer, 1, star_table=galaxy_table)
        return task_buffer.read_results(1)[0]
    finally:
        galaxy_table.close()
        task_buffer.close()


def test_gpu_task_dispatch_uses_foundational_spine_routes():
    stars = build_resolved_foundational_stars()
    id_to_index = {str(star["id"]): index for index, star in enumerate(stars)}

    math_result = _dispatch_spine_task(
        stars=stars,
        router_id="math_question_router",
        task_type="MATH",
        option_count=4,
    )
    question_result = _dispatch_spine_task(
        stars=stars,
        router_id="question_router",
        task_type="QUESTION",
        option_count=4,
    )
    grammar_result = _dispatch_spine_task(
        stars=stars,
        router_id="grammar_router",
        task_type="GRAMMAR",
        option_count=4,
    )
    general_result = _dispatch_spine_task(
        stars=stars,
        router_id="general_router",
        task_type="GENERAL",
        option_count=4,
    )
    game2d_result = _dispatch_spine_task(
        stars=stars,
        router_id="game2d_router",
        task_type="GAME_2D",
        option_count=4,
    )
    chat_result = _dispatch_spine_task(
        stars=stars,
        router_id="chat_router",
        task_type="CHAT",
        option_count=4,
    )

    assert math_result["route_depth"] >= 2
    assert math_result["recursion_depth_used"] >= 2
    assert math_result["winner_role_id"] != ROLE_ROUTER
    assert math_result["validator_star_index"] in {
        id_to_index["math_answer_validator"],
        id_to_index["math_normalization_validator"],
        id_to_index["math_unit_magnitude_validator"],
    }
    assert math_result["executor_star_index"] in {
        id_to_index["math_compute_executor"],
        id_to_index["math_word_problem_executor"],
        id_to_index["math_quantity_binding_executor"],
        id_to_index["math_goal_trace_executor"],
        id_to_index["math_operation_chain_executor"],
        id_to_index["math_additive_composition_executor"],
        id_to_index["math_multiplicative_comparison_executor"],
        id_to_index["math_fraction_of_total_executor"],
        id_to_index["math_remainder_after_use_executor"],
        id_to_index["math_rate_revenue_projection_executor"],
        id_to_index["math_multi_step_composition_executor"],
        id_to_index["math_unit_preserving_projection_executor"],
        id_to_index["math_answer_materializer"],
    }
    assert math_result["route_trace_star_indices"][0] in {
        id_to_index["math_question_router"],
        id_to_index["math_surface_bridge"],
        id_to_index["math_additive_composition_executor"],
        id_to_index["math_multiplicative_comparison_executor"],
        id_to_index["math_fraction_of_total_executor"],
        id_to_index["math_remainder_after_use_executor"],
        id_to_index["math_rate_revenue_projection_executor"],
        id_to_index["math_multi_step_composition_executor"],
        id_to_index["math_unit_preserving_projection_executor"],
    }
    assert math_result["route_trace_role_ids"][0] in {
        ROLE_ROUTER,
        ROLE_EXECUTOR,
    }

    assert question_result["route_depth"] >= 2
    assert question_result["recursion_depth_used"] >= 2
    assert question_result["winner_role_id"] == ROLE_VALIDATOR
    assert question_result["executor_star_index"] in {
        id_to_index["question_subject_grounding_executor"],
        id_to_index["knowledge_lookup_executor"],
        id_to_index["question_option_elimination_executor"],
        id_to_index["question_definition_lookup_executor"],
        id_to_index["question_evidence_compare_executor"],
        id_to_index["question_choice_materializer"],
        id_to_index["question_open_text_materializer"],
    }
    assert question_result["validator_star_index"] in {
        id_to_index["question_evidence_validator"],
        id_to_index["question_choice_alignment_validator"],
        id_to_index["question_answer_validator"],
    }
    assert question_result["route_trace_star_indices"][0] in {
        id_to_index["question_router"],
        id_to_index["question_surface_bridge"],
    }
    assert question_result["route_trace_role_ids"][0] == ROLE_ROUTER

    assert grammar_result["route_depth"] >= 2
    assert grammar_result["recursion_depth_used"] >= 2
    assert grammar_result["winner_role_id"] == ROLE_VALIDATOR
    assert grammar_result["validator_star_index"] >= 0
    assert str(stars[grammar_result["validator_star_index"]].get("selection_role") or "") == "validator"
    assert ROLE_ROUTER in list(grammar_result.get("route_trace_role_ids") or [])

    assert general_result["route_depth"] >= 2
    assert general_result["recursion_depth_used"] >= 2
    assert general_result["winner_role_id"] == ROLE_VALIDATOR
    assert general_result["validator_star_index"] >= 0
    assert str(stars[general_result["validator_star_index"]].get("selection_role") or "") == "validator"
    assert str(stars[general_result["validator_star_index"]].get("route_family") or "") == "GENERAL"
    assert general_result["route_trace_star_indices"][0] in {
        id_to_index["general_router"],
        id_to_index["general_surface_bridge"],
    }
    assert general_result["route_trace_role_ids"][0] == ROLE_ROUTER

    assert game2d_result["route_depth"] >= 2
    assert game2d_result["recursion_depth_used"] >= 2
    assert game2d_result["winner_role_id"] == ROLE_VALIDATOR
    assert game2d_result["executor_star_index"] in {
        id_to_index["game2d_state_parse_executor"],
        id_to_index["game2d_delta_extractor_executor"],
        id_to_index["game2d_transform_inference_executor"],
        id_to_index["game2d_action_materializer"],
        id_to_index["game2d_grid_materializer"],
    }
    assert game2d_result["validator_star_index"] in {
        id_to_index["game2d_state_transition_validator"],
        id_to_index["game2d_output_validator"],
    }
    assert game2d_result["route_trace_star_indices"][0] in {
        id_to_index["game2d_router"],
        id_to_index["game2d_surface_bridge"],
    }
    assert game2d_result["route_trace_role_ids"][0] == ROLE_ROUTER

    assert chat_result["route_depth"] >= 2
    assert chat_result["recursion_depth_used"] >= 2
    assert chat_result["winner_role_id"] == ROLE_VALIDATOR
    assert chat_result["validator_star_index"] >= 0
    assert str(stars[chat_result["validator_star_index"]].get("selection_role") or "") == "validator"
    assert str(stars[chat_result["validator_star_index"]].get("route_family") or "") == "CHAT"
    assert chat_result["route_trace_star_indices"][0] == id_to_index["chat_router"]
    assert chat_result["route_trace_role_ids"][0] == ROLE_ROUTER
