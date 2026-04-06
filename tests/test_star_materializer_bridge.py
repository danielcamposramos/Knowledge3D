from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from knowledge3d.knowledgeverse.galaxy_manager import normalize_disk_entry
from knowledge3d.knowledgeverse.foundational_galaxy_builder import build_foundational_galaxy_table
from knowledge3d.knowledgeverse.foundational_operations_bootstrap import (
    _foundational_reality_entries,
    _grammar_entries,
)
from knowledge3d.knowledgeverse.galaxy_vram_table import GalaxyVRAMTable
from knowledge3d.knowledgeverse.knowledge_gap_inventory import (
    curated_math_question_coverage_packets,
    curated_math_question_required_ids,
)
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.objects_3d_galaxy import default_3d_objects_entries
from knowledge3d.knowledgeverse import route_contract
from knowledge3d.knowledgeverse.resident_route_metadata import resident_route_registry_summary
from knowledge3d.knowledgeverse.sovereign_hot_path import (
    BUILD_FEED_VERSION,
    FEED_SOURCE_VERSION,
    ROUTE_FAMILY_FLAG_MASK,
    ROUTE_FAMILY_FLAG_SHIFT,
    _encode_runtime_flags,
)
from knowledge3d.knowledgeverse.tool_galaxy import default_tool_entries


def _compile_feed(runtime, *, catalog):
    feed_source = runtime.refresh_feed_source(catalog=list(catalog))
    build_feed = runtime.refresh_build_feed()
    return feed_source, build_feed


def test_runtime_materialize_result_prefers_grounded_choice_over_raw_option_index(tmp_path, monkeypatch):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_runtime_choice_alignment",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    monkeypatch.setattr(
        kv,
        "_runtime_materialize_text_answer",
        lambda **_: "Paris",
    )

    packet = kv.materialize_runtime_result(
        task={
            "query": "What is the capital of France?",
            "options": ["Berlin", "Madrid", "Paris", "Rome"],
        },
        route_family="QUESTION",
        answer_kind="choice",
        answer_index=3,
        stars=[],
    )

    assert packet["answer_choice"] == "Paris"
    assert packet["answer_text"] == "Paris"
    assert packet["answer_kind"] == "choice"


def test_runtime_answer_label_blocks_humanized_internal_labels(tmp_path):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_runtime_label_filter",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )

    assert kv._runtime_answer_label("Anti Pattern Missing Validator Traversal") == ""
    assert kv._runtime_answer_label("Question Open Text Materializer") == ""


def _spine_catalog_and_sources():
    ordered_ids = [
        "math_question_router",
        "math_compute_executor",
        "math_answer_materializer",
        "math_answer_validator",
        "question_router",
        "question_subject_grounding_executor",
        "knowledge_lookup_executor",
        "question_option_elimination_executor",
        "question_choice_materializer",
        "question_evidence_validator",
        "question_answer_validator",
    ]
    source_map = {
        str(star["id"]): dict(star)
        for star in build_foundational_galaxy_table()
        if str(star.get("id") or "") in set(ordered_ids)
    }
    catalog = [
        {
            "galaxy": "Math",
            "id": "math_question_router",
            "domain_hash": 11.0,
            "subject_hash": 7.0,
            "embedding16": [0.0625] * 16,
            "confidence": 0.90,
            "gpu_source_class": 1,
        },
        {
            "galaxy": "Math",
            "id": "math_compute_executor",
            "domain_hash": 11.0,
            "subject_hash": 9.0,
            "embedding16": [0.125] * 16,
            "confidence": 0.80,
            "gpu_source_class": 1,
        },
        {
            "galaxy": "Math",
            "id": "math_answer_materializer",
            "domain_hash": 11.0,
            "subject_hash": 11.0,
            "embedding16": [0.15625] * 16,
            "confidence": 0.82,
            "gpu_source_class": 1,
        },
        {
            "galaxy": "Math",
            "id": "math_answer_validator",
            "domain_hash": 11.0,
            "subject_hash": 13.0,
            "embedding16": [0.1875] * 16,
            "confidence": 0.85,
            "gpu_source_class": 1,
        },
        {
            "galaxy": "Question",
            "id": "question_router",
            "domain_hash": 13.0,
            "subject_hash": 5.0,
            "embedding16": [0.25] * 16,
            "confidence": 0.88,
            "gpu_source_class": 2,
        },
        {
            "galaxy": "Question",
            "id": "question_subject_grounding_executor",
            "domain_hash": 13.0,
            "subject_hash": 6.0,
            "embedding16": [0.28125] * 16,
            "confidence": 0.84,
            "gpu_source_class": 2,
        },
        {
            "galaxy": "Question",
            "id": "knowledge_lookup_executor",
            "domain_hash": 13.0,
            "subject_hash": 8.0,
            "embedding16": [0.3125] * 16,
            "confidence": 0.83,
            "gpu_source_class": 2,
        },
        {
            "galaxy": "Question",
            "id": "question_option_elimination_executor",
            "domain_hash": 13.0,
            "subject_hash": 9.0,
            "embedding16": [0.34375] * 16,
            "confidence": 0.82,
            "gpu_source_class": 2,
        },
        {
            "galaxy": "Question",
            "id": "question_choice_materializer",
            "domain_hash": 13.0,
            "subject_hash": 10.0,
            "embedding16": [0.359375] * 16,
            "confidence": 0.82,
            "gpu_source_class": 2,
        },
        {
            "galaxy": "Question",
            "id": "question_evidence_validator",
            "domain_hash": 13.0,
            "subject_hash": 11.0,
            "embedding16": [0.3671875] * 16,
            "confidence": 0.85,
            "gpu_source_class": 2,
        },
        {
            "galaxy": "Question",
            "id": "question_answer_validator",
            "domain_hash": 13.0,
            "subject_hash": 12.0,
            "embedding16": [0.375] * 16,
            "confidence": 0.86,
            "gpu_source_class": 2,
        },
    ]
    return catalog, source_map


def _route_family_catalog_and_sources():
    ordered_ids = [
        "math_surface_bridge",
        "math_question_router",
        "math_compute_executor",
        "math_word_problem_executor",
        "math_quantity_binding_executor",
        "math_goal_trace_executor",
        "math_operation_chain_executor",
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
        "anti_pattern_generic_language_numeric_winner",
        "anti_pattern_unit_magnitude_mismatch",
        "anti_pattern_unchecked_unit_transfer",
        "anti_pattern_numeric_without_materialization",
        "anti_pattern_generic_language_factual_winner",
        "anti_pattern_missing_evidence_consistency",
        "anti_pattern_empty_route_dispatch",
        "anti_pattern_shallow_router_stop",
        "anti_pattern_unsupported_option_leap",
        "anti_pattern_option_emission_without_comparison",
        "anti_pattern_validator_as_answer_leakage",
        "anti_pattern_answer_format_mismatch",
        "anti_pattern_symbol_meaning_drift",
        "anti_pattern_wrong_family_grounding",
        "anti_pattern_action_without_state_transition",
        "anti_pattern_grid_without_transform_inference",
        "anti_pattern_chat_ungrounded_response",
        "anti_pattern_missing_validator_traversal",
    ]
    ordered_ids = list(dict.fromkeys(ordered_ids + list(curated_math_question_required_ids())))
    source_map = {
        str(star["id"]): dict(star)
        for star in build_foundational_galaxy_table()
        if str(star.get("id") or "") in set(ordered_ids)
    }
    catalog = []
    for index, star_id in enumerate(ordered_ids):
        source = dict(source_map[star_id])
        family = str(source.get("route_family") or "GENERAL").strip().upper()
        galaxy = {
            "MATH": "Math",
            "QUESTION": "Question",
            "GRAMMAR": "Grammar",
            "GENERAL": "General",
            "CHAT": "Chat",
            "GAME_2D": "Reality",
        }.get(family, "General")
        catalog.append(
            {
                "galaxy": galaxy,
                "id": star_id,
                "domain_hash": float(index + 1),
                "subject_hash": float((index % 7) + 1),
                "embedding16": list(source.get("embedding", [])[:16]),
                "confidence": 0.75 + (0.01 * (index % 5)),
                "gpu_source_class": int(source.get("star_type", 0) or 0),
            }
        )
    return catalog, source_map


def test_star_materializer_matches_python_pack_and_csr(tmp_path, monkeypatch):
    catalog, source_map = _spine_catalog_and_sources()
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_materializer_parity",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    runtime = kv._get_sovereign_hot_path()
    reference_table = GalaxyVRAMTable(max_stars=32)
    expected_reader_table = GalaxyVRAMTable(max_stars=32)
    reader_table = GalaxyVRAMTable(max_stars=32)

    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question"])

    try:
        reference_stars = runtime._build_stars_from_catalog(catalog)
        reference_table.load_stars(reference_stars)
        expected_bundle = reference_table.export_artifact_bundle()
        expected_reader_table.load_artifact_bundle(expected_bundle)
        expected_readback = expected_reader_table.read_stars(len(catalog))

        feed_source_summary, feed_summary = _compile_feed(runtime, catalog=catalog)
        build_feed = runtime._load_build_feed(str(feed_summary["build_feed_signature"]))
        summary = runtime._build_stars_from_build_feed(build_feed)
        actual_bundle = runtime.star_table.export_artifact_bundle()
        reader_table.load_artifact_bundle(actual_bundle)
        rebuilt = reader_table.read_stars(len(catalog))

        assert feed_source_summary["mode"] == "feed_source_compile"
        assert feed_summary["mode"] == "build_feed_compile"
        assert summary["build_backend"] == "gpu_build_feed_v2"
        assert summary["build_feed_version"] == BUILD_FEED_VERSION
        assert summary["build_feed_signature"] == feed_summary["build_feed_signature"]
        assert summary["feed_source_signature"] == feed_source_summary["feed_source_signature"]
        assert summary["boot_finalize_s"] >= 0.0
        assert summary["validation_route_trit"] >= -1
        for index in range(len(catalog)):
            actual_star = rebuilt[index]
            expected_star = expected_readback[index]
            assert int(actual_star["layer_id"]) == int(expected_star["layer_id"])
            assert bool(actual_star["answer_eligible"]) is bool(expected_star["answer_eligible"])
            for dim, expected_value in enumerate(expected_star["embedding"]):
                assert math.isclose(
                    float(actual_star["embedding"][dim]),
                    float(expected_value),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                )
        assert len(actual_bundle["router_offsets"]) == len(expected_bundle["router_offsets"]) == len(catalog)
        assert len(actual_bundle["router_counts"]) == len(expected_bundle["router_counts"]) == len(catalog)
        assert len(actual_bundle["executor_offsets"]) == len(expected_bundle["executor_offsets"]) == len(catalog)
        assert len(actual_bundle["executor_counts"]) == len(expected_bundle["executor_counts"]) == len(catalog)
        assert len(actual_bundle["validator_offsets"]) == len(expected_bundle["validator_offsets"]) == len(catalog)
        assert len(actual_bundle["validator_counts"]) == len(expected_bundle["validator_counts"]) == len(catalog)
        assert len(actual_bundle["anti_pattern_offsets"]) == len(expected_bundle["anti_pattern_offsets"]) == len(catalog)
        assert len(actual_bundle["anti_pattern_counts"]) == len(expected_bundle["anti_pattern_counts"]) == len(catalog)
        assert len(actual_bundle["ref_indices"]) > 0
        assert len(expected_bundle["ref_indices"]) > 0
        assert rebuilt[0]["layer_id"] == 4
        assert rebuilt[0]["semantic_position"][2] == 1.0
        assert rebuilt[3]["layer_id"] == 4
        assert rebuilt[3]["semantic_position"][2] == 1.0
    finally:
        reader_table.close()
        expected_reader_table.close()
        reference_table.close()
        runtime.close()
        kv._sovereign_hot_path = None


def test_feed_source_reports_meaning_family_health(tmp_path, monkeypatch):
    catalog, source_map = _route_family_catalog_and_sources()
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_family_health",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    runtime = kv._get_sovereign_hot_path()

    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question", "Grammar", "General", "Chat"])

    try:
        summary = runtime.refresh_feed_source(catalog=list(catalog))
        health = dict(summary["route_family_health"])

        assert summary["mode"] == "feed_source_compile"
        assert health["MATH"]["routers"] >= 1
        assert health["MATH"]["executors"] >= 5
        assert health["MATH"]["materializers"] >= 1
        assert health["MATH"]["validators"] >= 3
        assert health["MATH"]["anti_patterns"] >= 3
        assert health["QUESTION"]["routers"] >= 1
        assert health["QUESTION"]["surface_bridges"] >= 1
        assert health["QUESTION"]["executors"] >= 3
        assert health["QUESTION"]["materializers"] >= 1
        assert health["QUESTION"]["validators"] >= 2
        assert health["QUESTION"]["anti_patterns"] >= 3
        assert health["GRAMMAR"]["surface_bridges"] >= 1
        assert health["GRAMMAR"]["executors"] >= 5
        assert health["GRAMMAR"]["materializers"] >= 1
        assert health["GRAMMAR"]["validators"] >= 2
        assert health["GRAMMAR"]["anti_patterns"] >= 2
        assert health["GENERAL"]["surface_bridges"] >= 1
        assert health["GENERAL"]["executors"] >= 4
        assert health["GENERAL"]["materializers"] >= 1
        assert health["GENERAL"]["validators"] >= 2
        assert health["GENERAL"]["anti_patterns"] >= 3
        assert health["GAME_2D"]["surface_bridges"] >= 1
        assert health["GAME_2D"]["routers"] >= 1
        assert health["GAME_2D"]["executors"] >= 5
        assert health["GAME_2D"]["materializers"] >= 2
        assert health["GAME_2D"]["validators"] >= 2
        assert health["GAME_2D"]["anti_patterns"] >= 3
        assert health["CHAT"]["routers"] >= 1
        assert health["CHAT"]["executors"] >= 2
        assert health["CHAT"]["validators"] >= 2
        assert health["CHAT"]["anti_patterns"] >= 2
        assert all(bucket["incomplete_validator_coverage"] == 0 for bucket in health.values())
        assert all(bucket["missing_materializer_paths"] == 0 for bucket in health.values())
        assert all(bucket["missing_reciprocal_links"] >= 0 for bucket in health.values())
        assert sum(bucket["missing_reciprocal_links"] for bucket in health.values()) > 0
    finally:
        runtime.close()
        kv._sovereign_hot_path = None


def test_build_feed_writes_meaning_family_route_audit(tmp_path, monkeypatch):
    catalog, source_map = _route_family_catalog_and_sources()
    storage_root = tmp_path / "kv_route_audit"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "audit_default_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "audit_house_sig"
    runtime = kv._get_sovereign_hot_path()

    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question", "Grammar", "General", "Chat"])
    monkeypatch.setattr(kv, "build_gpu_catalog_only", lambda *, galaxy_names=None: list(catalog))

    try:
        _feed_source_summary, feed_summary = _compile_feed(runtime, catalog=catalog)
        audit_path = storage_root / "checkpoints" / "meaning_family_route_audit.json"
        closure_audit_path = storage_root / "checkpoints" / "meaning_route_closure_audit.json"
        coverage_audit_path = storage_root / "checkpoints" / "meaning_knowledge_coverage_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        closure_audit = json.loads(closure_audit_path.read_text(encoding="utf-8"))
        coverage_audit = json.loads(coverage_audit_path.read_text(encoding="utf-8"))

        assert feed_summary["meaning_family_route_audit_path"] == str(audit_path)
        assert feed_summary["meaning_family_route_audit_passed"] is True
        assert feed_summary["meaning_route_closure_audit_path"] == str(closure_audit_path)
        assert feed_summary["meaning_route_closure_audit_passed"] is True
        assert feed_summary["meaning_knowledge_coverage_audit_path"] == str(coverage_audit_path)
        assert feed_summary["meaning_knowledge_coverage_audit_passed"] is True
        assert audit["passed"] is True
        assert closure_audit["passed"] is True
        assert coverage_audit["passed"] is True
        assert audit["build_backend"] == "gpu_build_feed_v2"
        assert audit["missing_explicit_route_family"] == 0
        assert audit["total_missing_reciprocal_links"] == 0
        assert audit["total_incomplete_validator_coverage"] == 0
        assert audit["families"]["GRAMMAR"]["actual"]["executors"] >= 4
        assert audit["families"]["GENERAL"]["actual"]["executors"] >= 3
        assert audit["families"]["CHAT"]["actual"]["validators"] >= 2
        assert audit["families"]["QUESTION"]["actual"]["anti_patterns"] >= 2
        assert audit["families"]["MATH"]["actual"]["validators"] >= 3
        assert audit["families"]["GAME_2D"]["actual"]["anti_patterns"] >= 3
        assert all(audit["families"][family]["meets_minima"].values() for family in audit["families"])
        assert closure_audit["families"]["GAME_2D"]["actual"]["surface_bridges"] >= 1
        assert closure_audit["families"]["MATH"]["actual"]["materializers"] >= 1
        assert closure_audit["families"]["QUESTION"]["actual"]["materializers"] >= 1
        assert closure_audit["families"]["GENERAL"]["actual"]["materializers"] >= 1
        assert closure_audit["families"]["GRAMMAR"]["actual"]["materializers"] >= 1
        assert closure_audit["route_broken_count"] == 0
        assert all(
            closure_audit["route_family_health"][family]["broken"] == 0
            for family in closure_audit["families"]
        )
        assert closure_audit["total_missing_materializer_paths"] == 0
        assert all(closure_audit["families"][family]["meets_minima"].values() for family in closure_audit["families"])
        assert coverage_audit["packet_count"] == len(curated_math_question_coverage_packets())
        assert coverage_audit["total_missing_required_ids"] == 0
        assert coverage_audit["total_failed_packets"] == 0
        assert coverage_audit["packets"]["question_open_text_packet"]["has_materializers"] is True
        assert coverage_audit["packets"]["general_open_text_packet"]["has_materializers"] is True
        assert coverage_audit["packets"]["math_operation_family_packet"]["route_chain_complete"] is True
    finally:
        runtime.close()
        kv._sovereign_hot_path = None


def test_force_rebuild_writes_fresh_artifact_then_next_boot_restores_it(tmp_path, monkeypatch):
    catalog, source_map = _spine_catalog_and_sources()
    storage_root = tmp_path / "kv_force_rebuild"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "unit_default_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "unit_house_sig"

    def _catalog_only(*, galaxy_names=None):
        kv._last_gpu_catalog_build_summary = {
            "cache_mode": "test",
            "signature": "unit_catalog_sig",
            "catalog_entries": len(catalog),
        }
        return list(catalog)

    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question"])
    monkeypatch.setattr(kv, "build_gpu_catalog_only", _catalog_only)
    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )

    try:
        runtime = kv._get_sovereign_hot_path()
        feed_source_summary, feed_summary = _compile_feed(runtime, catalog=catalog)
        monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: (_ for _ in ()).throw(AssertionError("catalog_source_reached")))
        monkeypatch.setattr(runtime, "_entry_metadata", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("entry_metadata_reached")))
        monkeypatch.setattr(runtime, "_translate_catalog_entries", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("translate_catalog_reached")))
        monkeypatch.setattr(runtime, "_ensure_bidirectional_symlinkage", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bidirectional_symlinkage_reached")))
        rebuilt = kv._boot_sovereign_runtime(force_reload=True, force_rebuild=True)
        runtime.invalidate_loaded_state()
        artifact = kv._boot_sovereign_runtime(force_reload=True)

        manifest_path = storage_root / "checkpoints" / "sovereign_runtime_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert rebuilt["mode"] == "rebuilt"
        assert rebuilt["build_backend"] == "gpu_build_feed_v2"
        assert manifest["version"] == 6
        assert manifest["build_backend"] == "gpu_build_feed_v2"
        assert manifest["build_feed_version"] == BUILD_FEED_VERSION
        assert manifest["build_feed_signature"] == feed_summary["build_feed_signature"]
        assert manifest["feed_source_version"] == FEED_SOURCE_VERSION
        assert manifest["feed_source_signature"] == feed_source_summary["feed_source_signature"]
        assert manifest["default_knowledge_signature"] == "unit_default_sig"
        assert manifest["house_signature_base"] == "unit_house_sig"
        assert manifest["build_decode_ptx_signature"]
        assert manifest["boot_finalize_ptx_signature"]
        assert manifest["materializer_ptx_signature"]
        assert manifest["csr_builder_ptx_signature"]
        assert manifest["hash_index_ptx_signature"]
        assert manifest["ref_hash_resolve_ptx_signature"]
        assert manifest["reverse_symlink_ptx_signature"]
        assert manifest["reverse_ref_hash_ptx_signature"]
        assert manifest["route_capability_trit_ptx_signature"]
        assert manifest["meaning_family_route_audit_path"]
        assert manifest["meaning_family_route_audit_passed"] is False
        assert manifest["meaning_route_closure_audit_path"]
        assert manifest["meaning_route_closure_audit_passed"] is False
        assert manifest["meaning_knowledge_coverage_audit_path"]
        assert manifest["meaning_knowledge_coverage_audit_passed"] is False
        assert artifact["mode"] == "artifact"
        assert artifact["default_knowledge_signature"] == "unit_default_sig"
        assert artifact["house_signature_base"] == "unit_house_sig"
    finally:
        runtime = getattr(kv, "_sovereign_hot_path", None)
        if runtime is not None:
            runtime.close()
            kv._sovereign_hot_path = None


def test_force_rebuild_fails_fast_on_missing_routing_metadata(tmp_path, monkeypatch):
    storage_root = tmp_path / "kv_missing_metadata"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "meta_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "meta_house"

    broken_catalog = [
        {
            "galaxy": "Math",
            "id": "broken_router",
            "domain_hash": 0.1,
            "subject_hash": 0.2,
            "embedding16": [0.125] * 16,
            "confidence": 0.9,
            "gpu_source_class": 1,
        }
    ]
    broken_source = {
        "id": "broken_router",
        "selection_role": "router",
        "layer_id": 4,
        "executor_refs": ["math_compute_executor"],
        "validator_refs": ["math_answer_validator"],
        "embedding16": [0.125] * 16,
        # missing answer_eligible and route_policy on purpose
    }

    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math"])
    monkeypatch.setattr(kv, "build_gpu_catalog_only", lambda *, galaxy_names=None: list(broken_catalog))
    monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: dict(broken_source))

    runtime = kv._get_sovereign_hot_path()
    with pytest.raises(ValueError, match="sovereign_build_metadata_invalid"):
        runtime.refresh_feed_source(catalog=list(broken_catalog))


def test_route_exempt_utility_registry_normalizes_legacy_operator_entries():
    source = next(
        entry
        for entry in default_3d_objects_entries()
        if str(entry.get("id") or "") == "obj3d_xform_apply"
    )
    normalized = normalize_disk_entry("3DObjects", dict(source))
    metadata = dict(normalized.get("metadata") or {})
    registry = resident_route_registry_summary()

    assert registry["route_exempt_utility"] >= 1
    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_family" not in normalized
    assert metadata["selection_role"] == "unknown"
    assert metadata["layer_id"] == 0
    assert metadata["answer_eligible"] is False
    assert metadata["sovereign_route_exempt"] is True


def test_route_exempt_utility_registry_normalizes_tool_surface_entries():
    source = next(
        entry
        for entry in default_tool_entries()
        if str(entry.get("id") or "") == "tool_mathcore_tier3_master_v1"
    )
    normalized = normalize_disk_entry("Tool", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_family" not in normalized
    assert metadata["selection_role"] == "unknown"
    assert metadata["layer_id"] == 0
    assert metadata["answer_eligible"] is False
    assert metadata["sovereign_route_exempt"] is True


def test_route_exempt_registry_normalizes_language_book_anchor_entries():
    source = {
        "id": "langbook_sec3_literals",
        "galaxy": "Book/LanguageFoundations",
        "category": "meaning_star",
        "metadata": {
            "router_refs": ["grammar_parse_executor"],
        },
    }
    normalized = normalize_disk_entry("Book/LanguageFoundations", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_family" not in normalized
    assert "router_refs" not in normalized
    assert metadata["sovereign_route_exempt"] is True


def test_route_exempt_registry_normalizes_halting_threshold_entries():
    source = {
        "id": "halting_threshold_math",
        "galaxy": "Tool",
        "category": "meta_rule",
        "selection_role": "validator",
        "metadata": {
            "selection_role": "validator",
            "layer": 4,
            "minimum_threshold": 0.3,
            "gap_threshold": 0.04,
            "agreement_threshold": 1.0,
        },
    }
    normalized = normalize_disk_entry("Tool", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_family" not in normalized
    assert "route_policy" not in normalized
    assert metadata["selection_role"] == "unknown"
    assert metadata["layer_id"] == 0
    assert metadata["answer_eligible"] is False
    assert metadata["sovereign_route_exempt"] is True


def test_route_exempt_registry_neutralizes_failure_patch_router_entries():
    source = {
        "id": "synset_08798062_n",
        "galaxy": "meaning_layer_stars",
        "category": "noun",
        "metadata": {
            "selection_role": "router",
            "router_refs": [
                "meta_four_way_reading_strategy",
                "meta_decompose_multi_step_word_problem",
            ],
            "failure_patch_family": "MATH",
            "failure_patch_source": "/tmp/failure_patch_log",
        },
    }
    normalized = normalize_disk_entry("meaning_layer_stars", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_family" not in normalized
    assert "router_refs" not in normalized
    assert metadata["selection_role"] == "unknown"
    assert metadata["layer_id"] == 0
    assert metadata["answer_eligible"] is False
    assert metadata["sovereign_route_exempt"] is True
    assert metadata["failure_patch_family"] == "MATH"


def test_route_exempt_registry_normalizes_canonical_meaning_layer_synset_entries():
    source = {
        "id": "synset_08798062_n",
        "galaxy": "meaning_layer_stars",
        "category": "noun",
        "metadata": {
            "meaning_star_id": "synset_08798062_n",
            "router_refs": [
                "meta_four_way_reading_strategy",
                "meta_decompose_multi_step_word_problem",
            ],
        },
    }
    normalized = normalize_disk_entry("meaning_layer_stars", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "router_refs" not in normalized
    assert metadata["selection_role"] == "unknown"
    assert metadata["answer_eligible"] is False
    assert metadata["sovereign_route_exempt"] is True
    assert metadata["meaning_star_id"] == "synset_08798062_n"


def test_route_exempt_registry_normalizes_reasoning_meaning_anchor_entries():
    source = {
        "id": "forward_entity_extraction",
        "galaxy": "reasoning_strategies",
        "category": "reasoning_strategy",
        "metadata": {
            "selection_role": "router",
            "route_policy": {"requires_executor": True},
            "grammar_refs": ["grammar_forward_entity_extraction"],
        },
    }
    normalized = normalize_disk_entry("reasoning_strategies", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_policy" not in normalized
    assert metadata["sovereign_route_exempt"] is True
    assert metadata["route_contract_schema_version"] == route_contract.ROUTE_CONTRACT_SCHEMA_VERSION


def test_route_exempt_registry_normalizes_reasoning_reality_support_rows():
    source = {
        "id": "reality_word_problem_goal_state",
        "galaxy": "Reality",
        "category": "goal_state",
        "metadata": {
            "selection_role": "router",
            "route_policy": {"answer_gate": True},
        },
    }
    normalized = normalize_disk_entry("Reality", dict(source))

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_policy" not in normalized


def test_route_exempt_registry_normalizes_foundational_reality_anchor_entries():
    source = {
        "id": "reality_anchor_college_physics_core",
        "galaxy": "Reality",
        "category": "definition",
        "metadata": {
            "selection_role": "router",
            "query_anchor": "college physics mechanics optics thermodynamics",
        },
    }
    normalized = normalize_disk_entry("Reality", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_family" not in normalized
    assert metadata["selection_role"] == "unknown"
    assert metadata["layer_id"] == 0
    assert metadata["answer_eligible"] is False
    assert metadata["sovereign_route_exempt"] is True


def test_route_exempt_registry_normalizes_game_mechanic_anchor_entries():
    source = {
        "id": "spatial_navigation_grid",
        "galaxy": "game_mechanics",
        "category": "game_mechanic",
        "metadata": {
            "selection_role": "router",
            "route_policy": {"branch_topk": 2},
        },
    }
    normalized = normalize_disk_entry("game_mechanics", dict(source))

    assert normalized["selection_role"] == "unknown"
    assert normalized["layer_id"] == 0
    assert normalized["answer_eligible"] is False
    assert normalized["sovereign_route_exempt"] is True
    assert "route_policy" not in normalized


def test_route_capable_registry_promotes_reasoning_pattern_entries():
    source = {
        "id": "pattern_arithmetic_next",
        "galaxy": "Grammar",
        "category": "pattern_rule",
        "metadata": {
            "router_refs": ["grammar_transform_executor"],
        },
    }
    normalized = normalize_disk_entry("Grammar", dict(source))
    metadata = dict(normalized.get("metadata") or {})
    registry = resident_route_registry_summary()

    assert registry["route_capable_legacy"] >= 16
    assert normalized["route_family"] == "GRAMMAR"
    assert normalized["selection_role"] == "executor"
    assert normalized["layer_id"] == 3
    assert normalized["answer_eligible"] is False
    assert normalized["route_policy"] == {
        "requires_validator": True,
        "answer_gate": True,
        "branch_topk": 2,
    }
    assert normalized["executor_refs"] == [
        "grammar_transform_executor",
        "grammar_answer_materializer",
    ]
    assert normalized["validator_refs"] == [
        "grammar_normalization_validator",
        "grammar_answer_validator",
    ]
    assert normalized["anti_pattern_refs"] == [
        "anti_pattern_missing_validator_traversal",
        "anti_pattern_answer_format_mismatch",
    ]
    assert metadata["route_family"] == "GRAMMAR"
    assert metadata["selection_role"] == "executor"
    assert metadata["layer_id"] == 3
    assert metadata["answer_eligible"] is False
    assert metadata["route_policy"] == {
        "requires_validator": True,
        "answer_gate": True,
        "branch_topk": 2,
    }
    assert metadata["executor_refs"] == [
        "grammar_transform_executor",
        "grammar_answer_materializer",
    ]
    assert metadata["validator_refs"] == [
        "grammar_normalization_validator",
        "grammar_answer_validator",
    ]
    assert metadata["router_refs"] == ["grammar_transform_executor"]


def test_route_capable_registry_promotes_consume_from_total_entry():
    source = {
        "id": "consume_from_total",
        "galaxy": "Grammar",
        "category": "math_word_problem",
        "metadata": {
            "router_refs": ["math_remainder_after_use_executor"],
        },
    }
    normalized = normalize_disk_entry("Grammar", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["route_family"] == "GRAMMAR"
    assert normalized["selection_role"] == "executor"
    assert normalized["layer_id"] == 3
    assert normalized["answer_eligible"] is False
    assert normalized["route_policy"] == {
        "requires_validator": True,
        "answer_gate": True,
        "branch_topk": 2,
    }
    assert normalized["executor_refs"] == [
        "grammar_transform_executor",
        "grammar_answer_materializer",
    ]
    assert normalized["validator_refs"] == [
        "grammar_normalization_validator",
        "grammar_answer_validator",
    ]
    assert normalized["anti_pattern_refs"] == [
        "anti_pattern_missing_validator_traversal",
        "anti_pattern_answer_format_mismatch",
    ]
    assert metadata["router_refs"] == ["math_remainder_after_use_executor"]


def test_route_capable_registry_promotes_grammar_support_executor_entries():
    source = {
        "id": "grammar_forward_entity_extraction",
        "galaxy": "Grammar",
        "category": "reading_rule",
        "metadata": {
            "selection_role": "executor",
        },
    }
    normalized = normalize_disk_entry("Grammar", dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["route_family"] == "GRAMMAR"
    assert normalized["selection_role"] == "executor"
    assert normalized["layer_id"] == 3
    assert normalized["answer_eligible"] is False
    assert normalized["route_policy"] == {
        "requires_validator": True,
        "answer_gate": False,
        "branch_topk": 2,
    }
    assert normalized["validator_refs"] == [
        "grammar_normalization_validator",
        "grammar_answer_validator",
    ]
    assert metadata["route_contract_schema_version"] == route_contract.ROUTE_CONTRACT_SCHEMA_VERSION


def test_foundational_operations_bootstrap_emits_math_word_problem_route_contracts():
    grammar_entries = {
        str(entry.get("id") or ""): entry
        for entry in _grammar_entries()
    }
    consume = grammar_entries["consume_from_total"]
    metadata = dict(consume.get("metadata") or {})

    assert metadata["route_family"] == "GRAMMAR"
    assert metadata["selection_role"] == "executor"
    assert metadata["layer_id"] == 3
    assert metadata["answer_eligible"] is False
    assert metadata["route_policy"] == {
        "requires_validator": True,
        "answer_gate": True,
        "branch_topk": 2,
    }
    assert metadata["route_contract_schema_version"] == route_contract.ROUTE_CONTRACT_SCHEMA_VERSION
    assert metadata["router_refs"] == [
        "grammar_transform_executor",
        "grammar_normalization_validator",
        "grammar_answer_validator",
        "grammar_answer_materializer",
        "math_remainder_after_use_executor",
    ]


def test_foundational_reality_bootstrap_emits_route_exempt_domain_support_entries():
    reality_entries = {
        str(entry.get("id") or ""): entry
        for entry in _foundational_reality_entries()
    }
    anchor = reality_entries["reality_anchor_computer_science_core"]
    anchor_metadata = dict(anchor.get("metadata") or {})
    chess = reality_entries["reality_chess_mating_patterns"]
    chess_metadata = dict(chess.get("metadata") or {})

    assert anchor["selection_role"] == "unknown"
    assert anchor["layer_id"] == 0
    assert anchor["answer_eligible"] is False
    assert anchor["sovereign_route_exempt"] is True
    assert anchor_metadata["layer"] == 2
    assert anchor_metadata["route_contract_schema_version"] == route_contract.ROUTE_CONTRACT_SCHEMA_VERSION

    assert chess["selection_role"] == "unknown"
    assert chess["layer_id"] == 0
    assert chess["answer_eligible"] is False
    assert chess["sovereign_route_exempt"] is True
    assert chess_metadata["layer"] == 2
    assert chess_metadata["route_contract_schema_version"] == route_contract.ROUTE_CONTRACT_SCHEMA_VERSION


@pytest.mark.parametrize(
    ("entry_id", "galaxy", "expected_executor_refs"),
    [
        (
            "reasoning_factual_lookup_top1",
            "General",
            ["general_lookup_executor", "general_answer_materializer"],
        ),
        (
            "reasoning_elimination_top1",
            "Question",
            ["question_option_elimination_executor", "question_choice_materializer"],
        ),
        (
            "quantity_role_initial",
            "Math",
            ["math_goal_trace_executor", "math_operation_chain_executor"],
        ),
        (
            "reasoning_chat_lookup_top1",
            "Chat",
            ["chat_grounding_executor"],
        ),
    ],
)
def test_route_capable_registry_wires_legacy_executors_to_materializer_chains(
    entry_id: str,
    galaxy: str,
    expected_executor_refs: list[str],
):
    source = {
        "id": entry_id,
        "galaxy": galaxy,
        "category": "reasoning_rule",
        "metadata": {},
    }
    normalized = normalize_disk_entry(galaxy, dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "executor"
    assert normalized["executor_refs"] == expected_executor_refs
    assert metadata["executor_refs"] == expected_executor_refs


@pytest.mark.parametrize(
    ("entry_id", "galaxy", "materializer_id"),
    [
        ("meta_four_way_reading_strategy", "Math", "math_answer_materializer"),
        ("meta_decompose_multi_step_word_problem", "Math", "math_answer_materializer"),
        ("meta_apply_backward_trace_before_emit", "Math", "math_answer_materializer"),
        ("meta_validate_units_before_answer", "Math", "math_answer_materializer"),
        ("meta_template_slot_binding", "Math", "math_answer_materializer"),
        ("meta_route_question_subject_before_elimination", "Question", "question_choice_materializer"),
        ("meta_verify_option_before_emit", "Question", "question_choice_materializer"),
    ],
)
def test_route_capable_registry_wires_meta_routers_to_family_materializers(
    entry_id: str,
    galaxy: str,
    materializer_id: str,
):
    source = {
        "id": entry_id,
        "galaxy": galaxy,
        "category": "meta_rule",
        "executor_refs": ["legacy_support_executor"],
        "metadata": {
            "executor_refs": ["legacy_support_executor"],
        },
    }
    normalized = normalize_disk_entry(galaxy, dict(source))
    metadata = dict(normalized.get("metadata") or {})

    assert normalized["selection_role"] == "router"
    assert normalized["route_policy"]["answer_gate"] is True
    assert materializer_id in normalized["executor_refs"]
    assert "legacy_support_executor" in normalized["executor_refs"]
    assert materializer_id in metadata["executor_refs"]
    assert "legacy_support_executor" in metadata["executor_refs"]


def test_duplicate_id_registry_normalizes_foundational_word_problem_aliases():
    grammar_source = {
        "id": "alias_word_problem_total_word_problem_total",
        "name": "WORD_PROBLEM_TOTAL Alias: word problem total",
        "domain": "grammar",
        "category": "alias_rule",
        "metadata": {
            "bootstrap": "deterministic_foundation_v1",
            "operation": "WORD_PROBLEM_TOTAL",
        },
    }
    math_source = {
        "id": "alias_word_problem_total_word_problem_total",
        "name": "WORD_PROBLEM_TOTAL Alias: word problem total",
        "domain": "math",
        "category": "alias_rule",
        "metadata": {
            "bootstrap": "deterministic_foundation_v1",
            "operation": "WORD_PROBLEM_TOTAL",
        },
    }
    grammar_normalized = normalize_disk_entry("Grammar", dict(grammar_source))
    math_normalized = normalize_disk_entry("Math", dict(math_source))
    registry = resident_route_registry_summary()

    assert registry["duplicate_id_repairs"] >= 10
    assert grammar_normalized["id"] == "grammar_alias_word_problem_total_word_problem_total"
    assert math_normalized["id"] == "math_alias_word_problem_total_word_problem_total"
    assert grammar_normalized["metadata"]["resident_id_repaired_from"] == "alias_word_problem_total_word_problem_total"
    assert math_normalized["metadata"]["resident_id_repaired_from"] == "alias_word_problem_total_word_problem_total"


def test_duplicate_id_registry_namespaces_language_meaning_layer_mirror_entries():
    source = {
        "id": "synset_00001740_a",
        "galaxy": "Language",
        "category": "meaning_star",
        "metadata": {
            "ingest_source": "meaning_layer",
            "meaning_star_id": "synset_00001740_a",
        },
    }

    normalized = normalize_disk_entry("Language", dict(source))

    assert normalized["id"] == "language_synset_00001740_a"
    assert normalized["metadata"]["meaning_star_id"] == "synset_00001740_a"
    assert normalized["metadata"]["resident_id_repaired_from"] == "synset_00001740_a"
    assert normalized["metadata"]["resident_id_repair_reason"] == "language_meaning_mirror_namespace"


def test_duplicate_id_registry_renames_reasoning_quantity_anchor_entries():
    source = {
        "id": "quantity_role_initial",
        "galaxy": "reasoning_strategies",
        "category": "reasoning_strategy",
        "metadata": {
            "grammar_refs": ["grammar_quantity_role_initial"],
        },
    }

    normalized = normalize_disk_entry("reasoning_strategies", dict(source))

    assert normalized["id"] == route_contract.REASONING_ANCHOR_ID_REPAIRS["quantity_role_initial"]
    assert normalized["metadata"]["resident_id_repaired_from"] == "quantity_role_initial"
    assert normalized["metadata"]["resident_id_repair_reason"] == "explicit_duplicate_id_override"
    assert normalized["sovereign_route_exempt"] is True


def test_feed_source_accepts_explicit_route_exempt_utility_entries(tmp_path, monkeypatch):
    storage_root = tmp_path / "kv_route_exempt_utility"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    catalog = [
        {
            "galaxy": "3DObjects",
            "id": "obj3d_xform_apply",
            "domain_hash": 0.25,
            "subject_hash": 0.75,
            "embedding16": [0.0625] * 16,
            "confidence": 0.9,
            "gpu_source_class": 0,
        }
    ]
    source = normalize_disk_entry(
        "3DObjects",
        next(
            entry
            for entry in default_3d_objects_entries()
            if str(entry.get("id") or "") == "obj3d_xform_apply"
        ),
    )
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["3DObjects"])
    monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: dict(source))

    runtime = kv._get_sovereign_hot_path()
    summary = runtime.refresh_feed_source(catalog=list(catalog))
    feed_source = runtime._load_feed_source(str(summary["feed_source_signature"]))
    entry = dict(feed_source["host_stars"][0])

    assert summary["mode"] == "feed_source_compile"
    assert entry["id"] == "obj3d_xform_apply"
    assert entry["selection_role"] == "unknown"
    assert entry["layer_id"] == 0
    assert entry["route_family"] == ""
    assert entry["answer_eligible"] is False
    assert entry["router_refs"] == []
    assert entry["executor_refs"] == []
    assert entry["validator_refs"] == []
    assert entry["anti_pattern_refs"] == []


def test_feed_source_parallel_matches_serial_output(tmp_path, monkeypatch):
    catalog, source_map = _route_family_catalog_and_sources()

    def _compile(storage_root, workers, chunk):
        kv = Knowledgeverse(
            storage_root=storage_root,
            eager_load_default_galaxies=False,
            start_live_loops=False,
        )
        monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question", "Grammar", "General", "Chat"])
        monkeypatch.setattr(
            kv,
            "_catalog_source_entry",
            lambda row: dict(source_map[str(row.get("id") or "")]),
        )
        runtime = kv._get_sovereign_hot_path()
        summary = runtime.refresh_feed_source(
            catalog=list(catalog),
            worker_count=workers,
            chunk_size=chunk,
        )
        feed_source = runtime._load_feed_source(str(summary["feed_source_signature"]))
        rows = Path(feed_source["paths"]["rows"]).read_bytes()
        refs = Path(feed_source["paths"]["refs"]).read_bytes()
        host_stars = [dict(star) for star in list(feed_source["host_stars"] or [])]
        runtime.close()
        kv._sovereign_hot_path = None
        return summary, rows, refs, host_stars

    serial_summary, serial_rows, serial_refs, serial_host_stars = _compile(
        tmp_path / "kv_feed_source_serial",
        1,
        3,
    )
    parallel_summary, parallel_rows, parallel_refs, parallel_host_stars = _compile(
        tmp_path / "kv_feed_source_parallel",
        2,
        3,
    )

    assert serial_summary["mode"] == "feed_source_compile"
    assert parallel_summary["mode"] == "feed_source_compile"
    assert serial_summary["feed_source_parallel"] is False
    assert parallel_summary["feed_source_parallel"] is True
    assert serial_rows == parallel_rows
    assert serial_refs == parallel_refs
    assert serial_host_stars == parallel_host_stars


def test_feed_source_duplicate_id_error_names_offending_indices(tmp_path, monkeypatch):
    storage_root = tmp_path / "kv_duplicate_id"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    catalog = [
        {
            "galaxy": "Math",
            "id": "duplicate_router",
            "domain_hash": 1.0,
            "subject_hash": 1.0,
            "embedding16": [0.0625] * 16,
            "confidence": 0.9,
            "gpu_source_class": 1,
        },
        {
            "galaxy": "Question",
            "id": "duplicate_router",
            "domain_hash": 2.0,
            "subject_hash": 2.0,
            "embedding16": [0.125] * 16,
            "confidence": 0.9,
            "gpu_source_class": 1,
        },
    ]
    source = {
        "id": "duplicate_router",
        "name": "Duplicate Router",
        "selection_role": "router",
        "layer_id": 4,
        "answer_eligible": False,
        "executor_refs": ["math_compute_executor"],
        "validator_refs": ["math_answer_validator"],
        "route_policy": {
            "requires_executor": True,
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 1,
        },
        "embedding16": [0.25] * 16,
    }
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question"])
    monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: dict(source))

    runtime = kv._get_sovereign_hot_path()
    with pytest.raises(
        ValueError,
        match=r"sovereign_feed_source_duplicate_id:duplicate_router:first_index=0:second_index=1",
    ):
        runtime.refresh_feed_source(catalog=list(catalog), worker_count=1, chunk_size=1)


def test_feed_source_uses_catalog_row_route_metadata_when_source_is_sparse(tmp_path, monkeypatch):
    storage_root = tmp_path / "kv_catalog_metadata_route_refs"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    source_map = {
        "question_reasoning_anchor_sparse": {
            "id": "question_reasoning_anchor_sparse",
            "name": "Question Reasoning Anchor",
        },
        "question_evidence_validator": next(
            dict(star)
            for star in build_foundational_galaxy_table()
            if str(star.get("id") or "") == "question_evidence_validator"
        ),
        "question_answer_validator": next(
            dict(star)
            for star in build_foundational_galaxy_table()
            if str(star.get("id") or "") == "question_answer_validator"
        ),
    }
    catalog = [
        {
            "galaxy": "Question",
            "id": "question_reasoning_anchor_sparse",
            "domain_hash": 1.0,
            "subject_hash": 1.0,
            "embedding16": [0.0625] * 16,
            "confidence": 0.82,
            "gpu_source_class": 0,
            "metadata": {
                "route_family": "QUESTION",
                "selection_role": "executor",
                "layer_id": 3,
                "answer_eligible": False,
                "executor_refs": ["question_option_elimination_executor"],
                "validator_refs": [
                    "question_evidence_validator",
                    "question_answer_validator",
                ],
                "anti_pattern_refs": [
                    "anti_pattern_missing_evidence_consistency",
                ],
                "route_policy": {
                    "requires_validator": True,
                    "answer_gate": True,
                    "branch_topk": 2,
                },
            },
        },
        {
            "galaxy": "Question",
            "id": "question_evidence_validator",
            "domain_hash": 2.0,
            "subject_hash": 2.0,
            "embedding16": [0.125] * 16,
            "confidence": 0.82,
            "gpu_source_class": 0,
        },
        {
            "galaxy": "Question",
            "id": "question_answer_validator",
            "domain_hash": 3.0,
            "subject_hash": 3.0,
            "embedding16": [0.1875] * 16,
            "confidence": 0.82,
            "gpu_source_class": 0,
        },
    ]

    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Question"])
    monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: dict(source_map[str(row.get("id") or "")]))

    runtime = kv._get_sovereign_hot_path()
    summary = runtime.refresh_feed_source(catalog=list(catalog), worker_count=1, chunk_size=1)
    feed_source = runtime._load_feed_source(str(summary["feed_source_signature"]))
    host_star = next(
        dict(star)
        for star in list(feed_source["host_stars"] or [])
        if str(star.get("id") or "") == "question_reasoning_anchor_sparse"
    )

    assert summary["mode"] == "feed_source_compile"
    assert len(list(host_star.get("validator_refs") or [])) == 2


def test_feed_source_cache_reused_when_signature_matches(tmp_path, monkeypatch):
    catalog, source_map = _spine_catalog_and_sources()
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_feed_source_cached",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "unit_default_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "unit_house_sig"
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question"])
    monkeypatch.setattr(kv, "build_gpu_catalog_only", lambda *, galaxy_names=None: list(catalog))
    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )

    runtime = kv._get_sovereign_hot_path()
    compiled = runtime.refresh_feed_source(catalog=list(catalog), worker_count=1, chunk_size=2)
    cached = runtime.refresh_feed_source()

    assert compiled["mode"] == "feed_source_compile"
    assert cached["mode"] == "feed_source_cached"
    assert cached["feed_source_signature"] == compiled["feed_source_signature"]
    assert cached["feed_source_worker_count"] == compiled["feed_source_worker_count"]
    assert cached["feed_source_chunk_size"] == compiled["feed_source_chunk_size"]


def test_force_rebuild_fails_fast_when_build_feed_is_missing(tmp_path):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_missing_feed",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "feed_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "house_sig"

    with pytest.raises(RuntimeError, match="sovereign_build_feed_missing"):
        kv._boot_sovereign_runtime(force_reload=True, force_rebuild=True)


def test_build_feed_fails_fast_when_feed_source_is_missing(tmp_path):
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_missing_feed_source",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "feed_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "house_sig"

    runtime = kv._get_sovereign_hot_path()
    with pytest.raises(RuntimeError, match="sovereign_feed_source_missing"):
        runtime.refresh_build_feed()


def test_force_rebuild_path_does_not_use_python_math_helpers(tmp_path, monkeypatch):
    catalog, source_map = _spine_catalog_and_sources()
    storage_root = tmp_path / "kv_no_python_math"
    kv = Knowledgeverse(
        storage_root=storage_root,
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "unit_default_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "unit_house_sig"

    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question"])
    monkeypatch.setattr(kv, "build_gpu_catalog_only", lambda *, galaxy_names=None: list(catalog))
    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )

    runtime = kv._get_sovereign_hot_path()
    runtime.refresh_feed_source(catalog=list(catalog))

    def _boom(*_args, **_kwargs):
        raise AssertionError("python_math_helper_reached")

    monkeypatch.setattr("knowledge3d.knowledgeverse.knowledgeverse.math.isclose", _boom)
    monkeypatch.setattr("knowledge3d.knowledgeverse.knowledgeverse.math.isfinite", _boom)
    monkeypatch.setattr("knowledge3d.knowledgeverse.knowledgeverse.math.sqrt", _boom)
    monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: (_ for _ in ()).throw(AssertionError("catalog_source_reached")))
    monkeypatch.setattr(runtime, "_entry_metadata", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("entry_metadata_reached")))
    monkeypatch.setattr(runtime, "_translate_catalog_entries", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("translate_catalog_reached")))
    monkeypatch.setattr(runtime, "_ensure_bidirectional_symlinkage", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bidirectional_symlinkage_reached")))
    runtime.refresh_build_feed()

    rebuilt = kv._boot_sovereign_runtime(force_reload=True, force_rebuild=True)
    assert rebuilt["mode"] == "rebuilt"
    assert rebuilt["build_backend"] == "gpu_build_feed_v2"


def test_build_feed_compile_does_not_use_python_translation_helpers(tmp_path, monkeypatch):
    catalog, source_map = _spine_catalog_and_sources()
    kv = Knowledgeverse(
        storage_root=tmp_path / "kv_feed_source_only",
        eager_load_default_galaxies=False,
        start_live_loops=False,
    )
    kv._default_galaxies_loaded = True
    kv._house_state_summary["default_knowledge_signature"] = "unit_default_sig"
    kv._house_state_summary["gpu_buffer_signature_base"] = "unit_house_sig"
    monkeypatch.setattr(kv, "_discover_live_galaxy_names", lambda: ["Math", "Question"])
    monkeypatch.setattr(kv, "build_gpu_catalog_only", lambda *, galaxy_names=None: list(catalog))
    monkeypatch.setattr(
        kv,
        "_catalog_source_entry",
        lambda row: dict(source_map[str(row.get("id") or "")]),
    )
    runtime = kv._get_sovereign_hot_path()
    feed_source = runtime.refresh_feed_source(catalog=list(catalog))

    monkeypatch.setattr(kv, "_catalog_source_entry", lambda row: (_ for _ in ()).throw(AssertionError("catalog_source_reached")))
    monkeypatch.setattr(runtime, "_entry_metadata", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("entry_metadata_reached")))
    monkeypatch.setattr(runtime, "_translate_catalog_entries", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("translate_catalog_reached")))
    monkeypatch.setattr(runtime, "_ensure_bidirectional_symlinkage", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bidirectional_symlinkage_reached")))

    build_feed = runtime.refresh_build_feed()
    assert feed_source["mode"] == "feed_source_compile"
    assert build_feed["mode"] == "build_feed_compile"
    assert build_feed["route_valid_count"] >= 2


def test_runtime_flags_encode_route_family_bits():
    encoded = _encode_runtime_flags(0x03, "MATH")
    assert (encoded & 0x03) == 0x03
    assert ((encoded & ROUTE_FAMILY_FLAG_MASK) >> ROUTE_FAMILY_FLAG_SHIFT) == 2

    encoded_grammar = _encode_runtime_flags(0x01, "GRAMMAR")
    assert (encoded_grammar & 0x01) == 0x01
    assert ((encoded_grammar & ROUTE_FAMILY_FLAG_MASK) >> ROUTE_FAMILY_FLAG_SHIFT) == 6

    encoded_neutral = _encode_runtime_flags(0x01, "")
    assert (encoded_neutral & ROUTE_FAMILY_FLAG_MASK) == 0
