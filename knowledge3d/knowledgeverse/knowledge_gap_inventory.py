"""Checked-in curated knowledge-gap inventory for foundational recovery waves."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CURATED_MATH_QUESTION_WAVE_ID = "e71_curated_math_question_wave_1"
CURATED_MATH_QUESTION_ARTIFACT_ROOT = (
    "/K3D/Knowledge3D.local/results/e70b_shutdown_2026-04-05/benchmarks/stage1"
)

CURATED_FAILURE_CLUSTERS: dict[str, dict[str, Any]] = {
    "multiplicative_comparison": {
        "family": "MATH",
        "symptom": "wrong operation-family selection on times-more word problems",
        "target_ids": [
            "math_multiplicative_comparison_executor",
            "meta_route_math_operation_family_before_emit",
            "anti_pattern_wrong_math_template_family",
        ],
    },
    "part_whole_aggregation": {
        "family": "MATH",
        "symptom": "part-whole and fraction-of-total questions collapse into generic arithmetic chains",
        "target_ids": [
            "math_additive_composition_executor",
            "math_fraction_of_total_executor",
            "math_unit_preserving_projection_executor",
        ],
    },
    "rate_and_remainder_projection": {
        "family": "MATH",
        "symptom": "rate, revenue, and remainder stories lose quantity-role structure before projection",
        "target_ids": [
            "math_remainder_after_use_executor",
            "math_rate_revenue_projection_executor",
            "math_multi_step_composition_executor",
        ],
    },
    "subject_grounded_choice_alignment": {
        "family": "QUESTION",
        "symptom": "option elimination reaches validators before subject grounding and clue comparison",
        "target_ids": [
            "question_evidence_compare_executor",
            "meta_route_definition_lookup_before_answer_emit",
            "anti_pattern_shallow_subjectless_option_elimination",
        ],
    },
    "open_text_grounded_projection": {
        "family": "QUESTION",
        "symptom": "open-ended answers leak validator or anti-pattern text instead of grounded text answers",
        "target_ids": [
            "question_open_text_materializer",
            "general_open_text_materializer",
            "anti_pattern_anti_pattern_text_as_answer",
            "anti_pattern_validator_without_grounded_materialization",
            "anti_pattern_open_question_as_choice",
        ],
    },
}

FORM_LAYER_PACKET_IDS = (
    "form_marker_fraction",
    "form_marker_times_more",
    "form_marker_rate_per",
    "form_marker_currency",
    "form_marker_choice_a",
    "form_marker_choice_b",
    "form_marker_choice_c",
    "form_marker_choice_d",
    "form_marker_yes_no",
    "form_marker_true_false",
    "form_marker_open_answer_delimiter",
    "form_marker_chess_notation",
)

MEANING_LAYER_PACKET_IDS = (
    "meaning_quantity_initial",
    "meaning_quantity_delta",
    "meaning_quantity_remainder",
    "meaning_quantity_multiplier",
    "meaning_quantity_fractional_part",
    "meaning_quantity_unit_price",
    "meaning_quantity_total",
    "meaning_relation_times_more",
    "meaning_relation_part_of",
    "meaning_relation_remaining_after_use",
    "meaning_relation_rate_to_total",
    "meaning_relation_definition_of",
    "meaning_relation_supports_option",
    "meaning_relation_contradicts_option",
    "meaning_answer_numeric",
    "meaning_answer_choice",
    "meaning_answer_open_text",
    "meaning_question_factual_recall",
    "meaning_question_definition_lookup",
    "meaning_question_domain_subject_grounding",
    "meaning_question_open_ended_projection",
)

LHE_DOMAIN_PACKET_IDS = (
    "domain_packet_mathematics_core",
    "domain_packet_logic_core",
    "domain_packet_computer_science_core",
    "domain_packet_linguistics_core",
    "domain_packet_physics_core",
    "domain_packet_chemistry_core",
    "domain_packet_chess_core",
    "domain_packet_factual_recall_core",
)

CURATED_ROUTE_PACKET_IDS = (
    "math_additive_composition_executor",
    "math_multiplicative_comparison_executor",
    "math_fraction_of_total_executor",
    "math_remainder_after_use_executor",
    "math_rate_revenue_projection_executor",
    "math_multi_step_composition_executor",
    "math_unit_preserving_projection_executor",
    "question_definition_lookup_executor",
    "question_evidence_compare_executor",
    "question_open_text_materializer",
    "general_domain_grounding_executor",
    "general_definition_projection_executor",
    "general_open_text_materializer",
    "meta_route_open_question_before_choice_emit",
    "meta_route_definition_lookup_before_answer_emit",
    "meta_route_math_operation_family_before_emit",
    "anti_pattern_anti_pattern_text_as_answer",
    "anti_pattern_validator_without_grounded_materialization",
    "anti_pattern_wrong_math_template_family",
    "anti_pattern_shallow_subjectless_option_elimination",
    "anti_pattern_open_question_as_choice",
)

CURATED_COVERAGE_PACKETS: dict[str, dict[str, Any]] = {
    "math_form_layer_packet": {
        "kind": "knowledge_packet",
        "family": "MATH",
        "required_ids": list(FORM_LAYER_PACKET_IDS[:6]),
    },
    "math_meaning_layer_packet": {
        "kind": "knowledge_packet",
        "family": "MATH",
        "required_ids": [
            "meaning_quantity_initial",
            "meaning_quantity_delta",
            "meaning_quantity_remainder",
            "meaning_quantity_multiplier",
            "meaning_quantity_fractional_part",
            "meaning_quantity_unit_price",
            "meaning_quantity_total",
            "meaning_relation_times_more",
            "meaning_relation_part_of",
            "meaning_relation_remaining_after_use",
            "meaning_relation_rate_to_total",
            "meaning_answer_numeric",
        ],
    },
    "math_operation_family_packet": {
        "kind": "route_packet",
        "family": "MATH",
        "required_ids": [
            "math_question_router",
            "math_word_problem_executor",
            "math_additive_composition_executor",
            "math_multiplicative_comparison_executor",
            "math_fraction_of_total_executor",
            "math_remainder_after_use_executor",
            "math_rate_revenue_projection_executor",
            "math_multi_step_composition_executor",
            "math_unit_preserving_projection_executor",
            "math_answer_materializer",
            "math_normalization_validator",
            "math_unit_magnitude_validator",
            "math_answer_validator",
            "meta_route_math_operation_family_before_emit",
            "anti_pattern_wrong_math_template_family",
        ],
        "materializer_ids": ["math_answer_materializer"],
        "anti_pattern_ids": [
            "anti_pattern_wrong_math_template_family",
            "anti_pattern_numeric_without_materialization",
        ],
    },
    "question_choice_grounding_packet": {
        "kind": "route_packet",
        "family": "QUESTION",
        "required_ids": [
            "question_router",
            "question_subject_grounding_executor",
            "knowledge_lookup_executor",
            "question_definition_lookup_executor",
            "question_evidence_compare_executor",
            "question_choice_materializer",
            "question_evidence_validator",
            "question_choice_alignment_validator",
            "question_answer_validator",
            "meta_route_definition_lookup_before_answer_emit",
            "anti_pattern_shallow_subjectless_option_elimination",
        ],
        "materializer_ids": ["question_choice_materializer"],
        "anti_pattern_ids": [
            "anti_pattern_option_emission_without_comparison",
            "anti_pattern_shallow_subjectless_option_elimination",
        ],
    },
    "question_open_text_packet": {
        "kind": "route_packet",
        "family": "QUESTION",
        "required_ids": [
            "question_surface_bridge",
            "question_router",
            "question_subject_grounding_executor",
            "question_definition_lookup_executor",
            "question_open_text_materializer",
            "question_evidence_validator",
            "question_answer_validator",
            "meta_route_open_question_before_choice_emit",
            "anti_pattern_anti_pattern_text_as_answer",
            "anti_pattern_validator_without_grounded_materialization",
            "anti_pattern_open_question_as_choice",
        ],
        "materializer_ids": ["question_open_text_materializer"],
        "anti_pattern_ids": [
            "anti_pattern_anti_pattern_text_as_answer",
            "anti_pattern_validator_without_grounded_materialization",
            "anti_pattern_open_question_as_choice",
        ],
    },
    "general_open_text_packet": {
        "kind": "route_packet",
        "family": "GENERAL",
        "required_ids": [
            "general_surface_bridge",
            "general_router",
            "general_domain_grounding_executor",
            "general_definition_projection_executor",
            "general_open_text_materializer",
            "general_grounding_validator",
            "general_consistency_validator",
            "general_answer_validator",
        ],
        "materializer_ids": ["general_open_text_materializer"],
        "anti_pattern_ids": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_empty_route_dispatch",
        ],
    },
    "lhe_domain_packets": {
        "kind": "knowledge_packet",
        "family": "GENERAL",
        "required_ids": list(LHE_DOMAIN_PACKET_IDS),
    },
}


def curated_math_question_knowledge_gap_inventory() -> dict[str, Any]:
    return {
        "wave_id": CURATED_MATH_QUESTION_WAVE_ID,
        "artifact_root": CURATED_MATH_QUESTION_ARTIFACT_ROOT,
        "focus_families": ["MATH", "QUESTION", "GENERAL", "GRAMMAR"],
        "failure_clusters": deepcopy(CURATED_FAILURE_CLUSTERS),
        "form_layer_packets": list(FORM_LAYER_PACKET_IDS),
        "meaning_layer_packets": list(MEANING_LAYER_PACKET_IDS),
        "route_packets": list(CURATED_ROUTE_PACKET_IDS),
        "lhe_domain_packets": list(LHE_DOMAIN_PACKET_IDS),
        "coverage_packets": deepcopy(CURATED_COVERAGE_PACKETS),
    }


def curated_math_question_coverage_packets() -> dict[str, dict[str, Any]]:
    return deepcopy(CURATED_COVERAGE_PACKETS)


def curated_math_question_required_ids() -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for packet in curated_math_question_coverage_packets().values():
        for entry_id in list(packet.get("required_ids") or []):
            normalized = str(entry_id).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
    return tuple(ordered)


__all__ = [
    "CURATED_COVERAGE_PACKETS",
    "CURATED_FAILURE_CLUSTERS",
    "CURATED_MATH_QUESTION_ARTIFACT_ROOT",
    "CURATED_MATH_QUESTION_WAVE_ID",
    "CURATED_ROUTE_PACKET_IDS",
    "FORM_LAYER_PACKET_IDS",
    "LHE_DOMAIN_PACKET_IDS",
    "MEANING_LAYER_PACKET_IDS",
    "curated_math_question_coverage_packets",
    "curated_math_question_knowledge_gap_inventory",
    "curated_math_question_required_ids",
]
