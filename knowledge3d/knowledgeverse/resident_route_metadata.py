"""Resident-route metadata repair registry for sovereign rebuild maintenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .objects_3d_galaxy import default_3d_objects_entries
from .reality_galaxy import default_reality_entries
from . import route_contract
from .tool_galaxy import default_tool_entries


def _entry_ids(entries: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        str(entry.get("id") or "").strip()
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("id") or "").strip()
    )


ROUTE_EXEMPT_UTILITY_IDS = (
    _entry_ids(default_3d_objects_entries())
    | _entry_ids(default_reality_entries())
    | _entry_ids(default_tool_entries())
    | route_contract.HALTING_PROFILE_IDS
    | route_contract.LANGUAGE_ROUTE_EXEMPT_IDS
)

GAME2D_LEGACY_META_ROUTER_IDS = frozenset(
    {
        "meta_prefer_shortest_walkable_route",
        "meta_route_to_switch_when_locked",
        "meta_match_key_before_door",
        "meta_seek_recharge_when_budget_low",
        "meta_probe_transform_blocks_on_mismatch",
        "meta_confirm_unlock_before_exit",
        "meta_learn_from_visual_transition",
        "meta_click_to_advance_after_completion",
        "meta_exploration_vs_exploitation",
        "meta_safety_first_strategy",
        "meta_budget_management_meta",
        "meta_jump_over_gap",
        "meta_use_teleporter_when_route_breaks",
        "meta_follow_conveyor_flow",
        "meta_push_crate_to_enable_path",
        "meta_avoid_hazard_without_protection",
        "meta_read_visual_affordance_before_action",
        "meta_backtrack_when_dependency_unsatisfied",
        "meta_use_checkpoint_after_progress",
        "meta_start_from_title_state",
        "meta_dismiss_transition_before_navigation",
        "meta_reset_context_after_level_completion",
        "meta_compare_budget_to_route_before_commit",
        "meta_reset_before_budget_depletion",
        "meta_reperceive_after_failure_flash",
    }
)

MATH_LEGACY_META_ROUTER_IDS = frozenset(
    {
        "meta_four_way_reading_strategy",
        "meta_decompose_multi_step_word_problem",
        "meta_apply_backward_trace_before_emit",
        "meta_validate_units_before_answer",
        "meta_template_slot_binding",
    }
)

QUESTION_LEGACY_META_ROUTER_IDS = frozenset(
    {
        "meta_route_question_subject_before_elimination",
        "meta_verify_option_before_emit",
    }
)

QUESTION_LEGACY_META_ANTI_PATTERN_IDS = frozenset({"meta_avoid_router_as_final_answer"})
MATH_LEGACY_META_ANTI_PATTERN_IDS = frozenset({"meta_avoid_isolated_template_halt"})

IDENTITY_REPAIR_OVERRIDES: dict[tuple[str, str], str] = {
    ("grammar", "alias_word_problem_total_word_problem_total"): "grammar_alias_word_problem_total_word_problem_total",
    ("grammar", "alias_word_problem_remainder_word_problem_remainder"): "grammar_alias_word_problem_remainder_word_problem_remainder",
    ("grammar", "alias_word_problem_rate_word_problem_rate"): "grammar_alias_word_problem_rate_word_problem_rate",
    ("grammar", "alias_word_problem_comparison_word_problem_comparison"): "grammar_alias_word_problem_comparison_word_problem_comparison",
    ("grammar", "alias_word_problem_percentage_word_problem_percentage"): "grammar_alias_word_problem_percentage_word_problem_percentage",
    ("math", "alias_word_problem_total_word_problem_total"): "math_alias_word_problem_total_word_problem_total",
    ("math", "alias_word_problem_remainder_word_problem_remainder"): "math_alias_word_problem_remainder_word_problem_remainder",
    ("math", "alias_word_problem_rate_word_problem_rate"): "math_alias_word_problem_rate_word_problem_rate",
    ("math", "alias_word_problem_comparison_word_problem_comparison"): "math_alias_word_problem_comparison_word_problem_comparison",
    ("math", "alias_word_problem_percentage_word_problem_percentage"): "math_alias_word_problem_percentage_word_problem_percentage",
    ("reasoning_strategies", "quantity_role_initial"): "reasoning_quantity_role_initial",
    ("reasoning_strategies", "quantity_role_delta"): "reasoning_quantity_role_delta",
}

ROUTE_CAPABLE_LEGACY_OVERRIDES: dict[str, dict[str, Any]] = {
    "pattern_arithmetic_next": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_answer_format_mismatch",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "pattern_geometric_next": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_answer_format_mismatch",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "consume_from_total": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_answer_format_mismatch",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "rate_application": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_unit_magnitude_mismatch",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "sequential_computation": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_shallow_router_stop",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 3,
        },
    },
    "comparison_delta": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_answer_format_mismatch",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "percentage_application": {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "grammar_transform_executor",
            "grammar_answer_materializer",
        ],
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_unit_magnitude_mismatch",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "answer_final_stack": {
        "route_family": "GRAMMAR",
        "selection_role": "validator",
        "layer_id": 4,
        "answer_eligible": True,
        "anti_pattern_refs": [
            "anti_pattern_answer_format_mismatch",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {"branch_topk": 0},
    },
    "reasoning_factual_lookup_top1": {
        "route_family": "GENERAL",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "general_lookup_executor",
            "general_answer_materializer",
        ],
        "validator_refs": [
            "general_consistency_validator",
            "general_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_generic_language_factual_winner",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "reasoning_chat_lookup_top1": {
        "route_family": "CHAT",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "chat_grounding_executor",
        ],
        "validator_refs": [
            "chat_grounding_validator",
            "chat_response_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_chat_ungrounded_response",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "reasoning_elimination_top1": {
        "route_family": "QUESTION",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "question_option_elimination_executor",
            "question_choice_materializer",
        ],
        "validator_refs": [
            "question_evidence_validator",
            "question_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_option_emission_without_comparison",
            "anti_pattern_unsupported_option_leap",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "reasoning_elimination_option_score": {
        "route_family": "QUESTION",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "question_option_elimination_executor",
            "question_choice_materializer",
        ],
        "validator_refs": [
            "question_evidence_validator",
            "question_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_option_emission_without_comparison",
            "anti_pattern_unsupported_option_leap",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "reasoning_comparison_top1": {
        "route_family": "GENERAL",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "general_compare_executor",
            "general_answer_materializer",
        ],
        "validator_refs": [
            "general_consistency_validator",
            "general_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_generic_language_factual_winner",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "reasoning_definition_top1": {
        "route_family": "GENERAL",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "general_lookup_executor",
            "general_answer_materializer",
        ],
        "validator_refs": [
            "general_consistency_validator",
            "general_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_generic_language_factual_winner",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "quantity_role_initial": {
        "route_family": "MATH",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "math_goal_trace_executor",
            "math_operation_chain_executor",
        ],
        "validator_refs": [
            "math_normalization_validator",
            "math_unit_magnitude_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "quantity_role_delta": {
        "route_family": "MATH",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "math_goal_trace_executor",
            "math_operation_chain_executor",
        ],
        "validator_refs": [
            "math_normalization_validator",
            "math_unit_magnitude_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
    "goal_type_factual_recall": {
        "route_family": "GENERAL",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "executor_refs": [
            "general_lookup_executor",
            "general_answer_materializer",
        ],
        "validator_refs": [
            "general_consistency_validator",
            "general_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_generic_language_factual_winner",
        ],
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
    },
}


def _router_override(
    route_family: str,
    *,
    branch_topk: int,
    executor_refs: list[str] | None = None,
    validator_refs: list[str] | None = None,
    anti_pattern_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "route_family": route_family,
        "selection_role": "router",
        "layer_id": 4,
        "answer_eligible": False,
        "executor_refs": list(executor_refs or []),
        "validator_refs": list(validator_refs or []),
        "anti_pattern_refs": list(anti_pattern_refs or []),
        "route_policy": {
            "requires_executor": True,
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": int(branch_topk),
        },
    }


def _anti_pattern_override(
    route_family: str,
    *,
    anti_pattern_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "route_family": route_family,
        "selection_role": "anti_pattern",
        "layer_id": 4,
        "answer_eligible": False,
        "anti_pattern_refs": list(anti_pattern_refs or []),
    }


ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        entry_id: _router_override(
            "GAME_2D",
            branch_topk=4,
            executor_refs=[
                "game2d_state_parse_executor",
                "game2d_delta_extractor_executor",
                "game2d_transform_inference_executor",
                "game2d_action_materializer",
                "game2d_grid_materializer",
            ],
            validator_refs=[
                "game2d_state_transition_validator",
                "game2d_output_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_wrong_family_grounding",
                "anti_pattern_action_without_state_transition",
                "anti_pattern_grid_without_transform_inference",
            ],
        )
        for entry_id in GAME2D_LEGACY_META_ROUTER_IDS
    }
)

ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        entry_id: _router_override(
            "MATH",
            branch_topk=3,
            validator_refs=[
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        )
        for entry_id in MATH_LEGACY_META_ROUTER_IDS
    }
)

ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        "meta_four_way_reading_strategy": _router_override(
            "MATH",
            branch_topk=3,
            executor_refs=[
                "math_quantity_binding_executor",
                "math_goal_trace_executor",
                "math_operation_chain_executor",
                "math_answer_materializer",
            ],
            validator_refs=[
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        ),
        "meta_decompose_multi_step_word_problem": _router_override(
            "MATH",
            branch_topk=3,
            executor_refs=[
                "math_word_problem_executor",
                "math_operation_chain_executor",
                "math_answer_materializer",
            ],
            validator_refs=[
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        ),
        "meta_apply_backward_trace_before_emit": _router_override(
            "MATH",
            branch_topk=3,
            executor_refs=[
                "math_goal_trace_executor",
                "math_operation_chain_executor",
                "math_answer_materializer",
            ],
            validator_refs=[
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        ),
        "meta_validate_units_before_answer": _router_override(
            "MATH",
            branch_topk=3,
            executor_refs=[
                "math_answer_materializer",
            ],
            validator_refs=[
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        ),
        "meta_template_slot_binding": _router_override(
            "MATH",
            branch_topk=3,
            executor_refs=[
                "math_quantity_binding_executor",
                "math_operation_chain_executor",
                "math_answer_materializer",
            ],
            validator_refs=[
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        ),
    }
)

ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        entry_id: _router_override(
            "QUESTION",
            branch_topk=3,
            validator_refs=[
                "question_evidence_validator",
                "question_choice_alignment_validator",
                "question_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_option_emission_without_comparison",
                "anti_pattern_validator_as_answer_leakage",
                "anti_pattern_empty_route_dispatch",
            ],
        )
        for entry_id in QUESTION_LEGACY_META_ROUTER_IDS
    }
)

ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        "meta_route_question_subject_before_elimination": _router_override(
            "QUESTION",
            branch_topk=3,
            executor_refs=[
                "question_subject_grounding_executor",
                "knowledge_lookup_executor",
                "question_option_elimination_executor",
                "question_choice_materializer",
            ],
            validator_refs=[
                "question_evidence_validator",
                "question_choice_alignment_validator",
                "question_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_option_emission_without_comparison",
                "anti_pattern_validator_as_answer_leakage",
                "anti_pattern_empty_route_dispatch",
            ],
        ),
        "meta_verify_option_before_emit": _router_override(
            "QUESTION",
            branch_topk=3,
            executor_refs=[
                "knowledge_lookup_executor",
                "question_option_elimination_executor",
                "question_choice_materializer",
            ],
            validator_refs=[
                "question_evidence_validator",
                "question_choice_alignment_validator",
                "question_answer_validator",
            ],
            anti_pattern_refs=[
                "anti_pattern_option_emission_without_comparison",
                "anti_pattern_validator_as_answer_leakage",
                "anti_pattern_empty_route_dispatch",
            ],
        ),
    }
)

ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        entry_id: _anti_pattern_override(
            "QUESTION",
            anti_pattern_refs=[
                "anti_pattern_missing_validator_traversal",
                "anti_pattern_unsupported_option_leap",
                "anti_pattern_option_emission_without_comparison",
            ],
        )
        for entry_id in QUESTION_LEGACY_META_ANTI_PATTERN_IDS
    }
)

ROUTE_CAPABLE_LEGACY_OVERRIDES.update(
    {
        entry_id: _anti_pattern_override(
            "MATH",
            anti_pattern_refs=[
                "anti_pattern_numeric_without_materialization",
                "anti_pattern_missing_validator_traversal",
                "anti_pattern_unchecked_unit_transfer",
            ],
        )
        for entry_id in MATH_LEGACY_META_ANTI_PATTERN_IDS
    }
)


@dataclass(frozen=True)
class RouteMetadataRepairResult:
    entry: dict[str, Any]
    bucket: str | None = None
    changed: bool = False
    original_id: str | None = None
    normalized_id: str | None = None


def _coerce_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _entry_id(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(
        entry.get("id")
        or entry.get("rule_id")
        or metadata.get("meaning_star_id")
        or ""
    ).strip()


def _normalized_entry_id(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    *,
    galaxy_name: str | None = None,
) -> tuple[str, str]:
    entry_id = _entry_id(entry, metadata)
    if not entry_id:
        return "", ""
    galaxy_key = str(
        galaxy_name
        or entry.get("galaxy")
        or metadata.get("galaxy")
        or entry.get("domain")
        or metadata.get("domain")
        or ""
    ).strip().lower()
    if (
        galaxy_key == "language"
        and entry_id.startswith("synset_")
        and str(metadata.get("ingest_source") or "").strip().lower() == "meaning_layer"
    ):
        replacement = route_contract.language_meaning_mirror_id(entry_id)
        entry["id"] = replacement
        metadata["resident_id_repaired_from"] = entry_id
        metadata["resident_id_repair_reason"] = "language_meaning_mirror_namespace"
        return entry_id, replacement
    replacement = str(IDENTITY_REPAIR_OVERRIDES.get((galaxy_key, entry_id)) or "").strip()
    if not replacement or replacement == entry_id:
        return entry_id, entry_id
    entry["id"] = replacement
    if str(entry.get("rule_id") or "").strip() == entry_id:
        entry["rule_id"] = replacement
    metadata["resident_id_repaired_from"] = entry_id
    metadata["resident_id_repair_reason"] = "explicit_duplicate_id_override"
    return entry_id, replacement


def _normalize_route_exempt(entry: dict[str, Any], metadata: dict[str, Any]) -> None:
    normalized = route_contract.apply_route_exempt_anchor_contract({**entry, "metadata": metadata})
    entry.clear()
    entry.update({key: value for key, value in normalized.items() if key != "metadata"})
    metadata.clear()
    metadata.update(dict(normalized.get("metadata") or {}))


def _normalize_route_capable(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    override: dict[str, Any],
) -> None:
    route_policy = dict(override.get("route_policy") or {})
    route_family = str(override.get("route_family") or "").strip()
    selection_role = str(override.get("selection_role") or "").strip().lower()
    layer_id = int(override.get("layer_id") or 0)
    answer_eligible = bool(override.get("answer_eligible", False))
    for container in (entry, metadata):
        if route_family:
            container["route_family"] = route_family
        container["selection_role"] = selection_role
        container["layer_id"] = layer_id
        container["answer_eligible"] = answer_eligible
        container["sovereign_route_exempt"] = False
        container["route_contract_schema_version"] = int(route_contract.ROUTE_CONTRACT_SCHEMA_VERSION)
        if route_policy:
            container["route_policy"] = dict(route_policy)
    for key in route_contract.ROUTE_ROLE_REF_KEYS:
        values = list(override.get(key) or [])
        if values:
            merged: list[str] = []
            for container in (entry, metadata):
                for raw_value in list(container.get(key) or []):
                    text = str(raw_value or "").strip()
                    if text and text not in merged:
                        merged.append(text)
            for raw_value in values:
                text = str(raw_value or "").strip()
                if text and text not in merged:
                    merged.append(text)
            entry[key] = list(merged)
            metadata[key] = list(merged)


def normalize_resident_route_metadata(
    entry: dict[str, Any],
    *,
    galaxy_name: str | None = None,
) -> RouteMetadataRepairResult:
    if not isinstance(entry, dict):
        return RouteMetadataRepairResult(entry={})
    raw = dict(entry)
    metadata = _coerce_metadata(raw)
    original_id, entry_id = _normalized_entry_id(raw, metadata, galaxy_name=galaxy_name)
    galaxy_key = str(
        galaxy_name
        or raw.get("galaxy")
        or metadata.get("galaxy")
        or raw.get("domain")
        or metadata.get("domain")
        or ""
    ).strip().lower()
    bucket: str | None = None
    if entry_id in ROUTE_EXEMPT_UTILITY_IDS or (galaxy_key, entry_id) in route_contract.route_exempt_anchor_keys():
        bucket = "route_exempt_utility"
        _normalize_route_exempt(raw, metadata)
    elif route_contract.is_foundational_route_exempt_substrate(raw, metadata, galaxy_key=galaxy_key):
        bucket = "route_exempt_utility"
        _normalize_route_exempt(raw, metadata)
    elif galaxy_key == "meaning_layer_stars" and (
        entry_id.startswith("synset_")
        or str(metadata.get("meaning_star_id") or "").strip().startswith("synset_")
    ):
        bucket = "route_exempt_utility"
        _normalize_route_exempt(raw, metadata)
    elif metadata.get("failure_patch_source") or metadata.get("failure_patch_family"):
        bucket = "route_exempt_utility"
        _normalize_route_exempt(raw, metadata)
    elif entry_id in route_contract.GRAMMAR_REASONING_EXECUTOR_CONTRACTS:
        bucket = "route_capable_legacy"
        promoted = route_contract.apply_route_capable_contract(
            {**raw, "metadata": metadata},
            dict(route_contract.GRAMMAR_REASONING_EXECUTOR_CONTRACTS[entry_id]),
        )
        raw = {key: value for key, value in promoted.items() if key != "metadata"}
        metadata = dict(promoted.get("metadata") or {})
    elif entry_id in ROUTE_CAPABLE_LEGACY_OVERRIDES:
        bucket = "route_capable_legacy"
        _normalize_route_capable(raw, metadata, dict(ROUTE_CAPABLE_LEGACY_OVERRIDES[entry_id]))
    if raw != entry:
        raw["metadata"] = metadata
        if bucket is None and original_id != entry_id:
            bucket = "duplicate_id_repair"
        return RouteMetadataRepairResult(
            entry=raw,
            bucket=bucket,
            changed=True,
            original_id=original_id or None,
            normalized_id=entry_id or None,
        )
    if bucket is None:
        return RouteMetadataRepairResult(
            entry=raw,
            bucket=None,
            changed=False,
            original_id=original_id or None,
            normalized_id=entry_id or None,
        )
    raw["metadata"] = metadata
    return RouteMetadataRepairResult(
        entry=raw,
        bucket=bucket,
        changed=(raw != entry),
        original_id=original_id or None,
        normalized_id=entry_id or None,
    )


def resident_route_registry_summary() -> dict[str, Any]:
    return {
        "route_exempt_utility": int(len(ROUTE_EXEMPT_UTILITY_IDS | route_contract.route_exempt_anchor_ids())),
        "route_capable_legacy": int(
            len(ROUTE_CAPABLE_LEGACY_OVERRIDES) + len(route_contract.GRAMMAR_REASONING_EXECUTOR_CONTRACTS)
        ),
        "duplicate_id_repairs": int(len(IDENTITY_REPAIR_OVERRIDES)),
    }


def repair_knowledgeverse_resident_route_metadata(
    knowledgeverse: Any,
    *,
    persist_to_disk: bool = True,
) -> dict[str, Any]:
    manager = getattr(knowledgeverse, "galaxy_manager")
    touched_galaxies: set[str] = set()
    bucket_counts = {
        "route_exempt_utility": 0,
        "route_capable_legacy": 0,
        "duplicate_id_repair": 0,
    }
    bucket_ids = {
        "route_exempt_utility": [],
        "route_capable_legacy": [],
        "duplicate_id_repair": [],
    }
    identity_repairs: list[dict[str, str]] = []
    for galaxy_name in knowledgeverse._discover_live_galaxy_names():
        galaxy = manager.get_galaxy(galaxy_name)
        entries = getattr(galaxy, "entries", None)
        if not isinstance(entries, list):
            entries = getattr(galaxy, "_extra_entries", None)
        if not isinstance(entries, list):
            continue
        changed_any = False
        for index, current in enumerate(list(entries)):
            if not isinstance(current, dict):
                continue
            repaired = normalize_resident_route_metadata(current, galaxy_name=str(galaxy_name))
            if not repaired.changed:
                continue
            entries[index] = dict(repaired.entry)
            changed_any = True
            if repaired.bucket is not None:
                bucket_counts[repaired.bucket] += 1
            entry_id = _entry_id(repaired.entry, _coerce_metadata(repaired.entry))
            if entry_id:
                bucket_key = repaired.bucket or "duplicate_id_repair"
                bucket_ids[bucket_key].append(entry_id)
            if (
                repaired.original_id
                and repaired.normalized_id
                and repaired.original_id != repaired.normalized_id
            ):
                identity_repairs.append(
                    {
                        "galaxy": str(galaxy_name),
                        "from": str(repaired.original_id),
                        "to": str(repaired.normalized_id),
                    }
                )
        if changed_any:
            touched_galaxies.add(str(galaxy_name))
    if persist_to_disk:
        for galaxy_name in sorted(touched_galaxies):
            manager._rewrite_galaxy_disk(galaxy_name, manager.get_galaxy(galaxy_name))
    return {
        "registry": resident_route_registry_summary(),
        "touched_galaxies": sorted(touched_galaxies),
        "route_exempt_utility": int(bucket_counts["route_exempt_utility"]),
        "route_capable_legacy": int(bucket_counts["route_capable_legacy"]),
        "duplicate_id_repair": int(bucket_counts["duplicate_id_repair"]),
        "modified_total": int(sum(bucket_counts.values())),
        "modified_ids": {bucket: sorted(values) for bucket, values in bucket_ids.items()},
        "identity_repairs": identity_repairs,
    }
