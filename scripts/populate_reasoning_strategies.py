from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.galaxy_population_utils import upsert_entries  # noqa: E402
from knowledge3d.knowledgeverse import route_contract  # noqa: E402


BOOTSTRAP_TAG = "phase_e38_four_way_reading_v2"
DEFAULT_HOUSE_DIR = Path("/K3D/Knowledge3D.local/house")
DEFAULT_GALAXY_DIR = Path("/K3D/Knowledge3D.local/galaxies")
DEFAULT_FAILURE_LOG_DIR = Path("/K3D/Knowledge3D.local/logs/e53_meaningful_mixed_rerun_20260331_233500")

REASONING_META_ROUTE_OVERRIDES: dict[str, dict[str, Any]] = {
    "meta_four_way_reading_strategy": {
        "route_family": "MATH",
        "validator_refs": ["math_normalization_validator", "math_unit_magnitude_validator", "math_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_numeric_without_materialization",
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_decompose_multi_step_word_problem": {
        "route_family": "MATH",
        "validator_refs": ["math_normalization_validator", "math_unit_magnitude_validator", "math_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_numeric_without_materialization",
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_apply_backward_trace_before_emit": {
        "route_family": "MATH",
        "validator_refs": ["math_normalization_validator", "math_unit_magnitude_validator", "math_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_numeric_without_materialization",
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_validate_units_before_answer": {
        "route_family": "MATH",
        "validator_refs": ["math_normalization_validator", "math_unit_magnitude_validator", "math_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_numeric_without_materialization",
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_template_slot_binding": {
        "route_family": "MATH",
        "validator_refs": ["math_normalization_validator", "math_unit_magnitude_validator", "math_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_numeric_without_materialization",
            "anti_pattern_unchecked_unit_transfer",
            "anti_pattern_missing_validator_traversal",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_route_question_subject_before_elimination": {
        "route_family": "QUESTION",
        "validator_refs": ["question_evidence_validator", "question_choice_alignment_validator", "question_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_option_emission_without_comparison",
            "anti_pattern_validator_as_answer_leakage",
            "anti_pattern_empty_route_dispatch",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_verify_option_before_emit": {
        "route_family": "QUESTION",
        "validator_refs": ["question_evidence_validator", "question_choice_alignment_validator", "question_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_option_emission_without_comparison",
            "anti_pattern_validator_as_answer_leakage",
            "anti_pattern_empty_route_dispatch",
        ],
        "route_policy": {"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
    },
    "meta_avoid_router_as_final_answer": {
        "route_family": "QUESTION",
        "selection_role": "anti_pattern",
        "anti_pattern_refs": [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_unsupported_option_leap",
            "anti_pattern_option_emission_without_comparison",
        ],
    },
    "meta_avoid_isolated_template_halt": {
        "route_family": "MATH",
        "selection_role": "anti_pattern",
        "anti_pattern_refs": [
            "anti_pattern_numeric_without_materialization",
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_unchecked_unit_transfer",
        ],
    },
    "halting_threshold_elimination": {
        "selection_role": "unknown",
        "answer_eligible": False,
        "layer_id": 0,
        "sovereign_route_exempt": True,
    },
    "halting_threshold_math": {
        "selection_role": "unknown",
        "answer_eligible": False,
        "layer_id": 0,
        "sovereign_route_exempt": True,
    },
    "halting_threshold_spatial": {
        "selection_role": "unknown",
        "answer_eligible": False,
        "layer_id": 0,
        "sovereign_route_exempt": True,
    },
    "halting_threshold_default": {
        "selection_role": "unknown",
        "answer_eligible": False,
        "layer_id": 0,
        "sovereign_route_exempt": True,
    },
}


def _merge_unique(*collections: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for values in collections:
        for raw_value in values:
            value = str(raw_value).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return ordered


def _annotate_entries(entries: list[dict[str, Any]], *, selection_role: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        entry_id = str(entry.get("id") or "").strip()
        override = dict(REASONING_META_ROUTE_OVERRIDES.get(entry_id) or {})
        if entry_id in route_contract.GRAMMAR_REASONING_EXECUTOR_CONTRACTS:
            override.update(dict(route_contract.GRAMMAR_REASONING_EXECUTOR_CONTRACTS[entry_id]))
        entry["selection_role"] = str(override.get("selection_role") or entry.get("selection_role") or selection_role).strip()
        for key, value in override.items():
            if key == "selection_role":
                continue
            entry[key] = value
        entry["answer_eligible"] = bool(entry.get("answer_eligible", entry["selection_role"] == "validator"))
        metadata = dict(entry.get("metadata") or {})
        metadata["selection_role"] = str(entry["selection_role"])
        metadata["answer_eligible"] = bool(entry["answer_eligible"])
        if "layer_id" in entry:
            metadata["layer_id"] = int(entry["layer_id"])
        if "sovereign_route_exempt" in entry:
            metadata["sovereign_route_exempt"] = bool(entry["sovereign_route_exempt"])
        metadata.setdefault("bootstrap", BOOTSTRAP_TAG)
        if "layer" in entry and "layer" not in metadata:
            metadata["layer"] = entry["layer"]
        for key in ("route_family", "route_policy", "validator_refs", "anti_pattern_refs", "executor_refs", "router_refs"):
            if key in entry:
                metadata[key] = entry[key]
        entry["metadata"] = metadata
        annotated.append(entry)
    return annotated


def _surface_forms(*forms: str) -> dict[str, Any]:
    ordered = [str(form).strip() for form in forms if str(form).strip()]
    primary = (ordered[0] if ordered else "reasoning strategy").lower().replace(" ", "_")
    return {
        "en": {"word_ref": f"word_{primary}", "char_refs": [], "surface_text": ordered or [primary]},
        "pt": {"word_ref": f"word_{primary}", "char_refs": [], "surface_text": ordered or [primary]},
    }


def _meaning_entry(
    entry_id: str,
    name: str,
    description: str,
    *,
    surface_forms: list[str],
    meaning_rpn: str,
    behavior_rpn: str,
    grammar_refs: list[str] | None = None,
    reality_refs: list[str] | None = None,
    meta_refs: list[str] | None = None,
    tags: list[str] | None = None,
    selection_role: str = "router",
    route_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_policy_payload = dict(route_policy or {})
    return {
        "id": entry_id,
        "star_id": entry_id,
        "name": name,
        "galaxy": "reasoning_strategies",
        "domain": "reasoning",
        "category": "reasoning_strategy",
        "layer": 2,
        "content": description,
        "summary": description,
        "description": description,
        "meaning_rpn": meaning_rpn,
        "behavior_rpn": behavior_rpn,
        "surface_forms": _surface_forms(*surface_forms),
        "grammar_refs": list(grammar_refs or []),
        "reality_refs": list(reality_refs or []),
        "meta_refs": list(meta_refs or []),
        "selection_role": str(selection_role),
        "metadata": {
            "bootstrap": BOOTSTRAP_TAG,
            "knowledge_category": "reasoning",
            "meaning_rpn": meaning_rpn,
            "behavior_rpn": behavior_rpn,
            "grammar_refs": list(grammar_refs or []),
            "reality_refs": list(reality_refs or []),
            "meta_refs": list(meta_refs or []),
            "selection_role": str(selection_role),
            "route_policy": route_policy_payload,
        },
        "route_policy": route_policy_payload,
        "tags": list(tags or []),
    }


def build_reasoning_meaning_entries() -> list[dict[str, Any]]:
    return [
        route_contract.apply_route_exempt_anchor_contract(row)
        for row in _annotate_entries([
        _meaning_entry(
            "forward_entity_extraction",
            "Forward Entity Extraction",
            "Read a problem left-to-right and bind quantities, units, entities, and action verbs into working memory.",
            surface_forms=["forward read", "entity extraction", "quantity binding"],
            meaning_rpn="TOKENS FOREACH IF NOUN THEN GALAXY_LOOKUP IF NUMBER THEN NUM_BIND IF VERB THEN ACTION_BIND",
            behavior_rpn="INPUT_SCAN LEFT_TO_RIGHT QUANTITY_BIND ENTITY_BIND ACTION_BIND",
            grammar_refs=[
                "grammar_forward_entity_extraction",
                "grammar_quantity_unit_binding",
                "grammar_quantity_role_initial",
                "grammar_quantity_role_delta",
                "grammar_quantity_role_multiplier",
                "grammar_template_slot_binding",
            ],
            reality_refs=["reality_word_problem_goal_state", "reality_dependency_dag"],
            meta_refs=["meta_four_way_reading_strategy"],
            tags=["reasoning", "math", "reading", "forward"],
        ),
        _meaning_entry(
            "backward_goal_tracing",
            "Backward Goal Tracing",
            "Start from the asked goal and trace backward through dependencies to determine which quantities and operations are required.",
            surface_forms=["backward read", "goal tracing", "dependency backtrace"],
            meaning_rpn="QUESTION_ENTITY GOAL_BIND DEPENDENCY_TRACE REVERSE",
            behavior_rpn="GOAL_IDENTIFY TRACE_BACKWARD REQUIRED_QUANTITIES_MARK",
            grammar_refs=["grammar_backward_goal_tracing", "grammar_dependency_dag_build"],
            reality_refs=["reality_word_problem_goal_state", "reality_dependency_dag"],
            meta_refs=["meta_apply_backward_trace_before_emit"],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
            tags=["reasoning", "math", "reading", "backward"],
        ),
        _meaning_entry(
            "operation_chain_construction",
            "Operation Chain Construction",
            "Compose a multi-step RPN program from the dependency graph instead of applying a single left-fold operation.",
            surface_forms=["operation chain", "multi-step program", "dependency execution"],
            meaning_rpn="DEPENDENCY_DAG TOPO_SORT OPERATION_BIND STORE RECALL CHAIN",
            behavior_rpn="SUBTASKS ORDER BUILD_RPN_CHAIN EXECUTE",
            grammar_refs=[
                "grammar_operation_chain_construction",
                "grammar_recursive_subtask_decomposition",
                "grammar_operation_chain_left_fold",
                "grammar_operation_chain_nested",
                "grammar_operation_chain_ratio",
                "grammar_operation_chain_difference_then_multiply",
                "grammar_template_slot_binding",
            ],
            reality_refs=["reality_operation_chain", "reality_dependency_dag"],
            meta_refs=["meta_decompose_multi_step_word_problem"],
            route_policy={"decompose_on_fail": True, "requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 4},
            tags=["reasoning", "math", "chain", "rpn"],
        ),
        _meaning_entry(
            "operation_chain_left_fold",
            "Operation Chain Left Fold",
            "Sequentially compose arithmetic steps from left to right when the quantities already appear in execution order.",
            surface_forms=["left fold chain", "sequential arithmetic chain", "left to right chain"],
            meaning_rpn="VALUES FOREACH APPLY_OP_LEFT_TO_RIGHT",
            behavior_rpn="SUBTASKS IN_ORDER EXECUTE_LEFT_FOLD",
            grammar_refs=["grammar_operation_chain_left_fold"],
            reality_refs=["reality_operation_chain"],
            meta_refs=["meta_decompose_multi_step_word_problem"],
            tags=["reasoning", "math", "chain", "sequential"],
        ),
        _meaning_entry(
            "operation_chain_nested",
            "Operation Chain Nested",
            "Compose a grouped program when one intermediate result must be solved before the outer operation can execute.",
            surface_forms=["nested chain", "grouped arithmetic", "inner before outer"],
            meaning_rpn="INNER_SUBTASK STORE OUTER_SUBTASK RECALL APPLY",
            behavior_rpn="BUILD_GROUPS SOLVE_INNER RECALL_OUTER",
            grammar_refs=["grammar_operation_chain_nested", "grammar_recursive_subtask_decomposition"],
            reality_refs=["reality_operation_chain", "reality_dependency_dag"],
            meta_refs=["meta_decompose_multi_step_word_problem"],
            tags=["reasoning", "math", "chain", "nested"],
        ),
        _meaning_entry(
            "operation_chain_ratio",
            "Operation Chain Ratio",
            "Handle ratio and scale language such as double, triple, or times as many by binding the multiplier role and applying it to the correct base quantity.",
            surface_forms=["ratio chain", "times as many", "double then compute"],
            meaning_rpn="BASE_QUANTITY MULTIPLIER_ROLE BIND SCALE_RESULT",
            behavior_rpn="ROLE_BIND MULTIPLY_BY_RATIO",
            grammar_refs=["grammar_operation_chain_ratio", "grammar_quantity_role_multiplier"],
            reality_refs=["reality_operation_chain"],
            meta_refs=["meta_four_way_reading_strategy"],
            tags=["reasoning", "math", "ratio", "scale"],
        ),
        _meaning_entry(
            "operation_chain_difference_then_multiply",
            "Operation Chain Difference Then Multiply",
            "Subtract one or more deltas from the initial quantity, then multiply the remainder by a rate or scale factor.",
            surface_forms=["difference then multiply", "subtract then scale", "remainder times rate"],
            meaning_rpn="INITIAL DELTAS SUBTRACT_CHAIN REMAINDER MULTIPLIER MUL",
            behavior_rpn="FIND_REMAINDER APPLY_RATE",
            grammar_refs=[
                "grammar_operation_chain_difference_then_multiply",
                "grammar_quantity_role_initial",
                "grammar_quantity_role_delta",
                "grammar_quantity_role_multiplier",
            ],
            reality_refs=["reality_operation_chain", "reality_dependency_dag"],
            meta_refs=["meta_decompose_multi_step_word_problem"],
            tags=["reasoning", "math", "difference", "multiply"],
        ),
        _meaning_entry(
            route_contract.REASONING_ANCHOR_ID_REPAIRS["quantity_role_initial"],
            "Quantity Role Initial",
            "The starting amount before any additions, removals, or scaling operations occur.",
            surface_forms=["initial amount", "started with", "had at first"],
            meaning_rpn="QUANTITY CONTEXT initial ROLE_BIND",
            behavior_rpn="ROLE_ASSIGN initial",
            grammar_refs=["grammar_quantity_role_initial"],
            reality_refs=["reality_word_problem_goal_state"],
            meta_refs=["meta_four_way_reading_strategy"],
            tags=["reasoning", "math", "quantity_role", "initial"],
        ),
        _meaning_entry(
            route_contract.REASONING_ANCHOR_ID_REPAIRS["quantity_role_delta"],
            "Quantity Role Delta",
            "A quantity that changes another amount through loss, gain, purchase, transfer, or removal.",
            surface_forms=["delta quantity", "lost amount", "change amount"],
            meaning_rpn="QUANTITY CONTEXT delta ROLE_BIND",
            behavior_rpn="ROLE_ASSIGN delta",
            grammar_refs=["grammar_quantity_role_delta"],
            reality_refs=["reality_dependency_dag"],
            meta_refs=["meta_four_way_reading_strategy"],
            tags=["reasoning", "math", "quantity_role", "delta"],
        ),
        _meaning_entry(
            "quantity_role_multiplier",
            "Quantity Role Multiplier",
            "A quantity that scales another value through multiplication, doubling, tripling, or rate application.",
            surface_forms=["multiplier quantity", "times as many", "rate quantity"],
            meaning_rpn="QUANTITY CONTEXT multiplier ROLE_BIND",
            behavior_rpn="ROLE_ASSIGN multiplier",
            grammar_refs=["grammar_quantity_role_multiplier"],
            reality_refs=["reality_dependency_dag"],
            meta_refs=["meta_four_way_reading_strategy"],
            tags=["reasoning", "math", "quantity_role", "multiplier"],
        ),
        _meaning_entry(
            "template_slot_binding",
            "Template Slot Binding",
            "Bind extracted quantities into named execution slots so a matching math template can emit the full RPN chain without Python-side slot logic.",
            surface_forms=["slot binding", "template slots", "bind quantities to template"],
            meaning_rpn="ROLE_MAP TEMPLATE_SLOTS BIND STORE",
            behavior_rpn="SLOT_FILL TEMPLATE_READY",
            grammar_refs=["grammar_template_slot_binding"],
            reality_refs=["reality_dependency_dag", "reality_operation_chain"],
            meta_refs=["meta_template_slot_binding"],
            tags=["reasoning", "math", "template", "binding"],
        ),
        _meaning_entry(
            "result_normalization_validation",
            "Result Normalization Validation",
            "Validate intermediate and final results for unit consistency, magnitude plausibility, and complete quantity consumption.",
            surface_forms=["result validation", "normalization", "unit check"],
            meaning_rpn="INTERMEDIATE_RESULTS UNIT_CHECK MAGNITUDE_CHECK ORPHAN_CHECK VALIDATE",
            behavior_rpn="UNITS_COMPARE MAGNITUDE_COMPARE ANSWER_SANITY_GATE",
            grammar_refs=["grammar_result_normalization", "grammar_validate_units_and_magnitude"],
            reality_refs=["reality_unit_consistency", "reality_operation_chain"],
            meta_refs=["meta_validate_units_before_answer"],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 2},
            tags=["reasoning", "math", "validation", "normalization"],
        ),
        _meaning_entry(
            "word_problem_multi_step_reasoning",
            "Word Problem Multi-Step Reasoning",
            "A four-pass reasoning procedure for multi-step question solving: forward extraction, backward tracing, chain construction, and validation.",
            surface_forms=["four-way reading", "multi-step word problem", "recursive decomposition"],
            meaning_rpn="forward_entity_extraction CALL backward_goal_tracing CALL operation_chain_construction CALL result_normalization_validation CALL",
            behavior_rpn="FOUR_PASS_REASONING EXECUTE UNTIL_CONVERGED",
            grammar_refs=[
                "grammar_forward_entity_extraction",
                "grammar_backward_goal_tracing",
                "grammar_operation_chain_construction",
                "grammar_operation_chain_left_fold",
                "grammar_operation_chain_nested",
                "grammar_operation_chain_ratio",
                "grammar_operation_chain_difference_then_multiply",
                "grammar_quantity_role_initial",
                "grammar_quantity_role_delta",
                "grammar_quantity_role_multiplier",
                "grammar_template_slot_binding",
                "grammar_result_normalization",
            ],
            reality_refs=["reality_dependency_dag", "reality_operation_chain", "reality_unit_consistency"],
            meta_refs=[
                "meta_four_way_reading_strategy",
                "meta_decompose_multi_step_word_problem",
                "meta_template_slot_binding",
            ],
            route_policy={"decompose_on_fail": True, "requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 4},
            tags=["reasoning", "math", "word_problem", "meta"],
        ),
        _meaning_entry(
            "question_subject_domain_routing",
            "Question Subject Domain Routing",
            "Use the surface subject and domain meaning as the primary router, then traverse into elimination, factual lookup, comparison, and verification executors before emitting an answer.",
            surface_forms=["subject routing", "domain routing", "question anchor routing"],
            meaning_rpn="QUESTION SUBJECT_HINT DOMAIN_ANCHOR ROUTE_EXECUTORS VERIFY_BEFORE_EMIT",
            behavior_rpn="SUBJECT_ROUTE EXECUTOR_ROUTE VALIDATOR_ROUTE",
            grammar_refs=[
                "grammar_subject_domain_alignment",
                "grammar_option_elimination",
                "grammar_compare_options_by_clues",
                "grammar_factual_lookup",
                "grammar_option_verification",
            ],
            reality_refs=["reality_question_answer_alignment"],
            meta_refs=[
                "meta_route_question_subject_before_elimination",
                "meta_verify_option_before_emit",
                "meta_avoid_router_as_final_answer",
                "meta_avoid_isolated_template_halt",
            ],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 4},
            tags=["reasoning", "question", "routing"],
        ),
        _meaning_entry(
            "question_option_elimination",
            "Question Option Elimination",
            "Discard answer options that contradict the routed subject clues, definitions, or world facts before the final comparison step.",
            surface_forms=["option elimination", "discard wrong options", "choice pruning"],
            meaning_rpn="OPTIONS CLUES CONTRADICTIONS ELIMINATE_REJECTED",
            behavior_rpn="OPTION_PRUNE CONTRADICTION_CHECK",
            grammar_refs=[
                "grammar_option_elimination",
                "grammar_compare_options_by_clues",
                "grammar_option_verification",
            ],
            reality_refs=["reality_question_answer_alignment"],
            meta_refs=["meta_verify_option_before_emit"],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
            tags=["reasoning", "question", "elimination"],
        ),
        _meaning_entry(
            "question_factual_lookup_validation",
            "Question Factual Lookup Validation",
            "Bind a subject-domain question to the matching fact or rule anchor, then verify that the candidate option is supported before emission.",
            surface_forms=["factual lookup", "question verification", "fact support"],
            meaning_rpn="QUESTION FACT_ANCHOR LOOKUP OPTION_SUPPORT VERIFY",
            behavior_rpn="FACT_RETRIEVE OPTION_VALIDATE",
            grammar_refs=[
                "grammar_factual_lookup",
                "grammar_option_verification",
                "grammar_subject_domain_alignment",
            ],
            reality_refs=["reality_question_answer_alignment"],
            meta_refs=["meta_verify_option_before_emit"],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 3},
            tags=["reasoning", "question", "factual"],
        ),
        _meaning_entry(
            "anti_pattern_router_final_answer",
            "Anti Pattern Router Final Answer",
            "A router or anchor is not itself the final answer unless it carries an explicit answer role. Route first, execute and validate second.",
            surface_forms=["router is not answer", "anchor is not final answer", "wrong router winner"],
            meaning_rpn="ROUTER_FOUND HOLD_EMIT EXECUTOR_ROUTE VALIDATOR_ROUTE",
            behavior_rpn="PREVENT_ROUTER_EMIT",
            grammar_refs=["grammar_subject_domain_alignment", "grammar_option_verification"],
            reality_refs=["reality_question_answer_alignment"],
            meta_refs=["meta_avoid_router_as_final_answer"],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 2},
            tags=["reasoning", "anti_pattern", "routing"],
            selection_role="anti_pattern",
        ),
        _meaning_entry(
            "anti_pattern_isolated_template_halt",
            "Anti Pattern Isolated Template Halt",
            "Do not halt on an isolated formula or template candidate when it has no route support from the active subject, option, or execution chain.",
            surface_forms=["isolated template halt", "template without support", "wrong formula attractor"],
            meaning_rpn="ISOLATED_TEMPLATE NO_ROUTE_SUPPORT HOLD_HALTING CONTINUE_TRAVERSAL",
            behavior_rpn="BLOCK_ISOLATED_TEMPLATE_HALT",
            grammar_refs=["grammar_compare_options_by_clues", "grammar_option_verification"],
            reality_refs=["reality_question_answer_alignment"],
            meta_refs=["meta_avoid_isolated_template_halt"],
            route_policy={"requires_executor": True, "requires_validator": True, "answer_gate": True, "branch_topk": 2},
            tags=["reasoning", "anti_pattern", "question"],
            selection_role="anti_pattern",
        ),
    ], selection_role="router")
    ]


def build_reasoning_reality_entries() -> list[dict[str, Any]]:
    return [
        route_contract.apply_route_exempt_anchor_contract(row)
        for row in _annotate_entries([
        {
            "id": "reality_word_problem_goal_state",
            "name": "Word Problem Goal State",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "goal_state",
            "content": "The explicit question target that determines what quantity or proof obligation must be produced.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "goal", "word_problem"],
        },
        {
            "id": "reality_dependency_dag",
            "name": "Reasoning Dependency DAG",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "dependency_structure",
            "content": "A directed acyclic graph of intermediate computations needed to reach a final answer.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "dag", "dependencies"],
        },
        {
            "id": "reality_operation_chain",
            "name": "Operation Chain",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "execution_structure",
            "content": "A multi-step ordered sequence of operations composed from the dependency graph and executed through RPN registers.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "rpn", "chain"],
        },
        {
            "id": "reality_unit_consistency",
            "name": "Unit Consistency",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "validation",
            "content": "A validation state asserting that units, scales, and answer type remain consistent across the reasoning chain.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "units", "validation"],
        },
        {
            "id": "reality_question_answer_alignment",
            "name": "Question Answer Alignment",
            "galaxy": "Reality",
            "domain": "reasoning",
            "category": "answer_validation",
            "content": "A valid answer candidate must align with the routed subject, surviving clues, and the requested output form.",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 2},
            "tags": ["reasoning", "question", "validation"],
        },
    ], selection_role="router")
    ]


def build_reasoning_grammar_rules() -> list[dict[str, Any]]:
    return _annotate_entries([
        {
            "id": "grammar_forward_entity_extraction",
            "name": "Forward Entity Extraction",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Read the prompt left-to-right and bind nouns, numbers, units, and verbs into working slots.",
            "rpn_program": "TOKENS FOREACH IF_NOUN GALAXY_LOOKUP IF_NUMBER NUM_BIND IF_UNIT UNIT_BIND IF_VERB ACTION_BIND",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "reading", "forward"],
        },
        {
            "id": "grammar_quantity_unit_binding",
            "name": "Quantity Unit Binding",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Bind quantities to their local units and entities before arithmetic composition.",
            "rpn_program": "QUANTITY UNIT ENTITY BIND_TRIPLE STORE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "quantity", "units"],
        },
        {
            "id": "grammar_backward_goal_tracing",
            "name": "Backward Goal Tracing",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Trace backward from the requested answer type through the required dependencies.",
            "rpn_program": "QUESTION GOAL_BIND DEPENDENCY_TRACE REVERSE_STORE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "reading", "backward"],
        },
        {
            "id": "grammar_dependency_dag_build",
            "name": "Dependency DAG Build",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Construct a DAG of intermediate operations required by the final goal.",
            "rpn_program": "BOUND_QUANTITIES GOAL_STATE DEPENDENCY_DAG_BUILD",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "dag", "dependencies"],
        },
        {
            "id": "grammar_operation_chain_construction",
            "name": "Operation Chain Construction",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Topologically sort the dependency DAG and emit a multi-step RPN chain using STORE and RECALL.",
            "rpn_program": "DEPENDENCY_DAG TOPO_SORT FOREACH OP_BIND STORE RECALL CHAIN_BUILD",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "rpn", "chain"],
        },
        {
            "id": "grammar_operation_chain_left_fold",
            "name": "Operation Chain Left Fold",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Apply a sequential arithmetic chain in the observed order when the problem already exposes the execution sequence.",
            "rpn_program": "VALUES FOREACH OP_RECALL APPLY_LEFT_FOLD",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "operation_pattern": True,
                "operation": "seq",
                "operation_chain": ["add", "sub"],
                "required_roles": ["initial", "delta"],
                "role_slots": ["initial", "delta_1", "delta_2"],
                "query_anchor": "left to right arithmetic chain sequential quantities",
                "aliases": ["left fold", "sequential chain", "ordered arithmetic"],
                "structural_cues": ["then", "after that", "next", "in order"],
            },
            "tags": ["reasoning", "rpn", "chain", "left_fold"],
        },
        {
            "id": "grammar_operation_chain_nested",
            "name": "Operation Chain Nested",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Solve an inner grouped sub-problem, store its result, then apply the outer operation.",
            "rpn_program": "INNER_GROUP EVAL STORE OUTER_GROUP RECALL APPLY",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "operation_pattern": True,
                "operation": "group",
                "operation_chain": ["sub", "mul"],
                "required_roles": ["initial", "delta", "multiplier"],
                "role_slots": ["initial", "delta_1", "multiplier"],
                "query_anchor": "grouped arithmetic solve inside first then outside",
                "aliases": ["nested chain", "grouped chain", "inner then outer"],
                "structural_cues": ["each", "per group", "remaining", "after removing"],
            },
            "tags": ["reasoning", "rpn", "chain", "nested"],
        },
        {
            "id": "grammar_operation_chain_ratio",
            "name": "Operation Chain Ratio",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Bind a multiplier role from ratio language such as double or times as many and apply it to the proper base quantity.",
            "rpn_program": "BASE_VALUE MULTIPLIER_ROLE RECALL MUL",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "operation_pattern": True,
                "operation": "mul",
                "operation_chain": ["mul"],
                "required_roles": ["initial", "multiplier"],
                "role_slots": ["initial", "multiplier"],
                "query_anchor": "times as many double triple multiply by ratio",
                "aliases": ["ratio chain", "times as many", "scale by multiplier"],
                "structural_cues": ["times", "double", "triple", "each"],
            },
            "tags": ["reasoning", "rpn", "ratio", "multiply"],
        },
        {
            "id": "grammar_operation_chain_difference_then_multiply",
            "name": "Operation Chain Difference Then Multiply",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Subtract one or more deltas from the initial quantity and then multiply the remainder by the role-bound scale value.",
            "rpn_program": "INITIAL RECALL DELTA_1 RECALL - DELTA_2 RECALL - MULTIPLIER RECALL *",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "operation_pattern": True,
                "operation": "mul",
                "operation_chain": ["sub", "sub", "mul"],
                "required_roles": ["initial", "delta", "multiplier"],
                "role_slots": ["initial", "delta_1", "delta_2", "multiplier"],
                "query_anchor": "subtract losses then multiply remainder by rate",
                "aliases": ["difference then multiply", "remainder times rate", "subtract then scale"],
                "structural_cues": ["left", "remain", "remaining", "times", "double", "twice"],
                "math_refs": ["operation_pattern_remainder_scale"],
            },
            "tags": ["reasoning", "rpn", "difference", "multiply"],
        },
        {
            "id": "grammar_recursive_subtask_decomposition",
            "name": "Recursive Subtask Decomposition",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Break a multi-step problem into sub-tasks, solve them individually, and compose their results.",
            "rpn_program": "QUERY PARTIAL_KNOWLEDGE DECOMPOSE SUBTASKS FOREACH SOLVE_SUBTASK COMPOSE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "recursive", "decomposition"],
        },
        {
            "id": "grammar_quantity_role_initial",
            "name": "Quantity Role Initial",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Bind quantities that denote the starting amount before changes occur.",
            "rpn_program": "IF TOKENS MATCH INITIAL_CUES THEN ROLE_BIND initial",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "quantity_role": "initial",
                "query_anchor": "initial amount started with had at first begins with",
                "aliases": ["initial", "starting amount", "had at first"],
                "structural_cues": ["had", "started with", "began with", "originally"],
            },
            "tags": ["reasoning", "quantity_role", "initial"],
        },
        {
            "id": "grammar_quantity_role_delta",
            "name": "Quantity Role Delta",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Bind quantities that change another amount through removal, gain, purchase, or transfer.",
            "rpn_program": "IF TOKENS MATCH DELTA_CUES THEN ROLE_BIND delta",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "quantity_role": "delta",
                "query_anchor": "lost gave spent bought removed added more",
                "aliases": ["delta", "change amount", "loss amount"],
                "structural_cues": ["lost", "gave", "spent", "bought", "removed", "more", "less"],
            },
            "tags": ["reasoning", "quantity_role", "delta"],
        },
        {
            "id": "grammar_quantity_role_multiplier",
            "name": "Quantity Role Multiplier",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "reading_rule",
            "content": "Bind quantities that scale another amount through multiplication, doubling, tripling, or rate application.",
            "rpn_program": "IF TOKENS MATCH MULTIPLIER_CUES THEN ROLE_BIND multiplier",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "quantity_role": "multiplier",
                "query_anchor": "times as many double triple rate per each",
                "aliases": ["multiplier", "scale value", "rate quantity"],
                "structural_cues": ["times", "double", "triple", "per", "each", "twice"],
            },
            "tags": ["reasoning", "quantity_role", "multiplier"],
        },
        {
            "id": "grammar_template_slot_binding",
            "name": "Template Slot Binding",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "composition_rule",
            "content": "Map extracted quantity roles into named template slots so an execution chain can be emitted without Python-side slot binding.",
            "rpn_program": "ROLE_MAP TEMPLATE_SLOT_NAMES BIND_SLOT_VALUES STORE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "template", "binding"],
        },
        {
            "id": "grammar_result_normalization",
            "name": "Result Normalization",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "validation_rule",
            "content": "Normalize the answer form, units, and magnitude before emitting a final result.",
            "rpn_program": "RESULT NORMALIZE_FORMAT UNIT_NORMALIZE MAGNITUDE_NORMALIZE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "normalization", "validation"],
        },
        {
            "id": "grammar_validate_units_and_magnitude",
            "name": "Validate Units And Magnitude",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "validation_rule",
            "content": "Check that units are consistent and the resulting magnitude is plausible for the requested answer.",
            "rpn_program": "UNITS_COMPARE MAGNITUDE_COMPARE PLAUSIBILITY_GATE",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 3},
            "tags": ["reasoning", "units", "magnitude", "validation"],
        },
        {
            "id": "grammar_subject_domain_alignment",
            "name": "Subject Domain Alignment",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "routing_rule",
            "content": "Align a question with subject and domain anchors, then traverse into the supported execution family instead of emitting the anchor itself.",
            "rpn_program": "QUESTION SUBJECT_HINT DOMAIN_ANCHOR MATCH ROUTE_EXECUTORS",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "query_anchor": "subject domain anchor question routing executor validator",
                "aliases": ["subject routing", "domain routing", "anchor routing"],
            },
            "tags": ["reasoning", "question", "routing"],
        },
        {
            "id": "grammar_option_elimination",
            "name": "Option Elimination",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "comparison_rule",
            "content": "Score options against routed clues, reject contradictions, and keep only the consistent survivors for final verification.",
            "rpn_program": "OPTIONS CLUES FOREACH CONTRADICTION_CHECK REJECT_IF_FAIL SURVIVORS_STORE",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "query_anchor": "multiple choice eliminate inconsistent options",
                "aliases": ["option pruning", "eliminate wrong choice", "contradiction pruning"],
            },
            "tags": ["reasoning", "question", "elimination"],
        },
        {
            "id": "grammar_compare_options_by_clues",
            "name": "Compare Options By Clues",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "comparison_rule",
            "content": "Compare surviving options against the routed clue set and prefer the option with the strongest grounded support chain.",
            "rpn_program": "SURVIVORS CLUES SUPPORT_SCORE RANK_SELECT",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "query_anchor": "compare answer options by clues and support",
                "aliases": ["clue comparison", "support comparison", "option ranking"],
            },
            "tags": ["reasoning", "question", "comparison"],
        },
        {
            "id": "grammar_factual_lookup",
            "name": "Factual Lookup",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "lookup_rule",
            "content": "Resolve a routed fact question by retrieving the anchored reality or grammar support chain before answer emission.",
            "rpn_program": "QUESTION FACT_ANCHOR LOOKUP SUPPORT_CHAIN STORE",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "query_anchor": "fact lookup anchored verification",
                "aliases": ["fact retrieval", "anchored lookup", "grounded lookup"],
            },
            "tags": ["reasoning", "question", "lookup"],
        },
        {
            "id": "grammar_option_verification",
            "name": "Option Verification",
            "galaxy": "Grammar",
            "domain": "reasoning",
            "category": "validation_rule",
            "content": "Require a surviving option to carry route support from executors or validators before it can emit the final answer.",
            "rpn_program": "OPTION SUPPORT_CHAIN VERIFY ELSE HOLD_EMIT",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 3,
                "query_anchor": "verify option before emit",
                "aliases": ["option validation", "support verification", "final answer gate"],
            },
            "tags": ["reasoning", "question", "validation"],
        },
    ], selection_role="executor")


def build_reasoning_meta_rules() -> list[dict[str, Any]]:
    return _annotate_entries([
        {
            "id": "meta_four_way_reading_strategy",
            "name": "Four Way Reading Strategy",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "For multi-step word problems, apply forward extraction, backward tracing, chain construction, then normalization.",
            "rpn_program": "IF WORD_PROBLEM THEN grammar_forward_entity_extraction CALL grammar_backward_goal_tracing CALL grammar_operation_chain_construction CALL grammar_result_normalization CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "word_problem"],
        },
        {
            "id": "meta_decompose_multi_step_word_problem",
            "name": "Decompose Multi-Step Word Problem",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "When direct resolution fails, decompose the problem into sub-tasks and solve recursively.",
            "rpn_program": "IF DIRECT_RESOLUTION_FAIL THEN grammar_recursive_subtask_decomposition CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "decomposition"],
        },
        {
            "id": "meta_apply_backward_trace_before_emit",
            "name": "Apply Backward Trace Before Emit",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Prevent answer emission until backward goal tracing has identified the required dependencies.",
            "rpn_program": "IF GOAL_DEPENDENCIES_MISSING THEN grammar_backward_goal_tracing CALL HOLD_EMIT",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "goal_trace"],
        },
        {
            "id": "meta_validate_units_before_answer",
            "name": "Validate Units Before Answer",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Require unit and magnitude validation before the final answer leaves the reasoning loop.",
            "rpn_program": "IF ANSWER_READY THEN grammar_validate_units_and_magnitude CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "validation"],
        },
        {
            "id": "meta_template_slot_binding",
            "name": "Template Slot Binding Meta Rule",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Require quantity roles to be bound into template slots before the execution chain is emitted.",
            "rpn_program": "IF TEMPLATE_SELECTED THEN grammar_template_slot_binding CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "meta", "template"],
        },
        {
            "id": "halting_threshold_elimination",
            "name": "Halting Threshold Elimination",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Choice-based questions halt on relative separation rather than absolute score.",
            "meaning_rpn": "minimum_threshold 0.0 gap_threshold 0.04 agreement_threshold 0.0",
            "rpn_program": "HALTING_PROFILE ELIMINATION MIN 0.0 GAP 0.04 AGREE 0.0",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 4,
                "minimum_threshold": 0.0,
                "gap_threshold": 0.04,
                "agreement_threshold": 0.0,
            },
            "tags": ["reasoning", "meta", "halting"],
        },
        {
            "id": "halting_threshold_math",
            "name": "Halting Threshold Math",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Numeric reasoning halts when consensus is both structurally valid and sufficiently separated.",
            "meaning_rpn": "minimum_threshold 0.3 gap_threshold 0.04 agreement_threshold 1.0",
            "rpn_program": "HALTING_PROFILE MATH MIN 0.3 GAP 0.04 AGREE 1.0",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 4,
                "minimum_threshold": 0.3,
                "gap_threshold": 0.04,
                "agreement_threshold": 1.0,
            },
            "tags": ["reasoning", "meta", "halting"],
        },
        {
            "id": "halting_threshold_spatial",
            "name": "Halting Threshold Spatial",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Spatial tasks need stronger confidence and agreement before a visual action is emitted.",
            "meaning_rpn": "minimum_threshold 0.3 gap_threshold 0.1 agreement_threshold 3.0",
            "rpn_program": "HALTING_PROFILE SPATIAL MIN 0.3 GAP 0.1 AGREE 3.0",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 4,
                "minimum_threshold": 0.3,
                "gap_threshold": 0.1,
                "agreement_threshold": 3.0,
            },
            "tags": ["reasoning", "meta", "halting"],
        },
        {
            "id": "halting_threshold_default",
            "name": "Halting Threshold Default",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Default halting policy for general queries when no stronger domain policy applies.",
            "meaning_rpn": "minimum_threshold 0.3 gap_threshold 0.1 agreement_threshold 3.0",
            "rpn_program": "HALTING_PROFILE DEFAULT MIN 0.3 GAP 0.1 AGREE 3.0",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 4,
                "minimum_threshold": 0.3,
                "gap_threshold": 0.1,
                "agreement_threshold": 3.0,
            },
            "tags": ["reasoning", "meta", "halting"],
        },
        {
            "id": "meta_route_question_subject_before_elimination",
            "name": "Route Question Subject Before Elimination",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "A question must route through subject and domain anchors before elimination or factual verification begins.",
            "rpn_program": "IF QUESTION THEN grammar_subject_domain_alignment CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "question", "routing"],
        },
        {
            "id": "meta_verify_option_before_emit",
            "name": "Verify Option Before Emit",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "meta_rule",
            "content": "Do not emit an option answer until it has route support from comparison, factual lookup, or validator chains.",
            "rpn_program": "IF OPTION_READY THEN grammar_option_verification CALL",
            "metadata": {"bootstrap": BOOTSTRAP_TAG, "layer": 4},
            "tags": ["reasoning", "question", "validation"],
        },
        {
            "id": "meta_avoid_router_as_final_answer",
            "name": "Avoid Router As Final Answer",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "anti_pattern",
            "content": "Routers and subject anchors seed traversal but cannot terminate the answer unless they also carry an explicit answer role.",
            "meaning_rpn": "router_only HOLD_EMIT executor_or_answer REQUIRED",
            "rpn_program": "IF ROUTER_ONLY_WINNER THEN HOLD_EMIT CONTINUE_TRAVERSAL",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 4,
                "selection_role": "anti_pattern",
                "semantic_gravity_polarity": -1,
            },
            "tags": ["reasoning", "anti_pattern", "router"],
            "selection_role": "anti_pattern",
        },
        {
            "id": "meta_avoid_isolated_template_halt",
            "name": "Avoid Isolated Template Halt",
            "galaxy": "Tool",
            "domain": "reasoning",
            "category": "anti_pattern",
            "content": "A formula or template candidate without route support from the active subject or execution chain cannot trigger halting.",
            "meaning_rpn": "isolated_template HOLD_HALTING route_support REQUIRED",
            "rpn_program": "IF TEMPLATE_WITHOUT_ROUTE_SUPPORT THEN CONTINUE_TRAVERSAL",
            "metadata": {
                "bootstrap": BOOTSTRAP_TAG,
                "layer": 4,
                "selection_role": "anti_pattern",
                "semantic_gravity_polarity": -1,
            },
            "tags": ["reasoning", "anti_pattern", "template"],
            "selection_role": "anti_pattern",
        },
    ], selection_role="router")


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _patch_refs_for_family(family: str) -> tuple[list[str], list[str]]:
    if family == "QUESTION":
        return (
            [
                "grammar_subject_domain_alignment",
                "grammar_option_elimination",
                "grammar_compare_options_by_clues",
                "grammar_factual_lookup",
                "grammar_option_verification",
            ],
            [
                "meta_route_question_subject_before_elimination",
                "meta_verify_option_before_emit",
                "meta_avoid_router_as_final_answer",
                "meta_avoid_isolated_template_halt",
            ],
        )
    return (
        [
            "grammar_forward_entity_extraction",
            "grammar_quantity_unit_binding",
            "grammar_quantity_role_initial",
            "grammar_quantity_role_delta",
            "grammar_quantity_role_multiplier",
            "grammar_backward_goal_tracing",
            "grammar_operation_chain_construction",
            "grammar_operation_chain_difference_then_multiply",
            "grammar_template_slot_binding",
            "grammar_result_normalization",
        ],
        [
            "meta_four_way_reading_strategy",
            "meta_decompose_multi_step_word_problem",
            "meta_apply_backward_trace_before_emit",
            "meta_validate_units_before_answer",
            "meta_avoid_router_as_final_answer",
        ],
    )


def _apply_router_patch(row: dict[str, Any], *, family: str, source_log_dir: str | Path) -> dict[str, Any]:
    patched = route_contract.apply_route_exempt_anchor_contract(dict(row))
    grammar_refs, meta_refs = _patch_refs_for_family(family)
    patched["grammar_refs"] = _merge_unique(patched.get("grammar_refs", []), grammar_refs)
    patched["meta_refs"] = _merge_unique(patched.get("meta_refs", []), meta_refs)
    metadata = dict(patched.get("metadata") or {})
    metadata["grammar_refs"] = list(patched["grammar_refs"])
    metadata["meta_refs"] = list(patched["meta_refs"])
    metadata["failure_patch_family"] = str(family)
    metadata["failure_patch_source"] = str(source_log_dir)
    patched["metadata"] = metadata
    return patched


def _mine_failure_patch_targets(log_dir: str | Path) -> dict[tuple[str, str], str]:
    root = Path(log_dir)
    family_by_file = {
        "gsm8k": "MATH",
        "math": "MATH",
        "amc_aime": "MATH",
        "omni_math": "MATH",
        "imo": "MATH",
        "mmlu": "QUESTION",
        "lhe": "QUESTION",
    }
    targets: dict[tuple[str, str], str] = {}
    for stem, family in family_by_file.items():
        path = root / f"{stem}.jsonl"
        if not path.exists():
            continue
        for row in _read_jsonl_rows(path):
            if row.get("correct") is True:
                continue
            task_result = row.get("task_result") if isinstance(row.get("task_result"), dict) else {}
            match = task_result.get("match") if isinstance(task_result.get("match"), dict) else {}
            galaxy = str(match.get("galaxy", "")).strip()
            entry_id = str(match.get("id", "")).strip()
            if not galaxy or not entry_id:
                continue
            if galaxy not in {"Language", "meaning_layer_stars"}:
                continue
            targets.setdefault((galaxy, entry_id), family)
    return targets


def patch_failure_router_entries(
    *,
    galaxy_dir: str | Path = DEFAULT_GALAXY_DIR,
    log_dir: str | Path = DEFAULT_FAILURE_LOG_DIR,
) -> dict[str, dict[str, int]]:
    root = Path(galaxy_dir)
    targets = _mine_failure_patch_targets(log_dir)
    stats: dict[str, dict[str, int]] = {}
    if not targets:
        return stats
    grouped: dict[str, dict[str, str]] = {}
    for (galaxy_name, entry_id), family in targets.items():
        grouped.setdefault(galaxy_name, {})[entry_id] = family
    for galaxy_name, target_map in grouped.items():
        path = root / f"{galaxy_name}.jsonl"
        rows = _read_jsonl_rows(path)
        if not rows:
            continue
        before = len(rows)
        replaced = 0
        patched_rows: list[dict[str, Any]] = []
        for row in rows:
            row_id = str(row.get("id", "")).strip()
            family = target_map.get(row_id)
            if family:
                patched_rows.append(_apply_router_patch(row, family=family, source_log_dir=log_dir))
                replaced += 1
            else:
                patched_rows.append(row)
        _write_jsonl_rows(path, patched_rows)
        stats[f"{galaxy_name}.jsonl"] = {
            "before": before,
            "after": len(patched_rows),
            "appended": 0,
            "replaced": replaced,
            "removed": 0,
        }
    return stats


def populate_reasoning_strategies(
    house_dir: str | Path = DEFAULT_HOUSE_DIR,
    *,
    galaxy_dir: str | Path = DEFAULT_GALAXY_DIR,
    failure_log_dir: str | Path = DEFAULT_FAILURE_LOG_DIR,
) -> dict[str, dict[str, int]]:
    root = Path(house_dir)
    root.mkdir(parents=True, exist_ok=True)
    stats = {
        "reasoning_strategies.jsonl": upsert_entries(
            root / "reasoning_strategies.jsonl",
            build_reasoning_meaning_entries(),
            remove_ids=set(route_contract.REASONING_ANCHOR_ID_REPAIRS.keys()),
        ),
        "Reality.jsonl": upsert_entries(root / "Reality.jsonl", build_reasoning_reality_entries()),
        "Grammar.jsonl": upsert_entries(root / "Grammar.jsonl", build_reasoning_grammar_rules()),
        "Tool.jsonl": upsert_entries(root / "Tool.jsonl", build_reasoning_meta_rules()),
    }
    stats.update(
        patch_failure_router_entries(
            galaxy_dir=galaxy_dir,
            log_dir=failure_log_dir,
        )
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate four-way reasoning strategy stars into the House")
    parser.add_argument("--house-dir", default=str(DEFAULT_HOUSE_DIR))
    parser.add_argument("--galaxy-dir", default=str(DEFAULT_GALAXY_DIR))
    parser.add_argument("--failure-log-dir", default=str(DEFAULT_FAILURE_LOG_DIR))
    args = parser.parse_args()
    stats = populate_reasoning_strategies(
        house_dir=args.house_dir,
        galaxy_dir=args.galaxy_dir,
        failure_log_dir=args.failure_log_dir,
    )
    for name, payload in stats.items():
        print(
            f"{name}: before={payload['before']} after={payload['after']} "
            f"appended={payload['appended']} replaced={payload['replaced']} removed={payload['removed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
