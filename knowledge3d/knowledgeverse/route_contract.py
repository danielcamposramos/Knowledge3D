"""Shared sovereign route-contract tables for generators and resident repair."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable

ROUTE_ROLE_REF_KEYS = (
    "router_refs",
    "executor_refs",
    "validator_refs",
    "anti_pattern_refs",
)

ROUTE_CONTRACT_SCHEMA_VERSION = 3

LANGUAGE_MEANING_MIRROR_PREFIX = "language_"

REASONING_ANCHOR_ID_REPAIRS = {
    "quantity_role_initial": "reasoning_quantity_role_initial",
    "quantity_role_delta": "reasoning_quantity_role_delta",
}

HALTING_PROFILE_IDS = frozenset(
    {
        "halting_threshold_elimination",
        "halting_threshold_math",
        "halting_threshold_spatial",
        "halting_threshold_default",
    }
)

LANGUAGE_ROUTE_EXEMPT_IDS = frozenset(
    {
        "langbook_ch1_symbols",
        "langbook_sec2_reference_words",
        "langbook_ch3_grammar",
        "langbook_sec3_literals",
        "langbook_sec3_sequences",
        "langbook_sec3_comparison",
        "langbook_page_emit_answer",
        "langbook_ch4_expression",
        "langbook_page_reading_practice",
        "meta_rule_parse_override_algebra",
        "meta_rule_parse_override_domain",
    }
)

GRAMMAR_REASONING_EXECUTOR_IDS = (
    "grammar_forward_entity_extraction",
    "grammar_quantity_unit_binding",
    "grammar_backward_goal_tracing",
    "grammar_dependency_dag_build",
    "grammar_operation_chain_construction",
    "grammar_operation_chain_left_fold",
    "grammar_operation_chain_nested",
    "grammar_operation_chain_ratio",
    "grammar_operation_chain_difference_then_multiply",
    "grammar_recursive_subtask_decomposition",
    "grammar_quantity_role_initial",
    "grammar_quantity_role_delta",
    "grammar_quantity_role_multiplier",
    "grammar_template_slot_binding",
    "grammar_result_normalization",
    "grammar_validate_units_and_magnitude",
    "grammar_subject_domain_alignment",
    "grammar_option_elimination",
    "grammar_compare_options_by_clues",
    "grammar_factual_lookup",
    "grammar_option_verification",
)

_GRAMMAR_REASONING_CHAIN_DECOMPOSITION_IDS = frozenset(
    {
        "grammar_dependency_dag_build",
        "grammar_operation_chain_construction",
        "grammar_operation_chain_left_fold",
        "grammar_operation_chain_nested",
        "grammar_operation_chain_ratio",
        "grammar_operation_chain_difference_then_multiply",
        "grammar_recursive_subtask_decomposition",
    }
)

_GRAMMAR_REASONING_OPTION_IDS = frozenset(
    {
        "grammar_option_elimination",
        "grammar_compare_options_by_clues",
        "grammar_option_verification",
    }
)

_GRAMMAR_REASONING_NORMALIZATION_IDS = frozenset(
    {
        "grammar_result_normalization",
        "grammar_validate_units_and_magnitude",
    }
)


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


def _entry_ids(entries: Iterable[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        str(entry.get("id") or "").strip()
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("id") or "").strip()
    )


def _entry_keys(entries: Iterable[dict[str, Any]]) -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            continue
        galaxy = str(entry.get("galaxy") or entry.get("domain") or "").strip().lower()
        keys.add((galaxy, entry_id))
    return frozenset(keys)


def _grammar_reasoning_executor_contract(entry_id: str) -> dict[str, Any]:
    anti_pattern_refs = _merge_unique(
        [
            "anti_pattern_missing_validator_traversal",
            "anti_pattern_symbol_meaning_drift",
        ],
        ["anti_pattern_option_emission_without_comparison"] if entry_id in _GRAMMAR_REASONING_OPTION_IDS else [],
        ["anti_pattern_answer_format_mismatch"] if entry_id in _GRAMMAR_REASONING_NORMALIZATION_IDS else [],
    )
    branch_topk = 3 if entry_id in _GRAMMAR_REASONING_CHAIN_DECOMPOSITION_IDS else 2
    return {
        "route_family": "GRAMMAR",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "validator_refs": [
            "grammar_normalization_validator",
            "grammar_answer_validator",
        ],
        "anti_pattern_refs": anti_pattern_refs,
        "route_policy": {
            "requires_validator": True,
            "answer_gate": False,
            "branch_topk": branch_topk,
        },
    }


GRAMMAR_REASONING_EXECUTOR_CONTRACTS: dict[str, dict[str, Any]] = {
    entry_id: _grammar_reasoning_executor_contract(entry_id)
    for entry_id in GRAMMAR_REASONING_EXECUTOR_IDS
}


@lru_cache(maxsize=1)
def route_exempt_anchor_ids() -> frozenset[str]:
    return frozenset(entry_id for _galaxy, entry_id in route_exempt_anchor_keys()) | HALTING_PROFILE_IDS | LANGUAGE_ROUTE_EXEMPT_IDS


@lru_cache(maxsize=1)
def route_exempt_anchor_keys() -> frozenset[tuple[str, str]]:
    from scripts.populate_game_mechanics import build_game_mechanics_entries, build_game_reality_entries
    from scripts.populate_reasoning_strategies import (
        build_reasoning_meaning_entries,
        build_reasoning_reality_entries,
    )

    return (
        _entry_keys(build_reasoning_meaning_entries())
        | _entry_keys(build_reasoning_reality_entries())
        | _entry_keys(build_game_mechanics_entries())
        | _entry_keys(build_game_reality_entries())
    )


def apply_route_exempt_anchor_contract(entry: dict[str, Any]) -> dict[str, Any]:
    raw = dict(entry)
    metadata = dict(raw.get("metadata") or {})
    for container in (raw, metadata):
        container["selection_role"] = "unknown"
        container["layer_id"] = 0
        container["answer_eligible"] = False
        container["sovereign_route_exempt"] = True
        container["route_contract_schema_version"] = int(ROUTE_CONTRACT_SCHEMA_VERSION)
        container.pop("route_family", None)
        container.pop("route_policy", None)
        for key in ROUTE_ROLE_REF_KEYS:
            container.pop(key, None)
    raw["metadata"] = metadata
    return raw


def apply_route_capable_contract(entry: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    raw = dict(entry)
    metadata = dict(raw.get("metadata") or {})
    route_family = str(contract.get("route_family") or "").strip()
    selection_role = str(contract.get("selection_role") or "").strip().lower()
    layer_id = int(contract.get("layer_id") or 0)
    answer_eligible = bool(contract.get("answer_eligible", False))
    route_policy = dict(contract.get("route_policy") or {})
    for container in (raw, metadata):
        if route_family:
            container["route_family"] = route_family
        container["selection_role"] = selection_role
        container["layer_id"] = layer_id
        container["answer_eligible"] = answer_eligible
        container["sovereign_route_exempt"] = False
        container["route_contract_schema_version"] = int(ROUTE_CONTRACT_SCHEMA_VERSION)
        if route_policy:
            container["route_policy"] = dict(route_policy)
        else:
            container.pop("route_policy", None)
    for key in ROUTE_ROLE_REF_KEYS:
        values = list(contract.get(key) or [])
        if values:
            raw[key] = list(values)
            metadata[key] = list(values)
        else:
            raw.pop(key, None)
            metadata.pop(key, None)
    raw["metadata"] = metadata
    return raw


def is_foundational_route_exempt_substrate(
    entry: dict[str, Any],
    metadata: dict[str, Any],
    *,
    galaxy_key: str = "",
) -> bool:
    entry_id = str(entry.get("id") or metadata.get("id") or "").strip()
    category = str(entry.get("category") or metadata.get("category") or "").strip().lower()
    resolved_galaxy = str(
        galaxy_key
        or entry.get("galaxy")
        or metadata.get("galaxy")
        or entry.get("domain")
        or metadata.get("domain")
        or ""
    ).strip().lower()
    if resolved_galaxy == "reality" and entry_id.startswith("reality_anchor_"):
        return True
    layer_value = metadata.get("layer", entry.get("layer"))
    try:
        if int(layer_value) == 2 and category in {"concept", "definition", "fact", "biology_fact"}:
            return True
    except (TypeError, ValueError):
        return False
    return False


def language_meaning_mirror_id(star_id: str) -> str:
    resolved = str(star_id or "").strip()
    if not resolved:
        return ""
    if resolved.startswith(LANGUAGE_MEANING_MIRROR_PREFIX):
        return resolved
    return f"{LANGUAGE_MEANING_MIRROR_PREFIX}{resolved}"
