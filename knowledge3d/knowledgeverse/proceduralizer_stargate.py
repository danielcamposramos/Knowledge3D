"""Deterministic Stargate-side normalization for proceduralizer bundles.

This module is the canonical post-model normalization surface before payload
rows are ingested into the Knowledgeverse.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from knowledge3d.ingestion.proceduralizer_contract import (
    ProceduralizerBundle,
    ProceduralizerPacket,
    ProceduralizerRequest,
    slugify_meaning_name,
)


def _sha(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _normalize_refs(value: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(value or []):
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def packet_id(packet: ProceduralizerPacket, request: ProceduralizerRequest) -> str:
    if packet.star_id:
        return packet.star_id
    if packet.proposed_star_id:
        return packet.proposed_star_id
    return f"{packet.layer_kind}_{slugify_meaning_name(packet.summary)}_{_sha(request.source_id + '|' + packet.summary)}"


def packet_galaxy(packet: ProceduralizerPacket) -> str:
    domain = str(packet.domain or "General").strip().lower()
    if domain == "mathematics":
        return "Math"
    if domain == "language":
        return "Grammar"
    if domain in {"visual", "audio"}:
        return domain.title()
    if packet.layer_kind == "form":
        return "Word"
    return "Reality"


def packet_route_contract(packet: ProceduralizerPacket) -> dict[str, Any] | None:
    if packet.route_contract:
        return dict(packet.route_contract)
    if packet.layer_kind not in {"rule", "meta_rule"}:
        return None
    domain = str(packet.domain or "General").strip().lower()
    if domain == "mathematics":
        return {
            "route_family": "MATH",
            "selection_role": "executor",
            "layer_id": 3,
            "answer_eligible": False,
            "route_policy": {"requires_validator": True, "answer_gate": False, "branch_topk": 2},
            "executor_refs": [
                "math_quantity_binding_executor",
                "math_goal_trace_executor",
                "math_operation_chain_executor",
            ],
            "validator_refs": [
                "math_normalization_validator",
                "math_unit_magnitude_validator",
                "math_answer_validator",
            ],
            "anti_pattern_refs": [
                "anti_pattern_unchecked_unit_transfer",
                "anti_pattern_missing_validator_traversal",
            ],
        }
    if domain == "language":
        return {
            "route_family": "QUESTION" if "question" in packet.meaning_rpn.lower() else "GRAMMAR",
            "selection_role": "executor",
            "layer_id": 3,
            "answer_eligible": False,
            "route_policy": {"requires_validator": True, "answer_gate": False, "branch_topk": 2},
            "executor_refs": [
                "question_subject_grounding_executor",
                "question_choice_materializer",
            ] if "question" in packet.meaning_rpn.lower() else [
                "grammar_forward_entity_extraction",
                "grammar_result_normalization",
            ],
            "validator_refs": [
                "question_choice_alignment_validator",
                "question_answer_validator",
            ] if "question" in packet.meaning_rpn.lower() else [
                "grammar_normalization_validator",
                "grammar_answer_validator",
            ],
            "anti_pattern_refs": [
                "anti_pattern_shallow_option_elimination_without_subject_grounding",
                "anti_pattern_validator_as_answer_leakage",
            ] if "question" in packet.meaning_rpn.lower() else [
                "anti_pattern_symbol_meaning_drift",
                "anti_pattern_answer_format_mismatch",
            ],
        }
    return {
        "route_family": "GENERAL",
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "route_policy": {"requires_validator": True, "answer_gate": False, "branch_topk": 2},
        "executor_refs": ["general_compare_executor", "general_evidence_executor"],
        "validator_refs": ["general_consistency_validator", "general_answer_validator"],
        "anti_pattern_refs": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_generic_language_factual_winner",
        ],
    }


def packet_metadata(packet: ProceduralizerPacket, request: ProceduralizerRequest) -> dict[str, Any]:
    route = packet_route_contract(packet)
    metadata: dict[str, Any] = {
        "source": f"proceduralizer_{request.source_kind}",
        "source_id": request.source_id,
        "source_path": request.source_path,
        "domain_hint": request.domain_hint,
        "ingest_mode": request.ingest_mode,
        "surface_forms": dict(packet.surface_forms),
        "symbol_refs": _normalize_refs(packet.symbol_refs),
        "word_refs": _normalize_refs(packet.word_refs),
        "taxonomy_refs": _normalize_refs(packet.taxonomy_refs),
        "grammar_refs": _normalize_refs(packet.grammar_refs),
        "reality_refs": _normalize_refs(packet.reality_refs),
        "meta_refs": _normalize_refs(packet.meta_refs),
        "relationships": list(packet.relationships),
        "confidence": float(packet.confidence),
        "needs_review": bool(packet.needs_review),
        "symlink": "character_galaxy|word_galaxy",
        "procedural_layer_kind": packet.layer_kind,
    }
    if route is not None:
        metadata.update(route)
    else:
        metadata.update(
            {
                "selection_role": "unknown",
                "layer_id": 0,
                "answer_eligible": False,
                "sovereign_route_exempt": True,
            }
        )
    return metadata


def packet_to_payload_row(packet: ProceduralizerPacket, request: ProceduralizerRequest) -> dict[str, Any]:
    galaxy = packet_galaxy(packet)
    entry_id = packet_id(packet, request)
    route = packet_route_contract(packet)
    entry: dict[str, Any] = {
        "id": entry_id,
        "name": packet.summary or entry_id,
        "domain": galaxy.lower(),
        "category": f"proceduralizer_{packet.layer_kind}",
        "rpn_program": packet.meaning_rpn or f"{str(packet.domain or 'General').upper()} CONTENT ENTRY",
        "metadata": packet_metadata(packet, request),
    }
    if route is not None:
        entry.update(route)
    else:
        entry.update(
            {
                "selection_role": "unknown",
                "layer_id": 0,
                "answer_eligible": False,
                "sovereign_route_exempt": True,
            }
        )
    return {"galaxy": galaxy, "entry": entry}


def bundle_to_payload_rows(bundle: ProceduralizerBundle, request: ProceduralizerRequest) -> list[dict[str, Any]]:
    if bundle.ingest_action != "augment":
        return []
    return [packet_to_payload_row(packet, request) for packet in bundle.knowledge_packets]


__all__ = [
    "bundle_to_payload_rows",
    "packet_galaxy",
    "packet_id",
    "packet_metadata",
    "packet_route_contract",
    "packet_to_payload_row",
]
