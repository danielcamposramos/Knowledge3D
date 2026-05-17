"""Deterministic Stargate-side normalization for proceduralizer bundles.

This module is the canonical post-model normalization surface before payload
rows are ingested into the Knowledgeverse.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

from knowledge3d.ingestion.proceduralizer_contract import (
    ProceduralizerBundle,
    ProceduralizerPacket,
    ProceduralizerRequest,
    slugify_meaning_name,
)
from knowledge3d.ingestion.universal_knowledge import load_all_omw


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_'-]+", re.IGNORECASE)
_FOUNDATIONAL_LAYER_IDS = {
    "form": 1,
    "meaning": 2,
    "rule": 3,
    "meta_rule": 4,
}
_HISTORY_TOKENS = frozenset(
    {
        "ancient",
        "archaeology",
        "bronze",
        "bce",
        "ce",
        "century",
        "civilization",
        "dynasty",
        "empire",
        "historical",
        "history",
        "kingdom",
        "mesopotamia",
        "neolithic",
        "paleolithic",
        "pharaoh",
        "prehistory",
        "rome",
        "sumer",
    }
)
_PREHISTORY_TOKENS = frozenset(
    {
        "paleolithic",
        "mesolithic",
        "neolithic",
        "prehistory",
        "stone",
        "hunter",
        "gatherer",
        "cuneiform",
    }
)
_LANGUAGE_TOKENS = frozenset(
    {
        "alphabet",
        "cuneiform",
        "grammar",
        "hieroglyphic",
        "language",
        "notation",
        "script",
        "symbol",
        "writing",
    }
)
_MATH_TOKENS = frozenset(
    {
        "angle",
        "arithmetic",
        "equation",
        "geometry",
        "mathematics",
        "numeral",
        "number",
        "ratio",
        "sexagesimal",
        "theorem",
        "triangle",
    }
)
_PHYSICS_TOKENS = frozenset(
    {
        "electromagnetic",
        "electric",
        "field",
        "force",
        "geomagnetism",
        "magnetic",
        "motion",
        "wave",
    }
)
_CHEMISTRY_TOKENS = frozenset(
    {
        "acid",
        "atom",
        "chemical",
        "chemistry",
        "compound",
        "element",
        "molecule",
        "reaction",
    }
)
_BIOLOGY_TOKENS = frozenset(
    {
        "anatomy",
        "biology",
        "cell",
        "genetic",
        "organism",
        "species",
    }
)
_TOOLS_TOKENS = frozenset({"protocol", "runtime", "system", "tool"})
_RELATION_ID_RE = re.compile(r"^[a-z][a-z0-9_]+$")
_EXTERNAL_REF_ID_RE = re.compile(r'"id"\s*:\s*"([A-Za-z][A-Za-z0-9_:-]+)"')
_EXTERNAL_REF_PREFIXES = (
    "artifact_",
    "civilization_",
    "concept_",
    "era_",
    "event_",
    "fact_",
    "grammar_",
    "high_school_",
    "notation_",
    "pattern_",
    "period_",
    "prehistory",
    "reality_",
    "rule_",
    "ruler_",
    "script_",
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


def _text_tokens(text: str) -> list[str]:
    normalized = str(text or "").replace("/", " ").replace("|", " ")
    return [token.lower() for token in _TOKEN_RE.findall(normalized)]


@lru_cache(maxsize=1)
def _english_synset_index() -> dict[str, list[str]]:
    synsets = load_all_omw()
    index: dict[str, list[str]] = {}
    for synset_id, entry in synsets.items():
        star_id = f"synset_{synset_id.replace('-', '_')}"
        for lemma in entry.lemmas.get("en", []):
            token = str(lemma or "").strip().lower()
            if not token:
                continue
            bucket = index.setdefault(token, [])
            if star_id not in bucket:
                bucket.append(star_id)
    return index


def _word_refs_from_text(text: str, *, limit: int = 6) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    index = _english_synset_index()
    for token in _text_tokens(text):
        for ref in index.get(token, []):
            if ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
            if len(refs) >= limit:
                return refs
    return refs


def _row_entry_text(row: dict[str, Any]) -> str:
    entry = dict(row.get("entry") or {})
    metadata = dict(entry.get("metadata") or {})
    parts: list[str] = [
        str(entry.get("id") or ""),
        str(entry.get("name") or ""),
        str(entry.get("rpn_program") or ""),
        str(metadata.get("source_path") or ""),
    ]
    surface_forms = metadata.get("surface_forms")
    if isinstance(surface_forms, dict):
        parts.extend(str(value or "") for value in surface_forms.values())
    relationships = metadata.get("relationships")
    if isinstance(relationships, list):
        for relation in relationships:
            if not isinstance(relation, dict):
                continue
            parts.extend(
                [
                    str(relation.get("from") or ""),
                    str(relation.get("relation") or ""),
                    str(relation.get("to") or ""),
                ]
            )
    return " ".join(part for part in parts if part).strip()


def _merge_enrichment_contexts(*contexts: dict[str, Any] | None) -> dict[str, Any]:
    merged_ids: set[str] = set()
    merged_tokens: dict[str, set[str]] = {}
    for context in contexts:
        if not isinstance(context, dict):
            continue
        merged_ids.update(str(item) for item in (context.get("id_set") or set()) if str(item or "").strip())
        for token, ids in dict(context.get("token_to_ids") or {}).items():
            token_text = str(token or "").strip().lower()
            if not token_text:
                continue
            bucket = merged_tokens.setdefault(token_text, set())
            for entry_id in ids or []:
                entry_text = str(entry_id or "").strip()
                if entry_text:
                    bucket.add(entry_text)
    return {"id_set": merged_ids, "token_to_ids": merged_tokens}


def _context_from_entry_dicts(entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    id_set: set[str] = set()
    token_to_ids: dict[str, set[str]] = {}
    for entry_candidate in entries:
        if not isinstance(entry_candidate, dict):
            continue
        entry = dict(entry_candidate.get("entry") or entry_candidate)
        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            continue
        id_set.add(entry_id)
        text = _row_entry_text({"entry": entry})
        for token in set(_text_tokens(text)):
            token_to_ids.setdefault(token, set()).add(entry_id)
    return {"id_set": id_set, "token_to_ids": token_to_ids}


@lru_cache(maxsize=4)
def _load_external_enrichment_context_cached(path_text: str, mtime_ns: int) -> dict[str, Any]:
    path = Path(path_text)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"id_set": set(), "token_to_ids": {}}
    id_set: set[str] = set()
    token_to_ids: dict[str, set[str]] = {}
    for match in _EXTERNAL_REF_ID_RE.finditer(text):
        entry_id = str(match.group(1) or "").strip()
        if not entry_id or not entry_id.startswith(_EXTERNAL_REF_PREFIXES):
            continue
        id_set.add(entry_id)
        for token in set(_text_tokens(entry_id.replace("_", " "))):
            token_to_ids.setdefault(token, set()).add(entry_id)
    return {"id_set": id_set, "token_to_ids": token_to_ids}


def load_external_enrichment_context(source: str | Path | None) -> dict[str, Any]:
    if source is None:
        return {"id_set": set(), "token_to_ids": {}}
    path = Path(source).expanduser()
    if not path.exists() or not path.is_file():
        return {"id_set": set(), "token_to_ids": {}}
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except Exception:
        return {"id_set": set(), "token_to_ids": {}}
    return _load_external_enrichment_context_cached(str(resolved), int(stat.st_mtime_ns))


def build_row_enrichment_context(
    rows: Iterable[dict[str, Any]],
    *,
    external_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _merge_enrichment_contexts(_context_from_entry_dicts(rows), external_context)


def _infer_taxonomy_refs(text: str, existing: list[str]) -> list[str]:
    refs = list(existing)
    tokens = set(_text_tokens(text))
    if tokens & _PREHISTORY_TOKENS:
        refs.extend(["prehistory", "high_school_world_history"])
    elif tokens & _HISTORY_TOKENS:
        refs.append("high_school_world_history")
    if tokens & _LANGUAGE_TOKENS:
        refs.append("concept_language")
    if tokens & _MATH_TOKENS:
        refs.append("concept_mathematics")
    if tokens & _PHYSICS_TOKENS:
        refs.append("concept_physics")
    if tokens & _CHEMISTRY_TOKENS:
        refs.append("concept_chemistry")
    if tokens & _BIOLOGY_TOKENS:
        refs.append("concept_biology")
    if tokens & _TOOLS_TOKENS:
        refs.append("concept_tool")
    return _normalize_refs(refs)


def _infer_symbol_refs(text: str, existing: list[str], *, context: dict[str, Any]) -> list[str]:
    refs = list(existing)
    lowered = str(text or "").lower()
    id_set = set(context.get("id_set") or set())
    if any(token in lowered for token in ("b.c.e", "bce", "bc")) and "notation_bce_dating" in id_set:
        refs.append("notation_bce_dating")
    if any(token in lowered for token in ("c.e", "ce", "ad")) and "notation_ce_dating" in id_set:
        refs.append("notation_ce_dating")
    return _normalize_refs(refs)


def _infer_reality_refs(row: dict[str, Any], *, context: dict[str, Any]) -> list[str]:
    entry = dict(row.get("entry") or {})
    metadata = dict(entry.get("metadata") or {})
    refs = list(metadata.get("reality_refs") or [])
    self_id = str(entry.get("id") or "").strip()
    id_set = set(context.get("id_set") or set())
    relationships = metadata.get("relationships")
    if isinstance(relationships, list):
        for relation in relationships:
            if not isinstance(relation, dict):
                continue
            for endpoint_key in ("from", "to"):
                endpoint = str(relation.get(endpoint_key) or "").strip()
                if not endpoint or endpoint == self_id or not _RELATION_ID_RE.match(endpoint):
                    continue
                refs.append(endpoint)
    lowered_tokens = set(_text_tokens(_row_entry_text(row)))
    token_index: dict[str, set[str]] = dict(context.get("token_to_ids") or {})
    scored: dict[str, int] = {}
    for token in lowered_tokens:
        if len(token) < 5:
            continue
        for candidate in token_index.get(token, set()):
            if candidate == self_id:
                continue
            scored[candidate] = scored.get(candidate, 0) + 1
    for candidate, score in sorted(scored.items(), key=lambda item: (-item[1], item[0])):
        if score < 2:
            continue
        if candidate in id_set:
            refs.append(candidate)
        if len(refs) >= 6:
            break
    return _normalize_refs(refs)


def _infer_grammar_refs(row: dict[str, Any]) -> list[str]:
    entry = dict(row.get("entry") or {})
    metadata = dict(entry.get("metadata") or {})
    refs = list(metadata.get("grammar_refs") or [])
    layer_kind = str(metadata.get("procedural_layer_kind") or "").strip().lower()
    text = _row_entry_text(row).lower()
    if layer_kind == "form":
        refs.extend(["grammar_forward_entity_extraction", "grammar_result_normalization"])
    if layer_kind == "meaning" and any(token in text for token in ("bce", "ce", "dynasty", "empire", "civilization", "kingdom")):
        refs.append("grammar_factual_lookup")
    if any(token in text for token in ("bce", "ce", "year", "century", "dynasty")):
        refs.append("grammar_subject_domain_alignment")
    return _normalize_refs(refs)


def second_pass_enrich_payload_row(row: dict[str, Any], *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(row)
    entry = dict(raw.get("entry") or {})
    metadata = dict(entry.get("metadata") or {})
    context = dict(context or {})
    layer_kind = str(metadata.get("procedural_layer_kind") or "").strip().lower()
    foundational_layer_id = int(_FOUNDATIONAL_LAYER_IDS.get(layer_kind, 0))
    text = _row_entry_text(raw)

    metadata["taxonomy_refs"] = _infer_taxonomy_refs(text, list(metadata.get("taxonomy_refs") or []))
    metadata["word_refs"] = _normalize_refs(list(metadata.get("word_refs") or []) + _word_refs_from_text(text))
    metadata["symbol_refs"] = _infer_symbol_refs(text, list(metadata.get("symbol_refs") or []), context=context)
    metadata["reality_refs"] = _infer_reality_refs({"entry": entry}, context=context)
    metadata["grammar_refs"] = _infer_grammar_refs({"entry": entry})
    metadata["foundational_layer_kind"] = layer_kind
    metadata["foundational_layer_id"] = foundational_layer_id
    metadata["meta_refs"] = _normalize_refs(list(metadata.get("meta_refs") or []) + ["second_pass_symlink_enriched"])

    entry["metadata"] = metadata
    entry["foundational_layer_kind"] = layer_kind
    entry["foundational_layer_id"] = foundational_layer_id
    raw["entry"] = entry
    return raw


def second_pass_enrich_payload_rows(
    rows: Iterable[dict[str, Any]],
    *,
    context: dict[str, Any] | None = None,
    external_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    row_list = [dict(row) for row in rows if isinstance(row, dict)]
    resolved_context = context or build_row_enrichment_context(row_list, external_context=external_context)
    return [second_pass_enrich_payload_row(row, context=resolved_context) for row in row_list]


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
    foundational_layer_id = int(_FOUNDATIONAL_LAYER_IDS.get(packet.layer_kind, 0))
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
        "sources": _normalize_refs(packet.sources),
        "relationships": list(packet.relationships),
        "confidence": float(packet.confidence),
        "needs_review": bool(packet.needs_review),
        "symlink": "character_galaxy|word_galaxy",
        "procedural_layer_kind": packet.layer_kind,
        "foundational_layer_kind": packet.layer_kind,
        "foundational_layer_id": foundational_layer_id,
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
    foundational_layer_id = int(_FOUNDATIONAL_LAYER_IDS.get(packet.layer_kind, 0))
    entry: dict[str, Any] = {
        "id": entry_id,
        "name": packet.summary or entry_id,
        "domain": galaxy.lower(),
        "category": f"proceduralizer_{packet.layer_kind}",
        "rpn_program": packet.meaning_rpn or f"{str(packet.domain or 'General').upper()} CONTENT ENTRY",
        "metadata": packet_metadata(packet, request),
        "foundational_layer_kind": packet.layer_kind,
        "foundational_layer_id": foundational_layer_id,
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
    rows = [packet_to_payload_row(packet, request) for packet in bundle.knowledge_packets]
    return second_pass_enrich_payload_rows(rows)


__all__ = [
    "build_row_enrichment_context",
    "bundle_to_payload_rows",
    "load_external_enrichment_context",
    "packet_galaxy",
    "packet_id",
    "packet_metadata",
    "packet_route_contract",
    "packet_to_payload_row",
    "second_pass_enrich_payload_row",
    "second_pass_enrich_payload_rows",
]
