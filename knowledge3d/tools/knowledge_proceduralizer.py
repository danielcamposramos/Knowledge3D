"""Proceduralize benchmark knowledge into MeaningCentricStar entries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from knowledge3d.ingestion.ollama_manager import OllamaManager
from knowledge3d.ingestion.proceduralizer_contract import (
    PROCEDURALIZER_MODEL_PROFILES,
    PROCEDURALIZER_SYSTEM_PROMPT,
    ProceduralizerBundle,
    ProceduralizerPacket,
    ProceduralizerRequest,
    parse_bundle,
)
from knowledge3d.ingestion.proceduralizer_wine import ProceduralizerWineBridge
from knowledge3d.ingestion.universal_knowledge import (
    build_meaning_layer_stars,
    iter_domains,
    iter_elements,
    iter_physical_constants,
    load_all_omw,
)
from knowledge3d.knowledgeverse.proceduralizer_stargate import bundle_to_payload_rows

from .augmentation_providers import AugmentationResult
from .content_to_stars import result_to_star, write_stars_jsonl


PROCEDURALIZATION_SYSTEM_PROMPT = PROCEDURALIZER_SYSTEM_PROMPT


SOURCE_MODEL_MAP = {
    "mmlu_train": PROCEDURALIZER_MODEL_PROFILES["quality"],
    "mmlu_val": PROCEDURALIZER_MODEL_PROFILES["quality"],
    "gsm8k_train": PROCEDURALIZER_MODEL_PROFILES["quality"],
}

MODEL_OPTIONS = {
    # Verified with `ollama show` on 2026-04-06:
    # - qwen3.5:397b-cloud context length = 262144
    # - kimi-k2-thinking:cloud context length = 262144
    # - glm-5:cloud context length = 202752
    # - deepseek-v3.2:cloud context length = 163840
    # Proceduralizer prompts are bounded well below those maxima, so the working
    # num_ctx is intentionally capped for better latency/cost while preserving
    # headroom. The proceduralizer clears context between distinct sources and
    # re-chunks oversized single-source content with overlap, so max model
    # context is not required for the default batch path.
    PROCEDURALIZER_MODEL_PROFILES["quality"]: {
        "temperature": 0.1,
        "num_predict": 3072,
        "num_ctx": 32768,
        "think": False,
    },
    PROCEDURALIZER_MODEL_PROFILES["audit_reasoning"]: {
        "temperature": 0.1,
        "num_predict": 4096,
        "num_ctx": 32768,
        "think": False,
    },
    PROCEDURALIZER_MODEL_PROFILES["long_context_engineering"]: {
        "temperature": 0.1,
        "num_predict": 3072,
        "num_ctx": 65536,
        "think": False,
    },
    PROCEDURALIZER_MODEL_PROFILES["balanced_fallback"]: {
        "temperature": 0.1,
        "num_predict": 2048,
        "num_ctx": 24576,
        "think": False,
    },
}

MMLU_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/MMLU/data")
GSM8K_DEFAULT_PATH = Path("/K3D/K3D_llama_cpp/datasets/GSM8K")

_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(r"####\s*([^\n]+)")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

PROCEDURALIZER_MAX_CONTENT_CHARS = 12000
PROCEDURALIZER_CHUNK_OVERLAP_CHARS = 900


def _hits(text: str, keywords: set[str]) -> bool:
    combined = str(text or "").lower()
    return any(keyword in combined for keyword in keywords)


def _subject_to_domain(subject: str) -> str:
    lowered = str(subject or "").lower()
    if any(keyword in lowered for keyword in ["math", "algebra", "calculus", "geometry", "statistics"]):
        return "Mathematics"
    if any(keyword in lowered for keyword in ["physics", "astronomy", "electrical"]):
        return "Physics"
    if "chemistry" in lowered:
        return "Physics"
    if any(keyword in lowered for keyword in ["biology", "anatomy", "medicine", "nutrition", "clinical"]):
        return "Biology"
    if any(keyword in lowered for keyword in ["computer", "machine_learning", "security"]):
        return "Tools"
    return "General"


def _slug(text: str) -> str:
    return _NON_ALNUM_RE.sub("_", str(text or "").strip().lower()).strip("_") or "entry"


def _sha(text: str, *, size: int = 12) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:size]


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


def _meaning_star_refs(question_text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[a-z][a-z_'-]+", str(question_text or "").lower())
    refs: list[str] = []
    seen: set[str] = set()
    index = _english_synset_index()
    for token in tokens:
        for star_id in index.get(token, []):
            if star_id in seen:
                continue
            seen.add(star_id)
            refs.append(star_id)
            if len(refs) >= limit:
                return refs
    return refs


def build_rag_context(domain: str, subject: str, question_text: str) -> str:
    """Build a compact reference menu of existing star ids."""
    refs: list[str] = []
    refs.append("## Existing star_ids (REFERENCE these, do not restate their content):")
    refs.append("")
    refs.append("### Taxonomy")
    refs.append(
        "concept_mathematics, concept_physics, concept_chemistry, "
        "concept_biology, concept_language, concept_tool"
    )

    combined = f"{domain} {subject} {question_text}".lower()

    if _hits(combined, {"chem", "element", "atom", "molecule", "compound", "reaction", "oxide", "acid", "metal", "halogen", "periodic", "bond", "bio", "anatomy", "medicine", "organic", "cell"}):
        refs.append("")
        refs.append("### Chemical elements (star_id = element_{symbol})")
        for element in list(iter_elements())[:36]:
            refs.append(
                f"  element_{element.symbol.lower()} = {element.name_en}, "
                f"Z={element.atomic_number}, mass={element.atomic_mass}"
            )

    if _hits(combined, {"phys", "force", "energy", "velocity", "gravity", "light", "planck", "boltzmann", "electric", "magnetic", "thermo", "momentum", "wave", "frequency", "astro", "optic"}):
        refs.append("")
        refs.append("### Physical constants (star_id = constant_{key})")
        for constant in iter_physical_constants():
            refs.append(f"  constant_{constant.key} = {constant.name} = {constant.value} {constant.unit}")

    if _hits(combined, {"math", "algebra", "calculus", "geometry", "unit", "convert", "distance", "speed", "mass", "temperature", "pressure", "econ", "statistic", "probability"}):
        refs.append("")
        refs.append("### Measurement units (star_id = unit_{domain}_{unit})")
        for domain_entry in list(iter_domains())[:8]:
            unit_refs = ", ".join(f"unit_{domain_entry.key}_{name}" for name in list(domain_entry.units.keys())[:4])
            refs.append(f"  {domain_entry.key}: {unit_refs}")

    if _hits(combined, {"material", "steel", "glass", "water", "wood", "concrete", "alloy"}):
        refs.append("")
        refs.append("### Materials (star_id = material_{name})")
        refs.append("  material_water, material_steel, material_glass, material_wood, material_concrete")

    refs.append("")
    refs.append("### Word meanings (star_id = synset_{id}, one star per meaning, multilingual)")
    meaning_refs = _meaning_star_refs(question_text)
    if meaning_refs:
        for star_id in meaning_refs:
            refs.append(f"  {star_id}")
    else:
        sample = build_meaning_layer_stars(min_languages=3, limit=3)
        for star in sample:
            refs.append(f"  {star.star_id}")
    refs.append("  Reference format: synset_XXXXXXXX_X")

    refs.append("")
    refs.append("### All available star_id prefixes:")
    refs.append(
        "  element_*, constant_*, unit_*, material_*, script_*, numeral_system_*, "
        "format_*, standard_size_*, synset_*"
    )
    return "\n".join(refs)


def build_request_from_entry(
    entry: dict[str, Any],
    *,
    quality_profile: str = "quality",
    source_kind: str = "benchmark",
) -> ProceduralizerRequest:
    question_text = str(entry.get("question") or entry.get("content") or "").strip()
    content = str(entry.get("content") or question_text).strip()
    domain_hint = str(entry.get("domain_hint") or "General").strip() or "General"
    source_id = str(entry.get("entry_id") or entry.get("id") or _slug(question_text) or "entry")
    source_path = str(entry.get("source_path") or entry.get("path") or entry.get("source") or "").strip()
    context_chunks: list[str] = []
    options = entry.get("options")
    if isinstance(options, list) and options:
        context_chunks.append("Options: " + " | ".join(str(item).strip() for item in options if str(item).strip()))
    answer = str(entry.get("correct_answer") or "").strip()
    if answer:
        context_chunks.append(f"Gold answer anchor: {answer[:240]}")
    subject = str(entry.get("subject") or "").strip()
    if subject:
        context_chunks.append(f"Subject anchor: {subject}")
    return ProceduralizerRequest(
        source_kind=source_kind,
        source_id=source_id,
        source_path=source_path,
        domain_hint=domain_hint,
        content=content,
        context_chunks=context_chunks,
        existing_ref_menu=build_rag_context(domain_hint, subject, question_text),
        quality_profile=quality_profile,
        ingest_mode="augment",
    )


def _packet_to_result(packet: ProceduralizerPacket, *, provider: str, raw_response: str) -> AugmentationResult:
    return AugmentationResult(
        summary=str(packet.summary or packet.proposed_star_id or packet.star_id or "proceduralized entry").strip(),
        entities=[{"type": "entity", "name": ref, "content": ref} for ref in packet.word_refs[:8]],
        relationships=list(packet.relationships),
        domain=str(packet.domain or "General").strip() or "General",
        meaning_rpn_hint=str(packet.meaning_rpn or "GENERAL CONTENT ENTRY").strip(),
        taxonomy_refs=list(packet.taxonomy_refs),
        surface_forms=dict(packet.surface_forms or {"en": packet.summary or "proceduralized entry"}),
        confidence=max(0.0, min(1.0, float(packet.confidence))),
        provider=provider,
        raw_response=raw_response,
    )


def _bundle_primary_result(bundle: ProceduralizerBundle, *, entry: dict[str, Any], provider: str, raw_response: str) -> AugmentationResult:
    for packet in bundle.knowledge_packets:
        if packet.layer_kind in {"meaning", "rule", "meta_rule"}:
            return _packet_to_result(packet, provider=provider, raw_response=raw_response)
    if bundle.knowledge_packets:
        return _packet_to_result(bundle.knowledge_packets[0], provider=provider, raw_response=raw_response)
    return _fallback_result(entry, raw_response, provider=provider)


def chunk_source_content(
    content: str,
    *,
    max_chars: int = PROCEDURALIZER_MAX_CONTENT_CHARS,
    overlap_chars: int = PROCEDURALIZER_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    text = str(content or "")
    if len(text) <= int(max_chars):
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, int(max_chars) - int(overlap_chars))
    while start < len(text):
        end = min(len(text), start + int(max_chars))
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _merge_receipts(chunk_receipts: list[Any], *, provider: str, model: str) -> Any:
    from knowledge3d.ingestion.proceduralizer_contract import ProceduralizerReceipt

    packets: list[ProceduralizerPacket] = []
    schema_ok = True
    failure_code = ""
    retry_after_utc = ""
    request_hash_parts: list[str] = []
    response_hash_parts: list[str] = []
    raw_paths: list[str] = []
    for receipt in chunk_receipts:
        schema_ok = schema_ok and bool(receipt.schema_ok)
        if not failure_code and str(receipt.failure_code or "").strip():
            failure_code = str(receipt.failure_code).strip()
        if not retry_after_utc and str(getattr(receipt, "retry_after_utc", "") or "").strip():
            retry_after_utc = str(receipt.retry_after_utc).strip()
        request_hash_parts.append(str(receipt.request_hash))
        response_hash_parts.append(str(receipt.response_hash))
        if str(receipt.raw_response_path or "").strip():
            raw_paths.append(str(receipt.raw_response_path))
        packets.extend(list(receipt.parsed_bundle.knowledge_packets))
        if str(receipt.failure_code or "").strip() == "plan_limit_consumed":
            break
    ingest_action = "augment" if packets else "skip"
    if failure_code == "plan_limit_consumed":
        ingest_action = "reject"
    elif any(receipt.parsed_bundle.ingest_action == "needs_context" for receipt in chunk_receipts) and not packets:
        ingest_action = "needs_context"
    bundle = ProceduralizerBundle(ingest_action=ingest_action, knowledge_packets=packets)
    return ProceduralizerReceipt(
        status="completed" if schema_ok and not failure_code else "invalid_json",
        provider=provider,
        model=model,
        latency_ms=sum(int(receipt.latency_ms) for receipt in chunk_receipts),
        request_hash=_sha("|".join(request_hash_parts), size=16),
        response_hash=_sha("|".join(response_hash_parts), size=16),
        raw_response_path="|".join(raw_paths),
        schema_ok=schema_ok,
        failure_code=failure_code,
        retry_after_utc=retry_after_utc,
        parsed_bundle=bundle,
    )


def load_mmlu_entries(
    data_dir: Path,
    split: str = "val",
    *,
    subjects: list[str] | None = None,
    limit_per_subject: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Load MMLU entries as proceduralization records."""
    split_key = str(split or "val").strip().lower()
    split_dir = Path(data_dir) / split_key
    if not split_dir.exists():
        return
    wanted = {str(subject).strip().lower() for subject in list(subjects or []) if str(subject).strip()}
    per_subject: dict[str, int] = {}
    pattern = "*.csv" if split_key == "auxiliary_train" else f"*_{split_key}.csv"
    for csv_path in sorted(split_dir.glob(pattern)):
        if split_key == "auxiliary_train":
            subject = csv_path.stem.strip().lower()
        else:
            suffix = f"_{split_key}"
            subject = csv_path.stem[: -len(suffix)].strip().lower() if csv_path.stem.endswith(suffix) else csv_path.stem.strip().lower()
        if wanted and subject not in wanted:
            continue
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for index, row in enumerate(reader):
                if len(row) < 6:
                    continue
                if limit_per_subject is not None and per_subject.get(subject, 0) >= int(limit_per_subject):
                    break
                question, option_a, option_b, option_c, option_d, correct_letter = [str(value).strip() for value in row[:6]]
                if not question:
                    continue
                options = [option_a, option_b, option_c, option_d]
                letter = correct_letter.upper()
                if letter not in {"A", "B", "C", "D"}:
                    continue
                correct_text = options[ord(letter) - ord("A")]
                per_subject[subject] = per_subject.get(subject, 0) + 1
                yield {
                    "entry_id": f"mmlu_{split_key}_{subject}_{index}",
                    "content": (
                        f"Subject: {subject.replace('_', ' ').title()}\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {letter}. {correct_text}\n"
                        f'Key Fact: The answer to "{question}" is "{correct_text}".'
                    ),
                    "subject": subject,
                    "domain_hint": _subject_to_domain(subject),
                    "source": f"mmlu_{split_key}",
                    "correct_answer": correct_text,
                    "correct_letter": letter,
                    "question": question,
                }


def load_math_entries(data_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Load GSM8K training entries as arithmetic-pattern records."""
    train_path = Path(data_dir) / "grade_school_math" / "data" / "train.jsonl"
    if not train_path.exists():
        return
    with train_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= int(limit):
                break
            raw = json.loads(line)
            question = str(raw.get("question") or "").strip()
            answer = str(raw.get("answer") or "").strip()
            match = _NUMBER_RE.search(answer)
            final_answer = match.group(1).strip() if match else answer.splitlines()[-1].strip()
            yield {
                "entry_id": f"gsm8k_train_{index}",
                "content": (
                    "Subject: Grade School Mathematics\n"
                    f"Problem: {question}\n"
                    f"Step-by-step Solution: {answer}\n"
                    f"Final Answer: {final_answer}\n"
                    "Extract the arithmetic PATTERN and RULES used in this solution."
                ),
                "subject": "arithmetic",
                "domain_hint": "Mathematics",
                "source": "gsm8k_train",
                "correct_answer": final_answer,
                "question": question,
            }


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", str(text or "")).strip()


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = _strip_thinking(raw)
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    for candidate in fenced:
        try:
            parsed = json.loads(candidate.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _fallback_result(entry: dict[str, Any], raw: str, provider: str = "ollama") -> AugmentationResult:
    domain = str(entry.get("domain_hint") or "General")
    summary = str(entry.get("question") or entry.get("subject") or entry.get("entry_id") or "proceduralized entry")
    return AugmentationResult(
        summary=summary,
        entities=[],
        relationships=[],
        domain=domain,
        meaning_rpn_hint=f"{domain.upper()} CONTENT ENTRY",
        taxonomy_refs=[f"concept_{domain.lower()}"] if domain != "General" else [],
        surface_forms={"en": summary},
        confidence=0.2,
        provider=provider,
        raw_response=str(raw or ""),
    )


def _parse_response(raw: str, entry: dict[str, Any], provider: str = "ollama") -> AugmentationResult:
    request = build_request_from_entry(entry)
    bundle, schema_ok, _ = parse_bundle(raw, request)
    if not schema_ok and not bundle.knowledge_packets:
        return _fallback_result(entry, raw, provider=provider)
    return _bundle_primary_result(bundle, entry=entry, provider=provider, raw_response=str(raw or ""))


def proceduralize_entry(
    entry: dict[str, Any],
    ollama: OllamaManager,
    model: str,
    options: dict[str, Any] | None,
    timeout: float,
    *,
    capture_dir: str | Path | None = None,
    model_profile: str = "quality",
) -> AugmentationResult | None:
    """Send one entry through the canonical WINE boundary and return the primary packet."""
    receipt = proceduralize_entry_receipt(
        entry,
        ollama=ollama,
        model=model,
        options=options,
        timeout=timeout,
        capture_dir=capture_dir,
        model_profile=model_profile,
    )
    if receipt.parsed_bundle.ingest_action != "augment":
        return _fallback_result(entry, receipt.failure_code or receipt.status, provider=receipt.provider)
    raw_response = ""
    for raw_path in [part for part in str(receipt.raw_response_path or "").split("|") if part.strip()]:
        path = Path(raw_path.strip())
        if path.exists():
            raw_response += path.read_text(encoding="utf-8")
    return _bundle_primary_result(
        receipt.parsed_bundle,
        entry=entry,
        provider=receipt.provider,
        raw_response=raw_response,
    )


def proceduralize_entry_receipt(
    entry: dict[str, Any],
    *,
    ollama: OllamaManager | None,
    model: str | None,
    options: dict[str, Any] | None,
    timeout: float,
    capture_dir: str | Path | None = None,
    model_profile: str = "quality",
    provider: str = "ollama",
    source_kind: str = "benchmark",
) -> Any:
    content = str(entry.get("content") or entry.get("question") or "").strip()
    context_chunks: list[str] = []
    options_list = entry.get("options")
    if isinstance(options_list, list) and options_list:
        context_chunks.append("Options: " + " | ".join(str(item).strip() for item in options_list if str(item).strip()))
    answer = str(entry.get("correct_answer") or "").strip()
    if answer:
        context_chunks.append(f"Gold answer anchor: {answer[:240]}")
    subject = str(entry.get("subject") or "").strip()
    if subject:
        context_chunks.append(f"Subject anchor: {subject}")
    receipt, _ = proceduralize_text_content(
        content=content,
        source_id=str(entry.get("entry_id") or entry.get("id") or "entry"),
        domain_hint=str(entry.get("domain_hint") or "General"),
        source_path=str(entry.get("source_path") or entry.get("path") or entry.get("source") or ""),
        context_chunks=context_chunks,
        model=model,
        timeout=timeout,
        capture_dir=capture_dir,
        provider=provider,
        model_profile=model_profile,
        options=options,
        ollama=ollama,
        source_kind=source_kind,
    )
    return receipt


def _response_meta_refs(result: AugmentationResult) -> list[str]:
    payload = _extract_json(result.raw_response)
    if not isinstance(payload, dict):
        return []
    refs: list[str] = []
    if isinstance(payload.get("knowledge_packets"), list):
        for packet in list(payload.get("knowledge_packets") or []):
            if not isinstance(packet, dict):
                continue
            refs.extend(str(item).strip() for item in list(packet.get("word_refs") or []) if str(item).strip())
            refs.extend(str(item).strip() for item in list(packet.get("taxonomy_refs") or []) if str(item).strip())
            refs.extend(str(item).strip() for item in list(packet.get("meta_refs") or []) if str(item).strip())
    for star_ref in list(payload.get("star_refs") or []):
        text = str(star_ref).strip()
        if text:
            refs.append(text)
    grammar_rules = payload.get("grammar_rules")
    if isinstance(grammar_rules, list) and grammar_rules:
        refs.append("grammar_rules:" + json.dumps(grammar_rules, ensure_ascii=False, sort_keys=True))
    return list(dict.fromkeys(refs))


def _payload_contract(entry: dict[str, Any], result: AugmentationResult) -> dict[str, Any]:
    domain = str(result.domain or entry.get("domain_hint") or "General").strip().lower()
    if domain == "mathematics":
        return {
            "galaxy": "Math",
            "route_family": "MATH",
            "category": "math_procedural_bridge",
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
            "galaxy": "Grammar",
            "route_family": "GRAMMAR",
            "category": "grammar_procedural_bridge",
            "executor_refs": [
                "grammar_slot_binding_executor",
                "grammar_sequence_executor",
                "grammar_transform_executor",
            ],
            "validator_refs": [
                "grammar_normalization_validator",
                "grammar_answer_validator",
            ],
            "anti_pattern_refs": [
                "anti_pattern_symbol_meaning_drift",
                "anti_pattern_answer_format_mismatch",
            ],
        }
    return {
        "galaxy": "Reality",
        "route_family": "GENERAL",
        "category": "general_procedural_bridge",
        "executor_refs": [
            "general_compare_executor",
            "general_evidence_executor",
        ],
        "validator_refs": [
            "general_consistency_validator",
            "general_answer_validator",
        ],
        "anti_pattern_refs": [
            "anti_pattern_missing_evidence_consistency",
            "anti_pattern_generic_language_factual_winner",
        ],
    }


def result_to_payload_row(result: AugmentationResult, entry: dict[str, Any]) -> dict[str, Any]:
    contract = _payload_contract(entry, result)
    summary = str(result.summary or entry.get("question") or entry.get("entry_id") or "procedural entry").strip()
    route_family = str(contract["route_family"]).strip().upper()
    question = str(entry.get("question") or "").strip()
    star_id = f"{route_family.lower()}_procedural_anchor_{_sha(summary + '|' + question)}"
    star = result_to_star(
        result,
        star_id=star_id,
        meta_refs=[
            f"source:{entry['source']}",
            f"subject:{entry['subject']}",
            *_response_meta_refs(result),
        ],
    )
    route_metadata = {
        "source": str(entry.get("source") or "").strip(),
        "subject": str(entry.get("subject") or "").strip(),
        "question": question,
        "confidence": float(result.confidence),
        "route_family": route_family,
        "selection_role": "executor",
        "layer_id": 3,
        "answer_eligible": False,
        "route_policy": {
            "requires_validator": True,
            "answer_gate": True,
            "branch_topk": 2,
        },
        "executor_refs": list(contract["executor_refs"]),
        "validator_refs": list(contract["validator_refs"]),
        "anti_pattern_refs": list(contract["anti_pattern_refs"]),
    }
    galaxy_entry = star.to_galaxy_entry(
        entry_id=star_id,
        name=summary or "Procedural Meaning Bridge",
        galaxy_name=str(contract["galaxy"]),
        category=str(contract["category"]),
        metadata=route_metadata,
    )
    galaxy_entry.update(
        {
            "route_family": route_family,
            "selection_role": "executor",
            "layer_id": 3,
            "answer_eligible": False,
            "route_policy": dict(route_metadata["route_policy"]),
            "executor_refs": list(contract["executor_refs"]),
            "validator_refs": list(contract["validator_refs"]),
            "anti_pattern_refs": list(contract["anti_pattern_refs"]),
        }
    )
    return {
        "galaxy": str(contract["galaxy"]),
        "entry": galaxy_entry,
    }


def packet_to_star(packet: ProceduralizerPacket, request: ProceduralizerRequest) -> Any:
    result = _packet_to_result(packet, provider="proceduralizer", raw_response="")
    return result_to_star(
        result,
        star_id=packet.star_id or packet.proposed_star_id or None,
        meta_refs=[
            f"source_kind:{request.source_kind}",
            f"source_id:{request.source_id}",
            *list(packet.meta_refs),
        ],
    )


def write_payload_jsonl(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_bundle_jsonl(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def proceduralize_entry_to_payload_rows(
    entry: dict[str, Any],
    *,
    model: str | None = None,
    timeout: float = 120.0,
    capture_dir: str | Path | None = None,
    provider: str = "ollama",
    model_profile: str = "quality",
    options: dict[str, Any] | None = None,
    ollama: OllamaManager | None = None,
    source_kind: str = "benchmark",
) -> tuple[list[dict[str, Any]], Any]:
    receipt = proceduralize_entry_receipt(
        entry,
        ollama=ollama,
        model=model,
        options=options,
        timeout=timeout,
        capture_dir=capture_dir,
        model_profile=model_profile,
        provider=provider,
        source_kind=source_kind,
    )
    request = build_request_from_entry(entry, quality_profile=model_profile, source_kind=source_kind)
    return bundle_to_payload_rows(receipt.parsed_bundle, request), receipt


def proceduralize_text_content(
    *,
    content: str,
    source_id: str,
    domain_hint: str = "General",
    source_path: str = "",
    context_chunks: list[str] | None = None,
    model: str | None = None,
    timeout: float = 120.0,
    capture_dir: str | Path | None = None,
    provider: str = "ollama",
    model_profile: str = "quality",
    options: dict[str, Any] | None = None,
    ollama: OllamaManager | None = None,
    source_kind: str = "text",
) -> tuple[Any, ProceduralizerRequest]:
    chunks = chunk_source_content(content)
    bridge = ProceduralizerWineBridge(
        provider=provider,
        default_timeout=timeout,
        capture_dir=capture_dir,
        ollama=ollama,
    )
    chunk_receipts: list[Any] = []
    first_request: ProceduralizerRequest | None = None
    for index, chunk in enumerate(chunks):
        request = ProceduralizerRequest(
            source_kind=source_kind,
            source_id=f"{source_id}#chunk={index:04d}" if len(chunks) > 1 else source_id,
            source_path=source_path,
            domain_hint=domain_hint,
            content=chunk,
            context_chunks=list(context_chunks or []),
            existing_ref_menu=build_rag_context(domain_hint, "", chunk[:400]),
            quality_profile=model_profile,
            ingest_mode="augment",
        )
        if first_request is None:
            first_request = request
        receipt = bridge.submit(
            request,
            model_profile=model_profile,
            model=model,
            timeout=timeout,
            options=options,
        )
        chunk_receipts.append(receipt)
        if str(receipt.failure_code or "").strip() == "plan_limit_consumed":
            break
    merged = _merge_receipts(
        chunk_receipts,
        provider=str(provider).strip().lower(),
        model=str(model or PROCEDURALIZER_MODEL_PROFILES.get(model_profile, "")),
    )
    return merged, first_request or ProceduralizerRequest(
        source_kind=source_kind,
        source_id=source_id,
        source_path=source_path,
        domain_hint=domain_hint,
        content=content,
        context_chunks=list(context_chunks or []),
        existing_ref_menu=build_rag_context(domain_hint, "", content[:400]),
        quality_profile=model_profile,
        ingest_mode="augment",
    )


def proceduralize_dataset(
    entries: Iterable[dict[str, Any]],
    *,
    model: str | None,
    timeout: float,
    output_path: Path,
    output_format: str = "stars",
    batch_size: int = 50,
    ollama: OllamaManager | None = None,
    options: dict[str, Any] | None = None,
    provider: str = "ollama",
    model_profile: str = "quality",
    capture_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    source_kind: str = "benchmark",
) -> dict[str, Any]:
    """Run batch proceduralization and write stars as JSONL."""
    manager = ollama or OllamaManager(default_timeout=timeout)
    stars = []
    payload_rows = []
    bundles = []
    processed = 0
    created = 0
    receipts: list[dict[str, Any]] = []
    schema_ok = 0
    ingest_actions: dict[str, int] = {"skip": 0, "augment": 0, "needs_context": 0, "reject": 0}
    duplicate_packet_ids: set[str] = set()
    duplicate_packet_count = 0
    stopped_due_to_plan_limit = False
    retry_after_utc = ""
    resolved_output_format = str(output_format).strip().lower()
    for entry in entries:
        processed += 1
        receipt = proceduralize_entry_receipt(
            entry,
            ollama=manager,
            model=model,
            options=options,
            timeout=timeout,
            capture_dir=capture_dir,
            model_profile=model_profile,
            provider=provider,
            source_kind=source_kind,
        )
        receipts.append(receipt.to_dict())
        if receipt.schema_ok:
            schema_ok += 1
        ingest_actions[receipt.parsed_bundle.ingest_action] = ingest_actions.get(receipt.parsed_bundle.ingest_action, 0) + 1
        if str(receipt.failure_code or "").strip() == "plan_limit_consumed":
            stopped_due_to_plan_limit = True
            retry_after_utc = str(receipt.retry_after_utc or "").strip()
        request = build_request_from_entry(entry, quality_profile=model_profile, source_kind=source_kind)
        rows = bundle_to_payload_rows(receipt.parsed_bundle, request)
        if receipt.parsed_bundle.ingest_action == "augment":
            for packet in receipt.parsed_bundle.knowledge_packets:
                packet_id = packet.star_id or packet.proposed_star_id
                if packet_id and packet_id in duplicate_packet_ids:
                    duplicate_packet_count += 1
                if packet_id:
                    duplicate_packet_ids.add(packet_id)
            if resolved_output_format == "payload":
                payload_rows.extend(rows)
            elif resolved_output_format == "bundle":
                bundles.append(
                    {
                        "request": request.to_dict(),
                        "receipt": receipt.to_dict(),
                    }
                )
            else:
                for packet in receipt.parsed_bundle.knowledge_packets:
                    stars.append(packet_to_star(packet, request))
            created += len(receipt.parsed_bundle.knowledge_packets)
        if stopped_due_to_plan_limit:
            break
        if batch_size and processed % int(batch_size) == 0:
            print(f"[knowledge_proceduralizer] processed={processed} created={created}")
    written_path = (
        write_payload_jsonl(payload_rows, output_path)
        if resolved_output_format == "payload"
        else write_bundle_jsonl(bundles, output_path)
        if resolved_output_format == "bundle"
        else write_stars_jsonl(stars, output_path)
    )
    summary = {
        "processed": processed,
        "created": created,
        "output_path": str(written_path),
        "model": str(model or PROCEDURALIZER_MODEL_PROFILES.get(model_profile, "")),
        "provider": provider,
        "model_profile": model_profile,
        "output_format": resolved_output_format,
        "schema_ok": schema_ok,
        "ingest_actions": ingest_actions,
        "duplicate_packet_count": duplicate_packet_count,
        "stopped_due_to_plan_limit": stopped_due_to_plan_limit,
        "retry_after_utc": retry_after_utc,
        "receipts": receipts,
    }
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_model_eval_harness(
    *,
    models: list[str],
    capture_dir: str | Path,
    timeout_seconds: float = 90.0,
    provider: str = "ollama",
    ollama: OllamaManager | None = None,
) -> dict[str, Any]:
    def _packet_metrics(bundle_dict: dict[str, Any]) -> dict[str, int]:
        packets = bundle_dict.get("knowledge_packets", []) if isinstance(bundle_dict, dict) else []
        packet_ids: list[str] = []
        symlink_ref_count = 0
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            packet_id = str(packet.get("star_id") or packet.get("proposed_star_id") or "").strip()
            if packet_id:
                packet_ids.append(packet_id)
            for key in ("symbol_refs", "word_refs", "taxonomy_refs", "grammar_refs", "reality_refs", "meta_refs"):
                value = packet.get(key)
                if isinstance(value, list):
                    symlink_ref_count += len([item for item in value if str(item or "").strip()])
        duplicate_packet_count = max(0, len(packet_ids) - len(set(packet_ids)))
        return {
            "packet_count": len([packet for packet in packets if isinstance(packet, dict)]),
            "symlink_ref_count": symlink_ref_count,
            "duplicate_packet_count": duplicate_packet_count,
        }

    def _build_eval_summary(
        *,
        summary_rows: list[dict[str, Any]],
        full_results: list[dict[str, Any]],
        stopped_due_to_plan_limit: bool,
        retry_after_utc: str,
    ) -> dict[str, Any]:
        return {
            "provider": provider,
            "models": summary_rows,
            "recommended_default": PROCEDURALIZER_MODEL_PROFILES["quality"],
            "stopped_due_to_plan_limit": bool(stopped_due_to_plan_limit),
            "retry_after_utc": retry_after_utc,
            "evaluated_models": len(full_results),
        }

    def _write_eval_artifacts(
        *,
        summary_rows: list[dict[str, Any]],
        full_results: list[dict[str, Any]],
        stopped_due_to_plan_limit: bool,
        retry_after_utc: str,
    ) -> dict[str, Any]:
        summary = _build_eval_summary(
            summary_rows=summary_rows,
            full_results=full_results,
            stopped_due_to_plan_limit=stopped_due_to_plan_limit,
            retry_after_utc=retry_after_utc,
        )
        (capture_root / "summary.execution.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (capture_root / "full_results.execution.json").write_text(json.dumps(full_results, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    capture_root = Path(capture_dir)
    capture_root.mkdir(parents=True, exist_ok=True)
    prompts = [
        {
            "id": "math_word_problem",
            "domain_hint": "Mathematics",
            "content": (
                "Problem: Maria buys 3 notebooks for $4 each and 2 pens for $1 each. "
                "How much does she spend in total? Final answer: 14."
            ),
        },
        {
            "id": "definition_fact",
            "domain_hint": "Language",
            "content": (
                "Definition: Mammals are warm-blooded vertebrates that feed milk to their young."
            ),
        },
        {
            "id": "bibliography_page",
            "domain_hint": "General",
            "content": (
                "References\n[1] Smith 2024 ... [2] Jones 2025 ... This page only lists references."
            ),
        },
    ]
    manager = ollama or OllamaManager(default_timeout=timeout_seconds)
    bridge = ProceduralizerWineBridge(
        provider=provider,
        default_timeout=timeout_seconds,
        capture_dir=capture_root / "captures",
        ollama=manager,
    )
    full_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stopped_due_to_plan_limit = False
    retry_after_utc = ""
    for model in models:
        model_results: list[dict[str, Any]] = []
        model_options = dict(MODEL_OPTIONS.get(str(model).strip(), {}))
        for prompt in prompts:
            request = ProceduralizerRequest(
                source_kind="text",
                source_id=str(prompt["id"]),
                source_path=f"eval://{prompt['id']}",
                domain_hint=str(prompt["domain_hint"]),
                content=str(prompt["content"]),
                context_chunks=[],
                existing_ref_menu=build_rag_context(str(prompt["domain_hint"]), "", str(prompt["content"])),
                quality_profile="quality",
                ingest_mode="augment",
            )
            receipt = bridge.submit(
                request,
                model=model,
                timeout=timeout_seconds,
                options=model_options,
            )
            parsed_bundle = receipt.parsed_bundle.to_dict()
            metrics = _packet_metrics(parsed_bundle)
            model_results.append(
                {
                    "prompt_id": prompt["id"],
                    "options": model_options,
                    "receipt": receipt.to_dict(),
                    "bundle": parsed_bundle,
                    "metrics": metrics,
                }
            )
            full_results_for_write = [*full_results, {"model": model, "results": model_results}]
            if str(receipt.failure_code or "").strip() == "plan_limit_consumed":
                stopped_due_to_plan_limit = True
                retry_after_utc = str(receipt.retry_after_utc or "").strip()
            _write_eval_artifacts(
                summary_rows=summary_rows,
                full_results=full_results_for_write,
                stopped_due_to_plan_limit=stopped_due_to_plan_limit,
                retry_after_utc=retry_after_utc,
            )
            if stopped_due_to_plan_limit:
                break
        schema_valid = sum(1 for row in model_results if bool(row["receipt"]["schema_ok"]))
        avg_latency_ms = int(sum(int(row["receipt"]["latency_ms"]) for row in model_results) / max(1, len(model_results)))
        summary_rows.append(
            {
                "model": model,
                "prompt_count": len(model_results),
                "schema_valid_count": schema_valid,
                "avg_latency_ms": avg_latency_ms,
                "augment_count": sum(1 for row in model_results if row["bundle"]["ingest_action"] == "augment"),
                "skip_count": sum(1 for row in model_results if row["bundle"]["ingest_action"] == "skip"),
                "needs_context_count": sum(1 for row in model_results if row["bundle"]["ingest_action"] == "needs_context"),
                "symlink_ref_count": sum(int(row["metrics"]["symlink_ref_count"]) for row in model_results),
                "duplicate_packet_count": sum(int(row["metrics"]["duplicate_packet_count"]) for row in model_results),
            }
        )
        full_results.append({"model": model, "results": model_results})
        _write_eval_artifacts(
            summary_rows=summary_rows,
            full_results=full_results,
            stopped_due_to_plan_limit=stopped_due_to_plan_limit,
            retry_after_utc=retry_after_utc,
        )
        if stopped_due_to_plan_limit:
            break
    summary = _write_eval_artifacts(
        summary_rows=summary_rows,
        full_results=full_results,
        stopped_due_to_plan_limit=stopped_due_to_plan_limit,
        retry_after_utc=retry_after_utc,
    )
    return summary


def _iter_limited(entries: Iterable[dict[str, Any]], count: int | None) -> Iterator[dict[str, Any]]:
    for index, entry in enumerate(entries):
        if count is not None and index >= int(count):
            break
        yield entry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, choices=["mmlu_val", "mmlu_train", "gsm8k_train"])
    parser.add_argument("--source-kind", default="benchmark", choices=["benchmark", "pdf", "manifest", "text"])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--subjects", default="", help="Comma-separated MMLU subjects")
    parser.add_argument("--limit-per-subject", type=int, default=None)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model-profile", default="quality")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/K3D/Knowledge3D.local/galaxies/proceduralized_stars.jsonl"),
    )
    parser.add_argument("--output-format", choices=["stars", "payload", "bundle"], default="stars")
    parser.add_argument("--emit", choices=["stars", "payload", "bundle"], default=None)
    parser.add_argument("--capture-dir", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--content", default="", help="Inline text content when --source-kind text")
    parser.add_argument("--content-file", type=Path, default=None, help="Text file when --source-kind text")
    parser.add_argument("--source-id", default="text_entry")
    parser.add_argument("--source-path", default="")
    parser.add_argument("--domain-hint", default="General")
    parser.add_argument("--mmlu-data", type=Path, default=MMLU_DEFAULT_PATH)
    parser.add_argument("--gsm8k-data", type=Path, default=GSM8K_DEFAULT_PATH)
    parser.add_argument("--eval-models", default="", help="Comma-separated explicit models for bounded proceduralizer eval")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    timeout = float(args.timeout_seconds if args.timeout_seconds is not None else args.timeout)
    if str(args.eval_models).strip():
        models = [item.strip() for item in str(args.eval_models).split(",") if item.strip()]
        summary = run_model_eval_harness(
            models=models,
            capture_dir=args.capture_dir or args.output.parent / "proceduralizer_model_eval",
            timeout_seconds=timeout,
            provider=str(args.provider).strip().lower(),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if str(args.source_kind).strip().lower() == "text":
        if args.content_file is not None:
            content = args.content_file.read_text(encoding="utf-8", errors="ignore")
        else:
            content = str(args.content or "")
        receipt, request = proceduralize_text_content(
            content=content,
            source_id=str(args.source_id),
            domain_hint=str(args.domain_hint),
            source_path=str(args.source_path),
            model=str(args.model).strip() or None,
            timeout=timeout,
            capture_dir=args.capture_dir,
            provider=str(args.provider).strip().lower(),
            model_profile=str(args.model_profile).strip().lower(),
            options=dict(MODEL_OPTIONS.get(str(args.model or PROCEDURALIZER_MODEL_PROFILES.get(str(args.model_profile).strip().lower(), PROCEDURALIZER_MODEL_PROFILES["quality"])), {})),
            source_kind="text",
        )
        output_format = str(args.emit or args.output_format).strip().lower()
        if output_format == "payload":
            rows = bundle_to_payload_rows(receipt.parsed_bundle, request)
            written = write_payload_jsonl(rows, args.output)
        elif output_format == "bundle":
            written = write_bundle_jsonl([{"request": request.to_dict(), "receipt": receipt.to_dict()}], args.output)
        else:
            stars = [packet_to_star(packet, request) for packet in receipt.parsed_bundle.knowledge_packets]
            written = write_stars_jsonl(stars, args.output)
        summary = {
            "processed": 1,
            "created": len(receipt.parsed_bundle.knowledge_packets),
            "output_path": str(written),
            "provider": args.provider,
            "model_profile": args.model_profile,
            "model": args.model or PROCEDURALIZER_MODEL_PROFILES.get(str(args.model_profile).strip().lower(), PROCEDURALIZER_MODEL_PROFILES["quality"]),
            "schema_ok": receipt.schema_ok,
            "ingest_action": receipt.parsed_bundle.ingest_action,
        }
        if args.summary_path is not None:
            args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    source = str(args.source or "gsm8k_train").strip().lower()
    if source.startswith("mmlu_"):
        split = "auxiliary_train" if source == "mmlu_train" else "val"
        subject_list = [item.strip() for item in str(args.subjects).split(",") if item.strip()]
        entries = load_mmlu_entries(
            args.mmlu_data,
            split=split,
            subjects=subject_list or None,
            limit_per_subject=args.limit_per_subject,
        )
    else:
        entries = load_math_entries(args.gsm8k_data, limit=args.count)
    resolved_model = str(args.model or SOURCE_MODEL_MAP[source]).strip()
    options = dict(MODEL_OPTIONS.get(resolved_model, {}))
    summary = proceduralize_dataset(
        _iter_limited(entries, args.count),
        model=resolved_model,
        timeout=timeout,
        output_path=args.output,
        output_format=str(args.emit or args.output_format).strip().lower(),
        options=options,
        provider=str(args.provider).strip().lower(),
        model_profile=str(args.model_profile).strip().lower(),
        capture_dir=args.capture_dir,
        summary_path=args.summary_path,
        source_kind="benchmark",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "GSM8K_DEFAULT_PATH",
    "MMLU_DEFAULT_PATH",
    "MODEL_OPTIONS",
    "PROCEDURALIZATION_SYSTEM_PROMPT",
    "SOURCE_MODEL_MAP",
    "_extract_json",
    "_hits",
    "_parse_response",
    "_payload_contract",
    "_subject_to_domain",
    "build_rag_context",
    "build_request_from_entry",
    "load_math_entries",
    "load_mmlu_entries",
    "packet_to_star",
    "proceduralize_entry_receipt",
    "proceduralize_entry_to_payload_rows",
    "proceduralize_dataset",
    "proceduralize_entry",
    "proceduralize_text_content",
    "result_to_payload_row",
    "run_model_eval_harness",
    "write_bundle_jsonl",
    "write_payload_jsonl",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
