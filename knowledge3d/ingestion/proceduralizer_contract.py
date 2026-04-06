"""Canonical ingestion-time contract for the knowledge proceduralizer.

This module defines the stable request/receipt/bundle contract used by the
proceduralizer before deterministic Stargate-side normalization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any


PROCEDURALIZER_MODEL_PROFILES: dict[str, str] = {
    "quality": "glm-5:cloud",
    "audit_reasoning": "kimi-k2-thinking:cloud",
    "long_context_engineering": "qwen3.5:397b-cloud",
    "balanced_fallback": "deepseek-v3.2:cloud",
}


PROCEDURALIZER_SYSTEM_PROMPT = """You are the K3D Knowledge Proceduralizer.

You operate only at ingestion time inside the Knowledgeverse Ingestion Stargate
(Region 7). You do NOT solve runtime tasks. You transmute source content into
canonical K3D knowledge packets.

K3D invariants:
- Follow the 4-layer order: Form -> Meaning -> Rules -> Meta-Rules
- Prefer references to existing canonical stars over new content
- Never duplicate lower-layer symbols, meanings, or facts when references exist
- Never use benchmark, dataset, source-file, or chunk names in star ids
- If the content is not knowledge, emit ingest_action=\"skip\"
- If the content is incomplete and needs more context, emit ingest_action=\"needs_context\"
- If the content is unsafe or unusable, emit ingest_action=\"reject\"
- Output strict JSON only. No prose. No markdown. No chain-of-thought.

Return this schema exactly:
{
  "ingest_action": "skip|augment|needs_context|reject",
  "knowledge_packets": [
    {
      "layer_kind": "form|meaning|rule|meta_rule",
      "star_id": "optional canonical id",
      "proposed_star_id": "optional new meaning-based id",
      "meaning_class": "fact|definition|pattern|rule|formula|symbol|taxonomy|bridge",
      "meaning_rpn": "compact English RPN",
      "summary": "one line summary",
      "domain": "Mathematics|Physics|Biology|Language|Tools|Visual|Audio|General",
      "surface_forms": {"en": "english label"},
      "symbol_refs": ["char_u0031"],
      "word_refs": ["synset_14845743_n"],
      "taxonomy_refs": ["concept_mathematics"],
      "grammar_refs": ["grammar_quantity_unit_binding"],
      "reality_refs": ["unit_money_dollar"],
      "meta_refs": ["source_span:0-120"],
      "relationships": [{"from": "x", "relation": "part_of", "to": "y"}],
      "route_contract": {
        "route_family": "MATH|QUESTION|GENERAL|GRAMMAR|GAME_2D",
        "selection_role": "executor|router|validator",
        "layer_id": 3,
        "answer_eligible": false,
        "route_policy": {"requires_validator": true, "answer_gate": false, "branch_topk": 2},
        "executor_refs": [],
        "validator_refs": [],
        "anti_pattern_refs": []
      },
      "confidence": 0.0,
      "needs_review": false
    }
  ]
}

Rules:
- knowledge_packets may be empty only for skip, needs_context, or reject
- route_contract is present only for truly route-capable Layer 3/4 packets
- Preserve evidence with compact source_span metadata, not quotes
- Mint proposed ids only when the concept is genuinely missing
- Every id and label must be meaning-named, never benchmark-named
"""


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_BENCHMARK_TOKEN_RE = re.compile(r"\b(mmlu|gsm8k|lhe|arc|imo|aime|amc|omni|benchmark)\b", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _sha(text: str, *, size: int = 12) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:size]


def slugify_meaning_name(text: str, *, fallback: str = "entry") -> str:
    slug = _NON_ALNUM_RE.sub("_", str(text or "").strip().lower()).strip("_")
    return slug or fallback


def request_hash(payload: dict[str, Any]) -> str:
    return _sha(json.dumps(payload, ensure_ascii=False, sort_keys=True), size=16)


def response_hash(text: str) -> str:
    return _sha(_THINK_RE.sub("", str(text or "")).strip(), size=16)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", str(text or "")).strip()


def extract_json_object(raw: str) -> dict[str, Any] | None:
    text = strip_thinking(raw)
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


@dataclass
class ProceduralizerRequest:
    source_kind: str
    source_id: str
    source_path: str
    domain_hint: str
    content: str
    context_chunks: list[str] = field(default_factory=list)
    existing_ref_menu: str = ""
    quality_profile: str = "quality"
    ingest_mode: str = "augment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProceduralizerPacket:
    layer_kind: str
    star_id: str = ""
    proposed_star_id: str = ""
    meaning_class: str = "entry"
    meaning_rpn: str = ""
    summary: str = ""
    domain: str = "General"
    surface_forms: dict[str, str] = field(default_factory=dict)
    symbol_refs: list[str] = field(default_factory=list)
    word_refs: list[str] = field(default_factory=list)
    taxonomy_refs: list[str] = field(default_factory=list)
    grammar_refs: list[str] = field(default_factory=list)
    reality_refs: list[str] = field(default_factory=list)
    meta_refs: list[str] = field(default_factory=list)
    relationships: list[dict[str, str]] = field(default_factory=list)
    route_contract: dict[str, Any] | None = None
    confidence: float = 0.0
    needs_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProceduralizerBundle:
    ingest_action: str
    knowledge_packets: list[ProceduralizerPacket] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingest_action": self.ingest_action,
            "knowledge_packets": [packet.to_dict() for packet in self.knowledge_packets],
        }


@dataclass
class ProceduralizerReceipt:
    status: str
    provider: str
    model: str
    latency_ms: int
    request_hash: str
    response_hash: str
    raw_response_path: str
    schema_ok: bool
    failure_code: str
    parsed_bundle: ProceduralizerBundle
    retry_after_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "raw_response_path": self.raw_response_path,
            "schema_ok": self.schema_ok,
            "failure_code": self.failure_code,
            "retry_after_utc": self.retry_after_utc,
            "parsed_bundle": self.parsed_bundle.to_dict(),
        }


def _normalize_id(value: Any, *, summary: str, layer_kind: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _BENCHMARK_TOKEN_RE.search(text):
        return ""
    slug = slugify_meaning_name(text, fallback=f"{layer_kind}_entry")
    return slug


def _normalize_ref_list(value: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(value, list):
        return out
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _normalize_relationships(value: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "from": str(item.get("from") or "").strip(),
                "relation": str(item.get("relation") or "").strip(),
                "to": str(item.get("to") or "").strip(),
            }
        )
    return out


def _normalize_route_contract(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    route_family = str(value.get("route_family") or "").strip().upper()
    selection_role = str(value.get("selection_role") or "").strip().lower()
    route_policy = value.get("route_policy")
    if not route_family or not selection_role or not isinstance(route_policy, dict):
        return None
    return {
        "route_family": route_family,
        "selection_role": selection_role,
        "layer_id": int(value.get("layer_id", 3)),
        "answer_eligible": bool(value.get("answer_eligible", False)),
        "route_policy": dict(route_policy),
        "executor_refs": _normalize_ref_list(value.get("executor_refs")),
        "validator_refs": _normalize_ref_list(value.get("validator_refs")),
        "anti_pattern_refs": _normalize_ref_list(value.get("anti_pattern_refs")),
    }


def _fallback_bundle(
    request: ProceduralizerRequest,
    *,
    failure_code: str,
    raw_text: str = "",
) -> ProceduralizerBundle:
    if failure_code == "non_knowledge":
        action = "skip"
    elif failure_code == "needs_context":
        action = "needs_context"
    elif failure_code == "reject":
        action = "reject"
    else:
        action = "augment"
    if action != "augment":
        return ProceduralizerBundle(ingest_action=action, knowledge_packets=[])
    summary = request.content.strip().splitlines()[0] if request.content.strip() else request.source_id
    packet = ProceduralizerPacket(
        layer_kind="meaning",
        proposed_star_id=f"meaning_anchor_{_sha(request.source_id + '|' + summary)}",
        meaning_class="entry",
        meaning_rpn=f"{str(request.domain_hint or 'General').upper()} CONTENT ENTRY",
        summary=summary[:240],
        domain=str(request.domain_hint or "General").strip() or "General",
        surface_forms={"en": summary[:240] or request.source_id},
        taxonomy_refs=[f"concept_{slugify_meaning_name(request.domain_hint, fallback='general')}"],
        meta_refs=[f"failure_code:{failure_code}", f"raw_excerpt:{strip_thinking(raw_text)[:96]}"] if raw_text else [f"failure_code:{failure_code}"],
        confidence=0.2,
        needs_review=True,
    )
    return ProceduralizerBundle(ingest_action="augment", knowledge_packets=[packet])


def parse_bundle(raw_text: str, request: ProceduralizerRequest) -> tuple[ProceduralizerBundle, bool, str]:
    payload = extract_json_object(raw_text)
    if not isinstance(payload, dict):
        return _fallback_bundle(request, failure_code="invalid_json", raw_text=raw_text), False, "invalid_json"

    action = str(payload.get("ingest_action") or "").strip().lower()
    if action not in {"skip", "augment", "needs_context", "reject"}:
        # Compat: accept old single-packet proceduralizer payloads as augment.
        action = "augment"

    packets_raw = payload.get("knowledge_packets")
    if packets_raw is None and any(key in payload for key in ("meaning_class", "meaning_rpn", "summary", "domain")):
        packets_raw = [payload]
    if not isinstance(packets_raw, list):
        packets_raw = []

    packets: list[ProceduralizerPacket] = []
    for item in packets_raw:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        layer_kind = str(item.get("layer_kind") or "meaning").strip().lower() or "meaning"
        packet = ProceduralizerPacket(
            layer_kind=layer_kind,
            star_id=_normalize_id(item.get("star_id"), summary=summary, layer_kind=layer_kind),
            proposed_star_id=_normalize_id(item.get("proposed_star_id"), summary=summary, layer_kind=layer_kind),
            meaning_class=str(item.get("meaning_class") or "entry").strip() or "entry",
            meaning_rpn=str(item.get("meaning_rpn") or f"{str(request.domain_hint or 'General').upper()} CONTENT ENTRY").strip(),
            summary=summary or request.source_id,
            domain=str(item.get("domain") or request.domain_hint or "General").strip() or "General",
            surface_forms={
                str(language).strip().lower(): str(text).strip()
                for language, text in dict(item.get("surface_forms") or {}).items()
                if str(language).strip() and str(text).strip()
            } or {"en": summary or request.source_id},
            symbol_refs=_normalize_ref_list(item.get("symbol_refs")),
            word_refs=_normalize_ref_list(item.get("word_refs") or item.get("star_refs")),
            taxonomy_refs=_normalize_ref_list(item.get("taxonomy_refs")),
            grammar_refs=_normalize_ref_list(item.get("grammar_refs")),
            reality_refs=_normalize_ref_list(item.get("reality_refs")),
            meta_refs=_normalize_ref_list(item.get("meta_refs")),
            relationships=_normalize_relationships(item.get("relationships")),
            route_contract=_normalize_route_contract(item.get("route_contract")),
            confidence=max(0.0, min(1.0, float(item.get("confidence", 0.35) or 0.35))),
            needs_review=bool(item.get("needs_review", False)),
        )
        if not packet.taxonomy_refs and packet.domain.lower() != "general":
            packet.taxonomy_refs = [f"concept_{slugify_meaning_name(packet.domain, fallback='general')}"]
        if not packet.proposed_star_id and not packet.star_id:
            packet.proposed_star_id = f"{layer_kind}_{slugify_meaning_name(packet.summary)}_{_sha(packet.summary + '|' + request.source_id)}"
        packets.append(packet)

    bundle = ProceduralizerBundle(ingest_action=action, knowledge_packets=packets)
    schema_ok = action in {"skip", "augment", "needs_context", "reject"}
    if action == "augment" and not packets:
        return _fallback_bundle(request, failure_code="empty_packets", raw_text=raw_text), False, "empty_packets"
    return bundle, schema_ok, "" if schema_ok else "invalid_schema"


__all__ = [
    "PROCEDURALIZER_MODEL_PROFILES",
    "PROCEDURALIZER_SYSTEM_PROMPT",
    "ProceduralizerBundle",
    "ProceduralizerPacket",
    "ProceduralizerReceipt",
    "ProceduralizerRequest",
    "extract_json_object",
    "parse_bundle",
    "request_hash",
    "response_hash",
    "slugify_meaning_name",
    "strip_thinking",
]
