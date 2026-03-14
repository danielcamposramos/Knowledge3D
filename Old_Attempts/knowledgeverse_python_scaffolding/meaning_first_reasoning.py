"""Meaning-first helpers for benchmark-facing reasoning paths.

This module extracts structured meaning atoms from Galaxy evidence rows.
It is intentionally generic: the same atom shape can be reused by LHE, Math,
ARC metadata bridges, or other validation surfaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MeaningAtom:
    atom_id: str
    concept_ref: str
    canonical_name: str
    domain: str
    subject: str
    subfield: str
    source_pass: str
    confidence: float
    forms: tuple[str, ...]
    related_refs: tuple[str, ...]
    symlinks: tuple[str, ...]
    semantics: str
    summary: str


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_keep_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return tuple(out)


def _tokenize_semantic(text: str) -> tuple[str, ...]:
    tokens = [tok for tok in re.split(r"[^a-z0-9_]+", str(text or "").lower()) if tok]
    return _dedupe_keep_order(tokens)


def _evidence_field_fragments(evidence_fields: dict[str, Any], entry: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, ...]:
    prioritized = [
        _coerce_text(evidence_fields.get("content")),
        _coerce_text(entry.get("content")),
        _coerce_text(evidence_fields.get("summary")),
        _coerce_text(entry.get("summary")),
        _coerce_text(evidence_fields.get("description")),
        _coerce_text(entry.get("description")),
        _coerce_text(evidence_fields.get("semantics")),
        _coerce_text(metadata.get("semantics")),
        _coerce_text(evidence_fields.get("usage_conditions")),
        _coerce_text(entry.get("rpn_program")),
        _coerce_text(evidence_fields.get("rpn_program")),
    ]
    return _dedupe_keep_order([value for value in prioritized if value and len(value) <= 240])


def meaning_atoms_from_row(
    row: dict[str, Any],
    *,
    source_pass: str = "evidence",
    rank_weight: float = 0.0,
    evidence_fields: dict[str, Any] | None = None,
) -> list[MeaningAtom]:
    entry = row.get("entry") if isinstance(row.get("entry"), dict) else {}
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    fields = evidence_fields if isinstance(evidence_fields, dict) else {}
    if not entry and not metadata:
        return []

    concept_ref = str(
        metadata.get("meaning_ref")
        or metadata.get("formalizes_ref")
        or metadata.get("reasons_about_ref")
        or entry.get("id")
        or ""
    ).strip()
    canonical_name = str(entry.get("name") or entry.get("title") or metadata.get("meaning_ref") or "").strip()
    if not concept_ref and not canonical_name:
        return []

    aliases = _coerce_list(metadata.get("aliases"))
    keywords = _coerce_list(metadata.get("keywords"))
    field_fragments = _evidence_field_fragments(fields, entry, metadata)
    forms = _dedupe_keep_order([canonical_name, *aliases, *keywords, *field_fragments])

    related = _dedupe_keep_order(
        [
            *[str(item.get("to", "")).strip() for item in metadata.get("relationships", []) if isinstance(item, dict)],
            *_coerce_list(metadata.get("related_concepts")),
            str(metadata.get("formalizes_ref") or "").strip(),
            str(metadata.get("reasons_about_ref") or "").strip(),
        ]
    )
    symlinks = _dedupe_keep_order(
        [
            *_coerce_list(metadata.get("symlinks")),
            *_coerce_list(metadata.get("word_refs")),
            str(metadata.get("formalizes_ref") or "").strip(),
            str(metadata.get("reasons_about_ref") or "").strip(),
        ]
    )

    semantics = _coerce_text(
        metadata.get("semantics")
        or fields.get("semantics")
        or fields.get("usage_conditions")
        or entry.get("summary")
    )
    summary = _coerce_text(
        fields.get("content")
        or entry.get("content")
        or fields.get("summary")
        or entry.get("summary")
        or fields.get("description")
        or entry.get("description")
    )

    subject = str(metadata.get("subject") or "").strip().lower()
    subfield = str(metadata.get("subfield") or "").strip().lower()
    domain = str(metadata.get("domain") or entry.get("domain") or "").strip().lower()
    confidence = float(metadata.get("confidence") or (0.55 + min(0.35, rank_weight * 0.25)))

    atom = MeaningAtom(
        atom_id=f"{source_pass}:{concept_ref or canonical_name.lower().replace(' ', '_')}",
        concept_ref=concept_ref or canonical_name.lower().replace(" ", "_"),
        canonical_name=canonical_name or concept_ref,
        domain=domain,
        subject=subject,
        subfield=subfield,
        source_pass=source_pass,
        confidence=confidence,
        forms=forms,
        related_refs=related,
        symlinks=symlinks,
        semantics=semantics,
        summary=summary,
    )
    return [atom]


def meaning_atoms_from_evidence_rows(evidence_rows: list[dict[str, Any]]) -> list[MeaningAtom]:
    atoms: list[MeaningAtom] = []
    for item in evidence_rows:
        row = item.get("row", {}) if isinstance(item.get("row"), dict) else {}
        rank_weight = float(item.get("rank_weight", 0.0))
        fields = item.get("fields", {}) if isinstance(item.get("fields"), dict) else {}
        atoms.extend(
            meaning_atoms_from_row(
                row,
                source_pass="evidence",
                rank_weight=rank_weight,
                evidence_fields=fields,
            )
        )
    return fuse_meaning_atoms(atoms)


def meaning_atoms_from_parse_entities(entities: list[dict[str, Any]]) -> list[MeaningAtom]:
    atoms: list[MeaningAtom] = []
    for index, entity in enumerate(entities):
        raw = str(entity.get("raw") or entity.get("value") or "").strip()
        value = str(entity.get("value") or "").strip()
        if not raw and not value:
            continue
        forms = _dedupe_keep_order([value, raw, *_tokenize_semantic(raw)])
        atom = MeaningAtom(
            atom_id=f"parse:{entity.get('source_pass', 'fusion')}:{index}",
            concept_ref=str(entity.get("meaning_ref") or entity.get("value") or f"parse_entity_{index}").strip(),
            canonical_name=value or raw,
            domain=str(entity.get("domain") or "").strip().lower(),
            subject=str(entity.get("subject") or "").strip().lower(),
            subfield=str(entity.get("subfield") or "").strip().lower(),
            source_pass=str(entity.get("source_pass") or "fusion"),
            confidence=float(entity.get("confidence") or 0.5),
            forms=forms,
            related_refs=(),
            symlinks=(),
            semantics=str(entity.get("role") or "").strip(),
            summary=raw,
        )
        atoms.append(atom)
    return fuse_meaning_atoms(atoms)


def fuse_meaning_atoms(atoms: list[MeaningAtom]) -> list[MeaningAtom]:
    merged: dict[str, MeaningAtom] = {}
    for atom in atoms:
        key = atom.concept_ref or atom.canonical_name.lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = atom
            continue
        merged[key] = MeaningAtom(
            atom_id=existing.atom_id,
            concept_ref=existing.concept_ref,
            canonical_name=existing.canonical_name if len(existing.canonical_name) >= len(atom.canonical_name) else atom.canonical_name,
            domain=existing.domain or atom.domain,
            subject=existing.subject or atom.subject,
            subfield=existing.subfield or atom.subfield,
            source_pass=existing.source_pass,
            confidence=max(existing.confidence, atom.confidence),
            forms=_dedupe_keep_order([*existing.forms, *atom.forms]),
            related_refs=_dedupe_keep_order([*existing.related_refs, *atom.related_refs]),
            symlinks=_dedupe_keep_order([*existing.symlinks, *atom.symlinks]),
            semantics=existing.semantics or atom.semantics,
            summary=existing.summary if len(existing.summary) >= len(atom.summary) else atom.summary,
        )
    return sorted(merged.values(), key=lambda item: (item.subject, item.subfield, item.canonical_name.lower(), item.concept_ref))
