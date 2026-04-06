# Procedural Memory Knowledge Representation Technology (PM-KR)

**Version**: 1.0  
**Status**: Candidate Specification (K3D Canonical Vocabulary)  
**License**: CC-BY-4.0 (Documentation), Apache 2.0 (Implementation)  
**Date**: February 19, 2026

---

## Abstract

This specification defines **Procedural Memory KR (PM-KR)** for Knowledge3D: a procedural-first, compression-aware knowledge representation model where knowledge is stored as executable programs plus symlinked references from **form to meaning**.

PM-KR defines:

- Canonical procedural storage (RPN/PTX-aligned) as source of truth.
- Layered composition (Form -> Meaning -> Rules -> Meta-Rules).
- Symlink-style references to avoid duplication (Save Information Principle).
- Dual-client consistency (human and synthetic users consume the same node truth).
- Sovereign boundary rules (PTX-only hot path; ingestion/migration outside hot path).

This document formalizes PM-KR as a candidate computer science technology within K3D and a base for W3C-facing output documents under `docs/W3C/`.

---

## 1. Scope

This specification applies to:

- Galaxy/House procedural memory nodes.
- Ingestion outputs that materialize into procedural memory.
- Specialist-readable knowledge assets used by sovereign hot-path inference.
- Compression and reference-preservation behavior for multi-layer knowledge.

This specification does **not** define:

- UI/visual design details (see spatial UI specs).
- Specific benchmark policies.
- External standards governance process (handled by W3C-facing documents).

---

## 2. Normative References

- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`
- `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md`
- `docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md`
- `docs/vocabulary/ADAPTIVE_PROCEDURAL_COMPRESSION_SPECIFICATION.md`
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md`

---

## 3. Terminology

- **Procedural Memory**: Knowledge stored as executable procedures (for example RPN programs), not static duplicated payloads.
- **Canonical Source**: The single authoritative procedural representation of a symbol/object/rule.
- **Symlink Reference**: Lightweight reference from one node to an already canonical node (for example `char_refs`, `word_refs`, `symbol_refs`).
- **Form**: Procedural representation of structure/appearance.
- **Meaning**: Semantic behavior, interpretation, or executable transformation semantics.
- **Hot Path**: Runtime inference path that must remain sovereign and PTX-only.
- **Ingestion Path**: Offline or mid-term path that constructs procedural memory from external data.

The keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

---

## 4. PM-KR Core Model

### 4.1 Layer Stack

PM-KR uses four compositional layers:

1. **Form Layer**: canonical procedural primitives (glyphs, shapes, base symbols).
2. **Meaning Layer**: semantic composition over form references.
3. **Rules Layer**: executable transformation programs referencing lower layers.
4. **Meta-Rules Layer**: strategy/control over rule selection and consolidation.

Higher layers MUST reference lower canonical layers rather than duplicate them.

### 4.2 Form -> Meaning Contract

- A node MAY carry both form and meaning procedures.
- If meaning is derived from form, derivation metadata SHOULD be explicit.
- If a lower-layer canonical node exists, higher layers MUST use references instead of payload copies.

### 4.3 Symlink Compression Contract

- References MUST be stable identifiers (or equivalent canonical keys).
- Duplicate payload storage across layers MUST NOT be used when reference resolution is possible.
- Reference graphs SHOULD be acyclic by design for deterministic reconstruction.

---

## 5. Normative Invariants

### 5.1 Canonicality Invariant

For each canonical concept/program, there MUST be one authoritative procedural source per versioned namespace.

### 5.2 Reference Preservation Invariant

If node B reuses canonical content from node A, B MUST reference A; B MUST NOT inline duplicate canonical payload unless explicitly marked as materialized cache.

### 5.3 Deterministic Reconstruction Invariant

Given canonical procedures + references + versioned metadata, reconstruction MUST be deterministic for a fixed runtime/kernel version.

### 5.4 Dual-Client Equivalence Invariant

Human and synthetic users MUST observe/consume the same underlying node truth, differing only in representation modality.

### 5.5 Sovereign Boundary Invariant

- Hot path MUST execute through sovereign PTX/RPN components.
- Ingestion/augmentation MAY use external tooling, but outputs MUST crystallize into PM-KR-compliant procedural memory before runtime use.

### 5.6 Auditability Invariant

Node provenance and transformation history SHOULD be recorded such that procedural origins and reference lineage are inspectable.

---

## 6. Minimal Node Contract (PM-KR Compliant)

A PM-KR-compliant node SHOULD expose:

- `id`: stable identifier.
- `layer`: one of `{form, meaning, rules, meta_rules}`.
- `form_program`: procedural source (where applicable).
- `meaning_program`: semantic/behavioral procedure (where applicable).
- `refs`: reference fields (`char_refs`, `word_refs`, `symbol_refs`, `rule_refs`, `component_refs`) as applicable.
- `metadata`: provenance, domain, version, confidence, timestamp.

Implementations MAY add fields, but MUST preserve these semantics.

---

## 7. Conformance Levels

### Level A: PM-KR Core

- Implements layer model.
- Enforces canonicality and reference preservation.
- Supports deterministic reconstruction.

### Level B: PM-KR Sovereign Runtime

Includes Level A plus:

- PTX/RPN hot path sovereignty enforcement.
- Explicit fail-fast behavior for unavailable sovereign backends.

### Level C: PM-KR Auditable Production

Includes Level B plus:

- Provenance/audit lineage.
- Compression/reference metrics reporting.
- Conformance test artifacts.

---

## 8. Validation Requirements

A PM-KR implementation SHOULD provide tests for:

1. Canonical node deduplication behavior.
2. Reference resolution correctness.
3. Deterministic reconstruction for fixed seeds/runtime.
4. Hot-path sovereignty checks (no hidden fallback).
5. Dual-client consistency for shared node IDs.

---

## 9. Relationship to Existing K3D Specifications

PM-KR consolidates and defines the procedural-memory aspects already present across K3D specs:

- Knowledgeverse memory topology and sovereignty boundaries.
- Dual-client shared reality contract.
- Foundational form/meaning/rule layering and symlink pattern.
- Adaptive procedural compression (PD04) as storage/runtime optimization.

PM-KR is therefore an umbrella KR technology vocabulary document, not a replacement of domain-specific specs.

---

## 10. Technology Position

PM-KR can be described as:

- A **new knowledge representation paradigm** in K3D.
- A **candidate technology specification** for broader computer science and web technology discussion.

To advance toward formal W3C Recommendation status, governance and external conformance pathways are required (for example W3C Community Group → Working Group publication flow), tracked via documents in `docs/W3C/`.

---

## 11. External Technology Pathway

PM-KR is positioned for external technology development via W3C Community Group process. Complete W3C-facing documentation package available at:

**`docs/W3C/`** (external technology package):
- `PM_KR_PROBLEM_STATEMENT.md` — Motivation and broader impact
- `PM_KR_NORMATIVE_MODEL.md` — Clean normative specification (RFC 2119 compliant)
- `PM_KR_CONFORMANCE_PROFILES.md` — Implementation guidance (Level A/B/C)
- `PM_KR_INTEROPERABILITY_GUIDE.md` — Migration strategies (RDF/OWL/JSON-LD)
- `PM_KR_EVIDENCE_VALIDATION_MATRIX.md` — K3D validation matrix with evidence-maturity tags (repo-verified, run-log verified, target/projection)
- `README.md` — Package overview and technology roadmap

**Proposed W3C Timeline**:
- Q2 2026: Community Group formation
- Q3-Q4 2026: Interoperability testing, third-party implementations
- Q1-Q2 2027: W3C Recommendation (if consensus achieved)

---

## 12. Changelog

**1.1 (2026-02-20)**:
- Added external technology pathway (§11)
- Cross-referenced W3C documentation package (`docs/W3C/`)
- Clarified relationship between K3D vocabulary spec (internal) vs W3C normative model (external)
- No changes to core PM-KR model (maintains v1.0 stability)

**1.0 (2026-02-19)**:
- Initial PM-KR technology document created
- Defines procedural-memory KR invariants, conformance levels, and sovereignty boundary alignment
- Establishes canonical K3D vocabulary specification
