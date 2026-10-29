# Procedural Memory Knowledge Representation: Normative Model

> **Status, 29 October 2026: Draft.** This is project input to the [PM-KR Community Group](https://www.w3.org/community/pm-kr/).
> Knowledge3D code is paused while work focuses on specifications and group deliverables, as [Daniel explained on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
> Requirements and examples below describe proposals, not implemented features or adopted W3C conformance rules.

> **Publication status, clarified 19 October 2026:** This is a Knowledge3D project proposal offered for PM-KR discussion. It is not an adopted Community Group report or a W3C standard. Requirement words describe the proposed model, not an approved W3C conformance regime. See the [document and evidence index](README.md).


**Document Type**: Knowledge3D candidate specification for PM-KR discussion
**Version**: 1.1
**Date**: February 20, 2026
**Authors**: Knowledge3D Project Contributors
**Status**: Draft Specification

---

## Abstract

This document proposes a **normative model** for Procedural Memory Knowledge Representation (PM-KR): executable procedures organized in compositional layers with explicit references.
Its goals are reference reuse, consistency between human and AI views, and a declared execution boundary.
These properties require implementation-specific tests.

---

## Status of This Document

This document is a **draft specification** proposed to the W3C AI Knowledge Representation Community Group. It defines normative requirements using RFC 2119 keywords (MUST, SHOULD, MAY).

**Implementers**: Conformance levels are defined in Section 8. Reference implementation: Knowledge3D (K3D) project.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Terminology](#2-terminology)
3. [Architectural Model](#3-architectural-model)
4. [Normative Data Model](#4-normative-data-model)
5. [Invariants](#5-invariants)
6. [Node Contract](#6-node-contract)
7. [Execution Semantics](#7-execution-semantics)
8. [Conformance](#8-conformance)
9. [Security and Privacy](#9-security-and-privacy)

---

## 1. Introduction

### 1.1 Motivation

This proposal addresses possible integration problems:
- **Duplication**: Related representations can repeat facts and drift when updated independently.
- **Opacity**: Different client views need explicit mappings and tests to establish consistent identity and meaning.
- **Dependencies**: The project proposes a declared runtime boundary; implementations must state the dependencies inside that boundary.

PM-KR addresses these via **procedural canonicalization** + **symlink composition** + **dual-client consistency**.

### 1.2 Design Goals

1. **Zero Duplication**: Canonical procedures referenced, not duplicated
2. **Shared Reality**: Humans and AI consume same procedural source
3. **Sovereign Execution**: Hot path = zero external dependencies
4. **Deterministic**: Same input → same output (reproducibility)
5. **Auditable**: Full provenance and execution traces

### 1.3 Scope

This specification defines:
- Compositional layer model (Form → Meaning → Rules → Meta-Rules)
- Node contract (minimal required fields)
- Normative invariants (MUST satisfy properties)
- Conformance levels (A/B/C implementation tiers)

This specification does NOT define:
- Specific execution engines (PTX, RPN, etc.) — see implementation guides
- Visual rendering (UI/UX) — see spatial UI specifications
- Benchmark policies — see evaluation frameworks

---

## 2. Terminology

### 2.1 Normative Keywords

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in RFC 2119.

### 2.2 PM-KR Concepts

**Procedural Memory**: Knowledge stored as executable procedures (not static payloads).

**Canonical Source**: Single authoritative procedural representation of a concept/symbol.

**Symlink Reference**: Lightweight reference from one node to a canonical node (analogous to filesystem symlinks).

**Form**: Procedural representation of structure/appearance (e.g., glyph geometry, visual rendering).

**Meaning**: Semantic behavior, interpretation, or executable transformation (e.g., mathematical operation, linguistic semantics).

**Hot Path**: Runtime inference path that MUST remain sovereign (zero external dependencies).

**Ingestion Path**: Offline or mid-term path that constructs procedural memory from external data sources.

**Dual-Client**: System where humans (visual perception) and AI (semantic processing) consume the same underlying data representation.

---

## 3. Architectural Model

### 3.1 Layer Composition

PM-KR organizes knowledge in **four compositional layers**:

```
┌─────────────────────────────────────┐
│ Meta-Rules Layer                    │  Strategy selection, control flow
│   ↓ references (rule_refs)          │
├─────────────────────────────────────┤
│ Rules Layer                         │  Transformation programs
│   ↓ references (symbol_refs)        │
├─────────────────────────────────────┤
│ Meaning Layer                       │  Semantic composition
│   ↓ references (word_refs, char_refs) │
├─────────────────────────────────────┤
│ Form Layer                          │  Canonical procedures
└─────────────────────────────────────┘
```

**Normative Requirement**: Higher layers MUST reference lower canonical layers rather than duplicate payload content.

### 3.2 Reference Graph

PM-KR nodes form a **directed acyclic graph (DAG)** of references:

- **Nodes**: Procedural memory units (characters, words, symbols, rules, meta-rules)
- **Edges**: Reference relationships (char_refs, word_refs, symbol_refs, rule_refs, component_refs)
- **Acyclicity**: Reference graphs SHOULD be acyclic by design for deterministic reconstruction

**Example**:
```
Word("rotation_task")
  ├─ char_refs → [Char('r'), Char('o'), Char('t'), Char('a'), Char('t'), Char('i'), Char('o'), Char('n')]
  ├─ char_refs → [Char('_')]
  └─ char_refs → [Char('t'), Char('a'), Char('s'), Char('k')]

Grammar("rotate_pattern")
  ├─ word_refs → [Word("rotation_task")]
  └─ transformation_rpn → "1 ROTATE 90 DEGREES_CW"
```

**Compression**: 12 char_refs (lightweight) vs 12 glyph duplications (heavy).

### 3.3 Form-Meaning Duality

PM-KR nodes MAY carry both `form_program` and `meaning_program`:

**Form Program**: Procedural representation of structure/appearance
- Visual glyphs (character rendering)
- Geometric shapes (drawing primitives)
- Audio waveforms (spectrograms)

**Meaning Program**: Semantic behavior/interpretation
- Mathematical operations (symbolic execution)
- Linguistic semantics (word meanings)
- Transformation rules (grammar patterns)

**Normative Requirement**: If meaning is derivable from form, derivation metadata SHOULD be explicit.

---

## 4. Normative Data Model

### 4.1 Minimal Node Schema

A PM-KR-compliant node MUST expose:

```json
{
  "id": "unique_stable_identifier",
  "layer": "form | meaning | rules | meta_rules",
  "form_program": "procedural_source_code",   // OPTIONAL if meaning-only
  "meaning_program": "semantic_procedure",    // OPTIONAL if form-only
  "refs": {
    "char_refs": ["char_id_1", "char_id_2"],  // OPTIONAL (depends on layer)
    "word_refs": ["word_id_1"],               // OPTIONAL
    "symbol_refs": ["symbol_id_1"],           // OPTIONAL
    "rule_refs": ["rule_id_1"],               // OPTIONAL
    "component_refs": ["component_id_1"]      // OPTIONAL (for molecules, etc.)
  },
  "metadata": {
    "provenance": "source_description",
    "domain": "math | visual | audio | language | physics",
    "version": "1.0",
    "confidence": 0.95,
    "timestamp": "ISO8601_datetime"
  }
}
```

**Extensibility**: Implementations MAY add fields, but MUST preserve these core semantics.

**Defeasible Reasoning Metadata (March 2026 Extension):**

Grammar-type nodes (Layer 3: Rules) MAY include defeasible logic fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rule_strength` | trit (-1, 0, +1) | 0 | Strict (+1), defeasible (0), or defeater (-1) |
| `superior_to` | array of node IDs | [] | Explicit superiority relations for conflict resolution |
| `trust_weight` | float [0, 1] | 1.0 | Source confidence, subject to sleep-time decay |

These fields enable non-monotonic reasoning: conclusions drawn from defeasible rules can be withdrawn when contradicted by superior evidence. The `superior_to` graph is processed by the sovereign `gre_defeasible_resolver` kernel at query time.

**Conformance:** Defeasible metadata is OPTIONAL at Level A. Level B (Sovereign Runtime) implementations SHOULD support defeasible resolution for Grammar nodes. Level C (Auditable Production) implementations MUST support defeasible resolution and log verdict events.

### 4.2 Reference Field Semantics

| Field | Layer | Points To | Example |
|-------|-------|-----------|---------|
| `char_refs` | Meaning, Rules | Form (characters) | Word → sequence of character IDs |
| `word_refs` | Rules, Meta-Rules | Meaning (words) | Grammar → semantic word IDs |
| `symbol_refs` | Rules | Meaning (math symbols) | Equation → operator/constant IDs |
| `rule_refs` | Meta-Rules | Rules (transformations) | Strategy → applicable rule IDs |
| `component_refs` | Meaning | Form/Meaning (atoms, molecules) | Molecule → constituent atom IDs |

**Normative Requirement**: All reference fields MUST use stable identifiers that resolve to canonical nodes.

### 4.3 Procedural Program Format

PM-KR does NOT mandate specific procedural languages, but implementations MUST document:
- **Language**: RPN, Lisp, Forth, etc.
- **Execution Environment**: PTX, WebAssembly, LLVM IR, etc.
- **Determinism Guarantees**: Fixed seed → fixed output

**K3D Reference Implementation**: Uses RPN (Reverse Polish Notation) programs executed via PTX kernels on NVIDIA GPUs.

---

## 5. Invariants

### 5.1 Canonicality Invariant

**Requirement**: For each canonical concept/program, there MUST be exactly one authoritative procedural source per versioned namespace.

**Rationale**: Prevents duplication, ensures single source of truth.

**Validation**: Content-addressable IDs or registry enforcement.

### 5.2 Reference Preservation Invariant

**Requirement**: If node B reuses canonical content from node A, then B MUST reference A; B MUST NOT inline duplicate canonical payload unless explicitly marked as materialized cache.

**Rationale**: Reusing references can avoid repeated payloads; net storage depends on the data, metadata and reference overhead.

**Example Violation**:
```json
// WRONG: Word duplicates character glyphs
{
  "id": "word_rotation",
  "chars": [
    {"glyph": "...", "font": "..."}, // DUPLICATE!
    {"glyph": "...", "font": "..."}  // DUPLICATE!
  ]
}

// CORRECT: Word references canonical characters
{
  "id": "word_rotation",
  "char_refs": ["char_r", "char_o", "char_t", ...]
}
```

### 5.3 Deterministic Reconstruction Invariant

**Requirement**: Given canonical procedures + references + versioned metadata, reconstruction MUST be deterministic for a fixed runtime/kernel version.

**Rationale**: Enables reproducibility, auditability, and verification.

**Validation**: Checksum verification (e.g., SHA-256 on reconstructed output).

### 5.4 Dual-Client Equivalence Invariant

**Requirement**: People and AI clients MUST observe/consume the same underlying node truth, differing only in representation modality (visual vs semantic).

**Rationale**: Ensures shared ground truth, enables verification.

**Example**:
```
Same procedural program:
├─ Human perception: Visual glyph rendering (GPU rasterization)
└─ AI perception: Semantic embedding (GPU vector operations)

Verification: Human click (x,y,z) → AI retrieves same node ID → Same metadata
```

### 5.5 Sovereign Boundary Invariant

**Requirement**:
- Hot path (runtime inference) MUST execute through sovereign components (zero external dependencies).
- Ingestion/augmentation MAY use external tooling, but outputs MUST crystallize into PM-KR-compliant procedural memory before runtime use.

**Rationale**: Guarantees determinism, security, and auditability.

**Example**:
```python
# Ingestion (ALLOWED: flexible, external tools)
import pdfplumber, numpy as np
text = pdfplumber.extract_text(pdf_path)  # External lib OK here

procedural_memory = convert_to_rpn(text)  # Crystallize to procedural form

# Hot path (REQUIRED: sovereign only)
result = execute_rpn_ptx(procedural_memory)  # Zero external dependencies!
```

### 5.6 Auditability Invariant

**Requirement**: Node provenance and transformation history SHOULD be recorded such that procedural origins and reference lineage are inspectable.

**Rationale**: Enables debugging, trust verification, and compliance auditing.

**Metadata Fields**:
- `provenance.source`: Original data source (PDF, API, human input)
- `provenance.transformation_chain`: Sequence of transformations applied
- `provenance.timestamp`: Creation/modification timestamps
- `provenance.agent`: Creating agent (human, AI, hybrid)

---

## 6. Node Contract

### 6.1 Mandatory Fields

All PM-KR nodes MUST include:
- `id` (string): Stable unique identifier
- `layer` (enum): One of `{form, meaning, rules, meta_rules}`
- `metadata.version` (string): Semantic version (e.g., "1.0.0")

### 6.2 Conditional Fields

Depending on layer, nodes MUST include:

**Form Layer**:
- `form_program` (string): Procedural source code for visual/structural representation

**Meaning Layer**:
- `meaning_program` (string): Semantic/behavioral procedure
- `refs` (object): At least one reference field (char_refs, word_refs, component_refs)

**Rules Layer**:
- `transformation_rpn` (string): Executable transformation program
- `refs.symbol_refs` OR `refs.word_refs` (array): Referenced lower-layer nodes

**Meta-Rules Layer**:
- `strategy_condition` (string): When to apply this meta-rule
- `refs.rule_refs` (array): Rules controlled by this meta-rule

### 6.3 Optional Fields

Implementations MAY add:
- `embeddings` (object): Matryoshka-tier embeddings for search/LOD (regenerable from procedural source)
- `cached_visual` (object): Pre-rendered visual cache (regenerable from form_program)
- `confidence` (number): Certainty score [0.0, 1.0]
- `usage_count` (integer): Access frequency (for eviction policies)

---

## 7. Execution Semantics

### 7.1 Reference Resolution

**Specification**: When a node references another (via `char_refs`, `word_refs`, etc.), implementations MUST:
1. Resolve reference to canonical node ID
2. Retrieve canonical node's procedural program
3. Execute/compose procedure as needed
4. Cache result if beneficial (mark as `materialized_cache`)

**Acyclicity Guarantee**: Reference graphs SHOULD be DAGs. If cycles detected, implementations MUST fail-fast with error.

### 7.2 Procedural Execution

**Specification**: When executing a `form_program` or `meaning_program`, implementations MUST:
1. Use documented execution environment (e.g., PTX, WebAssembly)
2. Guarantee determinism (fixed seed → fixed output)
3. Maintain sovereignty (zero external dependencies in hot path)
4. Provide error traces (for debugging)

**K3D design reference**: RPN programs execute through PTX kernels; telemetry must measure the declared execution boundary.
This draft establishes no latency result.

### 7.3 Dual-Client Rendering

**Specification**: For dual-client systems, the same procedural program MUST generate:
- **Human perception**: Visual geometry, textures, spatial layout (via GPU rasterization)
- **AI perception**: Semantic embeddings, graph topology (via GPU vector operations)

**Validation**: Humans and AI querying same node ID MUST retrieve identical metadata.

---

## 8. Conformance

### 8.1 Conformance Levels

**Level A: PM-KR Core**
- Implements 4-layer model (Form → Meaning → Rules → Meta-Rules)
- Enforces Canonicality Invariant (one canonical source per concept)
- Enforces Reference Preservation Invariant (symlink composition)
- Supports Deterministic Reconstruction Invariant (checksums pass)

**Level B: PM-KR Sovereign Runtime**

Includes Level A, plus:
- Enforces Sovereign Boundary Invariant (PTX/RPN hot path, zero external deps)
- Implements fail-fast behavior for unavailable sovereign backends
- Provides execution telemetry (latency, GPU call counts)

**Level C: PM-KR Auditable Production**

Includes Level B, plus:
- Enforces Auditability Invariant (provenance tracking, transformation chains)
- Provides compression metrics (reference graph size vs payload size)
- Includes conformance test suite (canonicality, reference resolution, determinism)

### 8.2 Conformance Testing

Implementations claiming PM-KR conformance SHOULD provide:

1. **Canonicality Tests**: Verify no duplicate canonical procedures
2. **Reference Resolution Tests**: Verify symlink graphs resolve correctly
3. **Determinism Tests**: Same seed → same output (10+ runs)
4. **Dual-Client Tests**: Human and AI retrieve same node metadata
5. **Sovereignty Tests**: Hot path uses zero external dependencies

**Proposed test suite**: A shared conformance suite remains a deliverable to agree and publish.
Repository test files are implementation inputs, not certification of this model.

---

## 9. Security and Privacy

### 9.1 Provenance Verification

**Threat**: Malicious nodes injected into procedural memory graph.

**Mitigation**: Implementations SHOULD:
- Cryptographically sign canonical procedures (e.g., SHA-256 + RSA signature)
- Verify provenance chains before execution
- Define and test an execution sandbox, memory-access checks and resource limits for untrusted procedures.

### 9.2 Reference Integrity

**Threat**: Broken references causing runtime failures.

**Mitigation**: Implementations MUST:
- Validate all references resolve to canonical nodes (fail-fast on broken refs)
- Implement reference counting or garbage collection (prevent dangling refs)
- Provide reference graph validation tools

### 9.3 Dual-Client Trust

**Threat**: Humans and AI perceive different realities (inconsistency attacks).

**Mitigation**: Implementations MUST:
- Guarantee Dual-Client Equivalence Invariant (same procedural source)
- Provide verification UI (human clicks → AI shows retrieved metadata)
- Log all procedural executions (audit trail)

---

## 10. Normative References

**Standards**:
- RFC 2119: Key words for use in RFCs to Indicate Requirement Levels
- glTF 2.0 Specification (Khronos Group)
- WebAssembly Core Specification (W3C)

**K3D Specifications**:
- Knowledgeverse Specification (7-region memory architecture)
- Dual-Client Contract Specification (procedural foundation)
- Adaptive Procedural Compression Specification (PD04 codecs)
- Sovereign NSI Specification (PTX-only hot path)

---

## Appendix A: Example Node Instances

### A.1 Form Layer: Character Node

```json
{
  "id": "char_latin_a",
  "layer": "form",
  "form_program": "BEZIER_CURVE [control_points...] PROCEDURAL_FONT_LATIN_A",
  "metadata": {
    "provenance": "Unicode U+0041, Latin script",
    "domain": "language",
    "version": "1.0",
    "timestamp": "2026-01-15T10:00:00Z"
  }
}
```

### A.2 Meaning Layer: Word Node

```json
{
  "id": "word_rotation",
  "layer": "meaning",
  "char_refs": ["char_r", "char_o", "char_t", "char_a", "char_t", "char_i", "char_o", "char_n"],
  "meaning_program": "CONCEPT_ROTATION SPATIAL_TRANSFORMATION ANGULAR",
  "metadata": {
    "provenance": "English lexicon",
    "domain": "language",
    "version": "1.0",
    "confidence": 0.98
  }
}
```

### A.3 Rules Layer: Grammar Pattern

```json
{
  "id": "grammar_rotate_90",
  "layer": "rules",
  "word_refs": ["word_rotation"],
  "transformation_rpn": "1 ROTATE 90 DEGREES_CW",
  "metadata": {
    "provenance": "ARC-AGI training data",
    "domain": "visual",
    "version": "1.0",
    "confidence": 0.95
  }
}
```

### A.4 Meta-Rules Layer: Strategy Selection

```json
{
  "id": "meta_rule_rotation_strategy",
  "layer": "meta_rules",
  "strategy_condition": "asymmetric_input AND rotation_task",
  "rule_refs": ["grammar_rotate_90", "grammar_rotate_180"],
  "confidence": 0.92,
  "metadata": {
    "provenance": "Shadow Copy learning (Week 21.2)",
    "domain": "visual",
    "version": "1.0"
  }
}
```

---

## Appendix B: Compression Analysis

This appendix describes measurements needed to evaluate reference reuse.
It contains no verified compression result.

### B.1 Character Galaxy (K3D Validation)

Compare complete font or glyph assets with procedure code, metadata and references for the same glyphs.
Include shaping behavior and reconstruction differences in the report.

### B.2 Semantic Tags (Math Benchmark)

Compare the actual serialized tags against canonical entries and references.
Record which tags can share an identity without changing their contextual meaning.

### B.3 PDF Knowledge (Overnight Ingestion)

Retain the original corpus manifest and all extracted content needed for the task.
A smaller derived knowledge representation is not lossless compression of the PDFs unless reconstruction is demonstrated.

---

## Document Status

**Version**: 1.1 (Draft Specification)
**Date**: February 20, 2026
**Status**: Proposed to W3C AI KR Community Group
**License**: CC-BY-4.0 (free to share, adapt, with attribution)
**Reference Implementation**: Knowledge3D (K3D) - https://github.com/danielcamposramos/Knowledge3D
**Test suite status**: No external conformance result is established by this document.

---

## Acknowledgments

The project credits its contributors in [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md):
- **Daniel Ramos** (Architect and founder)
- **AI Partners**: Claude, Codex, Grok, GLM, Kimi, DeepSeek, Qwen
- Public feedback is summarized with sources in the [explainer](EXPLAINER.md#stakeholder-feedback).

**Foundational Standards**:
- glTF 2.0 (Khronos Group) — 3D asset format
- WebXR Device API (W3C) — spatial interfaces
- PTX ISA (NVIDIA) — GPU execution environment

**Philosophy**: We patent nothing. We publish everything. We build in the open.

---

**End of Normative Model Specification**
