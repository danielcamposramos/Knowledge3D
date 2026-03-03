# How PM-KR and K3D Work (Technical Deep Dive)

## 1) System Objective
PM-KR defines a knowledge representation model where canonical procedural artifacts are stored once and reused through references across form, meaning, rules, and meta-rules. K3D is the operational reference implementation that validates this model with deterministic runtime behavior and sovereign hot-path constraints.

## 2) Architectural Surfaces
K3D describes a three-surface execution environment:
- House: persistent domain memory and organized artifacts
- Galaxy Universe: active VRAM reasoning workspace
- Cranium: active routing, composition, and execution logic

The important behavior is not the labels. It is the contracts between them:
- Ingestion and augmentation can be expensive and asynchronous.
- Runtime reasoning must be bounded, inspectable, and deterministic.
- Canonical representations must survive handoffs across all surfaces.

## 3) Layer Model
PM-KR uses four compositional layers.

1. Form Layer
Canonical procedural primitives (for example character forms, shape generators, base symbols).

2. Meaning Layer
Semantic interpretation and domain meaning attached to form references.

3. Rules Layer
Executable transformation logic over meaning nodes.

4. Meta-Rules Layer
Selection and control strategies for rules under context.

Normative direction: higher layers reference lower layers; they do not duplicate canonical payload.

## 4) Symlink-Style Composition
A PM-KR node is a structured unit with stable ID, layer assignment, optional form and meaning programs, and reference fields such as `char_refs`, `word_refs`, `symbol_refs`, and `rule_refs`.

When a concept is reused, a new node does not clone full payload. It stores references to the existing canonical nodes. This enables:
- compression
- consistency
- update locality
- lineage tracing

This is similar to symlink semantics in operating systems, but applied to knowledge artifacts.

## 5) Ingestion Pipeline
The current pipeline has two major tracks.

Track A: Benchmarks
Question and answer material is enriched into PM-KR-compatible payload rows, preserving supervision context and procedural hints.

Track B: PDF corpus
Pages are extracted, classified, and selectively augmented before conversion into structured payload rows.

Key implementation properties:
- resumable page staging
- checkpoint rebuilding of payload outputs
- skip-source logs for encrypted or unreadable files
- deterministic rebuild from stage state

This is intentionally not hot-path logic. It is fundamental construction work.

## 6) Runtime Reasoning Path
The runtime path is designed for bounded behavior.

Canonical flow:
1. Query arrives.
2. Router identifies relevant pattern family and specialist scope.
3. Galaxy retrieval returns candidate procedural assets.
4. Composer builds executable program (for example RPN form).
5. Sovereign kernel path executes.
6. Result and telemetry are recorded.

Success criteria include:
- no hidden fallback in hot path
- deterministic replay for fixed conditions
- traceability of selected patterns and execution steps

## 7) Example: Simple RPN Program
A minimal equation example:

If `2x + 3 = 11`, solve for `x`.

Canonical transform:
`x = (11 - 3) / 2`

RPN sequence:
```text
11 3 - 2 /
```

This is not presented as a full symbolic math system. It is a transparent procedural core where each operation is explicit and inspectable.

## 8) Sovereign Runtime Boundaries
K3D distinguishes between:
- hot path: strict sovereignty and deterministic execution requirements
- ingestion path: broader tooling allowed, output must crystallize into PM-KR artifacts

This separation avoids false purity while keeping runtime guarantees clear.

## 9) Audit and Provenance
PM-KR emphasizes auditability by design.

A production-oriented node/report lifecycle should include:
- source provenance
- transformation metadata
- execution trace references
- boundary policy context when relevant

For governance, this supports reproducible review instead of narrative-only claims.

## 10) Why This Is Different From Typical LLM-Only Pipelines
LLM-only pipelines often centralize knowledge in model weights or transient prompt chains. PM-KR pushes knowledge into explicit, reusable procedural memory objects with stable identity and reference composition.

Practical effects:
- easier inspection of what was used
- easier comparison across versions
- reduced duplication pressure in multi-team environments
- clear integration path for human-facing and machine-facing outputs from one source

## 11) Seven-Region Knowledgeverse View (Conceptual)
The Knowledgeverse specification describes a unified memory topology with pinned kernels, active galaxy context, house context, world view, model weights, audit region, and ingestion/staging region.

You can treat this as a systems map for responsibility partitioning:
- reasoning and retrieval
- persistence and organization
- synchronization and auditing
- ingestion and transformation

The map is useful because it prevents architecture drift. Teams can state where a capability belongs and where it does not.

## 12) Conformance Orientation
The W3C package describes conformance profiles and validation evidence pathways.

A robust implementation should show:
- canonicality and reference-preservation checks
- deterministic reconstruction checks
- dual-client consistency checks
- runtime sovereignty checks

## 13) Practical Integration Outlook
PM-KR is intended to interoperate with existing standards ecosystems. It does not require abandoning declarative semantics. It adds executable procedural continuity across representations.

Near-term integrations target:
- W3C CG workflow and cross-CG collaboration
- web runtime discussions (WebML, WebGPU-adjacent concerns)
- enterprise knowledge systems needing auditability and lower duplication overhead

## 14) Summary
PM-KR in K3D can be summarized as:
- canonical procedural memory
- reference-first composition
- deterministic, auditable execution behavior
- explicit separation of ingestion complexity and runtime guarantees

That combination is what makes the model suitable for standardization discussion: it is concrete enough to test and broad enough to interoperate.

## 15) Implementation Checklist for Engineering Teams
For teams evaluating implementation or pilot adoption, this checklist is a practical starting point:

1. Canonical Node Registry
Define stable IDs and versioning rules for canonical nodes before broad ingestion.

2. Reference Policy
Enforce \"reference when possible\" at ingestion and transformation time.

3. Determinism Contract
Document runtime seed, kernel version, and environment constraints used for deterministic replay.

4. Audit Output Format
Choose a trace format that can be consumed by both engineering and compliance teams.

5. Boundary Policy Attachment
Attach boundary metadata to critical procedural assets where execution or disclosure is sensitive.

6. Conformance Gate
Add test gates for canonicality, reference preservation, deterministic reconstruction, and runtime sovereignty before release.

This checklist keeps PM-KR implementation grounded in operational readiness, not only conceptual alignment.

Sources:
- docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md
- docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md
- docs/W3C/PM_KR_NORMATIVE_MODEL.md
- docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md
- README.md
