# Draft WebML Proposal: Procedural Knowledge Profile for WebML

Date: 2026-03-03
Status: Draft for internal review before WebML submission
Target venue: `https://github.com/webmachinelearning/proposals`

## 1) Title

Procedural Knowledge Profile for WebML: Composable, Auditable, and Portable Reasoning Artifacts

## 2) Abstract

WebML proposals today strongly cover runtime APIs, model placement, and agent/tool interoperability, but they do not yet define a shared procedural knowledge representation layer for reusable reasoning artifacts.

This proposal introduces an incubatable PM-KR-aligned profile for WebML ecosystems: a compact, executable, and auditable representation for knowledge units that can be composed and referenced across workflows. The profile is explicitly complementary to WebNN and WebMCP directions. It does not replace runtime execution APIs or protocol stacks; it standardizes the exchange and composition of procedural knowledge artifacts used by those systems.

The expected benefits are practical:
1. Better interoperability for representation-level tooling.
2. Improved traceability for agent-assisted workflows.
3. Lower duplication pressure via reference-based composition.

Evidence grounding follows two tiers: observed repository metrics (for example, documented 70% class compression snapshots in PM-KR evidence docs) and scenario projections (for example, carbon/latency scenarios in the Carbon Blueprint), with clear separation between the two.

## 3) Motivation and Use Cases

### 3.1 Why this is needed now

In `webmachinelearning/proposals`, active threads already show demand for:
- Portable graph/model representations (#16).
- Lightweight descriptors for placement decisions (#15).
- Tool/agent interoperability surfaces (#12/#14).
- Hybrid distribution and caching concerns (#5).

These threads solve adjacent parts of the stack. What is missing is a common representation profile for procedural knowledge artifacts that can be reused across these tracks.

### 3.2 Use Case A: Representation portability beyond model topology

Problem:
- Graph portability work is active, but graph syntax alone does not encode richer procedural meaning contracts.

Proposed PM-KR profile role:
- Add optional metadata and compositional references for procedural semantics.
- Keep host format compatibility while enabling reusable reasoning units.

Outcome:
- Toolchains can exchange not only graph structure but also portable procedural knowledge components.

### 3.3 Use Case B: Agentic workflows with auditable execution traces

Problem:
- Agent protocols expose tools, but tool behavior semantics are often opaque in practice.

Proposed PM-KR profile role:
- Represent tool knowledge artifacts as explicit procedural units with inspectable structure.
- Enable deterministic trace capture for governance and debugging.

Outcome:
- Better accountability for assistant/tool interactions without prescribing a single agent protocol.

### 3.4 Use Case C: Hybrid client/cloud systems with lower duplication

Problem:
- Hybrid systems repeatedly transfer similar structures or recreate equivalent artifacts.

Proposed PM-KR profile role:
- Reference-based composition and canonical IDs for reusable procedural components.

Outcome:
- Reduced duplication pressure in transport/storage workflows and cleaner cross-origin caching strategies.

## 4) Technical Specification (Incubation Scope)

### 4.1 Non-goals

This proposal does not define:
1. A replacement for WebNN runtime APIs.
2. A replacement for WebMCP protocol APIs.
3. Browser-level scheduling/offloading policy.

### 4.2 Core artifacts

A profile-compliant artifact includes:
1. `id`: stable canonical identifier.
2. `form_program`: optional procedural representation of structural form.
3. `meaning_program`: optional procedural semantics payload.
4. `refs`: compositional references to other artifacts.
5. `context_rules`: optional context-dependent execution hints.
6. `provenance`: optional metadata for source and transformation chain.

### 4.3 Minimal JSON shape (illustrative)

```json
{
  "id": "pmkr:concept:linear_equation.solve.v1",
  "form_program": ["TOKENIZE", "a", "x", "b", "c"],
  "meaning_program": ["PUSH", "c", "PUSH", "b", "SUB", "PUSH", "a", "DIV"],
  "refs": [
    "pmkr:op:sub.v1",
    "pmkr:op:div.v1"
  ],
  "context_rules": {
    "algebra": ["strict"],
    "education": ["show_steps"]
  },
  "provenance": {
    "source": "webml-proposal-incubation",
    "version": "0.1.0"
  }
}
```

### 4.4 Integration points

#### With WebNN graph efforts (#16)
- Map graph nodes to optional PM-KR references for richer procedural semantics.
- Keep graph execution independent from PM-KR adoption.

#### With DAOP-like descriptors (#15)
- Allow PM-KR artifact metadata to enrich weightless descriptors for better placement heuristics.

#### With WebMCP-like tracks (#12/#14)
- Use PM-KR artifact IDs as stable semantics handles for tool capabilities.

## 5) Performance Analysis (Observed vs Projected)

### 5.1 Observed (repository-documented)

From PM-KR evidence docs in this repository:
- Compression class evidence includes documented ~70% snapshot reductions in specific datasets.
- Sovereign/runtime docs report small-model procedural reasoning architecture and traceability orientation.

Interpretation:
- Representation-level deduplication and procedural composition are plausible and already documented in reference artifacts.

### 5.2 Projected (scenario-level, clearly labeled)

From `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md`:
- Compression scenarios include 200:1 to 1000:1 ranges in specific contexts.
- Latency and carbon impact figures are scenario projections and must not be presented as universal production baselines.

Proposal policy:
- WebML submission must label these as projected scenarios pending independent validation in WebML-specific prototypes.

### 5.3 Suggested prototype metrics for WebML incubation

For a first validation cycle, measure:
1. Artifact size reduction versus naive duplicated representation.
2. Serialization/deserialization latency overhead.
3. Traceability overhead (time and bytes).
4. Interop success rate across at least two toolchains.

## 6) Comparison with Existing Approaches

### 6.1 Versus graph-only portability
- Graph portability standardizes topology exchange.
- PM-KR profile adds reusable procedural semantics and compositional references.

### 6.2 Versus model compression alone
- Compression techniques reduce model size.
- PM-KR profile targets representational duplication and semantic composability across artifacts.

### 6.3 Versus prompt-only RAG pipelines
- Prompt/RAG surfaces optimize retrieval and generation UX.
- PM-KR profile provides deterministic, inspectable procedural units as retrieval targets.

### 6.4 Complement stance

PM-KR profile is explicitly additive:
- Use WebNN for execution APIs.
- Use WebMCP for protocol-level agent/tool exchange.
- Use PM-KR profile for composable procedural knowledge artifacts.

## 7) Standards Alignment

This proposal aligns with:
1. WebML proposal process (issue-first incubation).
2. PM-KR CG documents in this repository (`docs/w3c-specifications/*`, `docs/W3C/*`, `docs/vocabulary/*`).
3. Interoperability direction with RDF/OWL/JSON-LD documented in PM-KR interoperability artifacts.

Implementation neutrality principle:
- The profile should not require one hardware backend.
- It should remain usable across WebNN/WebGPU and other conforming runtimes.

## 8) Implementation Plan

### Phase 1 (0-6 weeks): explainer and profile draft
- Create a focused explainer with explicit goals/non-goals.
- Define minimal JSON profile and conformance checklist v0.

### Phase 2 (6-12 weeks): reference adapters
- Build adapter A: graph-oriented artifact -> PM-KR profile mapping.
- Build adapter B: PM-KR profile -> graph/tool annotation mapping.

### Phase 3 (12-16 weeks): interoperability prototype
- Run two-toolchain interop demo.
- Produce measurable artifact-size and traceability reports.

### Phase 4 (16+ weeks): community review and next-step decision
- Feed results back to WebML proposals repo.
- Decide whether to spin out to dedicated incubation repo.

## 9) Security and Privacy Considerations

1. Keep user data out of canonical artifact IDs.
2. Require explicit provenance fields for generated/transformed artifacts.
3. Support policy hooks for privacy/transparency boundaries.
4. Preserve deterministic traceability for audits, debugging, and safety review.

This aligns with current WebML concerns visible in fact-checking/offloading discussions, where governance and trust boundaries are central.

## 10) References

### WebML proposal landscape
- `https://github.com/webmachinelearning/proposals`
- Open issues analyzed: #1-#16
- Key alignment issues: #5, #12, #15, #16

### PM-KR/K3D grounding docs (repository-local)
- `docs/w3c-specifications/library/PM_KR_CG_CHARTER.md`
- `docs/w3c-specifications/workshop/phase1-data-model/spec-draft.md`
- `docs/W3C_PM_KR_COMMUNITY_GROUP_MISSION.md`
- `docs/W3C_PM_KR_OBJECTIVES_v1.2.md`
- `docs/W3C/PM_KR_NORMATIVE_MODEL.md`
- `docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md`
- `docs/W3C/PM_KR_INTEROPERABILITY_GUIDE.md`
- `docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md`
- `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md`

---

## Appendix: Submission-Ready Short Form (for WebML issue template)

### Proposal name
Procedural Knowledge Profile for WebML

### Short description
A complementary profile for representing composable, auditable procedural knowledge artifacts in WebML ecosystems. The profile targets interoperability gaps between model graph portability, hybrid placement workflows, and agent/tool integration. It is additive to existing WebNN/WebMCP directions.

### Example use cases
1. Attach reusable procedural semantics to graph artifacts for toolchain portability.
2. Use stable procedural artifact IDs as tool semantics handles in agent workflows.
3. Reduce duplication in hybrid client/cloud workflows through reference composition.

### A rough idea or two about implementation
Start with a minimal JSON profile and two adapters (graph->profile and profile->graph/tool annotation), then validate interoperability across two toolchains. Keep scope in explainer-land first; only pursue formal spec progression after prototype evidence.
