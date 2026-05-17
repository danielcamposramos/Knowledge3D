# WebML Competitive Analysis for PM-KR Proposal

Date: 2026-03-03
Goal: Position PM-KR as complementary to ongoing WebML proposal directions.

## 1. Closest Existing Proposals

| Issue | Existing Proposal Focus | Overlap with PM-KR | PM-KR Differentiator | Complement Strategy |
|---|---|---|---|---|
| #16 | WebNN Graph DSL and portable file format | Representation portability | PM-KR adds compositional semantics (form->meaning->rules) and cross-modal procedural references | Propose PM-KR profile as optional semantic/procedural extension layer over graph artifacts |
| #15 | Dynamic AI Offloading Protocol (DAOP) | Lightweight model descriptors, runtime decisions | PM-KR focuses on knowledge/program representation, not scheduling policy | Provide PM-KR descriptors as richer input to offloading estimators |
| #12 | WebMCP API | Agent/tool interoperability | PM-KR focuses on canonical procedural knowledge units and auditable execution traces | Use PM-KR artifacts as reusable tool semantics payloads in MCP workflows |
| #5 | Hybrid AI Exploration | Model distribution, caching, partitioning | PM-KR emphasizes deduplicated procedural composition for reusable knowledge | Pair PM-KR deduplication profile with hybrid model distribution efforts |
| #8 | Prompt API local RAG use cases | Retrieval and prompting UX | PM-KR emphasizes deterministic procedural memory, not prompt UX surface | Use PM-KR as structured retrieval substrate behind RAG pipelines |
| #9 | Local Inference extension | Fast prototyping path | PM-KR is a specification-ready representation layer, not extension delivery mechanism | Pilot PM-KR artifacts inside extension prototypes to generate evidence |

## 2. Where PM-KR Must Avoid Competition Framing

Do not position PM-KR as replacing:
1. WebNN/WebGPU runtime APIs.
2. WebMCP protocol efforts.
3. Hybrid AI architecture initiatives.

Recommended framing:
- PM-KR = representation and compositional semantics layer that interoperates with all three.

## 3. Differentiation Axes That Matter to Reviewers

### A) Deterministic procedural artifacts
- WebML proposals mostly focus on APIs and workflows.
- PM-KR contributes explicit executable knowledge units and composition semantics.

### B) Auditable form-to-meaning contract
- Existing proposals discuss capability and performance.
- PM-KR contributes inspectable structure for traceability and cross-client consistency.

### C) Deduplication-aware knowledge composition
- Existing proposals emphasize model transport/caching.
- PM-KR adds canonical references reducing representational duplication pressure.

### D) Standards bridge orientation
- PM-KR can bridge to RDF/OWL/JSON-LD interoperability work while still fitting WebML runtime ecosystem.

## 4. Risk Assessment

### Risk 1: "Too broad" perception
Mitigation:
- Submit a narrow, incubatable v1 profile (one concrete artifact format and one prototype path).

### Risk 2: "Out of scope for WebML" perception
Mitigation:
- Anchor proposal to active needs in #16/#15 (portable descriptors and lightweight execution planning).

### Risk 3: Overclaiming performance/carbon metrics
Mitigation:
- Separate observed repository metrics from scenario projections.
- Provide explicit citations and caveats.

## 5. Recommended Proposal Position Statement

"PM-KR is a complementary procedural knowledge representation profile for WebML ecosystems. It does not replace WebNN execution APIs or WebMCP protocols; it standardizes reusable, auditable, and composable knowledge artifacts that can be exchanged across these systems."

## 6. Practical Collaboration Targets

Short-list for early collaboration:
1. Engage #16 authors on representation convergence points.
2. Engage #15 authors on weightless descriptor enrichment.
3. Engage WebMCP track (#12/#13/#14) on tool semantics portability.

This collaboration-first posture is consistent with observed successful paths in the repository.
