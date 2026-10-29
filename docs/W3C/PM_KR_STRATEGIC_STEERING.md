# PM-KR Strategic Steering for K3D Development

> **Status, 29 October 2026: Superseded roadmap.** This is project input to the [PM-KR Community Group](https://www.w3.org/community/pm-kr/).
> Knowledge3D code is paused while work focuses on specifications and group deliverables, as [Daniel explained on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
> Requirements and examples below describe proposals, not implemented features or adopted W3C conformance rules.

**Status:** Historical project roadmap; current priorities are in the [explainer](EXPLAINER.md).
**Last Updated:** February 26, 2026
**Source:** Knowledge3D project planning input; not a group decision.

---

## Executive Summary

This historical draft records proposed work on access control, provenance, planning, interoperability, documentation and conformance tests.
Its schedules and APIs are planning sketches, not implementation results or Community Group commitments.
The current focus is specifications and a small shared vocabulary, as [Daniel explained on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).

---

## The 6 Strategic Imperatives

### 1. Formalizing Access Control and Sovereignty Boundaries

**Status:** Project proposal; no participant endorsement is asserted.

**The Challenge:**
The proposed design needs explicit boundaries between shared knowledge and permissioned execution.

**K3D Development Goal:**
Rigorously formalize K3D's **House (private/sovereign) and Galaxy (public/canonical)** architecture into a standardized access control model.

**Implementation Requirements:**

1. **Security Metadata Layer:**
   - Implement JSON-LD security annotations on procedural programs
   - Align with Model Context Protocol (MCP) tool annotations
   - Support .NET Code Access Security (CAS) patterns
   - Integrate BPMN security extensions for workflow metadata

2. **Permission Schema:**
   ```json
   {
     "@context": "https://pm-kr.org/contexts/security.jsonld",
     "@type": "ProceduralProgram",
     "id": "galaxy:math:sqrt",
     "program": ["DUP", "0", "GT", "ASSERT", "SQRT"],
     "permissions": {
       "visibility": "public",
       "execution": "authenticated",
       "minimumTrustLevel": "verified",
       "requiresConsent": true,
       "auditRequired": true
     },
     "preconditions": [
       {"type": "Constraint", "expr": "input > 0"}
     ]
   }
   ```

3. **House-Galaxy Firewall:**
   - Galaxy Universe = public, discoverable, read-only by default
   - House Universe = private, owner-controlled, write-enabled
   - Portal system enforces permission checks
   - TRM routing respects visibility boundaries

4. **Human-in-the-Loop:**
   - Sensitive operations require explicit user consent
   - Audit journal logs all permission escalations
   - Confirmation prompts for cross-boundary operations

**Specification Update:**
- [docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md](../vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) — Add Region 8: Security & Access Control
- [docs/vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md](../vocabulary/THREE_BRAIN_SYSTEM_SPECIFICATION.md) — Formalize House-Galaxy boundary enforcement

**Timeline:** Phase K (Q2 2026)

---

### 2. Procedural Canonicalization (C14N) and Cryptographic Trust

**Status:** Project proposal; no participant endorsement is asserted.

**The Challenge:**
Procedural programs need a defined canonical encoding before their signatures can be checked reproducibly.
This draft establishes no compression result.

**K3D Development Goal:**
Develop deterministic ways to canonicalize RPN procedural programs to support **Verifiable Procedural Credentials**.

**Implementation Requirements:**

1. **Procedural C14N Algorithm:**
   - Deterministic serialization of RPN programs (canonical order)
   - Stable JSON-LD context resolution (no ambiguity)
   - Hash-based content addressing (symlink integrity)
   - Reproducible across platforms (IEEE 754 floating-point handling)

2. **Hypothetical credential use case:**
   ```json
   {
     "@context": [
       "https://pm-kr.org/contexts/procedural.jsonld",
       "https://w3id.org/did/v1",
       "https://w3id.org/vc/v1"
     ],
     "@type": "VerifiableProceduralCredential",
     "id": "did:example:carbon_credit_verification_v2",
     "issuer": "did:example:publisher",
     "issuanceDate": "2026-02-26T12:00:00Z",
     "credentialSubject": {
       "@type": "ProceduralProgram",
       "program": [
         "LOAD", "satellite_data",
         "CALL", "did:example:forest_analysis_v2",
         "CALL", "did:example:carbon_calculation",
         "VERIFY", "stakeholder_signatures",
         "ISSUE", "carbon_credit"
       ],
       "preconditions": [
         {"type": "DataAvailability", "source": "Sentinel-2"}
       ]
     },
     "proof": {
       "type": "Ed25519Signature2020",
       "created": "2026-02-26T12:00:00Z",
       "proofPurpose": "assertionMethod",
       "verificationMethod": "did:example:publisher#key-1",
       "proofValue": "z3j4k5L6m7N8..."
     }
   }
   ```

3. **Cryptographic Verification:**
   - Sign canonical RPN programs with Ed25519/ECDSA
   - Verify signatures before execution (trust chain)
   - Support W3C DIDs (Decentralized Identifiers)
   - Integrate with W3C Verifiable Credentials data model

4. **Provenance Audit:**
   - Every procedural program has cryptographic provenance
   - Audit journal stores signature verification logs
   - Reproducible execution traces (same input → same output + same trace)

**Specification Update:**
- New: `docs/W3C/PM_KR_CANONICALIZATION_SPEC.md` — Procedural C14N algorithm
- New: `docs/W3C/PM_KR_VERIFIABLE_CREDENTIALS.md` — Integration with W3C VCs
- Update: [docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md](../vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) — Add cryptographic audit layer

**Timeline:** Phase L (Q3 2026)

---

### 3. Enabling AI Planning (STRIPS and PDDL)

**Status:** Project proposal; no participant endorsement is asserted.

**The Challenge:**
The proposed planning layer would compose procedures into workflows.
This remains a design topic, not an implemented planning capability established here.

**K3D Development Goal:**
Expand K3D's procedural metadata to include **preconditions and postconditions (effects)**, inspired by AI planning languages like STRIPS and PDDL.

**Implementation Requirements:**

1. **Planning Metadata Schema:**
   ```json
   {
     "@context": "https://pm-kr.org/contexts/planning.jsonld",
     "@type": "ProceduralProgram",
     "id": "galaxy:reality:physics:collision",
     "program": [
       "LOAD", "body1", "LOAD", "body2",
       "VELOCITY", "MASS", "MOMENTUM",
       "ELASTIC_COLLISION", "UPDATE_VELOCITIES"
     ],
     "preconditions": [
       {"type": "Assertion", "expr": "body1.mass > 0"},
       {"type": "Assertion", "expr": "body2.mass > 0"},
       {"type": "Constraint", "expr": "distance(body1, body2) < collision_threshold"}
     ],
     "effects": [
       {"type": "StateChange", "target": "body1.velocity", "operator": "update"},
       {"type": "StateChange", "target": "body2.velocity", "operator": "update"},
       {"type": "Assertion", "expr": "momentum_conserved(before, after)"}
     ]
   }
   ```

2. **TRM Planning Layer:**
   - TRM queries Galaxy for procedures matching desired effects
   - Constructs multi-step workflows (procedure chaining)
   - Validates preconditions before execution
   - Verifies postconditions after execution
   - Backtracking on constraint violations

3. **STRIPS/PDDL Compatibility:**
   - Bidirectional translation: STRIPS ↔ PM-KR
   - PDDL domain files → Galaxy population
   - PM-KR procedures → PDDL actions (for external planners)

4. **Use Case Examples:**
   - **Robotics:** Plan manipulation tasks (grasp → move → place)
   - **Math:** Plan proof strategies (axiom selection → theorem chaining)
   - **Chemistry:** Plan synthesis routes (precursor selection → reaction chaining)

**Specification Update:**
- New: `docs/W3C/PM_KR_PLANNING_METADATA.md` — STRIPS/PDDL alignment
- Update: [docs/vocabulary/MATH_CORE_SPECIFICATION.md](../vocabulary/MATH_CORE_SPECIFICATION.md) — Add planning metadata to math procedures
- Update: [docs/vocabulary/REALITY_ENABLER_SPECIFICATION.md](../vocabulary/REALITY_ENABLER_SPECIFICATION.md) — Add preconditions/effects to physics systems

**Timeline:** Phase M (Q3-Q4 2026)

---

### 4. Prioritizing Interoperability and Tooling

**Driver:** W3C PM-KR Community consensus

**The Challenge:**
PM-KR cannot exist in a vacuum — it must complement existing Semantic Web technologies (RDF, OWL, JSON-LD) and integrate with enterprise workflows.

**K3D Development Goal:**
Build bidirectional translation tools so legacy RDF graphs can be compressed into PM-KR procedural nodes, and PM-KR nodes can be published as valid JSON-LD for external discovery.

**Implementation Requirements:**

1. **CLI Converters:**
   - `rdf2pmkr` — Ingest RDF/OWL ontologies, convert to Galaxy entries
   - `pmkr2jsonld` — Export Galaxy Universe as browsable JSON-LD
   - `pmkr2rdf` — Flatten procedural programs back to RDF triples (for interop)
   - `pmkr-validate` — Conformance testing (Level A/B/C)

2. **Example Workflow:**
   ```bash
   # Ingest existing knowledge graph
   rdf2pmkr --input schema_org_ontology.rdf --output galaxy_universe.json

   # K3D populates Galaxy Universe with procedural programs
   python scripts/enhance_galaxy.py

   # Export for external systems (search engines, triple stores)
   pmkr2jsonld --input galaxy_universe.json --output public_galaxy.jsonld

   # Validate conformance
   pmkr-validate --level B --input galaxy_universe.json
   ```

3. **JSON-LD Context Publishing:**
   - Host canonical contexts at `https://pm-kr.org/contexts/`
   - Version contexts (e.g., `procedural-v1.jsonld`, `planning-v2.jsonld`)
   - Provide Schema.org alignment mappings

4. **Enterprise Integration:**
   - SPARQL endpoint for Galaxy queries (read-only)
   - GraphQL API for procedural program discovery
   - REST API for TRM routing (inference-as-a-service)

**Specification Update:**
- New: `docs/W3C/PM_KR_CLI_TOOLS.md` — Converter specifications
- Update: [docs/W3C/PM_KR_INTEROPERABILITY_GUIDE.md](PM_KR_INTEROPERABILITY_GUIDE.md) — Add tooling examples

**Timeline:** Phase N (Q4 2026)

---

### 5. Lowering Cognitive Load for Developer Adoption

**Status:** Project proposal; no participant endorsement is asserted.

**Key Insight:**
The proposed documentation should introduce the user problem before unfamiliar project terminology.

**The Challenge:**
K3D's architecture is revolutionary but unfamiliar. Developers need **familiar analogies** to onboard quickly.

**K3D Development Goal:**
Map K3D concepts to familiar industry paradigms — file systems, object-oriented programming, graph databases, spatial computing.

**Implementation Requirements:**

1. **Documentation Analogies:**

   | K3D Concept | Familiar Analogy | Why It Helps |
   |-------------|------------------|--------------|
   | **House Universe** | File system directories | "Your private workspace, like ~/Documents" |
   | **Rooms** | Folders | "Organized by purpose, like /project/src" |
   | **Galaxy Universe** | Public npm registry | "Canonical modules, everyone references" |
   | **Procedural Programs (RPN)** | Functions/methods | "Executable logic, composable" |
   | **Symlink-Style Refs** | Object inheritance | "Reuse, don't duplicate" |
   | **TRM Navigation** | Gremlin graph queries | "Traverse, filter, combine nodes" |
   | **PTX Kernels** | LLVM IR / WebAssembly | "Low-level, portable, fast" |

2. **API Familiarity:**
   ```python
   # File system analogy
   house = k3d.House.open("~/my_house")
   room = house.get_room("math_workspace")
   node = room.get_node("pythagorean_theorem")

   # Graph database analogy
   galaxy = k3d.Galaxy.connect()
   results = galaxy.query("MATCH (n:Math) WHERE n.topic = 'algebra' RETURN n")

   # Object-oriented analogy
   class MathProcedure(k3d.ProceduralProgram):
       def __init__(self, rpn_program):
           self.program = rpn_program

       def execute(self, inputs):
           return k3d.Cranium.execute(self.program, inputs)
   ```

3. **Visual Documentation:**
   - Interactive tutorials (ThreeJS Galaxy viewer)
   - Side-by-side code comparisons (traditional vs. K3D)
   - Architecture diagrams with familiar labels

4. **Migration Guides:**
   - "From Pandas/NumPy to Galaxy Universe"
   - "From TensorFlow/PyTorch to PTX Kernels"
   - "From Neo4j/ArangoDB to TRM Navigation"

**Specification Update:**
- New: `docs/DEVELOPER_ONBOARDING_GUIDE.md` — Analogies, tutorials, migration paths
- Update: [CLAUDE.md](../../CLAUDE.md) and [CODEX.md](../../CODEX.md) — Add analogy sections

**Timeline:** Phase J (Q2 2026) — Documentation sprint

---

### 6. Achieving "Level C" Conformance (Auditable Production)

**Input:** Project draft [conformance profiles](PM_KR_CONFORMANCE_PROFILES.md).

**The Challenge:**
No conformance level is established here.
Independent verification requires public test artifacts and an agreed evaluation process.

**K3D Development Goal:**
Reach **Level C (Auditable Production)** by finalizing externalized test suites, cryptographic signatures, and provenance audit logs for third-party verification.

**Conformance Levels:**

| Level | Requirements | K3D Status |
|-------|--------------|------------|
| **Level A** | Basic PM-KR implementation (JSON-LD + RPN) | No conformance result established |
| **Level B** | Sovereign execution (PTX-only hot path) | No conformance result established |
| **Level C** | Auditable production (external tests, crypto signatures) | Proposed evaluation work |

**Implementation Requirements:**

1. **Externalized Test Suite:**
   - Independent test runner (not K3D-internal)
   - Downloadable from w3c-cg/pm-kr repo
   - Runs against K3D via public API
   - JSON output for CI/CD integration

2. **Provenance Exporters:**
   ```bash
   # Export cryptographic audit log
   k3d-export-audit --format jsonld --output audit_log.jsonld

   # Verify execution trace
   k3d-verify-trace --input task_123_trace.json --signature ed25519
   ```

3. **Third-Party Validation:**
   - Academia can reproduce benchmarks
   - Industry can audit security (permission checks, signature verification)
   - W3C CG can validate conformance claims

4. **Continuous Integration:**
   - GitHub Actions: Run Level A/B/C tests on every commit
   - Publish conformance badge (like W3C HTML validator)
   - Automated test reports to PM-KR mailing list

**Specification Update:**
- Update: [docs/W3C/PM_KR_CONFORMANCE_PROFILES.md](PM_KR_CONFORMANCE_PROFILES.md) — Add Level C checklist
- New: `docs/W3C/PM_KR_INDEPENDENT_TEST_SUITE.md` — Test runner specification
- New: `scripts/export_audit_log.py` — Provenance exporter

**Timeline:** Phase O (Q4 2026)

---

## Strategic Transformation Summary

The proposed work connects execution control to permission checks, provenance and exchange tests.
These are development topics, not evidence of web-scale operation, interoperability or certification.

---

## Influence Mapping: Expert Contributions

The earlier roster and assigned contributions are withdrawn as evidence because this draft did not cite public records establishing them.
Project credits remain in [ATTRIBUTIONS.md](../../ATTRIBUTIONS.md).
The [explainer](EXPLAINER.md#stakeholder-feedback) describes the public October discussion with source links.

---

## Implementation Phases

These dated phases are the original proposal; they are superseded by the code pause and current specification work.

**Phase J (Q2 2026): Developer UX**
- [ ] Write Developer Onboarding Guide (analogies)
- [ ] Create interactive tutorials (Galaxy viewer)
- [ ] Migration guides (Pandas → Galaxy, Neo4j → TRM)

**Phase K (Q2 2026): Access Control**
- [ ] Implement security metadata schema
- [ ] Build House-Galaxy firewall (permission checks)
- [ ] Add human-in-the-loop confirmation prompts

**Phase L (Q3 2026): Cryptographic Trust**
- [ ] Design Procedural C14N algorithm
- [ ] Integrate W3C DIDs and Verifiable Credentials
- [ ] Build signature verification layer

**Phase M (Q3-Q4 2026): AI Planning**
- [ ] Add STRIPS-style preconditions/effects metadata
- [ ] Build TRM planning layer (workflow generation)
- [ ] Create STRIPS ↔ PM-KR translator

**Phase N (Q4 2026): Interoperability**
- [ ] Build `rdf2pmkr` CLI converter
- [ ] Build `pmkr2jsonld` exporter
- [ ] Host canonical JSON-LD contexts
- [ ] SPARQL endpoint for Galaxy queries

**Phase O (Q4 2026): Level C Conformance**
- [ ] Externalize test suite (w3c-cg/pm-kr repo)
- [ ] Build provenance exporters
- [ ] Third-party validation (academia + industry)
- [ ] Publish conformance badge

---

## Alignment with Existing K3D Architecture

The roadmap refers to the project's proposed procedural execution, shared knowledge and spatial interface designs.
Security metadata, canonicalization, planning, interoperability and conformance artifacts require implementation-specific evidence.
No completion status or current code inventory is claimed here.

---

## Success Metrics

**Historical evaluation targets, not an agreed schedule or measured outcome:**

1. **Security:**
   - 100% of Galaxy procedures have permission metadata
   - House-Galaxy firewall blocks unauthorized access (0% leakage)
   - Human-in-the-loop confirmation on sensitive ops (audit logged)

2. **Cryptographic Trust:**
   - All canonical Galaxy procedures cryptographically signed
   - A credential use case with published consent and provenance requirements
   - Reproducible execution traces (same input → same trace + same signature)

3. **AI Planning:**
   - TRM generates multi-step workflows (at least 3-step chains validated)
   - STRIPS ↔ PM-KR translator functional (bidirectional, no loss)
   - Planning metadata on ≥50% of Reality Galaxy procedures

4. **Interoperability:**
   - `rdf2pmkr` ingests Schema.org ontology (100% coverage)
   - `pmkr2jsonld` exports Galaxy as valid JSON-LD (W3C validator passes)
   - SPARQL endpoint serves ≥1,000 queries/day (external clients)

5. **Developer Adoption:**
   - Onboarding guide reduces "time to first contribution" by 50%
   - ≥10 external developers contribute (tracked via GitHub)
   - ≥5 migration guides published (Pandas, TensorFlow, Neo4j, etc.)

6. **Level C Conformance:**
   - Independent test suite runs on ≥3 different implementations
   - Third-party audit (academia or industry) validates sovereignty claims
   - An independently published conformance assessment against an agreed profile

---

## Conclusion: From Prototype to Technology

The useful next step is a bounded specification and exchange example that other participants can evaluate.
This roadmap supplies no evidence of adoption, external validation or a completed web standard.

**Maintained by:** Daniel Ramos, Knowledge3D.
**Discussion:** [PM-KR Community Group](https://www.w3.org/community/pm-kr/).

---

## References

- [PM-KR Community Group Page](https://www.w3.org/community/pm-kr/)
- [PM-KR Standards Repo](https://github.com/w3c-cg/pm-kr)
- [PM-KR Problem Statement](PM_KR_PROBLEM_STATEMENT.md)
- [PM-KR Conformance Profiles](PM_KR_CONFORMANCE_PROFILES.md)
- [PM-KR Interoperability Guide](PM_KR_INTEROPERABILITY_GUIDE.md)
- [K3D vs State of the Art 2026](K3D_VS_STATE_OF_THE_ART_2026.md)
- [Hyper-Modular Architecture Definition](HYPER_MODULAR_DEFINITION.md)

**END OF PM-KR STRATEGIC STEERING DOCUMENT**
