# PM-KR Strategic Steering for K3D Development

**Status:** Active Development Roadmap
**Last Updated:** February 26, 2026
**Source:** W3C PM-KR Community Group early ingressor discussions + NotebookLM strategic analysis

---

## Executive Summary

The formation of the W3C Procedural Memory Knowledge Representation (PM-KR) Community Group shifts Knowledge3D's development trajectory from being a standalone, highly advanced prototype into becoming the **foundational reference implementation** for a global web technology.

**Analogy:** If PM-KR is the HTML/CSS specification, K3D is the WebKit browser engine that proves it works.

Based on intense discussions and expert feedback from PM-KR's early ingressors (Manu Sporny, Adam Sobieski, Jonathan DeRouchie, Milton Ponson, Christoph Dorn, ixo.world, and others), K3D's development is now steered by **6 strategic imperatives** that transform it from a cognitive OS into a **secure, cryptographically verifiable, and highly interoperable web technology**.

---

## The 6 Strategic Imperatives

### 1. Formalizing Access Control and Sovereignty Boundaries

**Driver:** Jonathan DeRouchie (persistent memory AI systems expert)

**The Challenge:**
Industry needs strict boundaries between public shared knowledge and private permissioned execution. Current AI systems blur these lines dangerously.

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

**Driver:** Manu Sporny (JSON-LD co-creator, RDF Canonicalization editor)

**The Challenge:**
K3D's 70% compression is validated, but procedural programs need **deterministic canonicalization** so they can be cryptographically signed and verified.

**K3D Development Goal:**
Develop deterministic ways to canonicalize RPN procedural programs to support **Verifiable Procedural Credentials**.

**Implementation Requirements:**

1. **Procedural C14N Algorithm:**
   - Deterministic serialization of RPN programs (canonical order)
   - Stable JSON-LD context resolution (no ambiguity)
   - Hash-based content addressing (symlink integrity)
   - Reproducible across platforms (IEEE 754 floating-point handling)

2. **Example Use Case (ixo.world):**
   ```json
   {
     "@context": [
       "https://pm-kr.org/contexts/procedural.jsonld",
       "https://w3id.org/did/v1",
       "https://w3id.org/vc/v1"
     ],
     "@type": "VerifiableProceduralCredential",
     "id": "did:ixo:carbon_credit_verification_v2",
     "issuer": "did:ixo:shaun-conway",
     "issuanceDate": "2026-02-26T12:00:00Z",
     "credentialSubject": {
       "@type": "ProceduralProgram",
       "program": [
         "LOAD", "satellite_data",
         "CALL", "did:ixo:forest_analysis_v2",
         "CALL", "did:ixo:carbon_calculation",
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
       "verificationMethod": "did:ixo:shaun-conway#key-1",
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

**Driver:** Adam Sobieski (W3C veteran, founded 3 CGs)

**The Challenge:**
Procedural knowledge isn't just for execution — AI agents need to **plan** with it. K3D currently executes RPN programs but doesn't enable AI to reason *over* procedures to generate complex workflows.

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

**Driver:** Jonathan DeRouchie (developer experience expert)

**Key Insight:**
*"Users are more likely to adopt a framework that uses or simplifies existing language, concept or structure."*

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

**Driver:** W3C PM-KR Conformance Profiles (see [PM_KR_CONFORMANCE_PROFILES.md](PM_KR_CONFORMANCE_PROFILES.md))

**The Challenge:**
K3D is currently "Provisional Level B+" — it proves zero-dependency GPU sovereignty but lacks externalized test suites and independent audit tooling.

**K3D Development Goal:**
Reach **Level C (Auditable Production)** by finalizing externalized test suites, cryptographic signatures, and provenance audit logs for third-party verification.

**Conformance Levels:**

| Level | Requirements | K3D Status |
|-------|--------------|------------|
| **Level A** | Basic PM-KR implementation (JSON-LD + RPN) | ✅ Complete |
| **Level B** | Sovereign execution (PTX-only hot path) | ✅ Complete |
| **Level C** | Auditable production (external tests, crypto signatures) | 🚧 In Progress (50%) |

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

| Before PM-KR | After PM-KR Steering |
|--------------|----------------------|
| **Standalone prototype** | **W3C reference implementation** |
| **Cognitive OS** | **Web standard (like WebKit for HTML)** |
| **Sovereignty focus** | **Sovereignty + Security + Provenance** |
| **Internal execution** | **Cryptographically verifiable execution** |
| **K3D-specific** | **Interoperable with Semantic Web** |
| **Developer-unfriendly** | **Familiar analogies (file system, OOP, graphs)** |
| **Self-validated** | **Third-party auditable (Level C)** |

---

## Influence Mapping: Expert Contributions

| Expert | Primary Contribution | K3D Impact |
|--------|---------------------|------------|
| **Manu Sporny** (JSON-LD co-creator) | Procedural C14N + Verifiable Credentials | Cryptographic trust layer |
| **Adam Sobieski** (W3C veteran) | STRIPS/PDDL planning metadata | AI reasoning over procedures |
| **Jonathan DeRouchie** (Persistent memory AI) | Access control + Developer UX | Security boundaries + analogies |
| **Milton Ponson** (Gödelian KR) | Domains of discourse formalization | House-Galaxy separation |
| **Christoph Dorn** (K3D main contributor, PM-KR group member) | "Semantic gravity cohered by meaning"; TerraVision heritage | Spatial navigation + force paradigm |
| **ixo.world** (Shaun Conway) | W3C DIDs + Verifiable Claims | Blockchain-backed provenance |
| **Nitin Pasumarthy** (LinkedIn GNNs) | Production KG scaling patterns | Galaxy Universe optimization |
| **Marko Rodriguez** (Apache TinkerPop) | Gremlin graph query analogies | Developer onboarding |

---

## Implementation Phases

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

**Already Implemented (Validated):**
1. ✅ **Hyper-Modular Architecture** (7 levels, 70% compression)
2. ✅ **Sovereign PTX Execution** (45+ kernels, zero external frameworks)
3. ✅ **Three-Brain System** (Cranium, Galaxy, House)
4. ✅ **Unified Persistent Memory** (Knowledgeverse 7-region substrate)
5. ✅ **Dual-Client Contract** (Human visual + AI executable)

**Now Adding (PM-KR Strategic Steering):**
6. 🚧 **Security & Access Control** (House-Galaxy firewall, permission metadata)
7. 🚧 **Cryptographic Provenance** (Procedural C14N, Verifiable Credentials)
8. 🚧 **AI Planning Layer** (STRIPS/PDDL metadata, TRM workflow generation)
9. 🚧 **Interoperability Tooling** (RDF converters, JSON-LD export)
10. 🚧 **Developer Familiarity** (Analogies, migration guides)
11. 🚧 **Level C Conformance** (External tests, audit logs)

---

## Success Metrics

**By Q4 2026, K3D should demonstrate:**

1. **Security:**
   - 100% of Galaxy procedures have permission metadata
   - House-Galaxy firewall blocks unauthorized access (0% leakage)
   - Human-in-the-loop confirmation on sensitive ops (audit logged)

2. **Cryptographic Trust:**
   - All canonical Galaxy procedures cryptographically signed
   - Verifiable Credential integration (at least 1 real-world use case with ixo.world)
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
   - W3C PM-KR CG issues "Level C Conformant" badge

---

## Conclusion: From Prototype to Technology

The W3C PM-KR Community Group transforms K3D's mission:

**Before:** Build the world's most advanced spatial AI cognitive OS.
**Now:** Build the world's most advanced spatial AI cognitive OS **AND** prove it can become a global web technology.

**The stakes are higher. The validation is external. The impact is permanent.**

If K3D succeeds in implementing these 6 strategic imperatives, it won't just be a research project — it will be **the reference implementation that proves procedural knowledge representation works at web scale**.

**This is K3D's path from revolutionary prototype to foundational infrastructure.**

---

**Last Updated:** February 26, 2026
**Maintained by:** Daniel Ramos (K3D architect), W3C PM-KR Community Group
**Feedback:** public-pm-kr@w3.org

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
