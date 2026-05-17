# Procedural Memory Knowledge Representation (PM-KR) Community Group Charter

**Status:** Active
**Founded:** February 24, 2026
**W3C Page:** https://www.w3.org/community/pm-kr/
**Mailing List:** public-pm-kr@w3.org

---

## Grounding Note

This charter is grounded in repository sources under `docs/` and `docs/w3c-specifications/`.
Quantitative and membership statements in this file are sourced from:
- `docs/W3C_PM_KR_COMMUNITY_GROUP_MISSION.md`
- `docs/W3C_PM_KR_OBJECTIVES_v1.2.md`
- `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md`
- `docs/w3c-specifications/workshop/phase1-data-model/spec-draft.md`

---

## Mission Statement

The Procedural Memory Knowledge Representation (PM-KR) Community Group develops a knowledge representation paradigm where **knowledge is stored once as executable procedures** (like font programs or mathematical formula definitions) and **referenced via symlink-style composition**, enabling both humans and AI systems to consume the same procedural source.

---

## Problem Statement

Current knowledge representation systems suffer from massive duplication and fragmentation:

**Example: The same knowledge element (e.g., a Unicode character, mathematical symbol, or spatial concept) is duplicated across:**
- Font files (Bézier curves in .ttf/.otf)
- AI embeddings (semantic vectors)
- Accessibility metadata (screen reader pronunciations)
- Visual renderings (rasterized bitmaps)
- Documentation (character descriptions)

**Result:** 6+ redundant copies of the same knowledge in incompatible formats.

**Problems:**
1. **Maintenance burden**: Updating one representation requires updating all copies
2. **Inconsistency**: Copies diverge over time (font glyph ≠ accessibility description)
3. **Storage waste**: Gigabytes of redundant data
4. **Performance**: Re-computing same knowledge repeatedly
5. **Security**: Attack surface multiplied across representations
6. **Licensing**: Unclear which representation governs rights

---

## Scope of Work

The PM-KR Community Group will study and develop specifications for:

### 1. **Data Models for Procedural Knowledge Representation**
- Procedural programs (executable, deterministic, reproducible)
- Metadata clusters (semantic meaning, language-specific context)
- Galaxy Universe (multi-modal workspace structure)
- Symlink patterns (reference-based composition, avoiding duplication)

### 2. **Execution Semantics**
- RPN-based execution (stack-based, deterministic)
- Conformance levels (Read, Execute, Learn)
- Runtime validation and boundary contracts

### 3. **Integration with W3C Technologies**
- RDF/OWL/JSON-LD interoperability
- Accessibility (WCAG compliance, pronunciation metadata)
- Internationalization (language-specific metadata)

### 4. **Conformance and Testing**
- Level 0: Read-only (parsing & schema validation)
- Level 1: Execution (procedural program execution)
- Level 2: Learning (adaptive reasoning)

---

## Expected Deliverables

Subject to group consensus, potential outputs include:

1. **Data Model Specification** (Phase 1)
   - Core vocabulary for procedural knowledge representation
   - JSON serialization format
   - Schema validation rules

2. **Execution Semantics Specification** (Phase 2)
   - RPN execution model
   - Runtime behavior definitions
   - Boundary contract semantics

3. **Conformance Specification** (Phase 3)
   - Conformance levels (0, 1, 2)
   - Test suites for each level
   - Implementation validation

4. **Integration Specification** (Phase 4)
   - RDF/OWL/JSON-LD mappings
   - W3C technology interoperability
   - Cross-CG collaboration patterns

5. **Use Case Documentation**
   - Real-world applications
   - Implementation examples
   - Performance benchmarks

---

## Design Principles

### 1. **Single Source of Truth**
Knowledge is stored once and referenced, not duplicated.

### 2. **Dual-Client Contract**
Humans and AI systems consume identical procedural data (Form + Meaning), ensuring transparency and verifiability.

### 3. **Implementation Neutrality**
Specifications define data models and semantics, not hardware or runtime requirements.

### 4. **Multi-Modal Support**
Unified workspace for text, visual, audio, physics, and other modalities.

### 5. **Procedural Foundation**
Knowledge encoded as executable programs (RPN) + metadata, not static data blobs.

---

## Relationship to Prior Work

This initiative draws inspiration from the Knowledge3D (K3D) research project by Daniel Ramos (EchoSystems AI Studios), which demonstrated:
- Procedural font representation (200:1 compression vs traditional formats; see Carbon Blueprint)
- Galaxy Universe architecture (multi-modal workspace; see PM-KR mission/objectives docs)
- RPN-based execution (microsecond-scale targets documented in K3D specs)
- 12 Gt CO₂ projected carbon savings over 10 years (scenario projection in Carbon Blueprint)

**Important:** While K3D research informs the group's direction, it will **not become a formal deliverable**. The group's specifications will be implementation-neutral and developed through community consensus.

---

## Participation

**Who can join:**
- W3C account required (free)
- W3C Membership NOT required for Community Groups

**Mailing list:** public-pm-kr@w3.org

**How to join:**
1. Create W3C account: https://www.w3.org/accounts/request
2. Join PM-KR CG: https://www.w3.org/community/pm-kr/join

---

## Chairs

- **Daniel Ramos** (EchoSystems AI Studios)
- **Milton Ponson** (Rainbow Warriors Core Foundation)

---

## Contributors (Initial Supporters)

- Adam Sobieski
- Milton Ponson
- Nitin Pasumarthy
- Jonathan DeRouchie
- Hanna Abi Akl
- Daniel Ramos

**Current Members (repository snapshot, Feb 28, 2026):** 18+
- Rensselaer Polytechnic Institute
- University of Brescia
- NLP (Indiana University)
- Digital Credentials Consortium
- Spectacular Voyage LLC
- University of the Fraser Valley
- Informatique et en Automatique (Inria)
- Blockchain Commons (Christopher Allen)
- Rainbow Warriors Core Foundation (Milton Ponson)
- Stream44.Studio (Christoph Dorn)
- Huawei (Wei Ding)
- And others

---

## Timeline

**Public comment period:** Through March 2026
**CG launch status:** Active since February 24, 2026
**First deliverables:** Data Model Specification (Phase 1) — Target: April 30, 2026

---

## Cross-CG Collaboration

PM-KR collaborates with:
- **CogAI** (Cognitive Architecture) — Dave Raggett (multimodal reasoning)
- **AIKR** (AI Knowledge Representation) — Paola Di Maio (trust/safety/interoperability)
- **WebML** (Web Machine Learning) — Anssi Kostiainen/Intel (lightweight inference)
- **Sustainable Web IG** — Tzviya Siegman (carbon impact, Web Sustainability Guidelines)

---

## References

- **PM-KR Page:** https://www.w3.org/community/pm-kr/
- **GitHub:** https://github.com/danielcamposramos/Knowledge3D
- **Carbon Blueprint:** [CARBON_BLUEPRINT_10_YEAR_PROJECTION.md](../../CARBON_BLUEPRINT_10_YEAR_PROJECTION.md)
- **Phase 1 Spec (Draft):** [spec-draft.md](../workshop/phase1-data-model/spec-draft.md)

---

**Sources:**
- [Proposed Group: PM-KR Community Group](https://www.w3.org/community/blog/2026/02/20/proposed-group-procedural-memory-knowledge-representation-community-group/)
- [PM-KR Community Group Page](https://www.w3.org/community/pm-kr/)
- [Call for Participation](https://www.w3.org/community/pm-kr/2026/02/23/call-for-participation-in-procedural-memory-knowledge-representation-community-group/)
- [Repository Mission Snapshot](../../W3C_PM_KR_COMMUNITY_GROUP_MISSION.md)
- [Repository Objectives Snapshot](../../W3C_PM_KR_OBJECTIVES_v1.2.md)

---

**Last Updated:** March 3, 2026
**Status:** Active Community Group
**Phase 1 Artifact:** [spec-draft.md](../workshop/phase1-data-model/spec-draft.md)
