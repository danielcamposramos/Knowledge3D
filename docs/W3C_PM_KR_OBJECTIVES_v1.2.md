# PM-KR Community Group — Objectives (v1.3)

**Last Updated:** February 28, 2026

---

## Mission

The PM-KR Community Group develops standards for **procedural knowledge representation** that enable both humans and AI systems to consume the same canonical knowledge sources.

**Core insight (Milton Ponson, mandala graph theory):** Nothing is "wrong" with declarative approaches — they're **necessary but insufficient**. PM-KR provides **procedural optimization given declarative foundation**.

**Boundary framework (Christoph Dorn):** Reality is not uniform, containing paradoxes and non-logical choices. PM-KR addresses hard/soft/blurred/broken boundaries at fractal levels, with structural transparency as the safety net for author accountability.

### Triple Foundation

PM-KR provides **procedural knowledge representation** where:
- **Declarative foundation** (Milton Ponson): Semantics, structure, mathematical rigor (mandala graph theory)
- **Procedural execution** (PM-KR): Runnable, renderable, multi-modal
- **Boundary framework** (Christoph Dorn): Hard/soft/blurred/broken boundaries at fractal levels
- **Structural transparency** (Christoph Dorn): Author accountability, safety net for AI systems
- **Reality modeling** (Christoph Dorn): Captures paradoxes, non-logical choices, ignorance (not uniform models)

**The result:** AI systems that are:
1. **Mathematically rigorous** (Milton's mandala graph theory)
2. **Philosophically grounded** (Christoph's boundary framework)
3. **Practically implementable** (reference implementations)
4. **Ethically accountable** (structural transparency = safety net)

---

## Problem Statement

Current knowledge representation systems suffer from massive duplication and fragmentation:

- The same knowledge is duplicated across fonts, embeddings, accessibility metadata, and visual renderings
- Human-readable and machine-readable formats diverge, requiring separate maintenance
- AI training data duplicates knowledge already available in structured forms
- Knowledge updates must be propagated across multiple systems manually

**Result:** Inefficiency, inconsistency, and unsustainability at scale.

---

## Approach: Declarative Foundation + Procedural Execution

**Declarative approaches** (RDF, OWL, JSON-LD) provide the **"know-what"** — foundational understanding (structure, relationships, semantics).

**PM-KR adds the "know-how"** — procedural execution layer that makes declarative knowledge **runnable, renderable, and multi-modal**.

**We're not replacing RDF/OWL/JSON-LD — we're adding the execution layer on top.**

### Core Principles

1. **Dual-Client Contract**: One canonical procedural source serves both humans (readable) and AI systems (executable)
2. **Compositional Architecture**: Knowledge units compose via symlink-style references (no duplication)
3. **Procedural Foundation**: Knowledge is executable programs (like TrueType fonts, mathematical formulas, physics simulations)
4. **Hyper-Modularity**: Atomic knowledge units that combine into complex structures
5. **Explicit Context Rules**: Context-dependent meanings handled via inspectable procedural rules (not implicit neural weights)

### Synergy with Declarative Standards

| **Approach** | **Role** | **Execution** | **Transparency** |
|--------------|----------|---------------|------------------|
| **Declarative (RDF/OWL)** | **Foundation** ("know-what") | ❌ No | ✅ Yes |
| **PM-KR (Procedural)** | **Execution layer** ("know-how") | ✅ Yes | ✅ Yes |
| **Declarative + PM-KR** | **Complete system** | ✅ Yes | ✅ Yes |

---

## Scope

### In Scope

**Core Standards:**
- Procedural knowledge data models
- Composition semantics (symlink references, deduplication)
- Execution semantics (RPN-based procedural programs)
- Context rule semantics (handling context-dependent meanings)
- Conformance levels (minimal, extended, sovereign)

**Domains:**
- Mathematical knowledge (symbols, operators, formulas)
- Spatial knowledge (geometric primitives, transformations)
- Linguistic knowledge (characters, glyphs, typography)
- Educational knowledge (textbooks, curricula, multi-modal rendering)
- Game mechanics (rules, systems, procedural generation)
- Scientific knowledge (protocols, experiments, simulations)

**Relationships with W3C Technologies:**
- JSON-LD (structured data representation)
- RDF/OWL (semantic web integration — PM-KR adds execution layer)
- Verifiable Credentials (knowledge provenance)
- CBOR-LD (compression for efficient transmission)

### Out of Scope

- Proprietary AI training formats (focus: open, standardized)
- Natural language understanding (we provide structured knowledge inputs)
- Inference engines (we define knowledge representation, not reasoning systems)

---

## Deliverables (2026-2027)

### Specifications

1. **PM-KR Core Specification v1.0**
   - Procedural knowledge data model
   - Composition semantics
   - Conformance levels
   - **Target:** Q4 2026

2. **PM-KR Execution Semantics v1.0**
   - RPN-based procedural programs
   - Interpreter requirements
   - Sandbox/security model
   - **Target:** Q1 2027

3. **PM-KR Context Rules Specification v1.0**
   - Context-dependent execution semantics
   - Inheritance and override rules
   - Multi-modal rendering patterns
   - **Target:** Q2 2027

4. **PM-KR JSON-LD Profile v1.0**
   - JSON-LD context definitions
   - Vocabulary terms
   - Canonical serialization rules
   - **Target:** Q2 2027

### Reference Implementation

- **Knowledge3D** (Python/CUDA): Sovereign spatial procedural knowledge system with GPU execution
- Demonstrates: Hyper-modular architecture, 50-90% compression, dual-client contract

### Use Case Documentation

- **Education:** Procedural textbooks for human and AI tutors (MIT OpenCourseWare example)
- **Gaming:** Executable rulebooks for game masters and AI assistants (D&D SRD example)
- **Science:** Procedural experimental protocols (Nature Protocols example)
- **Accessibility:** Multi-modal knowledge rendering (visual, audio, tactile from ONE source)
- **AI Training:** Canonical knowledge sources (Wikipedia procedural KB — query, don't duplicate)

---

## Current Members (18+ as of Feb 28, 2026)

**Chairs:**
- Daniel Campos Ramos (Founder, Knowledge3D Project)
- Milton Ponson (Co-Chair, Mathematical Foundations)

**Notable Organizations:**
- MIT Digital Credentials Consortium
- Huawei Technologies (W3C Advisory Board)
- Digital Bazaar (Manu Sporny, JSON-LD co-creator)
- LinkedIn (Knowledge Graphs)
- University of Brescia, Italy
- Indiana University
- Rensselaer Polytechnic Institute
- INRIA, France
- Cogsonomy

---

## Liaisons with Other W3C Groups

**Active Collaborations:**
- Verifiable Credentials WG: Knowledge provenance and attribution
- JSON-LD WG: Procedural JSON-LD extensions
- Spatial Data on the Web WG: Spatial knowledge representation
- Credentials CG: Academic credentials and structured educational knowledge

**Potential Future Liaisons:**
- Semantic Web Interest Group
- Web Machine Learning WG
- RDF-star WG (nested knowledge structures)

---

## Roadmap

**Q1 2026 (Current):**
- ✅ Community Group launched (Feb 20, 2026)
- ✅ 18+ members recruited
- ✅ Mission statement revised based on community feedback (v1.0 → v1.2)
- 🔄 Initial specification drafts (PM-KR Core v0.1)

**Q2 2026:**
- Publish PM-KR Core Specification v0.5 (draft for community review)
- Gather community feedback (use cases, implementations)
- Establish liaisons with related W3C WGs
- Expand reference implementations (JavaScript, Rust)

**Q3 2026:**
- Publish PM-KR Core Specification v1.0 (candidate)
- Host W3C TPAC breakout session
- Empirical validation studies (compression, performance, accuracy)

**Q4 2026:**
- Finalize PM-KR Core Specification v1.0
- Begin PM-KR Execution Semantics v1.0 draft
- Publish conformance test suite

**2027:**
- PM-KR Execution Semantics v1.0
- PM-KR Context Rules Specification v1.0
- PM-KR JSON-LD Profile v1.0
- Industry adoption (textbook publishers, game companies, AI platforms)

---

## Why PM-KR Matters Now

### Technical Impact

- **Standardize** procedural knowledge representation for AI systems
- **Reduce** knowledge duplication across systems (50-90% compression demonstrated)
- **Enable** dual-client knowledge sources (humans + AI from same source)
- **Complement** declarative standards (add execution layer to RDF/OWL/JSON-LD)

### Societal Impact

- **Education:** Accessible, multi-modal textbooks (visual, audio, tactile)
- **Sustainability:** Reduce AI's carbon footprint via compression (no training data duplication)
- **Accessibility:** Knowledge rendered for diverse human needs (same source)
- **Reproducibility:** Scientific protocols as executable procedures

### Economic Impact

- **Efficiency:** Companies stop duplicating knowledge infrastructure
- **Innovation:** New applications enabled by compositional knowledge
- **Open Standards:** Prevent proprietary lock-in, foster ecosystem growth

---

## How to Participate

**Join:** https://www.w3.org/community/pm-kr/ (no W3C membership required)

**Contribute:**
- Review specifications in GitHub: https://github.com/danielcamposramos/Knowledge3D/tree/main/docs/vocabulary
- Propose use cases from your domain
- Build prototypes in your preferred language
- Participate in discussions: public-pm-kr@w3.org

---

## Contact

**Chairs:**
- Daniel Campos Ramos: capitain_jack@yahoo.com
- Milton Ponson: rwiciamsd@gmail.com

**Mailing List:** public-pm-kr@w3.org

**GitHub:** https://github.com/danielcamposramos/Knowledge3D

---

## Acknowledgments

PM-KR builds on decades of W3C work in semantic web, linked data, and web standards.

**Special thanks:**
- **Dave Raggett** (W3C veteran) — Critical feedback on declarative vs procedural positioning
- **Milton Ponson** (Co-Chair) — Mathematical insight: declarative foundation + procedural optimization synergy (mandala graph theory)
- **Christoph Dorn** (Systems Architect) — Boundary framework: hard/soft/blurred/broken boundaries, structural transparency as safety net for author accountability
- **Manu Sporny** (JSON-LD co-creator) — JSON-LD integration guidance
- **Tim Berners-Lee** — Linked Data principles
- All 18+ early ingressors who joined in the first week

---

**Version:** 1.3 (Triple Foundation: Milton + Christoph + Implementation)
**Last Updated:** February 28, 2026
