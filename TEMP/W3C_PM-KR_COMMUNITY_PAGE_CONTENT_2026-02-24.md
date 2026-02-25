# Procedural Memory Knowledge Representation (PM-KR) Community Group

## Mission

The PM-KR Community Group develops standards for **procedural knowledge representation** that enable both humans and AI systems to consume the same canonical knowledge sources. Our work addresses knowledge duplication, fragmentation, and the need for executable, compositional knowledge structures in the age of AI.

## Problem Statement

Current knowledge representation systems suffer from massive duplication and fragmentation:

- **The same knowledge** (e.g., a Unicode character, mathematical symbol, spatial concept) is duplicated across fonts, embeddings, accessibility metadata, and visual renderings
- **Human-readable** and **machine-readable** formats diverge, requiring separate maintenance
- **AI training data** duplicates knowledge already available in structured forms
- **Knowledge updates** must be propagated across multiple systems manually

**Result:** Inefficiency, inconsistency, and unsustainability at scale.

## Approach

PM-KR proposes a paradigm shift: **store knowledge once as executable procedures, reference via composition**.

### Core Principles

1. **Dual-Client Contract**: One canonical procedural source serves both humans (readable) and AI systems (executable)
2. **Compositional Architecture**: Knowledge units compose via symlink-style references (no duplication)
3. **Procedural Foundation**: Knowledge is executable programs (like font programs, mathematical formula definitions)
4. **Hyper-Modularity**: Atomic knowledge units that combine into complex structures

### Conceptual Analogy

Think of how **TrueType fonts** work:
- One `.ttf` file contains procedural glyph definitions (Bézier curves, hinting programs)
- Renders at any size/weight/style without separate files
- All applications reference the same font file

PM-KR extends this principle to **all knowledge domains**: math symbols, spatial concepts, procedural rules, educational content, game mechanics, etc.

## Scope

### In Scope

**Core Standards:**
- Procedural knowledge data models (JSON-LD based)
- Composition semantics (symlink references, deduplication)
- Execution semantics (RPN-based procedural programs)
- Conformance levels (minimal, extended, sovereign)

**Domains:**
- Mathematical knowledge (symbols, operators, formulas)
- Spatial knowledge (geometric primitives, transformations)
- Linguistic knowledge (characters, glyphs, typography)
- Educational knowledge (textbooks, curricula)
- Game mechanics (rules, systems, procedural generation)
- Scientific knowledge (protocols, experiments, simulations)

**Relationships with W3C Technologies:**
- JSON-LD (structured data representation)
- RDF/OWL (semantic web integration)
- Verifiable Credentials (knowledge provenance)
- CBOR-LD (compression for efficient transmission)

### Out of Scope

- Proprietary AI training formats (our focus: open, standardized)
- Natural language understanding (we provide structured knowledge inputs)
- Inference engines (we define knowledge representation, not reasoning systems)

## Deliverables

### Specifications (Target: 2026-2027)

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

3. **PM-KR JSON-LD Profile v1.0**
   - JSON-LD context definitions
   - Vocabulary terms
   - Canonical serialization rules
   - **Target:** Q2 2027

### Reference Implementation

- **Knowledge3D** (Python/CUDA): Sovereign spatial procedural knowledge system with GPU execution

### Use Case Documentation

- Education: Procedural textbooks for human and AI tutors
- Games: Executable rulebooks for game masters and AI assistants
- Science: Procedural experimental protocols
- Accessibility: Multi-modal knowledge rendering (visual, audio, tactile)

## Current Members (as of Feb 24, 2026)

**Chairs:**
- Daniel Campos Ramos (Founder, Knowledge3D Project)
- Milton Ponson (Co-Chair, Mathematical Foundations)

**Notable Participants (18+ members):**

**Organizations:**
- MIT Digital Credentials Consortium (Alex Higuera, Software Engineer)
- Huawei Technologies (Wei Ding, W3C Advisory Board, Standards & IP Director)
- Digital Bazaar (Manu Sporny, Co-creator of JSON-LD)
- LinkedIn (contributors in Knowledge Graphs)
- University of Brescia, Italy (Anisa Rula, Knowledge Graphs expert)
- Indiana University (Damir Cavar, Computational Linguistics)
- Rensselaer Polytechnic Institute (Henrique Santos, DARPA Machine Common Sense)
- INRIA, France (research contributors)
- Cogsonomy (Xavier Aimé, 25+ years Knowledge Engineering)

**Individual Contributors:**
- Adam Sobieski (W3C veteran, 10+ years experience)
- Hanna Abi Akl (Knowledge representation research)
- Charles Waweru (Semantic web technologies)
- Manh Thanh Le (Data modeling)
- Paul Murdock (Knowledge systems)
- Dave Richardson (Spectacular Voyage LLC)
- Saatvika Sathi (AI/ML research)
- [Additional members joining daily]

## Liaisons with Other W3C Groups

**Existing Collaborations:**
- **Verifiable Credentials WG**: Knowledge provenance and attribution
- **JSON-LD WG**: Procedural JSON-LD extensions
- **Spatial Data on the Web WG**: Spatial knowledge representation
- **Credentials CG**: Academic credentials and structured educational knowledge

**Potential Future Liaisons:**
- Semantic Web Interest Group
- Web Machine Learning WG
- RDF-star WG (nested knowledge structures)

## Resources

**GitHub Repository:**
- https://github.com/danielcamposramos/Knowledge3D
- Specifications: `/docs/vocabulary/`
- Reference implementations: `/knowledge3d/`

**Mailing List:**
- public-pm-kr@w3.org
- Archives: https://lists.w3.org/Archives/Public/public-pm-kr/

**Community Page:**
- https://www.w3.org/community/pm-kr/

**NotebookLM Research Space:**
- https://notebooklm.google.com/notebook/1bd10bda-8900-4c41-931e-c9ec67ac865f
- (Collaborative research and discussion)

## How to Participate

### Joining the Group

1. Visit: https://www.w3.org/community/pm-kr/
2. Click "Join this group"
3. No W3C membership required (Community Group is open to all)

### Contributing

**Ways to contribute:**
- **Use case documentation**: Share your domain's knowledge representation needs
- **Implementation**: Build prototypes in your preferred language
- **Specification feedback**: Review drafts, suggest improvements
- **Research**: Empirical validation, mathematical proofs, performance benchmarks

**Getting started:**
1. Introduce yourself on the mailing list (public-pm-kr@w3.org)
2. Review specifications in GitHub (`/docs/vocabulary/`)
3. Try the reference implementations (Knowledge3D, FangKnight, etc.)
4. Join discussions, propose ideas, contribute code

### Communication Channels

**Mailing List (primary):**
- public-pm-kr@w3.org
- For specification discussions, proposals, community updates

**GitHub Issues:**
- Technical discussions, implementation questions
- Specification clarifications

## Background: Why PM-KR Matters Now

### The AI Knowledge Gap

Current AI systems face a fundamental challenge:
- **Training data** = terabytes of unstructured text
- **Structured knowledge** = locked in silos (databases, APIs, embeddings)
- **Human knowledge** ≠ **machine knowledge** (separate formats, separate maintenance)

**PM-KR addresses this** by providing a canonical procedural layer that both humans and AI systems consume.

### Sustainability Angle

**Carbon impact of knowledge duplication:**
- AI training datasets duplicate Wikipedia, textbooks, documentation
- Each company trains on the same knowledge independently
- Storage + compute = massive carbon footprint

**PM-KR's contribution:**
- 1 canonical source → N systems reference it (no duplication)
- Compression (symlinks, deduplication) = 50-90% size reduction
- Sovereign execution (local GPU, no cloud API calls) = energy efficiency

### Real-World Validation

**PM-KR is grounded in empirical results:**
- Knowledge3D project demonstrates hyper-modular architecture
- Compression benchmarks show 50-90% storage reduction
- Working prototype prove dual-client viability
- MIT, Huawei, and JSON-LD co-creator joined within 4 days (Feb 20-24, 2026)

## Roadmap (2026-2027)

**Q1 2026 (Current):**
- ✅ Community Group launched (Feb 20, 2026)
- ✅ 18+ members recruited (MIT, Huawei, Manu Sporny, etc.)
- 🔄 Initial specification drafts (PM-KR Core v0.1)
- 🔄 Reference implementations (Knowledge3D, FangKnight)

**Q2 2026:**
- Publish PM-KR Core Specification v0.5 (draft)
- Gather community feedback (use cases, implementations)
- Establish liaisons with related W3C WGs
- Expand reference implementations (JavaScript, Rust)

**Q3 2026:**
- Publish PM-KR Core Specification v1.0 (candidate)
- Host W3C TPAC breakout session (community outreach)
- Empirical validation studies (compression, performance, accuracy)
- Industry adoption outreach (education, gaming, enterprise)

**Q4 2026:**
- Finalize PM-KR Core Specification v1.0
- Begin PM-KR Execution Semantics v1.0 draft
- Publish conformance test suite
- Celebrate first anniversary! 🎉

**2027 and Beyond:**
- PM-KR Execution Semantics v1.0
- PM-KR JSON-LD Profile v1.0
- Industry adoption (textbook publishers, game companies, AI platforms)
- Integration with W3C standards (Verifiable Credentials, RDF-star)

## Sustainability and Impact Goals

### Technical Impact

- **Standardize** procedural knowledge representation for AI systems
- **Reduce** knowledge duplication across systems (50-90% compression)
- **Enable** dual-client knowledge sources (humans + AI from same source)

### Societal Impact

- **Education**: Accessible, multi-modal textbooks (visual, audio, tactile)
- **Sustainability**: Reduce AI's carbon footprint via compression
- **Accessibility**: Knowledge rendered for diverse human needs

### Economic Impact

- **Efficiency**: Companies stop duplicating knowledge infrastructure
- **Innovation**: New applications enabled by compositional knowledge
- **Open Standards**: Prevent proprietary lock-in, foster ecosystem growth

## Contact

**Chairs:**
- Daniel Campos Ramos: capitain_jack@yahoo.com, daniel@echosystems.ai
- Milton Ponson: [contact via mailing list]

**Mailing List:**
- public-pm-kr@w3.org

**GitHub:**
- https://github.com/danielcamposramos/Knowledge3D

## Acknowledgments

PM-KR builds on decades of W3C work in semantic web, linked data, and web standards. Special thanks to:
- Manu Sporny (JSON-LD co-creator)
- Tim Berners-Lee (Linked Data principles)
- JSON-LD Working Group
- Verifiable Credentials Community
- All 18+ founding members who joined in the first 4 days

**Let's build the future of knowledge representation together.** 🚀

**Last Updated:** February 24, 2026
**Version:** 1.0