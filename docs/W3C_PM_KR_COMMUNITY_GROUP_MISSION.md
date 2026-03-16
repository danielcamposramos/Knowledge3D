# PM-KR Community Group — Mission Statement

**Last Updated:** February 28, 2026
**Version:** 1.1 (Revised based on community feedback)

---

## Mission

The PM-KR Community Group develops standards for **procedural knowledge representation** that enable both humans and AI systems to consume the same canonical knowledge sources. Our work addresses knowledge duplication, fragmentation, and the need for executable, compositional knowledge structures in the age of AI.

---

## Problem Statement

Current knowledge representation systems suffer from massive duplication and fragmentation:

- **The same knowledge** (e.g., a Unicode character, mathematical symbol, spatial concept) is duplicated across fonts, embeddings, accessibility metadata, and visual renderings
- **Human-readable** and **machine-readable** formats diverge, requiring separate maintenance
- **AI training data** duplicates knowledge already available in structured forms
- **Knowledge updates** must be propagated across multiple systems manually

**Result:** Inefficiency, inconsistency, and unsustainability at scale.

---

## Why Procedural? Building on Declarative Foundations

**Question:** Why is a procedural approach necessary? What's wrong with declarative approaches?

**Short answer:** Nothing is "wrong" with declarative approaches — they're **necessary but insufficient**. PM-KR provides **procedural optimization given declarative foundation** (insight from PM-KR Co-Chair Milton Ponson's mandala graph theory).

### The Declarative Foundation + Procedural Execution Synergy

**Declarative approaches** (RDF, OWL, JSON-LD metadata) excel at **describing relationships** — the **"know-what"** (facts, concepts, structure). They form the **foundational understanding** of the environment.

**What declarative approaches provide:**
- ✅ Explicit relationships (e.g., `:integral math:relatedTo :summation`)
- ✅ Semantic interoperability (common vocabularies, ontologies)
- ✅ Transparency (human-readable triples, inspectable graphs)

**What declarative approaches don't provide:**
- ❌ Execution logic: Declaring "a chair is furniture" doesn't tell you HOW to render a chair at different scales, orientations, or styles
- ❌ Context-dependent behavior: The symbol "∫" means different things in calculus (integration), physics (work calculation), and probability (expectation) — declaring these relationships doesn't provide the EXECUTION logic needed to compute results
- ❌ Multi-modal realization: Declarative systems require separate descriptions for visual rendering, audio pronunciation, tactile representation, and computational execution

**PM-KR's contribution:** Add **procedural execution layer** on top of declarative foundation — the **"know-how"** (methods to act upon that environment)

**Example: Mathematical Symbol "∫" (Integral)**

**Declarative approach (RDF/OWL):**
```turtle
:integral a math:Symbol ;
    rdfs:label "integral" ;
    schema:description "Represents integration in calculus" ;
    math:relatedTo :summation, :area_under_curve .
```

**Problem:** This describes WHAT the symbol is, but not HOW to:
- Render it visually (glyph shape, size, positioning)
- Execute it computationally (integration algorithm)
- Pronounce it (screen readers)
- Apply it contextually (definite vs indefinite integrals)

**PM-KR procedural approach:**
```json
{
  "@id": "pm-kr:integral",
  "visual_rpn": ["CURVE", "S_SHAPE", "SCALE", "POSITION"],
  "execution_rpn": ["INTEGRATE", "BOUNDS", "FUNCTION", "COMPUTE"],
  "audio_rpn": ["PRONOUNCE", "integral", "CONTEXT", "calculus"],
  "contexts": {
    "calculus": {"execution": "riemann_sum_rpn"},
    "physics": {"execution": "work_integral_rpn"},
    "probability": {"execution": "expectation_rpn"}
  }
}
```

**Result:** ONE source provides visual rendering, computational execution, audio pronunciation, and context-specific behavior.

### Declarative + Procedural Synergy (The Complete Picture)

Dave Raggett's feedback and Milton Ponson's mandala graph theory highlight a critical insight:

| **Approach** | **Role** | **Execution** | **Transparency** | **Composability** | **Context Handling** |
|--------------|----------|---------------|------------------|-------------------|----------------------|
| **Declarative (RDF/OWL)** | **Foundation** ("know-what") | ❌ No (describes, doesn't execute) | ✅ Transparent (explicit triples) | ⚠️ Limited (static relationships) | ⚠️ Requires manual context modeling |
| **Neural Networks** | Execution (opaque) | ✅ Yes (procedural execution) | ❌ Opaque (black box weights) | ❌ No (monolithic models) | ✅ Learned from data (but unexplainable) |
| **PM-KR (Procedural)** | **Execution layer** ("know-how") | ✅ Yes (RPN programs execute) | ✅ Transparent (inspectable procedures) | ✅ Yes (symlink composition) | ✅ Explicit context rules (inspectable) |
| **Declarative + PM-KR** | **Complete system** | ✅ Yes (procedural on declarative foundation) | ✅ Transparent (both layers inspectable) | ✅ Yes (semantic + procedural composition) | ✅ Declarative semantics + procedural execution |

**Key insight from Milton Ponson (mandala graph theory):** In practice, complex AI systems combine both. **Declarative knowledge forms the foundational understanding** (structure, relationships, semantics), while **procedural knowledge provides the methods to act** upon that environment (execution, rendering, computation).

**PM-KR's positioning:** We're not replacing declarative standards (RDF/OWL/JSON-LD) — we're **adding the execution layer** that makes them **runnable, renderable, and multi-modal**.

### Handling Context-Dependent Meanings

**Question:** What about the likelihood that different terms have subtly different meanings in their respective context of use?

**PM-KR's Answer: Explicit Context Rules**

Rather than relying on implicit neural representations OR forcing manual ontology engineering, PM-KR uses **procedural context rules**:

**Example: "Chair" in Different Contexts**

```json
{
  "@id": "pm-kr:chair",
  "base_visual_rpn": ["SEAT", "BACK", "LEGS", "ASSEMBLE"],
  "contexts": {
    "furniture_catalog": {
      "visual_rpn": ["BASE", "MATERIAL_TEXTURE", "PHOTOREALISTIC"],
      "metadata": ["dimensions", "price", "availability"]
    },
    "architectural_bim": {
      "visual_rpn": ["BASE", "COLLISION_MESH", "LOAD_BEARING"],
      "physics_rpn": ["WEIGHT", "FLOOR_CONTACT", "STABILITY"]
    },
    "game_environment": {
      "visual_rpn": ["BASE", "LOW_POLY", "TEXTURE_ATLAS"],
      "interaction_rpn": ["SITTABLE", "THROWABLE", "DESTRUCTIBLE"]
    },
    "accessibility": {
      "audio_rpn": ["PRONOUNCE", "chair"],
      "tactile_rpn": ["3D_PRINT_MESH", "HAPTIC_FEEDBACK"]
    }
  }
}
```

**Result:**
- **Same base knowledge** (chair = seat + back + legs)
- **Context-specific execution** (furniture catalog vs BIM vs game vs accessibility)
- **Composable** (contexts can inherit/override base procedures)
- **Transparent** (every context rule is inspectable RPN program)

**This addresses Dave's concern:** PM-KR doesn't force terms into single meanings — it provides PROCEDURAL EXECUTION that adapts to context, unlike declarative approaches that require manual context modeling or neural approaches that hide context in weights.

---

## Approach

PM-KR proposes a paradigm shift: **store knowledge once as executable procedures, reference via composition**.

### Core Principles

1. **Dual-Client Contract**: One canonical procedural source serves both humans (readable) and AI systems (executable)
2. **Compositional Architecture**: Knowledge units compose via symlink-style references (no duplication)
3. **Procedural Foundation**: Knowledge is executable programs (like font programs, mathematical formula definitions, physics simulations)
4. **Hyper-Modularity**: Atomic knowledge units that combine into complex structures
5. **Explicit Context Rules**: Context-dependent meanings handled via inspectable procedural rules (not implicit neural weights)

### Conceptual Analogy

Think of how **TrueType fonts** work:

- One `.ttf` file contains procedural glyph definitions (Bézier curves, hinting programs)
- Renders at any size/weight/style without separate files
- All applications reference the same font file
- **Context-aware**: Hinting programs adjust based on pixel size, rendering context

PM-KR extends this principle to **all knowledge domains**: math symbols, spatial concepts, procedural rules, educational content, game mechanics, scientific protocols, etc.

---

## Why PM-KR Is Needed: Concrete Applications

### 1. Education: Procedural Textbooks

**Problem:** Current textbooks are static (print) or duplicated (separate e-book, audiobook, Braille versions).

**PM-KR Solution:**
- Store ONE procedural textbook source
- Render visually (e-book), audibly (screen reader), tactilely (Braille printer), interactively (AI tutor)
- **Application:** MIT OpenCourseWare could publish procedural calculus textbooks that AI tutors (GPT, Claude, Gemini) AND students consume from the SAME source

**Impact:**
- Accessibility: Automatic multi-modal rendering (visual, audio, tactile)
- AI integration: Tutors execute the same procedures students read
- Maintenance: Update once, all modalities update automatically

---

### 2. Gaming: Executable Rulebooks

**Problem:** Game rules are documented in PDF (humans read) but duplicated as code (game engines execute). Rule changes require manual synchronization.

**PM-KR Solution:**
- Store game rules as procedural RPN programs
- Human game masters READ the rules (rendered as text/diagrams)
- AI game masters EXECUTE the rules (same source)
- Rules engine uses the same procedural source

**Application:** Dungeons & Dragons SRD (System Reference Document) as PM-KR procedural rules

**Impact:**
- Consistency: Rules documentation = rules execution (zero divergence)
- Modding: Community creates procedural rule extensions (composable)
- AI DM: AI Dungeon Master executes canonical D&D rules (no hallucination)

---

### 3. Science: Procedural Experimental Protocols

**Problem:** Scientific protocols are described in papers (text) but re-implemented manually in labs (code/robots). Reproducibility crisis.

**PM-KR Solution:**
- Store experimental protocols as procedural programs
- Scientists READ the protocol (rendered as text with diagrams)
- Lab robots EXECUTE the protocol (same source)
- Reproducibility: Exact same procedure executed across labs

**Application:** Nature Protocols could publish procedural biochemistry experiments

**Impact:**
- Reproducibility: Protocol description = protocol execution
- Automation: Lab robots execute canonical procedures (no manual translation)
- Verification: Auditable execution traces (Verifiable Credentials for provenance)

---

### 4. Accessibility: Multi-Modal Knowledge Rendering

**Problem:** Blind users rely on screen readers (audio) or Braille (tactile), but these are SEPARATE from visual rendering. Updates to visual content don't propagate to audio/tactile.

**PM-KR Solution:**
- Store knowledge as procedural programs with multi-modal execution rules
- Visual rendering: Execute visual_rpn
- Audio rendering: Execute audio_rpn (pronunciation, descriptions)
- Tactile rendering: Execute tactile_rpn (3D-printable meshes, haptic feedback)

**Application:** Mathematical textbooks with visual equations, spoken descriptions, tactile 3D-printed graphs

**Impact:**
- Inclusion: Blind students access same knowledge as sighted students (not separate "accessible version")
- Synchronization: Update equation once, all modalities update
- Personalization: Users choose preferred modalities (visual + audio, tactile only, etc.)

---

### 5. AI Training: Canonical Knowledge Sources

**Problem:** AI companies duplicate Wikipedia, textbooks, documentation in training datasets. Same knowledge trained N times across N companies.

**PM-KR Solution:**
- Publish knowledge as procedural PM-KR sources
- AI systems query procedural knowledge during inference (not training)
- No duplication: ONE canonical source, N systems reference it

**Application:** Wikipedia as PM-KR procedural knowledge base (infoboxes, formulas, diagrams)

**Impact:**
- Carbon reduction: Train once, reference forever (no per-company duplication)
- Accuracy: AI systems query canonical source (no hallucination)
- Provenance: Verifiable Credentials track which knowledge was used

---

## Scope

### In Scope

**Core Standards:**

- Procedural knowledge data models
- Composition semantics (symlink references, deduplication)
- Execution semantics (RPN-based procedural programs)
- Context rules (handling context-dependent meanings)
- Conformance levels (minimal, extended, sovereign)

**Domains:**

- Mathematical knowledge (symbols, operators, formulas)
- Spatial knowledge (geometric primitives, transformations)
- Linguistic knowledge (characters, glyphs, typography)
- Educational knowledge (textbooks, curricula, multi-modal rendering)
- Game mechanics (rules, systems, procedural generation)
- Scientific knowledge (protocols, experiments, simulations)
- Accessibility (multi-modal rendering: visual, audio, tactile)

**Relationships with W3C Technologies:**

- JSON-LD (structured data representation)
- RDF/OWL (semantic web integration — PM-KR augments with execution)
- Verifiable Credentials (knowledge provenance, audit trails)
- CBOR-LD (compression for efficient transmission)

### Out of Scope

- Proprietary AI training formats (our focus: open, standardized)
- Natural language understanding (we provide structured knowledge inputs)
- Inference engines (we define knowledge representation, not reasoning systems)
- Replacing declarative approaches (PM-KR complements RDF/OWL with execution layer)

---

## Deliverables

### Specifications (Target: 2026-2027)

1. **PM-KR Core Specification v1.0**
   - Procedural knowledge data model
   - Composition semantics
   - Context rule semantics
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

4. **PM-KR Context Rules Specification v1.0**
   - Context-dependent execution semantics
   - Inheritance and override rules
   - Multi-modal rendering patterns
   - **Target:** Q2 2027

### Reference Implementation

- **Knowledge3D** (Python/CUDA): Sovereign spatial procedural knowledge system with GPU execution
- Demonstrates hyper-modular architecture, compression (50-90%), dual-client contract

### Use Case Documentation

- **Education**: Procedural textbooks for human and AI tutors (MIT OpenCourseWare example)
- **Games**: Executable rulebooks for game masters and AI assistants (D&D SRD example)
- **Science**: Procedural experimental protocols (Nature Protocols example)
- **Accessibility**: Multi-modal knowledge rendering (visual, audio, tactile)
- **AI Training**: Canonical knowledge sources (Wikipedia procedural knowledge base)

---

## Current Members (as of Feb 28, 2026)

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

---

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

---

## Addressing Dave Raggett's Feedback

**Dave's Questions:**

1. **"Why is a procedural approach the best solution to synonyms in knowledge representation? What's wrong with a declarative approach?"**
   - **Answer:** See "Why Procedural? Addressing Declarative Alternatives" section above. Declarative approaches describe relationships but don't provide EXECUTION logic needed for rendering, computation, or multi-modal access. PM-KR provides procedural execution while maintaining transparency (unlike neural networks).

2. **"What about the likelihood that different terms have subtly different meanings in their respective context of use?"**
   - **Answer:** See "Handling Context-Dependent Meanings" section. PM-KR uses explicit context rules (procedural programs) that adapt execution based on context (furniture catalog vs BIM vs game environment vs accessibility). Unlike neural networks (context hidden in weights) or declarative systems (manual context modeling), PM-KR provides inspectable, composable context rules.

3. **"Neural networks shine as a procedural solution, despite lacking transparency, in contrast to PKN, which demonstrates the potential for combining declarative approaches with a qualitative treatment of metadata."**
   - **Answer:** PM-KR combines the best of both: PROCEDURAL EXECUTION (like neural networks) with TRANSPARENCY (like declarative systems) plus COMPOSABILITY (which neither provides). See comparison table in "Why Procedural?" section.

4. **"I recommend working on explaining PM-KR in respect to why it is needed and what applications it targets."**
   - **Answer:** See "Why PM-KR Is Needed: Concrete Applications" section with 5 detailed examples: education (MIT OpenCourseWare), gaming (D&D SRD), science (Nature Protocols), accessibility (multi-modal textbooks), AI training (Wikipedia procedural KB).

---

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

---

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
3. Try the reference implementation (Knowledge3D)
4. Join discussions, propose ideas, contribute code

### Communication Channels

**Mailing List (primary):**

- public-pm-kr@w3.org
- For specification discussions, proposals, community updates

**GitHub Issues:**

- Technical discussions, implementation questions
- Specification clarifications

---

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
- Working prototypes prove dual-client viability
- MIT, Huawei, and JSON-LD co-creator joined within 4 days (Feb 20-24, 2026)

---

## Roadmap (2026-2027)

**Q1 2026 (Current):**

- ✅ Community Group launched (Feb 20, 2026)
- ✅ 18+ members recruited (MIT, Huawei, Manu Sporny, etc.)
- 🔄 Initial specification drafts (PM-KR Core v0.1)
- 🔄 Reference implementation (Knowledge3D)

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
- PM-KR Context Rules Specification v1.0
- Industry adoption (textbook publishers, game companies, AI platforms)
- Integration with W3C standards (Verifiable Credentials, RDF-star)

---

## Sustainability and Impact Goals

### Technical Impact

- **Standardize** procedural knowledge representation for AI systems
- **Reduce** knowledge duplication across systems (50-90% compression)
- **Enable** dual-client knowledge sources (humans + AI from same source)
- **Provide transparency** (procedural execution with inspectable rules, unlike neural networks)

### Societal Impact

- **Education**: Accessible, multi-modal textbooks (visual, audio, tactile)
- **Sustainability**: Reduce AI's carbon footprint via compression
- **Accessibility**: Knowledge rendered for diverse human needs
- **Reproducibility**: Scientific protocols as executable procedures

### Economic Impact

- **Efficiency**: Companies stop duplicating knowledge infrastructure
- **Innovation**: New applications enabled by compositional knowledge
- **Open Standards**: Prevent proprietary lock-in, foster ecosystem growth

---

## Contact

**Chairs:**

- Daniel Campos Ramos: capitain_jack@yahoo.com
- Milton Ponson: [contact via mailing list]

**Mailing List:**

- public-pm-kr@w3.org

**GitHub:**

- https://github.com/danielcamposramos/Knowledge3D

---

## Acknowledgments

PM-KR builds on decades of W3C work in semantic web, linked data, and web standards. Special thanks to:

- **Dave Raggett** (W3C veteran, feedback on declarative vs procedural approaches)
- Manu Sporny (JSON-LD co-creator)
- Tim Berners-Lee (Linked Data principles)
- JSON-LD Working Group
- Verifiable Credentials Community
- All 18+ early ingressors who joined in the first 4 days

**Let's build the future of knowledge representation together.** 🚀

---

**Last Updated:** February 28, 2026
**Version:** 1.1 (Revised based on Dave Raggett's feedback)
