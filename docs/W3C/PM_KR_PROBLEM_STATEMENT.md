# Procedural Memory Knowledge Representation: Problem Statement

> **Status, 29 October 2026: Draft.** This is project input to the [PM-KR Community Group](https://www.w3.org/community/pm-kr/).
> Knowledge3D code is paused while work focuses on specifications and group deliverables, as [Daniel explained on 28 October](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
> Requirements and examples below describe proposals, not implemented features or adopted W3C conformance rules.

**Document Type**: W3C Community Group Contribution (Draft)
**Version**: 1.0
**Date**: February 20, 2026
**Authors**: Knowledge3D Project Contributors
**Status**: Draft Proposal

---

## Abstract

This draft asks how people and AI could use the same knowledge while retaining the views each needs.
PM-KR proposes reusable procedures, explicit meanings and references between entries.
Storage savings and consistency across clients are questions to test, not results established by this problem statement.

---

## 1. The Knowledge Duplication Crisis

### 1.1 Current State: Static Payload Duplication

The following hypothetical examples illustrate duplication that can arise when related representations are maintained separately.
They do not describe every knowledge graph, font system or deployment.

**Example 1: Character Knowledge**
```
Traditional System (per-language duplication):
├─ English: "A" stored with [font, unicode, meaning, pronunciation]
├─ Spanish: "A" stored with [font, unicode, meaning, pronunciation]
├─ French:  "A" stored with [font, unicode, meaning, pronunciation]
└─ Risk: separately maintained copies can drift
```

**Problem illustrated**: Independently maintained copies can drift.
Sharing a glyph does not imply sharing a character identity or meaning in every language and context.

**Example 2: Semantic Knowledge**
```
Traditional Knowledge Graph:
├─ "rotation_task": "Task involves rotating elements" (string payload)
├─ "reflection_task": "Task involves reflecting elements" (string payload)
└─ 400 task descriptions × 3 semantic tags = 1,200 duplicate strings
```

**Problem**: Semantic metadata duplicated rather than composed from canonical word meanings.

### 1.2 Consequences of Duplication

1. **Storage Explosion**
   - Independently maintained text, embeddings and metadata can duplicate facts.
   - Measure the complete stored representation, including references and indexes, before claiming a saving.

2. **Maintenance Burden**
   - Update "rotation" semantics → must find all 400 duplicates
   - Fix glyph rendering bug → must update all language variants
   - Version control nightmare: which copy is canonical?

3. **Consistency Violations**
   - Similar-looking characters can have different identities; visual similarity alone cannot justify merging them.
   - Semantic drift: "rotation" means different things in different contexts
   - No single source of truth

4. **Computational Waste**
   - Duplicate embeddings computed independently
   - Redundant similarity calculations (comparing near-identical payloads)
   - Cache pollution (duplicates evict unique knowledge)

---

## 2. The Procedural-Static Divide

### 2.1 Form Without Meaning

The proposed model distinguishes visual form, semantic identity and executable behavior.
The example below illustrates their association; it is not a description of font internals.

**Example: Mathematical Symbol "π"**

Traditional System:
```json
{
  "symbol": "π",
  "visual": "glyph_reference",  // Reference to a glyph in a font
  "meaning": "Ratio of circumference to diameter",  // String description
  "execution": "use_python_math_pi()"  // External dependency!
}
```

**Problems**:
- Visual form (font) separate from meaning (description)
- Meaning is textual description, not executable
- Execution environments and dependencies need to be declared for reproducible evaluation.
- The visual and machine views need an explicit shared identity and consistency tests.

### 2.2 The Dual-Client Opacity Problem

**Human Perspective**:
- Navigates 3D knowledge space
- Sees visual glyphs, geometric shapes, spatial layout
- Trusts AI is "seeing" the same thing

**AI Perspective**:
- Processes high-dimensional embeddings
- Executes opaque transformations
- No guarantee it corresponds to what human sees

**Question**: Do the human and machine views identify the same object and version, and where do their interpretations differ?

---

## 3. The Compression-Meaning Tradeoff

### 3.1 Current Compression Approaches

Candidate baselines include existing fonts, knowledge graphs, content-addressed storage, embedding compression and ordinary file compression.
The comparison must preserve the same content and account for indexes, references, execution costs and reconstruction loss.
Semantic similarity alone does not establish that two entries are interchangeable.

### 3.2 The Missing Compression Paradigm

**What's needed**:
1. **Procedural Canonicalization**: Store ONE executable procedure for "A" (glyph generation)
2. **Symlink Composition**: Words reference character procedures (not duplicate glyphs)
3. **Meaning-Aware Compression**: "rotation_task" = word_ref("rotation") + word_ref("task")
4. **Dual-Client Consistency**: Same procedural program generates both human visual + AI semantic

**Goal**: Reduce repeated storage without losing the meaning required by a use case.

---

## 4. The Sovereignty Crisis

### 4.1 External Dependency Cascade

Modern AI systems rely on external libraries at inference time:

**Example: Math Reasoning Chain**
```python
# Pseudo-code for typical AI math solver
def solve_equation(problem):
    # Step 1: Parse (external NLP library)
    parsed = spacy.parse(problem)  # ← External dependency!

    # Step 2: Symbolize (external CAS)
    symbolic = sympy.sympify(parsed)  # ← External dependency!

    # Step 3: Solve (external solver)
    solution = sympy.solve(symbolic)  # ← External dependency!

    # Step 4: Format (external)
    formatted = latex(solution)  # ← External dependency!

    return formatted
```

**Problems**:
- 4 external dependencies in hot path (numpy, sympy, scipy, spacy)
- Each dependency = potential security risk, version conflict, licensing issue
- No guarantee of determinism (library updates change behavior)
- GPU sovereignty lost (CPU preprocessing required)

### 4.2 The Sovereign Inference Requirement

**What AI systems need**:
- **Zero external dependencies** in inference hot path
- **Deterministic execution** (same input → same output, always)
- **GPU-native** (no CPU preprocessing bottlenecks)
- **Auditable** (trace every operation to source code)

These are design requirements of the proposed K3D execution profile, not a survey of other AI systems.

---

## 5. The PM-KR Solution Thesis

### 5.1 Core Principles

**Principle 1: Procedural Canonicalization**
- Knowledge = executable procedures (RPN programs)
- ONE canonical procedure per concept/symbol
- Visual form + semantic meaning unified in same procedure

**Principle 2: Symlink Composition**
- Higher layers reference lower canonical procedures
- Words = char_refs (not duplicate glyphs)
- Semantic tags = word_refs (not duplicate strings)
- Goal: reuse canonical content through references; measure any storage saving.

**Principle 3: Dual-Client Reality**
- Same procedural program generates:
  - Human perception (visual rendering via GPU)
  - AI perception (semantic embeddings via same GPU kernels)
- Shared ground truth = procedural source code

**Principle 4: Sovereign Execution**
- Hot path = PTX kernels only (GPU-native, zero external dependencies)
- Ingestion path = flexible (use any tools to generate procedural memory)
- Result: Deterministic, auditable, sovereign inference

### 5.2 Architectural Innovation

**PM-KR introduces a 4-layer compositional stack**:

```
Meta-Rules Layer (Strategy)
    ↓ references
Rules Layer (Transformations)
    ↓ references
Meaning Layer (Semantics)
    ↓ references
Form Layer (Canonical Procedures)
```

**Example: Mathematical Equation Solving**

```python
# Form Layer: Canonical symbol procedures
symbols = {
    "π": {
        "visual_rpn": "CIRCLE 1.0 RADIUS ...",  # Procedural glyph
        "math_rpn": "3.14159265358979323846 PUSH",  # Executable constant
    },
    "x": {
        "visual_rpn": "GLYPH_X PROCEDURAL_FONT_LATIN ...",
        "math_rpn": "VARIABLE 0 PUSH",  # Variable slot
    }
}

# Meaning Layer: Semantic composition
equation = {
    "form_refs": [word_ref("solve"), symbol_ref("x")],
    "meaning_rpn": "SOLVE_LINEAR_1VAR ...",  # Procedural solver
}

# Rules Layer: Transformation rules
grammar = {
    "pattern": "solve {equation}",
    "rule_refs": [equation_ref("linear_solver")],
    "transformation_rpn": "PARSE SYMBOLIZE SOLVE FORMAT ...",
}

# Meta-Rules Layer: Strategy selection
strategy = {
    "condition": "linear_equation",
    "rule_ref": grammar_ref("linear_solver"),
    "confidence": 0.95,
}
```

**Result**:
- **Zero duplication**: All symbols canonical, all higher layers use references
- **Dual-client**: Visual rendering + semantic execution from same RPN
- **Sovereign**: All math operations via PTX kernels (no sympy/numpy)
- **Compression goal**: compare complete procedural and baseline representations on the same inputs.

---

## 6. Validation Evidence (K3D Implementation)

This problem statement establishes no benchmark result.
The [evidence matrix](PM_KR_EVIDENCE_VALIDATION_MATRIX.md) describes the artifacts needed to evaluate the proposal.

### 6.1 Compression Results

Compare identical content, including procedure code, references, metadata, indexes and caches.
Retain source assets and reconstruction checks alongside byte counts.

### 6.2 Sovereignty Validation

Trace the actual execution path, including failed tasks and preloaded dependencies.
A count of solved tasks or an import scan alone does not establish GPU execution.

### 6.3 Dual-Client Consistency

Check identity, version and meaning across human and machine views.
Accessible presentation requires additional user and assistive-technology testing.

### 6.4 Production Metrics

No production deployment or operational readiness is established here.
Any future report needs a public revision, reproducible command, inputs, environment and output artifacts.

---

## 7. Broader Impact

### 7.1 Computer Science Implications

**The proposal explores these design choices:**

| Traditional Assumption | PM-KR Alternative |
|----------------------|------------------|
| Knowledge = static data | Knowledge = executable procedures |
| Compression ↔ meaning tradeoff | Compression preserves meaning via references |
| AI and humans see different things | Dual-client shared reality via procedural source |
| Inference requires external libraries | Sovereign PTX-only hot path |

These choices need comparison with existing representations on bounded use cases.

### 7.2 Web Standards Implications

**Candidate spatial-web uses to evaluate:**

1. **Spatial Knowledge Navigation** (not hypertext search)
   - Users walk through 3D knowledge spaces
   - Semantic proximity = spatial proximity
   - Galaxy Universe as addressable 3D RAM

2. **Federated Knowledge Networks** (not centralized databases)
   - Houses (like websites) connected via Portals (like hyperlinks)
   - Procedural memory as standard interchange format
   - glTF + PM-KR extensions as transport layer

3. **Human-AI Shared Reality** (not separate interfaces)
   - Same glTF files rendered for both clients
   - Procedural programs generate dual perceptions
   - Verifiable shared ground truth

**Proposal**: PM-KR as candidate W3C technology for spatial knowledge representation.

---

## 8. Open Questions for Community

### 8.1 Community Group Path

**Questions**:
1. Should PM-KR be a standalone W3C specification or glTF extension?
2. How to formalize conformance testing for external implementations?
3. What governance model for canonical procedure registries?

### 8.2 Interoperability Concerns

**Questions**:
1. How to migrate existing knowledge graphs to PM-KR?
2. Can PM-KR interop with RDF/OWL/JSON-LD?
3. What translation losses occur (static → procedural)?

### 8.3 Performance Tradeoffs

**Questions**:
1. When does procedural overhead exceed static lookup?
2. How to balance compression vs execution latency?
3. What are optimal symlink graph depths?

---

## 9. Call to Action

### 9.1 For W3C Community

**We propose**:
1. **Community Group participation** through the [existing PM-KR group](https://www.w3.org/community/pm-kr/)
2. **Draft specification development** (normative model, test suites)
3. **Interoperability testing** with existing KR standards (RDF, JSON-LD)

### 9.2 For Implementation Partners

**We invite**:
1. **Reference implementations** in other domains (database systems, knowledge graphs)
2. **Benchmark comparisons** (PM-KR vs traditional approaches)
3. **Conformance testing** against K3D validation suite

### 9.3 For Researchers

**We encourage**:
1. **Theoretical analysis** of PM-KR compression bounds
2. **Empirical studies** on dual-client consistency guarantees
3. **Security analysis** of sovereign execution models

---

## 10. Conclusion

The proposal connects reusable procedures to meaning and provenance for people and AI.
Its next step is a small vocabulary and knowledge repository, as [Daniel proposed on 28 October 2026](https://lists.w3.org/Archives/Public/public-pm-kr/2026Oct/0016.html).
Compression, accessible equivalence, scale and execution behavior remain evaluation questions.

---

## References

**K3D Specifications**:
- Knowledgeverse Specification (7-region memory architecture)
- Dual-Client Contract Specification (procedural foundation)
- Adaptive Procedural Compression Specification (PD04 codecs)
- Sovereign NSI Specification (PTX-only hot path)

**Standards references**:
- [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html), maintained by Khronos
- WebXR Device API (spatial interfaces)
- RDF 1.1 Specification (comparison baseline)

**Academic Context**:
- Knowledge Representation and Reasoning (KR&R) research
- Semantic Web standards evolution
- Embodied AI and spatial cognition

---

**Document Status**: Draft for W3C Community Group Discussion
**License**: CC-BY-4.0 (free to share, adapt, with attribution)
**Repository**: https://github.com/danielcamposramos/Knowledge3D

---

## Appendix A: Quick Comparison

| Aspect | Traditional KR | PM-KR |
|--------|---------------|-------|
| **Storage** | Static payloads | Executable procedures |
| **Duplication** | Depends on the representation and reuse model | Proposed canonical references; savings need measurement |
| **Compression** | Lossless and lossy approaches are available | Proposed reference reuse; semantic preservation needs tests |
| **Human-AI** | Separate views | Shared procedural reality |
| **Dependencies** | External libs (numpy, scipy) | Sovereign PTX kernels |
| **Determinism** | Must specify runtime and inputs | Required by the proposed profile for a fixed runtime; needs tests |
| **Auditability** | Opaque transformations | Traceable RPN programs |

PM-KR proposes procedures and references as a common source for human and machine views.
