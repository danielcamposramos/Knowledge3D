# PM-KR Email: Related Work, Form → Meaning, & Procedural Metadata

**To:** public-pm-kr@w3.org
**Subject:** Re: Procedural Metadata - Related Work, Form → Meaning Principle, JSON-LD Response
**Date:** February 25, 2026

---

Hi PM-KR community,

**Welcome to our newest member, Aswin Chandrasekaran!** The community continues to grow with diverse expertise. 🎉

Thanks to Adam's excellent intervention about procedural knowledge examples, I've compiled a comprehensive bibliography of related work:

**📚 PM-KR Related Work & References**
https://www.w3.org/community/pm-kr/pm-kr-related-work-references/

This covers:
- Existing standards (Schema.org, BPMN, BPEL)
- Cognitive science (ACT-R, SOAR, Tulving's memory systems)
- W3C technologies (JSON-LD, RDF, OWL, PROV-O)
- Procedural content generation (TrueType fonts, game systems)
- Industry applications (Watson, Wolfram Alpha)

**We're actively seeking contributions** — if you know of relevant papers, projects, or standards, please share them with the list or add them via GitHub.

---

## The Form → Meaning Principle (Core to PM-KR)

While researching K3D, I realized there's a fundamental principle underlying PM-KR that connects to how humans naturally process knowledge:

**Humans don't store separate "font files" when recognizing letters.**

When you see "A" — whether in Times New Roman, Arial, handwritten, or calligraphy — you recognize it as "A". You don't maintain separate mental representations for each visual form. **The form varies, but the meaning is universal.**

This "form → meaning" principle is **foundational to human knowledge**, and it applies far beyond typography:

### Everything Starts with Form

**In human history, all knowledge begins with form:**
- Mathematical concepts → drawings on clay tablets, papyrus, chalkboards
- Scientific discoveries → schematics, diagrams, blueprints
- Engineering designs → technical drawings with annotations
- Musical notation → visual symbols on staves
- Chemical structures → structural formulas (drawings + meaning)

**Letters themselves** are just very complex drawings with compositional meanings (grammar, language, semantics). A schematic is nothing more than drawings with contextual meaning attached.

### How This Connects to PM-KR's Architecture

PM-KR's layered architecture reflects this natural progression:

**Layer 0: Drawing Primitives (Pure Form)**
- LINE, CIRCLE, RECT, BEZIER (procedural geometric programs)
- No inherent meaning — just form

**Layer 1: Characters (Form + Minimal Context)**
- "A" as procedural Bézier curves
- Universal form (before language-specific meaning)
- Can render as visual glyph, Braille dots, phonetic sound

**Layer 2+: Compositional Meaning (Form + Context)**
- Words, grammar, semantics
- Same forms compose into language-specific meanings
- Multiple Layer 2s (Occidental, Oriental, etc.) share Layer 1 substrate

**The key insight:** We don't duplicate the form at each layer. We **reference it** (symlink pattern) and add contextual metadata as we move up the stack.

---

## Why This Matters for AI

Current AI systems violate the "form → meaning" principle:

**Problem:**
- LLMs memorize "A" in Times New Roman (training data)
- LLMs memorize "A" in Arial (more training data)
- LLMs memorize "A" in handwriting samples (even more training data)
- Each representation duplicated across embeddings, model weights, retrieval systems

**PM-KR approach:**
- Store "A" ONCE as procedural form (Layer 1)
- Reference it across contexts (fonts, languages, modalities)
- Attach meaning via compositional metadata (not duplication)

**Result:** 50-90% compression + dual-client capability (humans and AI consume the same canonical source).

---

## Multi-Modal Extension

This principle extends naturally to multi-modal AI:

**Visual reasoning (ARC-AGI):**
- Grids are drawings (Layer 0 primitives)
- Patterns have compositional meaning (Layer 2+ grammar)

**Mathematical reasoning:**
- Symbols are visual forms (∫, Σ, π as procedural drawings)
- Meaning emerges from compositional context (∫ in calculus vs. ∫ in physics)

**Spatial reasoning:**
- 3D structures start as geometric primitives (form)
- Meaning = physics constraints + interaction rules (context)

**All use the same substrate** — form at the bottom, meaning via composition.

---

## Response to Adam & Christoph: Procedural Metadata

Thanks to Adam and Christoph for raising the critical topic of **metadata on procedural knowledge**! This is exactly where PM-KR's JSON-LD foundation shines.

### Adam's Examples: Attributes, Annotations, Decorators

Adam showed how modern programming languages attach metadata to functions:

**C# (attributes):**
```csharp
[Precondition("x > 0")]
[Effect("result >= x")]
[SecurityPermission(SecurityAction.Demand)]
public double sqrt(double x) { ... }
```

**Java (annotations):**
```java
@Precondition("balance >= amount")
@Effect("balance -= amount")
@AccessControl(role="accountHolder")
public void withdraw(double amount) { ... }
```

**JavaScript (decorators):**
```javascript
@rateLimit(100)
@requireAuth
@audit
function processPayment(...) { ... }
```

These are **fantastic** examples of procedural metadata in practice!

### PM-KR's Approach: JSON-LD Context on RPN Programs

PM-KR extends this principle using **JSON-LD** for extensible metadata on procedural knowledge:

**Example: Math operation with metadata**
```json
{
  "@context": "https://pm-kr.org/contexts/math.jsonld",
  "@type": "ProceduralProgram",
  "id": "math:sqrt",
  "program": ["DUP", "0", "GT", "ASSERT", "SQRT"],
  "preconditions": [
    {"type": "Constraint", "expr": "input > 0"}
  ],
  "effects": [
    {"type": "Assertion", "expr": "output^2 ≈ input"}
  ],
  "metadata": {
    "complexity": "O(1)",
    "precision": "IEEE 754 double",
    "domain": "real numbers (positive)"
  }
}
```

**Example: Access-controlled action**
```json
{
  "@context": "https://pm-kr.org/contexts/security.jsonld",
  "@type": "ProceduralProgram",
  "id": "bank:withdraw",
  "program": ["BALANCE", "SWAP", "GTE", "ASSERT", "SUB", "STORE_BALANCE"],
  "preconditions": [
    {"type": "SecurityConstraint", "role": "accountHolder"},
    {"type": "BusinessRule", "expr": "balance >= amount"}
  ],
  "effects": [
    {"type": "StateChange", "target": "balance", "operation": "subtract"},
    {"type": "AuditLog", "event": "withdrawal", "timestamp": "auto"}
  ],
  "permissions": {
    "requiresAuth": true,
    "minimumTrustLevel": "verified",
    "auditRequired": true
  }
}
```

### Christoph's Use Case: Schema-Mapped Metadata

Christoph mentioned wanting to **attach any type of metadata by mapping to a schema**. This is **exactly** what JSON-LD excels at!

**PM-KR's extensibility:**
```json
{
  "@context": [
    "https://pm-kr.org/contexts/core.jsonld",
    "https://your-domain.com/contexts/custom.jsonld"
  ],
  "@type": "ProceduralProgram",
  "id": "custom:myAction",
  "program": ["...RPN program..."],

  "customMetadata": {
    "@type": "YourCustomSchema",
    "field1": "value1",
    "field2": "value2",
    "nestedSchema": {
      "@type": "AnotherSchema",
      "data": "..."
    }
  }
}
```

**Key advantages:**
1. **Open-world assumption** — any schema can be referenced via JSON-LD `@context`
2. **Compositional** — multiple contexts can be combined
3. **Machine-readable** — RDF triples enable automated reasoning
4. **Human-readable** — JSON syntax familiar to developers

### Connection to STRIPS, PDDL, and Planning

Adam mentioned **STRIPS** (Stanford Research Institute Problem Solver) and **PDDL** (Planning Domain Definition Language) — these are **foundational** to AI planning systems!

**STRIPS-style representation in PM-KR:**
```json
{
  "@type": "ProceduralProgram",
  "id": "blocks:stack",
  "preconditions": [
    {"predicate": "clear", "args": ["?x"]},
    {"predicate": "holding", "args": ["?y"]}
  ],
  "effects": [
    {"operation": "add", "predicate": "on", "args": ["?y", "?x"]},
    {"operation": "delete", "predicate": "holding", "args": ["?y"]},
    {"operation": "delete", "predicate": "clear", "args": ["?x"]},
    {"operation": "add", "predicate": "clear", "args": ["?y"]},
    {"operation": "add", "predicate": "armEmpty", "args": []}
  ],
  "program": ["PLACE", "?y", "ON", "?x", "RELEASE"]
}
```

This enables **AI planning systems** to reason over procedural knowledge — not just execute it, but **plan with it**.

### Security & Access Control (Adam's Links)

Adam shared excellent resources on securing actions:

**PM-KR can integrate with existing security frameworks:**
- **.NET Code Access Security** → metadata specifies required permissions
- **Java CAS** → role-based access control via JSON-LD annotations
- **MCP (Model Context Protocol)** → tool metadata for AI safety
- **BPMN security extensions** → workflow-level authorization

**Example: MCP-style tool metadata**
```json
{
  "@context": "https://modelcontextprotocol.io/schemas/tools.jsonld",
  "@type": "Tool",
  "name": "file_delete",
  "description": "Deletes a file from the filesystem",
  "program": ["PATH", "VALIDATE", "DELETE"],
  "dangerLevel": "high",
  "requiresConfirmation": true,
  "userPermissions": ["filesystem:write", "filesystem:delete"],
  "auditLog": true
}
```

This is **exactly** the kind of metadata-driven security we need for agentic AI systems!

### "Form → Meaning" Extends to Metadata

Here's the key insight: **Metadata is contextual meaning attached to procedural form.**

**Same RPN program, different contexts:**
```json
// Mathematical context
{
  "@context": "math.jsonld",
  "program": ["DUP", "MUL"],
  "metadata": {"operation": "square", "domain": "real numbers"}
}

// Physics context
{
  "@context": "physics.jsonld",
  "program": ["DUP", "MUL"],
  "metadata": {"operation": "energy", "formula": "E = mc²", "units": "joules"}
}

// Security context
{
  "@context": "security.jsonld",
  "program": ["DUP", "MUL"],
  "metadata": {"complexity": "O(1)", "sideEffects": "none", "safe": true}
}
```

**The form (RPN program) is the same — the meaning changes via metadata context.**

This is the "form → meaning" principle applied to procedural metadata!

---

## Connection to TrueType Fonts (Our Core Analogy)

TrueType fonts already implement "form → meaning":

1. **One `.ttf` file** stores procedural glyph definitions (form)
2. **Rendering context** determines appearance (size, weight, anti-aliasing)
3. **Semantic context** provides meaning (language, typography rules)

PM-KR asks: **What if we extend this principle to ALL knowledge domains?**

- Math symbols (not just letters)
- Spatial concepts (not just 2D glyphs)
- Procedural rules (not just static shapes)
- Multi-modal knowledge (visual, audio, tactile from one form)

---

## Why This Is Relevant to W3C

The Web already faces this duplication problem:

- Same content exists in multiple formats (HTML, PDF, EPUB)
- Accessibility requires separate versions (Braille, audio, large print)
- Translations duplicate structure (only content changes)
- APIs duplicate knowledge across systems

**PM-KR's "form → meaning" architecture** could provide:
- One canonical procedural source (form)
- Multi-modal rendering (visual, tactile, audio)
- Language-agnostic substrate (Layer 1 shared, Layer 2+ diverge)
- Verifiable knowledge provenance (same source = guaranteed consistency)

This aligns with W3C's mission: **one Web, accessible to all, in all modalities.**

---

## Invitation for Discussion

I'd love to hear the group's thoughts on:

**Form → Meaning Principle:**
1. **Does the "form → meaning" principle resonate?** (Or is this over-simplifying?)
2. **What other domains exhibit this pattern?** (Music? Chemistry? Programming languages?)
3. **Are there existing formalizations we should reference?** (Semiotics, cognitive science, linguistics?)

**Procedural Metadata (Adam & Christoph's topic):**
4. **Which metadata schemas are most critical?** (Security, planning, accessibility, provenance?)
5. **How should PM-KR integrate with existing standards?** (STRIPS, PDDL, MCP, BPMN security?)
6. **What use cases need extensible metadata?** (Christoph's "any schema" requirement?)
7. **Should we formalize a PM-KR metadata vocabulary?** (Or just rely on JSON-LD extensibility?)

**W3C Integration:**
8. **How could W3C standards leverage this?** (Accessibility, Semantic Web, Linked Data?)
9. **Which W3C groups should we coordinate with?** (Web Machine Learning, Security, Credentials?)

---

## Resources

**Related Work Bibliography:**
https://www.w3.org/community/pm-kr/pm-kr-related-work-references/

**PM-KR Community Page:**
https://www.w3.org/community/pm-kr/procedural-memory-knowledge-representation-pm-kr-community-group/

**GitHub (Specifications + Implementation):**
https://github.com/danielcamposramos/Knowledge3D

---

Looking forward to the discussion!

Best regards,
**Daniel Campos Ramos**
PM-KR Co-Chair
Knowledge3D Project

---

**P.S.** - If you're interested in contributing references to the Related Work page, you can:
- Email public-pm-kr@w3.org with citations
- Open GitHub issue: https://github.com/danielcamposramos/Knowledge3D/issues

---

## END OF EMAIL DRAFT
