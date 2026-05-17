# 📚 Library Room — Reference Material (READ-ONLY)

**Cognitive Function:** RETRIEVAL + CITATION
**Mode:** Read-only reference
**Agent Specialist:** Citation specialist

---

## 🧠 Agent Instructions

**WHEN YOU ENTER THIS ROOM:**
```yaml
mode: retrieval
specialist: citation_specialist
constraints:
  read_only: true
  modification_forbidden: true
  citation_required_when_referencing: true

allowed_operations:
  - query: Search for prior art, related specs, research
  - read: Access full content of reference materials
  - cite: Reference with markdown links [Title](relative/path)
  - extract: Pull relevant quotes for Workshop specs

forbidden_operations:
  - modify: Do not edit existing files
  - create: Do not create new files (file issues in Workshop feedback instead)
  - delete: Do not remove any content
```

**PURPOSE:**
This room contains consolidated reference material for PM-KR specification development. Think of it as the foundational knowledge base that Workshop specifications build upon.

---

## 📂 Contents

### `PM_KR_CG_CHARTER.md` — PM-KR Community Group Charter
- Grounded charter snapshot for PM-KR social/specification work.

### `PM_KR_WEBML_GROUNDING_MATRIX.md` — WebML Proposal Claim Map
- Claim-to-source matrix used to keep Intel/WebML submissions evidence-based.

### `prior-art/` — Research Papers & Related Work
- Procedural Knowledge Networks (PKN)
- Neuroscience research (Milton's bilingual brain studies)
- Computational linguistics foundations
- Knowledge representation paradigms

### `related-specs/` — W3C & Standards Documents
- W3C RDF Specification
- W3C OWL Specification
- W3C JSON-LD Specification
- WebNN API Specification
- Other relevant W3C standards

### `community-input/` — Expert Contributions
- Dave Raggett (W3C CogAI) feedback
- Anssi Kostiainen (Intel WebML) insights
- Wei Ding (Huawei) enterprise perspectives
- Milton Ponson (computational linguistics)
- Christoph Dorn (boundaries framework)

---

## 🔗 How to Use This Room

### For Specification Writing (Workshop):
1. **Research phase**: Read relevant prior art before drafting
2. **Citation**: Reference Library materials in Workshop specs
3. **Validation**: Check if your approach aligns with established research

**Example citation:**
```markdown
## 2.3 Procedural vs Declarative Knowledge

As documented in [PKN research](../library/prior-art/pkn-procedural-knowledge-networks.md),
procedural knowledge representation offers advantages for...
```

### For Community Engagement (Living Room):
1. **Context building**: Reference Library materials when explaining PM-KR to new members
2. **Prior art acknowledgment**: Show how PM-KR builds on established research

### For Introspection (Bathtub):
1. **Consistency checking**: Verify Workshop specs align with prior art
2. **Gap identification**: Identify missing references that should be in Library

---

## ➕ Adding New References

**Process:**
1. **Human (Daniel) manually adds** new reference materials
2. Files should be markdown (.md) or linked PDFs
3. Use descriptive filenames: `author-year-topic.md`
4. Update this README.md if new subfolder created

**Agents do NOT add files to Library** — this is curated reference material only.

---

## 📊 Current Status

**Prior Art:** [To be populated]
**Related Specs:** [To be populated]
**Community Input:** [To be populated]

**Last Updated:** March 3, 2026
**Curator:** Daniel Ramos (PM-KR Co-Chair)
