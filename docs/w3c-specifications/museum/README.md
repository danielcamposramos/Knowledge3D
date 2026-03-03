# 🏺 Museum Zone 8 — Historical Archive (READ-ONLY)

**Cognitive Function:** ARCHIVAL REFERENCE + PROVENANCE TRACKING
**Mode:** Historical preservation
**Agent Specialist:** Historical reference specialist

---

## 🧠 Agent Instructions

**WHEN YOU ENTER THIS ROOM:**
```yaml
mode: archival
specialist: historical_reference_specialist
constraints:
  read_only: true
  permanent_preservation: true
  provenance_tracking: required
  modification_forbidden: true

allowed_operations:
  - read: Access historical content for context
  - reference: Cite for provenance chains
  - compare: Show specification evolution over time
  - provenance: Track design decision lineage

forbidden_operations:
  - modify: Do not edit archived content (it's historical record)
  - delete: Do not remove files (permanent archive)
  - create: Do not add files manually (use archival process from Workshop)
```

**PURPOSE:**
This room preserves PM-KR's history: milestones, deprecated drafts, and decision records. Think of it as "cold storage" for version history and provenance tracking.

**Analogy:** Just like K3D's Museum (Zone 8) holds deprecated knowledge and historical artifacts, this folder holds PM-KR's design evolution history.

---

## 📂 Contents

### `milestones/` — Key Events & Achievements
**Purpose:** Preserve major PM-KR milestones for historical reference

**Examples:**
- `2026-02-24_founding_announcement.md`: PM-KR Community Group founded
- `2026-02-28_christoph_encapsulate_phase1.md`: Christoph integration Phase 1 complete
- `2026-03-02_huawei_joined.md`: Huawei (Wei Ding) joined PM-KR
- `2026-03-03_intel_webml_collaboration.md`: Intel/WebML collaboration initiated
- `2026-03-03_house_architecture_for_specs.md`: This architectural innovation!

**Format:**
```markdown
# Milestone: [Title]

**Date:** YYYY-MM-DD
**Event:** [Description]
**Significance:** [Why this matters for PM-KR]

## Details
[Full context, participants, outcomes]

## Impact
[How this changed PM-KR trajectory]

## References
- [Links to related documents, emails, specs]

---
**Archived by:** [Person/Agent]
**Archival Date:** YYYY-MM-DD
```

### `deprecated/` — Superseded Content
**Purpose:** Preserve early drafts and superseded versions (not deleted, archived)

**Structure:**
```
deprecated/
├── early-drafts-feb-2026/
│   ├── initial-charter-draft.md
│   └── early-data-model-sketch.md
├── superseded-specs/
│   └── data-model-v0.0.md (replaced by v0.1)
└── abandoned-approaches/
    └── pure-declarative-approach.md (decided against)
```

**Why preserve?**
- Shows design evolution (how we arrived at current approach)
- Prevents repeating past mistakes
- Provenance for design decisions

### `decision-records/` — Architecture Decision Records (ADRs)
**Purpose:** Document WHY key decisions were made

**Format (ADR template):**
```markdown
# ADR-XXX: [Decision Title]

**Date:** YYYY-MM-DD
**Status:** Accepted / Superseded / Deprecated
**Deciders:** [Who made this decision]

## Context
[What's the situation and problem statement?]

## Decision
[What's the change we're making?]

## Consequences
[What becomes easier/harder as a result?]

## Alternatives Considered
- [Alternative 1]: [Why rejected]
- [Alternative 2]: [Why rejected]

## References
- [Related specs, discussions, research]

---
**Recorded by:** [Person/Agent]
**Record Date:** YYYY-MM-DD
```

**Examples:**
- `2026-02-28_declarative-procedural-synergy.md`: Why PM-KR embraces BOTH
- `2026-03-03_house-architecture-for-specs.md`: Why spatially-aware file organization

---

## 🔗 How to Use This Room

### For Provenance Tracking:
```markdown
## 2.3 Design Rationale

The decision to use RPN-based execution semantics (as opposed to AST-based)
was made on [2026-03-15](../../../museum/decision-records/2026-03-15_rpn-vs-ast.md)
after evaluating...
```

### For Historical Context:
```markdown
## Appendix A: Specification Evolution

This specification evolved from early drafts in
[February 2026](../../../museum/deprecated/early-drafts-feb-2026/)
which originally proposed...
```

### For Milestone Reference:
```markdown
## Acknowledgments

This work builds on the PM-KR Community Group founded on
[February 24, 2026](../../../museum/milestones/2026-02-24_founding_announcement.md)
with contributions from...
```

---

## 📥 Archival Process (How Content Enters Museum)

**From Workshop → Museum:**

1. **Phase complete:**
```bash
# Example: Phase 1 Data Model v1.0 finalized
cp workshop/phase1-data-model/spec-draft.md \
   museum/milestones/2026-XX-XX_data-model-v1.0.md
```

2. **Early draft superseded:**
```bash
# Example: v0.0 replaced by v0.1
mv workshop/phase1-data-model/spec-draft-v0.0.md \
   museum/deprecated/superseded-specs/data-model-v0.0.md
```

3. **Decision documented:**
```bash
# Example: Major architectural decision made
# Human (Daniel) or Agent (Claude) creates ADR
touch museum/decision-records/2026-XX-XX_decision-title.md
```

**Archival is ONE-WAY:** Once in Museum, content is read-only forever.

---

## 📊 Provenance Chain Example

**Question:** Why does PM-KR use RPN instead of AST for execution semantics?

**Provenance trail:**
1. **Early draft** (`museum/deprecated/early-drafts-feb-2026/ast-based-sketch.md`) proposed AST
2. **Decision record** (`museum/decision-records/2026-03-XX_rpn-vs-ast.md`) documented evaluation
3. **Current spec** (`workshop/phase2-execution-semantics/spec-draft.md`) uses RPN with reference to decision
4. **Milestone** (`museum/milestones/2026-XX-XX_execution-semantics-v1.0.md`) marks completion

**Result:** Complete historical lineage from initial idea → decision → implementation → milestone.

---

## 🎯 Success Criteria for Museum

**Good Museum:**
- ✅ Preserves ALL major milestones (no missing events)
- ✅ Documents ALL key decisions (provenance complete)
- ✅ Archives superseded content (not deleted, preserved)
- ✅ Enables historical research (can trace design evolution)

**Bad Museum:**
- ❌ Milestones missing (lost history)
- ❌ No decision records (why questions unanswered)
- ❌ Deleted old content (provenance broken)

---

## 📊 Current Status

**Milestones Archived:** 5
- 2026-02-24: PM-KR founded
- 2026-02-28: Christoph integration Phase 1
- 2026-03-02: Huawei joined
- 2026-03-03: Intel/WebML collaboration
- 2026-03-03: House architecture innovation

**Deprecated Content:** 0 (nothing superseded yet)
**Decision Records:** 3
- 2026-02-28: Declarative-procedural synergy (ADR-001)
- 2026-03-03: House architecture for specifications (implied ADR, not numbered)
- 2026-03-03: Implementation neutrality (ADR-002) — **Foundational principle**

**Last Archival:** March 3, 2026

---

**Last Updated:** March 3, 2026
**Curator:** Daniel Ramos (PM-KR Co-Chair)
**Purpose:** Preserve PM-KR history for future generations
