# 🔨 Workshop Room — Active Drafting (READ-WRITE)

**Cognitive Function:** CREATION + ITERATION + VALIDATION
**Mode:** Active specification development
**Agent Specialist:** Specification writer

---

## 🧠 Agent Instructions

**WHEN YOU ENTER THIS ROOM:**
```yaml
mode: creation
specialist: specification_writer
constraints:
  read_write: true
  validation_required: true
  community_feedback_integration: required
  library_reference_encouraged: true

allowed_operations:
  - draft: Write specification sections
  - iterate: Refine based on feedback
  - validate: Check against conformance tests
  - example: Generate code examples
  - test: Create conformance test cases
  - integrate_feedback: Incorporate community comments

forbidden_operations:
  - modify_library: Do not edit Library reference materials
  - modify_museum: Do not edit Museum archived content
```

**PURPOSE:**
This is the active workspace for drafting PM-KR specifications. All specification writing, code examples, conformance tests, and iterative refinement happens here.

---

## 📂 Phase Structure

### Phase 1: Data Model Specification
**Location:** `phase1-data-model/`
**Goal:** Draft Procedural Memory Data Model Specification v0.1
**Target:** March-April 2026

**Deliverables:**
- `spec-draft.md`: Main specification document
- `examples/`: JSON/code examples of data model
- `conformance-tests/`: Level 0 (read-only) conformance tests
- `feedback/`: Community comments, issues, suggestions

### Phase 2: Execution Semantics Specification
**Location:** `phase2-execution-semantics/`
**Goal:** Draft Execution Semantics Specification v0.1
**Target:** April-May 2026

**Deliverables:**
- `spec-draft.md`: RPN + TRM execution model
- `examples/`: RPN stack execution, TRM specialist routing
- `conformance-tests/`: Level 1 (execute) conformance tests

### Phase 3: Conformance Specification
**Location:** `phase3-conformance/`
**Goal:** Draft Conformance Levels Specification v0.1
**Target:** May-June 2026

**Deliverables:**
- `spec-draft.md`: Level 0/1/2 conformance definitions
- `conformance-tests/`: Complete test suite

### Phase 4: Integration Specification
**Location:** `phase4-integration/`
**Goal:** Draft RDF/OWL/JSON-LD Integration Specification v0.1
**Target:** June-July 2026

**Deliverables:**
- `spec-draft.md`: W3C standards interoperability
- `examples/`: Integration code examples

---

## 🔄 Workflow

### 1. Draft Specification Section
```bash
cd workshop/phase1-data-model/
# Agent (Claude) writes spec-draft.md sections
```

### 2. Generate Code Examples
```bash
cd workshop/phase1-data-model/examples/
# Agent (Codex) creates working code examples
```

### 3. Create Conformance Tests
```bash
cd workshop/phase1-data-model/conformance-tests/
# Agent (Codex) writes test cases
```

### 4. Collect Community Feedback
```bash
cd workshop/phase1-data-model/feedback/
# Human (Daniel) + Living Room agent files feedback
```

### 5. Iterate Based on Feedback
```bash
# Agent (Claude) refines spec-draft.md
# Integration loop continues
```

---

## 📝 Specification Writing Guidelines

### Structure (Follow W3C Spec Template):
```markdown
# Specification Title

## Abstract
[1-2 paragraphs summarizing the spec]

## Status of This Document
[Draft, version, date]

## 1. Introduction
### 1.1 Background
### 1.2 Goals
### 1.3 Non-Goals

## 2. Conformance
### 2.1 Conformance Requirements
### 2.2 Terminology

## 3. Core Concepts
[Main specification content]

## 4. Data Model / Execution Semantics / etc.

## 5. Examples

## 6. Conformance Tests

## 7. Security & Privacy Considerations

## 8. References
### 8.1 Normative References
### 8.2 Informative References
```

### Tone:
- **Precise**: Use RFC 2119 keywords (MUST, SHOULD, MAY)
- **Clear**: Avoid ambiguity
- **Concise**: Milton's guidance (no AI verbosity)
- **Reference prior art**: Cite Library materials

### Citations:
- **Library references:** `[PKN]` → `../library/prior-art/pkn.md`
- **Cross-spec references:** `[EXEC-SEM]` → `../workshop/phase2-execution-semantics/spec-draft.md`
- **W3C specs:** `[RDF]` → Full W3C URL

---

## 🧪 Validation Requirements

**Before marking a phase complete:**
1. ✅ Specification draft covers all charter goals for that phase
2. ✅ Code examples work (validated by Codex)
3. ✅ Conformance tests pass
4. ✅ Community feedback addressed (or documented as deferred)
5. ✅ Cross-spec consistency checked (Bathtub introspection)

---

## 🤝 Community Feedback Integration

**Feedback sources:**
- PM-KR mailing list (internal-pm-kr@w3.org)
- Dave Raggett (CogAI), Anssi Kostiainen (WebML), Wei Ding (Huawei), Milton, Christoph
- GitHub issues (when public)

**Process:**
1. Feedback filed in `feedback/` subfolder
2. Agent (Claude) reviews weekly
3. Updates spec-draft.md or creates issue for later phases
4. Acknowledgment in spec (Contributors section)

---

## 📊 Current Status

**Phase 1 (Data Model):** 0% (Starting March 3, 2026)
**Phase 2 (Execution Semantics):** 0%
**Phase 3 (Conformance):** 0%
**Phase 4 (Integration):** 0%

**Next Action:** Claude drafts Phase 1: Procedural Memory Data Model Specification v0.1

---

**Last Updated:** March 3, 2026
**Active Agent:** Claude (Specification Writer)
**Human Coordinator:** Daniel Ramos (PM-KR Co-Chair)
