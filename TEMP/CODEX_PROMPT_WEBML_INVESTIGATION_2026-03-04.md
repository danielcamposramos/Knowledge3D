# Codex Task: Comprehensive WebML Proposals Investigation

**Date:** March 3, 2026
**Assigned by:** Daniel Ramos (PM-KR Co-Chair)
**Context:** Intel/WebML collaboration via Anssi Kostiainen

---

## Background

Anssi Kostiainen (Intel, WebML CG) responded to PM-KR collaboration request:

> "You may submit a new proposal here: https://github.com/webmachinelearning/proposals
> The group will review submitted proposals from time to time."

**Strategic goal:** Submit comprehensive, grounded proposal that impresses Intel/WebML with deep understanding of their needs + PM-KR's procedural reasoning value proposition.

---

## Task Overview

**Objective:** Investigate WebML proposals repository thoroughly, understand submission patterns, identify gaps PM-KR can fill, draft world-class proposal.

**Approach:** Christoph-inspired deep investigation (grounded truth, not theoretical speculation).

---

## Grounding Contract (Mandatory)

Use only claims that can be traced to concrete repository documents.

**Primary evidence set (must cite path + section/line in outputs):**
- `docs/w3c-specifications/library/PM_KR_CG_CHARTER.md`
- `docs/w3c-specifications/workshop/phase1-data-model/spec-draft.md`
- `docs/W3C_PM_KR_COMMUNITY_GROUP_MISSION.md`
- `docs/W3C_PM_KR_OBJECTIVES_v1.2.md`
- `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md`
- `docs/W3C/PM_KR_EVIDENCE_VALIDATION_MATRIX.md`
- `docs/vocabulary/PROCEDURAL_MEMORY_KR_STANDARD_SPECIFICATION.md`
- `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md`
- `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md`

**Grounding rules:**
1. If a metric is not verifiable in local docs, exclude it.
2. Mark projections as projections, not observed production results.
3. Distinguish repository snapshots from live W3C web state.
4. Keep proposal claims implementation-neutral and standards-appropriate.

---

## Phase 1: Repository Setup

### 1.1 Clone WebML Proposals Repository

**Target:** https://github.com/webmachinelearning/proposals

**Location:** Clone as **sister folder** to K3D project (NOT inside K3D to avoid cross-cloning).

**Expected structure:**
```
/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/
├── Knowledge3D/           # K3D project (existing)
└── webmachinelearning-proposals/  # WebML proposals (new clone)
```

**Command:**
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/"
git clone https://github.com/webmachinelearning/proposals webmachinelearning-proposals
cd webmachinelearning-proposals
```

### 1.2 Initial Repository Analysis

**Questions to answer:**
1. How many total proposals? (open + closed)
2. What's the proposal format/template?
3. What's the typical proposal length? (lines, sections)
4. How are proposals categorized? (labels, tags, directories)
5. What's the review process? (GitHub issues? PRs? Discussion threads?)

---

## Phase 2: Deep Issue Investigation

### 2.1 All Open Issues

**Instructions:**
1. List all open issues (15+ mentioned, but verify actual count)
2. For each issue, extract:
   - Title
   - Author
   - Date opened
   - Labels/tags
   - Summary (1-2 sentences)
   - Current status (discussion ongoing? blocked? awaiting implementation?)
   - Links to related issues/PRs

**Output:** `TEMP/WEBML_OPEN_ISSUES_ANALYSIS.md`

### 2.2 All Closed Issues

**Instructions:**
1. List all closed issues
2. For each closed issue, extract:
   - Title
   - Author
   - Date opened/closed
   - Resolution (accepted? rejected? merged?)
   - Key discussion points (why accepted/rejected?)
   - Implementation status (if accepted)

**Output:** `TEMP/WEBML_CLOSED_ISSUES_ANALYSIS.md`

### 2.3 Pattern Recognition

**Analyze:**
1. **Successful proposals:** What made them accepted?
   - Common characteristics (format, depth, use cases, implementation clarity)
   - Sponsorship (Intel? Google? Other orgs?)
   - Technical scope (narrow? broad?)

2. **Rejected proposals:** Why were they rejected?
   - Out of scope?
   - Insufficient detail?
   - Overlapping with existing work?
   - Poor use case justification?

3. **Long-running discussions:** What causes proposals to stall?
   - Lack of consensus?
   - Implementation complexity?
   - Competing approaches?

**Output:** `TEMP/WEBML_PROPOSAL_PATTERNS.md`

---

## Phase 3: Gap Analysis

### 3.1 Current WebML Scope

**Extract from repository:**
- What does WebML currently cover? (API surface, supported operations)
- What's on the roadmap? (upcoming features, planned extensions)
- What's explicitly out of scope? (if documented)

### 3.2 Identify Gaps PM-KR Can Fill

**Questions:**
1. **Procedural reasoning gap:** Does WebML have proposals for lightweight reasoning (not just model execution)?
2. **Compression gap:** Are there proposals for procedural knowledge compression (vs traditional model compression)?
3. **Sustainability gap:** Are there proposals addressing carbon footprint reduction?
4. **Multi-modal gap:** Are there proposals for unified multi-modal reasoning substrates?
5. **Explainability gap:** Are there proposals for transparent, traceable reasoning (stack-based, auditable)?

**Output:** `TEMP/WEBML_GAPS_PM-KR_CAN_FILL.md`

---

## Phase 4: Competitive Analysis

### 4.1 Similar Proposals

**Search for:**
- Lightweight inference proposals
- Compression/optimization proposals
- Multi-modal reasoning proposals
- Explainable AI proposals

**For each similar proposal:**
- How is PM-KR's approach different?
- What's PM-KR's competitive advantage? (compression ratio, latency, carbon impact)
- How can PM-KR complement (not compete with) existing proposals?

**Output:** `TEMP/WEBML_COMPETITIVE_ANALYSIS.md`

---

## Phase 5: Proposal Draft (Comprehensive)

### 5.1 Proposal Structure

**Follow WebML's format** (extracted from repository analysis).

**Required sections (adapt based on repository template):**

1. **Title**
   - Clear, concise (under 80 chars)
   - Example: "Procedural Reasoning Substrates for WebNN: Lightweight Inference via Procedural Knowledge Representation"

2. **Abstract** (150-250 words)
   - Problem statement
   - Proposed solution
   - Key benefits (quantified)

3. **Motivation & Use Cases** (500-1000 words)
   - Real-world scenarios (browser-based reasoning, edge AI, sustainable inference)
   - Concrete examples (with before/after comparisons)
   - Performance benchmarks (latency, memory, carbon)

4. **Technical Specification** (1000-2000 words)
   - Data model (PM-KR procedural programs + metadata)
   - Execution semantics (RPN stack-based processor)
   - WebNN integration points (where does procedural layer fit?)
   - API surface (proposed extensions to WebNN)

5. **Performance Analysis** (500-1000 words)
   - Include only locally sourced metrics with explicit citations.
   - Candidate metrics to validate from docs (use only if confirmed):
     - Latency/throughput claims from `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md`
     - Compression ranges from `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md` and PM-KR evidence docs
     - Parameter-efficiency claims from `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md` / `docs/RESULTS_HISTORICAL.md`
     - Carbon projections from `docs/CARBON_BLUEPRINT_10_YEAR_PROJECTION.md` (label as projection)

6. **Comparison with Existing Approaches** (500 words)
   - How procedural reasoning differs from:
     - Traditional model execution (LLMs, CNNs)
     - Model compression techniques (quantization, pruning)
     - Knowledge distillation
   - **Complementary positioning:** "Use LLMs when needed, procedural when sufficient"

7. **Standards Alignment** (300-500 words)
   - PM-KR Community Group specification (Phase 1 in progress)
   - Implementation-neutral (WebGPU, Metal, Vulkan compatible)
   - Cross-CG synergy (CogAI, Sustainable Web IG)

8. **Implementation Plan** (500 words)
   - Phase 1: Data model finalization (April 30, 2026)
   - Phase 2: Execution semantics specification (June 30, 2026)
   - Phase 3: WebNN integration prototype (Q3 2026)
   - Phase 4: Conformance testing (Q4 2026)

9. **Security & Privacy Considerations** (300 words)
   - Boundary contracts (from Christoph's Sovereign Systems Charter synthesis)
   - Privacy/Transparency Dial (4-level policy)
   - Auditable execution (stack-based traceability)

10. **References**
    - PM-KR Charter: [link]
    - K3D GitHub: https://github.com/danielcamposramos/Knowledge3D
    - Carbon Blueprint: [link]
    - Phase 1 spec (draft): [link]
    - Relevant WebML issues: [list issues this proposal addresses]

**Output:** `TEMP/WEBML_PROPOSAL_DRAFT_PM-KR_PROCEDURAL_REASONING.md`

---

## Phase 6: Validation & Refinement

### 6.1 Self-Review Checklist

Before submitting to Daniel for review, verify:

- [ ] Proposal follows WebML's format (based on repository analysis)
- [ ] All claims are **grounded in evidence** (K3D measurements, PM-KR spec progress)
- [ ] Use cases are **concrete, not theoretical** (real browser/edge scenarios)
- [ ] **No overpromising:** Timeline is realistic (based on PM-KR Phase 1-4 roadmap)
- [ ] **Complementary positioning:** Not competing with WebML, extending it
- [ ] **Intel-friendly:** Aligns with Intel NPU roadmap, Anssi's priorities
- [ ] **Carbon impact quantified:** References CARBON_BLUEPRINT_10_YEAR_PROJECTION.md
- [ ] **Cross-references resolved:** All links work, all citations present
- [ ] **Professional tone:** W3C-appropriate, not marketing hype
- [ ] **Metric hygiene:** Every numeric claim mapped to a local source or removed

### 6.2 Gap Check

**Questions:**
1. Does this proposal address gaps identified in Phase 3?
2. Does this proposal differentiate from similar proposals (Phase 4)?
3. Does this proposal incorporate lessons from rejected proposals?
4. Does this proposal align with Christoph's boundary contracts + privacy/transparency principles?

---

## Deliverables Summary

**Primary outputs:**
1. `TEMP/WEBML_OPEN_ISSUES_ANALYSIS.md` (all open issues analyzed)
2. `TEMP/WEBML_CLOSED_ISSUES_ANALYSIS.md` (all closed issues analyzed)
3. `TEMP/WEBML_PROPOSAL_PATTERNS.md` (success/failure patterns)
4. `TEMP/WEBML_GAPS_PM-KR_CAN_FILL.md` (gap analysis)
5. `TEMP/WEBML_COMPETITIVE_ANALYSIS.md` (differentiation strategy)
6. **`TEMP/WEBML_PROPOSAL_DRAFT_PM-KR_PROCEDURAL_REASONING.md`** (final proposal draft)

**Secondary outputs:**
- Local clone of `webmachinelearning-proposals` repository
- Analysis notes (any additional findings worth documenting)

---

## Timeline

**Phase 1 (Repository Setup):** 30 minutes
**Phase 2 (Deep Issue Investigation):** 2-3 hours
**Phase 3 (Gap Analysis):** 1-2 hours
**Phase 4 (Competitive Analysis):** 1-2 hours
**Phase 5 (Proposal Draft):** 3-4 hours
**Phase 6 (Validation):** 1 hour

**Total estimated time:** 8-12 hours

**Deadline:** March 5-6, 2026 (give Daniel time to review before submitting)

---

## Success Criteria

**This proposal is successful if:**
1. **Grounded in WebML's actual needs** (not theoretical speculation)
2. **Impresses Intel/Anssi** with depth of understanding
3. **Positions PM-KR as complementary** (not competing) to WebML
4. **Quantifies benefits** (latency, compression, carbon) with evidence
5. **Provides clear implementation path** (realistic timeline, achievable milestones)
6. **Incorporates Christoph's sovereignty principles** (boundary contracts, privacy/transparency)
7. **Addresses known gaps** in WebML's current scope

---

## Notes for Codex

**Strategic context:**
- This is Intel (major W3C player) watching PM-KR's first formal proposal
- Anssi's response was formal/bureaucratic, not warm (we need to earn credibility)
- WebML has rigorous review process (proposals reviewed "from time to time" = selective)
- PM-KR is new CG (founded Feb 2026), needs to prove technical depth

**Christoph's guidance (applied):**
- "Begin from principled perspectives in everything" → Include boundary contracts, privacy/transparency dial
- "See if we can construct real systems around that" → Proposal includes implementation plan, not just theory
- "Fractal in K3D that will result in something new" → Show how procedural reasoning enables new capabilities, not just optimizes existing ones

**Daniel's priorities:**
- "Grounded truth" → All claims backed by K3D measurements, PM-KR spec progress
- "Impress Intel guys" → Professional, thorough, no handwaving
- "Once and for all" → Comprehensive proposal, not rushed/incomplete

---

**Ready when you are, Codex. Make PM-KR proud.** 🎯

---

**Assigned:** March 3, 2026
**Coordinator:** Daniel Ramos (PM-KR Co-Chair)
**Partner:** Claude (Architecture Support)
