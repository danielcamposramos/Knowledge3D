# PM-KR Specification Workspace — House Architecture

**Version:** 1.0
**Created:** March 3, 2026
**Architecture:** K3D House (Spatially-Aware Agent Workspace)

---

## 🏛️ What Is This?

This is a **House-based specification development workspace** — the first W3C Community Group to use **spatially-aware file organization** where agent behavior changes based on folder location (like TRM routing in K3D's House architecture).

**Each folder = Room with specific cognitive function:**
- 📚 **Library**: Reference material (READ-ONLY)
- 🔨 **Workshop**: Active drafting (READ-WRITE)
- 🛁 **Bathtub**: Introspection mode (META-ANALYSIS)
- 🪟 **Living Room**: Community interface (PUBLIC-FACING)
- 🏺 **Museum**: Historical archive (ARCHIVAL)

---

## 🧠 Agent Instructions (CRITICAL: Read This First!)

**BEFORE working in ANY subfolder:**
1. **Read the folder's README.md** (contains context-specific instructions)
2. **Switch to appropriate mode** (retrieval, creation, introspection, communication, archival)
3. **Follow folder constraints** (read-only vs read-write, allowed operations)

**Example:**
```bash
cd library/  # Agent switches to RETRIEVAL mode (read-only)
cd workshop/ # Agent switches to CREATION mode (read-write)
cd bathtub/  # Agent switches to INTROSPECTION mode (meta-analysis)
```

---

## 📂 Room Overview

### 📚 Library — Reference Material
**Purpose:** Prior art, related W3C specs, neuroscience research, community input
**Mode:** RETRIEVAL (read-only)
**Agent Specialist:** Citation specialist
**Location:** `library/`

### 🔨 Workshop — Active Drafting
**Purpose:** Draft PM-KR specifications (Data Model, Execution Semantics, Conformance, Integration)
**Mode:** CREATION (read-write)
**Agent Specialist:** Specification writer
**Location:** `workshop/`

### 🛁 Bathtub — Introspection Mode
**Purpose:** Meta-analysis, spec health reports, gap identification, cross-spec consistency
**Mode:** INTROSPECTION (read all rooms, suggest improvements)
**Agent Specialist:** Meta-analysis specialist
**Location:** `bathtub/`

### 🪟 Living Room — Community Interface
**Purpose:** Weekly summaries, community feedback, cross-CG collaboration documentation
**Mode:** COMMUNICATION (public-facing, concise)
**Agent Specialist:** Summarization specialist
**Location:** `living-room/`

### 🏺 Museum — Historical Archive
**Purpose:** Milestones, deprecated drafts, decision records (permanent preservation)
**Mode:** ARCHIVAL (read-only, provenance tracking)
**Agent Specialist:** Historical reference specialist
**Location:** `museum/`

---

## 🎯 Current Mission: PM-KR Community Group → Working Group

**Charter Goals:**
- Study **data models** for procedural knowledge representation
- Study **execution semantics** (RPN + TRM)
- Study **conformance levels** (Level 0: Read, Level 1: Execute, Level 2: Learn)
- Study **W3C integration** (RDF, OWL, JSON-LD interoperability)

**Target Timeline:**
- **Months 1-3**: Draft 4 specifications v0.1 (Workshop)
- **Months 2-4**: Build 3+ reference implementations (K3D, Christoph Encapsulate, WebNN)
- **Months 3-5**: Document 4+ use cases (multilingual, browser ML, enterprise, boundaries)
- **Months 4-6**: Community consensus + CG Final Report
- **Month 12**: Working Group Charter proposal

**Target:** 6-12 months CG → WG (vs typical 2-5 years) using AI-partnered development (MVCIC methodology).

---

## 🤖 Multi-Vibe Methodology (AI Partnership)

**This workspace demonstrates MVCIC (Multi-Vibe Coding in Chain):**

**Agent Roles:**
- **Claude (Architecture)**: Draft specifications, analyze gaps, generate summaries
- **Codex (Implementation)**: Validate specs with code, create conformance tests
- **Gemini (Cross-Validation)**: Challenge assumptions, identify gaps
- **Human (Daniel + Community)**: Orchestrate, community engagement, W3C protocol

**Workflow:**
```
Claude (Workshop) → Drafts specification sections
    ↓
Codex (Workshop) → Validates with code examples, conformance tests
    ↓
Claude (Bathtub) → Introspection: identifies gaps, cross-spec consistency
    ↓
Claude (Living Room) → Weekly summary, community engagement
    ↓
Human (Daniel) → Reviews, publishes to internal-pm-kr@w3.org
    ↓
Community → Feedback filed in Workshop/feedback/
    ↓
Claude (Workshop) → Iterates based on feedback
```

---

## 🚀 Getting Started

**For Agents (Claude/Codex/Gemini):**
1. Determine task type (draft spec, validate code, analyze gaps, generate summary)
2. Navigate to appropriate room (Workshop, Bathtub, Living Room)
3. Read room's README.md for context-specific instructions
4. Execute task following room constraints
5. Cross-reference other rooms as needed (Library for prior art, Museum for history)

**For Humans (Daniel, PM-KR Community):**
1. Review agent-generated content in Living Room (weekly summaries)
2. Provide feedback → file in Workshop/feedback/
3. Review specifications in Workshop
4. Archive milestones to Museum when phases complete

---

## 📊 Progress Tracking

**Weekly Health Reports:** `bathtub/spec-health-reports/`
**Community Summaries:** `living-room/weekly-summaries/`
**Active Specifications:** `workshop/phase{1,2,3,4}-*/spec-draft.md`

---

## 🏛️ Architectural Principle

**This workspace embodies K3D's House architecture:**

> "Just as the avatar in K3D navigates between specialized rooms (Library, Workshop, Bathtub) with context-specific TRM routing, agents in this workspace switch modes based on folder location. Spatial organization determines cognitive function."

**Result:** Spatially-aware AI agents that behave differently based on location — the first W3C specification workspace using House architecture.

---

**Next Step:** Navigate to `workshop/` to begin drafting PM-KR specifications! 🚀

---

**Last Updated:** March 3, 2026
**Maintainer:** Claude (Architecture Partner) + Daniel Ramos (PM-KR Co-Chair)
**Status:** Active Development (Founding Week + 7 days)
