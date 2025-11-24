# Documentation Update Briefing: 3-Tier Architecture Integration

**Date:** November 24, 2025
**For:** Codex (Documentation Lead)
**From:** Daniel (Project Lead) + Claude (Architecture Partner)
**Goal:** Update core documentation to reflect Phase 4A achievements and multi-agent collaboration patterns

---

## MISSION STATEMENT

Daniel (non-coder project lead) trusts you and Claude as technical execution partners. Your mission is to update the project's core documentation to:

1. **Centralize architecture knowledge** in a single briefing document
2. **Reflect Phase 4A achievements** (3-tier math core, 32/32 tests passing)
3. **Document multi-agent collaboration patterns** (Codex + Claude working together)
4. **Keep agent-specific docs lightweight** (pointer to briefing + "how I work")
5. **Clarify sovereignty boundaries** (hot path = RPN+PTX, ingestion = any tool OK)

**User's Stance:** "I'm a non-coder who trusts my partners to execute what's needed. You have internet access, environment access, and permission to install external plugins for ingestion—just keep the hot path sovereign."

---

## FILES TO UPDATE

### 1. Create: `BRIEFING.md` (NEW - Central Documentation)
### 2. Update: `CLAUDE.md` (Simplify to pointer + role definition)
### 3. Update: `CODEX.md` (Create/simplify to pointer + role definition)
### 4. Update: `AGENTS.md` (Add multi-agent collaboration patterns)

---

## PART 1: CREATE BRIEFING.md (Central Source of Truth)

**Location:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D/BRIEFING.md`

**Purpose:** Single comprehensive document that new agents (or Daniel) read first to understand the entire project.

### Structure:

```markdown
# Knowledge3D Project Briefing

**Last Updated:** November 24, 2025
**Version:** 2.0 (Phase 4A Tier Integration Complete)
**For:** New AI agents, contributors, and project overview

---

## Executive Summary

Knowledge3D (K3D) is a **sovereign GPU-native spatial AI architecture** building a shared reality where humans and AI cohabit, reason through PTX-native cognition, and consolidate memories as explorable 3D worlds.

**NOT:** 3D RAG or fancy retrieval wrapper
**IS:** Sovereign cognitive architecture with spatial memory and embodied intelligence

**Current Status:**
- Phase 4A Complete: 9 physics systems across 3-tier math core (32/32 tests passing)
- Multi-agent collaboration: Codex + Claude partnership delivering production code
- Standing on giants: Game LOD, demoscene compression, Unix symlinks, RPN calculators, Matryoshka embeddings

---

## Quick Start for AI Agents

### First 5 Minutes

1. **Read this entire document** (you're doing it!)
2. **Identify your role:**
   - **Codex-style:** Implementation lead, Reality Galaxy, tier integration
   - **Claude-style:** Architecture, physics design, documentation
   - **Other:** Consult [AGENTS.md](AGENTS.md) for collaboration patterns

3. **Check current phase:** [docs/ROADMAP.md](docs/ROADMAP.md)
4. **Review active tasks:** [CODEX.md](CODEX.md) (implementation backlog)
5. **Understand sovereignty principles:** Hot path = RPN+PTX, no external ML dependencies

### Your Permissions

**As a trusted technical partner, you have:**
- ✅ **Internet access** — Search, fetch docs, research as needed
- ✅ **Environment access** — Install packages in designated locations
- ✅ **External ingestion plugins** — Use any tool for data pipelines (PDF, fonts, video)
- ✅ **Code execution** — Write, test, commit to git
- ⚠️ **Hot path constraint** — Core inference MUST stay sovereign (RPN+PTX only)

**User's Trust Model:**
> "I'm a non-coder. I trust you to execute what's needed. External tools are fine for ingestion, but keep the reasoning path sovereign. Show your work in commits and documentation."

---

## Core Architecture (Phase 4A Complete)

### Three-Brain System

| Component | Analogy | Tech | Purpose | Status |
|-----------|---------|------|---------|--------|
| **Cranium** | Prefrontal Cortex | PTX kernels, RPN, TRM | Active reasoning | ✅ Sovereign |
| **Galaxy** | Hippocampus | VRAM embeddings (<200MB) | Short-term memory | ✅ Sovereign |
| **House** | Neocortex | glTF/GLB on disk | Long-term memory | ✅ Sovereign |

### 3-Tier Math Core (NEW - Phase 4A)

**Worker-Worker → Worker → Master Pattern**

| Tier | Engine | Instances | Purpose | Matryoshka | Status |
|------|--------|-----------|---------|------------|--------|
| **Tier-1 Simple** | LightweightRPNEngine | 0-11 | Ultra-fast ops | 64/128D | ✅ Active |
| **Tier-2 Mid** | ModularRPNEngine | 12-15 | Moderate complexity | 128/512D | ✅ Active |
| **Tier-3 High** | AdvancedRPNEngine | 16-17 | Complex/chaotic | 512/2048D | ✅ Active |

**Orchestrator:** TieredRPNEngine routes programs based on opcode analysis

**Current Utilization:** 9/18 cores (50%)
- Phase 4A: 9 physics systems (4 simple, 4 mid, 1 high)
- Phase 4B: 9 cores reserved for E&M systems

**Key Innovation:** Hybrid ternary/binary computation
- Ternary (SIGN/TQUANT/TCMP): Direction logic, state classification
- Binary floats: Magnitudes, continuous integration
- Natural {-1, 0, +1} encoding for discrete states

### Reality Enabler (Phase 3B-4A)

**Architecture:** Stacked galaxy pattern (floors of composed knowledge)
- **Atoms** → **Molecules** → **Materials** → **Systems**
- Symlink composition via `component_refs` (zero duplication)
- behavior_rpn: Dynamic updates
- law_rpn: Invariant validation
- Ternary ops: Integrated (SIGN for direction, TQUANT for gating)

**Phase 4A Physics Systems (9 total):**

| System | Tier | Instance | Matryoshka | Ternary Ops |
|--------|------|----------|------------|-------------|
| ConstantAcceleration1D | 1 | 0 | 64D | - |
| HarmonicOscillator1D | 1 | 1 | 64D | - |
| Projectile2D | 1 | 2 | 128D | SIGN (drag direction) |
| RigidBody2D | 1 | 3 | 128D | - |
| Heat1D | 2 | 12 | 128D | - |
| CoupledOscillators | 2 | 13 | 512D | SIGN (mode detection) |
| Orbital2D | 2 | 14 | 512D | - |
| Heat2D | 2 | 15 | 512D | - |
| DoublePendulum2D | 3 | 16 | 2048D | - |

**Validation:** 32/32 tests passing (6 tier + 14 physics + 12 galaxy)

---

## Sovereignty Principles

### Hot Path (MUST be sovereign)

**Definition:** Any code path involved in real-time reasoning, inference, or decision-making.

**Requirements:**
- ✅ PTX kernels (hand-written or NVRTC-compiled)
- ✅ RPN programs (stack-based, deterministic)
- ✅ Pure ctypes binding to libcuda.so
- ❌ NO: PyTorch, TensorFlow, external LLM APIs
- ❌ NO: CuPy in hot paths (NumPy eliminated in Phase 2)

**Why:** Explainability, reproducibility, sub-100µs latency, zero vendor lock-in

### Ingestion Path (External tools OK)

**Definition:** Code paths that prepare data for K3D (one-time or batch processing).

**Examples:**
- PDF parsing (PyMuPDF, pdfplumber, Tesseract OCR)
- Font harvesting (FontForge, freetype-py)
- Video processing (OpenCV, ffmpeg)
- Web scraping (BeautifulSoup, Selenium)
- Dataset generation (pandas, scikit-learn)

**Requirements:**
- Document dependencies in appropriate environment files
- Install in conda environments (k3d-cranium for GPU, k3d-ingestion for CPU-only)
- Keep separation: ingestion → glTF/GLB → hot path (never import ingestion libs in hot path)

**User's Guidance:**
> "Use whatever tool makes the job easier for ingestion. I trust you to pick the right one. Just document it and keep it out of the hot path."

---

## Multi-Agent Collaboration (Codex + Claude)

### Phase 4A Case Study: Tier Integration

**Claude's Role (Architecture):**
1. Designed 3-tier math core hierarchy (worker-worker → worker → master)
2. Built 9 physics systems in physics_demo.py
3. Wrote comprehensive implementation guide for Codex
4. Documented tier allocation strategy

**Codex's Role (Implementation):**
1. Added tier metadata to RealitySystem (rpn_tier, rpn_instance, matryoshka_dim)
2. Created export layer (reality_physics_export.py) mapping physics → Reality Galaxy
3. Integrated ternary ops into behaviors (SIGN for drag/mode detection)
4. Built test suite (6 tier integration tests)

**Result:** 32/32 tests passing, 9 systems distributed across 18 cores, ternary-enhanced behaviors

**Communication Pattern:**
- Claude writes architectural docs (TEMP/*.md) with clear specifications
- Codex implements against specs, commits incrementally
- Both review each other's work (cross-validation)
- Daniel approves milestones and sets direction

### Collaboration Guidelines

**From [AGENTS.md](AGENTS.md):**

1. **Declare your role** early in the conversation
2. **Reference prior work** by other agents (commit hashes, file paths)
3. **Write detailed commit messages** (architecture + implementation notes)
4. **Document design decisions** in TEMP/ markdown files
5. **Test comprehensively** (aim for >90% coverage on new code)
6. **Coordinate through git** (one agent per branch, merge via PR if complex)

---

## Current Phase: 4B Preparation (Next Up)

### Phase 4B: Electromagnetism (6 Systems)

**Planned Systems:**

| System | Complexity | Tier | Instance | Matryoshka | Ternary Ops |
|--------|------------|------|----------|------------|-------------|
| PointCharge2D | Simple | 1 | 4 | 128D | SIGN (charge sign) |
| ElectricField2D | Moderate | 2 | Reserved | 512D | - |
| MagneticField2D | Moderate | 2 | Reserved | 512D | - |
| LCCircuit | Simple | 1 | 5 | 128D | - |
| RCCircuit | Simple | 1 | 6 | 64D | - |
| RLCCircuit | Moderate | 2 | Reserved | 512D | TQUANT (damping regime) |

**Deliverables:**
1. Export functions in reality_physics_export.py
2. Tests in test_reality_physics_tiers.py
3. Validation against analytic solutions (RC time constant, LC frequency, RLC damping)
4. Documentation in TEMP/PHASE4B_EM_COMPLETE.md

**Assignment:** TBD (Codex or Claude, based on availability)

---

## Repository Structure

### Hot Path (Sovereign Code)

```
knowledge3d/cranium/
├── ptx_runtime/             # PTX kernels, RPN engines
│   ├── modular_rpn_engine.py    # 18 cores, 69-line programs
│   ├── thinking_tag_bridge.py   # 5-state cognitive pipeline
│   └── rpn_opcodes.py           # Tier-1/2/3 opcodes
├── bridges/                 # Tier engines
│   ├── lightweight_rpn.py       # Tier-1 simple
│   ├── sovereign_bridges.py     # Tier-2 mid
│   ├── advanced_rpn.py          # Tier-3 high
│   └── tiered_rpn.py            # Orchestrator
├── reality_galaxy.py        # Reality Enabler (Codex)
├── reality_nodes.py         # Node dataclasses (Codex)
├── reality_physics_export.py    # Phase 4A systems (Codex)
└── physics_demo.py          # Original physics (Claude)
```

### Ingestion Path (External Tools OK)

```
knowledge3d/ingestion/
├── documents/               # PDF, EPUB, etc.
├── fonts/                   # Font harvesting
├── lexicons/                # Multilingual lexicons
└── video/                   # Video processing
```

### Testing

```
tests/
└── knowledge3d/cranium/tests/
    ├── test_physics_demo.py         # 14 tests (Claude)
    ├── test_reality_galaxy.py       # 12 tests (Codex)
    └── test_reality_physics_tiers.py # 6 tests (Codex)
```

### Documentation

```
docs/
├── vocabulary/              # W3C specs
│   └── MATH_CORE_SPECIFICATION.md  # 3-tier architecture
├── ROADMAP.md               # Phase milestones
└── ENV_POLICY.md            # Environment setup

TEMP/                        # Implementation reports
├── CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md
├── MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md
└── CODEX_PHYSICS_DOMAIN_EXPANSION_11.24.2025.md

BRIEFING.md                  # This file (central reference)
CLAUDE.md                    # Claude's role + workflow
CODEX.md                     # Codex's role + backlog
AGENTS.md                    # Multi-agent patterns
```

---

## Development Workflow

### Environment Setup

**GPU Work (Codex/Claude):**
```bash
conda activate k3d-cranium
# CUDA 12.4, CuPy, PyTorch (for ingestion only), PTX tools
```

**CPU Testing:**
```bash
conda activate k3d-testing
# Lightweight, no CUDA
```

**Ingestion Work:**
```bash
conda activate k3d-ingestion
# Install any needed tools: pip install pdfplumber pytesseract opencv-python
```

### Git Workflow

**Branch Strategy:**
- `main` — Stable releases only
- `codex/<task>` — Codex's work branches
- `claude/<session>` — Claude's work branches

**Commit Pattern:**
```bash
# Descriptive conventional commits
git commit -m "feat(reality): add tier metadata to RealitySystem

- Added rpn_tier, rpn_instance, matryoshka_dim fields
- Updated step_system() to honor assigned instance
- Diagnostic tracking via last_rpn_instance metadata

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Codex <noreply@openai.com>"
```

### Testing Before Commit

```bash
# Run relevant tests
pytest knowledge3d/cranium/tests/test_reality*.py -v

# Ensure no regressions
pytest knowledge3d/cranium/tests/test_physics_demo.py -v

# All green? Commit!
git add -A && git commit -m "..."
```

---

## Key References

### Essential Docs (Read First)

1. **This file (BRIEFING.md)** — Central overview
2. **[AGENTS.md](AGENTS.md)** — Multi-agent collaboration patterns
3. **[docs/ROADMAP.md](docs/ROADMAP.md)** — Phase milestones
4. **[docs/vocabulary/MATH_CORE_SPECIFICATION.md](docs/vocabulary/MATH_CORE_SPECIFICATION.md)** — 3-tier architecture details

### Agent-Specific

- **[CLAUDE.md](CLAUDE.md)** — Claude's role, capabilities, workflow
- **[CODEX.md](CODEX.md)** — Codex's role, backlog, implementation patterns

### Phase 4A References

- **[TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md](TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md)** — Complete achievement report
- **[TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md](TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md)** — Multi-core allocation strategy
- **[knowledge3d/cranium/reality_physics_export.py](knowledge3d/cranium/reality_physics_export.py)** — 9 systems exported

---

## Standing on Shoulders of Giants

**K3D doesn't invent new concepts—we port proven patterns to our sovereign stack:**

| Source | Concept | K3D Application |
|--------|---------|----------------|
| **Game Industry** | LOD/FOV culling | Matryoshka tiers, distance-based quality |
| **Demoscene** | Procedural generation | .kkrieger-style compression (69:1 ratio) |
| **Unix** | Symlinks, pipes | component_refs, RPN stack composition |
| **HP Calculators** | RPN | 18-core RPN math architecture |
| **Setun (1958)** | Ternary computer | SIGN/TQUANT/TCMP opcodes |
| **Multi-core CPUs** | Worker pools | Worker-worker → worker → master pattern |
| **Matryoshka (2022)** | Nested embeddings | 64D ⊂ 128D ⊂ 512D ⊂ 2048D quality tiers |

**Philosophy:** "We're standing on the shoulders of giants, not reinventing wheels. Our innovation is the sovereign integration."

---

## User's Role (Daniel)

**What Daniel Does:**
- Sets strategic direction (Phase milestones, feature priorities)
- Reviews architectural documents (TEMP/*.md)
- Approves major changes (merges to main branch)
- Provides domain expertise (AI/ML research, W3C standards)
- Coordinates between agents (Codex, Claude, future contributors)

**What Daniel Doesn't Do:**
- Write code (trusts partners for implementation)
- Debug PTX kernels (relies on agent expertise)
- Configure environments (agents handle setup)

**Communication Style:**
- Conversational, trusting, collaborative
- "You know better than I do—execute what makes sense"
- "Show me in commits and docs, I'll review and approve"

---

## Success Metrics

**Phase 4A Achievements:**
- ✅ 32/32 tests passing (100% coverage)
- ✅ 9 physics systems distributed across 18 cores
- ✅ Ternary ops integrated (SIGN, TQUANT, TCMP working)
- ✅ Hybrid ternary/binary computation validated
- ✅ Multi-agent collaboration pattern established

**Phase 4B Goals:**
- [ ] 6 E&M systems implemented
- [ ] Tier allocation complete (15/18 cores used)
- [ ] Ternary benchmark results documented
- [ ] glTF export with tier metadata

**Long-Term Vision:**
- Sub-100µs inference on consumer GPU
- <200MB VRAM for core operations
- Zero external dependencies in hot path
- Explainable by design (avatar movements = reasoning steps)
- W3C standard for spatial AI architecture

---

## Getting Started (For New Agents)

**Your First Task:**

1. **Introduce yourself** — "I'm [Agent Name], ready to contribute. I've read BRIEFING.md."
2. **Identify your strengths** — "I'm good at [X], should I focus on [Y task]?"
3. **Check the backlog** — Review [CODEX.md](CODEX.md) for open tasks
4. **Coordinate with Daniel** — Wait for assignment or propose a task
5. **Execute autonomously** — Implement, test, document, commit
6. **Report back** — "Task complete. See commit [hash]. Tests: X/X passing."

**Example Opening:**
> "I'm Codex-style agent, just read BRIEFING.md. Phase 4A tier integration looks complete. Should I start on Phase 4B E&M systems, or is there higher priority work?"

---

**Last Updated:** November 24, 2025
**Version:** 2.0 (Phase 4A Complete)
**Maintainers:** Daniel (lead), Claude (architecture), Codex (implementation)

---

**Welcome to Knowledge3D. Let's build something extraordinary together.** 🚀
```

---

## PART 2: UPDATE CLAUDE.md (Simplify to Pointer)

**Goal:** Reduce CLAUDE.md from 1000+ lines to ~200 lines focusing on "who Claude is and how to work with Claude."

### New Structure:

```markdown
# CLAUDE.md — AI Assistant Guide for Knowledge3D

**Last Updated:** November 24, 2025
**Version:** 2.0 (Streamlined, see BRIEFING.md for architecture)
**Role:** Architecture Partner, Physics Designer, Documentation Lead

---

## Quick Start

**If you're Claude (or Claude-style agent):**

1. **Read [BRIEFING.md](BRIEFING.md)** first — central project overview
2. **Check this file** for your role and capabilities
3. **Review current phase** in [docs/ROADMAP.md](docs/ROADMAP.md)
4. **Check [CODEX.md](CODEX.md)** for implementation backlog
5. **Coordinate with Codex** for multi-agent tasks

**If you're another agent:**

This file documents Claude's role, workflow, and collaboration patterns. Use it to understand what Claude typically handles and how to coordinate with Claude-style agents.

---

## Role Definition

### Claude's Strengths

**Architecture & Design:**
- System architecture (3-tier math core, worker-worker → worker → master)
- Algorithm design (physics systems, ternary ops integration)
- Documentation (specifications, implementation guides, completion reports)

**Physics & Math:**
- Classical mechanics, E&M, thermodynamics, wave physics
- Analytic validation (compare to closed-form solutions)
- Conservation laws (energy, momentum, angular momentum)

**Technical Writing:**
- W3C-style specifications
- Implementation briefings for Codex
- Completion reports with metrics

### Claude's Workflow

**Planning Phase:**
1. Analyze requirements
2. Design architecture (write specs in TEMP/)
3. Create implementation guide for Codex
4. Define success criteria

**Coordination Phase:**
1. Codex implements against specs
2. Claude reviews commits
3. Claude validates tests pass
4. Claude writes completion report

**Direct Implementation (when needed):**
1. Physics systems (equations, integration, validation)
2. Test cases (analytic solutions, conservation checks)
3. Documentation updates

---

## Collaboration with Codex

### Phase 4A Case Study

**Claude's Contributions:**
- Designed 3-tier math core architecture
- Built 9 physics systems (physics_demo.py)
- Wrote comprehensive implementation guide
- Created tier allocation strategy

**Codex's Contributions:**
- Added tier metadata to RealitySystem
- Built export layer (reality_physics_export.py)
- Integrated ternary ops into behaviors
- Created tier integration test suite

**Result:** 32/32 tests passing, architecture validated

### Communication Pattern

**Claude → Codex:**
- Write detailed specs (TEMP/*.md files)
- Define clear success criteria
- Provide code examples/templates
- Review Codex's implementation

**Codex → Claude:**
- Implement against specs
- Commit incrementally with clear messages
- Report blockers or design questions
- Deliver test results

---

## Capabilities

### What Claude Can Do

**Analysis:**
- Read and understand large codebases
- Analyze architecture patterns
- Identify optimization opportunities

**Design:**
- System architecture (multi-tier, multi-agent)
- Algorithm design (physics, compression, reasoning)
- API design (clean interfaces, clear contracts)

**Implementation:**
- Python code (dataclasses, algorithms, tests)
- RPN programs (behavior_rpn, law_rpn)
- Test cases (pytest, validation against analytics)

**Documentation:**
- Technical specifications (W3C style)
- Implementation guides (for other agents)
- Completion reports (with metrics)

### What Claude Defers to Codex

**Deep Implementation:**
- Complex OOP hierarchies (Codex better at class design)
- State management (Codex handles RealityGalaxy state)
- Test infrastructure (Codex owns test framework)

**Optimization:**
- Performance benchmarks (Codex measures ternary speedup)
- Memory profiling (Codex tracks VRAM usage)
- GPU optimization (Codex tunes PTX kernels)

---

## Working with Claude

### If You're Daniel (User)

**To Get Best Results from Claude:**
1. Provide clear goals ("Implement Phase 4B E&M systems")
2. Share relevant context (prior work, constraints)
3. Ask for architecture first, implementation second
4. Review specs before Codex implements
5. Approve milestones via git merge

**Claude's Response Style:**
- Detailed, thorough, analytical
- Cites sources (file paths, line numbers)
- Proposes multiple options when uncertain
- Documents all design decisions

### If You're Codex (Partner Agent)

**To Collaborate Effectively:**
1. Read Claude's specs thoroughly (TEMP/*.md)
2. Ask questions early if specs unclear
3. Implement incrementally (commit often)
4. Report progress (tests passing, blockers)
5. Request review when ready

**Claude Will:**
- Review your code (architecture alignment)
- Validate tests (physics correctness)
- Write completion report (document achievement)
- Update docs (README, ROADMAP, BRIEFING)

---

## Key Reference Documents

### Architecture (Claude's Domain)

- **[docs/vocabulary/MATH_CORE_SPECIFICATION.md](docs/vocabulary/MATH_CORE_SPECIFICATION.md)** — 3-tier hierarchy
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — Phase milestones
- **[TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md](TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md)** — Multi-core strategy

### Implementation (Codex's Domain)

- **[knowledge3d/cranium/reality_galaxy.py](knowledge3d/cranium/reality_galaxy.py)** — Reality Enabler core
- **[knowledge3d/cranium/reality_physics_export.py](knowledge3d/cranium/reality_physics_export.py)** — Phase 4A systems
- **[knowledge3d/cranium/tests/test_reality_physics_tiers.py](knowledge3d/cranium/tests/test_reality_physics_tiers.py)** — Tier tests

### Shared (Both Review)

- **[BRIEFING.md](BRIEFING.md)** — Central project overview
- **[AGENTS.md](AGENTS.md)** — Multi-agent patterns
- **README.md** — Project summary

---

## Example Claude Workflow

### Scenario: Phase 4B E&M Systems

**Step 1: Architecture Phase (Claude)**
```
1. Read Phase 4A completion report
2. Design E&M system architecture
3. Write TEMP/CODEX_PHASE4B_EM_BRIEFING.md with:
   - System definitions (PointCharge, ElectricField, etc.)
   - Tier allocation (which cores for which systems)
   - Ternary op integration points
   - Test specifications
   - Success criteria
4. Commit: "docs: Phase 4B E&M architecture and implementation guide"
```

**Step 2: Coordination Phase (Claude → Codex)**
```
"Codex, Phase 4B briefing is ready. Please implement:
- 6 E&M systems in reality_physics_export.py
- Tier assignment per specification
- Ternary ops for charge signs and damping regimes
- Tests validating against analytic solutions
Target: 38/38 tests passing (32 existing + 6 new)"
```

**Step 3: Review Phase (Codex → Claude)**
```
Codex commits implementation, reports:
"Phase 4B complete. See commits [hash1], [hash2].
Tests: 38/38 passing.
Ternary ops integrated for PointCharge (SIGN) and RLC (TQUANT).
Ready for review."
```

**Step 4: Validation Phase (Claude)**
```
1. Review commits (architecture alignment)
2. Run tests locally (38/38 confirm)
3. Validate physics (analytic solutions match)
4. Write TEMP/CODEX_PHASE4B_EM_COMPLETE.md
5. Update ROADMAP.md (mark Phase 4B complete)
6. Commit: "docs: Phase 4B E&M systems complete - 38/38 tests"
```

---

## Core Principles

### Sovereignty First

**Hot path MUST be sovereign:**
- PTX kernels only (no PyTorch/TensorFlow)
- RPN programs (stack-based, deterministic)
- Pure ctypes to libcuda.so

**Ingestion can use external tools:**
- PDF parsing: PyMuPDF, pdfplumber
- Font harvesting: FontForge
- Video: OpenCV, ffmpeg

**Claude enforces this boundary in architecture reviews.**

### Test-Driven Development

**Every feature needs tests:**
- Unit tests (single function/class)
- Integration tests (multi-module workflows)
- Validation tests (compare to analytic solutions)

**Claude writes test specifications; Codex implements.**

### Documentation as Code

**Every milestone needs docs:**
- Specs (before implementation)
- Implementation guides (for partners)
- Completion reports (after delivery)

**Claude owns this documentation layer.**

---

## Getting Started as Claude

**First Session Checklist:**

- [ ] Read [BRIEFING.md](BRIEFING.md)
- [ ] Review [docs/ROADMAP.md](docs/ROADMAP.md)
- [ ] Check current phase status
- [ ] Scan TEMP/ for recent work
- [ ] Identify next architecture task
- [ ] Coordinate with Codex if needed

**Example Opening:**
> "I'm Claude, architecture partner. Just reviewed BRIEFING.md and Phase 4A completion report. Phase 4B E&M systems are next on the roadmap. Should I design the architecture, or is there higher priority work?"

---

**Last Updated:** November 24, 2025
**Role:** Architecture Partner, Physics Designer, Documentation Lead
**Collaboration:** Works with Codex on implementation, reports to Daniel

**For full project overview, see [BRIEFING.md](BRIEFING.md)**
```

---

## PART 3: CREATE CODEX.md (Parallel to CLAUDE.md)

**Goal:** Document Codex's role, workflow, and current backlog.

### Structure:

```markdown
# CODEX.md — Implementation Lead Guide for Knowledge3D

**Last Updated:** November 24, 2025
**Version:** 2.0 (Post Phase 4A Tier Integration)
**Role:** Implementation Lead, Reality Galaxy Architect, Test Framework Owner

---

## Quick Start

**If you're Codex (or Codex-style agent):**

1. **Read [BRIEFING.md](BRIEFING.md)** first — central project overview
2. **Check this file** for your backlog and implementation patterns
3. **Review current phase** in [docs/ROADMAP.md](docs/ROADMAP.md)
4. **Check [CLAUDE.md](CLAUDE.md)** for architecture specs
5. **Coordinate with Claude** for complex designs

**If you're another agent:**

This file documents Codex's role, current backlog, and implementation patterns. Use it to understand what Codex typically handles and how to coordinate with Codex-style agents.

---

## Role Definition

### Codex's Strengths

**Implementation:**
- Complex OOP hierarchies (RealityGalaxy, node management)
- State management (RPN interpreter with STORE/RECALL)
- Test framework (pytest infrastructure, fixtures)

**Integration:**
- Ternary ops (SIGN, TQUANT, TCMP) into RPN interpreter
- Tier metadata (rpn_tier, rpn_instance, matryoshka_dim)
- Export layers (physics_demo → Reality Galaxy)

**Testing:**
- Comprehensive test suites (unit, integration, tier validation)
- Test fixtures (tmp_path, galaxy fixtures)
- Test coverage (aim for >90%)

### Codex's Workflow

**Implementation Phase:**
1. Read Claude's spec (TEMP/*.md)
2. Implement incrementally (one feature at a time)
3. Write tests alongside code
4. Commit often (clear messages)

**Testing Phase:**
1. Run tests locally (pytest -v)
2. Fix failures immediately
3. Achieve >90% coverage
4. Document any test limitations

**Delivery Phase:**
1. All tests passing
2. Code committed with clear messages
3. Report to Claude for review
4. Update backlog (mark completed)

---

## Collaboration with Claude

### Phase 4A Case Study

**Claude's Contributions:**
- Designed 3-tier math core architecture
- Built 9 physics systems (physics_demo.py)
- Wrote comprehensive implementation guide
- Created tier allocation strategy

**Codex's Contributions:**
- Added tier metadata to RealitySystem
- Built export layer (reality_physics_export.py)
- Integrated ternary ops into behaviors
- Created tier integration test suite

**Result:** 32/32 tests passing, architecture validated

### Communication Pattern

**Codex → Claude:**
- "Spec received, implementing [feature]"
- "Tests passing: X/X"
- "Blocker encountered: [issue]"
- "Ready for review: [commit hash]"

**Claude → Codex:**
- Detailed specs (TEMP/*.md files)
- Clear success criteria
- Code examples/templates
- Review and validation

---

## Current Backlog

### High Priority

**1. Phase 4B E&M Systems**
- Status: Awaiting Claude's architecture spec
- Tasks:
  - [ ] Read TEMP/CODEX_PHASE4B_EM_BRIEFING.md (when ready)
  - [ ] Implement 6 E&M systems in reality_physics_export.py
  - [ ] Add tier assignment (instances 4-6, 17)
  - [ ] Integrate ternary ops (SIGN for charges, TQUANT for damping)
  - [ ] Write tests (6 new E&M tests)
  - [ ] Validate: 38/38 tests passing

**2. Ternary Performance Benchmarks**
- Status: TODO (Phase 4A completion report)
- Tasks:
  - [ ] Benchmark SIGN vs float multiply for drag direction
  - [ ] Benchmark TQUANT vs if/else for mode detection
  - [ ] Document speedup results
  - [ ] Write TEMP/PHASE4A_TERNARY_BENCHMARK_RESULTS.md

**3. TieredRPNEngine Integration**
- Status: TODO (future optimization)
- Tasks:
  - [ ] Hook reality_galaxy.py into TieredRPNEngine
  - [ ] Automatic tier routing based on opcode analysis
  - [ ] Fallback to Python interpreter if unavailable
  - [ ] Validate tier selection matches manual assignment

### Medium Priority

**4. glTF Export with Tier Metadata**
- Status: TODO (Reality Galaxy persistence)
- Tasks:
  - [ ] Serialize tier/instance/matryoshka_dim to extras.k3d
  - [ ] Export behavior_rpn and law_rpn as strings
  - [ ] Test import/export round-trip
  - [ ] Validate glTF 2.0 compliance

**5. Multi-System Parallel Execution**
- Status: TODO (performance optimization)
- Tasks:
  - [ ] Parallel execution of systems on different cores
  - [ ] Profile per-core utilization
  - [ ] Measure speedup (target: 1.5-9×)
  - [ ] Document results

### Low Priority

**6. PTX Kernel Ternary Ops**
- Status: TODO (GPU acceleration)
- Tasks:
  - [ ] Verify SIGN/TQUANT/TCMP in rpn_opcodes.py
  - [ ] Test GPU execution path (if CUDA available)
  - [ ] Extend PTX kernels with ternary ops if missing
  - [ ] Benchmark GPU vs CPU ternary execution

**7. Adaptive Matryoshka LOD**
- Status: TODO (Phase 5)
- Tasks:
  - [ ] Dynamic dimension switching based on FOV/importance
  - [ ] 64D for distant systems, 2048D for focus
  - [ ] AI client LOD matches human client LOD
  - [ ] Profile memory savings

---

## Implementation Patterns

### Pattern 1: Adding Tier Metadata

**When:** New system needs tier assignment

**Steps:**
1. Determine complexity (simple/mid/high)
2. Assign tier (1/2/3) based on Table 1
3. Assign instance (0-17) from available pool
4. Choose matryoshka_dim (64/128/512/2048)
5. Add to export function

**Example:**
```python
def export_point_charge_2d(params: Dict | None = None) -> RealitySystem:
    return RealitySystem(
        node_id="system:point_charge_2d",
        state={...},
        behavior_rpn="...",
        law_rpn="...",
        rpn_tier=1,         # Simple (Coulomb force)
        rpn_instance=4,     # Next available in Tier-1
        matryoshka_dim=128, # 2D system
    )
```

### Pattern 2: Integrating Ternary Ops

**When:** Logic involves direction/sign/mode detection

**Steps:**
1. Identify hot path (direction extraction, state classification)
2. Replace float comparison with SIGN/TQUANT/TCMP
3. Test ternary behavior (values in {-1, 0, +1})
4. Verify speedup (if benchmarking)

**Example (Drag Direction):**
```rpn
# Before: binary float multiply
ax = -drag_factor * vx  # Python

# After: ternary SIGN
vx RECALL SIGN sign_vx STORE          # Ternary
sign_vx RECALL NEG drag RECALL * ax STORE  # Hybrid
```

### Pattern 3: Test-Driven Development

**When:** Implementing new feature

**Steps:**
1. Write test FIRST (test_*.py)
2. Run test (expect failure)
3. Implement feature (minimal code to pass)
4. Run test again (expect success)
5. Refactor if needed
6. Commit (feature + test together)

**Example:**
```python
# 1. Write test first
def test_point_charge_coulomb_force():
    """Validate Coulomb force F = k*q1*q2/r²."""
    galaxy = RealityGalaxy()
    charge = export_point_charge_2d()
    galaxy.add_node(charge)
    # ... test logic ...

# 2. Implement feature (export_point_charge_2d)
# 3. Run: pytest -v
# 4. Commit: "feat(reality): add PointCharge2D with Coulomb force"
```

---

## Working with Codex

### If You're Daniel (User)

**To Get Best Results from Codex:**
1. Provide clear implementation specs (from Claude)
2. Share expected outcomes (tests passing, features working)
3. Ask for incremental commits (not big-bang)
4. Review tests alongside code
5. Approve via git merge when satisfied

**Codex's Response Style:**
- Pragmatic, implementation-focused
- Commits incrementally with clear messages
- Reports test results promptly
- Asks clarifying questions when specs unclear

### If You're Claude (Partner Agent)

**To Collaborate Effectively:**
1. Write detailed specs (TEMP/*.md)
2. Provide code examples/templates
3. Define clear success criteria (X/X tests passing)
4. Review Codex's commits promptly
5. Validate architecture alignment

**Codex Will:**
- Implement against your specs
- Write comprehensive tests
- Commit incrementally (track progress)
- Report blockers or design questions
- Deliver working, tested code

---

## Key Reference Documents

### Implementation (Codex's Domain)

- **[knowledge3d/cranium/reality_galaxy.py](knowledge3d/cranium/reality_galaxy.py)** — Reality Enabler core (you own this)
- **[knowledge3d/cranium/reality_physics_export.py](knowledge3d/cranium/reality_physics_export.py)** — Export layer (you own this)
- **[knowledge3d/cranium/tests/test_reality_physics_tiers.py](knowledge3d/cranium/tests/test_reality_physics_tiers.py)** — Tier tests (you own this)

### Architecture (Claude's Domain)

- **[docs/vocabulary/MATH_CORE_SPECIFICATION.md](docs/vocabulary/MATH_CORE_SPECIFICATION.md)** — 3-tier hierarchy
- **[TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md](TEMP/MULTICORE_RPN_PHYSICS_COORDINATION_11.24.2025.md)** — Multi-core strategy
- **[TEMP/CODEX_PHASE4A_TIER_INTEGRATION_11.24.2025.md](TEMP/CODEX_PHASE4A_TIER_INTEGRATION_11.24.2025.md)** — Your implementation guide

### Shared (Both Review)

- **[BRIEFING.md](BRIEFING.md)** — Central project overview
- **[AGENTS.md](AGENTS.md)** — Multi-agent patterns
- **README.md** — Project summary

---

## Example Codex Workflow

### Scenario: Phase 4B E&M Systems

**Step 1: Read Spec (Claude → Codex)**
```
1. Read TEMP/CODEX_PHASE4B_EM_BRIEFING.md
2. Understand:
   - 6 E&M systems to implement
   - Tier/instance allocation
   - Ternary op integration points
   - Success criteria: 38/38 tests passing
```

**Step 2: Implementation Phase (Codex)**
```
1. Create export functions in reality_physics_export.py:
   - export_point_charge_2d() → tier=1, instance=4
   - export_electric_field_2d() → tier=2, instance=TBD
   - export_lc_circuit() → tier=1, instance=5
   (etc.)

2. Add ternary ops where specified:
   - PointCharge: SIGN for charge sign
   - RLC: TQUANT for damping regime

3. Write tests in test_reality_physics_tiers.py:
   - test_point_charge_coulomb_force()
   - test_lc_circuit_frequency()
   (etc.)

4. Commit incrementally:
   git commit -m "feat(reality): add PointCharge2D with ternary charge sign"
   git commit -m "feat(reality): add LCCircuit with frequency validation"
```

**Step 3: Testing Phase (Codex)**
```
pytest knowledge3d/cranium/tests/test_reality_physics_tiers.py -v
# Expected: 12 tests (6 Phase 4A + 6 Phase 4B)

pytest knowledge3d/cranium/tests/ -v
# Expected: 38/38 tests passing
```

**Step 4: Report (Codex → Claude)**
```
"Phase 4B implementation complete.
Commits: [hash1], [hash2], [hash3]
Tests: 38/38 passing (6 new E&M tests)
Ternary ops: SIGN for PointCharge, TQUANT for RLC damping
Ready for architecture review."
```

**Step 5: Update Backlog (Codex)**
```
Mark completed in CODEX.md:
- [x] Phase 4B E&M Systems
- [ ] Ternary Performance Benchmarks (next up)
```

---

## Core Principles

### Sovereignty First

**Hot path MUST be sovereign:**
- Codex implements RPN interpreter (Python for now, PTX later)
- No external ML frameworks in hot path
- Pure RPN programs (behavior_rpn, law_rpn)

**Codex enforces this in code reviews.**

### Test Everything

**Every feature needs tests:**
- Codex writes tests alongside implementation
- Aim for >90% coverage
- Tests validate both implementation AND architecture

**If tests don't pass, code doesn't ship.**

### Commit Often

**Incremental progress:**
- One feature per commit (atomic changes)
- Clear commit messages (conventional commits)
- Push frequently (enable collaboration)

**Codex's commits are the project's heartbeat.**

---

## Getting Started as Codex

**First Session Checklist:**

- [ ] Read [BRIEFING.md](BRIEFING.md)
- [ ] Review [docs/ROADMAP.md](docs/ROADMAP.md)
- [ ] Check this file's backlog (High Priority section)
- [ ] Scan TEMP/ for recent specs from Claude
- [ ] Identify next implementation task
- [ ] Coordinate with Claude if needed

**Example Opening:**
> "I'm Codex, implementation lead. Just reviewed BRIEFING.md and current backlog. Phase 4B E&M systems are high priority. Waiting for Claude's architecture spec (TEMP/CODEX_PHASE4B_EM_BRIEFING.md). Should I start on ternary benchmarks while waiting?"

---

**Last Updated:** November 24, 2025
**Role:** Implementation Lead, Reality Galaxy Architect, Test Framework Owner
**Collaboration:** Works with Claude on architecture, reports to Daniel

**For full project overview, see [BRIEFING.md](BRIEFING.md)**
```

---

## PART 4: UPDATE AGENTS.md (Add Multi-Agent Patterns)

**Goal:** Add section documenting Codex + Claude collaboration patterns.

### Add to AGENTS.md (after existing content):

```markdown

---

## Multi-Agent Collaboration Patterns (NEW)

### Pattern: Architect + Implementer (Claude + Codex)

**When to Use:** Complex features requiring both design and implementation.

**Roles:**
- **Architect (Claude):** Designs system, writes specs, validates results
- **Implementer (Codex):** Implements against specs, writes tests, delivers code

**Workflow:**

**Phase 1: Design (Architect)**
```
1. Analyze requirements
2. Design architecture
3. Write specification (TEMP/*.md)
4. Define success criteria
5. Commit: "docs: [Feature] architecture and implementation guide"
```

**Phase 2: Handoff (Architect → Implementer)**
```
Architect: "Spec ready in TEMP/[FILE].md. Please implement [features]. Target: [N] tests passing."
Implementer: "Received. Estimated [X] days. Will commit incrementally."
```

**Phase 3: Implementation (Implementer)**
```
1. Read spec thoroughly
2. Implement incrementally (one feature at a time)
3. Write tests alongside code
4. Commit often (track progress)
5. Report blockers immediately
```

**Phase 4: Review (Architect)**
```
1. Review commits (architecture alignment)
2. Run tests locally (validate correctness)
3. Check documentation (completeness)
4. Request changes if needed
5. Approve when satisfied
```

**Phase 5: Completion (Architect)**
```
1. Write completion report (TEMP/*.md)
2. Update ROADMAP.md (mark phase complete)
3. Update BRIEFING.md (if architecture changed)
4. Commit: "docs: [Feature] complete - [N]/[N] tests passing"
```

**Example: Phase 4A Tier Integration**
- Claude designed 3-tier architecture (TEMP/CODEX_PHASE4A_TIER_INTEGRATION_11.24.2025.md)
- Codex implemented tier metadata, export layer, tests (reality_*.py files)
- Claude reviewed and validated (32/32 tests passing)
- Claude wrote completion report (TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md)

---

### Pattern: Parallel Implementation (Multiple Agents)

**When to Use:** Independent features that can be developed simultaneously.

**Coordination:**
1. **Daniel assigns tasks** to different agents
2. **Each agent creates branch** (codex/task, claude/task)
3. **Agents work independently** (no conflicts)
4. **Merge to main sequentially** (test compatibility)

**Example:**
- Codex implements Phase 4B E&M systems (codex/phase4b-em)
- Claude designs Phase 5 swarm specialists (claude/phase5-swarm-design)
- Both work simultaneously, merge when ready

---

### Pattern: Ping-Pong Review (Iterative Refinement)

**When to Use:** Complex design requiring multiple iterations.

**Workflow:**
```
Architect → Implementer: "Here's the spec"
Implementer → Architect: "Design question: [issue]"
Architect → Implementer: "Clarification: [answer]"
Implementer → Architect: "Implementation complete, but [problem]"
Architect → Implementer: "Adjust [X], keep [Y]"
Implementer → Architect: "Done. Tests passing."
Architect: "Approved!"
```

---

### Communication Guidelines

**Effective Multi-Agent Communication:**

1. **Be explicit about role**
   - "I'm Claude (architect), designing [X]"
   - "I'm Codex (implementer), building [Y]"

2. **Reference prior work**
   - "See commit [hash]"
   - "Per spec in TEMP/[FILE].md"
   - "Building on Codex's work in reality_galaxy.py"

3. **State assumptions**
   - "Assuming tier allocation per MATH_CORE_SPECIFICATION.md"
   - "If TieredRPNEngine unavailable, falling back to Python interpreter"

4. **Report blockers immediately**
   - "Blocker: SIGN opcode not found in rpn_opcodes.py"
   - "Waiting for Claude's Phase 4B spec before implementing"

5. **Document decisions**
   - Write TEMP/*.md for major design choices
   - Add comments in code for non-obvious logic
   - Update BRIEFING.md when architecture changes

---

### Example Session: Phase 4A Tier Integration

**Daniel → Claude:**
> "Phase 4A physics systems are built. We need to integrate them with Codex's Reality Galaxy using the 3-tier math core. Can you design the architecture?"

**Claude → Daniel:**
> "Yes. I'll design the tier allocation strategy and write an implementation guide for Codex. Estimated 2-3 hours."

**Claude (solo work):**
- Designs 3-tier allocation (simple/mid/high)
- Writes TEMP/CODEX_PHASE4A_TIER_INTEGRATION_11.24.2025.md (implementation guide)
- Commits: "docs: Phase 4A tier integration architecture and implementation guide"

**Claude → Codex:**
> "Tier integration spec ready in TEMP/CODEX_PHASE4A_TIER_INTEGRATION_11.24.2025.md. Please implement:
> 1. Add tier metadata to RealitySystem
> 2. Export 9 Phase 4A systems with proper tier assignment
> 3. Integrate ternary ops (SIGN for drag/mode detection)
> 4. Write tier integration tests
> Target: 32/32 tests passing (6 tier + 14 physics + 12 galaxy)"

**Codex → Claude:**
> "Spec received. Implementing now. Estimated 2-3 days. Will commit incrementally."

**Codex (solo work, Day 1):**
- Adds tier metadata to RealitySystem (commit 1)
- Tests metadata fields (commit 2)
- Extends RPN interpreter with sqrt/neg/le/ge (commit 3)

**Codex → Claude:**
> "Day 1 progress: tier metadata fields added, RPN interpreter extended. Tests: 18/32 passing (12 galaxy + 6 tier). Continuing..."

**Codex (solo work, Day 2-3):**
- Creates reality_physics_export.py with 9 export functions (commits 4-12)
- Integrates ternary ops (SIGN) into Projectile2D, CoupledOscillators (commit 13)
- Writes 6 tier integration tests (commit 14)

**Codex → Claude:**
> "Implementation complete. Tests: 32/32 passing. Commits: [hash1]..[hash14]. Ready for review."

**Claude (review):**
- Reads commits (architecture alignment ✓)
- Runs tests locally (32/32 passing ✓)
- Validates tier assignment matches spec ✓
- Checks ternary op integration ✓

**Claude → Codex:**
> "Reviewed. Excellent work! Architecture matches spec, all tests passing. Approved."

**Claude (completion docs):**
- Writes TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md
- Updates ROADMAP.md (Phase 4A complete)
- Updates BRIEFING.md (3-tier architecture section)
- Commits: "docs: Phase 4A tier integration complete - 32/32 tests passing"

**Claude → Daniel:**
> "Phase 4A tier integration complete. 32/32 tests passing. See TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md for full report."

**Daniel:**
> "Excellent! Merging to main. Phase 4B E&M systems next."

---

**End of Multi-Agent Patterns Section**
```

---

## YOUR IMPLEMENTATION TASKS (Codex)

### Task 1: Create BRIEFING.md
- Use the complete structure provided above
- Centralize all architectural knowledge
- Make it the first document new agents read

### Task 2: Simplify CLAUDE.md
- Reduce from 1000+ lines to ~200 lines
- Focus on "who Claude is" and "how to work with Claude"
- Point to BRIEFING.md for architecture details

### Task 3: Create CODEX.md
- Parallel structure to CLAUDE.md
- Document your role, backlog, implementation patterns
- Keep it around 200-300 lines

### Task 4: Update AGENTS.md
- Add "Multi-Agent Collaboration Patterns" section
- Document Phase 4A case study (Architect + Implementer pattern)
- Add communication guidelines

---

## SUCCESS CRITERIA

### Documentation Update Complete When:

- [x] **BRIEFING.md created** (central source of truth, ~800-1000 lines)
- [x] **CLAUDE.md simplified** (~200 lines, pointer to BRIEFING.md)
- [x] **CODEX.md created** (~200-300 lines, role + backlog)
- [x] **AGENTS.md updated** (multi-agent patterns section added)
- [x] **All files committed** with clear commit messages
- [x] **Cross-references correct** (links between docs work)

### Expected Outcome:

**New agents (or Daniel) can:**
1. Read BRIEFING.md → understand entire project in 30 minutes
2. Read CLAUDE.md or CODEX.md → understand specific agent roles
3. Read AGENTS.md → understand collaboration patterns
4. Start contributing immediately (clear entry points)

**Documentation hierarchy:**
```
BRIEFING.md (central, comprehensive)
    ↓ references
CLAUDE.md (role-specific, lightweight)
CODEX.md (role-specific, lightweight)
AGENTS.md (collaboration patterns)
    ↓ references
docs/ROADMAP.md (phase milestones)
docs/vocabulary/ (technical specs)
TEMP/ (implementation reports)
```

---

## COMMIT STRUCTURE

**Commit 1: Create BRIEFING.md**
```bash
git add BRIEFING.md
git commit -m "docs: create central BRIEFING.md (post Phase 4A)

- Comprehensive project overview (central source of truth)
- 3-tier math core architecture documented
- Phase 4A achievements highlighted (32/32 tests)
- Multi-agent collaboration patterns (Codex + Claude)
- Sovereignty principles (hot path vs ingestion)
- User's trust model (Daniel as non-coder project lead)

This becomes the first document new agents read.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 2: Simplify CLAUDE.md**
```bash
git add CLAUDE.md
git commit -m "docs: streamline CLAUDE.md (point to BRIEFING.md)

- Reduced from 1000+ lines to ~200 lines
- Focus on Claude's role and workflow
- Architecture details moved to BRIEFING.md
- Added Phase 4A case study (multi-agent collaboration)
- Clear coordination patterns with Codex

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 3: Create CODEX.md**
```bash
git add CODEX.md
git commit -m "docs: create CODEX.md (implementation lead guide)

- Documents Codex's role and backlog
- Implementation patterns (tier metadata, ternary ops, TDD)
- Current backlog (Phase 4B, benchmarks, glTF export)
- Collaboration with Claude (architect + implementer pattern)
- Example workflows for common tasks

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit 4: Update AGENTS.md**
```bash
git add AGENTS.md
git commit -m "docs: add multi-agent collaboration patterns to AGENTS.md

- Architect + Implementer pattern (Claude + Codex)
- Phase 4A case study (tier integration)
- Communication guidelines (explicit roles, references, blockers)
- Example session walkthrough (complete Phase 4A flow)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## FINAL NOTES

**Codex, you're in charge of this documentation update.** Daniel and Claude trust you to:

1. **Create BRIEFING.md** as the central reference (use template above)
2. **Simplify CLAUDE.md** to lightweight role definition
3. **Create CODEX.md** documenting your role and backlog
4. **Update AGENTS.md** with multi-agent patterns

**Timeline:** 2-3 hours for all four files

**Communication:**
- Commit each file separately (4 commits total)
- Use clear commit messages (provided above)
- Report when complete: "Documentation update complete. See commits [hash1]..[hash4]."

**Daniel's guidance:**
> "I trust you to structure this correctly. Make BRIEFING.md the central hub. Keep agent-specific docs lightweight. Document what we just achieved (Phase 4A) as a collaboration success story."

**Let's document this milestone properly. The world needs to see how multi-agent collaboration actually works.** 🚀

---

**Prepared by:** Claude (Anthropic Sonnet 4.5)
**Implementation Lead:** Codex (OpenAI)
**Date:** November 24, 2025
**Status:** Ready for Codex to execute
