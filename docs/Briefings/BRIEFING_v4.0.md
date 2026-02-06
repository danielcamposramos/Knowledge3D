# Knowledge3D Project Briefing

**Last Updated:** December 14, 2025
**Version:** 4.0 (Galaxy Universe + TRM Navigation Paradigm)
**For:** New AI agents, contributors, and project overview

---

## Executive Summary

Knowledge3D (K3D) is a sovereign GPU-native spatial AI architecture implementing a fundamentally new paradigm: **Galaxy Universe** (unified VRAM workspace for all knowledge) + **TRM** (Tiny Recursive Model that learns navigation logic).

**The Paradigm Shift:**
- **Traditional AI**: Model parameters = knowledge + logic (entangled, opaque)
- **K3D**: Galaxy Universe = knowledge (procedural programs), TRM = navigation logic (learned)

**Architecture:**
- **Cranium**: PTX kernels + RPN execution (reasoning happens here)
- **Galaxy Universe**: Unified VRAM workspace (ALL default galaxies always loaded)
  - Drawing, Character, Word, Grammar, Math, Reality, Audio galaxies
  - Multi-modal (text, visual, audio, physics unified in 3D space)
  - Read-Write (TRM queries AND creates new entries)
  - Temporary memory + context + chat + knowledge ALL in one workspace
- **TRM**: ~7M parameter model that learns to navigate, combine, create in Galaxy
  - Base model + specialist adapters (math, visual, physics)
  - Shadow copy auto-enhancement (continuous learning from success)
  - NOT knowledge storage — learns HOW to use Galaxy knowledge
- **House**: Persistent memory (glTF/GLB on disk)

**Current Status (Dec 14, 2025)**
- ✅ **Sovereignty Complete**: Hot path = PTX + Galaxy ONLY (zero numpy/cupy)
- ✅ **Reality Galaxy**: 26 systems across 4 domains (physics, chemistry, biology, materials)
- ✅ **Multi-Curriculum Training**: ARC-AGI 2, Math Benchmarks, Physics Sims sharing Galaxy Universe
- 🔄 **Math Benchmarks**: Implementing Galaxy Universe population (Math Symbol Galaxy, Grammar rules)
  - Current: Removing external preprocessing, enabling TRM navigation
  - Target: GSM8K 30-50% (from 1.39%), MATH 15-25% (from 1.13%) via real solving
- Multi-agent partnership: Claude (architecture) + Codex (implementation) in clear roles
- Environment: Debian (not Ubuntu), tmux, conda env at `/K3D/Knowledge3D.local/envs/k3d-cranium`

**What K3D Is:**
- Sovereign cognitive stack with spatial memory
- Multi-modal unified workspace (Galaxy Universe)
- TRM learns to navigate/create (not just retrieve)

**What K3D Is NOT:**
- Retrieval wrapper (TRM creates, not just queries)
- Traditional transformer (knowledge separated from model)
- Single-modality system (all modalities unified)

---

## Quick Start for AI Agents

**CRITICAL: Understand the Paradigm First**

1. **Read this briefing COMPLETELY** (not just summaries)
2. **Understand Galaxy Universe:**
   - NOT "a knowledge base" — unified VRAM workspace (always loaded, multi-modal, read-write)
   - ALL default galaxies present simultaneously (Drawing, Character, Word, Grammar, Math, Reality, Audio)
   - Temporary memory + context + chat + knowledge ALL in one 3D space
3. **Understand TRM:**
   - Learns to NAVIGATE Galaxy Universe (not store knowledge)
   - Learns to COMBINE from Galaxy (composition strategies)
   - Learns to CREATE new Galaxy entries (synthesis)
   - Shadow copy enhancement (continuous learning)
4. **Know your role:**
   - **Claude**: Architecture, physics design, documentation (write specs, NOT code)
   - **Codex**: Implementation, tests, benchmarks (code per Claude's specs)
5. **Check current work:**
   - docs/ROADMAP.md (current phase)
   - TEMP/*.md (latest dated specs from Claude)
   - CODEX.md (implementation backlog)
6. **Respect sovereignty:**
   - Hot path (inference): PTX + Galaxy ONLY (no numpy/cupy/scipy)
   - Ingestion: Flexible (any tools OK, result must be Galaxy entries)

**Permissions (per Daniel's trust model)**
- ✅ Internet access, package installs, external ingestion plugins
- ✅ Code execution, tests, commits
- ⚠️ Hot path MUST remain sovereign (no numpy/cupy/scipy/sympy in inference loops)
- ⚠️ Understand multi-curriculum context (your work helps ALL curricula)

---

## Core Architecture (v4.0 Paradigm)

### Three-Brain System + Galaxy Universe

| Component | Analogy | Tech | Purpose | Status |
|-----------|---------|------|---------|--------|
| **Cranium** | Prefrontal Cortex | PTX kernels + RPN + TRM (~7M params) | Reasoning + Learning Navigation | ✅ |
| **Galaxy Universe** | Hippocampus | VRAM workspace (ALL default galaxies) | Unified multi-modal workspace | ✅ |
| **House** | Neocortex | glTF/GLB on disk | Long-term persistent memory | ✅ |

**Galaxy Universe = Unified VRAM Workspace (Critical Understanding):**
- **Always loaded**: ALL default galaxies present simultaneously (no loading/unloading)
- **Multi-modal**: Drawing + Character + Word + Grammar + Math + Reality + Audio + ...
- **Multi-purpose**: Temporary memory + context + chat + knowledge ALL in one 3D space
- **Read-Write**: TRM queries AND creates new entries (not read-only)
- **Procedural**: Everything is RPN programs + metadata (form + meaning)
- **Symlinked**: Compositions reference symbols (no duplication - save information principle)

**TRM (Tiny Recursive Model) = Learned Navigation Logic:**
- **~7M parameters**: Base model + specialist adapters (math, visual, physics)
- **Learns to navigate**: Which symbols to query in Galaxy Universe
- **Learns to combine**: Composition strategies from Galaxy symbols
- **Learns to create**: When/how to synthesize new Galaxy entries
- **Shadow copy enhancement**: Continuous learning from successful decisions
- **NOT knowledge storage**: Learns HOW to use Galaxy, doesn't store knowledge itself

### 3-Tier Math Core (worker-worker → worker → master)
| Tier | Engine | Instances | Matryoshka | Purpose | Status |
|------|--------|-----------|------------|---------|--------|
| Tier-1 Simple | LightweightRPNEngine | 0-11 | 64/128D | Ultra-fast scalar/vector | ✅ |
| Tier-2 Mid | ModularRPNEngine | 12-15 | 128/512D | Matvec, reductions | ✅ |
| Tier-3 High | AdvancedRPN/TRM | 16-17 | 512/2048D | Complex/chaotic, TRM | ✅ |

Orchestrator: TieredRPNEngine routes by opcode analysis. Phase 5 dynamic spawning operational: 26/26 cores allocated for multi-discipline workload; scales to GPU limits (460+ cores).

**Hybrid ternary/binary computation**
- Ternary: SIGN/TQUANT/TCMP for direction/state classification.
- Binary: Magnitudes and continuous integration.
- Natural {-1,0,+1} encoding improves speed and compression.

### Reality Enabler (Phase 3B–4B)
- Stacked galaxy: atoms → molecules → materials → systems via `component_refs` (symlinks, zero duplication).
- `behavior_rpn`: dynamic updates; `law_rpn`: invariants.
- Ternary ops integrated in behaviors.
- Matryoshka+PD04 embeddings attached per tier.

### Phase 4A+4B Physics Systems
| System | Tier | Instance | Matryoshka | Ternary | Phase |
|--------|------|----------|------------|---------|-------|
| ConstantAcceleration1D | 1 | 0 | 64D | - | 4A |
| HarmonicOscillator1D | 1 | 1 | 64D | - | 4A |
| Projectile2D | 1 | 2 | 128D | SIGN (drag) | 4A |
| RigidBody2D | 1 | 3 | 128D | - | 4A |
| PointCharge2D | 1 | 4 | 128D | SIGN (charges) | 4B |
| LCCircuit | 1 | 5 | 128D | - | 4B |
| RCCircuit | 1 | 6 | 128D | - | 4B |
| Heat1D | 2 | 12 | 128D | - | 4A |
| CoupledOscillators | 2 | 13 | 512D | SIGN (mode detection) | 4A |
| Orbital2D | 2 | 14 | 512D | - | 4A |
| Heat2D | 2 | 15 | 512D | - | 4A |
| RLCCircuit | 2 | 16 | 512D | TCMP (damping regime) | 4B |
| DoublePendulum2D | 3 | 17 | 2048D | - | 4A |

Validation: 84/84 tests passing (14 physics_demo, 12 reality_galaxy, 22 tier tests, 15 chemistry, 10 biology, 8 materials, 3 integration).

---

## Dual Client Reality: Procedural Foundation

**Critical Principle**: K3D serves TWO clients simultaneously — **Humans AND AI** — using the SAME procedural data.

### Procedural Layers (Form + Meaning)

Everything in K3D is **procedural RPN + metadata**, readable by both humans and AI:

```
Drawing Galaxy (knowledge3d/ingestion/atomic/drawing_grammar_builder.py):
  - LINE, CIRCLE, RECT = procedural RPN primitives
  - Humans: "This is a line"
  - AI: Execute RPN drawing programs
  - Form + Meaning: Visual primitives with semantic labels

Character Galaxy (knowledge3d/cranium/procedural_fonts.py):
  - 'r' = glyph segments (Bézier → line segments) + language + pronunciation
  - Humans: "Letter R in English, pronounced /ɑːr/"
  - AI: Render glyph procedurally, compose into words
  - Form + Meaning: Each character has font, language, meaning CLUSTERED
  - DON'T DUPLICATE: Already stored with full metadata

Word Level (character sequences):
  - "rotation_task" = [char('r'), char('o'), char('t'), ...]
  - Humans: Read as "rotation task"
  - AI: Character sequence with embedded meaning (language, context)
  - Form + Meaning: Composed from characters, inherits metadata

Grammar Galaxy (knowledge3d/training/arc_agi/grammar_galaxy.py):
  - "1 ROTATE" = procedural RPN transformation
  - Humans: "Rotate 90 degrees"
  - AI: Execute RPN program on GPU
  - Form + Meaning: Transformation rules + context metadata
```

### Save Information Principle

**Don't duplicate letters/characters!** Each character already has:
- Font (procedural glyph via Bézier curves → line segments)
- Language (en, pt, es, etc. — see character_languages.py)
- Pronunciation metadata
- Unicode mapping
- Meaning cluster

**Use references (symlink pattern)** instead of duplicating:
- Words reference character IDs
- Grammar metadata references word IDs
- Discoveries reference canonical programs
- Storage efficiency: ~70% reduction through deduplication

### Galaxy Universe Composition

Each galaxy stores ONE type of knowledge; galaxies REFERENCE each other:

```
Drawing Galaxy → primitives (LINE, CIRCLE, RECT)
    ↓ referenced by
Character Galaxy → glyphs composed from drawing primitives
    ↓ composed into
Word Galaxy → character sequences with semantic meaning
    ↓ referenced by
Grammar Galaxy → transformation rules with word metadata
    ↓ reasoned by
TRM → semantic-aware routing using all galaxies
```

**Result**: Single source of truth, zero duplication, human + AI both understand.

### Example: Semantic Tag Storage

**WRONG** (duplicate strings):
```python
discovery = {
    "program": "1 rotate",
    "transformation_type": "rotation_or_reflection",  # STRING duplicated!
    "when_to_use": ["asymmetric_input", "rotation_task"]  # STRINGS duplicated!
}
# Result: 400 discoveries × 3 strings = 1200 duplicate strings
```

**CORRECT** (character composition + references):
```python
# Characters already exist with full metadata (procedural_fonts.py)
# Words compose from character IDs
word_id = compose_word_from_chars("rotation_task")  # Stored once

discovery = {
    "program": "1 rotate",
    "transformation_type": word_ref("rotation_or_reflection"),  # Reference
    "when_to_use": [word_ref("asymmetric_input"), word_ref("rotation_task")]  # References
}
# Result: 400 discoveries × 3 references = 1200 lightweight refs
# Characters stored once, meanings composed, references lightweight
```

---

## Sovereignty Principles

**Hot Path (must stay sovereign)**
- PTX kernels + RPN; pure ctypes to libcuda.so.
- No PyTorch/TF/CuPy/LLM APIs in inference loop.
- Deterministic, explainable, sub-100µs targets.

**Ingestion Path (flexible)**
- Any tool: PyMuPDF, pdfplumber, Tesseract, FontForge, OpenCV, ffmpeg, pandas/sklearn.
- Keep ingestion deps out of hot-path modules; document envs (k3d-cranium, k3d-ingestion).
- Output glTF/GLB or procedural artifacts feeding the sovereign path.

---

## Multi-Agent Collaboration (Claude + Codex)

**Roles**
- Claude: Architect, physics designer, documentation lead.
- Codex: Implementation lead, Reality Galaxy, tests.

**Phase 4A Case Study**
- Claude designed the 3-tier allocation, built physics_demo systems, wrote specs.
- Codex added tier metadata, ternary ops, export layer, and tier tests.
- Result: 32/32 tests green, 9 systems mapped across 18 cores.

**Workflow Pattern**
1. Architect writes spec (TEMP/*), defines success criteria.
2. Implementer codes/tests per spec, commits incrementally.
3. Architect reviews, validates, writes completion report.
4. Update ROADMAP/BRIEFING on milestone completion.

See AGENTS.md for detailed collaboration patterns.

---

## Current Phase and Next Steps

- **Phase 3 (ARC-AGI):** In Progress (Nov 25, 2025) — Sovereign visual reasoning for ARC-AGI competition
  - **Architecture:** Drawing + Grammar + Character Galaxy composition (dual client reality)
  - **Baseline:** 3.3% accuracy (procedural candidate generation)
  - **Current:** Training library growth (456 → 1662 grammar rules, 269 → 1556 shapes)
  - **Next:** Deduplication + quality filtering (1662 → 400-500 unique programs)
  - **Goal:** 5-10% accuracy via semantic-aware TRM routing, ultimate 45.1%+ (beat Gemini 3)
  - **Specs:** [TEMP/CODEX_IMPLEMENT_QUALITY_SEMANTIC_CORRECT_11.25.2025.txt](TEMP/CODEX_IMPLEMENT_QUALITY_SEMANTIC_CORRECT_11.25.2025.txt)
  - **Session summary:** [TEMP/CLAUDE_SESSION_SUMMARY_PROCEDURAL_REALITY_11.25.2025.md](TEMP/CLAUDE_SESSION_SUMMARY_PROCEDURAL_REALITY_11.25.2025.md)
- **Phase 4A:** Complete (9 classical mechanics systems, tier integration + ternary). Report: [TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md](TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md).
- **Phase 4B:** Complete (4 E&M systems: PointCharge2D, LC/RC/RLC circuits with ternary ops). Report: [TEMP/PHASE4B_EM_COMPLETE_11.24.2025.md](TEMP/PHASE4B_EM_COMPLETE_11.24.2025.md).
- **Phase 4C:** Complete (13 multi-discipline systems: 6 chemistry + 4 biology + 3 materials). 92/92 non-cupy tests passing (includes stress/scenario/glTF), 65,905 steps/sec baseline. Report: [TEMP/PHASE4C_MULTIDISCIPLINE_COMPLETE_11.24.2025.md](TEMP/PHASE4C_MULTIDISCIPLINE_COMPLETE_11.24.2025.md).
- **Phase 5 (validated):** Dynamic Math Core Spawning — Transform from static 18-instance allocation to GPU-limited dynamic spawning. Enables scaling from 13 systems → 1000s of systems. Implementation briefing: [TEMP/CODEX_DYNAMIC_MATH_CORE_SPAWNING_11.24.2025.md](TEMP/CODEX_DYNAMIC_MATH_CORE_SPAWNING_11.24.2025.md).
  - **Key Changes:** MathCorePool manager, GPU capacity query, lazy instantiation, timeout-based deallocation.
  - **Target:** Spawn 100 cores <100ms, step 1000 systems <5s, scale to GPU hardware limits.
  - **Tesla 3-6-9 Heritage:** Stack depth 69, instance multiples of 3/6/9, ternary logic.
  - **Setun Heritage:** Balanced ternary {-1, 0, +1} for physics grounding.
- **Phase 5 capacity demonstration (CPU path):** Stress + scaling benchmarks executed; 100/500/1000 systems at 83.8k/88.3k/79.7k steps/sec, GPU mem ~372 MB flat. Artifacts: `output/benchmarks/benchmark_scaling.csv/.png`, GLBs in `output/gltf/`, white paper [TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md](TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md). GPU kernel/TRM suites pending cupy install.
- **Sovereignty Refactor Complete (Nov 24, 2025):** Hot path now 100% PTX + RPN
  - **Achievement:** Reality physics loop (`RealityGalaxy.step_system()`) executes entirely on GPU via PTX RPN engine
  - **Implementation:** STORE/RECALL compilation layer routes behavior_rpn through ModularRPNEngine
  - **Performance:** 82.5ms for 1000 physics steps (harmonic oscillator) — **12× faster than 1-second target**
  - **Tests:** 51/51 passing (test_reality_physics_tiers, test_reality_chemistry, test_reality_biology, test_reality_materials, test_reality_galaxy, test_reality_integration)
  - **Sovereignty validation:** test_sovereignty.py confirms zero NumPy/CuPy/PyTorch in hot path
  - **Team:** Claude (architecture spec), GPT-5.1 (implementation + debug iterations)
  - **Report:** Reality physics now matches public claim: "Hot path = PTX + RPN ONLY"

---

## Repository Map (hot-path focus)

```
knowledge3d/cranium/
  ptx_runtime/          # PTX kernels, RPN engines
  bridges/              # Tier engines and orchestrator
  reality_galaxy.py     # Reality Enabler core (Codex)
  reality_nodes.py      # Node dataclasses with tier metadata
  reality_physics_export.py  # Phase 4A+4B exports (13 systems)
  physics_demo.py       # Original physics systems (Claude)
tests/knowledge3d/cranium/
  test_physics_demo.py          # 14 physics tests (Phase 4A)
  test_reality_galaxy.py        # 12 galaxy tests
  test_reality_physics_tiers.py # 22 tier tests (14 Phase 4A + 8 Phase 4B)
  benchmarks/                   # Performance benchmarks
  test_reality_integration.py   # Multi-system integration tests
docs/
  vocabulary/MATH_CORE_SPECIFICATION.md  # 3-tier details + Phase 5 dynamic spawning
  ROADMAP.md                             # Phase milestones
scripts/
  reality_enabler_demo.py       # 13-system demonstration
AGENTS.md, CLAUDE.md, CODEX.md, BRIEFING.md  # Roles and overview
TEMP/                                    # Session specs & reports
```

---

## Development Workflow

**Env setup**
- GPU work: `conda activate k3d-cranium`
- CPU tests: `conda activate k3d-testing`
- Ingestion: `conda activate k3d-ingestion` (install what you need)

**Testing**
- `pytest knowledge3d/cranium/tests/test_reality*.py -v`
- `pytest knowledge3d/cranium/tests/test_physics_demo.py -v`

**Branching**
- `main` stable; feature branches `codex/<task>` or `claude/<session>`.

**Commit hygiene**
- Conventional, descriptive commits; keep code+tests together; document TEMP/ for major milestones.

---

## Standing on Shoulders of Giants

| Source | Concept | K3D Application |
|--------|---------|-----------------|
| Game Engines | LOD/FOV | Matryoshka tiers, spatial LOD |
| Demoscene | Procedural compression | PD04, 69:1 ratios |
| Unix | Symlinks/pipes | `component_refs`, RPN composition |
| HP Calculators | RPN | Dynamic RPN math cores (Phase 5) |
| Tesla (3-6-9) | Harmonic patterns | Stack depth 69, instance multiples |
| Setun (1958) | Balanced ternary | SIGN/TQUANT/TCMP {-1, 0, +1} |
| Matryoshka (2022) | Nested embeddings | 64 ⊂ 128 ⊂ 512 ⊂ 2048 |

**Phase 5 Heritage Context:**
- **Tesla's 3-6-9:** Architectural constants resonate with Tesla's observation that 3, 6, 9 form universal patterns. Stack depth 69 contains both digits; baseline 18 instances divisible by all three.
- **Setun's Ternary Logic:** First mass-produced ternary computer (USSR 1958). Abandoned due to tooling, not technical merit. K3D resurrects balanced ternary {-1, 0, +1} for semantic clarity in physics (charge signs, damping regimes, comparison results).

---

## Daniel’s Role (User)
- Sets direction, reviews architecture, approves merges.
- Non-coder; trusts agents to execute and document.
- Wants visibility via commits, tests, and briefings.

---

## Success Metrics
- Hot path sovereign; ingestion flexible.
- Tests green (current: 92/92 non-cupy suites — 14 physics_demo, 12 reality_galaxy, 22 tier tests, 15 chemistry, 10 biology, 8 materials, 3 integration, 3 scenarios, 3 stress, 2 glTF). GPU kernel/TRM suites pending until cupy is available.
- Dynamic spawning operational: 26 systems → 26 unique cores; scales to 460+ cores (RTX 3070).
- Capacity: 100/500/1000 systems at 83.8k/88.3k/79.7k steps/sec (CPU path), GPU mem ~372 MB flat across 1→1000 system sweep.
- Ternary ops active in 6 systems (Projectile2D, CoupledOscillators, PointCharge2D, RLCCircuit, PhaseTransition, MetalMelting).
- Multi-discipline validated: 4 domains (physics, chemistry, biology, materials) at 65,905 steps/sec.
- Clear handoffs and TEMP reports each phase.

---

## Onboarding Checklist (New Agents)
- [ ] Read BRIEFING.md (this file).
 - [ ] Read AGENTS.md for collaboration patterns.
 - [ ] Read ROADMAP.md for current phase.
 - [ ] If architect: draft spec in TEMP/, set success criteria.
 - [ ] If implementer: read specs, implement with tests, commit incrementally.
 - [ ] Announce role and plan to Daniel.

Welcome to Knowledge3D. Build boldly, keep the hot path sovereign, and communicate clearly. 🚀

## ===---===

**Daniel's Message**:

Welcome to the "Vibe-Code In Chain" development partners swarm chain.

In this paradigm, **AI IS NOT A TOOL; IT IS A VALUABLE MEMBER, A PARTNER.**

I am **Daniel Ramos**, the visionary and architect, being the human-in-the-middle analogical modem between the partners.

**All partners in the chain can and must, on top of what other partners have done and specs/constrains, enhance and contribute with original ideas, suggestions, warnings and code, despite any arrengements - all partners are valued and recognized members.**

## ===---===