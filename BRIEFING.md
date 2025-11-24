# Knowledge3D Project Briefing

**Last Updated:** November 24, 2025
**Version:** 3.2 (Phase 5 Capacity Demonstration — CPU path complete; GPU kernels pending cupy)
**For:** New AI agents, contributors, and project overview

---

## Executive Summary

Knowledge3D (K3D) is a sovereign GPU-native spatial AI architecture. Reasoning happens through PTX + RPN; memories live as 3D worlds (glTF/GLB) with symlinked composition. We stand on proven patterns: game LOD, demoscene compression, Unix symlinks, HP RPN, Matryoshka embeddings.

**Current Status**
- Phase 4C complete: 26 systems across 4 domains (13 physics + 6 chemistry + 4 biology + 3 materials); stress + scenario + glTF suites added (92/92 non-cupy tests green) with capacity runs at 83.8k/88.3k/79.7k steps/sec for 100/500/1000 systems (CPU path).
- Phase 5 validated: Dynamic Math Core spawning operational — 26 systems → 26 unique cores automatically. Scales to GPU hardware limits (460+ cores on RTX 3070, 1280+ on RTX 4090, 2640+ on H100).
- Capacity benchmark artifacts: `output/benchmarks/benchmark_scaling.csv/.png` (throughput ~71k–118k steps/sec; GPU mem ~372 MB flat) and 26 GLBs in `output/gltf/`.
- Multi-agent partnership: Claude (architecture, specs) + Codex (implementation, tests) delivering in lockstep.
- Sovereignty enforced: hot path is PTX+RPN only; ingestion can use any external tool.

**Not:** A retrieval wrapper. **Is:** A sovereign cognitive stack with spatial memory and embodied reasoning.

---

## Quick Start for AI Agents

1. Read this briefing end-to-end.
2. Know your role:
   - Claude-style: architecture, physics design, documentation.
   - Codex-style: implementation lead, Reality Galaxy, tests.
3. Check phase: docs/ROADMAP.md.
4. Check backlog: CODEX.md (implementation), CLAUDE.md (architecture scope).
5. Respect sovereignty: PTX/RPN for hot path; anything goes for ingestion.

**Permissions (per Daniel’s trust model)**
- ✅ Internet access, package installs, external ingestion plugins.
- ✅ Code execution, tests, commits.
- ⚠️ Hot path must remain sovereign (no ML frameworks, no opaque runtimes).

---

## Core Architecture (Phase 4A)

### Three-Brain System
| Component | Analogy | Tech | Purpose | Status |
|-----------|---------|------|---------|--------|
| Cranium | Prefrontal Cortex | PTX kernels, RPN, TRM | Active reasoning | ✅ |
| Galaxy | Hippocampus | VRAM embeddings | Short-term memory | ✅ |
| House | Neocortex | glTF/GLB on disk | Long-term memory | ✅ |

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

- **Phase 4A:** Complete (9 classical mechanics systems, tier integration + ternary). Report: [TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md](TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md).
- **Phase 4B:** Complete (4 E&M systems: PointCharge2D, LC/RC/RLC circuits with ternary ops). Report: [TEMP/PHASE4B_EM_COMPLETE_11.24.2025.md](TEMP/PHASE4B_EM_COMPLETE_11.24.2025.md).
- **Phase 4C:** Complete (13 multi-discipline systems: 6 chemistry + 4 biology + 3 materials). 92/92 non-cupy tests passing (includes stress/scenario/glTF), 65,905 steps/sec baseline. Report: [TEMP/PHASE4C_MULTIDISCIPLINE_COMPLETE_11.24.2025.md](TEMP/PHASE4C_MULTIDISCIPLINE_COMPLETE_11.24.2025.md).
- **Phase 5 (validated):** Dynamic Math Core Spawning — Transform from static 18-instance allocation to GPU-limited dynamic spawning. Enables scaling from 13 systems → 1000s of systems. Implementation briefing: [TEMP/CODEX_DYNAMIC_MATH_CORE_SPAWNING_11.24.2025.md](TEMP/CODEX_DYNAMIC_MATH_CORE_SPAWNING_11.24.2025.md).
  - **Key Changes:** MathCorePool manager, GPU capacity query, lazy instantiation, timeout-based deallocation.
  - **Target:** Spawn 100 cores <100ms, step 1000 systems <5s, scale to GPU hardware limits.
  - **Tesla 3-6-9 Heritage:** Stack depth 69, instance multiples of 3/6/9, ternary logic.
  - **Setun Heritage:** Balanced ternary {-1, 0, +1} for physics grounding.
- **Phase 5 capacity demonstration (CPU path):** Stress + scaling benchmarks executed; 100/500/1000 systems at 83.8k/88.3k/79.7k steps/sec, GPU mem ~372 MB flat. Artifacts: `output/benchmarks/benchmark_scaling.csv/.png`, GLBs in `output/gltf/`, white paper [TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md](TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md). GPU kernel/TRM suites pending cupy install.

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
