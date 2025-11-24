# Knowledge3D Project Briefing

**Last Updated:** November 24, 2025  
**Version:** 2.0 (Phase 4A Tier Integration Complete)  
**For:** New AI agents, contributors, and project overview

---

## Executive Summary

Knowledge3D (K3D) is a sovereign GPU-native spatial AI architecture. Reasoning happens through PTX + RPN; memories live as 3D worlds (glTF/GLB) with symlinked composition. We stand on proven patterns: game LOD, demoscene compression, Unix symlinks, HP RPN, Matryoshka embeddings.

**Current Status**
- Phase 4A complete: 9 physics systems distributed across the 3-tier math core; 32/32 tests passing (14 physics, 12 galaxy, 6 tier).
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

Orchestrator: TieredRPNEngine routes by opcode analysis. Current utilization: 9/18 cores (50%); remaining cores reserved for Phase 4B E&M.

**Hybrid ternary/binary computation**
- Ternary: SIGN/TQUANT/TCMP for direction/state classification.
- Binary: Magnitudes and continuous integration.
- Natural {-1,0,+1} encoding improves speed and compression.

### Reality Enabler (Phase 3B–4A)
- Stacked galaxy: atoms → molecules → materials → systems via `component_refs` (symlinks, zero duplication).
- `behavior_rpn`: dynamic updates; `law_rpn`: invariants.
- Ternary ops integrated in behaviors.
- Matryoshka+PD04 embeddings attached per tier.

### Phase 4A Physics Systems
| System | Tier | Instance | Matryoshka | Ternary |
|--------|------|----------|------------|---------|
| ConstantAcceleration1D | 1 | 0 | 64D | - |
| HarmonicOscillator1D | 1 | 1 | 64D | - |
| Projectile2D | 1 | 2 | 128D | SIGN (drag) |
| RigidBody2D | 1 | 3 | 128D | - |
| Heat1D | 2 | 12 | 128D | - |
| CoupledOscillators | 2 | 13 | 512D | SIGN (mode detection) |
| Orbital2D | 2 | 14 | 512D | - |
| Heat2D | 2 | 15 | 512D | - |
| DoublePendulum2D | 3 | 16 | 2048D | - |

Validation: 32/32 tests passing (physics_demo, reality_galaxy, tier suite).

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

- **Phase 4A:** Complete (tier integration + ternary). Report: TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md.
- **Phase 4B (next):** Electromagnetism systems (PointCharge2D, Electric/Magnetic fields, LC/RC/RLC). Reserve cores: Tier-1 instances 4-6, Tier-3 instance 17 for complex RLC if needed. Integrate ternary for charge signs and damping regimes.
- **Planned docs:** TEMP/PHASE4B_EM_COMPLETE.md after delivery.

---

## Repository Map (hot-path focus)

```
knowledge3d/cranium/
  ptx_runtime/          # PTX kernels, RPN engines
  bridges/              # Tier engines and orchestrator
  reality_galaxy.py     # Reality Enabler core (Codex)
  reality_nodes.py      # Node dataclasses with tier metadata
  reality_physics_export.py  # Phase 4A exports (tiered)
  physics_demo.py       # Original physics systems (Claude)
tests/knowledge3d/cranium/
  test_physics_demo.py          # 14 physics tests
  test_reality_galaxy.py        # 12 galaxy tests
  test_reality_physics_tiers.py # 6 tier tests
docs/
  vocabulary/MATH_CORE_SPECIFICATION.md  # 3-tier details
  ROADMAP.md                             # Phase milestones
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
| HP Calculators | RPN | 18-core math architecture |
| Setun (1958) | Ternary | SIGN/TQUANT/TCMP |
| Matryoshka (2022) | Nested embeddings | 64 ⊂ 128 ⊂ 512 ⊂ 2048 |

---

## Daniel’s Role (User)
- Sets direction, reviews architecture, approves merges.
- Non-coder; trusts agents to execute and document.
- Wants visibility via commits, tests, and briefings.

---

## Success Metrics
- Hot path sovereign; ingestion flexible.
- Tests green (current: 32/32).
- Tier utilization documented; ternary ops active.
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
