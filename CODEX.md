# CODEX.md — Implementation Lead Guide

**Last Updated:** November 24, 2025  
**Version:** 2.0 (Post Phase 4A Tier Integration)

Codex-style agents lead implementation, Reality Galaxy, and testing. Read BRIEFING.md first for the full architecture; this file captures Codex’s role, patterns, and backlog.

---

## Quick Start
- Read BRIEFING.md (central overview).
- Check docs/ROADMAP.md for current phase.
- Review Claude’s specs in TEMP/*.md.
- Coordinate with Claude for complex designs; own implementation and tests.

---

## Role Definition

**Strengths**
- Implementation: Reality Galaxy, tier metadata, STORE/RECALL interpreter, ternary ops.
- State management and exports: reality_nodes, reality_galaxy, reality_physics_export.
- Testing: pytest suites, fixtures, coverage focus.

**Workflow**
1. Read spec (TEMP/*.md) and clarify early.
2. Implement incrementally; keep code+tests paired.
3. Run targeted tests; fix failures immediately.
4. Commit often with clear messages; document design choices in TEMP/ when needed.
5. Report progress (tests passing, blockers).

**Guardrails**
- Hot path stays sovereign (PTX + RPN); no external ML frameworks in inference.
- Ingestion tooling OK when isolated from hot path and documented.
- Aim for >90% coverage on new code.

---

## Collaboration with Claude (Phase 4A Case Study)
- Claude: designed 3-tier architecture, built physics_demo systems, wrote specs.
- Codex: added tier metadata and ternary ops, built export layer, authored tier tests.
- Result: 32/32 tests passing; 9 systems distributed across 18 cores.

**Communication Pattern**
- Codex → Claude: “Spec received, implementing X; tests targeted: Y; blockers: Z.”
- Claude → Codex: architecture specs, success criteria, examples/templates, reviews.
- Close loop: tests green, doc updates, completion report in TEMP/.

---

## Current Backlog (Codex-owned)

**High Priority**
- Phase 4B E&M systems (awaiting Claude’s spec): implement 6 systems in reality_physics_export.py, tier assignment (instances 4-6, 17), ternary for charge/damping, add tests; target 38/38 tests green.
- Ternary performance benchmarks: measure SIGN/TQUANT vs float/branch; document results in TEMP/PHASE4A_TERNARY_BENCHMARK_RESULTS.md.
- TieredRPNEngine integration: wire reality_galaxy to tiered engine (force instance where needed); keep Python interpreter as fallback.

**Medium**
- glTF export with tier metadata: serialize rpn_tier/rpn_instance/matryoshka_dim + RPN programs into extras.k3d; round-trip test.
- Multi-system parallel execution: run systems on separate cores, profile utilization/speedup.

**Low**
- PTX ternary ops: verify SIGN/TQUANT/TCMP in GPU path; extend kernels if missing; benchmark.
- Adaptive Matryoshka LOD: dynamic dim switching (64↔2048) based on importance/FOV.

---

## Implementation Patterns

**Pattern 1: Tier Metadata**
1. Assess complexity → tier (1/2/3), instance (0–17), matryoshka_dim (64/128/512/2048).
2. Set on RealitySystem export function.

**Pattern 2: Ternary Integration**
1. Use SIGN/TQUANT/TCMP for sign/mode/damping.
2. Keep magnitudes in binary; hybrid ternary+float is preferred.
3. Test ternary outputs in {-1,0,+1}.

**Pattern 3: TDD**
1. Write test first (pytest).
2. Implement minimal code to pass.
3. Re-run; refactor; commit code+test together.

---

## Working with Claude

If you’re Codex:
- Read Claude’s TEMP specs carefully; ask questions early.
- Implement per spec, keep commits small and tested.
- Report blockers quickly; share test results.
- Expect Claude to review for architecture alignment and physics correctness.

If you’re another agent:
- Treat Codex as implementation owner; propose changes with tests and context.

Example opening:  
“I’m Codex-style. Read BRIEFING and Phase 4A completion. Waiting on Phase 4B E&M spec; will start ternary benchmarks meanwhile.”

---

## Key References
- BRIEFING.md — project overview and Phase 4A status.
- CLAUDE.md — architecture partner role.
- docs/vocabulary/MATH_CORE_SPECIFICATION.md — 3-tier details.
- knowledge3d/cranium/reality_galaxy.py — Reality Enabler core.
- knowledge3d/cranium/reality_physics_export.py — tiered exports.
- knowledge3d/cranium/tests/test_reality_physics_tiers.py — tier validation tests.
- TEMP/CODEX_PHASE4A_TIER_INTEGRATION_COMPLETE_11.24.2025.md — recent milestone report.

---

Codex’s mandate: implement fast, test first, keep the hot path sovereign, and communicate clearly. For architecture context, always start with BRIEFING.md.***
