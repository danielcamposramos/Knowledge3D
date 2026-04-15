# Codex Batch 3 — Opcodes + Kernels (Third Wave)

**Date:** 2026-04-13
**Parent spec:** `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`
**Predecessor:** `TEMP/CODEX_BATCH2_OPCODES_AND_KERNELS_2026-04-13.md` (green)
**Role:** Codex implements. Claude wrote the parent spec.
**Status:** Spec for implementation after Daniel approval.
**Scope guarantee:** Runtime-first and benchmark-facing only. No broad
ingestion pivot, no loader cleanup wave, no new general runtime.

---

## 0. Where We Stand After Batch 2

Batch 2 closed the second reasoning-kernel wave on `main`:

- S7-S9 landed the dynamic N selector, halting integration, and the
  sovereign sleep-time perf consumer that feeds calibration back into
  the swarm.
- S10 landed `TRESOLVE`, `TORDER`, and `TSUBSUME` as the first
  scalar-first resolution family.
- S11 landed the abductive completion tranche:
  `ABDUCE_HALT`, `SCUNION`, `ICHECK`, `ABDRES`, `ABDNEG`.
- S12 landed the DPLL/SAT substrate via `TBCP` and `TLEARNT`.
- All of the above compile and run behind `K3D_REASONING_OPCODES_V1`.

The next step is not catalog growth. The next step is the next
benchmark-facing reasoning wave that directly extends the Batch 2
substrate already in place.

---

## 1. Hard Rules

- **`K3D_REASONING_OPCODES_V1` gates all Batch 3 surfaces.**
- **PTX + Galaxy + RPN + TRM only** in hot path. No Python
  orchestration, no hot-path fallbacks, no dynamic heaps.
- **Reuse Batch 2 substrates first**: unification arena, tableaux branch
  handles, perf/halting scaffolding, `context_id`, `ethical_trit`, and
  the already-landed sleep calibration path.
- **Bounded, tick-local kernel surfaces only.** If a surface needs an
  unbounded clause heap, frame heap, or graph runtime, it is out of
  scope for Batch 3.
- **Sleep/ingestion changes stay deferred** unless strictly required for
  one canonical source star, such as `reasoning_kbo_precedence`.
- **No score-regression gate** in this batch. Benchmarks remain required
  activity, but numbers are allowed to move as semantics deepen.
- **No Python agenda manager** for Rete. Agenda behavior, if present, is
  bounded GPU substrate only.
- **Do not block spec or implementation on `kimi_swarm`** until the
  outer Codex MCP timeout is raised to 300 seconds. The container is
  already at 300 seconds; the verified blocker is still client-side.

---

## 2. Slice Ordering

1. **S13 — Superposition / Rewriting Core**
2. **S14 — Structured Logic Closure**
3. **S15 — Abductive LP Upgrade**
4. **S16 — Subjective Logic**
5. **S17 — Bi-Abduction / Frame Inference**
6. **S18 — Rete / Semantic Rule Engines**

Order rationale:

- S13 builds directly on Batch 2 resolution/unification and upgrades the
  deductive core first.
- S14 adds bounded closure operators that fit the same tableaux-style
  substrate.
- S15 deepens the abductive path only after the rewrite/resolution layer
  is stronger.
- S16 is largely self-contained once the ternary opinion substrate is
  stable.
- S17 depends on the same bounded-handle discipline and must stay small.
- S18 comes last because it touches agenda behavior and rule-cluster
  execution surfaces, but still stays bounded and GPU-only.

---

## 3. S13 — Superposition / Rewriting Core

**Focus**

- `TSUPERPOS (0xC4)`
- `TREWRITE (0xC5)`
- upgrade `TORDER` from Batch 2's scalar-first KBO placeholder to a
  precedence-star-backed ordering surface

**Implementation shape**

- Thin CUDA kernels only.
- Reuse Batch 2 resolution/unification substrate and substitution arena.
- Introduce exactly one canonical precedence source star:
  `reasoning_kbo_precedence`.
- Keep clause/term representation bounded and static. No runtime heap.
- Use fixed clause handles with a 64-literal ceiling and a 32-node term
  scratch window per active superposition/rewrite attempt. Overflow
  returns `0` deterministically.

**Operational semantics**

- `TORDER` reads symbol precedence ranks from
  `reasoning_kbo_precedence` and upgrades KBO ordering from the current
  scalar placeholder to precedence-aware comparison.
- `TREWRITE` performs one ordered rewrite step on a bounded term handle
  and returns a rewritten handle or `0`.
- `TSUPERPOS` emits one critical-pair handle when compatible equality
  and target terms overlap under the existing unification surface;
  otherwise it returns `0`.

**Tests**

- ordered rewrite fires only when precedence allows
- superposition emits a critical-pair handle on compatible equalities
- non-overlapping rules return `0`

---

## 4. S14 — Structured Logic Closure

**Focus**

- `DL_SATURATE (0xB4)`
- `BLOCKING_CHECK (0xB5)`
- `EULER_COMPLETE (0xB3)`

**Implementation shape**

- DL saturation and blocking use the same bounded branch/closure
  substrate as tableaux.
- Euler closure is a thin transitive/property-chain completion kernel,
  not a new graph runtime.
- All surfaces stay bounded and tick-local.
- Use fixed-capacity branch buffers: 64 assertions per branch, 16 branch
  handles per lane, and 32 property-chain edges per closure call.
  Overflow returns `STAR_FILTERED` or `0`, never heap growth.

**Operational semantics**

- `DL_SATURATE` performs one bounded tableau saturation pass over the
  current branch handle and returns a saturation status handle.
- `BLOCKING_CHECK` suppresses repeated expansion by comparing the active
  node summary against its predecessor chain within the existing branch
  handle.
- `EULER_COMPLETE` computes one transitive/property-chain completion
  pass over a bounded edge set and returns a fixpoint flag or updated
  handle.

**Tests**

- small DL tableau reaches saturation without clash
- blocking suppresses repeated expansion
- Euler/property-chain closure reaches fixpoint on a tiny graph

---

## 5. S15 — Abductive LP Upgrade

**Focus**

- `ALPCHAIN (0xB7)`
- upgrade Batch 2 `ICHECK` and `ABDRES` from minimal scalar semantics to
  Horn-style backward-chain semantics
- keep `ABDUCE_HALT` as the same halting surface

**Implementation shape**

- Define one bounded Horn clause encoding for the kernel surface.
- `ALPCHAIN` performs one backward-chain step.
- `ICHECK` validates integrity constraints over the same representation.
- `ABDRES` returns assumption-pool handles from the same bounded
  substrate.
- Use a fixed Horn window: up to 32 Horn rules, each with 1 head and up
  to 4 body literals, plus an 8-entry assumption pool per lane.

**Operational semantics**

- `ALPCHAIN` resolves one goal literal against one Horn head and emits a
  new residual goal handle.
- `ICHECK` evaluates integrity constraints against the same bounded goal
  and assumption representation; failed constraints return `0`.
- `ABDRES` collects unresolved literals into the bounded assumption pool
  and returns the updated pool handle.
- `ABDUCE_HALT` remains the same halting surface and is not redefined in
  this batch.

**Tests**

- one-step backward chain resolves a goal against a Horn rule
- integrity constraint blocks an otherwise-valid chain
- abductive resolution grows the assumption pool exactly once

---

## 6. S16 — Subjective Logic

**Focus**

- `EBELIEF (0xB0)`

**Implementation shape**

- Use ternary / `TQUANT`-compatible opinion triples already aligned with
  the substrate.
- No new probabilistic runtime.
- Implement evidence-to-belief update and uncertainty-threshold output
  only.
- Fix the opinion surface to a bounded triple:
  `(belief_q15, disbelief_q15, uncertainty_q15)` plus an evidence-mass
  input in the same q15 scaling.

**Operational semantics**

- `EBELIEF` updates one bounded opinion triple from evidence mass and
  outputs the updated opinion plus a thresholded status trit:
  `ok / defeasible / unresolved`.
- Contradictory evidence increases disbelief and/or uncertainty, but
  never collapses directly into hard-accept without crossing the same
  threshold rules.

**Tests**

- evidence update shifts belief toward supported hypothesis
- uncertainty shrinks with consistent evidence
- contradictory evidence stays defeasible instead of hard-accepting

---

## 7. S17 — Bi-Abduction / Frame Inference

**Focus**

- `BIDUCE (0xB1)`
- `FRAME (0xB2)`

**Implementation shape**

- Operate over bounded pre/post frame handles tied to Reality Galaxy
  semantics.
- No free-form separation-logic heap in Batch 3.
- Infer missing frame deltas from pre/post summaries only.
- Use fixed frame summaries with 16 slots each for preconditions,
  postconditions, preserved facts, and missing assumptions.

**Operational semantics**

- `FRAME` extracts the preserved summary between bounded pre/post handles
  and returns a compact frame handle or `0`.
- `BIDUCE` compares bounded pre/post summaries, infers missing
  assumptions, and returns a paired result handle containing both the
  missing assumptions and preserved frame summary.
- Incompatible summaries return `0` deterministically; no heap growth,
  no external solver.

**Tests**

- simple pre/post pair yields a nonzero frame handle
- incompatible pre/post returns `0`
- bi-abduction emits missing assumptions plus preserved frame summary

---

## 8. S18 — Rete / Semantic Rule Engines

**Focus**

- `RETE_ALPHA_TEST (0xE0)`
- `RETE_BETA_JOIN (0xE1)`
- `AGENDA_INSERT (0xE2)`

**Implementation shape**

- Use Galaxy rule-cluster tags plus already-landed `context_id` and
  `ethical_trit`.
- No Python agenda manager.
- Bounded GPU agenda only.
- Keep this batch to alpha/beta/agenda substrate, not full
  semantic-engine ingestion.
- Standardize on a fixed-capacity 32-entry agenda per lane with
  block-local bitonic insertion order. Overflow rejects deterministically
  with a saturated-agenda status code.

**Operational semantics**

- `RETE_ALPHA_TEST` filters fact handles by bounded predicate mask,
  `context_id`, and `ethical_trit` compatibility.
- `RETE_BETA_JOIN` joins bounded left/right token handles only when
  alpha-vetted bindings match under the existing substitution surface.
- `AGENDA_INSERT` inserts one bounded activation into the fixed-capacity
  agenda, preserving deterministic priority order by bitonic insertion.

**Tests**

- alpha test filters facts by bounded predicate mask
- beta join emits a joined handle only when both sides match
- agenda insert preserves bounded priority order and rejects overflow
  deterministically

---

## 9. Tests And Gates

Every slice ships with:

- sovereignty grep on modified surfaces
- focused `pytest` additions for the slice
- `git diff --check`
- targeted CUDA/PTX compile checks for any new kernel modules

Batch-level gates:

- `tests/test_batch3_superposition_rewrite.py`
- `tests/test_batch3_structured_logic_closure.py`
- `tests/test_batch3_abductive_lp.py`
- `tests/test_batch3_subjective_logic.py`
- `tests/test_batch3_biabduction.py`
- `tests/test_batch3_rete.py`

Notes:

- There is **no score-regression gate** in the spec body. Benchmark
  scores may legitimately move as semantics deepen.
- A Claude review checkpoint is optional after S15 or after S18, but it
  is not a blocker to landing the spec or beginning implementation.

---

## 10. Explicit Defers

- large HS ingestion waves from the parent spec
- full Frame Galaxy / Hypothesis Galaxy population
- loader cleanup
- daemon migration
- ethics ontology ingestion expansion
- broad benchmark retuning
- any unbounded clause heap
- any free-form separation-logic heap
- any new general graph runtime

---

## 11. Handoff Checklist

- Batch 3 remains runtime-first and benchmark-facing.
- All slices stay behind `K3D_REASONING_OPCODES_V1`.
- No slice introduces Python orchestration or fallback logic.
- Any minimal ingestion change is limited to one canonical source star
  such as `reasoning_kbo_precedence`.
- Existing Batch 2 substrates are reused before new surfaces are
  invented.
- Focused tests and compile gates land with each slice.
- `kimi_swarm` may be used for token economy only after the outer Codex
  MCP timeout is raised to 300 seconds; until then, implementation does
  not block on it.
