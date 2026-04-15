# Codex Batch 6 — Finish Reasoning Paradigm Dispatch Wiring

**Date:** 2026-04-14
**Parent spec:** `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`
**Predecessors:**
- Batches 1-4 (reasoning opcode surface + per-paradigm `rpn_*.cu`)
- Batch 5 (CBR / SUPERPOS / BIDUCE / EBELIEF / RETE dispatch wired,
  `ReasoningTickIO` / `ReasoningLaneOutput` ABI, real halting contract)
**Role:** Codex implements. Claude wrote this spec.
**Status:** Spec for implementation after Daniel approval.
**Scope guarantee:** Runtime-first wiring only. No new reasoning
kernels, no new opcodes, no ingestion pivot. Batch 6 drains the
remaining reasoning paradigms into the same persistent-swarm dispatch
pattern Batch 5 introduced, so that by end of batch every
`rpn_*.cu` slice with a master-plan paradigm mapping is callable
from a swarm lane.

---

## 0. Where We Stand After Batch 5

Batch 5 landed the shared tick-IO ABI and wired five paradigms into
`k3d_swarm_persistent.cu`:

```
REASONING_SLOT_CBR       → rpn_case_tick
REASONING_SLOT_SUPERPOS  → rpn_superpos_tick
REASONING_SLOT_BIDUCE    → rpn_frame_tick
REASONING_SLOT_EBELIEF   → rpn_ebelief_tick
REASONING_SLOT_RETE      → rpn_rete_tick
```

Halting contract is real (`lane_outputs[lane].halt_flag`), paradigm
assignment is GPU-only via `swarm_assign_paradigm_slot`, and
`tests/test_batch5_*` land `16 passed` alongside clean regression on
Batches 2-4. Validation evidence is in Codex's 2026-04-14 landing
report.

**Gap:** Batch 5 §10 "Explicit Defers" listed seven paradigms whose
`rpn_*.cu` slices exist and are test-green in isolation but are still
**not** dispatched from the persistent swarm:

```
rpn_tableaux.cu      (TSPLIT / TCLOSE / TEXPAND)
rpn_resolve.cu       (TRESOLVE)
rpn_unify.cu         (TUNIFY)
rpn_subsume.cu       (TSUBSUME)
rpn_alp.cu           (ALPCHAIN + ICHECK / ABDRES upgrades)
rpn_ctx_switch.cu    (CTX_SWITCH microtheory filter)
rpn_dpll.cu          (TBCP / TLEARNT)
```

Additionally, these three slices exist but are conceptually "support"
kernels that the other paradigms invoke rather than paradigms in their
own right:

```
rpn_abduce.cu        (ABDUCE / EXPLAIN / SUSPECT — already reached
                      through CBR's causal retrieval path in Batch 5)
rpn_abduce_ext.cu    (ABDUCE_HALT / SCUNION / ICHECK / ABDRES / ABDNEG)
rpn_order.cu         (TORDER — used by SUPERPOS/REWRITE under the hood)
rpn_rewrite.cu       (TREWRITE — used by SUPERPOS)
rpn_halt.cu          (HALT_SET / HALT_SYNC — used by every halting path)
```

Batch 6 wires the **seven paradigms** in the first list into the
persistent swarm dispatcher. The second list does not get new
`REASONING_SLOT_*` IDs; those slices keep their current role as inner
kernels called by the paradigm-level tick functions.

No new kernels, no new opcodes, no new headers beyond an extended
`reasoning_tick_io.cuh` enum and extended `reasoning_tick_entrypoints.cuh`
inline functions.

---

## 1. Hard Rules

- **`K3D_REASONING_OPCODES_V1` continues to gate every new surface.**
- **PTX + Galaxy + RPN + TRM only** in the hot path. No Python
  orchestration, no hot-path fallbacks, no dynamic heaps, no cupy.
- **No ABI break.** `ReasoningTickIO` stays 32 bytes. `ReasoningLaneOutput`
  stays 64 bytes. `SwarmTickControl` stays 32 bytes. `static_assert`s
  preserved.
- **No new rpn_*.cu files.** Each target slice gets a `__device__`
  tick entry added in-place, behind the same `K3D_REASONING_OPCODES_V1`
  gate as Batch 5.
- **No removal or rewrite** of the existing `__global__` surfaces used
  by Batches 2-4 tests. Those must stay green at the end of Batch 6.
- **Dispatch remains a flat switch**, no function pointer tables,
  no runtime indirection. Paradigm slot IDs packed low-to-high so the
  existing `swarm_assign_paradigm_slot` helper keeps working unchanged.
- **Bounded per-lane work.** Every new tick entry point honors the same
  bounded budget already used by its `__global__` counterpart. No heap
  growth, no scratch allocation, no new VRAM pools.
- **No score-regression gate.** Numbers may move as new paradigms
  become live. Broader benchmark retune is tracked separately.
- **No Python paradigm selection.** Host may only set
  `paradigm_mask` bits on the host-mapped `SwarmTickControl`.
- **ALPCHAIN stays bounded.** Horn window is identical to Batch 3 §5
  (32 rules, 1 head + 4 body, 8 assumption pool entries). No unbounded
  clause or goal heap.

---

## 2. Slice Ordering

1. **S25 — Extend `ReasoningParadigmSlot` enum and `paradigm_mask` semantics**
2. **S26 — `rpn_tableaux_tick` + dispatch wire**
3. **S27 — `rpn_resolve_tick` + `rpn_unify_tick` + `rpn_subsume_tick` wire**
4. **S28 — `rpn_alp_tick` wire (ALPCHAIN + ICHECK/ABDRES/ABDUCE_HALT)**
5. **S29 — `rpn_dpll_tick` wire (TBCP + TLEARNT)**
6. **S30 — `rpn_ctx_switch_tick` wire + context-filtered dispatch regression**
7. **S31 — Multi-paradigm concurrency stress (popcount = 7)**
8. **S32 — Rete agenda bitonic ordering gate (§7 item 10 cleanup)**

Order rationale:

- S25 lands the enum extension first so every following slice compiles
  against a stable paradigm ID.
- S26 (tableaux) is next because the master plan §7 item 5 names
  tableaux "the cheapest ATP addition with LHE multi-hop payoff," and
  it already fits the swarm natively per §2.
- S27 adds the resolution/unification/subsumption family together —
  they share the Batch 2 substitution arena and are useless apart.
- S28 lands ALPCHAIN using the Horn window already sized by Batch 3.
- S29 adds the DPLL substrate last among the ATP families because it
  depends on the resolution family being live first.
- S30 enables CTX_SWITCH as a context filter over the persistent swarm
  dispatch, closing master plan §8 item 4.
- S31 is the concurrency stress: all seven new slots + the five Batch 5
  slots active simultaneously (`paradigm_mask` popcount = 12). This
  catches any warp-divergence meltdown before benchmarks hit it.
- S32 closes master plan §7 item 10 by adding the bitonic agenda-order
  gate to the existing Rete slot, fulfilling the last explicit §7
  checklist item.

---

## 3. S25 — Extend `ReasoningParadigmSlot` Enum

**Focus**

- Extend `knowledge3d/cranium/cuda/reasoning_tick_io.cuh`:

```c
enum ReasoningParadigmSlot : uint32_t {
    REASONING_SLOT_NONE       = 0u,
    REASONING_SLOT_CBR        = 1u,
    REASONING_SLOT_SUPERPOS   = 2u,
    REASONING_SLOT_BIDUCE     = 3u,
    REASONING_SLOT_EBELIEF    = 4u,
    REASONING_SLOT_RETE       = 5u,
    REASONING_SLOT_TABLEAUX   = 6u,   // NEW — rpn_tableaux_tick
    REASONING_SLOT_RESOLUTION = 7u,   // NEW — rpn_resolve_tick
    REASONING_SLOT_ALPCHAIN   = 8u,   // NEW — rpn_alp_tick
    REASONING_SLOT_DPLL       = 9u,   // NEW — rpn_dpll_tick
    REASONING_SLOT_CTX_SWITCH = 10u,  // NEW — rpn_ctx_switch_tick
    REASONING_SLOT_SUBSUME    = 11u,  // NEW — rpn_subsume_tick
    REASONING_SLOT_UNIFY      = 12u,  // NEW — rpn_unify_tick
    // 13..15 reserved for a later batch
};
```

**Rules**

- `K3D_REASONING_PARADIGM_MAX` stays 16.
- Enum entries are packed low-to-high so `swarm_assign_paradigm_slot`
  keeps returning the `popcount`-th set bit correctly without
  rework.
- `paradigm_mask` bit `k` continues to mean "paradigm slot `k` is
  active this query."
- No change to `ReasoningTickIO` or `ReasoningLaneOutput` byte layout.

**Tests**

- `tests/test_batch6_paradigm_slot_abi.py`:
  - ctypes sizeof unchanged (ABIs stable).
  - Enum values compile-checked via a small CUDA harness that reads
    each constant and asserts its numeric value matches the spec.
  - `static_assert`s fire in nvcc compile.

---

## 4. S26 — `rpn_tableaux_tick` Wire

**Focus**

- Add a `__device__ __forceinline__ void rpn_tableaux_tick(const
  ReasoningTickIO& io, ReasoningLaneOutput* out)` to
  `reasoning_tick_entrypoints.cuh`, calling into bounded primitives
  already present inside `knowledge3d/cranium/cuda/rpn_tableaux.cu`.

**Implementation shape**

- The tick entry performs one bounded tableau step per call:
  1. Read the current branch handle from `io.query_handle`.
  2. Invoke one `TSPLIT` + one `TEXPAND` on a per-lane branch buffer of
     64 assertions and 16 branch handles (Batch 3 §4 caps).
  3. Invoke `TCLOSE` to check for clash.
  4. If clash reached: write `halt_flag=1, result_handle=closed_id,
     belief_q15=32768`. If saturated without clash: `halt_flag=1,
     result_handle=branch_id, belief_q15=16384`. Else: `halt_flag=0`.
- Scratch lives in a per-lane fixed-size struct on the stack of the
  tick entry, not in shared memory.
- Context and ethical filtering honored exactly as in Batch 5 (drop
  stars where `context_id` mismatches `io.context_id != 0`, or where
  `ethical_trit == -1`).

**Dispatch wire**

Extend the `k3d_swarm_dispatch_paradigm` switch in
`k3d_swarm_persistent.cu`:

```c
case REASONING_SLOT_TABLEAUX: rpn_tableaux_tick(io, out); break;
```

**Tests**

- `tests/test_batch6_tableaux_dispatch.py`:
  - Seed a 3-node branch with a known clash. Run persistent swarm
    with `paradigm_mask = (1u << REASONING_SLOT_TABLEAUX)`. Assert
    `halt_flag == 1` and `belief_q15 >= 16384` within ≤ 64 ticks.
  - Seed a 3-node branch that saturates without clash. Assert
    `halt_flag == 1` and `belief_q15` is the saturation value.
  - Batch 3 regression: `tests/test_batch3_*tableau*.py` stays green.

---

## 5. S27 — Resolution Family Wire (TRESOLVE + TUNIFY + TSUBSUME)

**Focus**

- Add three inline `__device__` tick entries to
  `reasoning_tick_entrypoints.cuh`:
  - `rpn_resolve_tick`
  - `rpn_unify_tick`
  - `rpn_subsume_tick`

**Implementation shape**

- All three share the Batch 2 substitution arena already declared
  inside `rpn_unify.cu`. No new arena allocation.
- `rpn_unify_tick` performs one unification step on a literal pair
  drawn from `io.query_handle`. On success, writes
  `result_handle = subst_id, halt_flag = 1`. On failure,
  `result_handle = 0, halt_flag = 1`.
- `rpn_resolve_tick` performs one resolution step reusing the Batch 2
  `TUNIFY` substitution. It reads two clause handles from the bounded
  per-lane resolution scratch (64-literal ceiling, same as Batch 3 S13),
  emits at most one resolvent handle per tick. If the resolvent is
  the empty clause, `halt_flag = 1, result_handle = 0xFFFFFFFFu`
  (reserved "empty-clause" sentinel). Otherwise it writes the new
  resolvent handle and `halt_flag = 0` until either clash or
  saturation is reached, then `halt_flag = 1`.
- `rpn_subsume_tick` checks whether clause A subsumes clause B under
  the existing unification surface. Returns `halt_flag = 1` and writes
  `result_handle = 1` (subsumes) or `0` (does not), plus
  `belief_q15 = 32768` on subsumption and `0` otherwise.

**Dispatch wire**

```c
case REASONING_SLOT_RESOLUTION: rpn_resolve_tick(io, out); break;
case REASONING_SLOT_UNIFY:      rpn_unify_tick(io, out);   break;
case REASONING_SLOT_SUBSUME:    rpn_subsume_tick(io, out); break;
```

**Tests**

- `tests/test_batch6_resolution_family_dispatch.py`:
  - Unify a pair that succeeds → nonzero subst handle.
  - Unify a pair that fails → zero handle, halt_flag = 1.
  - Resolve two complementary single-literal clauses → empty-clause
    sentinel, halt_flag = 1.
  - Subsume A ⊑ B small case → result_handle = 1, belief_q15 = 32768.
  - Batch 2 regression: `tests/test_batch2_resolution_opcodes.py`
    stays green.

---

## 6. S28 — `rpn_alp_tick` Wire (Abductive LP)

**Focus**

- Add `__device__ __forceinline__ void rpn_alp_tick(const
  ReasoningTickIO& io, ReasoningLaneOutput* out)` to
  `reasoning_tick_entrypoints.cuh`, wrapping the existing
  `rpn_alp.cu` backward-chain / integrity-check / assumption-pool
  primitives.

**Implementation shape**

- Horn window is identical to Batch 3 §5: 32 rules, 1 head + ≤ 4
  body literals, 8-entry assumption pool.
- Tick performs one backward-chain step:
  1. Read current goal handle from `io.query_handle`.
  2. Invoke `ALPCHAIN` once against the Horn window.
  3. If the goal literal resolves, invoke `ICHECK` over the same
     representation. On success, invoke `ABDRES` to append any
     unresolved literal into the assumption pool, then check
     `ABDUCE_HALT`.
  4. On `ABDUCE_HALT == 1`: `halt_flag = 1, result_handle = pool_id,
     belief_q15 = 32768`. On integrity failure:
     `halt_flag = 1, result_handle = 0, belief_q15 = 0`.
     Otherwise `halt_flag = 0`, next tick continues the chain.
- No heap. No new assumption pool structure. All state fits in the
  existing per-lane Horn window declared by Batch 3.

**Dispatch wire**

```c
case REASONING_SLOT_ALPCHAIN: rpn_alp_tick(io, out); break;
```

**Tests**

- `tests/test_batch6_alp_dispatch.py`:
  - Single-step backward chain resolves a goal against a Horn rule →
    result_handle != 0 when `ABDUCE_HALT` fires.
  - Integrity constraint blocks an otherwise-valid chain →
    result_handle = 0, halt_flag = 1.
  - Abductive resolution grows the assumption pool exactly once per
    tick and eventually halts.
  - Batch 3 regression: `tests/test_batch3_abductive_lp.py` stays
    green.

---

## 7. S29 — `rpn_dpll_tick` Wire (SAT Substrate)

**Focus**

- Add `__device__ __forceinline__ void rpn_dpll_tick(const
  ReasoningTickIO& io, ReasoningLaneOutput* out)` to
  `reasoning_tick_entrypoints.cuh`, wrapping the Batch 2 `TBCP` +
  `TLEARNT` primitives.

**Implementation shape**

- Tick performs one two-watch BCP step followed by at most one 1-UIP
  learn step (the same semantics already in `rpn_dpll.cu`). Clause
  representation is identical to Batch 2 §12 — bounded clause arena
  with per-lane 128-literal ceiling, no heap.
- On `⊥`: `halt_flag = 1, result_handle = 0, belief_q15 = 0`.
- On satisfying assignment: `halt_flag = 1, result_handle = model_id,
  belief_q15 = 32768`.
- Otherwise `halt_flag = 0`.

**Dispatch wire**

```c
case REASONING_SLOT_DPLL: rpn_dpll_tick(io, out); break;
```

**Tests**

- `tests/test_batch6_dpll_dispatch.py`:
  - Seed a trivially satisfiable 2-clause CNF → halt with
    result_handle != 0.
  - Seed a trivially unsatisfiable 2-clause CNF → halt with
    result_handle = 0.
  - Seed a 4-clause CNF that requires one learnt clause to close →
    halt within ≤ 16 ticks, `belief_q15 = 32768`.
  - Batch 2 regression: `tests/test_batch2_dpll_opcodes.py` stays
    green.

---

## 8. S30 — `rpn_ctx_switch_tick` Wire + Context Filter

**Focus**

- Add `__device__ __forceinline__ void rpn_ctx_switch_tick(const
  ReasoningTickIO& io, ReasoningLaneOutput* out)` to
  `reasoning_tick_entrypoints.cuh`.
- Wire it into the persistent-swarm dispatch switch.

**Implementation shape**

- `rpn_ctx_switch_tick` reads `io.context_id` and writes a per-lane
  "context view" handle into `out->result_handle`. Semantics match
  master plan §8.1: "filter subsequent Galaxy reads to stars whose
  `context_id` matches top-of-stack, or `0`."
- The filter handle is a bounded struct (context ID + inclusivity
  flag + ethical default trit) packed into the opaque payload bytes
  of `ReasoningLaneOutput::payload[0..7]`.
- **No new galaxy.** No `MT_INDEX`. All reads stay against the
  canonical star schema where `context_id` is a per-star field.
- Tick always halts in one step: `halt_flag = 1`.

**Dispatch wire**

```c
case REASONING_SLOT_CTX_SWITCH: rpn_ctx_switch_tick(io, out); break;
```

**Cross-slice context regression**

- When `paradigm_mask` includes `REASONING_SLOT_CTX_SWITCH` plus at
  least one paradigm from {CBR, TABLEAUX, ALPCHAIN, DPLL}, the
  swarm must honor the context filter emitted by the context lane on
  the next tick. The simplest way to enforce this is for the other
  tick entry points to read the context-view payload from their own
  `ReasoningLaneOutput` (which is per-lane, so they re-read their
  own lane's context handle set on the prior tick) and skip stars
  whose `context_id` mismatches.
- No shared mutable state between lanes.

**Tests**

- `tests/test_batch6_ctx_switch_dispatch.py`:
  - Seed two Galaxy stars, one with `context_id = 0`, one with
    `context_id = 42`. Run with
    `paradigm_mask = (1u << REASONING_SLOT_CTX_SWITCH) | (1u << REASONING_SLOT_CBR)`
    and `io.context_id = 42`. Assert the CBR lane only retrieves the
    `context_id = 42` star.
  - Repeat with `io.context_id = 0`: both stars are retrievable.
  - Repeat with `io.context_id = 42` and `ethical_trit == -1` on the
    target star: retrieval blocked, `halt_flag = 1, result_handle = 0`.

---

## 9. S31 — Multi-Paradigm Concurrency Stress

**Focus**

- Run the persistent swarm with **12 paradigm slots active**
  (all five Batch 5 slots + all seven Batch 6 slots) and
  `phys_lane_id ∈ [0..31]`. Verify no hang, no NaN, no non-halting
  tick, no warp lockstep meltdown.

**Rules**

- No new kernel code. The test exercises the existing dispatch
  switch under maximum legal fanout.
- `n_floor = 12`, `n_hard_max = 32`.
- Test must compute `popcount(paradigm_mask) == 12` on the host
  side as a sanity check.

**Tests**

- `tests/test_batch6_multi_paradigm_stress.py`:
  - All 12 slots active, 32 lanes, 256 ticks. Assert every lane
    halts at least once during the run, `halt_epoch >= 1`, no
    NaN/Inf in `belief_q15`, and no lane writes outside its own
    `ReasoningLaneOutput` slot.
  - Log the dispatch histogram — each paradigm must be exercised by
    at least one lane during the run.

---

## 10. S32 — Rete Agenda Bitonic Ordering Gate

**Focus**

- Close master plan §7 item 10 by asserting a deterministic bitonic
  priority order on the Rete agenda already built inside
  `rpn_rete.cu`.

**Implementation shape**

- No new kernel work beyond a test fixture that verifies the agenda
  insert order is bitonic-stable under bounded insertion and
  overflow rejection (32-entry cap from Batch 3 §8).
- If `rpn_rete.cu` already enforces bitonic order, the test is a
  regression gate. If the current implementation only sorts on
  evict, the test escalates to a Claude review note — Codex must not
  relax the 32-entry cap or the deterministic rejection on overflow.

**Tests**

- `tests/test_batch6_rete_agenda_bitonic.py`:
  - Insert 64 activations with increasing priority into a 32-entry
    agenda. Assert the top 32 survive in priority order and the
    lowest 32 are rejected deterministically with the existing
    saturated-agenda status code.
  - Regression: `tests/test_batch3_rete.py` stays green and
    `tests/test_batch5_reasoning_dispatch_switch.py` stays green.

---

## 11. Tests And Gates

Every slice ships with:

- sovereignty grep on modified surfaces
- focused `pytest` additions for the slice
- `git diff --check`
- targeted CUDA/PTX compile checks for any new header content or
  `__device__` entry points
- no new Python logic in the reasoning path

Batch-level gates:

- `tests/test_batch6_paradigm_slot_abi.py`
- `tests/test_batch6_tableaux_dispatch.py`
- `tests/test_batch6_resolution_family_dispatch.py`
- `tests/test_batch6_alp_dispatch.py`
- `tests/test_batch6_dpll_dispatch.py`
- `tests/test_batch6_ctx_switch_dispatch.py`
- `tests/test_batch6_multi_paradigm_stress.py`
- `tests/test_batch6_rete_agenda_bitonic.py`

Non-regression required (existing green stays green):

- `tests/test_batch2_*.py`
- `tests/test_batch3_*.py`
- `tests/test_batch4_*.py`
- `tests/test_batch5_*.py`
- `tests/test_n_chain_persistent_launch.py`

Notes:

- **No score-regression gate** in this batch.
- A Claude review checkpoint is optional after S30 or after S32, but
  it is not a blocker to landing the spec or beginning implementation.

---

## 12. Explicit Defers

- HS curriculum ingestion waves (Batch 7+ / parallel ingestion track).
- Reasoning taxonomy catalogue ingestion (master plan §9.1) — that is
  the next ingestion wave after reasoning dispatch is complete.
- Frame Galaxy / Hypothesis Galaxy population beyond the fixtures
  that Batch 5/6 tests seed.
- KBO precedence star population beyond the minimal Batch 3 seed.
- Sleep-time consolidation upgrades driven by the new paradigms
  (Phase D workstream).
- `knowledgeverse.py` 4000→200 line reduction.
- Daemon migration.
- Broad benchmark retuning and score-regression gating.
- Any Python-side paradigm selection, scoring, or dispatch.

---

## 13. Runway Toward HS Curriculum Materialization

Batch 6 closes the last runtime gap between the master plan §2
paradigm mapping and the persistent swarm. Once S25–S32 land:

1. Every master-plan reasoning paradigm has a live `__device__` tick
   entry callable from the persistent swarm.
2. Context filtering works via `CTX_SWITCH` on the canonical star
   schema, no new index.
3. Ethical trit filtering works via the Batch 1 schema extension
   honored by every tick entry.
4. Dispatch is a flat switch, bounded, sovereign, and free of Python.

At that point the sovereign reasoning substrate is **ready to consume
the HS curriculum payload**. The next spec (Batch 7) should be the
reasoning-taxonomy ingestion wave (master plan §9.1), followed by the
HS catalogue ingestion order from master plan §9.4:

```
Batch 7  — Reasoning taxonomy ingestion (§9.1 all four files)
Batch 8  — HS math clusters 1 → 2 → 3
Batch 9  — HS natural + earth/space sciences
Batch 10 — HS languages + linguistics (with context_id per region)
Batch 11 — HS history + civics + economics
Batch 12 — HS humanities + philosophy + ethics (sets ethical_trit)
Batch 13 — HS applied / CS / health / psych / sociology
Batch 14 — Cross-cultural glue (saudades, calendars, units, proverbs)
```

Those are ingestion-path batches, so Python tools are allowed there
(master plan §6 + CLAUDE.md "Ingestion Path = Flexible") provided the
result is sovereign Galaxy entries. Claude will spec each as it comes.

---

## 14. Handoff Checklist

- Batch 6 is runtime-first and wiring-only. No new kernels, no new
  opcodes, no new Galaxy writes.
- All new surfaces stay behind `K3D_REASONING_OPCODES_V1`.
- No slice introduces Python orchestration or hot-path fallback logic.
- Existing Batch 1-5 `__global__` launch surfaces and their tests stay
  green.
- `ReasoningTickIO`, `ReasoningLaneOutput`, and `SwarmTickControl`
  ABIs stay byte-stable; `static_assert`s preserved.
- `ReasoningParadigmSlot` enum extends low-to-high with the seven new
  IDs; reserved range 13..15 left untouched.
- Dispatch helper is a flat `switch`, no function pointer tables.
- Multi-paradigm stress test proves concurrent dispatch works at
  popcount = 12.
- Sovereignty grep (`numpy|cupy|scipy|sympy|re`) clean on all modified
  surfaces.
