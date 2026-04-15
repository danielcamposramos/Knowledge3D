# Codex Batch 5 — Swarm → Reasoning Kernel Dispatch Wiring

**Date:** 2026-04-13
**Parent spec:** `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`
**Predecessors:**
- `TEMP/CODEX_BATCH1_OPCODES_AND_KERNELS_2026-04-13.md` (landed)
- `TEMP/CODEX_BATCH2_OPCODES_AND_KERNELS_2026-04-13.md` (landed)
- `TEMP/CODEX_BATCH3_OPCODES_AND_KERNELS_2026-04-13.md` (landed)
- Batch 4 (CBR + context + model-check + n-sweep, landed)
**Role:** Codex implements. Claude wrote this spec.
**Status:** Spec for implementation after Daniel approval.
**Scope guarantee:** Runtime-first wiring only. No new reasoning kernels,
no new opcodes, no ingestion pivot, no loader cleanup wave. This batch
makes the Batch 1-4 kernels that already compile actually run inside the
persistent swarm hot path.

---

## 0. Where We Stand After Batch 4

Batches 1-4 fully landed the reasoning opcode surface and one `.cu`
slice per paradigm family inside `knowledge3d/cranium/cuda/`:

```
rpn_abduce.cu        rpn_abduce_ext.cu    rpn_alp.cu
rpn_case.cu          rpn_ctx_switch.cu    rpn_dpll.cu
rpn_ebelief.cu       rpn_frame.cu         rpn_halt.cu
rpn_order.cu         rpn_resolve.cu       rpn_rete.cu
rpn_rewrite.cu       rpn_subsume.cu       rpn_superpos.cu
rpn_tableaux.cu      rpn_unify.cu
```

All slices compile behind `K3D_REASONING_OPCODES_V1`. All have focused
`pytest` gates (`tests/test_batch{2,3,4}_*.py`). All reasoning opcode
constants in `rpn_opcodes.py` bind to the master-plan addresses (post
2026-04-13 Kimi cleanup).

**Gap:** `knowledge3d/cranium/cuda/k3d_swarm_persistent.cu` — the
sovereign persistent cooperative kernel that is supposed to drive the
hot path — still runs a placeholder `lane_summary_mix` and a synthetic
`local_halt`. None of the Batch 1-4 `rpn_*.cu` slices are actually
**called** from a swarm lane. They are green in isolation and dead in
composition.

Batch 5 closes that gap. No new paradigms, no new opcodes, no new
kernels. Only dispatch wiring, one shared tick-IO ABI, and a real
halting contract.

---

## 1. Hard Rules

- **`K3D_REASONING_OPCODES_V1` continues to gate every new surface.**
- **PTX + Galaxy + RPN + TRM only** in the hot path. No Python
  orchestration, no hot-path fallbacks, no dynamic heaps, no cupy.
- **Reuse existing slices.** No new `rpn_*.cu` files are created in this
  batch. Existing slices may grow `__device__` tick entry points, but
  their `__global__` launch surfaces must remain callable and
  test-green.
- **No ABI break** for `SwarmTickControl` total size. The existing
  reserved slot (`uint32_t _rsv`) is renamed to `paradigm_mask`; total
  stays at 32 bytes. `static_assert` preserved.
- **Bounded per-lane output.** Each lane writes at most a fixed-size
  reasoning result record (≤ 64 bytes) to a pre-allocated VRAM output
  array sized to `K3D_SWARM_N_HARD_MAX`. No heap.
- **Halting contract is real.** `halting_gate_step` must see the lane's
  reasoning halt bit, not a synthetic `(cross_ref | local_summary | 1u)`
  placeholder.
- **No score-regression gate** in this batch. End-to-end wiring may
  legitimately move benchmark numbers. A single smoke test proves the
  path works; broader benchmark deltas are tracked separately.
- **No Python dispatch selection.** Paradigm slot assignment per lane
  happens on GPU only. Python may set the `paradigm_mask` bitfield on
  the host-mapped `SwarmTickControl` and nothing else.
- **No new kernels, no new opcodes, no new stars** except the one
  canonical output record layout.

---

## 2. Slice Ordering

1. **S19 — Reasoning Tick-IO ABI (single source of truth)**
2. **S20 — Per-Lane Paradigm Assignment From `paradigm_mask`**
3. **S21 — CBR First-Light Dispatch in Persistent Swarm**
4. **S22 — Extend Dispatch to TSUPERPOS, BIDUCE, EBELIEF, RETE**
5. **S23 — Real Halting Contract From Lane Output**
6. **S24 — End-to-End Reasoning Smoke Test**

Order rationale:

- S19 defines the shared ABI so every rpn_*.cu slice can expose a
  uniform `__device__` entry point without churning its launch surface.
- S20 adds paradigm-to-lane routing on GPU only, driven by the
  reserved-slot rename, so the rest of the batch has a selector.
- S21 wires the first real reasoning dispatch (CBR) end-to-end inside
  the persistent swarm. CBR is chosen first because Batch 4 already
  ships a green `rpn_case.cu` plus an existing `tests/test_batch4_context_cbr.py`
  that can seed the smoke test.
- S22 extends the same dispatcher pattern to four more paradigms.
  Everything else (ALPCHAIN, resolution/tableaux/DPLL) can be wired in
  a later batch without changing the ABI.
- S23 replaces the synthetic `local_halt` with the real halt bit coming
  out of the reasoning tick-IO record.
- S24 closes the batch with a single end-to-end smoke test that proves
  a query traverses persistent swarm → reasoning dispatch → halt →
  readable output. No benchmark retune.

---

## 3. S19 — Reasoning Tick-IO ABI

**Focus**

- Introduce one canonical `ReasoningTickIO` struct.
- Introduce one canonical `ReasoningLaneOutput` record.
- Land both in a single header under
  `knowledge3d/cranium/cuda/reasoning_tick_io.cuh`.

**Struct shape (authoritative)**

```c
// knowledge3d/cranium/cuda/reasoning_tick_io.cuh
#pragma once
#include <stdint.h>

#define K3D_REASONING_LANE_OUTPUT_BYTES 64u
#define K3D_REASONING_PARADIGM_MAX 16u

// Paradigm slot IDs. Order matches bit index in SwarmTickControl.paradigm_mask.
// Unassigned slots dispatch nothing.
enum ReasoningParadigmSlot : uint32_t {
    REASONING_SLOT_NONE      = 0u,
    REASONING_SLOT_CBR       = 1u,
    REASONING_SLOT_SUPERPOS  = 2u,
    REASONING_SLOT_BIDUCE    = 3u,
    REASONING_SLOT_EBELIEF   = 4u,
    REASONING_SLOT_RETE      = 5u,
    // reserved 6..15 for a later batch (ALPCHAIN, resolution,
    // tableaux, DPLL, subsume, etc.)
};

struct __align__(16) ReasoningTickIO {
    const uint8_t* __restrict__ galaxy_atlas;
    uint32_t phys_lane_id;
    uint32_t tick_seed;
    uint32_t paradigm_slot;     // ReasoningParadigmSlot value
    uint32_t query_handle;      // bounded query id (Galaxy star id, truncated)
    uint32_t context_id;        // microtheory context from Batch 1 star schema
    int8_t   ethical_trit;      // master plan int8 (-1/0/+1)
    uint8_t  _pad0[3];
};

static_assert(sizeof(ReasoningTickIO) == 32,
              "ReasoningTickIO ABI must remain 32 bytes");

// Fixed-size per-lane output. Opaque bytes — each paradigm writes its
// own bounded handle layout inside this window.
struct __align__(16) ReasoningLaneOutput {
    uint32_t halt_flag;         // 1 if this lane converged this tick
    uint32_t result_handle;     // paradigm-specific bounded handle (0 = empty)
    uint32_t belief_q15;        // q15 confidence (0..32768)
    uint32_t _pad0;
    uint8_t  payload[K3D_REASONING_LANE_OUTPUT_BYTES - 16u];
};

static_assert(sizeof(ReasoningLaneOutput) == K3D_REASONING_LANE_OUTPUT_BYTES,
              "ReasoningLaneOutput ABI must remain 64 bytes");
```

**Operational semantics**

- One `ReasoningTickIO` is materialized per lane per tick, entirely on
  GPU. Python never touches it.
- One `ReasoningLaneOutput` is reserved per lane for the lifetime of
  the swarm (sized to `K3D_SWARM_N_HARD_MAX`). Swarm writes in place
  every tick. Host may read asynchronously via mapped memory.
- `paradigm_slot == REASONING_SLOT_NONE` means "no reasoning dispatch
  this tick" and must write `halt_flag = 1, result_handle = 0` so the
  halting gate still converges deterministically.

**Tests**

- `tests/test_batch5_reasoning_tick_io_abi.py` — ctypes sizeof checks
  for both structs; header compiles under nvcc with
  `K3D_REASONING_OPCODES_V1`.

---

## 4. S20 — Per-Lane Paradigm Assignment From `paradigm_mask`

**Focus**

- Rename `SwarmTickControl._rsv` → `SwarmTickControl.paradigm_mask`
  in `knowledge3d/cranium/cuda/n_selector.cu`.
- Add one `__device__ __forceinline__ uint32_t
  swarm_assign_paradigm_slot(const SwarmTickControl* control, uint32_t
  phys_lane_id)` helper that round-robins assigned bits across lanes.

**Implementation shape**

- `paradigm_mask` is a bitmask. Bit `k` set means "paradigm slot `k` is
  active this query." Bit 0 (`REASONING_SLOT_NONE`) is always set
  implicitly when the mask is zero so the swarm still halts cleanly.
- `swarm_assign_paradigm_slot` computes an assignment by:
  1. `popcount(mask)` → number of active slots.
  2. If `popcount == 0`, return `REASONING_SLOT_NONE`.
  3. `slot_index = phys_lane_id % popcount`.
  4. Walk the mask bits low-to-high and return the `slot_index`-th set
     bit as the paradigm slot id.
- No Python scheduling. Assignment is pure GPU and deterministic per
  `(paradigm_mask, phys_lane_id)`.

**Rules**

- ABI: `SwarmTickControl` total size must remain 32 bytes. The existing
  `static_assert` must continue to hold.
- All paradigm-mask consumers use the helper — no inline popcount
  drift.

**Tests**

- `tests/test_batch5_paradigm_assignment.py` — host-side parity table
  (ctypes + small CUDA harness): for `paradigm_mask ∈ {0, 0b10,
  0b110, 0b11110}` and `phys_lane_id ∈ [0..31]`, verify the returned
  slot matches a reference Python-side computation used only as
  oracle, never in hot path.

---

## 5. S21 — CBR First-Light Dispatch In Persistent Swarm

**Focus**

- Expose a `__device__` tick entry point on `rpn_case.cu`:
  ```c
  // Added to rpn_case.cu, behind K3D_REASONING_OPCODES_V1.
  __device__ void rpn_case_tick(const ReasoningTickIO& io,
                                ReasoningLaneOutput* out);
  ```
- Wire `k3d_swarm_persistent.cu` so every active lane:
  1. Builds a `ReasoningTickIO`.
  2. Calls `swarm_assign_paradigm_slot(...)`.
  3. Dispatches `rpn_case_tick(io, &lane_out[phys_lane_id])` when the
     assigned slot is `REASONING_SLOT_CBR`.
  4. Writes `halt_flag=1, result_handle=0` when the assigned slot is
     `REASONING_SLOT_NONE`.

**Rules**

- The existing `rpn_case` `__global__` launch surface used by Batch 4
  tests must remain callable and test-green. S21 adds a new
  `__device__` entry, it does not rewrite the old one.
- `rpn_case_tick` must be **bounded**: at most 4 case retrievals, 4
  rebind attempts, 1 revise pass per tick. These are the same caps
  already used inside the existing `rpn_case.cu` kernel. No new heap.
- `rpn_case_tick` reads `io.context_id` and `io.ethical_trit` from the
  Batch 1 star schema extensions. Cases tagged with an incompatible
  context or `ethical_trit == -1` (forbidden) must be filtered out and
  the lane must still write a clean output record with
  `result_handle=0, halt_flag=1`.

**Tests**

- `tests/test_batch5_cbr_first_light.py`:
  1. Compile the persistent swarm with
     `paradigm_mask = (1u << REASONING_SLOT_CBR)`.
  2. Seed one Galaxy case that matches a synthetic query star.
  3. Launch persistent swarm for a bounded epoch (≤ 64 ticks).
  4. Read host-mapped `lane_outputs[0]` and assert
     `result_handle != 0` and `halt_flag == 1`.
  5. Assert the existing `tests/test_batch4_context_cbr.py` is still
     green (it exercises the `__global__` surface).

---

## 6. S22 — Extend Dispatch To TSUPERPOS, BIDUCE, EBELIEF, RETE

**Focus**

- Add `__device__` tick entry points to the four existing slices:
  - `rpn_superpos.cu` → `rpn_superpos_tick(io, out)`
  - `rpn_frame.cu`    → `rpn_frame_tick(io, out)`
  - `rpn_ebelief.cu`  → `rpn_ebelief_tick(io, out)`
  - `rpn_rete.cu`     → `rpn_rete_tick(io, out)`

**Dispatch shape inside the swarm**

Centralize the dispatch in a tiny inline helper inside
`k3d_swarm_persistent.cu`:

```c
__device__ __forceinline__ void k3d_swarm_dispatch_paradigm(
    uint32_t slot,
    const ReasoningTickIO& io,
    ReasoningLaneOutput* out
) {
    switch (slot) {
        case REASONING_SLOT_CBR:      rpn_case_tick(io, out);     break;
        case REASONING_SLOT_SUPERPOS: rpn_superpos_tick(io, out); break;
        case REASONING_SLOT_BIDUCE:   rpn_frame_tick(io, out);    break;
        case REASONING_SLOT_EBELIEF:  rpn_ebelief_tick(io, out);  break;
        case REASONING_SLOT_RETE:     rpn_rete_tick(io, out);     break;
        default:                      // REASONING_SLOT_NONE and reserved
            out->halt_flag = 1u;
            out->result_handle = 0u;
            out->belief_q15 = 0u;
            break;
    }
}
```

**Rules**

- The switch is literal and flat. No function pointer tables. No
  runtime indirection. Warp divergence inside the switch is
  acceptable for this batch because paradigm assignment is
  lane-uniform within a warp when `popcount(paradigm_mask) <= 4` and
  the assignment helper groups bits low-to-high; multi-paradigm
  runs still function correctly, they simply diverge.
- Every tick entry point must honor the same bounded-tick budget
  used by its existing `__global__` surface. No tick may allocate
  memory.
- ALPCHAIN / resolution / tableaux / DPLL / subsume are **explicitly
  deferred** to a later batch. They stay as `__global__`-only surfaces
  for now and keep their Batch 2/3 tests green.

**Tests**

- `tests/test_batch5_reasoning_dispatch_switch.py`:
  - For each of `{CBR, SUPERPOS, BIDUCE, EBELIEF, RETE}`, launch the
    persistent swarm with only that bit set in `paradigm_mask`, seed a
    tiny Galaxy fixture per paradigm, and assert that
    `lane_outputs[0].halt_flag == 1` within ≤ 64 ticks and the
    paradigm-specific `result_handle` is nonzero.
  - The four Batch 3 tests
    (`test_batch3_superposition_rewrite.py`,
    `test_batch3_biabduction.py`,
    `test_batch3_subjective_logic.py`,
    `test_batch3_rete.py`)
    must stay green — they exercise the `__global__` surfaces
    untouched by S22.

---

## 7. S23 — Real Halting Contract From Lane Output

**Focus**

- Replace the synthetic `local_halt` in `k3d_swarm_persistent.cu`
  (the current `(cross_ref | local_summary | 1u) != 0u` placeholder)
  with the real `ReasoningLaneOutput::halt_flag` written by the
  dispatched tick entry point.

**Implementation shape**

- After the dispatch helper returns, each active lane (lane `0` of its
  warp only, to preserve the existing 1-vote-per-warp pattern) feeds
  `lane_outputs[phys_lane_id].halt_flag` into `halting_gate_step`.
- Inactive lanes (`phys_lane_id >= n_active`) still contribute
  `halt_flag = 1` so the gate converges deterministically.
- `lane_perf_write` continues to record the same `LanePerf` record
  (cycles, entropy, belief_delta, specialist id). Belief delta now
  reads `lane_outputs[phys_lane_id].belief_q15 / 32768.0f` instead of
  the synthetic `(local_summary & 0xFFFFu) + 1` formula.

**Rules**

- No change to `halting_gate_step` signature. Only its `local_halt`
  argument source changes.
- Existing `SwarmKernelControl::halting_counter` and `halt_epoch`
  semantics stay identical. Batch 2 `halting_gate` tests remain the
  authoritative gate.
- The `lane_summary_mix` helper is deleted.

**Tests**

- `tests/test_batch5_halting_contract.py`:
  - Seed a paradigm fixture that halts on tick 1, another that halts
    on tick 5, a third that halts on tick 16. In each case verify the
    persistent swarm transitions to `K3D_SWARM_FLAG_COMPLETE` within
    the correct tick count and that `halt_epoch` advances by exactly
    one per halt.
  - Regression: `tests/test_batch2_halting_gate.py` (existing) stays
    green.

---

## 8. S24 — End-to-End Reasoning Smoke Test

**Focus**

- Land **one** end-to-end smoke test that proves the full wire works
  for at least one benchmark-shaped query.

**Test shape**

- `tests/test_batch5_end_to_end_reasoning_smoke.py`:
  1. Use the already-green Batch 4 CBR fixture from
     `tests/test_batch4_context_cbr.py` as seed.
  2. Build a Galaxy atlas with exactly one relevant case.
  3. Set `paradigm_mask = (1u << REASONING_SLOT_CBR)`, `n_floor = 4`,
     `n_hard_max = 16`.
  4. Launch `k3d_swarm_sovereign` for up to 64 ticks.
  5. Assert: swarm halts, `lane_outputs[0].result_handle` points at
     the seeded case, `lane_outputs[0].belief_q15 >= 16384` (≥ 0.5
     q15), `halt_epoch == 1`.
  6. Assert: **no Python reasoning logic** runs in the path. The test
     must grep the imports of the module it exercises for
     `numpy|cupy|scipy|sympy|re` as a sovereignty sanity check.

**Rules**

- Exactly one smoke test in this batch. Broader benchmark expansion
  stays out of scope. The intent is "end-to-end wire is live," not
  "benchmarks are retuned."

---

## 9. Tests And Gates

Every slice ships with:

- sovereignty grep on modified surfaces
- focused `pytest` additions for the slice
- `git diff --check`
- targeted CUDA/PTX compile checks for any new headers or
  `__device__` entry points
- no new Python logic in the reasoning path

Batch-level gates:

- `tests/test_batch5_reasoning_tick_io_abi.py`
- `tests/test_batch5_paradigm_assignment.py`
- `tests/test_batch5_cbr_first_light.py`
- `tests/test_batch5_reasoning_dispatch_switch.py`
- `tests/test_batch5_halting_contract.py`
- `tests/test_batch5_end_to_end_reasoning_smoke.py`

Non-regression required (existing green stays green):

- `tests/test_batch2_*.py`
- `tests/test_batch3_*.py`
- `tests/test_batch4_*.py`

Notes:

- **No score-regression gate** in this batch. The wire may legitimately
  move benchmark numbers because reasoning kernels are now actually
  dispatched. Broader benchmark sweeps are tracked separately.
- A Claude review checkpoint is optional after S22 or after S24, but
  it is not a blocker to landing the spec or beginning implementation.

---

## 10. Explicit Defers

- Wiring ALPCHAIN, resolution, tableaux, DPLL, subsume, ctx_switch,
  order, halt, unify into the persistent swarm (same pattern, later
  batch).
- Frame Galaxy / Hypothesis Galaxy population.
- KBO precedence star population beyond the minimal Batch 3 seed.
- Full CTX_SWITCH integration against real microtheory Galaxy data.
- Ethics ontology ingestion expansion.
- Broad benchmark retuning and score-regression gating.
- Any Python-side paradigm selection or scoring.
- `knowledgeverse.py` 4000→200 line reduction (Phase D, separate
  workstream).
- Daemon migration (Phase C, separate workstream).

---

## 11. Cleanup Reconciliation Note (post-Kimi 2026-04-13)

Batch 5 lands *after* the Kimi-fabrication cleanup documented in
`TEMP/CLAUDE_KIMI_VERIFICATION_AND_CLEANUP_DIRECTIVE_2026-04-13.md`.
As of this spec:

- `rpn_opcodes.py` reasoning block is authoritative and binds to the
  master plan (34 reasoning opcode constants, 0 fabricated leaks).
- The orphan `knowledge3d/cranium/galaxy/star_schema.h`,
  `docs/GALAXY_SCHEMA_EXTENSIONS.md`, and repo-root
  `test_opcode_changes.py` have been deleted.
- The Qdrant `k3d_specifications` collection was not corrupted: its
  ingest scope is `docs/vocabulary/*.md` only
  (`scripts/ingest_specs_to_qdrant.py` line 31), and Kimi's fabricated
  doc sat in `docs/` root, outside scope.
- The canonical `context_id : u32` / `ethical_trit : int8` star schema
  from Batch 1 remains the only truth. Batch 5 `ReasoningTickIO`
  explicitly uses the `int8_t` encoding from that schema.

---

## 12. Handoff Checklist

- Batch 5 is runtime-first and wiring-only. No new kernels, no new
  opcodes, no new stars.
- All new surfaces stay behind `K3D_REASONING_OPCODES_V1`.
- No slice introduces Python orchestration or hot-path fallback logic.
- Existing Batch 1-4 `__global__` launch surfaces and their tests stay
  green.
- The `SwarmTickControl` ABI stays 32 bytes; `_rsv` is renamed to
  `paradigm_mask` in place.
- `ReasoningTickIO` is 32 bytes, `ReasoningLaneOutput` is 64 bytes,
  both with preserved `static_assert`s.
- Dispatch helper is a flat `switch`, no function pointer tables.
- Smoke test proves the full path: Python boot → persistent swarm →
  real reasoning tick → halt → readable output handle.
- Sovereignty grep (`numpy|cupy|scipy|sympy|re`) clean on all modified
  surfaces.
