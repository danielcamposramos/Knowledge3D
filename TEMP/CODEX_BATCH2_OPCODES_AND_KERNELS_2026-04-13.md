# Codex Batch 2 — Opcodes + Kernels (Second Wave)

**Date:** 2026-04-13
**Parent spec:** `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`
**Predecessor:** `TEMP/CODEX_BATCH1_OPCODES_AND_KERNELS_2026-04-13.md` (green)
**Role:** Codex implements. Claude wrote this spec.
**Scope guarantee:** Each slice is **independently landable**. No big-bang.

---

## 0. Where We Stand After Batch 1

Batch 1 landed green:

- Opcode registry reserved 0xA0–0xF1 (docs).
- Star schema: `context_id : u32` + `ethical_trit : i8`, 408-byte record,
  device-side stride now single-sourced in `device_functions.cuh` with a
  CI grep guard.
- Persistent cooperative kernel `k3d_swarm_sovereign` shell with virtual
  lane masking, `d_n_active` **clamped to 9**.
- `lane_perf_ring.cu` writer + 1 Mi ring; **no consumer yet**.
- First opcode tranche: `CTX_SWITCH` (0xB6), `TUNIFY` (0xC0),
  `TSPLIT/TCLOSE/TEXPAND` (0xD0–0xD2),
  `ABDUCE/EXPLAIN/SUSPECT` (0xA0–0xA2).
- Ethical gate plumbed in `gre_defeasible_resolver` stage 3.
- `K3D_REASONING_OPCODES_V1` compile-time flag is the rollback handle.

**Batch 2 goal:** convert that foundation into a **working N-scalable
swarm with sovereign sleep-time calibration**, and fill the ATP /
abductive opcode families that B1 started.

---

## 1. Hard Rules (unchanged, restated for Codex)

- **PTX + Galaxy + RPN + TRM only** in hot path. No numpy / cupy / scipy
  / sympy / Python regex in any hot-path or sleep-path code. Tests only.
- **Sleep is sovereign too.** The lane-perf consumer in Slice S9 is a
  PTX kernel, not Python. (Ref: `feedback_no_fallbacks_ever_including_sleeptime`.)
- **Persistent cooperative kernel**, no Dynamic Parallelism, no runtime
  malloc.
- **`N = 9` floor stays.** Slice S7 raises the ceiling; the floor never
  moves below 9 except under Tier-3 degradation.
- **No benchmark score-regression gate.** Daniel: "Those 10/10 and 20/20
  were achieved with python orchestration, so we're still to see those
  scores with a sovereign substrate." Benchmarks run as normal activity;
  numbers are expected to change (usually upward) as the sovereign path
  grows. Don't block a slice on score parity.
- Sovereignty grep and unit tests are still hard gates per slice.

---

## 2. Slice Ordering

1. **S7 — Dynamic N selector** (unlock the clamp; wire §5.3 formula).
2. **S8 — Halting opcodes** `HALT_SET` (0xF0) + `HALT_SYNC` (0xF1).
3. **S9 — Sleep-time PTX consumer** for the lane perf ring (writes the
   `swarm_perf_calibration` star consumed by S7).
4. **S10 — Resolution family** `TRESOLVE` (0xC1), `TORDER` (0xC2),
   `TSUBSUME` (0xC3).
5. **S11 — Abductive completion** `ABDUCE_HALT` (0xA3), `SCUNION` (0xA4),
   `ICHECK` (0xA5), `ABDRES` (0xA6), `ABDNEG` (0xA7).
6. **S12 — DPLL/SAT support** `TBCP` (0xD3), `TLEARNT` (0xD4).

Dependencies:

- S7 → can ship on B1 foundation alone, but its calibration feedback
  loop is only closed once S9 lands. Order matters: ship S7 with a
  synthetic `swarm_perf_calibration` star (bootstrap values), then let
  S9 start updating it.
- S8 → depends on S7 only to the extent that HALT_SYNC reads `n_active`.
  Can land in parallel with S7 behind the same flag.
- S9 → independent of S7/S8. Pure sleep-time kernel, consumes the B1
  ring.
- S10/S11/S12 → depend on B1's `TUNIFY` (S10, S11) and `TSPLIT/TCLOSE`
  (S12). All behind `K3D_REASONING_OPCODES_V1`.

All new kernels gate-compile behind `K3D_REASONING_OPCODES_V1`. Rollback
is `-UK3D_REASONING_OPCODES_V1` + a test-suite rerun.

---

## 3. S7 — Dynamic N Selector

**Goal:** replace the B1 clamp `d_n_active = 9` with the §5.3 formula,
recomputed at every tick boundary (between `grid.sync` calls). `N_floor`
stays at 9. `N_hard_max` stays at 1024.

### 3.1 Files to create / modify

- `knowledge3d/cranium/cuda/n_selector.cu`        (new, device-side)
- `knowledge3d/cranium/cuda/k3d_swarm_persistent.cu` (modify — call the
  selector at tick boundary)
- `knowledge3d/cranium/bridges/n_chain_swarm_bridge.py` (modify — host
  publishes free-VRAM + deadline via host-mapped struct)
- `knowledge3d/cranium/cuda/swarm_perf_calibration_reader.cuh` (new,
  header — reads the sleep-time calibration star that S9 populates)

### 3.2 Host → device control struct (host-mapped pinned)

```c
struct __align__(16) SwarmTickControl {
    uint32_t vram_free_mib;         // driver query, updated by host each tick
    uint32_t t_remaining_us;        // deadline budget this tick
    uint32_t n_cand_frustum;        // frustum-culled candidate count
    uint32_t h_belief_q10;          // belief entropy, bits × 1024 (u32)
    uint32_t n_floor;               // always 9 unless tier-3 sets otherwise
    uint32_t n_hard_max;            // always 1024 in this batch
    uint32_t sleep_calibration_n_hint; // last sleep-time utility maximum; 0 = bootstrap
    uint32_t _rsv;
};
```

Allocated host-side via `cudaHostAlloc(..., cudaHostAllocMapped)`. The
bridge writes to it before each `grid.sync` release. Device reads via
the mapped pointer.

### 3.3 Selector (device)

```cuda
__device__ __forceinline__ uint32_t n_selector(
    const SwarmTickControl& c,
    uint32_t                lane_perf_hint)  // from S9 calibration
{
    // Per-lane budget is a compile-time constant (parent spec §5.2 = 14.68 MiB).
    // Use 15 MiB for conservative integer math; keep floor at 9.
    const uint32_t PER_LANE_MIB = 15u;

    uint32_t n_vram    = (c.vram_free_mib * 90u / 100u) / PER_LANE_MIB;

    // entropy_boost = 1 + H / H_max  ∈ [1.0, 2.0], scaled ×1024 in q10
    uint32_t boost_q10 = 1024u + min(c.h_belief_q10, 1024u);
    uint32_t n_entropy = (uint32_t)(((uint64_t)n_vram * boost_q10) >> 10);

    // T_per_lane ≈ 48 μs (parent spec §5.3)
    uint32_t n_deadline = c.t_remaining_us / 48u;

    uint32_t n = min(n_entropy, min(n_deadline, c.n_cand_frustum));

    // Sleep-time calibration hint: if S9 has recorded a utility peak,
    // clamp proposal toward it (±25%).
    if (lane_perf_hint != 0u) {
        uint32_t lo = lane_perf_hint * 3u / 4u;
        uint32_t hi = lane_perf_hint * 5u / 4u;
        if (n < lo) n = lo;
        if (n > hi) n = hi;
    }

    if (n < c.n_floor)    n = c.n_floor;
    if (n > c.n_hard_max) n = c.n_hard_max;
    return n;
}
```

**Invariants:**

- N is **fixed intra-tick** (`bar.sync` count cannot change mid-kernel).
- N is **re-evaluated only at tick boundary**, after `grid.sync`, by
  lane 0 of block 0, which publishes to `d_n_active`.
- No lane ever reads `vram_free_mib` directly; only the selector.

### 3.4 Tick-boundary glue

In `k3d_swarm_persistent.cu`, inside the persistent outer loop:

```cuda
grid.sync();                       // end of tick
if (phys_lane_id == 0 && ctaid.x == 0) {
    uint32_t hint = swarm_perf_calibration_load(/*galaxy*/ galaxy_atlas);
    *d_n_active  = n_selector(*d_tick_ctrl, hint);
    *g_halting_counter = 0u;       // reset for next tick
}
grid.sync();                       // publish
```

Keep the existing virtual-lane mask logic: `active = phys_lane_id < *d_n_active`.

### 3.5 Tier-3 degradation hook (PTX-side only)

On detected VRAM page fault or on `trap 0xDEAD_N` signal:

- Lane 0 writes `n_floor = 9` (already the default) and
  `n_hard_max = (current * 80%)` into `SwarmTickControl`.
- Sleep-time consumer (S9) records the event.
- **No host fallback in this batch.** Tier-3 "host recovery" is a later
  wave; here we only implement the telemetry + ceiling reduction on
  GPU.

### 3.6 S7 exit tests

- `tests/test_n_selector_formula.py` — CPU unit test against the math,
  no GPU needed. Sweep VRAM/deadline/entropy/hint combinations, assert
  the clamp and the calibration band.
- `tests/test_n_selector_gpu.py` (CUDA-gated) — launches the persistent
  kernel, programmatically varies `SwarmTickControl`, asserts
  `d_n_active` tracks the formula tick-by-tick.
- Sovereignty grep on touched files.
- Bench sanity: run ARC/Math through the new bridge with the flag ON;
  record scores, do not block on them.

---

## 4. S8 — Halting Opcodes

**Goal:** expose the halting gate to RPN programs so the TRM game loop
can halt lanes on its own, without Python polling.

### 4.1 File

- `knowledge3d/cranium/cuda/rpn_halt.cu` (new, thin dispatcher)

### 4.2 Semantics

```
0xF0  HALT_SET
    Stack: [..., pred:u32] → [...]
    Effect: writes pred (0 or 1) to this lane's `local_halt` slot in
            shared memory. Pred > 0 counts as halt.

0xF1  HALT_SYNC
    Stack: [...] → [..., halted_now:u32]
    Effect: __syncthreads(), then calls halting_gate_step() exactly
            once (idempotent via a tick-scoped sentinel), returns
            the value of `halting_flag` after the reduction.
```

`HALT_SET` is a pure lane-local write — no atomics, no sync. Cheap to
call mid-program (e.g., inside a loop).

`HALT_SYNC` is the synchronisation point. It is idempotent within a
tick: second calls in the same tick are no-ops (guarded by a
`g_tick_epoch` counter).

### 4.3 Integration with the existing halting gate

The existing `halting_gate_step(...)` from B1 stays untouched. `HALT_SYNC`
is a wrapper:

```cuda
__device__ uint32_t op_halt_sync(LaneState& ls, SwarmGlobals& g) {
    __syncthreads();
    uint32_t epoch_seen = ls.halt_epoch_seen;
    if (epoch_seen != g.tick_epoch) {
        halting_gate_step(
            *g.d_n_active,
            ls.phys_lane_id,
            ls.local_halt != 0,
            g.halting_flag,
            &g.halting_counter);
        ls.halt_epoch_seen = g.tick_epoch;
    }
    return *g.halting_flag;
}
```

### 4.4 Dispatcher hook

In the RPN dispatcher (same `switch(opcode)` touched by B1 S5), add:

```cuda
case 0xF0: op_halt_set(stack);                 break;
case 0xF1: out = op_halt_sync(lane_state, g);  break;
```

Both behind `K3D_REASONING_OPCODES_V1`.

### 4.5 S8 exit tests

- `tests/test_op_halt.py` (CUDA-gated):
  - 9 lanes, each runs an RPN program `[HALT_SET(1) HALT_SYNC]`.
    Assert `halting_flag` flips to 1 and all lanes observe it.
  - 9 lanes, 8 set halt / 1 does not. Assert flag stays 0.
  - Idempotency: call `HALT_SYNC` twice in the same tick, assert
    `halting_counter` increments only once.
- Sovereignty grep.

---

## 5. S9 — Sleep-Time PTX Perf-Ring Consumer

**Goal:** consume the `LanePerf` ring from B1 S4 inside a sleep-time PTX
kernel (pattern: see `sleep_time_micro.cu`), and write the
`swarm_perf_calibration` meaning-star that S7's selector reads.

**No Python.** The B1 spec mentioned a Python analytics stub; it was
never landed. Batch 2 originates the consumer as PTX.

### 5.1 Files

- `knowledge3d/cranium/cuda/sleep_perf_consumer.cu` (new)
- `knowledge3d/cranium/bridges/sleep_perf_consumer_bridge.py` (new,
  bootstrap only — ingests the star schema, launches the kernel, hands
  off)
- `knowledge3d/ingestion/star_crafter.py` (add seed star
  `swarm_perf_calibration` with zeroed metrics so S7 has something to
  read on first boot)

### 5.2 Kernel signature

```cuda
extern "C" __global__ void k3d_sleep_perf_consume(
    const LanePerf* __restrict__ ring,
    uint32_t                     ring_size,      // power-of-two
    uint32_t                     ring_head,      // latest head from bridge
    uint8_t*       __restrict__  galaxy_atlas,   // read-write, sleep-only
    uint32_t                     calib_star_off  // byte offset in atlas
);
```

Launch: 1 block × 1024 threads, cooperative not required (single block).
Sleep-time only, so launch cost is irrelevant and we can read the full
ring in one sweep.

### 5.3 Aggregation (on-device, sovereign)

Per-block shared memory:

```cuda
__shared__ uint64_t  sum_cycles[16];   // 16 N-buckets (log-spaced: 9, 16, 32, 64, ..., 1024)
__shared__ uint64_t  sum_delta_q20[16];
__shared__ uint32_t  sample_count[16];
```

Each thread strides the ring, classifies `sample.n_active` into the 16
buckets, and accumulates `cycles_consumed` and `belief_delta * 2^20`
(kept as integer q20 to stay off of floats). Block-level reduction via
`__shfl_down_sync`.

Utility metric **on-device, integer only**:

```
utility_q20[b] = sum_delta_q20[b] / sum_cycles[b]    // u64/u64 integer div
```

Find argmax bucket `b*` with at least `MIN_SAMPLES = 256`. Report the
midpoint N of that bucket as `n_hint`. If no bucket qualifies, write
`n_hint = 0` (S7 treats this as "no hint, use pure formula").

### 5.4 Calibration star payload (in-place update)

Write into the star record at `calib_star_off`:

```c
struct SwarmPerfCalibration {
    uint32_t n_hint;                // argmax bucket midpoint, or 0
    uint32_t sample_count_total;    // for freshness
    uint32_t last_tick_epoch;       // when this was last updated
    uint32_t utility_peak_q20;      // the winning utility
    uint32_t bucket_samples[16];    // for diagnostics
    uint32_t bucket_utility_q20[16];
    uint32_t _pad[4];               // keep struct ≤ 128 B for alignment
};
```

**This is the sole sleep-time write to a Galaxy star**, consistent with
the parent spec §6 checklist line "Galaxy writes only at ingestion or
sleep-time".

### 5.5 Bridge

`sleep_perf_consumer_bridge.py` is pure I/O:

1. Queries device ring head via host-mapped pointer.
2. Launches the kernel once per sleep cycle.
3. Returns success. No numerical work in Python. No numpy.

### 5.6 S9 exit tests

- `tests/test_sleep_perf_consumer.py` (CUDA-gated):
  - Pre-fill the ring with 4096 synthetic samples crafted so
    `utility[N=64]` is clearly the winner.
  - Launch kernel, read back the calibration star.
  - Assert `n_hint ∈ [48, 80]` (bucket midpoint tolerance).
- `tests/test_sleep_perf_consumer_empty.py`:
  - Empty ring → `n_hint == 0`, no crash.
- Sovereignty grep.

---

## 6. S10 — Resolution Family (`TRESOLVE`, `TORDER`, `TSUBSUME`)

**Goal:** complete the first-order logic path seeded by B1's `TUNIFY`.
Reuses the same per-lane substitution arena.

### 6.1 Files

- `knowledge3d/cranium/cuda/rpn_resolve.cu`    (new)
- `knowledge3d/cranium/cuda/rpn_order.cu`       (new)
- `knowledge3d/cranium/cuda/rpn_subsume.cu`     (new)

### 6.2 Semantics

```
0xC1  TRESOLVE
    Stack: [..., clause_a_ptr, clause_b_ptr] → [..., resolvent_ptr]
    Effect: binary resolution.  Picks a literal L in A and ¬L in B,
            unifies via op_tunify (0xC0), emits the resolvent.
            Returns 0 if no resolvent exists.

0xC2  TORDER
    Stack: [..., term_ptr] → [..., kbo_weight:u32]
    Effect: computes KBO (Knuth-Bendix) weight using the canonical
            symbol precedence stored in the Reasoning Galaxy.
            Precedence star is ingested once; TORDER only reads.

0xC3  TSUBSUME
    Stack: [..., clause_ptr_candidate, clause_ptr_existing] → [..., u32]
    Effect: returns 1 if `existing` subsumes `candidate`
            (∃σ. candidate_literals ⊇ σ(existing_literals)),
            else 0.  Uses op_tunify for each literal pair.
```

### 6.3 Implementation notes

- **Clause layout in Galaxy**: each clause is a meaning-star with a
  short RPN body `[literals...]`. Codex: confirm the clause schema with
  the star_crafter team before coding; if no clause schema exists yet,
  add one as an ingestion-time update.
- **KBO precedence**: ingest once as a star
  `reasoning_kbo_precedence` with a u8 rank per symbol. TORDER reads it
  via one coalesced load per term node.
- **Subst arena reuse**: TRESOLVE pushes a new arena frame per
  resolvent attempt; arena pointer is a lane-local stack.
- **No heap**: if a clause exceeds 64 literals, return 0 and emit
  `STAR_FILTERED`. 64 is the same bit-packed max used by B1 tableaux.

### 6.4 S10 exit tests

- `tests/test_op_resolve.py`:
  - Resolve `{P(X), Q(X)}` with `{¬P(a), R(a)}` → `{Q(a), R(a)}`.
  - No resolution when literals don't complement.
- `tests/test_op_order.py`:
  - Given precedence `f > g > a`, assert `TORDER(f(a))` >
    `TORDER(g(a))`.
- `tests/test_op_subsume.py`:
  - `{P(X)}` subsumes `{P(a), Q(a)}`.
  - `{P(X), Q(X)}` does not subsume `{P(a)}`.

All sovereign. All behind the compile flag.

---

## 7. S11 — Abductive Completion

**Goal:** finish the Peirce abductive path started by B1's ABDUCE /
EXPLAIN / SUSPECT. Unblocks MMLU hypothesis exploration once Galaxy
coverage catches up.

### 7.1 Files

- `knowledge3d/cranium/cuda/rpn_abduce_ext.cu`   (new)

### 7.2 Semantics

```
0xA3  ABDUCE_HALT
    Stack: [..., hyp_id, simplicity_score:u32] → [..., halted:u32]
    Effect: if simplicity_score > THRESHOLD (from Reasoning Galaxy
            star `abduce_halt_threshold`), sets this lane's local_halt
            and pushes 1; else pushes 0. Does NOT call HALT_SYNC —
            that is the program's job.

0xA4  SCUNION
    Stack: [..., mask_a, mask_b] → [..., mask_union]
    Effect: warp-level popcount-aware set union. Uses __popc and
            __shfl_xor_sync for O(log 32) merge.  Used by set-cover
            greedy abduction.

0xA5  ICHECK
    Stack: [..., hyp_id, constraint_star_id] → [..., ok:u32]
    Effect: ternary AND over the constraint's RPN program applied to
            the hypothesis (uses existing TAND opcode internally).
            Returns 1 if the constraint holds, 0 otherwise,
            STAR_FILTERED on missing data.

0xA6  ABDRES
    Stack: [..., goal_clause_ptr, kb_clause_ptr] → [..., assumptions_ptr]
    Effect: abductive resolution.  Like TRESOLVE, but any literal of
            the goal that fails to resolve is collected as an
            *assumption* (a new star written to a per-lane scratch
            assumption pool).  The pool is committed to Galaxy only
            if the program subsequently calls ABDUCE_HALT with a
            successful score.

0xA7  ABDNEG
    Stack: [..., goal_lit_ptr, budget:u32] → [..., finitely_failed:u32]
    Effect: negation-as-failure. Attempts exhaustive resolution with
            `budget` steps. Returns 1 if the search space is finitely
            exhausted without success, 0 otherwise.
```

### 7.3 Dependencies

- `ABDRES` depends on S10 `TRESOLVE`.
- `ABDNEG` depends on S10 `TRESOLVE` and on S8 `HALT_SET` (uses it
  internally to abandon when budget is exhausted).
- `SCUNION` is self-contained.
- `ICHECK` uses the existing ternary opcodes (TAND/TCOMP) from B1 S0
  infrastructure.
- `ABDUCE_HALT` depends on the ingestion-time `abduce_halt_threshold`
  star; ship a default of `THRESHOLD = 0x80000000u` (midpoint) until
  sleep-time calibration updates it.

### 7.4 S11 exit tests

- `tests/test_op_abduce_halt.py`: threshold above score → no halt;
  below → halt set.
- `tests/test_op_scunion.py`: merges `0xAAAA_AAAA | 0x5555_5555 =
  0xFFFF_FFFF`, warp-level.
- `tests/test_op_icheck.py`: constraint `∀x. P(x) → Q(x)` against two
  hypotheses.
- `tests/test_op_abdres.py`: resolves partially, collects assumption,
  asserts pool grew by exactly 1 star.
- `tests/test_op_abdneg.py`: finite search exhausts within budget → 1;
  infinite → 0.

---

## 8. S12 — DPLL/SAT Support (`TBCP`, `TLEARNT`)

**Goal:** extend the tableaux foundation from B1 into a CDCL-capable
micro-solver. Enables Reality-Galaxy constraint satisfaction and MMLU
logic puzzles without a host solver.

### 8.1 Files

- `knowledge3d/cranium/cuda/rpn_dpll.cu`  (new)

### 8.2 Semantics

```
0xD3  TBCP
    Stack: [..., clause_set_ptr, trail_ptr] → [..., conflict_clause_ptr]
    Effect: two-watched-literal Boolean constraint propagation.
            Propagates unit clauses until fixpoint.  Returns the
            conflict clause pointer on conflict, or 0 on fixpoint.
            Runs in a single warp with warp-level watch updates.

0xD4  TLEARNT
    Stack: [..., conflict_clause_ptr, trail_ptr] → [..., learnt_ptr]
    Effect: first-UIP conflict clause learning. Walks the
            implication graph backward, records the 1-UIP cut, and
            commits the learnt clause to a per-lane clause pool.
            Minimisation: self-subsumption only (cheap), no full
            resolution chain.
```

### 8.3 Constraints

- Max clause size: 64 literals (2×u64 bit-packed, same as B1
  tableaux).
- Max trail depth per lane: 256 (bounded by shared mem).
- Conflict clause pool per lane: 128 clauses × 16 B = 2 KiB, living in
  the lane's existing 3.5 KiB TPACK stack region.
- **No garbage collection** for learnt clauses in this batch. When the
  pool fills, TLEARNT returns 0 (drop the clause). Sleep-time
  consolidator handles actual LBD-based deletion in a later wave.

### 8.4 S12 exit tests

- `tests/test_op_tbcp.py`:
  - `{¬p ∨ q}, trail = {p}` → propagates `q`, no conflict.
  - `{p}, {¬p}` → returns the first clause as the conflict.
- `tests/test_op_tlearnt.py`:
  - Simple 3-variable conflict, assert 1-UIP clause is
    `{¬x ∨ ¬y}` (or equivalent).
- Pool-full test: 128 learnt → 129th returns 0, no crash.

---

## 9. Non-Regression + CI Gates

Every slice must ship with:

- Unit tests for its new surface.
- Sovereignty grep `rg -n 'numpy|cupy|scipy|sympy|re\.' <touched hot+sleep files>`
  returns **zero** hits. `knowledge3d/sleep/` is included in this grep
  starting from S9 (sleep is sovereign).
- Opcode-budget grep: reservation count in
  `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` is monotone
  non-decreasing.
- Star-stride guard (B1's grep) continues to pass.
- `K3D_REASONING_OPCODES_V1` off → full test suite still green.
- `K3D_REASONING_OPCODES_V1` on → new tests green.

**No score-regression gate in this batch.** Benchmark numbers are
expected to move as the sovereign substrate grows; Daniel's directive
is to stop blocking on the Python-baseline scores.

---

## 10. What's NOT in Batch 2 (deferred)

- **Bi-abduction** (`BIDUCE` 0xB1, `FRAME` 0xB2) — needs Frame Galaxy
  stubs; earliest Batch 3.
- **Superposition** (`TSUPERPOS` 0xC4, `TREWRITE` 0xC5) — builds on
  S10; Batch 3.
- **Rete** (`RETE_ALPHA_TEST` 0xE0, `RETE_BETA_JOIN` 0xE1, `AGENDA_INSERT`
  0xE2) — independent, cheap, land whenever drools-style rule coverage
  is needed.
- **Subjective / belief** (`EBELIEF` 0xB0) — needs the belief Galaxy to
  be populated.
- **Euler-path / DL saturation** (`EULER_COMPLETE` 0xB3, `DL_SATURATE`
  0xB4, `BLOCKING_CHECK` 0xB5) — Batch 3 or 4.
- **ALP backward-chain** (`ALPCHAIN` 0xB7) — Batch 3.
- **Full philosophy ontology** for `ethical_trit`. B1 ships the seed
  allowlist; full ontology is ingestion-only work, does not block this
  batch.
- **Galaxy-wide HS catalogue ingestion** (parent spec §9.2).
- **Daemon / always-on migration** (Phase C).
- **`loader.py` cupy/numpy cleanup** — separate sovereignty wave.

These follow the parent spec's §7 order in Batch 3.

---

## 11. Hand-off Checklist

Codex, before starting:

- [ ] Confirm Batch 1 re-greened: `pytest -q
      tests/test_star_materializer_bridge.py
      tests/test_star_schema_context_ethical.py` → 50 passed;
      `K3D_PYTEST_PROBE_CUDA=1 pytest -q
      tests/test_n_chain_persistent_launch.py` → 2 passed.
- [ ] Read parent spec `CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`
      §5 (N-swarm), §7 (landing order), §8 (Daniel's decisions).
- [ ] Start with S7 (dynamic N) — the biggest unlock; ship it with a
      zeroed `swarm_perf_calibration` star so the calibration-hint code
      path runs end-to-end on day 1.
- [ ] S9 can land in parallel with S7; they meet at the calibration
      star, not in code.
- [ ] S8 can land in parallel with S7/S9; it only touches the
      dispatcher and the existing gate.
- [ ] S10 → S11 → S12 is a natural sequence but each is independently
      testable. If S10 slips, skip to S12 (it only depends on B1
      tableaux).
- [ ] Claude review checkpoint **after S7 + S9** land together — that
      is the moment the dynamic-N loop closes and we want to inspect
      the first real calibration data.
- [ ] No `--no-verify`. No fallbacks. We fix or we fix.

**End of Batch 2 spec.**
