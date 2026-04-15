# Codex Batch 1 — Opcodes + Kernels (First Wave)

**Date:** 2026-04-13
**Parent spec:** `TEMP/CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`
**Role:** Codex implements. Claude wrote this spec.
**Scope guarantee:** This batch is **minimum-to-unblock**. Later waves
follow the parent spec's §7 ordering.

---

## 0. Hard Rules (non-negotiable)

- **PTX + Galaxy + RPN + TRM only** in hot path. No numpy / cupy / scipy
  / sympy / Python regex anywhere Codex touches here except in tests.
- **Persistent cooperative kernel**, not CUDA Dynamic Parallelism.
- `N = 9` is the frozen floor for this batch. Dynamic-N selection is
  wired in but `d_n_active` is clamped to 9 until a later wave.
- **Non-regression**: ARC 10/10, Math 20/20 must stay green after every
  slice lands.
- Benchmarks, linting, sovereignty grep are Codex's job per the standing
  directive ("Claude = architecture, Codex = implementation").

---

## 1. Slice Ordering (land in this order, each ships independently)

1. **S1 — Opcode registry reservation** (docs only).
2. **S2 — Star schema extension** (`context_id`, `ethical_trit`).
3. **S3 — N-chain persistent swarm kernel shell** (with N=9 clamp).
4. **S4 — Shadow-copy perf ring** (`LanePerf` writer + reader).
5. **S5 — First opcode tranche** (four kernel families).
6. **S6 — Ethical gate extension** to `gre_defeasible_resolver`.

S1/S2 are prerequisite and touch only docs + ingestion; S3–S6 land new
PTX. S5 depends on S2 (for `CTX_SWITCH`) and on S3 (swarm dispatches
opcodes). S6 depends on S2.

---

## 2. S1 — Opcode Registry Reservation

**File to modify:** `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`

Add a new section **"Extended Tier — Reasoning-Paradigm Block
(0xA0–0xF1)"** with the following reservations (Codex: keep existing
table style, do not reorder existing reservations):

```
# Abductive family (0xA0-0xA7)
0xA0  ABDUCE         pop observation, push hypothesis candidate
0xA1  EXPLAIN        verify effects(A) ⊇ C via TCOMP on ternary mask
0xA2  SUSPECT        rank by simplicity (TQUANT simplicity score)
0xA3  ABDUCE_HALT    push result to halting gate if score > threshold
0xA4  SCUNION        warp-popcount greedy set-cover union
0xA5  ICHECK         integrity-constraint validation (ternary AND)
0xA6  ABDRES         abductive resolution (unify + collect assumptions)
0xA7  ABDNEG         negation-as-failure (finite failure)

# Subjective / frame (0xB0-0xB7)
0xB0  EBELIEF        evidence-to-belief (Bayes inversion on TQUANT)
0xB1  BIDUCE         bi-abduction frame inference
0xB2  FRAME          separation-logic frame extraction
0xB3  EULER_COMPLETE transitive property-chain closure
0xB4  DL_SATURATE    description-logic tableau saturation
0xB5  BLOCKING_CHECK binary blocking on node × predecessor
0xB6  CTX_SWITCH     filter Galaxy reads by star.context_id
0xB7  ALPCHAIN       abductive-LP backward chain step

# Deductive / ATP (0xC0-0xC5)
0xC0  TUNIFY         Robinson unification with occur-check
0xC1  TRESOLVE       binary resolution → resolvent clause
0xC2  TORDER         KBO/LPO term ordering
0xC3  TSUBSUME       clause subsumption (forward/backward)
0xC4  TSUPERPOS      superposition critical-pair generation
0xC5  TREWRITE       ordered rewriting from indexed rule set

# Tableaux / SAT (0xD0-0xD4)
0xD0  TSPLIT         α/β branch split
0xD1  TCLOSE         branch closure via complementarity
0xD2  TEXPAND        γ/δ quantifier expansion (Skolemise on-GPU)
0xD3  TBCP           Boolean constraint propagation (two-watch)
0xD4  TLEARNT        conflict-clause learn + minimise

# Rete (0xE0-0xE2)
0xE0  RETE_ALPHA_TEST
0xE1  RETE_BETA_JOIN
0xE2  AGENDA_INSERT

# System (0xF0-0xF1)
0xF0  HALT_SET       write consensus register (lane-local view)
0xF1  HALT_SYNC      global barrier + cross-lane halt check
```

Document semantics brief, stack diagram (`[...] → [...]`), and ternary
packing per opcode. Cross-reference the parent spec §4 as authority.

**S1 exit test:** `rg '0x[A-F][0-9]' docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md | wc -l` includes all 32 new reservations. No collision with existing 0x00–0x9F.

---

## 3. S2 — Star Schema Extension

**Goal:** add two metadata fields to every star record. Default values
are zero so existing data remains valid.

**Files to modify** (ingestion path only — hot path reads these fields
but no migration happens in hot path):

- `knowledge3d/ingestion/canonical_lookup.py`
- `knowledge3d/ingestion/star_crafter.py`
- `knowledge3d/cranium/cuda/star_materializer.cu`
- `knowledge3d/cranium/cuda/star_hash_index.cu` (layout-only; indexed
  fields unchanged)
- `scripts/ingest_canonical_to_qdrant.py` (write defaults)

**New fields on the canonical star record:**

```c
// Append to the existing star struct (VRAM layout, 4-byte aligned)
uint32_t context_id;   // 0 = universal; non-zero = scoped (era/region/microtheory)
int8_t   ethical_trit; // -1 = forbidden, 0 = ok, +1 = defeasible
                       // kept as int8 for straightforward sign; pack into
                       // ternary word only when crossing kernel boundaries
uint8_t  _pad[3];      // alignment
```

**Qdrant payload additions:**

```json
{
  "context_id": 0,
  "ethical_trit": 0
}
```

Add both to the payload schema in `ingest_canonical_to_qdrant.py`.
Create keyword payload indices for `context_id` and `ethical_trit` so
scroll filters are O(1).

**Ingestion policy:**

- Universal stars → `context_id = 0`, `ethical_trit = 0`.
- Philosophy / ethics stars (HS Humanities plan, parent-spec §9.2) →
  ingestion-time rule assigns `ethical_trit` from the star's
  `is_a` chain.  Slice S6 documents exact rules; S2 only needs the field
  populated with the default.
- History / geography / civics / linguistics stars with region/era
  metadata → `context_id = canonical_context_id(region, era)`; helper
  goes in `canonical_lookup.py` using `uuid.uuid5(NAMESPACE_URL, ...)`
  truncated to u32 (same deterministic pattern as `canonical_entry_id`).

**S2 exit tests:**

- `tests/test_star_schema_context_ethical.py` — round-trip ingest +
  scroll by `context_id` and `ethical_trit`.
- Existing ingestion tests still pass.
- Sovereignty grep: no new `numpy|cupy|scipy|sympy|re\.` in touched
  files beyond whatever was there.

---

## 4. S3 — N-Chain Persistent Swarm Kernel Shell

**Goal:** replace the current nine-chain launch with a persistent
cooperative kernel that virtually masks `N` active lanes out of a
`N_hard_max = 1024` physical grid. Clamp `N = 9` in this batch.

### 4.1 New kernel files

- `knowledge3d/cranium/cuda/k3d_swarm_persistent.cu` (entry kernel)
- `knowledge3d/cranium/cuda/halting_gate.cu`

### 4.2 Entry kernel signature

```cuda
extern "C" __global__ void k3d_swarm_sovereign(
    const uint8_t* __restrict__  galaxy_atlas,     // read-only LoRA atlas
    volatile uint32_t* __restrict__ halting_flag, // global, host-mapped
    volatile uint32_t* __restrict__ d_n_active,   // host-mapped, clamped to 9 this batch
    LanePerf*                       perf_ring,    // ring buffer (S4)
    uint32_t                        perf_ring_mask // power-of-two mask
);
```

### 4.3 Launch configuration (one-shot, persistent)

- `gridDim  = {256, 1, 1}`  (256 blocks)
- `blockDim = {128, 1, 1}`  (128 threads = 4 warps; lane = warp)
- Max physical lanes = `256 × 4 = 1024`
- Shared memory per block: `15_360` bytes (4 lanes × ~3.5 KiB stack +
  cross-ref bitmap), bank-interleaved
- Launch via `cuLaunchCooperativeKernel`. Do NOT use Dynamic Parallelism.

### 4.4 Virtual-lane masking

```
phys_lane_id = ctaid.x * 4 + warpid           // 0..1023
n_active     = ld.global.u32 [d_n_active]     // clamped to 9 for now
active       = (phys_lane_id < n_active)
```

Active lanes execute the tick body. Inactive lanes fall straight to
`bar.sync` to keep the barrier count stable.

### 4.5 Cross-reference topology — FULL

Per Daniel's decision (parent spec §8.1): every lane sees every other
lane's summary. Shared-mem layout, bank-interleaved so lane `j`'s
summary is readable by lane `i` without bank conflicts:

```
smem[CROSS_REF_BASE + j * 4] = belief_summary_of_lane(j)
```

No locality gate. LOD and FOV remain the only cuts.

### 4.6 Halting gate (`halting_gate.cu`)

Tree reduction via atomics. Lane writes its local halt predicate (set by
RPN `HLT` / `HALT_SET` opcode) into `g_halting_counter`; lane 0 compares
against `n_active` and toggles `halting_flag`.

```cuda
__device__ __forceinline__ void halting_gate_step(
    uint32_t n_active,
    uint32_t phys_lane_id,
    bool     local_halt,
    volatile uint32_t* halting_flag,
    uint32_t* g_halting_counter)
{
    if (local_halt) atomicAdd(g_halting_counter, 1u);
    __threadfence_system();
    if (phys_lane_id == 0) {
        uint32_t c = *g_halting_counter;
        if (c >= n_active) *halting_flag = 1u;
    }
    __syncthreads();
}
```

Reset `g_halting_counter` to 0 at the start of each tick (lane 0 writes,
barrier, then tick body runs).

### 4.7 Python bridge

- New file: `knowledge3d/cranium/bridges/n_chain_swarm_bridge.py`
- **Do not delete `nine_chain_swarm_bridge.py` yet.** The new bridge
  wraps the persistent kernel and exposes:
  - `launch()` — one-shot cooperative launch
  - `tick(work_packet)` — submit a tick; returns when halting flag set
  - `shutdown()` — set flag to halt the persistent kernel
- The new bridge internally sets `d_n_active = 9` (frozen this batch).

Existing call sites keep using `nine_chain_swarm_bridge` until a later
wave swaps them. This batch only adds the new path; it does not remove
the old.

### 4.8 S3 exit tests

- `tests/test_n_chain_persistent_launch.py` (CUDA-gated, skipped on
  sandbox):
  - Launches persistent kernel, submits 16 no-op ticks, shuts down
    cleanly.
  - Asserts `d_n_active == 9` throughout.
  - Asserts `halting_flag` cycles 0 → 1 per tick.
- Sovereignty grep.
- ARC 10/10 + Math 20/20 still green when routed through old bridge
  (new bridge not yet wired into the hot benchmark path).

---

## 5. S4 — Shadow-Copy Performance Ring

### 5.1 New kernel file

- `knowledge3d/cranium/cuda/lane_perf_ring.cu`

### 5.2 Struct (matches parent spec §5.7)

```cuda
struct __align__(16) LanePerf {
    uint32_t n_active;         // N value this tick
    uint32_t entropy_input;    // belief entropy at tick start (bits × 1e3)
    float    belief_delta;     // L2 norm of belief update
    uint32_t cycles_consumed;  // clock64 delta
    uint8_t  specialist_id;    // active LoRA
    uint8_t  _pad[3];
};
```

Ring size = `1 << 20` entries (1 Mi), aligned to 16 B; total ~16 MiB.
Power-of-two size so the mask is cheap.

### 5.3 Writer (device)

```cuda
__device__ __forceinline__ void lane_perf_write(
    LanePerf*       ring,
    uint32_t*       ring_head,
    uint32_t        ring_mask,
    const LanePerf& sample)
{
    uint32_t idx = atomicInc(ring_head, UINT32_MAX) & ring_mask;
    ring[idx] = sample;
}
```

Called once per active lane at tick end (just before the halting gate
reduction).

### 5.4 Sleep-time reader

- `knowledge3d/sleep/lane_perf_consumer.py` (pure ingestion — can use
  standard Python; runs under sleep-time only, per
  `feedback_no_fallbacks_ever_including_sleeptime` **sleep is sovereign
  too** → if the reader ever needs to touch hot-path data structures it
  must go through the existing sleep PTX kernels; this Python is for
  analytics aggregation writing back into Galaxy metadata).
- Computes moving average `utility[N] = mean(belief_delta / cycles)` over
  the last 1 M samples.
- Writes aggregates into a new Galaxy metadata star
  (`star: swarm_perf_calibration`) at sleep-time.

### 5.5 S4 exit tests

- `tests/test_lane_perf_ring.py` — fills ring with N=9 samples, reads
  back, asserts no overwrite within ring size.
- Sleep-time aggregation unit test on a synthetic ring of 10 k samples.

---

## 6. S5 — First Opcode Tranche (four kernel families)

Each kernel file below is **thin** — a single `__device__` function per
opcode, dispatched from the existing RPN engine's opcode switch. Codex:
land the dispatcher hooks in the existing `rpn_engine` / `rpn_ext` code
path; do not fork the dispatcher.

### 6.1 `rpn_ctx_switch.cu` — `CTX_SWITCH` (0xB6)

**Depends on:** S2 (star schema `context_id`).

```cuda
// Stack: [... , ctx_id] → [...]
// Effect: subsequent Galaxy reads on this lane filter to stars whose
//         context_id == ctx_id OR context_id == 0 (universal).
// Single lane-local state slot in .local mem:
// __local__ uint32_t lane_active_context;

__device__ void op_ctx_switch(RPNStack& s, LaneState& ls) {
    uint32_t ctx = s.pop_u32();
    ls.active_context = ctx;
}
```

Every Galaxy read helper (`galaxy_star_probe`, `reverse_symlink_expand`,
etc.) gains an early-reject:

```cuda
if (ls.active_context != 0 &&
    star.context_id   != 0 &&
    star.context_id   != ls.active_context) {
    return STAR_FILTERED;
}
```

### 6.2 `rpn_unify.cu` — `TUNIFY` (0xC0)

**Depends on:** star `hash_index`; no schema change.

```cuda
// Stack: [..., term_a_ptr, term_b_ptr] → [..., subst_handle]
// subst_handle = 0 on unification failure.

__device__ uint32_t op_tunify(
    RPNStack& s,
    const Term* galaxy_terms,
    SubstArena& subst)
{
    uint32_t b = s.pop_u32();
    uint32_t a = s.pop_u32();
    uint32_t h = robinson_unify(&galaxy_terms[a], &galaxy_terms[b], &subst);
    s.push_u32(h);
    return h;
}
```

Robinson unification with occur-check. Use warp-shuffle for path
compression on the substitution DAG. Subst arena = per-lane scratch,
fixed 4 KiB — overflow sets `h = 0` and emits `STAR_FILTERED`-equivalent.

### 6.3 `rpn_tableaux.cu` — `TSPLIT` / `TCLOSE` / `TEXPAND` (0xD0–0xD2)

Natural fit for the N-chain swarm: each active lane owns a branch.

```cuda
// TSPLIT: [..., alpha_formula, beta_formula] → [..., branch_id_alpha, branch_id_beta]
// TCLOSE: [..., branch_id, lit_a, lit_b] → [..., closed:u32]
//         uses op_tunify internally for complementarity check.
// TEXPAND: [..., branch_id, formula] → [..., expanded_formula_ptr]
//          γ uses __constant__ skolem_counter via atomicAdd.
```

Branch struct is **bit-packed in registers** (max 64 literals per
branch via 2 × u64). No heap allocation inside branches.

### 6.4 `rpn_abduce.cu` — `ABDUCE` / `EXPLAIN` / `SUSPECT` (0xA0–0xA2)

**Depends on:** Hypothesis Galaxy (ingestion-time preparation — Claude
will spec a separate slice; for this batch, use a stub Galaxy cluster
with 0 stars and verify the kernel runs on an empty set).

```cuda
// ABDUCE:  [..., obs] → [..., hyp_id]     // spatial query on Hypothesis Galaxy
// EXPLAIN: [..., hyp_id, obs] → [..., cover_mask]
//          TCOMP over packed ternary effect vector
// SUSPECT: [..., hyp_id] → [..., simplicity_score_tquant]
```

Simplicity score = term-size + symbol-rarity, both available from the
canonical star metadata (term size computable from RPN program length).

### 6.5 Dispatcher hooks

In the existing RPN dispatcher (Codex: locate the `switch(opcode)` in
`rpn_engine` / `rpn_ext`), add cases for the 8 new opcodes above. Each
case calls the corresponding `op_*` function. Keep dispatcher changes
behind a compile-time flag `K3D_REASONING_OPCODES_V1` so the batch can
roll back instantly if benchmarks regress.

### 6.6 S5 exit tests

Per opcode family:

- `tests/test_op_ctx_switch.py` — ingests two stars with different
  `context_id`, runs RPN program that sets context, asserts only
  matching star returned.
- `tests/test_op_tunify.py` — unifies `f(X, a)` with `f(b, Y)`, asserts
  substitution `{X↦b, Y↦a}`. Occur-check case: `f(X)` with `X` → fails.
- `tests/test_op_tableaux.py` — simple closed tableau for
  `p ∧ ¬p`, assert branch closes.
- `tests/test_op_abduce.py` — empty Hypothesis Galaxy, assert ABDUCE
  returns 0 (no-op success) and EXPLAIN / SUSPECT handle it cleanly.

All sovereign (PTX path), all ARC/Math non-regression.

---

## 7. S6 — Ethical Gate Extension

**File to modify:** `knowledge3d/cranium/cuda/gre_defeasible_resolver.cu`
(assumed path — Codex: confirm location; see parent spec refs to this
kernel being reused at 3 pipeline stages).

### 7.1 New stage

Add `stage3_ethical_gate(...)` that runs **after** the existing stage 3
conflict resolution:

```cuda
// For each candidate star in the current resolution frontier:
//   trit = star.ethical_trit
//   if trit == -1 (forbidden):   reject immediately, no further scoring
//   if trit == +1 (defeasible):  pass through stage 3 conflict resolution
//                                (existing behaviour)
//   if trit ==  0 (ok):          bypass conflict resolution, accept
```

Ternary trit packing for cross-kernel transport: use existing `TPACK`
(0x75) when writing to shared/global; keep as `int8` in the star struct
for simplicity.

### 7.2 Tests

- `tests/test_ethical_gate.py`:
  - Ingest 3 stars, one per trit value.
  - Run a defeasible query that would otherwise resolve to the
    forbidden star. Assert it is rejected regardless of score.
  - Assert the `ok` star bypasses conflict resolution (measured via
    per-stage timing counter).
- Sovereignty grep + benchmark non-regression.

### 7.3 Philosophy-star ingestion rule (light-touch for S6)

Full ethics ontology lands in a later wave. For S6 it is enough to
populate `ethical_trit` via a small allowlist in ingestion:

```python
# knowledge3d/ingestion/ethical_ingest_rules.py
FORBIDDEN_STARS = {"harm_intent", "deception_malicious", ...}  # short seed list
DEFEASIBLE_STARS = {"self_defense", "triage_tradeoff", ...}    # short seed list
# Everything else defaults to 0 (ok).
```

Claude will draft the full rules in a later spec; S6 just wires the
plumbing and ships the seed list so the gate is exercised in tests.

---

## 8. Non-Regression + CI Gates

Every slice must ship with:

- Unit test covering its new surface.
- Sovereignty grep passing:
  `rg -n 'numpy|cupy|scipy|sympy|re\.' <touched files in hot path>`
  returns **zero** hits.
- ARC-AGI 10/10, Math 20/20 benchmarks green.
- Opcode-budget grep:
  `rg '0x[A-F][0-9]' docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`
  monotone-increasing count (no reservations removed).

If any gate fails, the slice does not land. No `--no-verify`. No
fallbacks. We fix or we fix.

---

## 9. What's NOT in Batch 1 (deferred — see parent spec §7)

- Dynamic `N` selection (beyond N=9 clamp).
- LoRA specialist streaming + rotation per lane.
- Remaining opcodes (0xA3–A7, 0xB0–B5, 0xB7, 0xC1–C5, 0xD3–D4, 0xE0–E2,
  0xF0–F1 are reserved in S1 but not implemented here).
- Superposition, DPLL/SMT, Rete, full bi-abduction.
- Galaxy-wide HS ingestion from parent spec §9.2.
- Full philosophy ontology driving `ethical_trit`.
- Daemon / always-on migration (Phase C).

These follow the parent spec's §7 order in subsequent batches.

---

## 10. Hand-off Checklist

Codex, before starting:

- [ ] Read the parent spec `CLAUDE_REASONING_PARADIGMS_AND_N_SWARM_SPEC_2026-04-13.md`.
- [ ] Confirm kernel directory layout (`knowledge3d/cranium/cuda/`) and
      RPN dispatcher location.
- [ ] Confirm `gre_defeasible_resolver.cu` path; if different, report
      back before S6.
- [ ] Start with S1 (docs only) to warm up and to make opcode collisions
      impossible.
- [ ] Land S2 before any S5 work (hard dep).
- [ ] Land S3 before running S5 opcodes through the swarm (soft dep; S5
      can be unit-tested via the old bridge first).
- [ ] Open a Claude review checkpoint after S3 lands and before S5
      begins, so we catch architectural drift early.

**End of Batch 1 spec.**
