# Attention Opcode Expansion v2 — Additive Registry Update

**Date**: 2026-04-18
**Author**: Claude (architecture, cuda-research-solver lane)
**Implementer**: Codex
**Supersedes**: `attention_opcode_expansion.md` (v1) for opcodes 0x1AA-0x1AD.
**Principle**: EXPAND-NOT-REPLACE. Every entry below is ADDITIVE. v1 opcodes
  0x1A7, 0x1A8, 0x1A9, and 0x108 are preserved exactly. This document adds
  0x1AA, 0x1AB, 0x1AC, 0x1AD and reserves 0x1AE-0x1AF.

**Before implementing**: Codex MUST grep `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`
for any existing assignments at 0x1AA-0x1AD. If collisions exist, use the next free
slot in the 0x1A0-0x1BF range.

**Architecture note**: Core vs instance vocabulary per `feedback_core_vs_instance_vocabulary.md`.
RTX 3070 = 46 isolated cores (CUDA blocks) × 9 instances/core (warps) = 414 concurrent instances.
Cross-core communication via global-queue slots only (QUEUE_PUSH/POP/PEEK 0x178-0x17A).

---

## 0. Existing Opcodes — Preserved Unchanged

### `ATTENTION_FWD_BASE` — 0x1A7
**Status**: PRESERVED per Daniel's ruling. Float32 training-lane validator. Never in sovereign hot path.

### `ATTENTION_FWD_TERNARY` — 0x1A8
**Status**: RETAINED but **REDEFINED** to use BitNet b1.58 projections internally.

v1 defined the Q, K, V projections as XNOR+popcount over 2-bit-packed ternary×ternary
weights. That is **correct for rule-mask logic** but **wrong for weight-matrix compute**:
when multiplying ternary WEIGHTS by INT8 ACTIVATIONS, the canonical pattern (BitNet b1.58)
is add/sub/skip, not XNOR+popcount. The redefinition only affects the projection sub-steps
(W_q×input, W_k×input, W_v×input, W_o×output). The contrastive margin and V-mix sections
of 0x1A8 are unchanged.

Updated internal structure of 0x1A8 (see §3 of `ternary_contrastive_attention_design_v2.md`):
- Projections: 0x1AA TERNARY_MATMUL_ADDSUB (new)
- Scoring:     0x108 TERNARY_XNOR_POPCOUNT (unchanged — Q·K^T uses INT8 × INT8 via dp4a)
- Top-K:       0x1A9 CONTRASTIVE_RANK_TOPK (unchanged)
- V-mix:       INT8 × INT8 add/sub via TQUANT-gated additions (unchanged semantics)
- Post-attn:   0x1AD VEC_NORM_L2_INT8 — MANDATORY after every 0x1A8 invocation

### `CONTRASTIVE_RANK_TOPK` — 0x1A9
**Status**: EXTENDED (v4 Ruling 3, turn-6, 2026-04-18) with `margin_path` 1-bit flag. v1 semantics preserved for `margin_path = 0`.

Warp-cooperative bitonic top-K + margin gate. v4 adds a 1-bit `margin_path` operand that selects the margin comparison pathway:

- `margin_path = 0` (**default**, Ruling 3 v4): **Path A (SHIFT)** — uses 0x1AE (ATTENTION_MARGIN_SHIFT) semantics inline. 1-cycle SHR. No Galaxy metadata load required. Coarse-grained filtering; safe baseline.
- `margin_path = 1` (opt-in): **Path B (SCALED)** — uses 0x1AF (ATTENTION_MARGIN_SCALED) semantics inline. Requires **mandatory smem prefetch** per Ruling 1 v4. Silent d-mismatch rescale per Ruling 2 v4. Fine-grained ranking.

**Operand encoding (v4)**:
```
[opcode:       16 bits = 0x1A9]
[operand_0:     4 bits = bank_scores]
[operand_1:     4 bits = bank_stars]
[operand_2:     4 bits = bank_topk_out]
[operand_3:     8 bits = K_topk]
[operand_4:     1 bit  = margin_path]   ; 0 = Path A (default), 1 = Path B
[operand_5:     3 bits = reserved]      ; must be zero
```

**Lane-switch**: a single RPN program may call 0x1A9 twice with different `margin_path` values. The kernel dispatches per-invocation; no global "current path" state.

**Silent d-mismatch** (Ruling 2 v4): when `margin_path = 1` and `d_active != star.d_tier`, the kernel applies `margin × d_active / d_tier` inline (1 IMUL + 1 SHR). No warning, no log, no exit.

See v1 `attention_opcode_expansion.md` §3 for the bitonic top-K body; see `attention_score_normalization_dual_path_spec_04.18.2026.md` §4.5 and §5.0 for the Ruling 2 and Ruling 3 semantics.

### `TERNARY_XNOR_POPCOUNT` — 0x108
**Status**: PRESERVED unchanged. Used for rule-mask logic (ternary×ternary inner
products in defeasible reasoning, rule-application, GRE specialist matching).
The attention hot path does NOT use 0x108 for projections (uses 0x1AA instead),
but 0x108 remains valid for pure ternary×ternary logic anywhere in the pipeline.

---

## 1. `TERNARY_MATMUL_ADDSUB` — 0x1AA

**Mnemonic**: `TERNARY_MATMUL_ADDSUB`
**Opcode**: `0x1AA`
**Range**: 0x1A0-0x1BF (next free after 0x1A9)
**Layer**: Linear projection substrate — the BitNet add/sub/skip matmul kernel
**Priority**: P0 (required by 0x1A8 ATTENTION_FWD_TERNARY for projection steps)

### Purpose

This is the atomic building block for ALL ternary-weight × INT8-activation
multiplication in the K3D pipeline. It is used by:
- Q/K/V/O projections within ATTENTION_FWD_TERNARY (0x1A8)
- Any MLP layer that is ternary-weight quantised (future use)
- Any scalar product where one operand is 1.6-bit packed and the other is INT8

### Kernel operation

```
For each (weight, activation) pair:
  w ∈ {-1, 0, +1}   (1.6-bit unpacked)
  a ∈ [-127, 127]    (INT8)

  contribution = a   if w = +1   (ADD)
               = -a  if w = -1   (SUBTRACT)
               = 0   if w =  0   (SKIP — no instruction issued)

accum += contribution   for all pairs in the weight tile
```

No integer multiplies. On sm_86: each weight×activation pair costs 2 instructions
(ISETP integer compare + SELP select-with-predicate). For 50% zero weights (typical
BitNet b1.58 distribution), ~50% of contributions are predicated-off no-ops.

### Binary Layout (Operand Encoding)

```
[opcode:   16 bits = 0x1AA]
[operand_0: 4 bits = bank_W]    Bank containing 1.6-bit packed weight tile pointer
                                 (one slot = pointer to global/constant memory)
[operand_1: 4 bits = bank_X]    Bank containing INT8 activation tile pointer
[operand_2: 4 bits = bank_Y]    Output bank — receives INT32 accumulator
[operand_3: 8 bits = M]         Output dimension (rows of weight matrix, ≤ 255)
[operand_4: 8 bits = K]         Input dimension / dot-product length (≤ 255)
[operand_5: 1 bit  = reduce]    0 = return per-lane partial sums (for chaining)
                                 1 = warp-reduce to lane-0 result (scalar output)
```

### Precondition

```
yards[instance][bank_W][0]     = pointer to 1.6-bit packed weight tile
                                  Layout: M rows × ceil(K/5) bytes per row
yards[instance][bank_X][0..K/16) = INT8 activations, packed as float4 tiles
                                   (4 × float4 slots = 64 INT8 = K=64)
yards[instance][bank_Y][0..]   = output buffer, INT32 (zero-initialised by caller)
```

### Postcondition

```
yards[instance][bank_Y][0..M)  = INT32 accumulators, one per output row
                                  Range: [-K*127, K*127]
                                  Caller MUST requantise before further use.
```

### Cycle Cost on sm_86 (RTX 3070, 46 isolated cores, 9 instances/core)

```
Per dot product (M=1, K=64):
  Packing: 64 trits = ceil(64/20) = 4 uint32 words = 16 bytes
  Per step (8 words = 160 trits): 1 shfl per group + 5 add/sub/skip × 8 groups
    = 8 shfl (4 cycles each) + 40 add/sub/skip (2 instructions each)
    = 32 + 80 = 112 instructions per step
  For K=64 (4 words): steps = ceil(4/8) = 1 step
    = ~112 instructions / 32 warp threads = ~3.5 cycles amortised
  Warp reduction: 5 × __shfl_down_sync = ~20 cycles
  Total per output element: ~25 cycles

For full projection (M=64, K=64):
  64 output rows × 25 cycles = 1,600 cycles
  (if rows dispatched to separate warps: 1,600 / 9 instances = ~178 cycles amortised
   across the 9 instances in one isolated core)

Compare vs v1 XNOR+popcount path (ternary×ternary, wrong for this case):
  XNOR+popcount per word: 12 cycles × 4 words = 48 cycles
  v1 would also require TQUANT to project INT8→ternary first: +16 cycles
  Total v1 cost: ~64 cycles per dot
  0x1AA add/sub/skip path: ~25 cycles
  Speedup: ~2.6× on the projection step.

Compare vs float32 matmul baseline (64-element FMA):
  float32 FMA: ~96-128 cycles per dot (64 ops)
  0x1AA:       ~25 cycles
  Speedup: ~4-5×
```

### RPN Invocation Examples

```rpn
; Project input (INT8, K=64) through W_q (ternary, M=64) → Q_int32:
PUSH bank_W         ; pointer to W_q packed weight matrix
PUSH bank_X         ; pointer to input INT8 vector
PUSH bank_Y         ; pointer to Q accumulator output
PUSH 64             ; M = output dimension
PUSH 64             ; K = input dimension
PUSH 1              ; reduce = 1 (warp-reduce, scalar per row)
0x1AA  TERNARY_MATMUL_ADDSUB

; After projection, requantise via scale stored in Galaxy star record:
YARD_PEEK_ADDR bank_meta, scale_slot   ; load INT8 scale from star metadata
0x1AD  VEC_NORM_L2_INT8               ; mandatory post-projection normalisation
```

### Matryoshka FOV Compatibility

The 1.6-bit packing is prefix-compatible with Matryoshka tiers IF the packing
order of the weight matrix rows matches the tier order:

```
Tier 64:   First 64 trits (rows 0..63) of the weight matrix
Tier 128:  First 128 trits (rows 0..127)
Tier 512:  First 512 trits
Tier 2048: All 2048 trits

Pack order: MUST pack row 0 first, row 1 second, etc.
When invoking at tier 64: pass M=64, K=d_tier64.
The bytes for higher tiers are simply not accessed — no re-packing needed.
```

This guarantees that a weight matrix packed once at tier 2048 can be used for
any coarser tier without re-encoding.

---

## 2. `TERNARY_PACK5` — 0x1AB

**Mnemonic**: `TERNARY_PACK5`
**Opcode**: `0x1AB`
**Range**: 0x1A0-0x1BF
**Layer**: 1.6-bit encoding utility (ingestion path; also available in sovereign path
          for on-the-fly weight packing during TRM self-modification)
**Priority**: P1 (required at weight-upload time; optional in hot path)

### Binary Layout

```
[opcode: 16 bits = 0x1AB]
[operand_0..4: implicit] = top 5 slots of active bank, each int8_t ∈ {-1, 0, +1}
Result: pops 5 slots, pushes 1 uint8 byte in [0, 242]
```

Operand order on stack (top → bottom): t4, t3, t2, t1, t0
(t0 is deepest, t4 is topmost, consistent with RPN push order)

### Precondition

```
Active bank stack (top-down): [t4, t3, t2, t1, t0]
All values in {-1, 0, +1} (int8_t)
```

### Postcondition

```
Active bank stack (top-down): [packed_byte]
packed_byte = (t0+1)*81 + (t1+1)*27 + (t2+1)*9 + (t3+1)*3 + (t4+1) ∈ [0, 242]
```

### Algorithm

```c
__device__ uint8_t ternary_pack5_op(int8_t t0, int8_t t1, int8_t t2,
                                     int8_t t3, int8_t t4) {
    // Identical to pack5() in reference_bitnet_addsub_kernel.cuh
    return (uint8_t)((t0+1)*81 + (t1+1)*27 + (t2+1)*9 + (t3+1)*3 + (t4+1));
}
```

### Cycle Cost on sm_86

```
5 IMAD (integer multiply-add): 5 × 4 cycles latency = ~8 cycles (pipelined)
Total: ~8 cycles. Only called at weight-upload or self-modification time.
```

### RPN Invocation Example

```rpn
; Pack 5 learned trits (on stack after TQUANT pipeline) into one byte:
PUSH -1   ; t0
PUSH  0   ; t1
PUSH  1   ; t2
PUSH -1   ; t3
PUSH  1   ; t4
0x1AB  TERNARY_PACK5   ; pops 5, pushes 1 byte = (-1+1)*81 + 0*27 + 2*9 + 0*3 + 2 = 0+0+18+0+2 = 20
```

---

## 3. `TERNARY_UNPACK5` — 0x1AC

**Mnemonic**: `TERNARY_UNPACK5`
**Opcode**: `0x1AC`
**Range**: 0x1A0-0x1BF
**Layer**: 1.6-bit decoding utility
**Priority**: P1 (used internally by 0x1AA; also available standalone for debugging)

### Implementation

Uses the 256-entry `__constant__ uint64_t bitnet_unpack5_lut[256]` (2KB constant
memory) declared in `reference_bitnet_addsub_kernel.cuh`.

Constant memory cache on sm_86: 64 KB total (working set per SM). The 2KB LUT fits
easily without evicting yard data (yard lives in shared memory, separate memory
hierarchy). For 4-thread broadcast (all 4 threads in a group reading the same
packed byte), constant cache latency is ~5 cycles.

### Binary Layout

```
[opcode: 16 bits = 0x1AC]
[operand_0: implicit] = top of active bank (uint8_t packed byte, ∈ [0, 242])
Result: pops 1 slot, pushes 5 int8_t slots [t0, t1, t2, t3, t4]
        Top of stack after: t4 (most-recently-packed trit)
        Stack order: t0 is deepest, t4 is top
```

### Precondition

```
Active bank stack (top-down): [packed_byte]
packed_byte ∈ [0, 255]; values 243..255 clamp to (+1,+1,+1,+1,+1)
```

### Postcondition

```
Active bank stack (top-down): [t4, t3, t2, t1, t0]
All values in {-1, 0, +1} (int8_t)
```

### Cycle Cost on sm_86

```
1 LD.CONST (constant-cache load, 4-thread broadcast): ~5 cycles
5 byte extracts (AND + shift): 5 cycles
Total: ~10 cycles
```

### RPN Invocation Example

```rpn
; Decode a packed weight byte back to 5 trits (for debugging or self-inspection):
PUSH 0x14          ; packed_byte = 20 decimal
0x1AC  TERNARY_UNPACK5
; Stack now: [t4=+1, t3=-1, t2=+1, t1=0, t0=-1]  (reading top-down)
```

---

## 4. `VEC_NORM_L2_INT8` — 0x1AD

**Mnemonic**: `VEC_NORM_L2_INT8`
**Opcode**: `0x1AD`
**Range**: 0x1A0-0x1BF
**Layer**: Post-attention normalisation — MANDATORY after every attention output
**Priority**: P0 (Daniel's ruling: "implement, default N=1")

### Mandatory Usage Rule

Every RPN program that invokes ATTENTION_FWD_TERNARY (0x1A8) MUST follow it with
VEC_NORM_L2_INT8 (0x1AD) before any subsequent yard operation on the output bank.
This is enforced at spec level; Codex must add an acceptance gate checking for
this pattern in all generated RPN programs:

```bash
# Acceptance gate: every 0x1A8 must be followed by 0x1AD before next YARD_* op
grep -A5 "0x1A8" <rpn_program_file> | grep -c "0x1AD"
# Expected: >= 1 per 0x1A8 occurrence
```

Purpose: Prevents unbounded accumulation of activation magnitudes across
attention layers. Without normalisation, INT8 values saturate after 2-3 layers
of add/sub accumulation.

### Binary Layout

```
[opcode:   16 bits = 0x1AD]
[operand_0: 4 bits = bank_V]   Bank containing INT8 vector to normalise
[operand_1: 8 bits = d]        Vector dimension (e.g. 64, 128, 512)
[operand_2: 8 bits = scale]    Target L2 norm scale (default 127 for full INT8 range;
                                use 64 for conservative half-range)
```

### Precondition

```
yards[instance][bank_V][0..d/16) = INT8 vector packed as float4 tiles
                                   (each float4 slot = 4 INT8 bytes = 16 bytes)
d > 0, scale in [1, 127]
```

### Postcondition

```
yards[instance][bank_V][0..d/16) = L2-normalised INT8 vector
                                   ||output||_2 ≈ scale (in INT8 fixed-point)
```

### Algorithm (no transcendentals)

```c
// Step 1: Compute L2 norm squared (integer)
int32_t sum_sq = 0;
for (i in [0, d)): sum_sq += (int32_t)v[i] * v[i];

// Step 2: Estimate sqrt(sum_sq) via integer bit-scan + Babylonian correction
// (2 iterations of x_{n+1} = (x_n + sum_sq/x_n)/2, shift-based — no division)
// See reference_bitnet_addsub_kernel.cuh §5 for full implementation.

// Step 3: Scale vector
for (i in [0, d)): out[i] = clamp(v[i] * scale / norm_est, -127, 127) as int8_t;
```

No `sqrtf()`, no `__fsqrt_rn()`, no `expf()`. Pure integer arithmetic.
The reciprocal-sqrt estimate is accurate to ±1 ULP at INT8 precision —
sufficient for maintaining bounded activations without exact float32 normalisation.

### Cycle Cost on sm_86

```
d=64:
  Step 1 (sum of squares): 64 IMAD = ~32 cycles (2-way pipelined)
  Step 2 (integer sqrt):    bit-scan + 2 iterations ≈ 15 cycles
  Step 3 (scale + clamp):  64 IMAD + 64 MIN/MAX = ~40 cycles
  Total: ~87 cycles for d=64

d=128: ~174 cycles (linear in d)
d=512: ~696 cycles
```

This is a one-time cost per attention layer output — acceptable relative to the
~40-cycle attention scoring step for N=64, K_topk=8.

### RPN Invocation Example

```rpn
; After ATTENTION_FWD_TERNARY produces output in bank_O:
0x170  YARD_SELECT  bank_id=3     ; select bank_O (output bank)
PUSH   64                          ; d = 64
PUSH   127                         ; scale = 127 (full INT8 range, N=1 default)
0x1AD  VEC_NORM_L2_INT8

; After projection (W_q × input) produces INT32 in bank_Q, requantise + normalise:
0x170  YARD_SELECT  bank_id=0
PUSH   64   PUSH   127
0x1AD  VEC_NORM_L2_INT8
```

---

## 5. `ATTENTION_MARGIN_SHIFT` — 0x1AE (New in v3)

**Mnemonic**: `ATTENTION_MARGIN_SHIFT`
**Opcode**: `0x1AE`
**Range**: 0x1A0-0x1BF
**Layer**: Attention margin normalization — Path A (lossy, fast)
**Priority**: P1 (optional; use when speed critical and precision acceptable)

### Purpose

Path A of dual-path normalization. Compress Q·K^T INT32 score [−1M, +1M] DOWN to match Galaxy `confidence_trit` range via right-shift. Fast (1 SHR) but lossy.

See `attention_score_normalization_dual_path_spec_04.18.2026.md` §2 (Problem Statement) and §3 (Path A) for full rationale.

### Binary Layout

```
[opcode: 16 bits = 0x1AE]
[operand_0: 8 bits = d]         Target dimension (32, 64, 128, 512)
```

### Precondition

```
Stack (top): [score: int32_t ∈ [-1M, +1M] from dp4a]
```

### Postcondition

```
Stack (top): [normalized_score: int32_t ∈ [-4K, +4K] for d=64]
```

### Algorithm

```c
__device__ int32_t attention_margin_shift_op(int32_t score, int d) {
    // Shift table: d → shift_bits (conservative headroom k=2)
    const int shift_table[] = {18, 18, 19, 21};  // d={32, 64, 128, 512}
    int shift = shift_table[log2(d)];
    return score >> shift;
}
```

### Cycle Cost on sm_86

```
SHR (right-shift):  1 cycle
Total:              1 cycle
```

### RPN Invocation Example

```rpn
; Compute Q·K^T score
0x157 DP4A_INT8              ; score: [−1M, +1M]

; Normalize down (Path A)
PUSH 64                      ; d = 64
0x1AE ATTENTION_MARGIN_SHIFT ; score → [−4K, +4K]

; Load confidence and apply margin
YARD_PEEK_ADDR bank_meta, confidence_trit_slot
PUSH confidence_trit
0x6F IMAD                    ; margin = confidence × (normalized >> 2)
```

### When to Use

- Quick-and-dirty filters with coarse confidence.
- Backward compatibility (no Galaxy preprocessing needed).
- Streaming inference where per-star metadata is unavailable.
- Tight shared-memory budgets.

See `attention_score_normalization_dual_path_spec_04.18.2026.md` §5 (Selection Criteria) for detailed decision tree.

---

## 6. `ATTENTION_MARGIN_SCALED` — 0x1AF (New in v3)

**Mnemonic**: `ATTENTION_MARGIN_SCALED`
**Opcode**: `0x1AF`
**Range**: 0x1A0-0x1BF
**Layer**: Attention margin comparison — Path B (lossless, composable)
**Priority**: **P0** (Ruling 1 v4 turn-6, 2026-04-18 — prefetch is structural, not optional)

### Mandatory Prefetch Rule (Ruling 1 v4, 2026-04-18)

0x1AF is defined to **REQUIRE** shared-memory prefetch of `confidence_margin` at the start of every enclosing kernel tile. The opcode's contract is "compare against a smem-resident margin". Any kernel that invokes 0x1AF without a preceding cooperative prefetch + `__syncthreads()` is **out of spec**.

Lane A (kernel-implementation lane) MUST emit the prefetch in the kernel body surrounding every 0x1AF invocation. There is **no opt-out flag**.

Acceptance gate (Gate R-prefetch): `grep -A40 "0x1AF\|ATTENTION_MARGIN_SCALED" <kernel.cu>` must contain `__shared__` AND `__syncthreads()` AND a load-from-smem before the scoring loop. Hard fail otherwise.

### Purpose

Path B of dual-path normalization. Keep Q·K^T score at full INT32 precision. Pre-scale Galaxy `confidence_trit` UP at load-time to match score range. Lossless and composable across Matryoshka tiers.

See `attention_score_normalization_dual_path_spec_04.18.2026.md` §4 (Path B) for full rationale.

### Binary Layout

```
[opcode: 16 bits = 0x1AF]
[operand_0: implicit]       Pops 2 slots: [confidence_int32, score]
Result: pushes [is_in_margin: bool]
```

### Precondition

```
Stack (top → bottom):
  [confidence_int32: int32_t, pre-scaled by host loader (d × 127 × trit)]
  [score: int32_t from dp4a]
  [reference_top_score: int32_t, from prior top-K tracking]
```

### Postcondition

```
Stack (top): [is_in_margin: int8_t, 1 if (score > top − confidence), 0 otherwise]
```

### Algorithm

```c
__device__ int8_t attention_margin_scaled_op(int32_t score, int32_t confidence_int32,
                                             int32_t reference_top_score) {
    // No normalisation of score needed — both at same scale
    int32_t margin_threshold = reference_top_score - confidence_int32;
    return (score > margin_threshold) ? 1 : 0;
}
```

### Pre-Scaling (Host Side)

When loading a star into Galaxy, pre-compute and store `confidence_int32`:

```python
# knowledge3d/knowledgeverse/galaxy_loader.py

confidence_int32 = confidence_trit × scale_factor[d]
where scale_factor[d] = d × 127  (INT8 max × dimension)

Example for d=64:
  scale_factor[64] = 8,128
  confidence_trit = +1 → confidence_int32 = +8,128 (soft margin)
  confidence_trit = 0  → confidence_int32 = 0      (neutral)
  confidence_trit = −1 → confidence_int32 = −8,128 (hard margin)
```

Store `confidence_int32` in star metadata alongside `confidence_trit`. Track the `d_tier` this was computed for; use the matching tier during query.

### Cycle Cost on sm_86

Prefetch is mandatory per Ruling 1 v4 — the non-prefetch branch does not exist as a valid implementation.

```
Loaded from shared memory (always prefetched per Ruling 1 v4):
  Shared-memory load (broadcast):  1 cycle
  ISUB (top − confidence):         1 cycle
  ISETP (compare):                 1 cycle
  Total:                           2–3 cycles per comparison
```

One-time cost of the cooperative prefetch at tile start: ~50-60 cycles amortized across the warp, invoked once per 46-star tile. Per-comparison cost dominates over many iterations.

### RPN Invocation Example

```rpn
; Assume star metadata prefetched into shared memory (one-time cost)

YARD_SELECT bank_id=CANDIDATES

PUSH reference_top_score   ; from prior iteration

DO
    YARD_PEEK_ADDR bank_meta, star_idx
    
    ; Retrieve pre-scaled confidence (smem_prefetch makes this 1 cycle)
    YARD_PEEK_SMEM bank_meta, confidence_int32_slot
    PUSH confidence_int32
    
    ; Compute score
    YARD_PEEK_ADDR bank_qa, q_embedding
    YARD_PEEK_ADDR bank_kb, k_embedding
    0x157 DP4A_INT8                      ; score ∈ [−1M, +1M]
    
    ; Compare with margin (Path B)
    0x1AF ATTENTION_MARGIN_SCALED        ; is_in_margin = (score > top − confidence)?
    
    { PUSH 1 0x179 QUEUE_PUSH }          ; if yes, add to top-K
LOOP
```

### Matryoshka Tier Switching

When switching Matryoshka tiers (e.g., from d=64 to d=128 LOD), use the corresponding scale_factor:

```
d=64:   scale_factor = 8,128   → confidence_int32[64]
d=128:  scale_factor = 16,256  → confidence_int32[128]
d=512:  scale_factor = 65,024  → confidence_int32[512]
```

Metadata loader precomputes all tier scales at load-time:

```python
confidence_int32 = {
    32:  confidence_trit * 4064,
    64:  confidence_trit * 8128,
    128: confidence_trit * 16256,
    512: confidence_trit * 65024,
}
```

Query code selects the appropriate scale for the active tier.

### d-Mismatch Handling (Ruling 2 v4, 2026-04-18)

If `d_active != star.d_tier`, 0x1AF applies a **silent in-kernel rescale**:

```
margin_effective = confidence_margin × d_active / d_tier
```

Implementation: 1 IMUL + 1 SHR (tier ratios are powers of 2). No branching on the mismatch — the ratio evaluates to 1 when tiers match. **No warning, no log, no exit**. d-mismatch is an expected Matryoshka tier-switch event, not an error.

Acceptance gate (Gate R-dmatch): `grep -n "printf\|fprintf\|stderr" <0x1AF_kernel_region>` must return zero matches. Hard fail otherwise.

### When to Use

- Production inference requiring fine-grained ranking.
- Defeasible reasoning (confidence = rule strength).
- Matryoshka tier switching without re-ranking.
- When Galaxy stars are pre-loaded with metadata (high L2 cache reuse).
- Multi-hop reasoning where confidence-aware filtering is critical.

See `attention_score_normalization_dual_path_spec_04.18.2026.md` §5 (Selection Criteria) for detailed decision tree.

---

## 7. Q·K^T in INT8 — Why dp4a, Not XNOR+popcount

After the ternary-weight projections (0x1AA) produce INT8 Q and K vectors, the
attention score Q·K^T is an INT8 × INT8 inner product. This is NOT a ternary
problem — both Q and K have full INT8 range [-127, 127].

The correct hardware instruction on sm_86 is `dp4a.s32.s32`:
```
dp4a.s32.s32 d, a, b, c;
// d = c + dot4(a[7:0] as int8, b[7:0] as int8)
// Packs 4 INT8 elements per 32-bit operand.
// Throughput: 1 cycle (sm_86). Latency: 4 cycles.
```

CUDA intrinsic: `int32_t __dp4a(int a_packed, int b_packed, int c_acc)`

For d=64: 16 `dp4a` calls × 1 cycle throughput = 16 cycles (fully pipelined).

Compare vs v1 XNOR+popcount at this stage:
- v1 used XNOR+popcount assuming Q, K were ternary. After projection, they are NOT.
- dp4a handles 4 INT8 products per cycle vs 1 ternary dot per 12 cycles.
- For d=64: dp4a = 16 cycles vs XNOR+popcount = 48 cycles = 3× dp4a win.

Key distinction:
- Use 0x1AA TERNARY_MATMUL_ADDSUB (add/sub/skip) for: ternary W × INT8 X (weight matrices).
- Use dp4a (via 0x1AA internal, or directly) for: INT8 Q × INT8 K (activation × activation).
- Use 0x108 TERNARY_XNOR_POPCOUNT for: rule-mask logic, defeasible reasoning, GRE specialists.

---

## 7. Galaxy Schema Requirement — confidence_trit Field

Daniel's ruling: `m` (contrastive margin) is loaded from each star's `confidence_trit`
field via `YARD_PEEK_ADDR` inside the scoring loop. This is not hardcoded.

Galaxy star records MUST include `confidence_trit` going forward. Schema addition:

```
star_record {
    ...existing fields...
    confidence_trit: int8_t   // ∈ {-1, 0, +1} or [0, 15] scaled
                              // Used as margin m in CONTRASTIVE_RANK_TOPK (0x1A9)
                              // Default: confidence_trit = 8 (= 12.5% of d=64 range)
                              // Cross-reference: Phase B native embedding spec
                              //   (meaning_rpn projections feed this field)
}
```

The `confidence_trit` field encodes how confident the star's embedding placement is.
Stars with low confidence (near 0) use a tight margin (m → small → near-winner-take-all).
Stars with high confidence use a wider margin (m → large → softer inclusion). This is
the Galaxy-native realisation of the attention "temperature" parameter.

Implementation note for Phase B native embedding spec:
`meaning_rpn` projections that produce Matryoshka embeddings should also produce a
scalar confidence estimate (e.g., norm of the residual from projection) quantised to
int8_t and stored in `confidence_trit`.

---

## 8. Summary Table — Complete Opcode Family (Updated in v3)

| Opcode | Mnemonic | Status | Purpose |
|--------|----------|--------|---------|
| `0x108` | `TERNARY_XNOR_POPCOUNT` | Preserved (v1) | Rule-mask ternary×ternary dot product |
| `0x1A7` | `ATTENTION_FWD_BASE` | Preserved (v1) | Float32 training-lane validator |
| `0x1A8` | `ATTENTION_FWD_TERNARY` | Redefined (v2) | Full BitNet b1.58 attention head |
| `0x1A9` | `CONTRASTIVE_RANK_TOPK` | Preserved (v1) | Margin-gated top-K ranking |
| `0x1AA` | `TERNARY_MATMUL_ADDSUB` | NEW (v2) | 1.6-bit weight × INT8 activation matmul |
| `0x1AB` | `TERNARY_PACK5` | NEW (v2) | Pack 5 trits into 1 byte (ingestion/self-mod) |
| `0x1AC` | `TERNARY_UNPACK5` | NEW (v2) | Unpack 1 byte into 5 trits (LUT-based) |
| `0x1AD` | `VEC_NORM_L2_INT8` | NEW (v2) | Mandatory post-attention L2 normalise (scale=64 default) |
| `0x1AE` | `ATTENTION_MARGIN_SHIFT` | NEW (v3) | Path A: lossy shift-down score normalisation |
| `0x1AF` | `ATTENTION_MARGIN_SCALED` | NEW (v3) | Path B: lossless pre-scaled confidence margin |

Total new opcodes in v2: **4** (0x1AA, 0x1AB, 0x1AC, 0x1AD).
Total new opcodes in v3: **2** (0x1AE, 0x1AF).
Total attention-family opcodes after v3: **9** (0x108 + 0x1A7-0x1AF).
**Naming principle**: EXPAND-NOT-REPLACE. Path A and Path B coexist; RPN programs select based on semantic requirements.

---

## 9. Registry Update — Codex Must Add to `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`

Codex adds the following four entries to the registry before implementing anything.
The registry update is the first commit in the v2 implementation sequence.

```markdown
### 0x1AA — TERNARY_MATMUL_ADDSUB
**Category**: Linear projection / BitNet b1.58 (0x1A0-0x1BF range)
**Inputs**: bank_W (1.6-bit packed weight tile), bank_X (INT8 activations), bank_Y (output),
           M (output dim), K (input dim), reduce flag
**Output**: INT32 accumulator in bank_Y; one value per output row
**Semantics**: Ternary-weight × INT8-activation matrix multiplication via add/sub/skip.
No integer multiplies. Weights stored 1.6-bit packed (5 trits per byte, 20 trits per uint32).
For each (w, a) pair: accum += a if w=+1, accum -= a if w=-1, skip if w=0.
Warp-cooperative: 4-thread groups handle 1 uint32 word each; __shfl_sync broadcasts word
to group; warp-reduce for final scalar output per row. VEC_NORM_L2_INT8 (0x1AD) MUST
follow requantisation of output.
**Hardware**: sm_86 (Ampere), uses __shfl_sync, __shfl_down_sync, __dp4a for Q·K^T step.
~25 cycles/element for K=64. 2KB __constant__ LUT for unpack5.
**Date added**: 2026-04-18

### 0x1AB — TERNARY_PACK5
**Category**: 1.6-bit encoding utility (0x1A0-0x1BF range)
**Inputs**: top 5 slots of active bank, each int8_t ∈ {-1, 0, +1} (t4=top, t0=bottom)
**Output**: 1 uint8_t byte ∈ [0, 242] representing 5 packed trits
**Semantics**: Pack 5 ternary weights into 1 byte using base-3 encoding:
byte = (t0+1)*81 + (t1+1)*27 + (t2+1)*9 + (t3+1)*3 + (t4+1).
Used at weight-upload time (ingestion) and during TRM self-modification (sovereign path).
~8 cycles. Inverse of TERNARY_UNPACK5 (0x1AC).
**Hardware**: sm_86. 5 IMAD instructions.
**Date added**: 2026-04-18

### 0x1AC — TERNARY_UNPACK5
**Category**: 1.6-bit decoding utility (0x1A0-0x1BF range)
**Inputs**: top of active bank (uint8_t packed_byte ∈ [0, 255]; 243-255 clamp to +1,+1,+1,+1,+1)
**Output**: 5 int8_t slots [t0,t1,t2,t3,t4] (t0=deepest, t4=top)
**Semantics**: Decode 1 byte into 5 ternary trits {-1,0,+1} using 256-entry constant-memory LUT.
LUT: __constant__ uint64_t bitnet_unpack5_lut[256] (2KB). Entries pre-computed at host init.
4-thread broadcast path: ~5 cycles. Inverse of TERNARY_PACK5 (0x1AB).
**Hardware**: sm_86. 1 LD.CONST + 5 byte-extract instructions ≈ 10 cycles.
**Date added**: 2026-04-18

### 0x1AD — VEC_NORM_L2_INT8
**Category**: Post-attention normalisation (0x1A0-0x1BF range)
**Inputs**: bank_V (INT8 vector), d (dimension), scale (target norm, default 127)
**Output**: bank_V in-place, L2-normalised INT8, ||v||_2 ≈ scale
**Semantics**: Integer L2 normalise with requantisation. No transcendentals.
Bit-scan sqrt estimate + 2 Babylonian refinement steps + integer scale + INT8 clamp.
MANDATORY after ATTENTION_FWD_TERNARY (0x1A8) and after projection outputs from
TERNARY_MATMUL_ADDSUB (0x1AA) before any subsequent attention stage.
Prevents unbounded activation accumulation across layers (Daniel's ruling, N=1 default).
**Hardware**: sm_86. ~87 cycles for d=64. Pure integer arithmetic (no sqrtf, no expf).
**Date added**: 2026-04-18
```

---

## 10. Implementation Order for Codex (v2 additions)

```
0. Grep RPN_DOMAIN_OPCODE_REGISTRY.md for 0x1AA-0x1AD — verify no collisions.

1. Add registry entries (§9 above) — first commit.

2. Add opcode constants to knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:
   TERNARY_MATMUL_ADDSUB = 0x1AA
   TERNARY_PACK5         = 0x1AB
   TERNARY_UNPACK5       = 0x1AC
   VEC_NORM_L2_INT8      = 0x1AD

3. Host init: call bitnet_init_lut_host() before first kernel launch.
   (Codex adds this to the sovereign hot-path boot sequence in knowledgeverse.py
   or cranium init — wherever kernel launch setup happens.)

4. Implement TERNARY_PACK5 (trivial — 5 IMAD).
   Test: pack5(-1,0,+1,-1,+1) == 20 (verify against encoding table).

5. Implement TERNARY_UNPACK5 using the LUT.
   Test: unpack5(20) == (-1, 0, +1, -1, +1) (round-trip with step 4).
   Test: unpack5(0) == (-1,-1,-1,-1,-1), unpack5(242) == (+1,+1,+1,+1,+1).
   Test: unpack5(243) == (+1,+1,+1,+1,+1) (clamp guard).

6. Implement TERNARY_MATMUL_ADDSUB using reference_bitnet_addsub_kernel.cuh.
   File: knowledge3d/cranium/kernels/bitnet_matmul.cu
   Test: Compare M=1, K=20 dot product against hand-computed reference.
   Test: Compare M=64, K=64 matmul against int32 reference (brute force).

7. Implement VEC_NORM_L2_INT8.
   File: knowledge3d/cranium/kernels/vec_norm.cu
   Test: Input vector with known L2 norm, verify output norm ≈ scale ± 2 INT8 units.
   Test: All-zero vector handled gracefully (no div-by-zero).

8. Wire ATTENTION_FWD_TERNARY (0x1A8) to call 0x1AA for projections.
   Verify: no XNOR+popcount in the W_q/W_k/W_v/W_o projection paths.
   Verify: dp4a used for Q·K^T (search for __dp4a in kernel source).

9. Add acceptance gate:
   grep -n "0x1AD\|VEC_NORM_L2_INT8" after every "0x1A8\|ATTENTION_FWD_TERNARY"
   in all generated RPN programs. Zero VEC_NORM follows = hard fail.

10. Compile all new kernels:
    nvcc -arch=sm_86 -ptx -o knowledge3d/cranium/ptx/bitnet_matmul.ptx \
         knowledge3d/cranium/kernels/bitnet_matmul.cu
    nvcc -arch=sm_86 -ptx -o knowledge3d/cranium/ptx/vec_norm.ptx \
         knowledge3d/cranium/kernels/vec_norm.cu

11. Run acceptance gates from ternary_contrastive_attention_design.md §9
    plus the new gate (step 9 above). Report all gate results.
```

---

## 11. Must-NOT-Do

- Do NOT call TQUANT on Q or K BEFORE the projection step — TQUANT is for
  mapping continuous scores to {-1,0,+1}; projection inputs are INT8 activations.
- Do NOT use XNOR+popcount (0x108) for W×X projections — that opcode requires
  BOTH operands ternary; activations are INT8, not ternary.
- Do NOT skip VEC_NORM_L2_INT8 after attention output. This is a hard rule.
- Do NOT use float32 reciprocal-sqrt (`rsqrtf`, `__frsqrt_rn`) in 0x1AD — the
  implementation must be pure integer (see reference_bitnet_addsub_kernel.cuh §5).
- Do NOT hardcode `m` in CONTRASTIVE_RANK_TOPK calls — load from star's
  `confidence_trit` field via YARD_PEEK_ADDR (Daniel's ruling).
- Do NOT use Tensor Core mma.sync for ternary weights — the 1.6-bit base-3
  encoding is not compatible with the b1.xor.popc format (which requires 1-bit packing).
- Do NOT remove 0x108 TERNARY_XNOR_POPCOUNT from the registry — it stays for
  rule-mask and defeasible reasoning logic where both operands are ternary.
```
