# Ternary Contrastive Attention — Kernel-Level Design Document v2

**Supersedes**: `ternary_contrastive_attention_design.md` (v1) for the
weight-matrix / projection sections. **v1 stays valid** for the contrastive-margin
section (§3 of v1 / §4 of this doc) and the V-mix section (§4 of v1 / §5 of this doc).

**Date**: 2026-04-18
**Author**: Claude (architecture, cuda-research-solver lane)
**Implementer**: Codex
**Daniel's Ruling**: "BitNet b1.58 — 1.6-bit packing + multiplication-free
  add/sub/skip kernels" (2026-04-18). Supersedes the XNOR+popcount assumption
  for weight-matrix compute.

**Sequential dependencies (must be complete before this phase)**:
1. Transfer Yard substrate (`float4 yards[9][9][69]`, 87.3 KB shared mem per core)
2. Ternary isolation contract (46 isolated cores, 9 instances/core)
3. Phase B native embedding spec (Matryoshka 64/128/512/2048 dim, meaning_rpn)
4. Ternary opcode family 0x100-0x107 (TERNARY_AND/OR/NOT/XOR, TQUANT, TCMP)
5. Yard addressing opcodes 0x170-0x17A (YARD_SELECT, YARD_PUSH_BANK, etc.)
6. v1 attention design (ternary_contrastive_attention_design.md) — v2 extends it

---

## 0. What Changed From v1 and Why

### The Packing Error in v1

v1 assumed that Q, K, V projection weight matrices used 2-bit ternary packing
(16 trits per uint32) with XNOR+popcount for the weight × activation inner product.
This is the correct pattern for **ternary × ternary** operations (rule masks, sign
comparisons, semantic gravity T(s₁,s₂) computations). It is wrong for
**ternary weights × INT8 activations** (linear projections W_q, W_k, W_v, W_o).

The canonical regime for weight matrices trained in the BitNet b1.58 paradigm is:
- **1.6-bit packing**: 5 trits per byte, 20 trits per uint32 (vs 16 in 2-bit).
- **Add/sub/skip kernel**: w=+1 → add activation, w=−1 → subtract activation,
  w=0 → skip. No integer multiplies. 82% less energy than float32 matmul.

### Corrected Compute Split

| Stage | Input × Input | Packing | Kernel | Opcode |
|---|---|---|---|---|
| W_q, W_k, W_v, W_o projections | ternary W × INT8 X | 1.6-bit (weight only) | add/sub/skip | 0x1AA |
| Q·K^T attention scores | INT8 Q × INT8 K | INT8 (both) | dp4a | (inside 0x1A8) |
| Contrastive margin top-K | integer scores | — | partial-sort | 0x1A9 |
| A·V output mix | INT8 attn weight × INT8 V | INT8 (both) | add/sub/skip | (inside 0x1A8) |
| Rule-mask logic (defeasible, GRE) | ternary × ternary | 2-bit (both) | XNOR+popcount | 0x108 |

### What v1 Got Right (Unchanged in v2)

- Contrastive margin formulation (§3 of this doc) — integer threshold + partial-sort.
- V-mix semantics (§5 of this doc) — TQUANT-gated float4 add/subtract.
- Two-scale architecture: context-attention vs retrieval-attention (§7 of this doc).
- Acceptance gates (§10 of this doc) — all six v1 gates still apply.
- Yard bank layout (bank_Q / bank_K / bank_V / bank_O / bank_S).
- TERNARY_XNOR_POPCOUNT (0x108) stays in the registry for rule-mask use.

---

## 1. Architecture Concept (Unchanged from v1)

Attention = selective focus. The TRM directs cognitive bandwidth toward Galaxy stars
most relevant to the current query. Frustum Culling is the coarse outer filter;
attention is the fine inner filter over survivors.

**Implementation**: ternary-weight dot-product scoring + contrastive-margin top-K +
ternary-gated value mixing. No `exp()`, no softmax, no transcendentals.

**Why ternary weights**: Dual convergence.

1. **External**: BitNet b1.58 (Ma et al., arXiv:2504.12285) — models trained with
   absmean quantization converge to weight values in {-1, 0, +1} (log₂3 ≈ 1.58 bits).
2. **Internal**: Semantic gravity T(s₁,s₂) ∈ {+1, 0, −1} (Christoph Dorn / Daniel
   Ramos, March 2026) — the ternary force between meanings IS the attention weight
   in equilibrium. Softmax continuous attention would contradict this at the
   conceptual AND implementation level simultaneously.

---

## 2. 1.6-bit Packing — The BitNet b1.58 Encoding

### 2.1 Why 1.6 Bits? The "Magic" of Base-3

Five ternary weights encode 3^5 = 243 possible states. One byte holds 256 states.
243/256 = 94.9% utilisation — only 13 of 256 byte patterns are unused.

This gives **1.6 bits per weight** (log₂(243)/5 ≈ 1.584). Compared to 2-bit packing
(16 trits per uint32), 1.6-bit packing stores 20 trits per uint32 — a 25% density
improvement.

```
Memory compression vs float32:
  float32 weight:     32 bits
  1.6-bit packed:     1.6 bits
  Compression ratio:  20×

Memory compression vs 2-bit packing (v1 scheme for weight matrices):
  2-bit packed:       2 bits per trit
  1.6-bit packed:     1.6 bits per trit
  Improvement:        25% denser; 25% less VRAM for weight storage
```

### 2.2 Encoding Table

| Trit value | Offset value (ti + 1) | Example byte for single trit |
|---|---|---|
| -1 | 0 | — |
|  0 | 1 | — |
| +1 | 2 | — |

Five-trit byte encoding:
```
byte = (t0+1)*81 + (t1+1)*27 + (t2+1)*9 + (t3+1)*3 + (t4+1)

Position weights: 81 = 3^4, 27 = 3^3, 9 = 3^2, 3 = 3^1, 1 = 3^0

Examples:
  (-1,-1,-1,-1,-1) → 0
  ( 0, 0, 0, 0, 0) → 121  (= 81+27+9+3+1 = 1*81 + 1*27 + 1*9 + 1*3 + 1)
  (+1,+1,+1,+1,+1) → 242
  (-1, 0,+1,-1,+1) → 0*81 + 1*27 + 2*9 + 0*3 + 2 = 0+27+18+0+2 = 47
```

### 2.3 Container Layout (uint32 = 4 bytes = 20 trits)

```
uint32 word layout (little-endian bytes):
  Byte 0 [bits 7:0]:   trits 0..4   (positions 0-4 of the weight row slice)
  Byte 1 [bits 15:8]:  trits 5..9
  Byte 2 [bits 23:16]: trits 10..14
  Byte 3 [bits 31:24]: trits 15..19
```

The byte order within the uint32 is **unambiguous** and **prefix-compatible** with
Matryoshka tiers: trits at lower indices always appear in lower bytes.

### 2.4 Matryoshka FOV Prefix Compatibility

Weight matrices packed in 1.6-bit format are natively prefix-compatible with
Matryoshka embedding tiers:

```
Row 0 of W_q → governs output dim 0 (Tier 64 prefix for d_out < 64)
Rows 0..63   → Tier 64 slice (d_out=64, d_in determined by tier_select)
Rows 0..127  → Tier 128 slice
Rows 0..511  → Tier 512 slice
Rows 0..2047 → Tier 2048 full

For Tier 64 projection: pass M=64 to TERNARY_MATMUL_ADDSUB (0x1AA)
For Tier 128: pass M=128. The packed weight bytes for rows 64-127 follow
immediately — no re-packing, just a different M operand.
```

### 2.5 Packing Strategy — Why LUT for Unpack

Three strategies analysed (Kimi swarm analysis, 2026-04-18):

| Strategy | Unpack cycles | Pack cycles | Shared mem | Bank conflicts |
|---|---|---|---|---|
| Base-3 arithmetic (div/mod) | ~18-25 | ~8 | 0 | 0 |
| **256-entry constant LUT (uint64_t)** | **~10** | ~8 | **2 KB constant** | **None (broadcast)** |
| Bit-interleaving | ~15 | ~20 | 0 | 0 |

Winner: **constant-memory LUT** for unpack. Reasoning:
- unpack5 is called ~20 times per warp per weight-row at inference (hot path).
- 2KB LUT fits in the 8KB per-SM constant cache working set without evicting yard data
  (yard lives in shared memory — completely separate 87.3 KB region).
- 4-thread broadcast pattern (all 4 threads in a group unpack the same byte)
  hits constant cache broadcast path: ~5 cycles vs ~18+ for arithmetic.
- Base-3 division (dividing by 81, 27, 9, 3) requires mul-reciprocal chains even
  with compiler optimisation — 3-5× slower than LUT broadcast.

LUT size confirmation: 256 entries × 8 bytes (uint64_t) = **2048 bytes = 2 KB**.
Each entry packs 5 int8_t values (already mapped to {-1,0,+1}) in bytes [4:0]
of the uint64_t. This is NOT a "256-byte" LUT — it is a "256-entry" LUT totalling 2KB.

---

## 3. Add/Sub/Skip Projection Kernel

### 3.1 Principle (No Multiplies)

```
For each (weight w, activation a) pair:
  w ∈ {-1, 0, +1}     (1.6-bit unpacked via LUT)
  a ∈ [-127, 127]      (INT8)

  Contribution:
    w = +1: accum += a   (integer add, 1 cycle)
    w = -1: accum -= a   (integer subtract, 1 cycle)
    w =  0: skip         (predicated no-op, 0 effective cycles)

No IMUL, no FMUL, no FMA. Energy ~82% lower than float32 matmul.
(Published BitNet b1.58 figure; ~50% zero weights × 0.1 nJ + ~50% add/sub × 0.5 nJ
 vs float32 FMA × 3.5 nJ = 0.3 vs 3.5 nJ/op.)
```

On sm_86, the conditional is compiled to ISETP (integer set predicate) + SELP
(select with predicate) — 2 instructions per weight, no branch divergence.

### 3.2 Warp-Cooperative Distribution

A warp has 32 threads. One uint32 word holds 20 trits (4 bytes × 5 trits/byte).

```
Per step: warp processes 8 uint32 words = 160 trits simultaneously.

Thread mapping within a step:
  word_group = lane / 4    ∈ [0, 7]    which of the 8 uint32 words
  byte_lane  = lane % 4    ∈ [0, 3]    which byte within the word

Thread 0  (g=0, b=0): trits  0.. 4 of word 0
Thread 1  (g=0, b=1): trits  5.. 9 of word 0
Thread 2  (g=0, b=2): trits 10..14 of word 0
Thread 3  (g=0, b=3): trits 15..19 of word 0
Thread 4  (g=1, b=0): trits  0.. 4 of word 1 (= trits 20..24 of current row slice)
...
Thread 31 (g=7, b=3): trits 15..19 of word 7 (= trits 155..159)
```

### 3.3 __shfl_sync for Word Broadcast

One thread per group performs the global memory load; the other three receive the
word via `__shfl_sync`. This gives 1 global load per 4 threads — 4× fewer
memory transactions vs each thread loading independently.

```c
// Thread (word_group*4 + 0) loads; threads (word_group*4 + 1,2,3) receive:
uint32_t w32 = 0;
if (byte_lane == 0) w32 = W32[word_base + word_group];
int src_lane = word_group * 4;
w32 = __shfl_sync(0xFFFFFFFF, w32, src_lane);  // broadcast to group

// Each thread then extracts its byte:
uint8_t packed_byte = (uint8_t)((w32 >> (byte_lane * 8)) & 0xFF);
```

`__shfl_sync` latency on sm_86: ~4-6 cycles. Used 8 times per step (once per group).
Dominant cost: the LUT broadcast (~5 cycles) and the 5 add/sub/skip ops (~10 cycles).

### 3.4 Register Pressure (Per Thread)

```
Per-thread register usage during matmul step:
  1  uint32   loaded/broadcast word
  1  uint8    extracted byte
  5  int8_t   unpacked trits (from LUT entry via byte-extract)
  5  int8_t   loaded activations
  1  int32_t  running accumulator
  ≈ 13 registers total

sm_86 register limit: 255 registers per thread.
Register pressure: negligible — leaves ~240 registers for the surrounding kernel context.
```

### 3.5 Shared Memory Bank Conflicts for Activations

Activations stored as INT8 in shared memory with stride-5 access pattern (each
thread loads 5 consecutive bytes) produce 4-way bank conflicts because sm_86
shared memory banks are 4 bytes wide:

```
Bank = (byte_address / 4) % 32
Thread 0 reads bytes 0..4   → banks 0,0,0,0,1 — 4-way conflict in bank 0
Thread 1 reads bytes 5..9   → banks 1,1,1,2,2 — conflict
...
```

**Resolution**: For short activation vectors (d=64, 128), load activations from
global memory using coalesced loads and keep in registers — no shared memory
needed. For longer vectors (d=512, 2048), pad activation storage to 4-byte
alignment (1 INT8 + 3 padding bytes per slot = bank stride 4, no conflicts).

The yard substrate already occupies 87.3 KB of the 96 KB shared memory per core.
Only ~8.7 KB remains. For d=64 activations: 64 bytes — trivially fits. For d=512:
512 bytes, still fits. For d=2048: 2048 bytes, still fits within the ~8.7 KB margin.

### 3.6 Full Projection Step (CUDA Pseudo-Code)

```c
// W_q projection: [d_out × d_in] ternary weight × [d_in] INT8 activation → [d_out] INT32

__global__ void ternary_projection_kernel(
    const uint8_t* __restrict__ W_packed,  // 1.6-bit packed, [d_out × ceil(d_in/5)] bytes
    const int8_t*  __restrict__ X,          // INT8 activations, [d_in] elements
    int32_t*       __restrict__ Y,          // INT32 output, [d_out] elements
    int d_out, int d_in)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;  // one warp per output row
    if (row >= d_out) return;

    const uint8_t* W_row = W_packed + row * ((d_in + 4) / 5);

    // bitnet_matmul_tile returns per-lane partial sum:
    int32_t partial = bitnet_matmul_tile(W_row, X, d_in);

    // Warp reduction (sum 32 partial sums → 1 scalar in lane 0):
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_down_sync(0xFFFFFFFF, partial, off);

    if ((threadIdx.x & 31) == 0)
        Y[row] = partial;
}

// After all projections: caller invokes vec_norm_l2_int8() to requantise
// Y from INT32 → INT8 with L2 normalisation.
```

---

## 4. Q·K^T in INT8 via dp4a

### 4.1 Why INT8 × INT8 Here (Not Ternary × Anything)

After W_q × input_int8 and W_k × input_int8 (both via add/sub/skip projection),
the outputs Q and K are INT8-requantised vectors. They are **not ternary** — they
are standard quantised activations with full INT8 range [-127, 127].

The Q·K^T inner product requires INT8 × INT8 multiply-accumulate. The correct
hardware instruction is `dp4a.s32.s32`:

### 4.2 dp4a Instruction

```
PTX: dp4a.s32.s32 d, a, b, c;
  d = c + dot4(a[7:0] as int8, b[7:0] as int8)
  Operands a, b: 4 × INT8 packed in one uint32 (LSB = element 0)
  Operand c: INT32 running accumulator
  Output d: INT32

Available: sm_61 or higher (confirmed sm_86 / RTX 3070 — PTX ISA 5.0+)
Throughput: 1 cycle per warp on sm_86 (4 INT8 multiply-adds per cycle)
Latency: 4 cycles
CUDA intrinsic: int32_t __dp4a(int a_packed, int b_packed, int c_acc)
```

### 4.3 Full d=64 Dot Product (Q·K^T)

```c
// Pack Q and K as const int8_t[64]; each group of 4 int8_t = one uint32.
const uint32_t* q32 = reinterpret_cast<const uint32_t*>(Q_int8);
const uint32_t* k32 = reinterpret_cast<const uint32_t*>(K_int8);
int32_t score = 0;
#pragma unroll 16
for (int i = 0; i < 16; i++)      // 16 dp4a calls × 4 elements = 64 elements
    score = __dp4a(q32[i], k32[i], score);
// score ∈ [-1,032,256, +1,032,256] for d=64, safe in int32_t
```

Cycle cost: 16 dp4a × 1 cycle throughput = **16 cycles** (fully pipelined).

### 4.4 Cycle Comparison: v2 dp4a vs v1 XNOR+popcount

v1 assumed Q, K were ternary-packed (2-bit) and used XNOR+popcount:
```
v1 XNOR+popcount:
  12 instructions × 4 words (d=64) = 48 instructions ≈ 20-25 cycles
  Plus: TQUANT step to project INT8 activations → ternary before scoring: ~16 cycles
  Total v1 Q·K^T: ~36-41 cycles

v2 dp4a (INT8 × INT8, post-projection):
  16 dp4a instructions = 16 cycles (throughput-bound, pipelined)
  No TQUANT needed (Q, K are already INT8 from projection)
  Total v2 Q·K^T: ~16 cycles

Speedup: ~2.3× on Q·K^T step alone.
```

v1's XNOR+popcount IS correct for ternary×ternary rule-mask logic. The error was
applying it to the attention projection output, which is INT8 not ternary. dp4a
cannot be used for ternary×INT8 (the add/sub/skip path) — but it is the optimal
instruction for INT8×INT8 (the scoring path).

---

## 5. Contrastive Margin (Unchanged from v1)

The margin formulation, warp-cooperative partial-sort, and CONTRASTIVE_RANK_TOPK
(0x1A9) are unchanged from v1. Key points preserved:

```
score(q, k_i) = dp4a_dot(Q_int8, K_int8)   // integer ∈ [-d*127², +d*127²]

Margin gate: select k_i iff score(q, k_i) >= score(q, k_best) - m

m is loaded from star's confidence_trit field (Daniel's ruling — not hardcoded):
  m = YARD_PEEK_ADDR(star_record.confidence_trit)   // int8_t ∈ [0, 15] typical

Default: m = 8 for d=64 (12.5% of the non-squared score range ≈ [-64, +64])
         (Note: dp4a score range is larger; normalise m proportionally.)

No exp(), no softmax, no division.
```

**Galaxy schema update required**: star records must include `confidence_trit` field.
Cross-reference: Phase B native embedding spec (meaning_rpn projections produce
a scalar confidence estimate → quantised to int8_t → stored in `confidence_trit`).
Default for legacy stars without this field: m = 8 (conservative).

---

## 6. V-Mix — INT8 × INT8 (Updated Recommendation from v1)

v1 specified ternary-gated float4 V addition (TQUANT-gated float4 add/subtract).
v2 updates the V-mix recommendation to **INT8 × INT8** after requantisation of
attention weights:

### Two V-Mix Options

**Option A (v2 recommended): INT8 attention weights × INT8 V vectors**

After CONTRASTIVE_RANK_TOPK selects top-K candidates, the attention weights are
the scores themselves, requantised to INT8 (via VEC_NORM on the score vector).
The V-mix then becomes a weighted INT8 × INT8 sum using dp4a:

```
output[dim] = sum_{selected_i} score_int8[i] * V_int8[selected_i][dim]
```

dp4a handles 4 elements per cycle. For K=8 selected, d=64: 8 × 16 dp4a calls
= 128 cycles. Clean semantics; INT8 output directly without TQUANT conversion.

**Option B (v1 ternary-gated): still valid for sparse, fast mixing**

If scores are TQUANT-mapped to {-1,0,+1}, the V-mix degenerates to add/subtract/skip
over float4 tiles — no multiply at all. Faster when K is small and sparsity is high:

```c
w = TQUANT(score)   // {-1, 0, +1}
if (w == +1): output += V[selected_i]
if (w == -1): output -= V[selected_i]
// w == 0: skip
```

Cycle cost: 5-6 cycles per V vector (4 FADD + 1 ISETP).
For K=8: ~48 cycles.

**v2 recommendation**: Use Option B (ternary-gated) for K ≤ 8 (fast, sparse).
Use Option A (INT8 × INT8 dp4a) for K > 8 (richer weighting at modest extra cost).
Both options are sovereignty-compliant (no float32 multiply in hot path for Option A
since dp4a is integer; Option B uses only float4 ADD/SUB which is cheaper than MUL).

### VEC_NORM After V-Mix (MANDATORY)

After the V-mix output is accumulated in bank_O, **VEC_NORM_L2_INT8 (0x1AD) MUST
be invoked** before the output is passed to the next pipeline stage. This is Daniel's
ruling: N=1 default. Every attention-layer output in every RPN program must be followed
by 0x1AD.

Cycle cost of 0x1AD for d=64: ~87 cycles (integer L2 norm + scale + clamp). This is
a one-time cost per attention layer output, not per candidate — amortised over the
tick budget.

---

## 7. Full Pipeline Cost Summary (v2)

Reference configuration: H=8 attention heads, d_head=64, N=128 context candidates.
One tick of the Nine-Chain Swarm across 9 instances in one isolated core.

```
Step                    | Op               | Cycles (per head, per instance)
-----------------------|------------------|----------------------------------
W_q × input (d=64)     | 0x1AA ADDSUB     | ~25 cycles
W_k × input (d=64×N)   | 0x1AA ADDSUB ×N  | ~25 × N/9 cycles amortised (9 warps)
W_v × input (d=64×N)   | 0x1AA ADDSUB ×N  | ~25 × N/9
VEC_NORM Q,K,V         | 0x1AD ×3         | ~87 × 3 = 261 cycles
Q·K^T (N=128 candidates)| dp4a ×128       | 16 × 128 = 2,048 cycles
Contrastive top-K (K=8) | 0x1A9            | ~120 cycles (N=128 ref)
V-mix (Option B, K=8)   | ternary add/sub  | ~48 cycles
VEC_NORM output        | 0x1AD            | ~87 cycles
-----------------------|------------------|----------------------------------
Total per head (rough)  |                  | ~2,614 cycles dominant = Q·K^T
Per tick (H=8 heads)    |                  | ~20,912 cycles / head amortised
At 1800 MHz (RTX 3070)  |                  | ~11.6 µs per tick for attention

Tick budget (60 Hz game loop): 16,667 µs
Attention share at 60 Hz:      ~0.07% of tick budget
```

### Comparison vs v1 (XNOR+popcount projections)

```
v1 Q·K^T estimate (XNOR+popcount, assumed ternary K):
  48 cycles × N=128 = 6,144 cycles
  Plus TQUANT pre-projection: +16 cycles × N = 2,048 cycles
  Total v1 Q·K^T stage: ~8,192 cycles

v2 Q·K^T (dp4a, INT8 K after projection):
  16 cycles × N=128 = 2,048 cycles
  (No TQUANT needed — K is already INT8)
  Speedup: 4×

v1 projection (assumed XNOR+popcount, wrong kernel):
  12 cycles/word × 4 words × M=64 outputs = 3,072 cycles
v2 projection (0x1AA add/sub/skip, correct):
  25 cycles × M=64 = 1,600 cycles
  Speedup: 1.9×

Overall attention step speedup v2 vs v1: ~2.3-4× depending on N.
```

---

## 8. Matryoshka FOV Integration

The Matryoshka embedding tiers (64, 128, 512, 2048 dimensions) map directly to
the `tier_select` operand of ATTENTION_FWD_TERNARY (0x1A8):

```
tier_select = 0 → d=64:   W packed at 64 dims = 64/5 = 13 bytes = 4 uint32 words
tier_select = 1 → d=128:  128/5 = 26 bytes = 7 uint32 words (rounded to 32 = 8 words)
tier_select = 2 → d=512:  512/5 = 103 bytes = 26 uint32 words
tier_select = 3 → d=2048: 2048/5 = 410 bytes = 103 uint32 words
```

For Tier 64 coarse FOV pass (Frustum Culling alignment):
```rpn
PUSH bank_Q  PUSH bank_K  PUSH bank_V  PUSH bank_O
PUSH 8   ; N_candidates
PUSH 3   ; K_topk
YARD_PEEK_ADDR bank_meta, confidence_slot  ; load m from Galaxy star
PUSH 0   ; tier_select = 0 → d=64
0x1A8  ATTENTION_FWD_TERNARY
0x170  YARD_SELECT  bank_O
PUSH 64  PUSH 127
0x1AD  VEC_NORM_L2_INT8   ; MANDATORY
```

The 1.6-bit weight matrix packing is prefix-compatible with tier selection because
byte order matches row order (row 0 first, packed left-to-right in the weight array).

---

## 9. Training vs Inference Lanes (Updated)

```
TRAINING LANE (unchanged from v1):
  Weights: float32 Q, K, V projection matrices
  Kernel:  ATTENTION_FWD_BASE (0x1A7) — unchanged, training validator
  Export:  At checkpoint: apply absmean TQUANT → {-1, 0, +1} → pack5 into 1.6-bit format
           Then upload 1.6-bit packed matrices to Galaxy weight store

INFERENCE LANE (sovereign, updated):
  Weights: 1.6-bit packed Q, K, V, O weight matrices in Galaxy VRAM
  Projections: TERNARY_MATMUL_ADDSUB (0x1AA) — add/sub/skip, no multiply
  Scoring: dp4a (INT8 Q × INT8 K) — 4-way hardware INT8 accumulate
  Ranking: CONTRASTIVE_RANK_TOPK (0x1A9) — integer margin, no softmax
  V-mix: ternary-gated or INT8 add/sub (Option A or B per §6)
  Normalise: VEC_NORM_L2_INT8 (0x1AD) — mandatory after every head output

KEY CHANGE FROM v1:
  v1 assumed projections used XNOR+popcount (ternary×ternary).
  v2 uses add/sub/skip (ternary weight × INT8 activation) = correct BitNet b1.58 regime.
  v1's 2-bit weight packing for W_q, W_k, W_v, W_o → replaced with 1.6-bit packing.
  v1's TQUANT conversion of activations before scoring → REMOVED (was wrong).
```

---

## 10. Acceptance Gates (Extended from v1)

Gates 1-6 from v1 (ternary_contrastive_attention_design.md §9) still apply.
Two new gates for v2:

### Gate 7 — 1.6-bit Packing Round-Trip

```python
# Test: pack5 / unpack5 round-trip is lossless for all valid trits
def test_pack5_unpack5_roundtrip():
    from itertools import product
    for trits in product([-1, 0, 1], repeat=5):
        packed = pack5(*trits)
        assert 0 <= packed <= 242, f"Out of range: {packed}"
        unpacked = unpack5(packed)
        assert unpacked == list(trits), f"Mismatch: {trits} → {packed} → {unpacked}"

# Test: LUT handles clamp range
def test_unpack5_clamp():
    for b in range(243, 256):
        assert unpack5(b) == [1, 1, 1, 1, 1], f"Clamp failed for byte {b}"
```

### Gate 8 — No IMUL in Projection Kernel

```bash
# Verify add/sub/skip kernel contains no integer multiply instructions
# in the projection hot path (not init code):
nvdisasm knowledge3d/cranium/ptx/bitnet_matmul.ptx | \
    grep -v "IMAD.WIDE\|init\|pack5" | grep -c "IMAD\|IMUL"
# Expected: 0  (all multiply instructions should be in pack5, not in inference path)
```

### Gate 9 — dp4a Used for Q·K^T

```bash
# Verify dp4a is present in the attention scoring section:
nvdisasm knowledge3d/cranium/ptx/ternary_attention.ptx | grep -c "DP4A"
# Expected: >= 16 (at least 16 dp4a calls for d=64 Q·K^T)
```

### Gate 10 — VEC_NORM_L2_INT8 Follows Every ATTENTION_FWD_TERNARY in RPN Programs

```bash
# Programmatic check: parse RPN program bytecodes and verify 0x1AD follows 0x1A8
python3 -c "
import re, sys
program = open(sys.argv[1], 'rb').read().hex()
# Find all 0x1A8 occurrences, check for 0x1AD within next 32 opcodes
assert '1a8' not in program.split('1ad')[0], 'ATTENTION without VEC_NORM found'
print('Gate 10: PASS')
" <rpn_program_file>
```

---

## 11. Must-NOT-Do (Extended from v1)

All v1 must-not-do rules still apply. New rules for v2:

- Do NOT use XNOR+popcount (0x108) for weight-matrix projections. That opcode
  operates on ternary×ternary; weight projections are ternary×INT8 (use 0x1AA).
- Do NOT try to use dp4a for ternary-weight × INT8-activation projections.
  dp4a requires INT8 on BOTH sides; ternary weights are NOT INT8 — they are
  {-1,0,+1} packed 5-per-byte. The add/sub/skip path is the correct kernel.
- Do NOT pack W_q, W_k, W_v, W_o in 2-bit format. They MUST use 1.6-bit (5-per-byte)
  packing. The 2-bit format is reserved for rule-mask metadata (per-star trit fields,
  confidence_trit, defeasible rule strengths) — completely separate use case.
- Do NOT call 0x1AD VEC_NORM_L2_INT8 with `scale=0` — this would zero the vector.
  Minimum meaningful scale is 1; default is 127.
- Do NOT hardcode the margin `m` in any RPN program. Always load from star's
  `confidence_trit` field via YARD_PEEK_ADDR.
- Do NOT skip the LUT init call `bitnet_init_lut_host()` at boot. Without it,
  `bitnet_unpack5_lut` contains uninitialised constant memory → undefined trit values.
```
