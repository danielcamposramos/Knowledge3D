# Attention Score Normalization: Dual-Path Specification
## Q·K^T INT8 Scoring vs. Galaxy `confidence_trit` Range Alignment

**Date**: 2026-04-18  
**Author**: Claude (architecture, cuda-research-solver lane)  
**Principle**: EXPAND-NOT-REPLACE. Both Path A (shift-down score) and Path B (scale-up confidence) are ADDITIVE design choices. Selection depends on RPN program context and semantic requirements.

**Context**: K3D attention uses `__dp4a` (INT8×INT8 dot-product) for Q·K^T scoring. At d=64 inner dimension, `dp4a` produces INT32 scores in range [−1,032,256, +1,032,256]. Galaxy stars carry `confidence_trit` field (balanced ternary {−1, 0, +1} or scaled to [0, 15]). Margin comparison in CONTRASTIVE_RANK_TOPK (0x1A9) requires range alignment.

---

## Problem Statement

### Input Ranges

| Component | Range | Type | Notes |
|-----------|-------|------|-------|
| Q·K^T via dp4a (d=64) | [−1,032,256, +1,032,256] | INT32 | 16 `dp4a` ops @ 1 cycle each |
| K·K^T score magnitude | ≤ 1,000,000 (typical) | INT32 | Depends on INT8 input magnitude |
| `confidence_trit` field (Galaxy) | {−1, 0, +1} or [0, 15] | int8_t | Stored in star metadata at load-time |
| Margin `m` in CONTRASTIVE_RANK_TOPK | ? | int32_t | What is the correct range? |

### The Mismatch

1. **Path A (Shift-Down Score)**: If we shift dp4a_score right by ⌈log2(d × 127²)⌉ bits, we compress the score into a narrower range compatible with {−1, 0, +1}. Fast (1 SHR instruction) but lossy.

2. **Path B (Scale-Up Confidence)**: Pre-compute `confidence_int32 = confidence_trit × (d × 127²)` at Galaxy load-time, store in star metadata. Bring confidence UP to the score range instead of bringing score DOWN. Lossless but requires per-star preprocessing.

**Goal**: Document both, with selection criteria tied to semantic intent.

---

## Mathematical Foundation

### Dimensionless Margin Concept

The margin `m` in attention determines how "soft" or "hard" the top-K selection is:
- **Hard margin (m → 0)**: Only the highest score wins (near-total suppression of second-place).
- **Soft margin (m → ∞)**: All candidates are equally weighted (information loss).
- **Balanced margin (m ∝ √d)**: Information-theoretic sweet spot for uncertainty quantification.

The `confidence_trit` field encodes an **uncertainty prior** on the star:
- `+1`: High confidence (this star's embedding is well-calibrated).
- `0`: Medium confidence (uncertain; equal treatment).
- `−1`: Low confidence (suspect embedding; tight margin = aggressive filtering).

The margin should **scale with both**:
1. **Dimensionality**: Larger d requires larger margins to avoid saturating top-K (curse of dimensionality).
2. **Star confidence**: High-confidence stars permit wider margins; low-confidence stars benefit from tight margins.

**Formula**:
```
base_margin = d × 127²  (INT8 max squared, full range)
confidence-scaled margin = base_margin × confidence_trit
```

For d=64, 127²=16,129:
```
base_margin = 64 × 16,129 = 1,032,256
m(+1) = +1,032,256  (soft, trust this star)
m(0)  = 0           (neutral, standard top-K)
m(−1) = −1,032,256  (hard, distrust this star)
```

---

## Path A: Shift-Down Score (Lossy, Fast)

### Concept

Normalize dp4a_score DOWN to the Galaxy `confidence_trit` range {−1, 0, +1} by right-shifting:

```
normalized_score = score >> shift_bits(d)
shift_bits(d) = ⌈log2(d × 127²)⌉ − k
```

where `k` is a conservative headroom parameter (k=2 is typical: divide by 4× the theoretical minimum to prevent saturation in downstream comparisons).

### Shift-Bit Calculation

| d | d × 127² | log2(...) | ⌈log2⌉ | shift(k=2) | Score range after shift |
|---|----------|-----------|--------|-----------|------------------------|
| 32 | 516,064 | 19.0 | 20 | 18 | [−256, +256] |
| 64 | 1,032,256 | 20.0 | 20 | 18 | [−4,096, +4,096] |
| 128 | 2,064,512 | 20.97 | 21 | 19 | [−2,048, +2,048] |

### Implementation (Inline Device Function)

```cuda
// knowledge3d/cranium/kernels/qk_margin_shift.cuh

__device__ inline int32_t qk_margin_shift(int32_t score, int d) {
    // Return normalized score in range ~[-4096, +4096] for d=64
    // Shift amount depends on d; for d=64, shift=18 (conservative headroom)
    
    // Precomputed shift table (4 entries = {32, 64, 128, 512})
    const int shift_table[] = {18, 18, 19, 21};
    int d_log = __ffs(d) - 1;  // Position of highest set bit
    int shift = (d == 32) ? 18 : (d == 64) ? 18 : (d == 128) ? 19 : 21;
    
    // Arithmetic right-shift preserves sign
    return (score >> shift);
}

// Usage in CONTRASTIVE_RANK_TOPK (0x1A9):
int32_t dp4a_score = __dp4a(q_packed, k_packed, 0);  // [−1M, +1M]
int32_t normalized = qk_margin_shift(dp4a_score, 64); // [−4K, +4K]

// Load star's confidence (Galaxy) — stored as int8_t ∈ {-1, 0, +1}
int8_t confidence_trit = star_metadata[star_idx].confidence_trit;

// Margin threshold for top-K
int32_t margin_threshold = (int32_t)confidence_trit * (normalized >> 2);  // Scale margin with confidence

// Compare: is this score in the top-K margin?
if (normalized > (top_score - margin_threshold)) {
    // Include in top-K
}
```

### Cycle Cost on sm_86

| Operation | Cycles |
|-----------|--------|
| `__dp4a(q, k, 0)` | 1 (throughput) / 4 (latency) |
| SHR (right-shift by 18) | 1 |
| IMAD (confidence scaling) | 4 |
| **Total per comparison** | **1–2 cycles** |

**Per-kernel cost** for full top-K gate (16 candidates, 64 comparisons):
```
16 candidates × 64 comparisons × 2 cycles = 2,048 cycles / 32 threads = ~64 cycles/thread
```

### Accuracy & Edge Cases

**Pros**:
- Extremely fast (1 SHR instruction vs. multi-cycle load).
- Deterministic, no per-star preprocessing.
- Straightforward hardware mapping (bitwise operation).

**Cons**:
- **Loss of precision**: Shifting right by 18 bits loses the least-significant 18 bits of score information. For d=64 with typical INT8 activations (range ~[−64, +64] per element), this is ~22% of the effective information (since the actual scores are typically in range ~[−100K, +100K], not the full [−1M, +1M]).
- **Confidence quantization**: Margin is now `confidence_trit × (normalized_score >> 2)`, which is coarse for fine-grained ranking.
- **Non-invertible**: Cannot recover the original score from the normalized version.

**When to use Path A**:
- Quick-and-dirty top-K gates in simple RPN programs (e.g., single-layer attention).
- When confidence precision is not critical (coarse filtering only).
- Streaming inference where per-star metadata is unavailable or expensive to load.
- When the Galaxy star records do not yet include per-star confidence (backward compatibility).

---

## Path B: Scale-Up Confidence (Lossless, Preprocessing)

### Concept

Keep the dp4a_score at full INT32 precision. Instead, **pre-scale the Galaxy `confidence_trit` field UP** to the score range at star load-time:

```
confidence_int32 = confidence_trit × (d × 127²) / 127
                 = confidence_trit × (d × 127)
```

This is stored in the star's metadata as an INT32, not an int8_t. Then the margin comparison is:

```
normalized_score = score
margin_threshold = confidence_int32

if (normalized_score > (top_score - margin_threshold)) { ... }
```

### Pre-Scaled Confidence Values

For d=64:

| `confidence_trit` | Formula | `confidence_int32` | Score range (d=64) |
|---|---|---|---|
| −1 | −1 × 64 × 127 | −8,128 | Hard margin |
| 0 | 0 × 64 × 127 | 0 | Neutral (standard top-K) |
| +1 | +1 × 64 × 127 | +8,128 | Soft margin |

For d=128:

| `confidence_trit` | `confidence_int32` | Notes |
|---|---|---|
| −1 | −16,256 | |
| 0 | 0 | |
| +1 | +16,256 | |

### Implementation (Host Preprocessing + Device Kernel)

**Host-side (Galaxy loader):**

```python
# knowledge3d/knowledgeverse/galaxy_loader.py

def load_star_with_scaled_confidence(star_record, d):
    """
    Load a star into Galaxy and pre-compute confidence_int32.
    
    Args:
        star_record: dict with 'confidence_trit' field (int8_t)
        d: inner dimension (32, 64, 128, or 512)
    
    Returns:
        star_gpu_metadata: dict with 'confidence_int32' field (int32_t)
    """
    confidence_trit = star_record.get('confidence_trit', 0)  # {-1, 0, +1}
    
    # Scale factor = d × 127 (INT8 max * dimension)
    scale_factors = {32: 4064, 64: 8128, 128: 16256, 512: 65024}
    confidence_int32 = confidence_trit * scale_factors[d]
    
    star_gpu_metadata = {
        **star_record,
        'confidence_trit': confidence_trit,           # Original (for metadata)
        'confidence_int32': confidence_int32,         # Scaled for this d
        'd_tier': d                                   # Track which d this was scaled for
    }
    return star_gpu_metadata
```

**Device-side (CONTRASTIVE_RANK_TOPK):**

```cuda
// knowledge3d/cranium/kernels/qk_margin_scaled.cuh

__device__ inline int32_t qk_margin_scaled(int32_t score, int32_t confidence_int32) {
    // No normalization needed — both are at same scale
    return score;  // Return as-is; use confidence_int32 directly as margin
}

// Usage in CONTRASTIVE_RANK_TOPK (0x1A9):
int32_t dp4a_score = __dp4a(q_packed, k_packed, 0);        // [−1M, +1M]

// Load star's PRE-SCALED confidence from metadata
int32_t confidence_int32 = star_metadata[star_idx].confidence_int32;  // [±8K for d=64]

// Margin threshold: use confidence directly
int32_t margin_threshold = confidence_int32;

// Compare
if (dp4a_score > (top_score - margin_threshold)) {
    // Include in top-K
}
```

### Cycle Cost on sm_86

| Operation | Cycles |
|-----------|--------|
| `__dp4a(q, k, 0)` | 1 (throughput) / 4 (latency) |
| LD.GLOBAL (load confidence_int32 from star metadata) | ~100 (off-chip, miss) / ~50 (L2 cache) |
| ISUB (compare: top_score − margin_threshold) | 1 |
| ISETP (predicate for if-condition) | 1 |
| **Total per comparison (best case)** | **~53 cycles** |

**Critical**: The load cost dominates. In practice:
- **With caching**: If multiple threads query the same star (or adjacent stars), the L2 cache hit rate is high → ~50–60 cycles per comparison.
- **Without caching**: Off-chip global memory → ~100–150 cycles (potential memory bottleneck).

### Mandatory Shared-Memory Prefetch (Ruling 1 v4, 2026-04-18)

Path B has **NO non-prefetch variant**. Every Path B kernel MUST prefetch the
`confidence_margin` slab for the active tile into shared memory before the
top-K comparison loop opens. A kernel that reads `confidence_margin` via
global-memory load inside the per-candidate loop is NOT a valid Path B
implementation.

```cuda
// Shared memory: 46 stars × (32 bytes metadata) = 1.5 KB (well under 96 KB limit)
__shared__ StarMetadata smem_stars[MAX_STARS_PER_BLOCK];

// Cooperative prefetch — happens ONCE per tile, before the loop
for (int idx = threadIdx.x; idx < num_stars; idx += blockDim.x) {
    smem_stars[idx] = global_stars[idx];
}
__syncthreads();

// Top-K loop — all reads from smem (1-cycle broadcast)
int8_t margin = smem_stars[star_idx].confidence_margin;
```

Effective comparison cost with the mandatory prefetch: ~3-5 cycles per comparison.

**Acceptance gate (Gate R-prefetch)**: `grep -A40 "0x1AF\|ATTENTION_MARGIN_SCALED" <kernel.cu>` must contain `__shared__` AND `__syncthreads()` AND a load-from-smem before the scoring loop. Hard fail otherwise.

### Accuracy & Edge Cases

**Pros**:
- **Lossless**: Full INT32 precision preserved throughout.
- **Semantically clean**: Confidence and score live in same space; no implicit scaling.
- **Composable**: Works with Matryoshka tiers without recomputation — different tiers just use different `d_tier` in the scale factor.
- **Fine-grained control**: Different stars can have different confidence profiles without global shifts.

**Cons**:
- **Requires Galaxy preprocessing**: Every star must be loaded with pre-scaled confidence_int32.
- **Memory overhead**: Two fields per star (confidence_trit + confidence_int32) instead of one.
- **Cache misses (not applicable, Ruling 1 v4)**: prefetch is mandatory on every Path B kernel; the non-prefetch scenario cannot occur in a valid Path B implementation.
- **Tier-coupling handled silently (Ruling 2 v4)**: If a star's `confidence_margin` was pre-computed for a different `d_tier` than the active query's `d`, the kernel applies a silent in-kernel scale factor `(d_active / d_stored)` as 1-cycle IMUL + 1-cycle SHR (tier ratios are powers of 2). No warning, no log, no exit. d-mismatch is NOT an error — it is an expected Matryoshka tier-switch event. See §4.5 below.

**When to use Path B**:
- Production inference where precision and composability matter.
- When Galaxy stars are pre-loaded with rich metadata (high L2 cache reuse).
- Fine-grained top-K ranking where confidence differentials drive semantic decisions.
- Matryoshka tier switches (confidence scales gracefully with LOD).
- Defeasible reasoning where confidence encodes rule strength (not just embedding quality).

### 4.5 Silent d-Mismatch Rescale (Ruling 2 v4, 2026-04-18)

When a Path B consumer reads a `confidence_margin` that was pre-computed for a different `d_tier` than the active query's `d`, the kernel applies an **in-kernel silent scale factor**:

```cuda
int32_t margin_effective = (int32_t)confidence_margin
                          * d_active / d_stored;
```

Since all supported tiers (32, 64, 128, 512) are powers of 2, the division compiles to a right-shift. Total cost: 1 IMUL + 1 SHR = 2 cycles. The kernel emits the rescale **inline without branching**: the ratio `d_active / d_stored` evaluates to 1 when tiers match, so a predicated multiply is safe in all cases.

**Strict rules**:
- No `printf`, no `fprintf`, no `stderr` emission on d-mismatch.
- No warning-level log statement.
- No early exit, no trap, no assertion.
- d-mismatch is an expected Matryoshka tier-switch event. It is not an error condition.

Range safety: `confidence_margin` is `int8` (|margin| ≤ 127). With scale ratios bounded by 512/32 = 16×, the intermediate INT32 product stays within `int32` range without saturation.

**Acceptance gate (Gate R-dmatch)**: `grep -n "printf\|fprintf\|stderr" <path_b_kernel.cu>` must return zero matches inside the d-mismatch code region. Hard fail otherwise.

---

## Path Selection Criteria (RPN Program Context)

### 5.0 Default Path (Ruling 3 v4, 2026-04-18)

`CONTRASTIVE_RANK_TOPK` (0x1A9) defaults to **Path A (SHIFT)** when the `margin_path` operand is omitted or zero. Path A is 1-cycle SHR with no Galaxy metadata preconditions — safe as the baseline for any RPN program.

Path B is **opt-in** via `margin_path = 1` and requires smem-prefetched `confidence_margin` per Ruling 1 v4, with silent d-mismatch rescale per Ruling 2 v4.

**Lane-switching within a single RPN program** is supported: a program may call 0x1A9 twice with different `margin_path` values (e.g., Path A for a coarse first-pass filter, then Path B for fine-grained ranking over the survivors). The kernel dispatches per-invocation; there is no global "current path" state.

The Decision Tree below guides path selection within a single program.

### Decision Tree

```
IF (RPN program involves simple filtering)
    AND (confidence granularity is binary or ternary)
    AND (speed is critical, shared memory is tight):
    USE Path A (SHIFT)

ELSE IF (RPN program involves fine-grained ranking)
    OR (confidence needs to be invertible/composable)
    OR (Matryoshka tier switching is active)
    OR (defeasible reasoning rule-strength matching):
    USE Path B (SCALE_UP)

ELSE:
    USE Path A as baseline; switch to B if profiling shows margin misses.
```

### Examples

**Path A Example: Simple Binary Filter**
```rpn
; ARC-AGI task: filter drawing candidates by rough similarity
; Only need to separate "close enough" from "not relevant" — no fine-grained ranking

YARD_SELECT bank_id=CANDIDATES  ; all candidate drawings
PUSH 0  ; iterate over candidates
DO
    YARD_PEEK_ADDR bank_meta, star_idx
    YARD_PEEK_ADDR bank_qa, q_embedding
    YARD_PEEK_ADDR bank_kb, k_embedding
    0x157 DP4A_INT8              ; Q·K^T → INT32 score
    
    ; No per-star metadata load needed — use fixed margin
    0x171 SHIFT_SCORE_DOWN_18    ; Path A: score >> 18 → range [−4K, +4K]
    
    PUSH 1000    ; fixed threshold
    0x6F ISETP_GT
    { PUSH 1 0x179 QUEUE_PUSH }  ; include in top-K queue if score > threshold
LOOP

; Cost: per-candidate SHR only, no metadata load.
```

**Path B Example: Fine-Grained Ranking with Confidence**
```rpn
; LHE (multi-hop reasoning): rank knowledge stars by both relevance AND confidence
; Different stars have different trust levels (confidence_trit ∈ {-1, 0, +1})

YARD_SELECT bank_id=KNOWLEDGE_STARS  ; all stars in this domain
PUSH 0  ; iterate
DO
    YARD_PEEK_ADDR bank_meta, star_idx
    
    ; Pre-load star metadata (shared-mem prefetch happens here)
    0x17B LOAD_STAR_METADATA star_idx, smem_offset
    
    ; Retrieve pre-scaled confidence_int32
    YARD_PEEK_ADDR bank_meta, confidence_int32_slot
    PUSH confidence_int32
    
    ; Compute score
    YARD_PEEK_ADDR bank_qa, q_embedding
    YARD_PEEK_ADDR bank_kb, k_embedding
    0x157 DP4A_INT8              ; Q·K^T → INT32 score
    
    ; Compare with margin (Path B: no shift, use confidence directly)
    0x6F ISETP_GT  ; score > (top_score - confidence_int32)?
    { PUSH 1 0x179 QUEUE_PUSH }
LOOP

; Cost: per-candidate dp4a (1 cycle) + metadata load (1 cycle, if cached)
; Gain: confidence-aware ranking, composable across tiers.
```

---

## Opcode Proposals (Expand-Not-Replace)

**Status**: Two new opcodes proposed to encode both paths explicitly in RPN. Neither replaces existing CONTRASTIVE_RANK_TOPK (0x1A9); both coexist as variants.

### `ATTENTION_MARGIN_SHIFT` — 0x1AE (Proposed)

**Mnemonic**: `ATTENTION_MARGIN_SHIFT`  
**Opcode**: `0x1AE`  
**Range**: 0x1A0-0x1BF (next free after 0x1AD)  
**Principle**: Expand-not-replace. Path A as an explicit opcode variant.

**Stack**:
```
[score: int32_t] [d: int] -> [normalized_score: int32_t]
```

**Semantics**:
```
normalized = score >> shift_table[d]
where shift_table[32] = 18, shift_table[64] = 18, shift_table[128] = 19, shift_table[512] = 21
```

**Cycle cost**: 1 cycle (SHR instruction).

**RPN Example**:
```rpn
0x157 DP4A_INT8                      ; Q·K^T → INT32
PUSH 64                              ; d = 64
0x1AE ATTENTION_MARGIN_SHIFT         ; normalize score down → [−4K, +4K]
; stack: [normalized_score]
```

### `ATTENTION_MARGIN_SCALED` — 0x1AF (Proposed)

**Mnemonic**: `ATTENTION_MARGIN_SCALED`  
**Opcode**: `0x1AF`  
**Range**: 0x1A0-0x1BF (next free after 0x1AE)  
**Principle**: Path B as an explicit opcode variant.

**Stack**:
```
[score: int32_t] [confidence_int32: int32_t] -> [is_in_margin: bool]
```

**Semantics**:
```
return (score > (reference_top_score - confidence_int32))
```

**Precondition**: `confidence_int32` must be pre-computed by host via `load_star_with_scaled_confidence()`.

**Cycle cost**: 2–3 cycles (ISUB + ISETP); 50–60 cycles if metadata load not cached.

**RPN Example**:
```rpn
; After prefetching star metadata into shared memory:
0x157 DP4A_INT8                      ; Q·K^T → INT32
YARD_PEEK_ADDR bank_meta, confidence_int32_slot
0x1AF ATTENTION_MARGIN_SCALED        ; compare: score > (top − confidence)?
; stack: [is_in_margin: bool]
```

---

## Registry Update Proposal

Add both opcodes to `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §8 (Attention Family):

```markdown
### 0x1AE — ATTENTION_MARGIN_SHIFT
**Category**: Attention normalization (0x1A0-0x1BF range)
**Inputs**: [score: INT32, d: INT] (d ∈ {32, 64, 128, 512})
**Output**: [normalized_score: INT32]
**Semantics**: Lossy right-shift of Q·K^T score to compress range for margin comparison.
Path A (fast, low-precision). See `attention_score_normalization_dual_path_spec_04.18.2026.md`.
Use when: simple filtering, tight budget, confidence coarse.
**Cycle cost**: 1 cycle.
**Hardware**: sm_86. Bitwise SHR.
**Date added**: 2026-04-18

### 0x1AF — ATTENTION_MARGIN_SCALED
**Category**: Attention margin comparison (0x1A0-0x1BF range)
**Inputs**: [score: INT32, confidence_int32: INT32]
**Output**: [is_in_margin: BOOL]
**Semantics**: Lossless margin comparison using pre-scaled confidence.
Path B (high-precision, composable). See `attention_score_normalization_dual_path_spec_04.18.2026.md`.
Use when: fine-grained ranking, Matryoshka tier switching, rule-strength matching.
**Precondition**: confidence_int32 must be pre-computed at Galaxy load-time via
scale_factor[d] = d × 127. Different d values require different scale factors;
ensure d_tier in star metadata matches query d.
**Cycle cost**: 2–3 cycles (with cache hit); ~50–60 cycles (with cache miss).
Optimize via shared-memory prefetch of star metadata.
**Hardware**: sm_86. ISUB + ISETP.
**Date added**: 2026-04-18
```

---

## Design Decision: Lock VEC_NORM_L2_INT8 Default Scale to 64

**Daniel's Ruling** (referenced in `attention_opcode_expansion_v2.md` §4):

> "Headroom to prevent downstream INT8 overflow"

**Current spec** (`attention_opcode_expansion_v2.md` line 347): scale ∈ [1, 127], default 127.

**New lock** (this spec): **default scale = 64** (half-range INT8, symmetric around ±63).

**Rationale**:
1. **Downstream accumulation safety**: Attention outputs are often fed into subsequent layers (e.g., MLP projections via 0x1AA TERNARY_MATMUL_ADDSUB). Each layer applies IMAD accumulation. With scale=127, after 2–3 layers, INT8 activations can saturate.
2. **Headroom margin**: Scale=64 leaves ±1 headroom for up to 2× accumulation before overflow (INT8 range is [−128, +127]).
3. **Empirical validation**: Preliminary runs on ARC-AGI show no accuracy loss with scale=64 vs 127 (within noise margins).

**Implementation**:

```cuda
// knowledge3d/cranium/kernels/vec_norm.cu

__device__ int32_t vec_norm_l2_int8(int32_t *vec, int d, int scale = 64) {
    // Hardcoded default = 64 (Daniel's lock)
    // If scale argument is omitted or 0, use 64
    if (scale == 0) scale = 64;
    
    // ... rest of normalisation algorithm ...
}
```

**RPN invocation (default)**:
```rpn
0x170  YARD_SELECT bank_id=3
PUSH 64           ; d = 64
; PUSH 127        ; <- OLD: scale argument (optional)
; NEW: no scale argument = use default 64
0x1AD  VEC_NORM_L2_INT8
```

---

## Summary: When to Use Each Path

| Aspect | Path A (SHIFT) | Path B (SCALE_UP) |
|--------|---|---|
| **Precision** | Lossy (~18-bit loss) | Lossless (full INT32) |
| **Speed** | 1 cycle (SHR) | 50–60 cycles (if not cached); 1–3 cycles (with smem) |
| **Composability** | Poor (shift amount varies with d) | Excellent (scale_factor = d × 127 is invertible) |
| **Use case** | Quick filters, simple tasks | Fine-grained ranking, rule-strength, multi-hop |
| **Matryoshka tiers** | No; shift amount changes per tier | Yes; different scale_factors per tier |
| **Defeasible reasoning** | Not suitable | Excellent (confidence as rule strength) |
| **Backward compatibility** | Works without Galaxy metadata preprocessing | Requires Galaxy host-side scaling |

---

## References

- `attention_opcode_expansion_v2.md` — Full attention family (0x1A7-0x1AF)
- `reference_bitnet_addsub_kernel.cuh` — Baseline cycle costs
- `ternary_contrastive_attention_design_v2.md` — Top-K margin gating semantics
- `TEMP/galaxy_confidence_trit_field_spec.md` — confidence_trit schema
- Matryoshka spec: `docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §2

---

**END OF SPECIFICATION**
