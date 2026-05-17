# Ternary Contrastive Attention — Kernel-Level Design Document

**Date**: 2026-04-18
**Author**: Claude (architecture, cuda-research-solver lane)
**Implementer**: Codex
**Daniel's Ruling**: "We need attention mechanism, but my guess is that this is logic (model weights) and must leverage ternary logic and contrastive learning."
**Expand-Not-Replace**: `ATTENTION_FWD_BASE` (0x1A7) is preserved; `ATTENTION_FWD_TERNARY` (0x1A8) is added alongside.

---

## 1. Concept / Implementation Split

Attention is NOT a neural-framework primitive here. It is two things, both realized in ternary:

**Concept**: Selective focus. The TRM directs cognitive bandwidth toward the sub-set of Galaxy stars most relevant to the current query. This is the same mechanism as FOV (Field of View) in the spatial sense — Frustum Culling is the coarse outer filter; attention is the fine inner filter over the candidates that survived culling. The two scales share the same kernel.

**Implementation**: Attention = ternary-weight dot-product scoring + contrastive-margin top-K selection + ternary-gated value mixing. No `exp()`, no softmax normalization, no transcendentals. Weights live in {-1, 0, +1}.

### 1.1 Why Ternary {-1, 0, +1} Is the Right Weight Space

Two independent converging motivations:

**External convergence — BitNet b1.58 regime** (Ma et al., arXiv:2504.12285, March 2026): Models trained with absmean quantization converge with weights in {-1, 0, +1} (log₂(3) ≈ 1.58 bits). At inference, ternary Q, K, V weights allow the attention dot product to be computed entirely with integer XOR + popcount arithmetic on packed 2-bit fields — no floating-point multiply-accumulate in the weight path. The 2B-parameter BitNet b1.58 model achieves full-quality inference with these constraints on CPU. On a GPU (sm_86) the speedup is larger because `popc.b32` is a single-cycle instruction and warp-parallel XNOR over 4×uint32 computes a d=64 ternary dot in ~17-20 instructions.

**Internal convergence — Semantic gravity cohered by meaning** (Christoph Dorn, March 2026, from Daniel Ramos): The K3D-native semantic force between two stars is `F = T(s₁,s₂) × M(s₁) × M(s₂) / d²` where T is the **ternary operator** ∈ {+1, 0, −1}. This is literally a ternary dot-product similarity between concept embeddings: +1 for affinity (attract), 0 for no evidence (neutral), −1 for contradiction (repel). Contrastive learning between meaning embeddings (not surface forms) IS the realization of T(s₁,s₂) in the composed head pipeline.

**Conclusion**: Attention weights must be ternary because the entire knowledge representation substrate is ternary. Softmax float32 attention would violate the semantic gravity formulation at the conceptual level and sovereignty at the implementation level simultaneously.

---

## 2. Q·K via XNOR + Popcount — Full Derivation

### 2.1 Packing Format

```
Encoding (K3D canonical 2-bit trit):
  +1 → 0b10   (high bit = 1, low bit = 0)
   0 → 0b01   (high bit = 0, low bit = 1)
  -1 → 0b00   (high bit = 0, low bit = 0)

One uint32 holds 16 trits. Trit i occupies bits [2i+1 : 2i].
For d_head = 64: 4 uint32 words per vector.
```

The encoding choice is deliberate: the zero sentinel (0b01) has a unique low-bit=1 pattern that allows zero-detection without an explicit compare instruction. The unused code 0b11 is illegal and treated as +1 for forward compatibility.

### 2.2 Extracting Bit Planes

From a packed word `w`:

```
sign_bits  = bits at odd positions  → (w & 0xAAAAAAAA)    high bit of each pair
zero_bits  = bits at even positions → (w & 0x55555555)    low bit of each pair

is_positive: high_bit=1, low_bit=0  → (w >> 1) & ~w & 0x55555555
is_negative: high_bit=0, low_bit=0  → ~(w | (w >> 1)) & 0x55555555
is_zero:     high_bit=0, low_bit=1  → w & ~(w >> 1) & 0x55555555

Validation:
  +1 (0b10): (w>>1)=...01, ~w=...01; 01 & 01 & 0x5..5 = set  ✓
   0 (0b01): (w>>1)=...00, ~w=...10; 00 &  ...  = clear       ✓ (is_positive=0)
             w=...01, ~(w>>1)=...11; 01 & 11 & 0x5..5 = set   ✓ (is_zero=1)
  -1 (0b00): ~(w|0)=...11, >> ...   = ...01 & 0x5..5 = set    ✓ (is_negative=1)
```

### 2.3 Zero-Mass Correction Problem

Simple XNOR on packed trits does NOT work directly because of the zero encoding. The XNOR truth table for 2-bit pairs reveals the problem:

```
q=+1(10) XNOR k=+1(10) → XNOR(10,10)=11  popcount=2  want: +1  (off by scale)
q=-1(00) XNOR k=-1(00) → XNOR(00,00)=11  popcount=2  want: +1  (off by scale)
q=+1(10) XNOR k=-1(00) → XNOR(10,00)=01  popcount=1  want: -1  (off by scale)
q= 0(01) XNOR k= 0(01) → XNOR(01,01)=11  popcount=2  want:  0  ← WRONG
q= 0(01) XNOR k=-1(00) → XNOR(01,00)=10  popcount=1  want:  0  ← WRONG
q=-1(00) XNOR k= 0(01) → XNOR(00,01)=10  popcount=1  want:  0  ← WRONG
q= 0(01) XNOR k=+1(10) → XNOR(01,10)=00  popcount=0  want:  0  ← correct
q=+1(10) XNOR k= 0(01) → XNOR(10,01)=00  popcount=0  want:  0  ← correct
```

The zero contributions are inconsistent — sometimes they register as agreements, sometimes as disagreements. Pure XNOR+popcount cannot be used without the zero mask.

### 2.4 Correct Formula — The Four-Mask Method

The correct ternary dot product across one uint32 (16 trits):

```
pos_q = is_positive mask of q    // set at trit positions where q = +1
neg_q = is_negative mask of q    // set at trit positions where q = -1
pos_k = is_positive mask of k
neg_k = is_negative mask of k

// Agreements: both +1 or both -1
agree = popcount(pos_q & pos_k) + popcount(neg_q & neg_k)

// Disagreements: one +1, one -1
disagree = popcount(pos_q & neg_k) + popcount(neg_q & pos_k)

dot_word = agree - disagree
```

### 2.5 Equivalent Two-Pass Sign-Plane Method (More Cache-Friendly)

An alternative with fewer intermediate registers, useful for loop unrolling:

```
// Extract sign plane (0 for -1 and 0, 1 for +1): shift high bits down to even positions
q_sign = (q & 0xAAAAAAAA) >> 1     // 16-bit packed in even positions
k_sign = (k & 0xAAAAAAAA) >> 1

// Non-zero mask: 1 where trit is not 0 (i.e., neither +1 nor -1 is "missing")
// nz = 1 means trit contributes to dot product
q_nz = ~(q & 0x55555555) & 0x55555555   // 1 at positions where q != 0
k_nz = ~(k & 0x55555555) & 0x55555555   // 1 at positions where k != 0
both_nz = q_nz & k_nz

// Sign XNOR over the 16 active positions
xor_sign = q_sign ^ k_sign           // 1 where signs differ
xnor_sign = ~xor_sign & 0x55555555   // 1 where signs match (in even positions only)

// Mask to valid positions only
match    = xnor_sign & both_nz        // agree AND both non-zero
mismatch = xor_sign  & both_nz        // disagree AND both non-zero

dot_word = popcount(match) - popcount(mismatch)
```

Instructions per word: 4×AND + 2×NOT + 1×XOR + 2×AND + 2×POPC + 1×SUB = 12 instructions
(vs. the 17-instruction full-path — the two-pass method saves by collapsing two `is_pos/is_neg` extractions into one sign plane)

### 2.6 Full d=64 Dot Product

Four words, unrolled:

```
// Words: q[0..3], k[0..3]
// Accumulate dot_sum (int32)

FOR word IN [0, 1, 2, 3]:
    dot_sum += ternary_dot_word(q[word], k[word])   // 12 instructions each

Total: 4 × 12 = 48 instructions
Range: dot_sum ∈ [-64, +64] (same range as float dot product of unit vectors scaled by 64)
```

### 2.7 Cycle Counts on sm_86 (Ampere)

```
Instruction throughput on sm_86 (RTX 3070):
  AND, OR, XOR, NOT: 1 cycle throughput, 4 cycles latency (integer pipeline)
  popc.b32:          1 cycle throughput, 4 cycles latency (SFU-adjacent)
  SUB.s32:           1 cycle throughput, 4 cycles latency

Per word (12 instructions, no data hazard stalls if pipelined):
  Throughput-bound:  12 cycles
  Latency-bound:     ~48 cycles (4 dep chains × 4)

For d=64 (4 words), with loop unrolling hiding latency:
  Typical (pipelined):    ~20-25 cycles per dot product per thread
  Worst-case (no unroll): ~80-90 cycles
  With warp-level parallelism (32 heads computed in parallel): 1-2 cycles amortized per head

Comparison: float32 dot product of 64 elements
  FMA: 64 FMA instructions, ~6-8 cycles each if pipelined = ~96-128 cycles
  Ternary XNOR+popcount: ~20-25 cycles = 4-5x speedup
```

---

## 3. Contrastive Margin Instead of Softmax

### 3.1 Why No Softmax

Softmax over K candidates requires:
1. `exp(s_i / sqrt(d))` per element — transcendental operation (~20-30 cycles per element on Ampere SFU, not the main ALU)
2. Sum reduction across K elements — inter-thread communication latency
3. Division by the sum — dependent on the reduction result

For K=64 candidates in the Nine-Chain Swarm yard, this is ~64 × 20 = 1280 cycles of SFU time just for the exp() calls, plus reduction overhead. This violates the sovereignty constraint (no transcendentals in hot path) and wastes hardware.

### 3.2 Contrastive Margin Formulation

At inference, the attention score is:

```
score(q, k_i) = ternary_dot(q, k_i)    // integer ∈ [-d, +d]
```

No division by `sqrt(d)` needed because the ternary dot product range is already bounded by `d` (dimension). The margin `m` establishes a minimum separation between selected and rejected candidates.

**Ranking criterion**: Candidate k_i is selected over k_j iff `score(q, k_i) > score(q, k_j)`.

**Margin gate**: After top-K selection, a candidate k_i is included only if `score(q, k_i) >= score(q, k_best) - m` where m is a hyperparameter (integer, suggested default: 8 for d=64, i.e., 12.5% of range).

**Training loss (contrastive pair-ranking)**:
```
L = max(0, m - (score(q, k_pos) - score(q, k_neg)))
```

where `k_pos` is a known-relevant candidate (positive pair) and `k_neg` is a known-irrelevant candidate (negative pair). This is the ternary realization of InfoNCE loss without the `exp()` denominator — it is structurally equivalent to the margin-based contrastive loss from the SimCLR family, but uses integer ternary dot products instead of cosine similarity over float32 vectors.

This maps to Christoph Dorn's semantic gravity contrastive principle: training teaches the system to attract semantically-related stars (+1) and repel contradictory ones (−1), with 0 for neutral-unknown pairs. The ternary force T(s₁,s₂) IS the attention weight in its equilibrium state.

### 3.3 Loss-at-Runtime Inference

At inference (no gradient, no training):

```
// Over K candidates from the Galaxy neighborhood (survivors of Frustum Cull):
FOR i IN [0, K):
    scores[i] = ternary_dot(q, k[i])   // integer

// Sort or partial-sort to get top-K (bitonic sort for K <= 32 in one warp):
top_k_indices = bitonic_topk(scores, K=8)   // 8-16 candidates typically
margin_threshold = scores[top_k_indices[0]] - m

// Margin gate: keep only candidates within margin of best
selected = [i for i in top_k_indices if scores[i] >= margin_threshold]
```

No exp(), no float normalization. Pure integer comparison. On sm_86:
```
Bitonic top-K for K=64, k=8 in one warp: ~100-150 cycles total
Integer comparison: 1 cycle
vs softmax over 64 elements: ~1280 cycles SFU + ~32 cycles reduction
Speedup: 8-12x
```

### 3.4 Expressiveness vs Standard Scaled Dot-Product Attention

Standard attention: continuous, differentiable, ordinal-aware.
Ternary contrastive attention: discrete, ordinal-preserving (ranking), lossy in magnitude.

The key insight: K3D does NOT need magnitude-continuous attention. The Galaxy is organized by semantic proximity — stars already carry their relevance signal in their spatial position (Morton octree) and embedding tier (Matryoshka prefix). Attention over the ternary embeddings is a second-pass refinement of candidates that already passed coarse filtering. The ranking signal (relative order of scores) is what matters, not the exact probability distribution.

**Where expressiveness is preserved**: The margin parameter m gives the system a tunable "attention width" — small m means near-winner-take-all (sharp), large m means soft inclusion (broad). This is the ternary equivalent of the temperature parameter in softmax.

**Where expressiveness is gained**: Ternary attention is sparse by construction — zero weights contribute nothing. This means the attention output is a sum over only the subset of V vectors corresponding to non-zero Q·K agreements. Sparse attention is faster AND reduces hallucination risk by not blending irrelevant context.

---

## 4. Value Mixing in the Yard

### 4.1 V Tile Layout

V vectors (values) reside in `yards[instance][bank][slot]` as `float4` tiles:

```
// For one attention head with d_head=64 and H key vectors:
// V tiles: yards[lane][bank_V][0..15]  -- 16 float4 slots = 64 floats = one d=64 V vector
// K tiles: registers (loaded from yard before dot product)
// Q vector: loaded from bank_Q, held in registers during scoring
// Output: written back to bank_O (bank_V reused if fused)

Bank assignment for one attention head:
  bank_Q = 0   (query vector)
  bank_K = 1   (key vectors, loaded one at a time)
  bank_V = 2   (value vectors, H slots deep)
  bank_O = 3   (output accumulator)
  bank_S = 4   (scores buffer, integer)
```

### 4.2 Ternary-Gated Value Mix

After top-K selection, the attention output is computed as a weighted sum of selected V vectors. Since attention weights are ternary-quantized (+1, 0, −1), the weighted sum degenerates to pure addition/subtraction:

```
output = zero vector

FOR selected_idx in top_k_selected:
    w = quantize_to_trit(score(q, k[selected_idx]))   // TQUANT → {-1, 0, +1}
    IF w == +1:
        output += V[selected_idx]    // float4 add
    ELIF w == -1:
        output -= V[selected_idx]    // float4 sub (subtract contradictory V)
    // w == 0: skip (no contribution)
```

This is NOT a full float32 matmul — it is repeated conditional float4 add/subtract gated by the trit value. On sm_86:
```
float4 ADD: 4 FADD, ~4 cycles latency, 1 cycle throughput
float4 SUB: same
Trit check: 1 ISETP (compare), 1 cycle
Per V vector: ~5-6 cycles
For top-8 selected out of K=64: ~48 cycles for the mix phase
```

### 4.3 Composition with Existing Opcodes

The value mix step uses:
- `TQUANT` (0x106) to map integer scores to {-1, 0, +1}
- `TERNARY_AND` (0x100) to gate +1 contributions
- `TERNARY_NOT` (0x102) + `TERNARY_AND` for −1 contributions (i.e., "negate-then-add")
- `YARD_PUSH_BANK` (0x171) to accumulate into bank_O
- `YARD_PEEK_ADDR` (0x173) for random-access V lookup by top-K index

The mix is ~80% existing primitives. Only the scoring step (ternary dot product) needs a new atomic opcode if cycle analysis shows the composition cost is prohibitive (see §6).

---

## 5. Composition, Not Monolithic Opcode

**Programs-before-opcodes principle** (K3D canonical): Build domain semantics as RPN programs over the existing opcode surface before adding new atomic opcodes. New opcodes are authorized only when composition cost is demonstrably wasteful on the target hardware.

The full attention head is an RPN program:

```
[load Q from bank_Q]
[load K_i from bank_K for each i]
  TERNARY_XNOR_POPCOUNT_WORD × 4     // or composed: AND + XOR + NOT + AND + POPC × 2 + SUB
  CONTRASTIVE_RANK_TOPK               // or composed: integer compare + bitonic sort
[for selected_i:]
  TQUANT                              // map score to {-1, 0, +1}
  TERNARY_AND / conditional YARD_POP  // gate V contribution
  YARD_PUSH_BANK bank_O               // accumulate output
```

Approximately 80% of this is expressible with:
- Existing yard ops: YARD_SELECT, YARD_PUSH_BANK, YARD_POP_BANK, YARD_PEEK_ADDR
- Existing ternary ops: TERNARY_AND, TERNARY_XOR, TERNARY_NOT, TQUANT, TCMP
- PTX primitives: popc.b32 (via inline PTX or a thin CUDA wrapper)

Only 2-4 new atomic opcodes are needed (see §6 and `attention_opcode_expansion.md`).

---

## 6. Two Scales: Context-Attention vs Retrieval-Attention

The same kernel, two invocations:

```
CONTEXT_ATTENTION (yard-local):
  Q, K, V: tiles within yards[instance][bank][0..68]
  Candidate set size K: 8-64 (single yard bank depth)
  Purpose: Attend over the current active reasoning context
           (Nine-Chain Swarm workers within one block)
  Tier used: tier_512 embeddings (fine-grained matching)

RETRIEVAL_ATTENTION (Galaxy neighborhood):
  Q: local query (from current tick)
  K, V: streamed in via global-queue slots (opcodes 0x178-0x17A)
  Candidate set size K: 64-512 (Galaxy neighborhood)
  Purpose: Attend over relevant Galaxy stars identified by LED-A*
           and Frustum Culling
  Tier used: tier_128 for scoring (fast), tier_512 for mix (fine)

Scale parameter: K (candidate set size) + tier_select
Everything else (the ternary dot product, the margin ranking, the V mix) is identical.
```

This is the ternary realization of the Matryoshka FOV/POV mechanism: the zoom level is expressed as the embedding tier used for K and V tiles, not as a different kernel.

---

## 7. Training vs Inference Lanes

```
TRAINING LANE:
  Weights: float32 Q, K, V projection matrices
  Kernel:  ATTENTION_FWD_BASE (0x1A7) — unchanged, kept per Daniel's ruling
  Loss:    Standard cross-entropy + contrastive pair-ranking margin loss
  Export:  At checkpoint boundary, apply TQUANT to project weights → {-1,0,+1}
           then pack into 2-bit trit format per §2.1

INFERENCE LANE (sovereign):
  Weights: ternary-packed Q, K, V projection matrices in 2-bit format
  Kernel:  ATTENTION_FWD_TERNARY (0x1A8)
  No exp(), no softmax, no float32 weight multiply
  Score:   ternary_dot + contrastive margin top-K
  Mix:     Ternary-gated float4 V addition

TQUANT AT EXPORT BOUNDARY:
  Location: checkpoint export script (ingestion path — float32 ok here)
  Operation: w_ternary = sign(w_float32) where |w_float32| > threshold
             threshold = absmean(|w_float32|) per BitNet b1.58 scheme
  No TQUANT at inference time for weight access (weights already packed)
  TQUANT IS used at inference time for score → weight mapping in V-mix step
```

Float32 weights never enter the sovereign hot path. Training is an ingestion-path activity; its outputs (ternary-packed weight tiles) are what the TRM game loop consumes.

---

## 8. Composition with Phase B Native Embedding

Phase B (`CLAUDE_CODEX_PHASE_B_NATIVE_EMBEDDING_04.18.2026.md`) produces 2048-dim meaning_rpn projections with the Matryoshka structure:

```
Dims 0-15:  Concept-class anchors (cross-modal, shared encoding for circle/cat/number)
Dims 0-63:  Coarse semantic class — tier_64
Dims 0-127: Structural category — tier_128
Dims 0-511: Fine concept matching — tier_512
Dims 0-2047: Full semantic fingerprint — tier_2048
```

Attention naturally implements the FOV/POV mechanism when paired with tier selection:

```
TIER_64 attention  → coarse FOV filter (is this region relevant at all?)
                     Matches Frustum Culling stage
TIER_128 attention → structural filter (is this the right domain shelf?)
                     Matches LED-A* path scoring
TIER_512 attention → fine candidate selection (which specific stars?)
                     Matches Nine-Chain Swarm worker scoring
TIER_2048 attention → deduplication / halting gate confirmation
                      Matches full-fidelity fingerprint at convergence check
```

The ternary Q·K dot product at tier_64 operates on 64-dim vectors = 4 uint32 words = 48 instructions total (4 words × 12 each). At tier_512 = 32 words = 384 instructions. Retrieval-attention at tier_128 = 8 words × 12 = 96 instructions per candidate. For a galaxy neighborhood of K=256 candidates at tier_128: 256 × 96 = 24,576 instructions across the warp (~768 cycles / 32 threads = 24 cycles amortized per candidate — well within the 16ms frame budget).

---

## 9. Acceptance Gates

All six gates below must pass before Codex declares `ATTENTION_FWD_TERNARY` production-ready.

### Gate 1 — No torch in attention paths
```bash
grep -rn "import torch\|from torch\|torch.nn\|torch.functional" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
    --include="*.py" --exclude-dir=Old_Attempts
# Expected: 0 lines
```

### Gate 2 — No softmax anywhere sovereign
```bash
grep -rn "torch.nn.functional.softmax\|F\.softmax\|softmax\b" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ knowledge3d/tablet/ \
    --include="*.py" --exclude-dir=Old_Attempts
# Expected: 0 lines
```

### Gate 3 — No exp() in sovereign attention kernel
```bash
grep -n "expf\|__expf\|exp2\.approx\|\.exp\b" \
    knowledge3d/cranium/kernels/attention_ternary.cu 2>/dev/null
# Expected: 0 lines (or file does not exist yet)
```

### Gate 4 — Ternary packing invariants
```python
# Test: round-trip packing preserves values
def test_trit_roundtrip():
    for val in [-1, 0, +1]:
        packed = pack_trit(val)   # {-1:0b00, 0:0b01, +1:0b10}
        assert packed in [0b00, 0b01, 0b10]
        assert unpack_trit(packed) == val

# Test: is_positive/is_negative/is_zero masks are mutually exclusive and cover all trits
def test_mask_partition():
    for w in [random.randint(0, 0xFFFFFFFF) for _ in range(1000)]:
        w = sanitize_packed_trits(w)  # clear 0b11 codes
        pos_mask = is_positive_mask(w)
        neg_mask = is_negative_mask(w)
        zer_mask = is_zero_mask(w)
        assert pos_mask & neg_mask == 0, "pos and neg must not overlap"
        assert pos_mask & zer_mask == 0, "pos and zero must not overlap"
        assert neg_mask & zer_mask == 0, "neg and zero must not overlap"
        assert (pos_mask | neg_mask | zer_mask) & 0x55555555 == 0x55555555, \
               "all trits accounted for"
```

### Gate 5 — Margin scoring Top-K matches float32 baseline (within epsilon)
```python
# Gold set: 100 query-key pairs with known float32 attention patterns
def test_margin_topk_matches_float32_baseline():
    for q, keys in gold_set:
        float32_topk = softmax_topk(q, keys, k=8)
        ternary_topk  = contrastive_margin_topk(quantize(q), quantize(keys), k=8, m=8)
        # Rank agreement: Kendall's tau >= 0.8
        tau = kendall_tau(float32_topk, ternary_topk)
        assert tau >= 0.8, f"Rank correlation too low: {tau}"
```

### Gate 6 — ATTENTION_FWD_BASE (0x1A7) still registered and not removed
```bash
grep -n "0x1A7\|ATTENTION_FWD\b\|ATTENTION_FWD_BASE" \
    docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md knowledge3d/cranium/ptx_runtime/rpn_opcodes.py
# Expected: ≥2 hits (one in registry, one in opcodes)
```

---

## 10. Must-NOT-Do List

- Do NOT add `exp()`, `log()`, `sqrt()` to the sovereign attention kernel path.
- Do NOT implement softmax as a "validation mode" — that path belongs to `ATTENTION_FWD_BASE` (0x1A7) which is the float32 training validator. It never runs in sovereign inference.
- Do NOT renumber or remove `ATTENTION_FWD` (0x1A7). It stays. The ternary variant is `ATTENTION_FWD_TERNARY` (0x1A8).
- Do NOT attempt to do Q·K with plain XNOR+popcount without the zero-mask correction — the formula produces wrong scores for zero trits (see §2.3).
- Do NOT move V tiles to a separate CUDA device array — they must live in the yard (shared memory) during the mix step for latency reasons.
- Do NOT implement attention as a monolithic kernel — use the RPN program approach in `reference_attention_rpn_program.md` so each step is independently testable and composable.
- Do NOT use `0b11` as a trit encoding — it is the unused/reserved code. TQUANT must map any `0b11` to +1 (or clamp) on input.
