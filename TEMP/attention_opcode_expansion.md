# Attention Opcode Expansion — Additive Registry Update

**Date**: 2026-04-18
**Author**: Claude (architecture, cuda-research-solver lane)
**Implementer**: Codex
**Principle**: EXPAND-NOT-REPLACE. Every entry below is ADDITIVE. No existing opcode is renumbered, removed, or modified. `ATTENTION_FWD_BASE` (0x1A7) is preserved exactly. This document adds 0x1A8, 0x108, and 0x1A9.

**Before implementing**: Codex MUST grep `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` for any existing assignments at 0x108, 0x1A8, 0x1A9. If collisions exist, use the next free slot in the respective range.

---

## 0. Existing Opcode Kept As-Is

### `ATTENTION_FWD_BASE` — 0x1A7

**Status**: PRESERVED UNCHANGED per Daniel's ruling ("do not replace, expand — change what's needed to make it happen").

**Purpose**: Single-head float32 attention. Training-lane validator. Debug path for comparing ternary vs float32 baseline scores. Used in acceptance gate 5 (rank correlation test).

**Not for sovereign inference**: `ATTENTION_FWD_BASE` must never appear in the sovereign hot path (inference, game loop, swarm scoring). It is training infrastructure. The sovereign inference path uses `ATTENTION_FWD_TERNARY` (0x1A8) exclusively.

**Grep check** (run before declaring this spec implemented):
```bash
grep -rn "0x1A7\|ATTENTION_FWD_BASE\|ATTENTION_FWD\b" \
    docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md \
    knowledge3d/cranium/ptx_runtime/rpn_opcodes.py
# → ≥2 hits: one in registry, one in opcodes
```

---

## 1. `ATTENTION_FWD_TERNARY` — 0x1A8

**Mnemonic**: `ATTENTION_FWD_TERNARY`
**Opcode**: `0x1A8`
**Range**: 0x1A0-0x1BF (bulk-lib purge new math range, next free after 0x1A7)
**Layer**: Composed head pipeline — attention scoring stage
**Priority**: P0 (required for Phase B+ sovereign inference)

### Binary Layout (Operand Encoding)

```
[opcode: 16 bits = 0x1A8]
[operand_0: 4 bits = bank_Q]       Q tile bank in yard
[operand_1: 4 bits = bank_K]       K tile bank in yard
[operand_2: 4 bits = bank_V]       V tile bank in yard
[operand_3: 4 bits = bank_O]       Output accumulator bank
[operand_4: 8 bits = N_candidates] Candidate count (up to 255)
[operand_5: 4 bits = K_topk]       Top-K selection count (1-15)
[operand_6: 8 bits = margin_m]     Contrastive margin (integer, default 8)
[operand_7: 4 bits = tier_select]  Embedding tier: 0=64, 1=128, 2=512, 3=2048
                                   (selects number of uint32 words: 4,8,32,128)
```

### Precondition (what must be in the yard before invocation)

```
yards[lane][bank_Q][0..15]         = Q vector, float4 format (d=64)
yards[lane][bank_Q][16..19]        = Q vector, ternary-packed (4 uint32 words)
yards[lane][bank_K][0..N*16)       = K matrix, float4 format
yards[lane][bank_K][N*16..N*20)    = K matrix, ternary-packed (N × 4 uint32 words)
yards[lane][bank_V][0..N*16)       = V matrix, float4 format
yards[lane][bank_O][0..15]         = Output accumulator, zero-initialized
```

If `tier_select` = 1 (tier_128, 128-dim vectors), then ternary form uses 8 words per vector, and float4 form uses 32 slots per vector. The bank must be sized accordingly.

### Postcondition (what is in the yard after)

```
yards[lane][bank_O][0..15]         = Attention output, d=64 float4, weighted V mix
yards[lane][bank_K][N*20..N*20+K_topk) = Selected candidate indices (integer)
yards[lane][bank_K][N*20+K_topk..N*20+K_topk*2) = Selected scores (integer, debug)
```

### Algorithm (what the kernel computes)

```
1. Load Q ternary words (4 words for tier_64, 8 for tier_128, etc.) from bank_Q
2. FOR i = 0 to N_candidates-1:
     a. Load K[i] ternary words from bank_K
     b. Compute dot_i = ternary_dot(Q, K[i]) using TERNARY_XNOR_POPCOUNT_WORD × n_words
     c. Write dot_i to score buffer (temporary register array or bank_O scratch)
3. top_k_idx = CONTRASTIVE_RANK_TOPK(scores, K_topk, margin_m)
4. Initialize output accumulator = 0
5. FOR selected_idx in top_k_idx:
     a. w = TQUANT(score[selected_idx])    // {-1, 0, +1}
     b. v = load_float4_tile(bank_V, selected_idx, 16 slots)
     c. IF w == +1: output += v
        IF w == -1: output -= v
        IF w ==  0: skip
6. Write output to bank_O
7. Write top_k_idx to bank_K[N*20..]
```

### Cycle Cost on sm_86 (RTX 3070)

```
N=64 candidates, K_topk=8, d=64 (tier_64):
  Step 2 scoring:  64 candidates × 7 instructions (with 0x108) = 448 instructions
                   ≈ 448 / 32 threads / 1.5 cycles avg = ~14 cycles amortized
  Step 3 top-K:    ~10 instructions ≈ 1 cycle amortized
  Step 5 V mix:    8 selected × ~100 instructions = 800 instructions
                   ≈ 800 / 32 = 25 cycles
  Total:           ~40 cycles per head (amortized, pipeline-saturated warp)

N=256, K_topk=16, d=128 (tier_128, retrieval-attention scale):
  Scoring:         256 × 11 instructions (8 words) = 2,816 ≈ 88 cycles
  Top-K:           ~10 cycles
  V mix:           16 × ~100 = 1,600 ≈ 50 cycles
  Total:           ~148 cycles per head
```

### RPN Invocation Examples

```rpn
; Context-attention (yard-local, tier_64, 8 candidates, top-3, margin=6):
PUSH 0  ; bank_Q
PUSH 1  ; bank_K
PUSH 2  ; bank_V
PUSH 3  ; bank_O
PUSH 8  ; N_candidates
PUSH 3  ; K_topk
PUSH 6  ; margin_m
PUSH 0  ; tier_select=0 → tier_64
0x1A8   ATTENTION_FWD_TERNARY

; Retrieval-attention (Galaxy neighborhood, tier_128, 64 candidates, top-8, margin=8):
PUSH 0 PUSH 1 PUSH 2 PUSH 3
PUSH 64 PUSH 8 PUSH 8 PUSH 1    ; tier_select=1 → tier_128
0x1A8   ATTENTION_FWD_TERNARY
```

---

## 2. `TERNARY_XNOR_POPCOUNT` — 0x108

**Mnemonic**: `TERNARY_XNOR_POPCOUNT`
**Opcode**: `0x108`
**Range**: 0x100-0x10F (ternary block, next free after TQUANT=0x106, TERNARY_XOR=0x107)
**Layer**: Ternary arithmetic substrate — micro-operation level
**Priority**: P0 (required by ATTENTION_FWD_TERNARY)

### Composition-vs-Atomic Decision

**Call**: Atomize. The composed alternative requires 12 instructions per word (AND×4, NOT×2, AND, XOR, AND, NOT, AND×2, POPC×2, SUB = 17 in the full-path version, ~12 in the optimized sign-plane version). For N=64 candidates × 4 words = 256 calls, the composition overhead is 256 × (12-1) = 2,816 extra instructions per attention head per tick. At 8 heads × ~60 TRM ticks/second, this is 8 × 60 × 2,816 = 1,351,680 extra instructions per second. Atomizing saves roughly 1.3M instructions/sec per lane — well above the cost-of-new-opcode threshold.

The composed path also has 4 dependency chains per word that cannot be hidden without unrolling, adding latency pressure. The atomic opcode's implementation can use PTX's native `popc.b32` with compiler-level scheduling.

### Binary Layout

```
[opcode: 16 bits = 0x108]
[operand_0: implicit] = top of active bank (q_word, uint32, ternary-packed)
[operand_1: implicit] = second element of active bank (k_word, uint32, ternary-packed)
Result: pops q_word and k_word, pushes partial_dot (int32) ∈ [-16, +16]
```

Both operands come from the active bank (stack-based, like all tier-1 ops). This is consistent with TERNARY_XOR (0x107), TERNARY_AND (0x100), etc.

### Precondition

```
Active bank stack (top-down):  [..., k_word_uint32, q_word_uint32]
Both words: ternary-packed, 2-bit per trit, encoding 0b10=+1, 0b01=0, 0b00=-1
Code 0b11 is treated as +1 (forward compatibility clamp)
```

### Postcondition

```
Active bank stack (top-down):  [..., partial_dot_int32]
partial_dot ∈ [-16, +16]   (one uint32 holds 16 trits, so range is [-16, +16])
```

### Algorithm (C pseudo-code for Codex to implement in CUDA/PTX)

```c
__device__ int32_t ternary_xnor_popcount_word(uint32_t q, uint32_t k) {
    // Sign-plane method (§2.5 of ternary_contrastive_attention_design.md)
    // Step 1: Extract sign planes (high bit of each 2-bit pair, at even positions after shift)
    uint32_t q_sign = (q & 0xAAAAAAAAu) >> 1;   // sign bits at even positions
    uint32_t k_sign = (k & 0xAAAAAAAAu) >> 1;

    // Step 2: Non-zero masks (bit 0 of each pair is 1 only for encoding 0b01 = trit 0)
    // zero_bit = bit0 AND NOT bit1, extracting to even positions
    uint32_t q_nz = ~(q & 0x55555555u) & 0x55555555u;   // 1 where q != 0
    uint32_t k_nz = ~(k & 0x55555555u) & 0x55555555u;   // 1 where k != 0
    uint32_t both_nz = q_nz & k_nz;                       // 1 where both non-zero

    // Step 3: Sign agreement / disagreement, masked to non-zero positions
    uint32_t xor_sign   = (q_sign ^ k_sign) & both_nz;    // 1 where signs differ AND both nz
    uint32_t match      = (~xor_sign) & both_nz & 0x55555555u; // 1 where signs match AND both nz

    // Step 4: Dot product = matches - mismatches
    int32_t agree    = __popc(match);
    int32_t disagree = __popc(xor_sign);
    return agree - disagree;
}
```

**PTX Note**: `__popc(x)` compiles to `popc.b32 dst, src` — a single-issue instruction on sm_86, 4-cycle latency, 1-cycle throughput. The full function body above is 12 instructions, maps to a single PTX inline function that the compiler can schedule aggressively.

### Cycle Cost on sm_86

```
7 bitwise instructions: AND×4, NOT×2, XOR×1 = 7 × 1 cycle throughput = 7 cycles (pipelined)
2 popc.b32: 2 × 1 cycle throughput = 2 cycles
1 SUB.s32: 1 cycle
Total throughput: ~10 cycles (latency-bound: ~20 cycles without unrolling)
Effective amortized cost (4 dep chains hidden by unrolling): ~12 cycles per word
```

### RPN Invocation Examples

```rpn
; Compute partial dot product for one word pair:
PUSH q_word_uint32        ; push packed Q trit word
PUSH k_word_uint32        ; push packed K trit word
0x108  TERNARY_XNOR_POPCOUNT   ; pops both, pushes int32 partial_dot ∈ [-16,+16]

; Full d=64 dot product (4 words, unrolled):
PUSH q0 PUSH k0   0x108  ; partial_0
PUSH q1 PUSH k1   0x108  ; partial_1
PUSH q2 PUSH k2   0x108  ; partial_2
PUSH q3 PUSH k3   0x108  ; partial_3
ADD ADD ADD               ; sum all partials → dot ∈ [-64, +64]

; d=128 dot product (8 words):
PUSH q0 PUSH k0 0x108  PUSH q1 PUSH k1 0x108  ADD
PUSH q2 PUSH k2 0x108  PUSH q3 PUSH k3 0x108  ADD
ADD
PUSH q4 PUSH k4 0x108  PUSH q5 PUSH k5 0x108  ADD
PUSH q6 PUSH k6 0x108  PUSH q7 PUSH k7 0x108  ADD
ADD
ADD   ; final sum → dot ∈ [-128, +128]
```

---

## 3. `CONTRASTIVE_RANK_TOPK` — 0x1A9

**Mnemonic**: `CONTRASTIVE_RANK_TOPK`
**Opcode**: `0x1A9`
**Range**: 0x1A0-0x1BF (next free after ATTENTION_FWD_TERNARY=0x1A8)
**Layer**: Ranking / selection — replaces softmax probability ranking
**Priority**: P0 (required by ATTENTION_FWD_TERNARY)

### Composition-vs-Atomic Decision

**Call**: Atomize. The composed alternative requires ~200 instructions (max-find via REDUCE, threshold compute, linear scan with conditional push, bitonic sort). For the attention use case called 8×60=480 times/second, saving 190 instructions × 480 = 91,200 instructions/second. Marginal on its own, but the composed path also requires temporary scratch space in the active bank that competes with the scoring loop. The atomic opcode has a cleaner precondition (scores already in a dedicated bank, no interference with scoring) and enables Codex to implement a purpose-optimized warp-level bitonic topK.

### Binary Layout

```
[opcode: 16 bits = 0x1A9]
[operand_0: implicit] = bank containing N score values (int32)
[operand_1: 8 bits]   = K_topk (number of candidates to select, 1-255)
[operand_2: 8 bits]   = margin_m (integer margin threshold)
Result: Replaces bank contents with K_topk selected INDICES (not scores)
        in descending score order, margin-gated.
```

### Precondition

```
Active bank (bank_S): [score_0, score_1, ..., score_{N-1}]  (int32 values)
N = current stack pointer (sp[bank_S]) — number of elements in bank
K_topk ≤ N
margin_m ≥ 0
```

### Postcondition

```
Active bank (bank_S): [idx_0, idx_1, ..., idx_{k-1}]  (int32 indices, k ≤ K_topk)
  - idx_0 is the index of the highest-scoring candidate
  - All indices satisfy: score[idx_i] >= score[idx_0] - margin_m
  - k ≤ K_topk (may be fewer if margin gate excludes some candidates)
sp[bank_S] = k (stack pointer set to number of selected indices)
```

### Algorithm

```c
// Warp-cooperative partial sort + margin gate
// For N ≤ 64 (fits in shared memory): bitonic sort then linear scan
// For N > 64: register-sorted top-K with heap
__device__ void contrastive_rank_topk(
    int32_t* scores, int N,           // scores buffer in shared memory
    int32_t* output_indices, int* k,  // output
    int K_topk, int margin_m
) {
    // Step 1: Warp-parallel argmax over scores
    //         Using warp shuffle reduction — log2(32) = 5 steps, 5 cycles
    int32_t best_score = warp_reduce_max(scores, N);

    // Step 2: Margin threshold
    int32_t threshold = best_score - margin_m;

    // Step 3: Count and collect qualifying indices
    //         (Warp-coalesced scan with ballot/prefix-sum)
    int qualifying_count = 0;
    for (int i = threadIdx.x; i < N; i += 32) {
        bool qualifies = (scores[i] >= threshold);
        uint32_t mask = __ballot_sync(0xFFFFFFFF, qualifies);
        // Warp prefix sum to assign output positions
        int lane = threadIdx.x & 31;
        int pos  = __popc(mask & ((1u << lane) - 1)) + qualifying_count;
        if (qualifies && pos < K_topk) output_indices[pos] = i;
        qualifying_count += __popc(mask);
    }

    // Step 4: Bitonic sort output_indices by score descending (for top-K ordering)
    bitonic_sort_by_score(output_indices, min(qualifying_count, K_topk), scores);

    *k = min(qualifying_count, K_topk);
}
```

### Cycle Cost on sm_86

```
N=64, K_topk=8, warp-cooperative:
  Step 1 warp reduce max:  5 warp shuffle steps ≈ 5 × 4 cycles = 20 cycles
  Step 2 threshold:        1 SUB, 1 cycle
  Step 3 ballot+prefix:    64/32 = 2 iterations × ~10 cycles = 20 cycles
  Step 4 bitonic sort K=8: O(K log² K) = 8×9 = ~72 operations ≈ 10 cycles
  Total: ~51 cycles

N=256, K_topk=16:
  Step 1-2:  ~25 cycles
  Step 3:    256/32 = 8 iterations × 10 = 80 cycles
  Step 4:    K=16 bitonic: ~15 cycles
  Total: ~120 cycles
```

### RPN Invocation Examples

```rpn
; Select top-8 from 64 scores in bank_4, margin=8:
0x170  YARD_SELECT    bank_id=4     ; activate score bank
PUSH   8                            ; K_topk=8
PUSH   8                            ; margin_m=8
0x1A9  CONTRASTIVE_RANK_TOPK        ; pops K_topk and margin, replaces bank with indices

; Select top-4 from 256 scores in bank_5, margin=12:
0x170  YARD_SELECT    bank_id=5
PUSH   4   PUSH   12
0x1A9  CONTRASTIVE_RANK_TOPK        ; bank_5 now has ≤4 selected indices
```

---

## 4. Summary Table — All New Opcodes

| Opcode | Mnemonic | Range | Priority | Replaces | Status |
|--------|----------|-------|----------|----------|--------|
| `0x1A7` | `ATTENTION_FWD_BASE` | 0x1A0-0x1BF | — | Kept as-is | Existing, preserved |
| `0x1A8` | `ATTENTION_FWD_TERNARY` | 0x1A0-0x1BF | P0 | Softmax float32 attention in sovereign path | NEW |
| `0x108` | `TERNARY_XNOR_POPCOUNT` | 0x100-0x10F | P0 | Composed AND+XOR+NOT+POPC sequence | NEW |
| `0x1A9` | `CONTRASTIVE_RANK_TOPK` | 0x1A0-0x1BF | P0 | Softmax top-K; composed bitonic sort sequence | NEW |

Total new opcodes: **3** (target was ≤4 — satisfied).

---

## 5. Registry Update — Codex Must Add to `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md`

Codex adds the following three entries to the registry before implementing anything. The registry update is the first commit in the attention implementation sequence.

```markdown
### 0x108 — TERNARY_XNOR_POPCOUNT
**Category**: Ternary arithmetic (0x100-0x10F range)
**Inputs**: top(active_bank) = q_word: uint32; second(active_bank) = k_word: uint32
**Output**: partial_dot: int32 ∈ [-16, +16]
**Semantics**: Ternary dot product over 16 packed trits in two uint32 words.
Encoding: 0b10=+1, 0b01=0, 0b00=-1. Zero-mass correction applied via sign-plane
method (see TEMP/ternary_contrastive_attention_design.md §2.5). Zero trits do not
contribute to the dot product. Result = count(sign_match AND both_nonzero) - count(sign_mismatch AND both_nonzero).
**Hardware**: sm_86 (Ampere), uses popc.b32 PTX intrinsic. ~12 cycles/word throughput.
**Date added**: 2026-04-18

### 0x1A8 — ATTENTION_FWD_TERNARY
**Category**: Attention / composed head (0x1A0-0x1BF range)
**Inputs**: operands encode bank_Q, bank_K, bank_V, bank_O, N_candidates, K_topk, margin_m, tier_select
**Output**: Attention output in bank_O; selected indices in bank_K scratch area
**Semantics**: Single-head ternary attention. Q, K scored via TERNARY_XNOR_POPCOUNT × n_words.
Top-K selected via CONTRASTIVE_RANK_TOPK. V mixed via TQUANT-gated float4 add/sub.
No softmax, no exp(). Sovereign inference only. For training path use ATTENTION_FWD_BASE (0x1A7).
Supports two scales: context-attention (yard-local) and retrieval-attention (Galaxy neighborhood
via global queue). Scale is K × tier_select parameter, not a different opcode.
**Hardware**: sm_86. ~40 cycles/head amortized for N=64, d=64, K_topk=8.
**Date added**: 2026-04-18

### 0x1A9 — CONTRASTIVE_RANK_TOPK
**Category**: Ranking / selection (0x1A0-0x1BF range)
**Inputs**: active_bank = N integer scores; K_topk: uint8; margin_m: uint8
**Output**: Active bank replaced with ≤K_topk selected indices (descending score order)
**Semantics**: Margin-gated top-K selection. Selects up to K_topk candidates whose score
is within margin_m of the best score. Implements the contrastive margin inference criterion:
no exp(), no softmax. Warp-cooperative bitonic sort internally.
Conceptually: the ternary inference realization of InfoNCE ranking without the denominator.
Replaces softmax top-K in the sovereign attention pipeline.
**Hardware**: sm_86. ~51 cycles for N=64, ~120 cycles for N=256. Warp-cooperative.
**Date added**: 2026-04-18
```

---

## 6. Implementation Order for Codex

```
1. Add registry entries (§5) to RPN_DOMAIN_OPCODE_REGISTRY.md — first commit.
2. Add opcode constants to knowledge3d/cranium/ptx_runtime/rpn_opcodes.py:
   TERNARY_XNOR_POPCOUNT = 0x108
   ATTENTION_FWD_TERNARY  = 0x1A8
   CONTRASTIVE_RANK_TOPK  = 0x1A9
3. Implement ternary_xnor_popcount_word() in CUDA device function:
   File: knowledge3d/cranium/kernels/ternary_attention_kernels.cu
   Test: standalone unit test for the 9 input cases in the truth table (§2.3 of design doc).
4. Implement contrastive_rank_topk() in the same file.
   Test: verify top-K indices match brute-force sort for random score arrays.
5. Implement ATTENTION_FWD_TERNARY kernel composing both above:
   File: knowledge3d/cranium/kernels/ternary_attention_kernels.cu (same file)
   This is the full pipeline per reference_attention_rpn_program.md.
6. Compile: nvcc -arch=sm_86 -ptx -o knowledge3d/cranium/ptx/ternary_attention.ptx
            knowledge3d/cranium/kernels/ternary_attention_kernels.cu
7. Write ctypes bridge: knowledge3d/cranium/bridges/ternary_attention_bridge.py
8. Run acceptance gates 1-6 from ternary_contrastive_attention_design.md §9.
9. Run rank-correlation test (gate 5) against ATTENTION_FWD_BASE float32 output.
10. Report all gate results with grep evidence.
```

---

## 7. Must-NOT-Do

- Do NOT implement `TERNARY_XNOR_POPCOUNT` as a Python function — it is a CUDA device function called from within the PTX kernel, not a Python-side operation.
- Do NOT add `__syncthreads()` inside `ternary_xnor_popcount_word()` — it is a pure register-level function with no shared memory access.
- Do NOT use `mma.sync` (Tensor Core) for the ternary dot product even though PTX supports `mma.sync.xor.popc` for binary weights — the 2-bit ternary encoding is NOT compatible with Tensor Core b1.xor.popc format (which requires 1-bit packing). Stick with scalar popc.b32.
- Do NOT implement `CONTRASTIVE_RANK_TOPK` with `thrust::sort` or any host-side call — it must be a device-side warp-cooperative operation.
- Do NOT use `0b11` (binary 3) as a valid trit code in test inputs — it is the undefined code and `TERNARY_XNOR_POPCOUNT` treats it as +1. Test only with valid codes {0b00, 0b01, 0b10}.
- Do NOT remove or modify the signature of `ATTENTION_FWD_BASE` (0x1A7) anywhere. Existing callers in training pipelines depend on it.
- Do NOT raise `TERNARY_XNOR_POPCOUNT` out of the 0x100-0x10F ternary block — future ternary opcodes 0x109-0x10F are reserved for the ternary arithmetic family. This opcode belongs there.
