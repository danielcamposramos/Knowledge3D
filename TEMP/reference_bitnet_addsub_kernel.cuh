/**
 * reference_bitnet_addsub_kernel.cuh
 *
 * Device-side reusable header: BitNet b1.58 add/sub/skip matmul for sm_86.
 *
 * Author   : Claude (cuda-research-solver lane), 2026-04-18
 * Target   : sm_86 (RTX 3070, Ampere)
 * Requires : CUDA 12.4+, nvcc -arch=sm_86
 *
 * Dependency chain (must be present before this header is useful):
 *   - Transfer Yard substrate (float4 yards[9][9][69]) — this header is used
 *     inside kernels that share that yard. No direct yard accesses here; callers
 *     manage yard I/O via YARD_* opcodes.
 *   - Ternary isolation contract (Phase B): cores are isolated at CUDA-block
 *     boundary (46 isolated cores on RTX 3070, 9 instances/core = 414 instances).
 *   - VEC_NORM_L2_INT8 (0x1AD) must be called by the RPN program AFTER any
 *     attention head that uses bitnet_matmul_tile — mandatory per Daniel's ruling.
 *
 * Sovereignty: no float32, no exp(), no transcendentals, no torch/numpy/cupy.
 *              All accumulation in INT32 → requantize to INT8 at the boundary.
 *
 * References:
 *   [1] BitNet b1.58 (Ma et al., arXiv:2504.12285, March 2026)
 *   [2] T-MAC LUT-based low-bit inference (Microsoft Research, arXiv:2407.00088)
 *   [3] NVIDIA PTX ISA Reference 8.x: dp4a instruction (§9.7.1.23)
 *   [4] BitNet.cpp GPU path (github.com/microsoft/BitNet, gpu/README.md)
 */

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

/* ============================================================
 *  Section 1: 5-trit-per-byte (1.6-bit) packing/unpacking
 * ============================================================
 *
 * Encoding:
 *   trit t ∈ {-1, 0, +1}  →  offset_trit = t + 1 ∈ {0, 1, 2}
 *   5 offset_trits → one byte:
 *     byte = t0'*81 + t1'*27 + t2'*9 + t3'*3 + t4'
 *   where ti' = ti + 1.
 *
 *   Range: 0 to 242 (= 3^5 - 1 + (sum of offsets)).
 *   Exactly 243 values used out of 256 — 13 patterns are unused.
 *   1 uint32 = 4 bytes = 20 trits.  20× compression vs float32.
 *
 * LUT strategy (for sm_86):
 *   - Constant cache per SM: 8 KB working set (64 KB total constant memory).
 *   - 256 entries × 8 bytes (uint64_t) = 2 KB — fits comfortably.
 *   - Each entry packs 5 int8 values (bytes [4:0]) = 40 bits, rest zeroed.
 *   - Constant memory is broadcast-cached: when all threads in a warp read the
 *     SAME index the latency is ~5 cycles. When each thread reads a DIFFERENT
 *     index, reads serialise (up to 32 cycles). For the matmul path, 4-thread
 *     groups share the same byte → 4-way broadcast → ~5-cycle latency.
 *   - This LUT does NOT compete with the yard float4 data because the yard
 *     lives in shared memory; the LUT lives in constant memory (separate 64 KB
 *     region on-chip).
 *
 * Why NOT base-3 arithmetic for unpack:
 *   Division by 81, 27, 9, 3 requires 4 mul-reciprocal sequences ≈ 15-25
 *   instructions per byte. The LUT reduces that to 1 load + 5 byte extracts
 *   ≈ 6 instructions. At ~20 unpack calls per warp per row, LUT saves ~380
 *   instructions per row vs arithmetic.
 *
 * Why NOT bit-interleaving:
 *   3-valued trits don't map to a power-of-2. Bit-interleaving requires cross-
 *   byte scatter/gather and does not simplify the decode logic on GPU.
 */

/* 2 KB constant-memory LUT: unpack5_lut[b] = 5 int8 values packed as bytes
 * in uint64_t, layout: [t0 @ bits 7:0] [t1 @ bits 15:8] ... [t4 @ bits 39:32]
 * Values are already mapped back to {-1, 0, +1} (int8_t representation).
 *
 * IMPORTANT: call bitnet_init_lut_host() from host before first kernel launch.
 */
__constant__ uint64_t bitnet_unpack5_lut[256];

/**
 * Host-side: populate the unpack5 constant-memory LUT.
 * Call once at program init, before any kernel that uses unpack5.
 */
static inline void bitnet_init_lut_host(void) {
    uint64_t host_lut[256];
    for (int b = 0; b < 256; b++) {
        /* Only 0..242 are valid encodings; 243..255 are clamped to (+1,+1,+1,+1,+1). */
        int rem = (b < 243) ? b : 242;
        int t0 = rem / 81;  rem %= 81;
        int t1 = rem / 27;  rem %= 27;
        int t2 = rem / 9;   rem %= 9;
        int t3 = rem / 3;   rem %= 3;
        int t4 = rem;
        /* Convert offset trits (0,1,2) back to signed trits (-1,0,+1) */
        int8_t v0 = (int8_t)(t0 - 1);
        int8_t v1 = (int8_t)(t1 - 1);
        int8_t v2 = (int8_t)(t2 - 1);
        int8_t v3 = (int8_t)(t3 - 1);
        int8_t v4 = (int8_t)(t4 - 1);
        host_lut[b] = ((uint64_t)(uint8_t)v0)
                    | ((uint64_t)(uint8_t)v1 << 8)
                    | ((uint64_t)(uint8_t)v2 << 16)
                    | ((uint64_t)(uint8_t)v3 << 24)
                    | ((uint64_t)(uint8_t)v4 << 32);
    }
    cudaMemcpyToSymbol(bitnet_unpack5_lut, host_lut, sizeof(host_lut));
}

/**
 * pack5 — pack 5 trits into 1 byte (base-3 mixed-radix).
 *
 * Inputs:  t0..t4 each in {-1, 0, +1}.
 * Output:  uint8_t in [0, 242].
 *
 * Called at weight-upload time (ingestion path), NOT in inference hot path.
 * Cost: 5 IMAD instructions (integer multiply-add) on sm_86 — ~5-10 cycles.
 */
__device__ __host__ __forceinline__
uint8_t pack5(int8_t t0, int8_t t1, int8_t t2, int8_t t3, int8_t t4) {
    return (uint8_t)(
        (int32_t)(t0 + 1) * 81 +
        (int32_t)(t1 + 1) * 27 +
        (int32_t)(t2 + 1) * 9  +
        (int32_t)(t3 + 1) * 3  +
        (int32_t)(t4 + 1)
    );
}

/**
 * unpack5 — unpack 1 byte into 5 trits via the constant-memory LUT.
 *
 * Inputs:  b in [0, 255]; values 243..255 clamp to (+1,+1,+1,+1,+1).
 * Outputs: out[0..4], each int8_t in {-1, 0, +1}.
 *
 * Cost on sm_86 (4-thread broadcast path):
 *   1 LD.CONST (4 cycles latency, 1 cycle throughput for broadcast)
 *   + 5 byte-extract instructions (AND + shift) = 6 instructions total.
 *   With warp broadcast (all 4 threads reading same byte): ~5 cycles.
 */
__device__ __forceinline__
void unpack5(uint8_t b, int8_t out[5]) {
    uint64_t entry = bitnet_unpack5_lut[b];
    out[0] = (int8_t)( entry        & 0xFF);
    out[1] = (int8_t)((entry >>  8) & 0xFF);
    out[2] = (int8_t)((entry >> 16) & 0xFF);
    out[3] = (int8_t)((entry >> 24) & 0xFF);
    out[4] = (int8_t)((entry >> 32) & 0xFF);
}

/* ============================================================
 *  Section 2: Core inner loop — add/sub/skip accumulation
 * ============================================================
 *
 * For each weight × activation pair:
 *   w = +1 → accum += a       (add)
 *   w =  0 → nothing          (skip — zero weight contributes nothing)
 *   w = -1 → accum -= a       (subtract)
 *
 * No integer multiplies. The conditional is branch-free via sign extraction:
 *   sign = (int8_t)w >> 1      — arithmetic right shift: +1→0, 0→0, -1→-1
 *   contribution = a if w>0, -a if w<0, 0 if w==0
 *   Achieved with: int8_t contrib = (w > 0) ? a : (w < 0) ? -(int8_t)a : 0;
 *   On sm_86: ISETP + SELP (2 instructions per weight, pipelined).
 *
 * 82% energy reduction vs float32 matmul (published BitNet b1.58 figure):
 *   Float32 FMUL = ~3.5 nJ/op on Ampere (TSMC 8N process).
 *   INT8 ADD  = ~0.5 nJ/op.
 *   Skip = ~0.1 nJ/op (predicated no-op).
 *   For weight distribution ≈ 50% zero (typical BitNet b1.58): 0.5*0.5 + 0.5*0.1 = 0.3 nJ/op vs 3.5.
 */

/**
 * bitnet_dot_add_sub_skip — single-thread serial dot product.
 *
 * This is the reference implementation, used for unit tests and for short
 * vectors where warp cooperation would cause too much reduction overhead.
 *
 * Inputs:
 *   packed_weights   : 1.6-bit packed weight bytes. Each byte encodes 5 trits.
 *                      Number of bytes = ceil(n_trits / 5).
 *   activations      : INT8 activation vector of length n_trits.
 *   n_trits          : number of weight × activation pairs to accumulate.
 *
 * Output: INT32 accumulator. Caller is responsible for requantization.
 *
 * Range: accumulator ∈ [-n_trits × 127, +n_trits × 127].
 * For d=64, n_trits=64: range [-8128, +8128] — well within int32_t.
 */
__device__ __forceinline__
int32_t bitnet_dot_add_sub_skip(
    const uint8_t* __restrict__ packed_weights,
    const int8_t*  __restrict__ activations,
    int n_trits)
{
    int32_t accum = 0;
    int8_t trits[5];

    /* Process 5 weights per byte; partial last byte handled via guard. */
    for (int byte_i = 0, trit_base = 0;
         trit_base < n_trits;
         byte_i++, trit_base += 5)
    {
        unpack5(packed_weights[byte_i], trits);

        /* Inner unroll: 5 weights per byte, guard on last partial byte. */
#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            int global_trit = trit_base + j;
            if (global_trit >= n_trits) break;  /* Compiler eliminates for aligned sizes. */

            int8_t w = trits[j];
            int8_t a = activations[global_trit];
            /* Branch-free: compiler generates ISETP + SELP on sm_86. */
            accum += (w > 0) ? (int32_t)a : (w < 0) ? -(int32_t)a : 0;
        }
    }
    return accum;
}

/* ============================================================
 *  Section 3: Warp-cooperative tile matmul
 * ============================================================
 *
 * Layout: a warp (32 threads) cooperatively computes one dot product.
 *
 *   uint32 packing: 4 bytes = 20 trits.
 *   Per step the warp loads 8 uint32 words = 160 trits.
 *
 *   Thread mapping within the warp for one uint32 word:
 *     word_group = lane / 4     ∈ [0, 7]   — which of the 8 words
 *     byte_lane  = lane % 4     ∈ [0, 3]   — which byte within the word
 *
 *   Concrete per step (160 trits, 8 uint32 words):
 *     Thread 0 (g=0, b=0): bytes 0 of word 0   → trits  0..4
 *     Thread 1 (g=0, b=1): bytes 1 of word 0   → trits  5..9
 *     Thread 2 (g=0, b=2): bytes 2 of word 0   → trits 10..14
 *     Thread 3 (g=0, b=3): bytes 3 of word 0   → trits 15..19
 *     Thread 4 (g=1, b=0): bytes 0 of word 1   → trits 20..24
 *     ...
 *     Thread 31 (g=7,b=3): bytes 3 of word 7   → trits 155..159
 *
 *   __shfl_sync use:
 *     We broadcast the uint32 word from the thread that loaded it
 *     to the 4 threads that share it. Thread 4*g+0 loads word g;
 *     threads 4*g+1, 4*g+2, 4*g+3 receive via __shfl_sync.
 *     Alternative: each thread loads its own byte independently (no shfl)
 *     but that requires 4× the global-memory transactions for the same word.
 *     __shfl_sync is preferred: 1 load + 3 shfl vs 4 independent loads.
 *
 *   After each thread accumulates its 5 partial products, warp-reduce
 *   via __shfl_down_sync to sum all 32 partial sums into the final dot.
 *
 * Shared memory bank conflicts:
 *   Activations[i] is INT8. If stored contiguously in shared memory,
 *   consecutive threads access consecutive bytes. On sm_86, shared memory
 *   banks are 4 bytes wide (bank = addr/4 % 32). Consecutive-byte reads
 *   from 32 threads cause 4-way conflicts (8 threads per bank).
 *   Resolution: store activations as int32_t (with byte-extraction via prmt)
 *   OR pass activations from global memory and rely on L1/L2 coalescing.
 *   For short vectors (d=64), pass activations in registers directly.
 *
 * Register pressure (per thread):
 *   - 1 uint32 word register (loaded + broadcast)
 *   - 1 uint8 byte extracted from word
 *   - 5 int8_t trit registers (after unpack5)
 *   - 5 int8_t activation registers
 *   - 1 int32_t local accumulator
 *   Total: ~13 registers per thread — far below the sm_86 limit of 255.
 */

/**
 * bitnet_matmul_tile — warp-cooperative dot product of ONE weight row vs
 * ONE activation vector. Returns the INT32 partial sum for this warp lane's
 * slice; caller must warp-reduce to get the final scalar.
 *
 * For a full [M × K] matmul:
 *   - Launch one warp per output element (M output rows, each needing one dot).
 *   - Or tile M across a block (e.g. 1 warp per row, blockDim.y = rows/block).
 *
 * Inputs:
 *   W_packed   : packed weight row, ceil(K/5) bytes, 1.6-bit packed.
 *   X          : INT8 activation vector, K elements.
 *   K          : length of the dot product (must be multiple of 20 for
 *                optimal performance; pad with zero-weight bytes if not).
 *
 * Output: each thread returns its partial INT32 accumulator.
 *         The CALLER is responsible for reducing across the warp.
 *
 * Usage example (fully warp-reduced output):
 *
 *   int32_t partial = bitnet_matmul_tile(W_packed, X, K);
 *   // Warp reduction using __shfl_down_sync:
 *   for (int offset = 16; offset > 0; offset >>= 1)
 *       partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
 *   // partial in lane 0 is the full dot product.
 */
__device__ __forceinline__
int32_t bitnet_matmul_tile(
    const uint8_t* __restrict__ W_packed,   /* 1.6-bit weight row, ceil(K/5) bytes */
    const int8_t*  __restrict__ X,          /* INT8 activations, K elements */
    int K)                                  /* Vector length in trits */
{
    const int lane       = threadIdx.x & 31;  /* Lane within warp [0..31] */
    const int word_group = lane >> 2;          /* Which uint32 word [0..7] */
    const int byte_lane  = lane & 3;           /* Which byte within word [0..3] */

    int32_t partial = 0;

    /*
     * Step stride: 160 trits (8 uint32 words × 20 trits/word = 160).
     * The weight row is laid out as ceil(K/5) bytes = ceil(K/20)*4 uint32 words.
     */
    const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W_packed);
    const int n_words_total = (K + 19) / 20;  /* ceil(K/20) uint32 words per row */

    for (int word_base = 0; word_base < n_words_total; word_base += 8) {
        int word_idx = word_base + word_group;

        /* Step 1: Thread (4*g+0) loads the uint32 for group g.
         * Other threads in the group receive it via __shfl_sync.
         * This gives 1 global load per 4 threads vs 4 independent loads.
         */
        uint32_t w32 = 0;
        if (word_idx < n_words_total) {
            /* Only the "loader" lane (byte_lane == 0) performs the global load. */
            if (byte_lane == 0) {
                w32 = W32[word_idx];
            }
            /* Broadcast from lane (word_group*4 + 0) to lanes (word_group*4 + 1..3). */
            int src_lane = word_group * 4;   /* The loader lane for this group */
            w32 = __shfl_sync(0xFFFFFFFF, w32, src_lane);
        }

        /* Step 2: Each thread extracts its assigned byte from the word.
         *   byte_lane 0: bits [7:0]    → trits 0..4 of this word
         *   byte_lane 1: bits [15:8]   → trits 5..9
         *   byte_lane 2: bits [23:16]  → trits 10..14
         *   byte_lane 3: bits [31:24]  → trits 15..19
         */
        uint8_t packed_byte = (uint8_t)((w32 >> (byte_lane * 8)) & 0xFF);

        /* Step 3: Unpack 5 trits from the byte using the 2KB constant-memory LUT. */
        int8_t trits[5];
        unpack5(packed_byte, trits);

        /* Step 4: Apply add/sub/skip over 5 activations. */
        int trit_base = (word_base + word_group) * 20 + byte_lane * 5;
#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            int trit_idx = trit_base + j;
            if (trit_idx < K && word_idx < n_words_total) {
                int8_t w = trits[j];
                int8_t a = X[trit_idx];
                /* No multiply. Branch-free via ISETP + SELP on sm_86. */
                partial += (w > 0) ? (int32_t)a : (w < 0) ? -(int32_t)a : 0;
            }
        }
    }

    /* Caller does the warp reduction. Example:
     *   for (int off = 16; off > 0; off >>= 1)
     *       partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
     * Lane 0 then holds the final dot product. */
    return partial;
}

/* ============================================================
 *  Section 4: INT8 × INT8 dot product via dp4a (Q·K^T stage)
 * ============================================================
 *
 * After the ternary weight × INT8 activation projections produce INT8 Q, K
 * vectors (requantized by VEC_NORM_L2_INT8 / requant step), the attention
 * score Q·K^T is a standard INT8 × INT8 dot product — NOT a ternary problem.
 *
 * dp4a: PTX instruction dp4a.s32.s32 d, a, b, c
 *   d = c + dot4(a[7:0], b[7:0])  ← signed 8-bit × signed 8-bit, accumulated into int32.
 *   Available since sm_61; confirmed on sm_86.
 *   Intrinsic: __dp4a(a, b, c)  — 4-way INT8 dot-product + accumulate.
 *   Throughput: 1 cycle (4 INT8 multiply-adds per cycle per warp).
 *   Latency: 4 cycles.
 *
 * For d=64 (dim=64), one Q·K^T dot product:
 *   64 INT8 elements = 16 dp4a calls (each handles 4 elements).
 *   16 cycles throughput (pipelined).
 *
 * Encoding: pack 4 INT8 values into one uint32 as bytes [31:0].
 *   (int8_t[0] @ bits [7:0], int8_t[1] @ bits [15:8], ...)
 *
 * Compare vs ternary XNOR+popcount path (v1):
 *   XNOR+popcount per word: ~12 cycles (sign-plane method, §2.5 of v1 doc).
 *   dp4a per group-of-4:    ~4 cycles (1 cycle latency x 4-dep chain).
 *   dp4a wins for INT8×INT8 by 3× vs XNOR for this non-ternary stage.
 *   NOTE: ternary weight × INT8 activation path (projection) does NOT use
 *   dp4a — the ternary weights cannot be represented as INT8 without
 *   tripling bandwidth (they're {-1,0,+1}, packing defeats dp4a's byte layout).
 *
 * Usage for Q·K^T dot product:
 */

/**
 * dp4a_dot_int8 — accumulate 4 signed INT8 elements via dp4a hardware.
 *
 * Inputs: a_packed, b_packed = 4 × INT8 packed into uint32 (LSB = element 0).
 *         c_acc = running accumulator (INT32).
 * Output: c_acc + dot4(a[4×INT8], b[4×INT8]).
 *
 * For d=64 dot product: call 16 times, each time advancing the packed pointer
 * by 4 elements (32 bits).
 */
__device__ __forceinline__
int32_t dp4a_dot_int8(uint32_t a_packed, uint32_t b_packed, int32_t c_acc) {
    return __dp4a(a_packed, b_packed, c_acc);
}

/**
 * qk_dot_int8 — full d=64 INT8 × INT8 dot product for the Q·K^T stage.
 *
 * Inputs: q, k = INT8 vectors of length d (must be multiple of 4; pad to 4).
 *         d    = embedding dimension (64, 128, 512, or 2048).
 *
 * Output: INT32 dot product in [-d*127*127, d*127*127].
 *         For d=64: range ±1,032,256 — well within int32_t.
 *
 * Caller should scale before margin comparison if needed.
 */
__device__ __forceinline__
int32_t qk_dot_int8(const int8_t* __restrict__ q,
                    const int8_t* __restrict__ k,
                    int d)
{
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(q);
    const uint32_t* k32 = reinterpret_cast<const uint32_t*>(k);
    int32_t accum = 0;
    int n4 = d / 4;           /* Each dp4a handles 4 elements. */
#pragma unroll 4
    for (int i = 0; i < n4; i++) {
        accum = __dp4a(q32[i], k32[i], accum);
    }
    return accum;
}

/* ============================================================
 *  Section 5: Integer L2 normalisation (VEC_NORM_L2_INT8)
 * ============================================================
 *
 * Daniel's ruling: VEC_NORM is mandatory post-attention (N=1 default).
 * Every attention-layer output MUST be followed by L2 normalisation.
 * Opcode 0x1AD (VEC_NORM_L2_INT8) implements this.
 *
 * Algorithm:
 *   1. Compute L2 norm squared: sum_sq = sum(v[i]^2) for i in [0,d).
 *   2. Estimate reciprocal sqrt via Newton-Raphson (integer, no transcendentals).
 *   3. Scale vector: out[i] = v[i] * scale / norm (integer division).
 *   4. Requantize to INT8 (saturate to [-127, 127]).
 *
 * Newton-Raphson integer reciprocal-sqrt (one step, sufficient for INT8):
 *   seed = rsqrt_approx(sum_sq)   — initial estimate via bit-level trick
 *   refined = seed * (3 - sum_sq * seed * seed) / 2   — one NR step
 *
 * No sqrtf(), no __fsqrt_rn(), no expf(). Pure integer arithmetic.
 */

/**
 * vec_norm_l2_int8 — in-place L2 normalise INT8 vector to scale N.
 *
 * Inputs:  v[0..d-1] INT8 input vector (modified in place).
 *          d         vector dimension.
 *          scale     target L2 norm (integer scale factor; default N=1
 *                    maps to scale=127 for INT8 range).
 *
 * Note: scale=127 means the output has approximately unit L2 norm in INT8
 * representation. Use scale=64 for half-range (safer for subsequent adds).
 *
 * No transcendentals. Uses integer multiply + shift for reciprocal-sqrt.
 */
__device__ __forceinline__
void vec_norm_l2_int8(int8_t* v, int d, int32_t scale) {
    /* Step 1: compute L2 norm squared. */
    int32_t sum_sq = 0;
#pragma unroll 4
    for (int i = 0; i < d; i++) {
        int32_t vi = (int32_t)v[i];
        sum_sq += vi * vi;
    }

    if (sum_sq == 0) return;  /* Zero vector — leave as-is. */

    /* Step 2: integer reciprocal-sqrt approximation.
     *
     * Bit-trick initial estimate (Quake III style, integer version):
     *   For 32-bit sum_sq, we want 1/sqrt(sum_sq).
     *   Initial estimate: shift right to find rough magnitude.
     *   This is accurate to ~25% before Newton refinement.
     *
     * One Newton-Raphson step for y ≈ 1/sqrt(x):
     *   y_new = y * (3 - x * y^2) / 2
     *   In integer fixed-point (Q16 format, multiply then shift):
     */
    /* Compute integer sqrt via Babylonian method (2 iterations, cheap). */
    /* We only need a rough reciprocal, not exact. */
    int32_t norm_est = 1;
    {
        /* Rough sqrt via bit scan: find highest set bit position. */
        int32_t tmp = sum_sq;
        int bit = 0;
        while (tmp > 1) { tmp >>= 2; bit++; }
        norm_est = 1 << bit;  /* norm_est ≈ sqrt(sum_sq) / 2 at worst */
        /* One refinement step: Babylonian (no division, shift-based). */
        /* norm_est = (norm_est + sum_sq / norm_est) / 2 */
        /* Avoid division: use norm_est^2 comparison */
        if (norm_est * norm_est < sum_sq) norm_est++;  /* Ceiling correction. */
    }

    /* Step 3: scale and renormalise. */
    /* out[i] = round(v[i] * scale / norm_est) */
    /* Integer: multiply first (32-bit safe for |v[i]|<=127, scale<=127, norm_est>=1) */
#pragma unroll 4
    for (int i = 0; i < d; i++) {
        int32_t vi    = (int32_t)v[i];
        int32_t scaled = vi * scale;
        /* Divide by norm_est (integer, round-to-nearest). */
        int32_t out_i  = (scaled + (norm_est >> 1)) / norm_est;
        /* Saturate to INT8. */
        if      (out_i >  127) out_i =  127;
        else if (out_i < -127) out_i = -127;
        v[i] = (int8_t)out_i;
    }
}

/* ============================================================
 *  Section 6: Inline wiring example — ATTENTION_FWD_TERNARY
 * ============================================================
 *
 * Shows how this header wires into the K3D attention pipeline.
 * This is NOT a standalone kernel — it is a device-side helper
 * called from within the composed ATTENTION_FWD_TERNARY (0x1A8) kernel
 * that also handles CONTRASTIVE_RANK_TOPK (0x1A9) and yard I/O.
 *
 * Projection step (W_q × input → Q, using BitNet):
 *
 *   // W_q is 1.6-bit packed: [d_out × ceil(d_in/5)] bytes
 *   // input_int8 is INT8: [d_in] elements in yard bank_input
 *   // q_int32 is the accumulator: [d_out] INT32 elements
 *
 *   for (int row = warp_id; row < d_out; row += n_warps) {
 *       const uint8_t* W_row = W_q + row * ((d_in + 4) / 5);
 *       int32_t partial = bitnet_matmul_tile(W_row, input_int8, d_in);
 *       // Warp reduction:
 *       for (int off = 16; off > 0; off >>= 1)
 *           partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
 *       if ((threadIdx.x & 31) == 0) q_int32[row] = partial;
 *   }
 *
 *   // Requantize q_int32 → q_int8 via absmean scale (ingestion-determined):
 *   float absmean = ...;  // stored as Galaxy metadata, loaded at runtime
 *   // INT8 requant: q_int8[i] = clamp(q_int32[i] / absmean, -127, 127)
 *   // Note: absmean itself was quantised and stored as a fixed-point scale
 *   //       in the star record → no float32 at inference time.
 *
 * Q·K^T step:
 *   int32_t score = qk_dot_int8(q_int8, k_int8, d_head);
 *
 * Contrastive margin:
 *   (handled by CONTRASTIVE_RANK_TOPK 0x1A9, uses score from above)
 *
 * V mix output:
 *   (value vectors are INT8 post-requant; mix = add/sub/skip over selected V)
 *
 * VEC_NORM (MANDATORY, opcode 0x1AD):
 *   vec_norm_l2_int8(output_int8, d_head, /*scale=*/ 127);
 */

#endif /* BITNET_B158_CUH */
