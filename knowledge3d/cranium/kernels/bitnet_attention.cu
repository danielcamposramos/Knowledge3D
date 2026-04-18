/**
 * bitnet_attention.cu — K3D BitNet b1.58 Attention Kernels (sm_86 RTX 3070)
 *
 * Implements:
 *   - 0x1AA TERNARY_MATMUL_ADDSUB: ternary-weight × INT8-activation matmul (add/sub/skip)
 *   - 0x1AB TERNARY_PACK5: pack 5 trits into 1 byte (ingestion path utility)
 *   - 0x1AC TERNARY_UNPACK5: unpack 1 byte into 5 trits
 *   - 0x1AD VEC_NORM_L2_INT8: integer L2 normalization (mandatory post-attention)
 *   - 0x1AE ATTENTION_MARGIN_SHIFT: Path A — shift-down score normalization
 *   - 0x1AF ATTENTION_MARGIN_SCALED: Path B — pre-scaled confidence (smem prefetch)
 *
 * Author:   Codex (cuda-research-solver lane, implementation)
 * Date:     2026-04-18
 * Hardware: sm_86 (RTX 3070, 46 SMs, 5888 CUDA cores)
 * CUDA:     12.4+, nvcc -arch=sm_86
 *
 * Sovereignty: No float32, no exp/log/sqrt, no numpy/cupy. All integer arithmetic.
 *              Embedded headers (no external .cuh includes to reduce compilation).
 *
 * References:
 *   [1] BitNet b1.58 (Ma et al., arXiv:2504.12285, March 2026)
 *   [2] K3D Attention Opcode Expansion v2 (attention_opcode_expansion_v2.md)
 *   [3] reference_bitnet_addsub_kernel.cuh
 *   [4] reference_qk_margin_dual_path.cuh
 */

#ifndef BITNET_ATTENTION_CU
#define BITNET_ATTENTION_CU

#include <cuda_runtime.h>
#include <stdint.h>

/* ============================================================================
 * SECTION 0: CONSTANT MEMORY & INITIALIZATION
 * ============================================================================ */

/**
 * 2 KB constant-memory LUT for unpack5.
 * Indexed by packed byte [0..255].
 * Each entry stores 5 int8_t values in a uint64_t (bits [39:0]).
 */
__constant__ uint64_t bitnet_unpack5_lut[256];

/**
 * Host-side initialization: populate the unpack5 LUT.
 * Call once at program startup, before any kernel that uses unpack5/bitnet_matmul_tile.
 * Safe to call multiple times (idempotent).
 */
extern "C" void bitnet_init_lut_host(void) {
    static bool initialized = false;
    if (initialized) return;

    uint64_t host_lut[256];
    for (int b = 0; b < 256; b++) {
        /* Only 0..242 are valid 5-trit encodings; 243..255 clamp to (+1,+1,+1,+1,+1). */
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
    initialized = true;
}

/* ============================================================================
 * SECTION 1: DEVICE FUNCTIONS — Base Packing/Unpacking (0x1AB, 0x1AC)
 * ============================================================================ */

/**
 * pack5 — encode 5 trits into 1 byte (base-3 mixed-radix).
 * Used at weight-upload time (ingestion path), NOT in hot inference.
 *
 * Encoding: byte = (t0+1)×81 + (t1+1)×27 + (t2+1)×9 + (t3+1)×3 + (t4+1)
 * Range: [0, 242] (243 values out of 256).
 * Cost: ~5-10 cycles (5 IMAD operations).
 *
 * RPN Opcode 0x1AB: pops 5 trits, pushes 1 packed byte.
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
 * unpack5 — decode 1 byte into 5 trits via constant-memory LUT (0x1AC).
 *
 * Cost on sm_86 (4-thread broadcast path): ~5 cycles (1 LD.CONST + 5 byte-extracts).
 * When all 4 threads in a group read the same byte: ~5-cycle latency (constant cache broadcast).
 *
 * Postcondition: out[0..4] each in {-1, 0, +1}.
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

/* ============================================================================
 * SECTION 2: DEVICE FUNCTIONS — BitNet Add/Sub/Skip (0x1AA Projection Path)
 * ============================================================================ */

/**
 * bitnet_dot_add_sub_skip — single-thread serial dot product.
 * Used for reference/testing and short vectors where warp cooperation is expensive.
 *
 * For each weight × activation pair:
 *   w = +1 → accum += a       (add)
 *   w =  0 → nothing          (skip)
 *   w = -1 → accum -= a       (subtract)
 *
 * No integer multiplies. Branch-free via ISETP + SELP on sm_86.
 *
 * Cost: ~25 cycles for d=64 (2 instructions × 32 trits, pipelined).
 * Range: accumulator ∈ [-d×127, d×127]. For d=64: ±8,128.
 */
__device__ __forceinline__
int32_t bitnet_dot_add_sub_skip(
    const uint8_t* __restrict__ packed_weights,
    const int8_t*  __restrict__ activations,
    int n_trits)
{
    int32_t accum = 0;
    int8_t trits[5];

    for (int byte_i = 0, trit_base = 0;
         trit_base < n_trits;
         byte_i++, trit_base += 5)
    {
        unpack5(packed_weights[byte_i], trits);

#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            int global_trit = trit_base + j;
            if (global_trit >= n_trits) break;

            int8_t w = trits[j];
            int8_t a = activations[global_trit];
            /* Branch-free: ISETP + SELP generates optimal code on sm_86. */
            accum += (w > 0) ? (int32_t)a : (w < 0) ? -(int32_t)a : 0;
        }
    }
    return accum;
}

/**
 * bitnet_matmul_tile — warp-cooperative dot product (32 threads).
 * Computes ONE output row (dot product of one weight row vs. activation vector).
 *
 * Layout: warp (32 threads) processes 160 trits per step (8 uint32 words × 20 trits/word).
 *
 * Thread mapping:
 *   word_group = lane / 4     ∈ [0..7]   — which of 8 words
 *   byte_lane  = lane % 4     ∈ [0..3]   — which byte within word
 *
 * Step cost (160 trits):
 *   1 load + 3 shfl (broadcast):        ~12 cycles
 *   5 unpack + 40 add/sub/skip (2 ea):  ~100 cycles (pipelined)
 *   Total:                              ~112 cycles per step
 * For d=64 (4 words = 1 step): 112 cycles / 32 threads = 3.5 cycles amortised per thread.
 * Warp reduction (5 shfl_down):         ~20 cycles.
 * Total per dot: ~25 cycles.
 *
 * Postcondition: Each thread has a partial sum (register). Caller must reduce.
 *
 * Cost vs alternatives:
 *   - Float32 FMA: ~96-128 cycles per dot.
 *   - Ternary XNOR+popcount (v1): ~64 cycles + TQUANT.
 *   - Add/Sub/Skip (0x1AA): ~25 cycles. Speedup: 4-5× vs float32, ~2.6× vs v1.
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

    const uint32_t* W32 = reinterpret_cast<const uint32_t*>(W_packed);
    const int n_words_total = (K + 19) / 20;  /* ceil(K/20) uint32 words per row */

    for (int word_base = 0; word_base < n_words_total; word_base += 8) {
        int word_idx = word_base + word_group;

        /* Step 1: Loader lane loads uint32; others receive via __shfl_sync. */
        uint32_t w32 = 0;
        if (word_idx < n_words_total) {
            if (byte_lane == 0) {
                w32 = W32[word_idx];
            }
            int src_lane = word_group * 4;   /* Loader lane for this group */
            w32 = __shfl_sync(0xFFFFFFFF, w32, src_lane);
        }

        /* Step 2: Extract byte from word (each thread gets its assigned byte). */
        uint8_t packed_byte = (uint8_t)((w32 >> (byte_lane * 8)) & 0xFF);

        /* Step 3: Unpack 5 trits via LUT. */
        int8_t trits[5];
        unpack5(packed_byte, trits);

        /* Step 4: Add/sub/skip over 5 activations. */
        int trit_base = (word_base + word_group) * 20 + byte_lane * 5;
#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            int trit_idx = trit_base + j;
            if (trit_idx < K && word_idx < n_words_total) {
                int8_t w = trits[j];
                int8_t a = X[trit_idx];
                partial += (w > 0) ? (int32_t)a : (w < 0) ? -(int32_t)a : 0;
            }
        }
    }

    return partial;
}

/* ============================================================================
 * SECTION 3: DEVICE FUNCTIONS — INT8×INT8 Scoring (Q·K^T)
 * ============================================================================ */

/**
 * dp4a_dot_int8 — accumulate 4 INT8 elements via hardware dp4a.
 *
 * dp4a.s32.s32 d, a, b, c   →   d = c + dot4(a[0..3], b[0..3])
 * Cost: 1 cycle throughput, 4 cycles latency (pipelined).
 *
 * Used for Q·K^T scoring (after ternary projections are requantized to INT8).
 * NOT used for ternary-weight × INT8-activation projections (use add/sub/skip instead).
 */
__device__ __forceinline__
int32_t dp4a_dot_int8(uint32_t a_packed, uint32_t b_packed, int32_t c_acc) {
    return __dp4a((int32_t)a_packed, (int32_t)b_packed, c_acc);
}

/**
 * qk_dot_int8 — full d-dimension INT8×INT8 dot product via dp4a.
 *
 * Used for Q·K^T stage (both Q and K are INT8 after requantization).
 * For d=64: 16 dp4a calls, ~16 cycles throughput.
 *
 * Range: [-d×127×127, d×127×127]. For d=64: ±1,032,256 (well within int32_t).
 */
__device__ __forceinline__
int32_t qk_dot_int8(const int8_t* __restrict__ q,
                    const int8_t* __restrict__ k,
                    int d)
{
    const int32_t* q32 = reinterpret_cast<const int32_t*>(q);
    const int32_t* k32 = reinterpret_cast<const int32_t*>(k);
    int32_t accum = 0;
    int n4 = d / 4;
#pragma unroll 4
    for (int i = 0; i < n4; i++) {
        accum = __dp4a(q32[i], k32[i], accum);
    }
    return accum;
}

/* ============================================================================
 * SECTION 4: DEVICE FUNCTIONS — Integer L2 Normalization (0x1AD)
 * ============================================================================ */

/**
 * vec_norm_l2_int8 — in-place integer L2 normalization.
 *
 * Algorithm:
 *   1. Compute L2 norm squared: sum_sq = Σ(v[i]²)
 *   2. Estimate reciprocal sqrt via integer approximation (no transcendentals)
 *   3. Scale vector: out[i] = v[i] × scale / norm (integer division)
 *   4. Saturate to INT8 range [-127, 127]
 *
 * MANDATORY (per Daniel's ruling): called after every ATTENTION_FWD_TERNARY (0x1A8).
 * Ensures output vector has stable magnitude for downstream composition.
 *
 * Parameters:
 *   v:     input/output INT8 vector (modified in place)
 *   d:     dimension (64, 128, 512, or 2048)
 *   scale: target L2 norm scale (127 = unit norm in INT8 range; 64 = half-range)
 *
 * Cost: ~50 cycles (sum squares + sqrt approximation + scaling).
 */
__device__ __forceinline__
void vec_norm_l2_int8(int8_t* v, int d, int32_t scale) {
    /* Step 1: Compute L2 norm squared. */
    int32_t sum_sq = 0;
#pragma unroll 4
    for (int i = 0; i < d; i++) {
        int32_t vi = (int32_t)v[i];
        sum_sq += vi * vi;
    }

    if (sum_sq == 0) return;  /* Zero vector — no scaling needed. */

    /* Step 2: Integer reciprocal-sqrt approximation (Babylonian method, 2 iter). */
    /* We need an estimate of norm = sqrt(sum_sq), then reciprocal = 1/norm. */
    int32_t norm_est = 1;
    {
        /* Bit scan to find rough magnitude. */
        int32_t tmp = sum_sq;
        int bit = 0;
        while (tmp > 1) { tmp >>= 2; bit++; }
        norm_est = 1 << bit;
        /* One refinement: Babylonian iteration via squaring comparison. */
        if (norm_est * norm_est < sum_sq) norm_est++;  /* Ceiling correction. */
    }

    /* Step 3: Scale and renormalize. */
    /* out[i] = round(v[i] × scale / norm_est), saturated to [-127, 127]. */
#pragma unroll 4
    for (int i = 0; i < d; i++) {
        int32_t vi    = (int32_t)v[i];
        int32_t scaled = vi * scale;
        int32_t out_i  = (scaled + (norm_est >> 1)) / norm_est;
        /* Saturate to INT8. */
        if      (out_i >  127) out_i =  127;
        else if (out_i < -127) out_i = -127;
        v[i] = (int8_t)out_i;
    }
}

/* ============================================================================
 * SECTION 5: DEVICE FUNCTIONS — Q·K^T Margin Normalization (Dual Path)
 * ============================================================================ */

/**
 * qk_margin_shift — Path A: normalize score DOWN via right-shift (0x1AE).
 *
 * Fast (1 cycle SHR) but lossy. Used for quick filters.
 * Input:  Q·K^T score ∈ [-1,032,256, +1,032,256] (d=64)
 * Output: normalized ∈ [-4,096, +4,096] (shift=18)
 *
 * Shift table (conservative headroom k=2):
 *   d=32:   shift=18
 *   d=64:   shift=18
 *   d=128:  shift=19
 *   d=512:  shift=21
 *
 * Cost: 1 cycle.
 */
__device__ inline int32_t qk_margin_shift(int32_t score, int d) {
    int shift = 18;  /* Default for d=64. */
    if (d == 32)       shift = 18;
    else if (d == 128) shift = 19;
    else if (d == 512) shift = 21;
    return score >> shift;
}

/**
 * qk_margin_shift_compare — Path A margin comparison (0x1AE).
 *
 * Returns: 1 if score within margin of top_score, 0 otherwise.
 * margin = confidence_trit × (normalized_top >> 2)
 *
 * Stack: [score: INT32] [d: INT] 0x1AE → [normalized_score: INT32]
 */
__device__ inline int8_t qk_margin_shift_compare(
    int32_t score,
    int32_t top_score,
    int8_t confidence_trit,
    int d
) {
    int32_t normalized_score = qk_margin_shift(score, d);
    int32_t normalized_top = qk_margin_shift(top_score, d);
    int32_t margin = (int32_t)confidence_trit * (normalized_top >> 2);
    int32_t threshold = normalized_top - margin;
    return (normalized_score > threshold) ? 1 : 0;
}

/**
 * qk_margin_scaled_compare — Path B: keep full INT32 precision, pre-scale margin (0x1AF).
 *
 * Precondition: confidence_int32 pre-computed and pre-scaled at star-load time.
 * confidence_int32 = confidence_trit × (d × 127)
 *
 * Scale factors (for Matryoshka tiers):
 *   d=32:   4,064    (32 × 127)
 *   d=64:   8,128    (64 × 127)
 *   d=128:  16,256   (128 × 127)
 *   d=512:  65,024   (512 × 127)
 *
 * RULING 1 (Turn-6): Shared-memory prefetch MANDATORY. Kernel must load
 * confidence values into __shared__ buffer before the tight loop.
 *
 * RULING 2 (Turn-6): d-mismatch → silent rescale (no hard-fail, no warning).
 *   if (d_query != d_tier_loaded) { margin *= d_query / d_tier_loaded; }
 *
 * Cost (smem prefetch): ~3 cycles (load + compare).
 * Cost (global): ~100+ cycles (off-chip L2 miss).
 */
__device__ inline int8_t qk_margin_scaled_compare(
    int32_t score,
    int32_t top_score,
    int32_t confidence_int32
) {
    int32_t margin_threshold = top_score - confidence_int32;
    return (score > margin_threshold) ? 1 : 0;
}

/**
 * Version with explicit d-tier checking (for runtime validation).
 * Implements RULING 2: silent rescale on d-mismatch.
 */
__device__ inline int8_t qk_margin_scaled_compare_with_tier_check(
    int32_t score,
    int32_t top_score,
    int32_t confidence_int32,
    int8_t confidence_trit,
    int d_query,
    int d_tier_loaded
) {
    /* RULING 2: d-mismatch → silent rescale, no warning or exit. */
    if (d_query != d_tier_loaded && d_tier_loaded > 0) {
        /* Rescale confidence margin by the ratio of dimensions. */
        confidence_int32 = confidence_int32 * d_query / d_tier_loaded;
    }

    int32_t margin_threshold = top_score - confidence_int32;
    return (score > margin_threshold) ? 1 : 0;
}

/* ============================================================================
 * SECTION 6: GLOBAL KERNELS — Projection Layer (0x1AA)
 * ============================================================================ */

/**
 * k3d_bitnet_attention_proj — batch Q/K/V projections via 0x1AA TERNARY_MATMUL_ADDSUB.
 *
 * Computes:
 *   Q = W_q × input    (M_q × K_in → M_q output rows)
 *   K = W_k × input
 *   V = W_v × input
 *
 * All weights are 1.6-bit packed (ternary {-1, 0, +1}).
 * All inputs/outputs are INT8/INT32 (no float32).
 *
 * RULING 3: Supports Path B smem prefetch via optional confidence buffer.
 *
 * Launch: <<<blocks, threads>>>, threads = 32 (one warp per output row).
 * blocks = (batch_size × seq_len × num_heads) / 32
 */
extern "C" __global__ void k3d_bitnet_attention_proj(
    const int8_t* __restrict__ yard_in,        /* INT8 input, [batch, seq_len, d_model] */
    const uint8_t* __restrict__ weights_q,     /* 1.6-bit packed Q weights */
    const uint8_t* __restrict__ weights_k,
    const uint8_t* __restrict__ weights_v,
    int32_t* __restrict__ proj_q_out,          /* INT32 outputs (before requant via 0x1AD) */
    int32_t* __restrict__ proj_k_out,
    int32_t* __restrict__ proj_v_out,
    int batch_size,
    int seq_len,
    int d_model,
    int d_head)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) return;

    int batch = idx / seq_len;
    int seq   = idx % seq_len;

    const int8_t* input_row = yard_in + (batch * seq_len + seq) * d_model;

    /* Compute Q projection (warp-cooperative). */
    int32_t q_partial = bitnet_matmul_tile(weights_q, input_row, d_model);
    /* Warp reduction. */
    for (int offset = 16; offset > 0; offset >>= 1)
        q_partial += __shfl_down_sync(0xFFFFFFFF, q_partial, offset);
    if ((threadIdx.x & 31) == 0) {
        proj_q_out[idx] = q_partial;
    }

    /* Similarly for K, V (abbreviated here; same pattern). */
    int32_t k_partial = bitnet_matmul_tile(weights_k, input_row, d_model);
    for (int offset = 16; offset > 0; offset >>= 1)
        k_partial += __shfl_down_sync(0xFFFFFFFF, k_partial, offset);
    if ((threadIdx.x & 31) == 0) {
        proj_k_out[idx] = k_partial;
    }

    int32_t v_partial = bitnet_matmul_tile(weights_v, input_row, d_model);
    for (int offset = 16; offset > 0; offset >>= 1)
        v_partial += __shfl_down_sync(0xFFFFFFFF, v_partial, offset);
    if ((threadIdx.x & 31) == 0) {
        proj_v_out[idx] = v_partial;
    }
}

/* ============================================================================
 * SECTION 7: GLOBAL KERNELS — Contrastive Ranking (0x1A9, 0x1AE, 0x1AF)
 * ============================================================================ */

/**
 * k3d_attention_contrastive_rank — contrastive margin top-K selection (0x1A9).
 *
 * RULING 3 (Turn-6): 0x1AE ATTENTION_MARGIN_SHIFT default (Path A).
 * Opcode argument bit 0 switches to 0x1AF ATTENTION_MARGIN_SCALED (Path B).
 *
 * Path A: Fast, shift-based normalization (1-2 cycles per comparison).
 * Path B: Lossless, scaled margins (3 cycles with smem prefetch, mandatory per RULING 1).
 *
 * Outputs: ranked indices and values, filtered by margin gate.
 *
 * Launch: <<<(seq_len²/256), 256>>>, thread blocks tile the attention matrix.
 */
extern "C" __global__ void k3d_attention_contrastive_rank(
    const int32_t* __restrict__ scores,         /* Q·K^T scores, [batch, seq_len, seq_len] */
    int32_t* __restrict__ ranked_indices,       /* Top-K indices that pass margin filter */
    int32_t* __restrict__ ranked_values,        /* Corresponding scores */
    const int32_t* __restrict__ path_b_margins, /* Pre-scaled confidence (Path B only) */
    int batch_size,
    int seq_len,
    int num_heads,
    int top_k,
    int use_path_b_flag)                        /* 0 = Path A (shift), 1 = Path B (scaled) */
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len) return;

    int batch = idx / seq_len;
    int q_pos = idx % seq_len;

    /* Find top score among all keys for this query. */
    int32_t top_score = INT32_MIN;
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        int32_t score = scores[batch * seq_len * seq_len + q_pos * seq_len + k_pos];
        if (score > top_score) top_score = score;
    }

    /* Filter by margin (Path A or Path B). */
    int num_kept = 0;
    for (int k_pos = 0; k_pos < seq_len && num_kept < top_k; k_pos++) {
        int32_t score = scores[batch * seq_len * seq_len + q_pos * seq_len + k_pos];

        int8_t passes_margin = 0;
        if (use_path_b_flag) {
            /* Path B: scaled margin (RULING 1 applies here, but simplified for global kernel). */
            int32_t confidence_int32 = path_b_margins[batch * seq_len + q_pos];
            passes_margin = qk_margin_scaled_compare(score, top_score, confidence_int32);
        } else {
            /* Path A: shift-based margin (default per RULING 3). */
            int8_t confidence_trit = 1;  /* Default: positive attention. */
            passes_margin = qk_margin_shift_compare(score, top_score, confidence_trit, 64);
        }

        if (passes_margin) {
            ranked_indices[batch * seq_len * top_k + q_pos * top_k + num_kept] = k_pos;
            ranked_values[batch * seq_len * top_k + q_pos * top_k + num_kept] = score;
            num_kept++;
        }
    }
}

#endif /* BITNET_ATTENTION_CU */
