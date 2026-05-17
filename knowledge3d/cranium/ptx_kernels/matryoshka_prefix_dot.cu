/*
 * matryoshka_prefix_dot.cu — Fused variable-width prefix dot product
 *
 * Spec: CLAUDE_CODEX_GPU_GAME_LOOP_CLOSURE_04.18.2026.md §5
 *       §5.1 VRAM layout, §5.2 fused prefix dot, §5.3 meta-rule tier selection
 *
 * Matryoshka embedding design:
 *   VRAM layout: row-major float[N][2048] — each row is a 2048-dim embedding.
 *   A k-dim prefix view is the first k columns of each row.
 *   Tier k ∈ {64, 128, 256, 512, 1024, 2048} is selected at runtime by the
 *   Layer 4 meta-rule RPN program meta_select_matryoshka_tier, which writes
 *   its result to shared memory register `tier_signal`.
 *   No Python decides the tier. No host-side if/else.
 *
 * Per-warp computation:
 *   One warp (32 lanes) handles one dot product of dimension k.
 *   Each lane accumulates k/32 multiply-accumulates over its strided elements.
 *   Final reduction: log2(32) = 5 rounds of __shfl_xor_sync butterfly.
 *   Lane 0 holds the result.
 *
 *   For k = 64:  each lane does  2 MACs
 *   For k = 128: each lane does  4 MACs
 *   For k = 256: each lane does  8 MACs
 *   For k = 512: each lane does 16 MACs
 *   For k=1024:  each lane does 32 MACs
 *   For k=2048:  each lane does 64 MACs
 *
 * Memory access pattern:
 *   Consecutive lanes access consecutive float elements (coalesced).
 *   Lane i reads elements at offsets i, i+32, i+64, ... up to k-1.
 *   This is stride-32 access into a contiguous float row — fully coalesced
 *   in a single global memory transaction per 32-element window.
 *
 * WHY not use shared memory tiling here:
 *   The query vector q (up to 2048 floats = 8 KB) fits in registers across
 *   a warp. Loading it once from global and accumulating in registers avoids
 *   shared memory bank conflicts and keeps occupancy high.
 *
 * Target: sm_86 (RTX 3070). __shfl_xor_sync available from sm_30.
 */

#include <cuda_runtime.h>
#include <cstdint>

/* Valid tier set — any other value is treated as 64 (minimum viable tier) */
#define K3D_MAT_TIER_64    64u
#define K3D_MAT_TIER_128  128u
#define K3D_MAT_TIER_256  256u
#define K3D_MAT_TIER_512  512u
#define K3D_MAT_TIER_1024 1024u
#define K3D_MAT_TIER_2048 2048u

/* Warp size constant — do NOT assume blockDim.x == 32 */
#define K3D_WARP_SIZE 32u
#define K3D_WARP_FULL_MASK 0xFFFFFFFFu

/* ---------------------------------------------------------------------------
 * matryoshka_prefix_dot — fused dot product q·k[0:dim].
 *
 * Called by one warp (32 consecutive threads in a block).
 * Each thread participates; lane 0 holds the result after return.
 * Other lanes return an UNDEFINED value — callers must not use them.
 *
 * Parameters:
 *   q   — device ptr to query embedding (float[2048], first `dim` elements used)
 *   k   — device ptr to key embedding   (float[2048], first `dim` elements used)
 *   dim — prefix dimension ∈ {64,128,256,512,1024,2048}
 *         If dim is not a valid tier, falls back to 64.
 *
 * Returns: dot product on lane 0 only.
 *
 * IMPORTANT: q and k must be aligned to at least 4 bytes (float alignment).
 *   For best performance, align to 128 bytes (cache line).
 * --------------------------------------------------------------------------- */
__device__ float matryoshka_prefix_dot(
    const float* __restrict__ q,
    const float* __restrict__ k,
    uint32_t dim)
{
    /* Clamp to valid tier; default to 64 on invalid input */
    if (dim != 64u && dim != 128u && dim != 256u &&
        dim != 512u && dim != 1024u && dim != 2048u)
    {
        dim = K3D_MAT_TIER_64;
    }

    const uint32_t lane = threadIdx.x & (K3D_WARP_SIZE - 1u);

    float acc = 0.0f;

    /*
     * Strided accumulation: lane i handles elements i, i+32, i+64, ...
     * up to dim-1. This is coalesced: all lanes in the warp read a
     * contiguous 32-float window per iteration.
     */
    for (uint32_t base = lane; base < dim; base += K3D_WARP_SIZE) {
        acc += q[base] * k[base];
    }

    /*
     * Warp butterfly reduction: 5 rounds for 32 lanes.
     * After each round, partial sums collapse by half.
     * After round 5: lane 0 holds the full dot product.
     */
    acc += __shfl_xor_sync(K3D_WARP_FULL_MASK, acc, 16u);
    acc += __shfl_xor_sync(K3D_WARP_FULL_MASK, acc,  8u);
    acc += __shfl_xor_sync(K3D_WARP_FULL_MASK, acc,  4u);
    acc += __shfl_xor_sync(K3D_WARP_FULL_MASK, acc,  2u);
    acc += __shfl_xor_sync(K3D_WARP_FULL_MASK, acc,  1u);

    /* Lane 0 holds the result. Other lanes' values are implementation-defined. */
    return acc;
}

/* ---------------------------------------------------------------------------
 * matryoshka_prefix_dot_batch — score n candidate keys against query q.
 *
 * Spec §5.2: one candidate per warp. Each warp computes one dot product.
 * The calling kernel allocates (n + 31) / 32 warps across its blocks.
 *
 * Parameters:
 *   q   — device ptr to query embedding float[2048]
 *   K   — device ptr to candidate key matrix float[n][2048] (row-major)
 *   n   — number of candidates
 *   dim — prefix dimension (same for all candidates; from tier_signal)
 *   out — device ptr to output float[n]; out[i] = q · K[i][0:dim]
 *
 * Caller's launch configuration:
 *   blockDim.x must be a multiple of 32.
 *   gridDim.x must cover ceil(n / (blockDim.x / 32)) warps.
 *   The per-warp index is: warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32
 *
 * Lane 0 of each warp writes the result to out[candidate_idx].
 * --------------------------------------------------------------------------- */
__device__ void matryoshka_prefix_dot_batch(
    const float* __restrict__ q,
    const float* __restrict__ K,
    uint32_t n,
    uint32_t dim,
    float* __restrict__ out)
{
    /* Global warp index */
    const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp_id       = global_thread / K3D_WARP_SIZE;
    const uint32_t lane          = global_thread & (K3D_WARP_SIZE - 1u);

    if (warp_id >= n) return;

    /* Point to this warp's candidate row: K[warp_id] */
    const float* __restrict__ k_row = K + static_cast<uint64_t>(warp_id) * 2048u;

    float dot = matryoshka_prefix_dot(q, k_row, dim);

    /* Only lane 0 writes the output (holds the valid reduction result) */
    if (lane == 0u) {
        out[warp_id] = dot;
    }
}

/* ---------------------------------------------------------------------------
 * matryoshka_prefix_dot_batch_kernel — __global__ entry for the batch variant.
 *
 * This is the callable kernel form. The __device__ version above is for
 * inline use inside the persistent tick. Both are provided so that:
 *   - persistent_tick.cu calls the __device__ function directly (no launch)
 *   - Tests and standalone benchmarks call this __global__ entry
 *
 * Launch config: grid = ceil(n/warps_per_block) blocks, block = 256 threads
 *   (8 warps per block, each warp handles one candidate)
 * --------------------------------------------------------------------------- */
__global__ void matryoshka_prefix_dot_batch_kernel(
    const float* __restrict__ q,
    const float* __restrict__ K,
    uint32_t n,
    uint32_t dim,
    float* __restrict__ out)
{
    matryoshka_prefix_dot_batch(q, K, n, dim, out);
}
