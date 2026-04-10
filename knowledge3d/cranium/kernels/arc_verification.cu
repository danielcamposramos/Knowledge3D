#include <stdint.h>

struct ArcGrid {
    uint8_t cells[900];
    uint16_t height;
    uint16_t width;
};

#define ARC_MAX_DEMO_PAIRS 5u

__device__ inline uint32_t arc_warp_sum(uint32_t value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
    }
    return value;
}

__device__ inline uint32_t arc_block_sum(uint32_t value) {
    __shared__ uint32_t warp_sums[32];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    value = arc_warp_sum(value);
    if (lane == 0u) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    uint32_t total = 0u;
    const uint32_t warp_count = (blockDim.x + 31u) >> 5u;
    if (warp == 0u) {
        total = (lane < warp_count) ? warp_sums[lane] : 0u;
        total = arc_warp_sum(total);
    }
    return total;
}

__device__ inline uint32_t arc_pair_mismatch_count(
    const ArcGrid* candidate,
    const ArcGrid* target
) {
    if (candidate->height != target->height || candidate->width != target->width) {
        return threadIdx.x == 0 ? 1u : 0u;
    }
    const uint32_t cell_count = static_cast<uint32_t>(candidate->height) * static_cast<uint32_t>(candidate->width);
    uint32_t local_mismatches = 0u;
    for (uint32_t idx = threadIdx.x; idx < cell_count; idx += blockDim.x) {
        if (candidate->cells[idx] != target->cells[idx]) {
            local_mismatches += 1u;
        }
    }
    return arc_block_sum(local_mismatches);
}

extern "C" __global__ void arc_verify_candidate(
    const ArcGrid* __restrict__ candidate,
    const ArcGrid* __restrict__ training_outputs,
    uint32_t n_pairs,
    uint32_t* out_match_count
) {
    __shared__ uint32_t match_count;
    if (threadIdx.x == 0) {
        match_count = 0u;
    }
    __syncthreads();

    const uint32_t pair_count = n_pairs > ARC_MAX_DEMO_PAIRS ? ARC_MAX_DEMO_PAIRS : n_pairs;
    for (uint32_t pair_idx = 0u; pair_idx < pair_count; ++pair_idx) {
        const uint32_t mismatches = arc_pair_mismatch_count(candidate, &training_outputs[pair_idx]);
        if (threadIdx.x == 0 && mismatches == 0u) {
            match_count += 1u;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_match_count = match_count;
    }
}

extern "C" __global__ void arc_score_candidates(
    const ArcGrid* __restrict__ candidates,
    uint32_t n_candidates,
    const ArcGrid* __restrict__ training_outputs,
    uint32_t n_pairs,
    uint32_t* out_scores
) {
    const uint32_t candidate_idx = blockIdx.x;
    if (candidate_idx >= n_candidates) {
        return;
    }
    __shared__ uint32_t match_count;
    if (threadIdx.x == 0) {
        match_count = 0u;
    }
    __syncthreads();

    const ArcGrid* candidate = &candidates[candidate_idx];
    const uint32_t pair_count = n_pairs > ARC_MAX_DEMO_PAIRS ? ARC_MAX_DEMO_PAIRS : n_pairs;
    for (uint32_t pair_idx = 0u; pair_idx < pair_count; ++pair_idx) {
        const uint32_t mismatches = arc_pair_mismatch_count(candidate, &training_outputs[pair_idx]);
        if (threadIdx.x == 0 && mismatches == 0u) {
            match_count += 1u;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_scores[candidate_idx] = match_count;
    }
}
