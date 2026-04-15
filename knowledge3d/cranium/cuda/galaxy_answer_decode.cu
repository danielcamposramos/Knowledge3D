#include <cuda_runtime.h>
#include <stdint.h>

#include "device_functions.cuh"

#define K3D_GALAXY_EMBEDDING_DIMS 64u
#define K3D_INVALID_STAR_INDEX 0xFFFFFFFFu

extern "C" __global__ void galaxy_answer_decode_top1(
    const float* __restrict__ y_new,
    const unsigned char* __restrict__ galaxy_table,
    unsigned int star_count,
    unsigned int embedding_dims,
    unsigned int require_answer_eligible,
    unsigned int* __restrict__ top_index,
    float* __restrict__ top_score,
    unsigned int* __restrict__ top_galaxy_id,
    unsigned int* __restrict__ top_role_id,
    unsigned long long* __restrict__ top_star_hash
) {
    __shared__ float shared_scores[256];
    __shared__ unsigned int shared_indices[256];

    const unsigned int tid = threadIdx.x;
    const unsigned int dim_count = embedding_dims > K3D_GALAXY_EMBEDDING_DIMS
        ? K3D_GALAXY_EMBEDDING_DIMS
        : embedding_dims;

    float y_norm_sq = 0.0f;
    for (unsigned int dim = 0u; dim < dim_count; ++dim) {
        const float value = y_new[dim];
        y_norm_sq += value * value;
    }
    const float y_norm = sqrtf(y_norm_sq);

    float best_score = -1.0e30f;
    unsigned int best_index = K3D_INVALID_STAR_INDEX;

    if (galaxy_table != nullptr && y_new != nullptr && y_norm > 1.0e-8f && dim_count > 0u) {
        for (unsigned int star_index = tid; star_index < star_count; star_index += blockDim.x) {
            const unsigned int base = star_index * GALAXY_STAR_RECORD_BYTES;
            const unsigned int flags = *reinterpret_cast<const unsigned int*>(
                galaxy_table + base + GALAXY_STAR_FLAGS_OFFSET
            );
            if ((flags & GALAXY_STAR_FLAG_ACTIVE) == 0u) {
                continue;
            }
            const unsigned int answer_eligible = *reinterpret_cast<const unsigned int*>(
                galaxy_table + base + GALAXY_STAR_ANSWER_ELIGIBLE_OFFSET
            );
            if (require_answer_eligible != 0u && answer_eligible == 0u) {
                continue;
            }

            const float* embedding = reinterpret_cast<const float*>(
                galaxy_table + base + GALAXY_STAR_EMBEDDING_OFFSET
            );
            float dot = 0.0f;
            float star_norm_sq = 0.0f;
            for (unsigned int dim = 0u; dim < dim_count; ++dim) {
                const float value = embedding[dim];
                dot += y_new[dim] * value;
                star_norm_sq += value * value;
            }
            const float star_norm = sqrtf(star_norm_sq);
            if (star_norm <= 1.0e-8f) {
                continue;
            }
            const float score = dot / (y_norm * star_norm);
            if (score > best_score || (score == best_score && star_index < best_index)) {
                best_score = score;
                best_index = star_index;
            }
        }
    }

    shared_scores[tid] = best_score;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_score = shared_scores[tid + stride];
            const unsigned int other_index = shared_indices[tid + stride];
            const unsigned int current_index = shared_indices[tid];
            if (
                other_score > shared_scores[tid]
                || (other_score == shared_scores[tid] && other_index < current_index)
            ) {
                shared_scores[tid] = other_score;
                shared_indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) {
        const unsigned int winner = shared_indices[0];
        top_index[0] = winner;
        top_score[0] = shared_scores[0];
        if (winner == K3D_INVALID_STAR_INDEX || galaxy_table == nullptr) {
            top_galaxy_id[0] = 0u;
            top_role_id[0] = 0u;
            top_star_hash[0] = 0ull;
            return;
        }
        const unsigned int base = winner * GALAXY_STAR_RECORD_BYTES;
        top_galaxy_id[0] = *reinterpret_cast<const unsigned int*>(
            galaxy_table + base + GALAXY_STAR_GALAXY_ID_OFFSET
        );
        top_role_id[0] = *reinterpret_cast<const unsigned int*>(
            galaxy_table + base + GALAXY_STAR_SELECTION_ROLE_OFFSET
        );
        top_star_hash[0] = *reinterpret_cast<const unsigned long long*>(
            galaxy_table + base + GALAXY_STAR_HASH_OFFSET
        );
    }
}
