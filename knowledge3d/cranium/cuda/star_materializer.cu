#include <stdint.h>
#include <math.h>

#include "device_functions.cuh"

#define STAR_RECORD_BYTES                  256
#define STAR_EMBEDDING_OFFSET                0
#define STAR_GALAXY_ID_OFFSET              128
#define STAR_TYPE_OFFSET                   132
#define STAR_SELECTION_ROLE_OFFSET         136
#define STAR_LAYER_ID_OFFSET               140
#define STAR_FLAGS_OFFSET                  144
#define STAR_ANSWER_ELIGIBLE_OFFSET        148
#define STAR_SEMANTIC_POLARITY_OFFSET      152
#define STAR_SEMANTIC_FOCUS_OFFSET         156
#define STAR_SEMANTIC_MASS_OFFSET          160
#define STAR_ATTRACTIVE_PRIOR_OFFSET       164
#define STAR_REPULSIVE_PRIOR_OFFSET        168
#define STAR_ROUTE_POLICY_OFFSET           172
#define STAR_STAR_HASH_OFFSET              176
#define STAR_ROUTER_REF_COUNT_OFFSET       184
#define STAR_ROUTER_REFS_OFFSET            188
#define STAR_EXECUTOR_REF_COUNT_OFFSET     196
#define STAR_EXECUTOR_REFS_OFFSET          200
#define STAR_VALIDATOR_REF_COUNT_OFFSET    208
#define STAR_VALIDATOR_REFS_OFFSET         212
#define STAR_ANTI_PATTERN_REF_COUNT_OFFSET 220
#define STAR_ANTI_PATTERN_REFS_OFFSET      224
#define STAR_POSITION_OFFSET               232
#define STAR_VELOCITY_OFFSET               244

#define STAR_NULL_REF 0xFFFFFFFFu

#define CATALOG_INPUT_ENTRY_BYTES          152

static __device__ __forceinline__ float read_f32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const float*>(ptr + offset);
}

static __device__ __forceinline__ unsigned int read_u32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const unsigned int*>(ptr + offset);
}

static __device__ __forceinline__ int read_i32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const int*>(ptr + offset);
}

static __device__ __forceinline__ unsigned long long read_u64(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const unsigned long long*>(ptr + offset);
}

extern "C" __global__ void star_materializer(
    unsigned char* __restrict__ galaxy_table,
    const unsigned char* __restrict__ input,
    unsigned int entry_count,
    unsigned int star_offset,
    unsigned int* __restrict__ router_offsets,
    unsigned int* __restrict__ router_counts,
    unsigned int* __restrict__ executor_offsets,
    unsigned int* __restrict__ executor_counts,
    unsigned int* __restrict__ validator_offsets,
    unsigned int* __restrict__ validator_counts,
    unsigned int* __restrict__ anti_pattern_offsets,
    unsigned int* __restrict__ anti_pattern_counts
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= entry_count) return;

    const unsigned int star_index = star_offset + i;
    const unsigned char* src = input + (i * CATALOG_INPUT_ENTRY_BYTES);
    unsigned char* dst = galaxy_table + (star_index * STAR_RECORD_BYTES);

    float embedding16[16];
    float embedding32[32];
    #pragma unroll
    for (int d = 0; d < 16; ++d) {
        embedding16[d] = read_f32(src, d * 4);
    }
    duplicate_and_normalize_embedding16_device(embedding16, embedding32);
    #pragma unroll
    for (int d = 0; d < 32; ++d) {
        *reinterpret_cast<float*>(dst + STAR_EMBEDDING_OFFSET + (d * 4)) = embedding32[d];
    }

    *reinterpret_cast<unsigned int*>(dst + STAR_GALAXY_ID_OFFSET) = read_u32(src, 64);
    *reinterpret_cast<unsigned int*>(dst + STAR_TYPE_OFFSET) = read_u32(src, 68);
    *reinterpret_cast<unsigned int*>(dst + STAR_SELECTION_ROLE_OFFSET) = read_u32(src, 72);
    *reinterpret_cast<unsigned int*>(dst + STAR_LAYER_ID_OFFSET) = read_u32(src, 76);
    *reinterpret_cast<unsigned int*>(dst + STAR_FLAGS_OFFSET) = read_u32(src, 80);
    *reinterpret_cast<unsigned int*>(dst + STAR_ANSWER_ELIGIBLE_OFFSET) = read_u32(src, 84);
    *reinterpret_cast<int*>(dst + STAR_SEMANTIC_POLARITY_OFFSET) = read_i32(src, 88);
    *reinterpret_cast<float*>(dst + STAR_SEMANTIC_FOCUS_OFFSET) = read_f32(src, 92);
    *reinterpret_cast<float*>(dst + STAR_SEMANTIC_MASS_OFFSET) = read_f32(src, 96);
    *reinterpret_cast<float*>(dst + STAR_ATTRACTIVE_PRIOR_OFFSET) = read_f32(src, 100);
    *reinterpret_cast<float*>(dst + STAR_REPULSIVE_PRIOR_OFFSET) = read_f32(src, 104);
    *reinterpret_cast<unsigned int*>(dst + STAR_ROUTE_POLICY_OFFSET) = read_u32(src, 108);
    *reinterpret_cast<unsigned long long*>(dst + STAR_STAR_HASH_OFFSET) = read_u64(src, 112);

    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        *reinterpret_cast<float*>(dst + STAR_POSITION_OFFSET + (d * 4)) = read_f32(src, 120 + (d * 4));
        *reinterpret_cast<float*>(dst + STAR_VELOCITY_OFFSET + (d * 4)) = 0.0f;
    }

    *reinterpret_cast<unsigned int*>(dst + STAR_ROUTER_REF_COUNT_OFFSET) = 0u;
    *reinterpret_cast<unsigned int*>(dst + STAR_EXECUTOR_REF_COUNT_OFFSET) = 0u;
    *reinterpret_cast<unsigned int*>(dst + STAR_VALIDATOR_REF_COUNT_OFFSET) = 0u;
    *reinterpret_cast<unsigned int*>(dst + STAR_ANTI_PATTERN_REF_COUNT_OFFSET) = 0u;

    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        *reinterpret_cast<unsigned int*>(dst + STAR_ROUTER_REFS_OFFSET + (slot * 4)) = STAR_NULL_REF;
        *reinterpret_cast<unsigned int*>(dst + STAR_EXECUTOR_REFS_OFFSET + (slot * 4)) = STAR_NULL_REF;
        *reinterpret_cast<unsigned int*>(dst + STAR_VALIDATOR_REFS_OFFSET + (slot * 4)) = STAR_NULL_REF;
        *reinterpret_cast<unsigned int*>(dst + STAR_ANTI_PATTERN_REFS_OFFSET + (slot * 4)) = STAR_NULL_REF;
    }

    router_offsets[star_index] = 0u;
    router_counts[star_index] = 0u;
    executor_offsets[star_index] = 0u;
    executor_counts[star_index] = 0u;
    validator_offsets[star_index] = 0u;
    validator_counts[star_index] = 0u;
    anti_pattern_offsets[star_index] = 0u;
    anti_pattern_counts[star_index] = 0u;
}
