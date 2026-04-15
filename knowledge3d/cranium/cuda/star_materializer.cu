#include <stdint.h>
#include <math.h>

#include "device_functions.cuh"



#define CATALOG_INPUT_ENTRY_BYTES          360
#define CATALOG_INPUT_CONTEXT_ID_OFFSET    340
#define CATALOG_INPUT_ETHICAL_TRIT_OFFSET  344

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
    unsigned char* dst = galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES);

    float embedding64[GPU_TASK_EMBED_DIMS];
    float normalized_embedding[GPU_TASK_EMBED_DIMS];
    #pragma unroll
    for (int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) {
        embedding64[d] = read_f32(src, d * 4);
    }
    normalize_embedding64_device(embedding64, normalized_embedding);
    #pragma unroll
    for (int d = 0; d < GPU_TASK_EMBED_DIMS; ++d) {
        *reinterpret_cast<float*>(dst + GALAXY_STAR_EMBEDDING_OFFSET + (d * 4)) = normalized_embedding[d];
    }

    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_GALAXY_ID_OFFSET) = read_u32(src, 256);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_TYPE_OFFSET) = read_u32(src, 260);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_SELECTION_ROLE_OFFSET) = read_u32(src, 264);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_LAYER_ID_OFFSET) = read_u32(src, 268);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_FLAGS_OFFSET) = read_u32(src, 272);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_ANSWER_ELIGIBLE_OFFSET) = read_u32(src, 276);
    *reinterpret_cast<int*>(dst + GALAXY_STAR_SEMANTIC_POLARITY_OFFSET) = read_i32(src, 280);
    *reinterpret_cast<float*>(dst + GALAXY_STAR_SEMANTIC_FOCUS_OFFSET) = read_f32(src, 284);
    *reinterpret_cast<float*>(dst + GALAXY_STAR_SEMANTIC_MASS_OFFSET) = read_f32(src, 288);
    *reinterpret_cast<float*>(dst + GALAXY_STAR_ATTRACTIVE_PRIOR_OFFSET) = read_f32(src, 292);
    *reinterpret_cast<float*>(dst + GALAXY_STAR_REPULSIVE_PRIOR_OFFSET) = read_f32(src, 296);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_ROUTE_POLICY_OFFSET) = read_u32(src, 300);
    *reinterpret_cast<unsigned long long*>(dst + GALAXY_STAR_HASH_OFFSET) = read_u64(src, 304);

    #pragma unroll
    for (int d = 0; d < 3; ++d) {
        *reinterpret_cast<float*>(dst + GALAXY_STAR_POSITION_OFFSET + (d * 4)) = read_f32(src, 312 + (d * 4));
        *reinterpret_cast<float*>(dst + GALAXY_STAR_VELOCITY_OFFSET + (d * 4)) = 0.0f;
    }
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_META_RULE_ADDR_OFFSET) = read_u32(src, 324);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_PROGRAM_FLAGS_OFFSET) = read_u32(src, 328);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_PROGRAM_LENGTH_OFFSET) = read_u32(src, 332);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_PROGRAM_OPCODE_COUNT_OFFSET) = read_u32(src, 336);
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_CONTEXT_ID_OFFSET) = read_u32(src, CATALOG_INPUT_CONTEXT_ID_OFFSET);
    *reinterpret_cast<int8_t*>(dst + GALAXY_STAR_ETHICAL_TRIT_OFFSET) =
        static_cast<int8_t>(read_i32(src, CATALOG_INPUT_ETHICAL_TRIT_OFFSET));

    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_ROUTER_REF_COUNT_OFFSET) = 0u;
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_EXECUTOR_REF_COUNT_OFFSET) = 0u;
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_VALIDATOR_REF_COUNT_OFFSET) = 0u;
    *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_ANTI_PATTERN_REF_COUNT_OFFSET) = 0u;

    #pragma unroll
    for (int slot = 0; slot < 2; ++slot) {
        *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_ROUTER_REFS_OFFSET + (slot * 4)) = GALAXY_NULL_REF;
        *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_EXECUTOR_REFS_OFFSET + (slot * 4)) = GALAXY_NULL_REF;
        *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_VALIDATOR_REFS_OFFSET + (slot * 4)) = GALAXY_NULL_REF;
        *reinterpret_cast<unsigned int*>(dst + GALAXY_STAR_ANTI_PATTERN_REFS_OFFSET + (slot * 4)) = GALAXY_NULL_REF;
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
