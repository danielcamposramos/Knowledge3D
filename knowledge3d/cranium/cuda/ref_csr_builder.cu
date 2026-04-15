#include <stdint.h>

#include "device_functions.cuh"

#define STAR_NULL_REF 0xFFFFFFFFu
#define ROLE_REF_LIMIT 2u
#define REF_TUPLE_BYTES 16

static __device__ __forceinline__ unsigned int tuple_u32(const unsigned char* ptr, const unsigned int offset) {
    return *reinterpret_cast<const unsigned int*>(ptr + offset);
}

extern "C" __global__ void ref_count_refs(
    const unsigned char* __restrict__ ref_tuples,
    unsigned int ref_count,
    unsigned int star_count,
    unsigned int* __restrict__ router_counts,
    unsigned int* __restrict__ executor_counts,
    unsigned int* __restrict__ validator_counts,
    unsigned int* __restrict__ anti_pattern_counts
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ref_count) return;

    const unsigned char* src = ref_tuples + (i * REF_TUPLE_BYTES);
    const unsigned int star_index = tuple_u32(src, 0);
    const unsigned int role_type = tuple_u32(src, 4);
    const unsigned int slot = tuple_u32(src, 12);
    if (star_index >= star_count) return;

    unsigned int* target = nullptr;
    switch (role_type) {
        case 0u: target = router_counts; break;
        case 1u: target = executor_counts; break;
        case 2u: target = validator_counts; break;
        case 3u: target = anti_pattern_counts; break;
        default: return;
    }
    atomicMax(target + star_index, slot + 1u);
}

extern "C" __global__ void ref_scan_offsets(
    unsigned int star_count,
    const unsigned int* __restrict__ router_counts,
    unsigned int* __restrict__ router_offsets,
    const unsigned int* __restrict__ executor_counts,
    unsigned int* __restrict__ executor_offsets,
    const unsigned int* __restrict__ validator_counts,
    unsigned int* __restrict__ validator_offsets,
    const unsigned int* __restrict__ anti_pattern_counts,
    unsigned int* __restrict__ anti_pattern_offsets
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    unsigned int running = 0u;
    for (unsigned int i = 0u; i < star_count; ++i) {
        router_offsets[i] = running;
        running += router_counts[i];
        executor_offsets[i] = running;
        running += executor_counts[i];
        validator_offsets[i] = running;
        running += validator_counts[i];
        anti_pattern_offsets[i] = running;
        running += anti_pattern_counts[i];
    }
}

extern "C" __global__ void ref_scatter_refs(
    unsigned char* __restrict__ galaxy_table,
    unsigned int* __restrict__ ref_indices,
    const unsigned char* __restrict__ ref_tuples,
    unsigned int ref_count,
    unsigned int star_count,
    const unsigned int* __restrict__ router_offsets,
    const unsigned int* __restrict__ executor_offsets,
    const unsigned int* __restrict__ validator_offsets,
    const unsigned int* __restrict__ anti_pattern_offsets
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ref_count) return;

    const unsigned char* src = ref_tuples + (i * REF_TUPLE_BYTES);
    const unsigned int star_index = tuple_u32(src, 0);
    const unsigned int role_type = tuple_u32(src, 4);
    const unsigned int ref_index = tuple_u32(src, 8);
    const unsigned int slot = tuple_u32(src, 12);
    if (star_index >= star_count) return;

    unsigned char* star_dst = galaxy_table + (star_index * GALAXY_STAR_RECORD_BYTES);
    unsigned int dst_offset = 0u;
    unsigned int inline_count_offset = 0u;
    unsigned int inline_refs_offset = 0u;

    switch (role_type) {
        case 0u:
            dst_offset = router_offsets[star_index] + slot;
            inline_count_offset = GALAXY_STAR_ROUTER_REF_COUNT_OFFSET;
            inline_refs_offset = GALAXY_STAR_ROUTER_REFS_OFFSET;
            break;
        case 1u:
            dst_offset = executor_offsets[star_index] + slot;
            inline_count_offset = GALAXY_STAR_EXECUTOR_REF_COUNT_OFFSET;
            inline_refs_offset = GALAXY_STAR_EXECUTOR_REFS_OFFSET;
            break;
        case 2u:
            dst_offset = validator_offsets[star_index] + slot;
            inline_count_offset = GALAXY_STAR_VALIDATOR_REF_COUNT_OFFSET;
            inline_refs_offset = GALAXY_STAR_VALIDATOR_REFS_OFFSET;
            break;
        case 3u:
            dst_offset = anti_pattern_offsets[star_index] + slot;
            inline_count_offset = GALAXY_STAR_ANTI_PATTERN_REF_COUNT_OFFSET;
            inline_refs_offset = GALAXY_STAR_ANTI_PATTERN_REFS_OFFSET;
            break;
        default:
            return;
    }

    ref_indices[dst_offset] = ref_index;
    if (slot < ROLE_REF_LIMIT) {
        *reinterpret_cast<unsigned int*>(star_dst + inline_refs_offset + (slot * 4)) = ref_index;
        atomicMax(reinterpret_cast<unsigned int*>(star_dst + inline_count_offset), slot + 1u);
    }
}
