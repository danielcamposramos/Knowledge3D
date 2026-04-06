#include <stdint.h>

#include "device_functions.cuh"

#define REF_TUPLE_BYTES 16u

static __device__ __forceinline__ unsigned int tuple_read_u32(
    const unsigned char* ptr,
    const unsigned int offset
) {
    return *reinterpret_cast<const unsigned int*>(ptr + offset);
}

static __device__ __forceinline__ void tuple_write_u32(
    unsigned char* ptr,
    const unsigned int offset,
    const unsigned int value
) {
    *reinterpret_cast<unsigned int*>(ptr + offset) = value;
}

extern "C" __global__ void reverse_symlink_expand(
    const unsigned char* __restrict__ forward_ref_tuples,
    unsigned int ref_count,
    unsigned int star_count,
    unsigned int* __restrict__ router_counts,
    unsigned int* __restrict__ executor_counts,
    unsigned int* __restrict__ validator_counts,
    unsigned int* __restrict__ anti_pattern_counts,
    unsigned char* __restrict__ expanded_ref_tuples
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ref_count) {
        return;
    }

    const unsigned char* src = forward_ref_tuples + (i * REF_TUPLE_BYTES);
    unsigned char* forward_dst = expanded_ref_tuples + (i * REF_TUPLE_BYTES);
    unsigned char* reverse_dst = expanded_ref_tuples + ((ref_count + i) * REF_TUPLE_BYTES);

    const unsigned int source_index = tuple_read_u32(src, 0u);
    const unsigned int role_type = tuple_read_u32(src, 4u);
    const unsigned int target_index = tuple_read_u32(src, 8u);
    const unsigned int slot = tuple_read_u32(src, 12u);

    tuple_write_u32(forward_dst, 0u, source_index);
    tuple_write_u32(forward_dst, 4u, role_type);
    tuple_write_u32(forward_dst, 8u, target_index);
    tuple_write_u32(forward_dst, 12u, slot);

    tuple_write_u32(reverse_dst, 0u, star_count);
    tuple_write_u32(reverse_dst, 4u, role_type);
    tuple_write_u32(reverse_dst, 8u, GALAXY_NULL_REF);
    tuple_write_u32(reverse_dst, 12u, 0u);

    if (source_index >= star_count || target_index >= star_count) {
        return;
    }

    unsigned int reverse_role_type = 0u;
    unsigned int reverse_slot = 0u;
    switch (role_type) {
        case 0u:
            reverse_role_type = 0u;
            reverse_slot = atomicAdd(router_counts + target_index, 1u);
            break;
        case 1u:
            reverse_role_type = 0u;
            reverse_slot = atomicAdd(router_counts + target_index, 1u);
            break;
        case 2u:
            reverse_role_type = 1u;
            reverse_slot = atomicAdd(executor_counts + target_index, 1u);
            break;
        case 3u:
            reverse_role_type = 3u;
            reverse_slot = atomicAdd(anti_pattern_counts + target_index, 1u);
            break;
        default:
            return;
    }

    tuple_write_u32(reverse_dst, 0u, target_index);
    tuple_write_u32(reverse_dst, 4u, reverse_role_type);
    tuple_write_u32(reverse_dst, 8u, source_index);
    tuple_write_u32(reverse_dst, 12u, reverse_slot);
}
