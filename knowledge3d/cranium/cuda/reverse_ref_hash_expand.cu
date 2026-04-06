#include <stdint.h>

#include "device_functions.cuh"

#define FEED_SOURCE_REF_BYTES 24u
#define BUILD_REF_HASH_BYTES 16u
#define FEED_SOURCE_REVERSE_ROLE_OFFSET 16u

static __device__ __forceinline__ unsigned int read_u32_ref(
    const unsigned char* ptr,
    const unsigned int offset
) {
    return *reinterpret_cast<const unsigned int*>(ptr + offset);
}

static __device__ __forceinline__ unsigned long long read_u64_ref(
    const unsigned char* ptr,
    const unsigned int offset
) {
    return *reinterpret_cast<const unsigned long long*>(ptr + offset);
}

static __device__ __forceinline__ void write_u32_ref(
    unsigned char* ptr,
    const unsigned int offset,
    const unsigned int value
) {
    *reinterpret_cast<unsigned int*>(ptr + offset) = value;
}

static __device__ __forceinline__ void write_u64_ref(
    unsigned char* ptr,
    const unsigned int offset,
    const unsigned long long value
) {
    *reinterpret_cast<unsigned long long*>(ptr + offset) = value;
}

static __device__ __forceinline__ unsigned int hash_lookup_device(
    const unsigned long long target_hash,
    const unsigned long long* __restrict__ hash_keys,
    const unsigned int* __restrict__ hash_values,
    const unsigned int table_capacity
) {
    if (target_hash == 0ull || table_capacity == 0u) {
        return GALAXY_NULL_REF;
    }
    unsigned int slot = static_cast<unsigned int>(target_hash % static_cast<unsigned long long>(table_capacity));
    for (unsigned int probe = 0u; probe < table_capacity; ++probe) {
        const unsigned long long candidate = hash_keys[slot];
        if (candidate == target_hash) {
            return hash_values[slot];
        }
        if (candidate == 0ull) {
            return GALAXY_NULL_REF;
        }
        slot = (slot + 1u) % table_capacity;
    }
    return GALAXY_NULL_REF;
}

extern "C" __global__ void reverse_ref_hash_expand(
    const unsigned char* __restrict__ feed_source_ref_rows,
    unsigned int ref_count,
    unsigned int star_count,
    const unsigned long long* __restrict__ hash_keys,
    const unsigned int* __restrict__ hash_values,
    unsigned int table_capacity,
    const unsigned char* __restrict__ galaxy_table,
    unsigned char* __restrict__ build_ref_hash_rows,
    unsigned int* __restrict__ unresolved_source_indices,
    unsigned long long* __restrict__ unresolved_target_hashes,
    unsigned int* __restrict__ unresolved_count
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ref_count) {
        return;
    }

    const unsigned char* src = feed_source_ref_rows + (i * FEED_SOURCE_REF_BYTES);
    const unsigned int source_index = read_u32_ref(src, 0u);
    const unsigned int role_type = read_u32_ref(src, 4u);
    const unsigned long long target_hash = read_u64_ref(src, 8u);
    const unsigned int reverse_role_type = read_u32_ref(src, FEED_SOURCE_REVERSE_ROLE_OFFSET);

    unresolved_source_indices[i] = GALAXY_NULL_REF;
    unresolved_target_hashes[i] = 0ull;

    unsigned char* forward_dst = build_ref_hash_rows + (i * BUILD_REF_HASH_BYTES);
    unsigned char* reverse_dst = build_ref_hash_rows + ((ref_count + i) * BUILD_REF_HASH_BYTES);

    write_u32_ref(forward_dst, 0u, star_count);
    write_u32_ref(forward_dst, 4u, role_type);
    write_u64_ref(forward_dst, 8u, 0ull);

    write_u32_ref(reverse_dst, 0u, star_count);
    write_u32_ref(reverse_dst, 4u, reverse_role_type);
    write_u64_ref(reverse_dst, 8u, 0ull);

    if (source_index >= star_count) {
        const unsigned int error_slot = atomicAdd(unresolved_count, 1u);
        unresolved_source_indices[error_slot] = source_index;
        unresolved_target_hashes[error_slot] = target_hash;
        return;
    }

    const unsigned int target_index = hash_lookup_device(target_hash, hash_keys, hash_values, table_capacity);
    if (target_index == GALAXY_NULL_REF || target_index >= star_count) {
        const unsigned int error_slot = atomicAdd(unresolved_count, 1u);
        unresolved_source_indices[error_slot] = source_index;
        unresolved_target_hashes[error_slot] = target_hash;
        return;
    }

    const unsigned char* source_star = galaxy_table + (source_index * GALAXY_STAR_RECORD_BYTES);
    const unsigned long long source_hash =
        *reinterpret_cast<const unsigned long long*>(source_star + GALAXY_STAR_HASH_OFFSET);
    if (source_hash == 0ull) {
        const unsigned int error_slot = atomicAdd(unresolved_count, 1u);
        unresolved_source_indices[error_slot] = source_index;
        unresolved_target_hashes[error_slot] = target_hash;
        return;
    }

    write_u32_ref(forward_dst, 0u, source_index);
    write_u32_ref(forward_dst, 4u, role_type);
    write_u64_ref(forward_dst, 8u, target_hash);

    write_u32_ref(reverse_dst, 0u, target_index);
    write_u32_ref(reverse_dst, 4u, reverse_role_type);
    write_u64_ref(reverse_dst, 8u, source_hash);
}
