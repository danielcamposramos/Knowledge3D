#include <stdint.h>

#include "device_functions.cuh"

static __device__ __forceinline__ unsigned long long read_u64_star(
    const unsigned char* ptr,
    const unsigned int offset
) {
    return *reinterpret_cast<const unsigned long long*>(ptr + offset);
}

extern "C" __global__ void star_hash_index_build(
    const unsigned char* __restrict__ galaxy_table,
    unsigned int star_count,
    unsigned long long* __restrict__ hash_keys,
    unsigned int* __restrict__ hash_values,
    unsigned int table_capacity,
    unsigned long long* __restrict__ collision_flag
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= star_count || table_capacity == 0u) {
        return;
    }

    const unsigned char* star = galaxy_table + (i * GALAXY_STAR_RECORD_BYTES);
    const unsigned long long star_hash = read_u64_star(star, GALAXY_STAR_HASH_OFFSET);
    if (star_hash == 0ull) {
        atomicCAS(reinterpret_cast<unsigned long long*>(collision_flag), 0ull, 1ull);
        return;
    }

    unsigned int slot = static_cast<unsigned int>(star_hash % static_cast<unsigned long long>(table_capacity));
    for (unsigned int probe = 0u; probe < table_capacity; ++probe) {
        unsigned long long previous = atomicCAS(
            reinterpret_cast<unsigned long long*>(hash_keys + slot),
            0ull,
            star_hash
        );
        if (previous == 0ull || previous == star_hash) {
            unsigned int existing = hash_values[slot];
            if (previous == star_hash && existing != i) {
                atomicCAS(reinterpret_cast<unsigned long long*>(collision_flag), 0ull, star_hash);
            } else {
                hash_values[slot] = i;
            }
            return;
        }
        slot = (slot + 1u) % table_capacity;
    }
    atomicCAS(reinterpret_cast<unsigned long long*>(collision_flag), 0ull, star_hash);
}
