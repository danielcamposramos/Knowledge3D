#pragma once

#include "cas_star_node.h"
#include <stdint.h>

#define SAS_HASHCONS_SIZE  (1u << 20)
#define SAS_HASHCONS_EMPTY 0xFFFFFFFFu

struct HashconsSlot {
    uint32_t key_opcode;
    uint32_t key_child0;
    uint32_t key_child1;
    uint32_t key_flags;
    uint32_t pool_idx;
    uint32_t _pad;
};

__device__ __forceinline__ uint32_t hashcons_slot(
    uint32_t opcode, uint32_t child0, uint32_t child1, uint32_t flags)
{
    uint32_t h = opcode * 2654435761u ^ child0 * 2246822519u
               ^ child1 * 3266489917u ^ flags * 668265263u;
    return h & (SAS_HASHCONS_SIZE - 1u);
}

__device__ __forceinline__ uint32_t hashcons_lookup(
    const HashconsSlot* __restrict__ table,
    uint32_t opcode, uint32_t child0, uint32_t child1, uint32_t flags)
{
    uint32_t slot = hashcons_slot(opcode, child0, child1, flags);
    for (uint32_t probe = 0; probe < 32u; ++probe) {
        const HashconsSlot& s = table[(slot + probe) & (SAS_HASHCONS_SIZE - 1u)];
        if (s.pool_idx == SAS_HASHCONS_EMPTY) return SAS_HASHCONS_EMPTY;
        if (s.key_opcode == opcode && s.key_child0 == child0 &&
            s.key_child1 == child1 && s.key_flags == flags) {
            return s.pool_idx;
        }
    }
    return SAS_HASHCONS_EMPTY;
}

__device__ __forceinline__ uint32_t hashcons_insert(
    HashconsSlot* __restrict__ table,
    uint32_t opcode, uint32_t child0, uint32_t child1, uint32_t flags,
    uint32_t pool_idx)
{
    uint32_t slot = hashcons_slot(opcode, child0, child1, flags);
    for (uint32_t probe = 0; probe < 32u; ++probe) {
        const uint32_t s = (slot + probe) & (SAS_HASHCONS_SIZE - 1u);
        uint32_t old = atomicCAS(&table[s].pool_idx, SAS_HASHCONS_EMPTY, pool_idx);
        if (old == SAS_HASHCONS_EMPTY) {
            table[s].key_opcode = opcode;
            table[s].key_child0 = child0;
            table[s].key_child1 = child1;
            table[s].key_flags = flags;
            return pool_idx;
        }
        if (table[s].key_opcode == opcode && table[s].key_child0 == child0 &&
            table[s].key_child1 == child1 && table[s].key_flags == flags) {
            return old;
        }
    }
    return pool_idx;
}
