#pragma once

#include <stdint.h>

#define K3D_REASONING_LANE_OUTPUT_BYTES 64u
#define K3D_REASONING_PARADIGM_MAX 16u

enum ReasoningParadigmSlot : uint32_t {
    REASONING_SLOT_NONE = 0u,
    REASONING_SLOT_CBR = 1u,
    REASONING_SLOT_SUPERPOS = 2u,
    REASONING_SLOT_BIDUCE = 3u,
    REASONING_SLOT_EBELIEF = 4u,
    REASONING_SLOT_RETE = 5u,
    REASONING_SLOT_TABLEAUX = 6u,
    REASONING_SLOT_RESOLUTION = 7u,
    REASONING_SLOT_ALPCHAIN = 8u,
    REASONING_SLOT_DPLL = 9u,
    REASONING_SLOT_CTX_SWITCH = 10u,
    REASONING_SLOT_SUBSUME = 11u,
    REASONING_SLOT_UNIFY = 12u,
};

struct __align__(16) ReasoningTickIO {
    const uint8_t* __restrict__ galaxy_atlas;
    uint32_t phys_lane_id;
    uint32_t tick_seed;
    uint32_t paradigm_slot;
    uint32_t query_handle;
    uint32_t context_id;
    int8_t ethical_trit;
    uint8_t _pad0[3];
};

static_assert(sizeof(ReasoningTickIO) == 32, "ReasoningTickIO ABI must remain 32 bytes");

struct __align__(16) ReasoningLaneOutput {
    uint32_t halt_flag;
    uint32_t result_handle;
    uint32_t belief_q15;
    uint32_t _pad0;
    uint8_t payload[K3D_REASONING_LANE_OUTPUT_BYTES - 16u];
};

static_assert(
    sizeof(ReasoningLaneOutput) == K3D_REASONING_LANE_OUTPUT_BYTES,
    "ReasoningLaneOutput ABI must remain 64 bytes"
);
