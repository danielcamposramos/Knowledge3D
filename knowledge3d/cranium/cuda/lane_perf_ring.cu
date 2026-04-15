#pragma once

#include <stdint.h>

#ifndef K3D_LANE_PERF_RING_ENTRIES
#define K3D_LANE_PERF_RING_ENTRIES (1u << 20)
#endif

struct __align__(16) LanePerf {
    uint32_t n_active;
    uint32_t entropy_input;
    float belief_delta;
    uint32_t cycles_consumed;
    uint8_t specialist_id;
    uint8_t _pad[3];
};

static_assert(sizeof(LanePerf) == 32, "LanePerf ABI must remain 32 bytes with 16-byte alignment");

__device__ __forceinline__ void lane_perf_write(
    LanePerf* ring,
    uint32_t* ring_head,
    uint32_t ring_mask,
    const LanePerf& sample
) {
    if (ring == nullptr || ring_head == nullptr || ring_mask == 0u) {
        return;
    }
    const uint32_t idx = atomicInc(ring_head, 0xFFFFFFFFu) & ring_mask;
    ring[idx] = sample;
}
