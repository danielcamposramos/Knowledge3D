#pragma once

#include <stdint.h>

#include "reasoning_tick_io.cuh"
#include "swarm_perf_calibration_reader.cuh"

struct __align__(16) SwarmTickControl {
    uint32_t vram_free_mib;
    uint32_t t_remaining_us;
    uint32_t n_cand_frustum;
    uint32_t h_belief_q10;
    uint32_t n_floor;
    uint32_t n_hard_max;
    uint32_t sleep_calibration_n_hint;
    uint32_t paradigm_mask;
};

static_assert(sizeof(SwarmTickControl) == 32, "SwarmTickControl ABI must remain 32 bytes");

static __device__ __forceinline__ uint32_t swarm_min_u32(uint32_t a, uint32_t b) {
    return a < b ? a : b;
}

static __device__ __forceinline__ uint32_t swarm_max_u32(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

static __device__ __forceinline__ uint32_t swarm_clamp_u32(uint32_t value, uint32_t minimum, uint32_t maximum) {
    return swarm_min_u32(swarm_max_u32(value, minimum), maximum);
}

static __device__ __forceinline__ uint32_t swarm_apply_hint_window(uint32_t candidate, uint32_t hint) {
    if (hint == 0u) {
        return candidate;
    }
    const uint32_t lower = swarm_max_u32(1u, (hint * 3u) / 4u);
    const uint32_t upper = swarm_max_u32(lower, (hint * 5u) / 4u);
    return swarm_clamp_u32(candidate, lower, upper);
}

static __device__ __forceinline__ uint32_t swarm_select_dynamic_n(
    const SwarmTickControl* control,
    const SwarmPerfCalibration* calibration
) {
    if (control == nullptr) {
        return 1u;
    }
    const uint32_t n_floor = swarm_max_u32(1u, control->n_floor);
    const uint32_t n_hard_max = swarm_max_u32(n_floor, control->n_hard_max);
    const uint32_t n_cand_frustum = swarm_max_u32(n_floor, control->n_cand_frustum);

    const uint32_t n_vram = ((control->vram_free_mib * 90u) / 100u) / 15u;
    const uint32_t boost_q10 = 1024u + swarm_min_u32(control->h_belief_q10, 1024u);
    const uint32_t n_entropy = (n_vram * boost_q10) >> 10;
    const uint32_t n_deadline = control->t_remaining_us / 48u;

    uint32_t n = swarm_min_u32(n_entropy, swarm_min_u32(n_deadline, n_cand_frustum));
    uint32_t hint = control->sleep_calibration_n_hint;
    if (hint == 0u) {
        hint = swarm_perf_calibration_hint(calibration);
    }
    n = swarm_apply_hint_window(n, hint);
    return swarm_clamp_u32(n, n_floor, n_hard_max);
}

static __device__ __forceinline__ uint32_t swarm_assign_paradigm_slot(
    const SwarmTickControl* control,
    uint32_t phys_lane_id
) {
    if (control == nullptr) {
        return REASONING_SLOT_NONE;
    }
    const uint32_t mask = control->paradigm_mask;
    const uint32_t active_slots = __popc(mask);
    if (active_slots == 0u) {
        return REASONING_SLOT_NONE;
    }
    const uint32_t slot_index = phys_lane_id % active_slots;
    uint32_t seen = 0u;
    for (uint32_t bit = 0u; bit < K3D_REASONING_PARADIGM_MAX; ++bit) {
        if (((mask >> bit) & 1u) == 0u) {
            continue;
        }
        if (seen == slot_index) {
            return bit;
        }
        ++seen;
    }
    return REASONING_SLOT_NONE;
}
