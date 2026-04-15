#pragma once

#include <stdint.h>

__device__ __forceinline__ void halting_gate_step(
    uint32_t n_active,
    uint32_t phys_lane_id,
    bool local_halt,
    volatile uint32_t* halting_flag,
    volatile uint32_t* g_halting_counter,
    uint32_t complete_flag
) {
    if (local_halt) {
        atomicAdd((unsigned int*)g_halting_counter, 1u);
    }
    __threadfence_system();
    if (phys_lane_id == 0u) {
        while (*g_halting_counter < n_active) {
        }
        *halting_flag = complete_flag;
        __threadfence_system();
    }
    __syncthreads();
}
