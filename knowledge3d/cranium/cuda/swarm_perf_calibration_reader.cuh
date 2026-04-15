#pragma once

#include <stdint.h>

struct __align__(16) SwarmPerfCalibration {
    uint32_t n_hint;
    uint32_t sample_count_total;
    uint32_t last_tick_epoch;
    uint32_t utility_peak_q20;
    uint32_t bucket_samples[16];
    uint32_t bucket_utility_q20[16];
    uint32_t _pad[4];
};

static_assert(sizeof(SwarmPerfCalibration) == 160, "SwarmPerfCalibration ABI must remain stable");

__device__ __forceinline__ uint32_t swarm_perf_calibration_hint(
    const SwarmPerfCalibration* calibration
) {
    if (calibration == nullptr) {
        return 0u;
    }
    return calibration->n_hint;
}
