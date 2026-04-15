#include <stdint.h>

#include "lane_perf_ring.cu"
#include "swarm_perf_calibration_reader.cuh"

static __device__ __forceinline__ uint32_t sleep_perf_bucket(uint32_t n_active) {
    if (n_active == 0u) {
        return 0u;
    }
    const uint32_t bucket = n_active - 1u;
    return bucket > 15u ? 15u : bucket;
}

static __device__ __forceinline__ uint32_t sleep_perf_abs_q20(float value) {
    const float scaled = (value < 0.0f ? -value : value) * 1048576.0f;
    return scaled <= 0.0f ? 0u : static_cast<uint32_t>(scaled + 0.5f);
}

extern "C" __global__ void k3d_sleep_perf_consume(
    const LanePerf* __restrict__ ring,
    uint32_t ring_size,
    uint32_t ring_head,
    SwarmPerfCalibration* __restrict__ calibration
) {
    if (calibration == nullptr || ring == nullptr || ring_size == 0u) {
        return;
    }

    __shared__ uint32_t bucket_samples[16];
    __shared__ uint32_t bucket_utility_q20[16];
    __shared__ uint32_t utility_peak_q20;
    __shared__ uint32_t sample_count_total;

    if (threadIdx.x < 16u) {
        bucket_samples[threadIdx.x] = 0u;
        bucket_utility_q20[threadIdx.x] = 0u;
    }
    if (threadIdx.x == 0u) {
        utility_peak_q20 = 0u;
        sample_count_total = ring_head > ring_size ? ring_size : ring_head;
    }
    __syncthreads();

    const uint32_t total = sample_count_total;
    for (uint32_t index = threadIdx.x; index < total; index += blockDim.x) {
        const LanePerf sample = ring[index];
        const uint32_t bucket = sleep_perf_bucket(sample.n_active);
        const uint32_t utility_q20 = sleep_perf_abs_q20(sample.belief_delta);
        atomicAdd(&bucket_samples[bucket], 1u);
        atomicAdd(&bucket_utility_q20[bucket], utility_q20);
        atomicMax(&utility_peak_q20, utility_q20);
    }
    __syncthreads();

    if (threadIdx.x == 0u) {
        uint32_t best_bucket = 0u;
        uint32_t best_avg_q20 = 0u;
        for (uint32_t bucket = 0u; bucket < 16u; ++bucket) {
            const uint32_t samples = bucket_samples[bucket];
            const uint32_t utility_sum = bucket_utility_q20[bucket];
            calibration->bucket_samples[bucket] = samples;
            calibration->bucket_utility_q20[bucket] = utility_sum;
            const uint32_t avg_q20 = samples == 0u ? 0u : (utility_sum / samples);
            if (avg_q20 >= best_avg_q20) {
                best_avg_q20 = avg_q20;
                best_bucket = bucket;
            }
        }
        calibration->n_hint = best_bucket + 1u;
        calibration->sample_count_total = total;
        calibration->last_tick_epoch = ring_head;
        calibration->utility_peak_q20 = utility_peak_q20;
        calibration->_pad[0] = 0u;
        calibration->_pad[1] = 0u;
        calibration->_pad[2] = 0u;
        calibration->_pad[3] = 0u;
        __threadfence_system();
    }
}
