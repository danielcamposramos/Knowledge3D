#include "device_functions.cuh"

extern "C" __global__ void sleep_time_micro(
    unsigned char* __restrict__ brain_state,
    int outcome_signal,
    unsigned char* __restrict__ galaxy_table,
    unsigned int chosen_star_index
) {
    float* reasoning = reinterpret_cast<float*>(brain_state + BRAIN_REASONING_OFFSET);
    float* chains = reinterpret_cast<float*>(brain_state + BRAIN_CHAINS_OFFSET);
    float* specialist_trace = reinterpret_cast<float*>(brain_state + BRAIN_SPECIALIST_TRACE_OFFSET);
    const float outcome = static_cast<float>(outcome_signal);

    for (int index = threadIdx.x; index < GPU_TASK_EMBED_DIMS; index += blockDim.x) {
        if (outcome > 0.5f) {
            reasoning[index] = tanhf(reasoning[index] * 1.05f);
        } else if (outcome < -0.5f) {
            reasoning[index] *= 0.85f;
        } else {
            reasoning[index] *= 0.98f;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int chain = 0; chain < GPU_TASK_NUM_CHAINS; ++chain) {
            const float resonance = specialist_trace[chain];
            if (outcome > 0.5f) {
                specialist_trace[chain] = device_clamp01(resonance + 0.05f);
            } else if (outcome < -0.5f) {
                specialist_trace[chain] = device_clamp01(resonance - 0.03f);
            } else {
                specialist_trace[chain] *= 0.995f;
            }
        }
        *reinterpret_cast<signed char*>(brain_state + BRAIN_TERNARY_OFFSET) =
            static_cast<signed char>(outcome_signal > 0 ? 1 : (outcome_signal < 0 ? -1 : 0));
    }
    __syncthreads();

    if (threadIdx.x == 0 && outcome > 0.5f) {
        int best_chain = 0;
        int worst_chain = 0;
        float best_res = specialist_trace[0];
        float worst_res = specialist_trace[0];
        for (int chain = 1; chain < GPU_TASK_NUM_CHAINS; ++chain) {
            if (specialist_trace[chain] > best_res) {
                best_res = specialist_trace[chain];
                best_chain = chain;
            }
            if (specialist_trace[chain] < worst_res) {
                worst_res = specialist_trace[chain];
                worst_chain = chain;
            }
        }
        if (best_chain != worst_chain) {
            float* src = chains + (best_chain * GPU_TASK_EMBED_DIMS);
            float* dst = chains + (worst_chain * GPU_TASK_EMBED_DIMS);
            for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                dst[index] = tanhf((0.70f * dst[index]) + (0.30f * src[index]));
            }
        }
    }
    __syncthreads();

    if (galaxy_table != nullptr && threadIdx.x == 0) {
        const unsigned int flags = galaxy_read_flags(galaxy_table, chosen_star_index);
        if ((flags & GALAXY_STAR_FLAG_LEARNABLE) != 0u) {
            float* star_embedding = reinterpret_cast<float*>(
                galaxy_table + (chosen_star_index * GALAXY_STAR_RECORD_BYTES) + GALAXY_STAR_EMBEDDING_OFFSET
            );
            const float* brain_reasoning = reinterpret_cast<const float*>(
                brain_state + BRAIN_REASONING_OFFSET
            );
            const float nudge = (outcome_signal > 0) ? 0.02f : ((outcome_signal < 0) ? -0.01f : 0.0f);
            for (int index = 0; index < GPU_TASK_EMBED_DIMS; ++index) {
                star_embedding[index] = tanhf(star_embedding[index] + (nudge * brain_reasoning[index]));
            }
        }
    }
    __syncthreads();
}
