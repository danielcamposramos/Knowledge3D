#define K3D_TRM_DEFINE_TRANSITION_TABLE 1

#include <cuda_runtime.h>

#include "../cuda/trm_game_loop.cuh"

extern "C" __global__ void trm_state_machine_step(
    EntityHotPath* entity_hot_paths,
    TRMStateMachine* state_machines,
    GPUEvent* ring_buffer,
    uint32_t* head_ptr,
    uint32_t* tail_ptr,
    uint32_t entity_count,
    float delta_time,
    unsigned long long tick
) {
    const uint32_t entity_idx = static_cast<uint32_t>(blockIdx.x);
    if (threadIdx.x != 0 || entity_idx >= entity_count) {
        return;
    }

    trm_state_machine_step_device(
        &entity_hot_paths[entity_idx],
        &state_machines[entity_idx],
        ring_buffer,
        head_ptr,
        tail_ptr,
        delta_time,
        static_cast<uint64_t>(tick)
    );
}
