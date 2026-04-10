#include <cuda_runtime.h>
#include "../cuda/trm_recursive_core.cuh"
#include "../kernels/entity_hot_path.h"

__device__ __forceinline__ void trm_physics_phase_stub(
    const void* physics_soa_ptr,
    const void* contact_soa_ptr,
    const void* event_queue_ptr,
    unsigned int body_count,
    float physics_dt,
    unsigned int solver_iterations
) {
    // PHYSICS_PHASE boundary for the sovereign rigid-body path.
    // The full composed dispatch is driven by the modular RPN physics opcodes
    // (0x150–0x17F) and bound SOA buffers. This stub keeps the fused-step slot
    // explicit in source while the PTX/module wiring is promoted incrementally.
    (void)physics_soa_ptr;
    (void)contact_soa_ptr;
    (void)event_queue_ptr;
    (void)body_count;
    (void)physics_dt;
    (void)solver_iterations;
}

__device__ __forceinline__ void trm_behavior_phase_stub(
    const EntityHotPath* __restrict__ entity_hot_paths,
    unsigned int entity_count,
    unsigned int frame_counter
) {
    // BEHAVIOR_PHASE boundary for avatar/entity execution.
    (void)entity_hot_paths;
    (void)entity_count;
    (void)frame_counter;
}

extern "C" __global__ void trm_step_fused(
    const float* __restrict__ q,
    const float* __restrict__ y,
    const float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ z_new,
    float* __restrict__ y_new,
    float* __restrict__ workspace,
    const void* __restrict__ physics_soa_ptr,
    const void* __restrict__ contact_soa_ptr,
    const void* __restrict__ event_queue_ptr,
    unsigned int body_count,
    float physics_dt,
    unsigned int solver_iterations,
    const void* __restrict__ entity_hot_path_ptr,
    unsigned int entity_count,
    unsigned int frame_counter
) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    __shared__ int steps_taken;
    __shared__ float drift_value;

    if (tid == 0) {
        steps_taken = 0;
        drift_value = 0.0f;
    }
    __syncthreads();

    for (int index = tid; index < GPU_TASK_TRM_DIMS; index += stride) {
        y_new[index] = y[index];
        z_new[index] = z[index];
    }
    __syncthreads();

    trm_recursive_core_device(
        q,
        y_new,
        z_new,
        W1,
        W2,
        W3,
        W4,
        workspace,
        tid,
        stride,
        1,
        0.0f,
        &steps_taken,
        &drift_value,
        &steps_taken,
        &drift_value
    );

    __syncthreads();
    if (tid == 0) {
        // SWARM_PHASE completed above. The sovereign physics phase belongs here,
        // before any draw/frustum/LOD surface is invoked.
        trm_physics_phase_stub(
            physics_soa_ptr,
            contact_soa_ptr,
            event_queue_ptr,
            body_count,
            physics_dt,
            solver_iterations
        );
        trm_behavior_phase_stub(
            static_cast<const EntityHotPath*>(entity_hot_path_ptr),
            entity_count,
            frame_counter
        );
    }
}
