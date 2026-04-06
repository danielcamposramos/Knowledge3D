#include <cuda_runtime.h>
#include "../cuda/trm_recursive_core.cuh"

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
    float* __restrict__ workspace
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
}
