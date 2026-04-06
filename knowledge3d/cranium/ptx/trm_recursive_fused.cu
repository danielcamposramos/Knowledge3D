#include <cuda_runtime.h>
#include "../cuda/trm_recursive_core.cuh"

extern "C" __global__ void trm_recursive_fused(
    const float* __restrict__ q,
    float* __restrict__ y,
    float* __restrict__ z,
    const float* __restrict__ W1,
    const float* __restrict__ W2,
    const float* __restrict__ W3,
    const float* __restrict__ W4,
    float* __restrict__ workspace,
    int* __restrict__ steps_out,
    float* __restrict__ drift_out,
    int max_steps,
    float epsilon
) {
    __shared__ int steps_shared;
    __shared__ float drift_shared;
    trm_recursive_core_device(
        q,
        y,
        z,
        W1,
        W2,
        W3,
        W4,
        workspace,
        static_cast<int>(threadIdx.x),
        static_cast<int>(blockDim.x),
        max_steps,
        epsilon,
        &steps_shared,
        &drift_shared,
        steps_out,
        drift_out
    );
}
