/*
 * spatial_pool.cu - Sovereign spatial pooling kernels
 *
 * Provides mean pooling across the spatial dimensions (H×W) of a feature map.
 * Designed for character embedding extraction where activations are already on
 * the GPU. Includes RPN-style safety guards (NaN/Inf sanitation and relaxed
 * ±10 clipping) to ensure gradient stability.
 */

#include <cuda_runtime.h>

extern "C" __global__ void spatial_mean_pool(
    const float* __restrict__ features,  // [H, W, C] flattened as HW contiguous
    float* __restrict__ output,          // [C]
    int H,
    int W,
    int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }

    float accum = 0.0f;
    int stride = C;
    int spatial_size = H * W;

    for (int hw = 0; hw < spatial_size; ++hw) {
        float val = features[hw * stride + c];
        if (isnan(val) || isinf(val)) {
            val = 0.0f;
        }
        accum += val;
    }

    float mean = accum / static_cast<float>(spatial_size);
    if (isnan(mean) || isinf(mean)) {
        mean = 0.0f;
    }
    mean = fmaxf(fminf(mean, 10.0f), -10.0f);

    output[c] = mean;
}
