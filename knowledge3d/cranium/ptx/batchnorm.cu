/*
 * batchnorm.cu - Batch Normalization Kernel
 *
 * Implements spatial batch normalization for feature map normalization.
 * Used in OCR pipeline to stabilize training and improve convergence.
 *
 * Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * Where:
 *   - mean, var: Computed per-channel statistics
 *   - gamma, beta: Learned affine parameters
 *   - eps: Small constant for numerical stability (1e-5)
 *
 * Performance target: <200µs for 256×256×128 input on RTX 3060
 */

#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 16

/*
 * batchnorm_forward - Batch normalization forward pass
 *
 * Grid: ((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE, C)
 * Block: (TILE_SIZE, TILE_SIZE, 1)
 *
 * Args:
 *   input: Input feature map [H, W, C]
 *   output: Output feature map [H, W, C]
 *   mean: Per-channel mean [C]
 *   var: Per-channel variance [C]
 *   gamma: Scale parameter [C]
 *   beta: Shift parameter [C]
 *   H, W: Spatial dimensions
 *   C: Number of channels
 *   eps: Epsilon for numerical stability
 */
extern "C" __global__ void batchnorm_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int H,
    int W,
    int C,
    float eps
) {
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int c = blockIdx.z;

    if (row >= H || col >= W) return;

    const int idx = (row * W + col) * C + c;

    // Load input
    float x = input[idx];

    // Normalize
    float normalized = (x - mean[c]) / sqrtf(var[c] + eps);

    // Scale and shift
    float y = gamma[c] * normalized + beta[c];

    // Write output
    output[idx] = y;
}

/*
 * batchnorm_compute_stats - Compute per-channel mean and variance
 *
 * Two-pass algorithm:
 *   Pass 1: Compute mean
 *   Pass 2: Compute variance
 *
 * Grid: (C, 1, 1)
 * Block: (256, 1, 1)
 *
 * Args:
 *   input: Input feature map [H, W, C]
 *   mean: Output mean [C]
 *   var: Output variance [C]
 *   H, W, C: Dimensions
 */
extern "C" __global__ void batchnorm_compute_stats(
    const float* __restrict__ input,
    float* __restrict__ mean,
    float* __restrict__ var,
    int H,
    int W,
    int C
) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int spatial_size = H * W;

    // Shared memory for reduction
    __shared__ float shared_sum[256];
    __shared__ float shared_var_sum[256];

    // Phase 1: Compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = i * C + c;
        sum += input[idx];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 computes final mean
    if (tid == 0) {
        mean[c] = shared_sum[0] / spatial_size;
    }
    __syncthreads();

    // Phase 2: Compute variance
    float mean_val = mean[c];
    float var_sum = 0.0f;

    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = i * C + c;
        float diff = input[idx] - mean_val;
        var_sum += diff * diff;
    }

    shared_var_sum[tid] = var_sum;
    __syncthreads();

    // Reduction for variance
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_var_sum[tid] += shared_var_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        var[c] = shared_var_sum[0] / spatial_size;
    }
}

/*
 * batchnorm_fused - Fused batch normalization (compute stats + normalize)
 *
 * Single-kernel implementation that computes statistics and normalizes
 * in one pass. More efficient but requires more shared memory.
 *
 * Grid: (C, 1, 1)
 * Block: (256, 1, 1)
 *
 * Args:
 *   input: Input feature map [H, W, C]
 *   output: Output feature map [H, W, C]
 *   gamma: Scale parameter [C]
 *   beta: Shift parameter [C]
 *   H, W, C: Dimensions
 *   eps: Epsilon for numerical stability
 */
extern "C" __global__ void batchnorm_fused(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int H,
    int W,
    int C,
    float eps
) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int spatial_size = H * W;

    __shared__ float shared_sum[256];
    __shared__ float shared_var_sum[256];
    __shared__ float channel_mean;
    __shared__ float channel_var;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        sum += input[i * C + c];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce mean
    for (int stride = 128; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        channel_mean = shared_sum[0] / spatial_size;
    }
    __syncthreads();

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        float diff = input[i * C + c] - channel_mean;
        var_sum += diff * diff;
    }

    shared_var_sum[tid] = var_sum;
    __syncthreads();

    // Reduce variance
    for (int stride = 128; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_var_sum[tid] += shared_var_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        channel_var = shared_var_sum[0] / spatial_size;
    }
    __syncthreads();

    // Normalize and apply affine transform
    float std_inv = 1.0f / sqrtf(channel_var + eps);
    float gamma_val = gamma[c];
    float beta_val = beta[c];

    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int idx = i * C + c;
        float x = input[idx];
        float normalized = (x - channel_mean) * std_inv;
        output[idx] = gamma_val * normalized + beta_val;
    }
}

/*
 * layernorm_forward - Layer normalization (normalize across channels)
 *
 * Alternative to batch norm that normalizes across feature dimension
 * instead of spatial dimension. Often more stable for small batches.
 *
 * Grid: ((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE, 1)
 * Block: (TILE_SIZE, TILE_SIZE, 1)
 *
 * Args:
 *   input: Input feature map [H, W, C]
 *   output: Output feature map [H, W, C]
 *   gamma: Scale parameter [C]
 *   beta: Shift parameter [C]
 *   H, W, C: Dimensions
 *   eps: Epsilon for numerical stability
 */
extern "C" __global__ void layernorm_forward(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int H,
    int W,
    int C,
    float eps
) {
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (row >= H || col >= W) return;

    const int pixel_idx = row * W + col;

    // Compute mean across channels
    float sum = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < C; ++c) {
        sum += input[pixel_idx * C + c];
    }
    float mean = sum / C;

    // Compute variance
    float var_sum = 0.0f;
    #pragma unroll 8
    for (int c = 0; c < C; ++c) {
        float diff = input[pixel_idx * C + c] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / C;

    // Normalize and apply affine transform
    float std_inv = 1.0f / sqrtf(var + eps);

    #pragma unroll 8
    for (int c = 0; c < C; ++c) {
        int idx = pixel_idx * C + c;
        float x = input[idx];
        float normalized = (x - mean) * std_inv;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}
